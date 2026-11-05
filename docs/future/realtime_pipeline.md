# Real-Time Automated Screening Pipeline — Design (Future Work)

> **Status: documented future work — NOT yet implemented.** This is an
> architecture/design proposal. The queue/worker, notification, and orchestration glue
> described here do not exist in the repo yet; the stages reuse existing components.

## Purpose

This document describes an **end-to-end automated pipeline** that takes a captured
fundus image all the way to a stored report and a doctor notification with no manual
steps in between. Most of the heavy lifting already exists as discrete, well-factored
components (shared preprocessing, the inference `Registry`, the ReportLab report
generator); the missing piece is **orchestration glue** — a queue and worker that chain
the stages reliably and asynchronously. The system stays a **decision-support screening
assistant**, not a regulated diagnostic device, and a clinician confirms every result.

---

## Pipeline Overview

```
 Image Captured ─► Preprocessing ─► AI Inference ─► Classification ─►
        Grad-CAM ─► PDF Report ─► Database Storage ─► Doctor Notification
```

### Stage → Component Mapping

| Stage | Maps to (existing) | New glue needed |
|---|---|---|
| Image Captured | upload (`webapp/app.py`) or camera watcher (`docs/future/camera_integration.md`) | enqueue job |
| Preprocessing | `models/common/preprocessing.py` `build_transforms` (called inside inference) | — |
| AI Inference | `webapp/inference.py` `Registry.analyze()` (3 EfficientNetV2-S models, TTA) | run inside worker |
| Classification | per-disease grade + confidence in `analyze()` output `findings` | — |
| Grad-CAM | heatmap produced by `Registry.analyze()` (`result["heatmap"]`) | — |
| PDF Report | `reports/report_generator.py` `generate_pdf(...)` | — |
| Database Storage | `Image`/`Prediction`/`Report` rows (`docs/future/patient_management.md`) | persist step |
| Doctor Notification | — | email / Slack / webhook sender |

The only genuinely new code is: an **enqueue** step, a **worker** that runs the chain,
a **notifier**, and status tracking. Stages 2–6 are existing functions called in order.

---

## Architecture

A producer enqueues a job; one or more workers consume jobs and execute the stage chain;
results are persisted and the doctor is notified. Decoupling capture from processing via
a queue gives back-pressure, retries, and horizontal scale.

```
 ┌────────────┐   enqueue    ┌──────────────┐   dequeue   ┌─────────────────────────┐
 │  Producer  │─────────────►│   Job Queue  │────────────►│        Worker(s)        │
 │ upload /   │   {job_id,   │  (in-proc /  │             │  preprocess → infer →   │
 │ camera     │    img ref}  │   Redis+RQ / │             │  classify → grad-cam →  │
 │ watcher    │              │   Celery)    │             │  pdf → persist → notify │
 └────────────┘              └──────────────┘             └───────────┬─────────────┘
        ▲                                                             │
        │  status / result (job_id)                                  ▼
        │                                          ┌──────────────────────────────┐
 ┌──────┴───────┐                                  │ DB: Image/Prediction/Report  │
 │   Doctor     │◄──── notification (email/slack)──┤ + job status (queued/done)   │
 │  Dashboard   │                                  └──────────────────────────────┘
 └──────────────┘
```

### Queue / worker options (scale ladder)
1. **In-process background thread / `queue.Queue`** — simplest; fine for the single-user
   desktop prototype. Survives only while the Flask process runs.
2. **Redis + RQ** — lightweight, easy; durable jobs, retries, multiple workers. Good
   default once volume grows.
3. **Celery (+ Redis/RabbitMQ)** — full-featured: scheduling, retries with backoff,
   result backend, monitoring (Flower), multi-machine workers. For real scale.

GPU note: each worker loads the `Registry` once at startup (model weights resident in
memory), not per job, to avoid reload latency. On the Kaggle-first design the desktop is
inference-only; a single worker with the 3 models loaded is the baseline.

---

## Sequence of Steps

```
1.  Capture           image arrives (upload route OR camera watcher)
2.  Enqueue           producer writes job {job_id, image_path/bytes, patient_id?} to queue;
                      DB job status = "queued"; respond 202 Accepted + job_id
3.  Dequeue           worker pulls job; status = "processing"
4.  Preprocess        build_transforms / TTA transforms applied (inside inference call)
5.  Inference         Registry.analyze(image)  -> DR/ROP/Glaucoma probs
6.  Classification    grades + confidences + screening thresholds extracted from findings
7.  Grad-CAM          heatmap from analyze() result decoded to an image
8.  PDF Report        generate_pdf(out, patient, findings, image, gradcam) -> pdf_path
9.  Persist           insert Image, Prediction(s), Report rows; status = "done"
10. Notify            send doctor notification (email/Slack/webhook) with link to report
11. Surface           dashboard polls status / receives push; clinician reviews & signs
12. On error          status = "failed", increment retry; after N retries -> alert + failed/
```

The current synchronous `/predict` and `/report` routes remain for interactive use; the
async pipeline is for hands-off, high-volume, or camera-driven flows.

---

## Notification Options

- **Email (SMTP):** simplest; send the report PDF (or a secure link) to the on-call
  ophthalmologist. Use a link rather than attaching PII where possible.
- **Slack / Teams webhook:** post a card to a clinic channel — patient ref, top finding,
  recommendation (Healthy / Follow-up / Refer / Urgent), link to dashboard.
- **Generic webhook:** POST the result JSON to an external EHR/dashboard endpoint for
  custom integrations.
- **Prioritized routing:** "Urgent" recommendations (e.g. ROP positive) can trigger an
  immediate Slack/SMS page, while routine results batch into a daily digest.

A `Notifier` abstraction with pluggable channels (`EmailNotifier`, `SlackNotifier`,
`WebhookNotifier`) selected by config keeps this open-ended.

---

## Latency & Throughput Considerations

- **Per-image latency** is dominated by AI inference. Three EfficientNetV2-S models with
  TTA run sequentially today; expect sub-second to a few seconds per image on GPU, longer
  on CPU/MPS. Grad-CAM adds one extra backward pass.
- **Reduce latency:** disable/limit TTA for real-time mode (TTA is configurable per
  config), run the 3 models in parallel (threads/processes or batched), keep weights
  warm in the worker, use FP16 on GPU.
- **Throughput / scale-out:** add more RQ/Celery workers; the queue smooths bursts (e.g.
  a screening camp uploading many images). A GPU worker batches multiple images per
  forward pass for higher images/sec.
- **PDF + DB writes** are cheap relative to inference and run after the GPU work, so they
  don't bottleneck the model.
- **Back-pressure & reliability:** the queue caps concurrent work and persists jobs so a
  worker crash doesn't lose captures; failed jobs retry with backoff, then land in
  `failed/` with an alert.
- **Observability:** record per-stage timings and job status; ties into the existing
  experiment/metrics tooling (`models/common/experiment_logger.py`,
  `models/evaluation/`) for monitoring drift and latency over time.

---

## Suggested File Layout (when implemented)

```
pipeline/
  __init__.py
  queue.py        # enqueue + backend selector (in-proc | RQ | Celery)
  worker.py       # the stage chain (steps 4–10), loads Registry once
  notify.py       # Notifier + Email/Slack/Webhook channels
  status.py       # job status model + helpers
scripts/
  run_worker.py   # `python scripts/run_worker.py` starts a worker
```

## Phasing

1. **MVP** — in-process background worker; capture → analyze → PDF → notify (email).
2. **Durable** — Redis + RQ; retries, multiple workers, job status in DB.
3. **Scale** — Celery, GPU batching, parallel model execution, prioritized routing.

## Scope Note
Automation never replaces clinician judgement: the pipeline produces a *draft* screening
report carrying the existing "not a diagnostic device" disclaimer; a qualified
ophthalmologist reviews and signs off before any clinical action.
