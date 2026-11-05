# Roadmap — Future Work

Where RetinAI goes next, ordered from "small fixes that close known gaps" to "big features
with full design docs already written." Each item ties back to a limitation
([LIMITATIONS.md](LIMITATIONS.md)) or a challenge ([CHALLENGES.md](CHALLENGES.md)).

Legend: 🟢 quick win · 🟡 medium · 🔴 large / research.

---

## 1. Close the known implementation gaps (near-term)

These are gaps between what the app claims and what it does — low effort, high honesty payoff.

- 🟢 **Wire calibration into serving.** Temperature scaling is computed offline
  (`models/evaluation/calibration.py`) but the live app shows raw softmax. Persist the fitted
  temperature per model and apply it in `webapp/inference.py` so displayed confidence is
  actually calibrated. *(LIMITATIONS §5)*
- 🟢 **Make the dashboard DR metric match the cited one.** Add a `test` block / referable-AUC to
  `results/dr/metrics.json` (or have the card read the referable metric) so the dashboard and
  the About page agree. *(LIMITATIONS §5)*
- 🟢 **Retrain/confirm DR with the intended preprocessing.** The persisted DR run recorded
  `circle_crop/ben_graham/clahe = false`; re-run with `configs/dr.yaml` as written and confirm
  the checkpoint matches the documented pipeline. *(LIMITATIONS §5)*
- 🟢 **Fix or delete the root `Dockerfile`.** It still deploys the legacy `unified_model/` app.
  Point it at `webapp/` or remove it so only `retinal-ai/Dockerfile` remains. *(LIMITATIONS §5)*
- 🟡 **Per-site threshold tuning.** External validation showed operating points don't transfer.
  Add a small calibration-set flow so a new deployment can re-tune its threshold. *(CHALLENGES §7)*
- 🟡 **Run the statistical tests.** Execute the implemented McNemar/DeLong
  (`models/evaluation/statistical_tests.py`) to confirm the ResNet50-for-ROP choice and report CIs
  for all headline AUCs. *(LIMITATIONS §6)*

## 2. Strengthen the models & evidence

- 🔴 **A truly held-out Glaucoma source.** G1020 is now in-distribution; validate zero-shot on a
  *new* source (REFUGE / AIROGS) to make an honest generalization claim. The config stub already
  exists (`configs/external_glaucoma_refuge.yaml`). *(CHALLENGES §5)*
- 🔴 **A public ROP external set.** ROP has no external validation. Seek/curate a second ROP
  source, or partner with a clinic, to measure real generalization. *(LIMITATIONS §2)*
- 🟡 **More DR data + better early-DR detection.** Add sources (DDR, IDRiD) and target the weak
  No-DR ↔ Mild boundary (ordinal-regression head, focal-tversky, hard-example mining). *(RESULTS)*
- 🔴 **Richer ROP output.** Move beyond binary to zone/stage/plus-disease staging, closer to
  clinical ROP grading. *(LIMITATIONS §3)*
- 🟡 **Uncertainty & OOD rejection.** Add an abstain path for ungradable / non-fundus images
  (entropy threshold, MC-dropout, or an OOD detector). *(LIMITATIONS §3)*
- 🟡 **Both-eye + laterality fusion.** Accept OD/OS as a pair and fuse for a per-patient result.

## 3. Explainability & fairness

- 🟡 **Per-disease heatmaps.** Generate Grad-CAM for every positive finding, not just the first.
- 🟡 **Finer attribution.** Add SHAP/LIME or Grad-CAM++ for sanity-checking beyond coarse CAMs.
- 🔴 **Fairness audit.** Break down performance by age/sex/device/image-quality where metadata
  allows, and report subgroup gaps. *(LIMITATIONS §2)*

## 4. Product features (design docs already written)

Three features have complete architecture/design docs in [future/](future/) — reusing the
existing preprocessing, `Registry`, and `generate_pdf` so most of the work is glue.

- 🟡 **Patient management & persistence** — [future/patient_management.md](future/patient_management.md).
  SQLite→PostgreSQL via SQLAlchemy, a full schema (Patient/Visit/Image/Prediction/Report/FollowUp/
  User/AuditLog), auth + roles, encryption, consent, and **longitudinal comparison** across visits.
  Replaces the in-memory `deque`. *(LIMITATIONS §5)*
- 🟡 **Fundus-camera integration** — [future/camera_integration.md](future/camera_integration.md).
  Phase 1: a `watchdog` **watched-folder** watcher that auto-processes new captures (achievable
  now). Phase 2: direct device / **DICOM** (C-STORE SCP + Worklist) behind a `CameraSource`
  abstraction.
- 🔴 **Real-time automated pipeline** — [future/realtime_pipeline.md](future/realtime_pipeline.md).
  A queue + worker (in-process → Redis/RQ → Celery) chaining capture → preprocess → infer →
  Grad-CAM → PDF → persist → **doctor notification** (email/Slack/webhook), with back-pressure,
  retries, and per-stage timings.

## 5. Deployment & ops

- 🟡 **GPU inference option** for lower latency (the CPU Space takes a few seconds/image).
- 🟡 **Model quantization** (INT8 / ONNX) for faster CPU inference and mobile.
- 🟡 **Drift & latency monitoring** — log per-stage timings and prediction distributions
  (hooks into `models/evaluation/`) to catch data drift in production.
- 🟢 **Automated tests + CI** — currently none; add unit tests for preprocessing/inference and a
  smoke test for the web routes. *(LIMITATIONS §5)*

## 6. Toward the clinic (long horizon, research)

- 🔴 **Prospective validation** on real screening captures — the single biggest gap between
  prototype and clinical usefulness.
- 🔴 **Active learning with ophthalmologists** — route low-confidence / disagreement cases to
  clinicians and feed corrections back into training.
- 🔴 **Regulatory pathway** — IEC 62304, DICOM conformance, and clinical-trial design are
  explicitly out of scope for this prototype but are the prerequisites for any real deployment.

---

## Suggested next three

If picking up this project tomorrow: **(1)** wire calibration into serving + reconcile the DR
dashboard metric (a day, closes two honesty gaps), **(2)** run McNemar/DeLong and add CIs (makes
the results defensible), **(3)** build the patient-management MVP from
[future/patient_management.md](future/patient_management.md) (turns a stateless demo into a usable
tool). See [LIMITATIONS.md](LIMITATIONS.md) for the full gap list.
