# BE Capstone Project

## Project Title

**RetinAI — Explainable Multi-Disease Retinal Screening from a Single Fundus Photograph**

[![Python](https://img.shields.io/badge/Python-3.11%2B-blue?style=flat&logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5-EE4C2C?style=flat&logo=pytorch)](https://pytorch.org/)
[![Flask](https://img.shields.io/badge/Flask-3.0-black?style=flat&logo=flask)](https://flask.palletsprojects.com/)
[![Android](https://img.shields.io/badge/Android-Jetpack_Compose-3DDC84?style=flat&logo=android)](https://developer.android.com/jetpack/compose)
[![HF Space](https://img.shields.io/badge/Deployed-HuggingFace_Spaces-FFD21E?style=flat&logo=huggingface)](https://champ610-retinal-ai.hf.space)

A clinical **decision-support** prototype that screens one colour fundus (retina) photograph for
three sight-threatening diseases at once — **Diabetic Retinopathy (DR)**, **Retinopathy of
Prematurity (ROP)** and **Glaucoma** — and returns an explainable, graded, reportable result.
Each disease has its own dedicated model, its own tuned operating point, a Grad-CAM heatmap and
a downloadable PDF report.

> [!WARNING]
> **Not a medical device.** This is a research and education prototype built for a final-year
> capstone project. It has no regulatory clearance, was never prospectively validated, and must
> never be used for real diagnosis. Every finding belongs to a qualified clinician.

> [!NOTE]
> ### 🌐 Live Deployment
> **Web application:** [https://champ610-retinal-ai.hf.space](https://champ610-retinal-ai.hf.space)
> — free-tier, CPU-only, a few seconds per image. The Android client points at this same server.

---

## Table of Contents

1. [Team Details](#team-details)
2. [Guide Details](#guide-details)
3. [Problem Statement](#problem-statement)
4. [Abstract](#abstract)
5. [Objectives](#objectives)
6. [Scope of the Project](#scope-of-the-project)
7. [Existing System](#existing-system)
8. [Proposed System](#proposed-system)
9. [System Architecture](#system-architecture)
10. [Hardware Requirements](#hardware-requirements)
11. [Software Requirements](#software-requirements)
12. [Technologies Used](#technologies-used)
13. [Methodology](#methodology)
14. [Project Timeline](#project-timeline)
15. [Weekly Progress Updates](#weekly-progress-updates)
16. [Design Files](#design-files)
17. [Circuit Diagram](#circuit-diagram)
18. [Flowchart / Algorithm](#flowchart--algorithm)
19. [Implementation Details](#implementation-details)
20. [Code Structure](#code-structure)
21. [How to Run the Project](#how-to-run-the-project)
22. [Testing and Results](#testing-and-results)
23. [Result Images / Videos](#result-images--videos)
24. [Applications](#applications)
25. [Advantages](#advantages)
26. [Limitations](#limitations)
27. [Future Scope](#future-scope)
28. [Research Paper / Publication](#research-paper--publication)
29. [References](#references)
30. [Repository Update Guidelines](#repository-update-guidelines)
31. [Declaration](#declaration)
32. [License](#license)

---

## Team Details

| Sr. No. | Name of Student | Roll No. | Branch | Email ID | GitHub ID |
|---|---|---|---|---|---|
| 1 | Praharsh Nagpure | `TO FILL` | Automation and Robotics | 2023.praharsh.nagpure@ves.ac.in | Coldbari |
| 2 | `TO FILL` | `TO FILL` | Automation and Robotics | `TO FILL` | `TO FILL` |
| 3 | `TO FILL` | `TO FILL` | Automation and Robotics | `TO FILL` | `TO FILL` |
| 4 | `TO FILL` | `TO FILL` | Automation and Robotics | `TO FILL` | `TO FILL` |

---

## Guide Details

**Project Guide:** `TO FILL`
**Department:** Automation and Robotics
**Institute:** Vivekanand Education Society's Institute of Technology (VESIT), Mumbai

---

## Problem Statement

> The aim of this project is to design and develop an explainable deep-learning screening
> assistant that detects Diabetic Retinopathy, Retinopathy of Prematurity and Glaucoma from a
> single colour fundus photograph, so that limited ophthalmologist time can be directed to the
> patients most likely to need it — and to establish, through device-controlled and
> source-controlled validation, how much of that screening performance is real.

Three diseases share one property that makes them worth screening: they are **irreversible once
symptomatic but treatable when caught early**, and all three are visible in a photograph of the
retina. They also share one obstacle — the photograph must be read by a trained
ophthalmologist, and there are not enough of them.

India has roughly **1 ophthalmologist per 100,000 people**, concentrated in cities. Diabetic
Retinopathy alone requires annual screening of every diabetic patient; Retinopathy of
Prematurity requires repeated bedside examination of every pre-term infant in a NICU within a
narrow treatment window. The bottleneck is not the camera. It is the reading.

---

## Abstract

Diabetic Retinopathy, Retinopathy of Prematurity and Glaucoma are among the leading causes of
preventable blindness worldwide, and all three are diagnosable from a colour fundus photograph.
Screening programmes are limited not by imaging capacity but by the availability of
ophthalmologists to read the images. This project develops **RetinAI**, a decision-support
system that screens a single fundus photograph for all three diseases and returns a graded risk
level, a Grad-CAM explanation and a printable clinical report.

The system uses **three dedicated convolutional models** rather than one shared multi-task
network — an earlier unified EfficientNet-B3 was built, measured, and abandoned when it
plateaued at 73% ROP accuracy and 0.50 DR macro-F1. Each disease now selects its own backbone
(ResNet50 for ROP, chosen by a five-architecture sweep; EfficientNetV2-S for DR and Glaucoma),
preprocessing pipeline and operating point through a config-driven training framework in which
a new experiment is a new YAML file rather than a code change. Training runs on Kaggle GPUs;
serving runs CPU-only on HuggingFace Spaces, with a Jetpack Compose Android client for capture.

The project's substantive contribution is not the headline accuracy but the **audit of it**.
A device-confound analysis showed that every ROP-positive test image originated from a single
camera, reducing the defensible ROP AUC from 0.927 pooled to **0.881 device-controlled**. A
source-confound analysis reduced Glaucoma from 0.967 pooled to **0.878 source-controlled**, with
zero-shot performance on withheld collections ranging 0.735–0.936. DR referable-disease AUC of
**0.906** held at **0.837** on external Messidor-2 (−6.9%, within the ≤10% pass criterion). A
gradability gate and a patient-context routing layer were added after a cross-disease audit
found the ROP model firing on 59 of 59 adult eyes — a disease of prematurity that cannot occur
in an adult. Every number is reported with the conditions under which it was measured.

---

## Objectives

1. To study the clinical screening pathways for Diabetic Retinopathy, Retinopathy of Prematurity
   and Glaucoma, and the published deep-learning approaches to each.
2. To design a config-driven training framework in which disease, dataset, architecture,
   preprocessing, resolution and hyperparameters are specified declaratively, so experiments are
   reproducible from a saved configuration.
3. To train one dedicated model per disease, selecting each architecture by measured comparison
   rather than assumption.
4. To validate every model **externally** — on data the model never saw — and to report the
   internal-to-external drop honestly rather than the internal number alone.
5. To audit each headline metric for confounds (imaging device, data source, input resolution,
   population prevalence) and to report the confound-controlled figure alongside the pooled one.
6. To make every prediction explainable through Grad-CAM, calibrated through temperature
   scaling, and safe through a gradability gate and patient-context routing.
7. To deliver a working product: a web application and an Android client that produce a
   clinician-readable PDF report.
8. To document the work, its limitations and its failures, and to prepare it for publication.

---

## Scope of the Project

**In scope**

- Three dedicated deep-learning classifiers (DR 5-class, ROP binary, Glaucoma binary) trained on
  pooled public and institutional retinal datasets.
- A shared, single-source-of-truth preprocessing pipeline used identically in training and serving.
- External validation with no retraining (Messidor-2 and IDRiD for DR; withheld-source protocol
  for Glaucoma).
- Confound audits: device, data source, input resolution, prevalence and subgroup fairness.
- Explainability (Grad-CAM), probability calibration (temperature scaling), and statistical
  intervals (bootstrap, cluster-bootstrap for patient-grouped data).
- A gradability gate that rejects non-retinal and ungradable input, and a patient-context router
  that suppresses clinically impossible findings.
- A Flask web application and an Android (Jetpack Compose) client, both producing PDF reports.
- Docker deployment to HuggingFace Spaces.

**Explicitly out of scope**

- Regulatory clearance, clinical trials, or prospective validation of any kind.
- Autonomous diagnosis. The system is decision *support*; a clinician confirms every result.
- Patient record persistence, authentication and multi-tenancy (designed, not built).
- Fundus camera hardware. The project consumes images from existing clinical cameras.

---

## Existing System

Retinal disease screening today follows one of three routes, each with a distinct failure mode.

| Existing approach | How it works | Limitations |
|---|---|---|
| **Manual ophthalmoscopy / fundus reading** | A trained ophthalmologist examines the eye or reads the photograph | Does not scale; ~1 ophthalmologist per 100,000 people in India; inter-grader agreement on DR grading is only QWK ≈ 0.83–0.85, so even the reference standard is noisy |
| **Tele-ophthalmology reading centres** | Images captured locally, transmitted to a central grader | Reduces travel but not grader workload; turnaround measured in days, which is too slow for the ROP treatment window |
| **Commercial autonomous AI (IDx-DR, EyeArt)** | FDA-cleared autonomous DR detection | Single-disease (DR only); closed, unexplainable output; per-use licence cost; validated on populations and cameras that may not match the deployment site |

**The gap common to all three:** each screens for one disease at a time, and none exposes *why*
it reached its conclusion or *under what conditions* its accuracy was measured. A screening
number quoted without its confounds — which camera, which source, which prevalence — is not a
number a clinic can plan around.

---

## Proposed System

RetinAI screens a single fundus photograph for all three diseases in one pass, and reports each
result with its evidence and its conditions.

**Main idea.** One image in; three independent, dedicated models run; three graded results out,
each with a heatmap and a recommendation, assembled into one PDF report.

**How it works.**

1. **Gate.** Before any model runs, a gradability gate checks that the input is actually a
   retinal photograph (redness ratio, saturation, red-hue fraction, blur variance, luminance,
   field-of-view ratio). Non-retinal or unusable input is rejected rather than scored.
2. **Route.** Patient context (infant vs adult) determines which models are clinically
   admissible. ROP cannot occur in an adult; DR and Glaucoma are not screened in a neonate.
   Findings outside the patient's world are suppressed rather than shown.
3. **Preprocess.** One shared pipeline — circle-crop to the retinal disc, CLAHE contrast
   equalisation on the LAB luminance channel, Ben-Graham background subtraction for DR.
4. **Infer.** Three dedicated models, each with 5-view test-time augmentation (original,
   horizontal flip, vertical flip, ±10° rotation), averaged.
5. **Explain and report.** Grad-CAM over the last convolutional layer, a graded risk level per
   disease, and a one-page ReportLab clinical PDF.

**Major components.** Config-driven training framework · three disease models · gradability gate
· patient-context router · Grad-CAM explainer · calibration layer · Flask web app · Android
client · PDF report generator · external-validation and confound-audit harness.

**Expected benefits.** One capture screens three diseases. Every output is explainable and
carries a stated measurement condition. The whole system runs on commodity CPU hardware, so a
deployment costs a server, not a licence.

---

## System Architecture

![System Architecture](images/system_architecture.png)

```
                    ┌──────────────── TRAINING (Kaggle GPU: P100 / T4) ────────────────┐
  Kaggle datasets ─► kaggle_kernels/train_*.py ─► prepare_data.py ─► train.py ─► results/<disease>/
  EyePACS · APTOS      pin torch 2.5.1+cu121      build pooled       config-driven     weights.pth
  SMDG-19 · G1020      glob-discover mounts       manifest           trainer loop      metrics.json
  ROP database         extract archives           patient-grouped    AMP · OneCycle    plots/ errors/
                    └───────────────────────────────────┬─────────────────────────────┘
                                                        │ weights pulled back (Git LFS)
                    ┌───────────────────────────────────▼──────── SERVING (CPU-only) ──┐
  Fundus image ─►  Gradability gate ─► Patient-context router ─► 3 × DiseaseModel      │
  (web upload or     reject non-        suppress clinically       shared preprocessing  │
   Android camera)   retinal input      impossible findings       + 5-view TTA          │
                                                                    ├─ predict() → grade · risk · confidence
                                                                    └─ gradcam() → heatmap overlay
                                                              ─► reports/report_generator.py → PDF
                    └──────────────────────────────────────────────────────────────────┘
```

The **same** preprocessing module (`models/common/preprocessing.py`) executes in both training
and serving. This is deliberate: a screening system whose train and serve pipelines diverge is
measuring one thing and deploying another.

The gate and the router are not decoration. They were added *after* measurement — see
[Testing and Results](#testing-and-results).

---

## Hardware Requirements

> **This is a software and machine-learning project.** No circuit was designed, no PCB was
> fabricated, and no microcontroller was programmed. The table below lists the compute and
> capture hardware the system runs on and consumes images from — none of it was built by this
> team. This section is completed for the template's sake and is stated plainly rather than
> left blank.

| Sr. No. | Component | Specification | Quantity | Purpose |
|---|---|---|---|---|
| 1 | Training GPU (cloud) | Kaggle NVIDIA P100 16 GB / T4 16 GB | 1 session | Model training; 12 h session cap, 2 concurrent sessions |
| 2 | Inference server (cloud) | HuggingFace Spaces free tier, 2 vCPU / 16 GB RAM, **no GPU** | 1 | Serving the three models + Grad-CAM |
| 3 | Development machine | Apple Silicon Mac, 16 GB RAM | 1 | Development, evaluation, report generation, MPS-accelerated local inference |
| 4 | Android device | Android 8.0+ (API 26), rear camera with autofocus | 1 | Image capture, on-device quality check, PDF export |
| 5 | Fundus camera *(third-party, not built)* | Adult: standard mydriatic/non-mydriatic fundus camera. Infant: RetCam-class wide-field contact camera | — | Source of the retinal photographs the system consumes |

---

## Software Requirements

| Sr. No. | Software / Tool | Version | Purpose |
|---|---|---|---|
| 1 | Python | 3.11+ | Primary language for training, evaluation and the web application |
| 2 | PyTorch + torchvision | 2.5.1 (cu121 on Kaggle) / 2.1.0+cpu (serving) | Model definition, training, inference |
| 3 | timm | latest | Additional backbones for the ROP staging sweep |
| 4 | OpenCV (headless) | 4.x | Circle-crop, CLAHE, Ben-Graham preprocessing |
| 5 | scikit-learn | 1.x | Metrics, QWK, ROC/AUC, bootstrap intervals |
| 6 | NumPy · pandas | latest | Manifests, arrays, tabular results |
| 7 | Flask + Jinja2 | 3.x | Web application and templating |
| 8 | ReportLab | 4.x | Clinical PDF report generation |
| 9 | Matplotlib | 3.x | Training curves, confusion matrices, all 12 evidence figures |
| 10 | PyYAML | 6.x | The config-driven experiment system |
| 11 | Docker | 24+ | Reproducible deployment image (HuggingFace Spaces) |
| 12 | Kotlin + Jetpack Compose | Kotlin 2.0, Compose BOM 2024.x | Android client |
| 13 | CameraX · OkHttp | latest | Android capture and networking |
| 14 | Kaggle CLI | 1.6+ | Pushing training kernels, pulling outputs |
| 15 | Git + Git LFS | 2.4x | Version control; model weights ship via LFS |

---

## Technologies Used

* **Deep learning** — PyTorch, EfficientNetV2-S, ResNet50, transfer learning from ImageNet,
  mixed-precision (AMP) training, OneCycleLR, gradient accumulation, Mixup regularisation
* **Loss design** — focal loss (γ = 2.0), class-balanced weights (effective-number method),
  weighted random sampling, ordinal regression head
* **Computer vision** — circle-crop segmentation, CLAHE, Ben-Graham background subtraction,
  5-view test-time augmentation, letterbox resizing
* **Explainable AI** — Grad-CAM via forward/backward hooks on the last convolutional layer
* **Statistics** — quadratic weighted kappa, bootstrap confidence intervals, cluster-bootstrap
  over patients, McNemar and DeLong tests, expected calibration error, temperature scaling
* **Web** — Flask, Jinja2, vanilla JavaScript, ReportLab
* **Mobile** — Kotlin, Jetpack Compose, Material 3, CameraX, OkHttp
* **Infrastructure** — Kaggle GPU kernels, Docker, HuggingFace Spaces, Git LFS, GitHub Actions CI

---

## Methodology

1. **Literature survey** — DR grading (EyePACS/Kaggle protocol, Gulshan et al.), ROP screening
   (ICROP classification), glaucoma from fundus (cup-to-disc ratio methods), and the confound
   literature on shortcut learning in medical imaging.
2. **Problem identification** — screening throughput, not imaging capacity, is the bottleneck;
   and published accuracies are frequently confounded.
3. **Requirement analysis** — one image, three diseases, explainable, deployable on CPU,
   defensible under audit.
4. **Baseline system** — a unified multi-task EfficientNet-B3 with shared backbone and two heads.
   Built, measured, and **rejected**: ROP 73.3% accuracy, DR macro-F1 0.497.
5. **System redesign** — a config-driven framework with one dedicated model per disease.
6. **Architecture selection by measurement** — a five-backbone sweep on the ROP set
   (EfficientNet-B0, EfficientNetV2-S, MobileNetV3-Large, ResNet50, DenseNet121) benchmarked on
   accuracy, macro-F1, QWK, AUC, parameter count and CPU/GPU latency.
7. **Training** — Kaggle GPU kernels, patient-grouped splits to prevent leakage, per-epoch
   resume-safe checkpointing against the 12-hour session cap.
8. **Evaluation** — internal held-out metrics with bootstrap intervals.
9. **External validation** — no retraining, on Messidor-2 and IDRiD (DR) and withheld sources
   (Glaucoma), with a PASS/REVIEW verdict on the internal→external drop.
10. **Confound audit** — device stratification (ROP), source stratification (Glaucoma), input
    resolution sensitivity (DR), PPV against realistic prevalence, and subgroup fairness.
11. **Safety layer** — gradability gate and patient-context routing, each calibrated on a
    training split and then *measured* on held-out data.
12. **Integration** — Flask web app, Android client, PDF reporting, Docker deployment.
13. **Documentation and publication** — this repository, the technical documentation set, and a
    paper in preparation.

---

## Project Timeline

| Week | Task Planned | Status |
|---|---|---|
| Week 1 (08 Jun) | Problem finalisation; baseline unified ROP+DR model | ✅ Completed |
| Week 2 (15 Jun) | Literature survey; dataset acquisition | ✅ Completed |
| Week 3 (22 Jun) | Config-driven framework; architecture sweep | ✅ Completed |
| Week 4 (29 Jun) | Referable-DR reframing; multi-source glaucoma; web app rebuild | ✅ Completed |
| Week 5–7 (06–26 Jul) | Presentation preparation; deck rebuild; dataset survey | ✅ Completed |
| Week 8 (27 Jul) | Safety, honesty and statistics overhaul; device-confound audit | ✅ Completed |
| Week 9 (03 Aug) | Android client; confound audits T12–T17; evidence figures | ✅ Completed |
| Week 10 (10 Aug) | ROP staging corpus; five-backbone staging benchmark | 🔄 In Progress |
| Week 11 | Statistical significance tests (McNemar / DeLong); calibration wired into serving | ⏳ Pending |
| Week 12 | Paper drafting and submission | ⏳ Pending |

---

## Weekly Progress Updates

> Backfilled from the development repository's commit history. Each row's commit link points at
> the work actually done that week.

| Week | Date | Work Completed | Work Planned for Next Week | Issues / Challenges | Commits |
|---|---|---|---|---|---|
| Week 1 | 08 Jun 2026 | Baseline built: unified ROP+DR EfficientNet-B3 with two heads and a Flask app. Code structure refactored. | Measure the baseline properly; survey datasets | Shared backbone forced a compromise representation across two very different diseases | 2 |
| Week 2 | 15–21 Jun 2026 | Literature survey; dataset acquisition (EyePACS, APTOS, SMDG-19, G1020); Kaggle environment setup | Rebuild as dedicated per-disease models | Baseline plateaued — ROP 73.3% acc, DR macro-F1 0.497 | 0 |
| Week 3 | 22–28 Jun 2026 | Config-driven Kaggle-first training framework; evaluation, validation and reporting layers; five-architecture ROP sweep (**ResNet50 best, AUC 0.964**); multi-disease web app; mixup regulariser | External validation; glaucoma multi-source | Kaggle P100 `sm_60` unsupported by default torch; 384px OOM on T4; `kaggle/` folder shadowed the PyPI package | 10 |
| Week 4 | 29 Jun–05 Jul 2026 | Referable-DR screening eval; multi-source glaucoma config; full Model Card & Methodology page; dashboard/UX overhaul; deck rebuilt for the three-model system | Presentation; then rigor pass | Glaucoma SMDG-only collapsed to **AUC 0.589 zero-shot on G1020** — a genuine domain shift | 7 |
| Week 5–7 | 06–26 Jul 2026 | Presentation preparation and delivery; dataset survey for a second ROP source | Full honesty and statistics overhaul | No repository activity — `TO FILL: reason (exams / presentation period)` | 0 |
| Week 8 | 27 Jul–02 Aug 2026 | Safety, honesty and statistics overhaul; **T9 device-confound audit** (real confound confirmed); cluster-bootstrap intervals; DR and glaucoma serving calibration; gradability gate FRR measured on adults; glaucoma source audit; Android client started | Complete Android client; finish confound audits | ROP's 0.927 was partly device recognition; a PHI egress risk was found and guarded | 21 |
| Week 9 | 03–09 Aug 2026 | **Android client complete** (CameraX capture, on-device quality gate, result/report screens, PDF export, instrumented tests); T12 glaucoma zero-shot range; T13 DR on IDRiD + resolution sensitivity; T16 cross-disease routing; T17 confusion matrices and clinic PPV; **12 evidence figures**; PHI incident remediated; ROP staging corpus consolidated from four sources | ROP 6-class ICROP staging; five-backbone benchmark | **Six infant patient photographs were being served publicly** — removed and swept; ROP model fires on 59/59 adult eyes | 43 |
| Week 10 | 10–16 Aug 2026 | *(in progress)* ROP staging plumbing, shortcut audit, Kaggle bundle rebuild, first real staging training run | Report staging results; run significance tests | `TO FILL` | — |

**Development repository:** the full commit history, code and results live in the private
development repository; this log book mirrors the milestones. Ask the guide if direct access to
the development history is required for evaluation.

---

## Design Files

| File Type | File Name / Link | Description |
|---|---|---|
| System architecture | [images/system_architecture.png](images/system_architecture.png) | Training and serving data flow |
| Flowchart | [images/flowchart.png](images/flowchart.png) | Gate → route → preprocess → infer → explain → report |
| Model evidence figures | [images/](images/) | 12 figures covering every headline claim |
| Software design | [software/software.md](software/software.md) | Module-by-module implementation notes |
| Literature survey | [docs/literarture_survey.md](docs/literarture_survey.md) | Reviewed work and how it shaped the design |
| Reference papers | [reference/paper.md](reference/paper.md) | IEEE-format reference list |
| CAD Model | Not applicable | Software-only project |
| Circuit Diagram | Not applicable | Software-only project |
| PCB Design | Not applicable | Software-only project |
| Simulation File | Not applicable | Software-only project |

---

## Circuit Diagram

**Not applicable.** RetinAI is a software and machine-learning system with no custom electronic
hardware. There is no circuit, no PCB and no microcontroller in this project. The
[System Architecture](#system-architecture) diagram serves the equivalent purpose — it shows how
data moves between the components.

---

## Flowchart / Algorithm

![Flowchart](images/flowchart.png)

### Algorithm

```
1.  Start
2.  Acquire a colour fundus image (web upload or Android camera capture)
3.  Collect patient context (age band: infant | adult)
4.  GRADABILITY GATE
      compute redness ratio, saturation, red-hue fraction,
              blur variance, luminance, field-of-view ratio
      IF any metric outside calibrated bounds:
          RETURN "ungradable" with the specific reason        → go to 12
5.  ROUTING
      IF patient is an infant  : admissible = { ROP }
      IF patient is an adult   : admissible = { DR, Glaucoma }
      IF no context supplied   : run all, and mark every result as unrouted
6.  PREPROCESS
      circle-crop to the retinal disc (tolerance 7)
      CLAHE on the LAB L-channel (clip 2.0, grid 8×8)
      Ben-Graham background subtraction        [DR only]
      resize to 384×384, ImageNet normalisation
7.  FOR each admissible disease model:
      build 5 TTA views (original, H-flip, V-flip, +10°, −10°)
      forward pass each view; average the softmax outputs
      apply the disease's operating threshold (or argmax for multiclass)
      map to grade → risk level → clinical recommendation
8.  EXPLAIN
      Grad-CAM on the last conv layer of the first abnormal finding
      overlay a JET colormap on the preprocessed image
9.  CALIBRATE
      apply the fitted temperature to the reported confidence
10. ASSEMBLE the per-disease results, heatmap and recommendations
11. GENERATE the PDF report (ReportLab, A4, with the disclaimer)
12. DISPLAY / download the result
13. Stop
```

---

## Implementation Details

### Hardware Implementation

Not applicable — no custom hardware was designed or fabricated. The system runs on commodity
cloud compute (Kaggle GPU for training, CPU-only HuggingFace Spaces for serving) and consumes
images from existing clinical fundus cameras. The Android client uses the host device's rear
camera through CameraX with autofocus, torch control and an alignment overlay.

### Software Implementation

**Config-driven training framework.** Every experiment is a YAML file. `train.py` loads a config,
seeds, picks a device, builds the pooled manifest, constructs dataloaders and a model from the
config, trains, reloads the best checkpoint, evaluates, computes metrics and writes an
auto-incrementing `results/experiment_NNN/` directory containing the exact config used. Any run
is reproducible from its own saved configuration. CLI overrides (`--set train.epochs=2`) allow
a variation without editing code.

**Shared preprocessing.** `models/common/preprocessing.py` is the single source of truth —
`circle_crop(tol=7)`, `clahe(clip=2.0, grid=8)` on the LAB luminance channel, `ben_graham(sigma)`
as `addWeighted(img, 4, blur, −4, 128)`, and `build_tta_transforms` producing the five serving
views. Imported by both the trainer and the web application.

**Training recipe (common to all three diseases).** Seed 42, 384×384 input, AdamW (lr 2e-4,
weight decay 1e-4), OneCycleLR, AMP mixed precision, batch 16 with gradient accumulation 2
(effective 32), focal loss γ = 2.0 with class-balanced weights and a weighted random sampler,
5-view TTA, early stopping on the disease's primary metric.

**Per-disease specifics.**

| | DR | ROP | Glaucoma |
|---|---|---|---|
| Architecture | EfficientNetV2-S | **ResNet50** (won the sweep) | EfficientNetV2-S |
| Classes | 5 (No / Mild / Moderate / Severe / Proliferative) | 2 (No ROP / ROP) | 2 (Non-glaucoma / Glaucoma) |
| Primary metric | QWK, screening endpoint referable (grade ≥ 2) | recall | AUC |
| Preprocessing | circle-crop + Ben-Graham + CLAHE | circle-crop + CLAHE + strong augmentation | circle-crop + CLAHE |
| Epochs / patience | 25 / 7 | 30 / 8 | 25 / 7 |
| Operating point | argmax; referable grade = 2 | threshold **0.1933** | threshold 0.5 |

**Data handling.** `build_manifest` pools heterogeneous sources into a uniform
`image_path,label,split` manifest. The ROP database is split **by patient**, not by image — the
first token of each filename is the patient ID, and 70/15/15 splits are taken over patients so
no infant appears in two splits. A dataset-QA pass checks for corrupt files, bad labels,
resolution outliers, blur (variance-of-Laplacian) and perceptual-hash duplicates before training.

**Safety layer.** The gradability gate's thresholds were calibrated on 400 images from the ROP
*train* split only; the test split, the HRF cohorts and EyePACS/APTOS were held out, so the
reported false-rejection rates are measurements rather than fits. The patient-context router
suppresses clinically impossible findings.

**Serving.** `webapp/inference.py` holds a `Registry` that loads all three models from
`registry.yaml` and degrades gracefully — a missing checkpoint marks that disease unavailable
and the rest still work. Grad-CAM is generic across backbones via forward/backward hooks.

**Android client.** Kotlin + Jetpack Compose, Material 3. CameraX capture with focus, torch and
an alignment overlay; an advisory on-device quality meter calibrated at a fixed scale; mandatory
patient-context entry before scanning; a result screen that honours the server's routing,
calibration and gate states; on-device PDF export with share and print; a file-backed report
store. Unit tests parse fixtures captured from live server responses.

**Deployment.** Docker on HuggingFace Spaces — `python:3.11-slim`, CPU-only torch 2.1.0, non-root
uid 1000, port 7860, weights via Git LFS.

---

## Code Structure

```text
RetinAI/
│
├── README.md
├── train.py · prepare_data.py · evaluate.py · benchmark.py · predict.py   ← entry points
├── requirements.txt · Dockerfile
│
├── configs/                    ← one YAML per experiment
│   ├── dr.yaml · rop.yaml · glaucoma.yaml · glaucoma_multisource.yaml
│   ├── sweep.yaml · sweep_rop.yaml · rop_staging.yaml
│   ├── ablation_dr.yaml
│   └── external_dr_messidor.yaml · external_glaucoma_refuge.yaml
│
├── models/                     ← the training framework
│   ├── common/                 ← config, datasets, data_prep, preprocessing,
│   │                             architectures, losses, metrics, train_utils,
│   │                             experiment_logger, data_quality
│   ├── evaluation/             ← calibration, statistical_tests, error_analysis
│   ├── validation/             ← external_validation
│   ├── comparison/             ← make_table (architecture sweeps)
│   └── experiments/            ← ablation_study
│
├── kaggle_kernels/             ← GPU training kernels + the validated workflow
├── data_prep/                  ← patient-level splitting
├── dataset_split/              ← the ROP train/val/test split by patient
│
├── results/                    ← weights.pth + metrics.json + audits, per disease
│   ├── dr/ · rop/ · glaucoma/
│   ├── rop_device_audit.md · gate_calibration.md · dr_intervals.md · sweep_rop.md
│   └── cross_disease_matrix.json · confusion_tables.json
│
├── graphs/                     ← 12 evidence figures, generated from results/
├── scripts/                    ← make_graphs.py, generate_thesis_assets.py, audits
├── reports/                    ← ReportLab clinical-PDF generator
│
├── webapp/                     ← the Flask application (the product)
├── retinal-ai/                 ← deployment mirror → HuggingFace Space (Docker)
├── android/                    ← Jetpack Compose Android client
├── tests/                      ← unit and integration tests
│
├── docs/                       ← PROJECT_OVERVIEW · RESULTS · LIMITATIONS ·
│                                 CHALLENGES · DATASETS · ROADMAP
├── thesis_assets/              ← collected figures and tables for the paper
├── unified_model/              ← LEGACY: the earlier single multi-task model
└── legacy/                     ← quarantined first-generation scripts
```

---

## How to Run the Project

### Step 1: Clone the Repository

```bash
git clone https://github.com/Coldbari/RetinAI_BE_Project_2026_2027.git
cd RetinAI_BE_Project_2026_2027
```

### Step 2: Install Dependencies

```bash
python -m venv .venv && source .venv/bin/activate      # Python 3.11+
pip install -r requirements.txt
```

### Step 3: Run

**Web application (the product):**

```bash
python webapp/app.py            # → http://127.0.0.1:5002
```

The app degrades gracefully: if a disease's `results/<disease>/weights.pth` is absent, that model
reports as unavailable and the other two still screen.

**Train a model** (training is Kaggle-first; nothing trains on a laptop):

```bash
python train.py --config configs/rop.yaml
python train.py --config configs/dr.yaml --set train.epochs=2 model.arch=efficientnet_b0
```

**Evaluate a checkpoint:**

```bash
python evaluate.py --config configs/rop.yaml --weights results/rop/weights.pth
```

**Regenerate every evidence figure from the recorded results:**

```bash
python scripts/make_graphs.py
```

**Android client:**

```bash
cd android && ./gradlew assembleDebug
# install the APK, then set the server URL in Settings
```

### Step 4: Observe the Output

Upload a fundus photograph and supply the patient's age band. The application returns, per
admissible disease: a grade, a calibrated confidence, a risk level, a clinical recommendation and
a Grad-CAM heatmap — plus a downloadable one-page PDF report. Non-retinal images are rejected by
the gate with a stated reason rather than scored.

**Or use the live deployment:** [https://champ610-retinal-ai.hf.space](https://champ610-retinal-ai.hf.space)

---

## Testing and Results

### Headline results, with the conditions they were measured under

| Disease | Backbone | Pooled (internal) | **Confound-controlled** | External (no retraining) |
|---|---|---|---|---|
| **DR** (referable, grade ≥ 2) | EfficientNetV2-S | **AUC 0.906** [0.893, 0.919] | — | Messidor-2 **0.837** (−6.9%, **PASS**) |
| DR (5-class) | EfficientNetV2-S | QWK 0.728 [0.709, 0.746], acc 74.4% | — | Messidor-2 QWK 0.654 |
| **ROP** | ResNet50 | AUC 0.927 | **0.881 device-controlled** | none — no public ROP set exists |
| **Glaucoma** *(exploratory)* | EfficientNetV2-S | AUC 0.967 [0.957, 0.976] | **0.878 source-controlled** | zero-shot 0.735 (ORIGA) – 0.936 (REFUGE1) |

**Read the confound-controlled column, not the pooled one.** The difference between them is the
finding, and it is the part of this project worth defending in a viva.

### Test log

| Test No. | Test Description | Expected Result | Actual Result | Status |
|---|---|---|---|---|
| 1 | Patient-level ROP split — no infant in two splits | 0 patients shared across splits | train 3,574 (131 pts) / val 928 (28) / test 1,502 (29); 0 shared | ✅ Pass |
| 2 | Five-backbone ROP architecture sweep | A defensible best backbone | ResNet50 — AUC 0.9635 vs next-best 0.8700, competitive latency | ✅ Pass |
| 3 | DR external validation on Messidor-2, no retraining | Internal→external drop ≤ 10% | referable AUC 0.906 → 0.837, drop 6.9% | ✅ Pass |
| 4 | DR external validation on IDRiD | Transfer holds | Transfers well; input resolution alone is worth 0.08 AUC | ✅ Pass |
| 5 | Glaucoma zero-shot on a source withheld from training | AUC ≥ 0.90 | **0.936 (REFUGE1) but 0.735 (ORIGA)** — depends on which source is held out | ⚠️ Partial |
| 6 | ROP device-confound audit | Device carries no label information | **FAIL — patient-level prevalence spread 30.8% across 3 devices; every ROP-positive test image comes from one device.** Defensible AUC 0.881, not 0.927 | ❌ Fail → re-split |
| 7 | Glaucoma source-confound audit | Pooled ≈ within-source | **FAIL — pooled 0.967 vs source-controlled 0.878.** 340 of 506 positives come from single-class sources | ❌ Fail → reported both |
| 8 | Cross-disease admissibility (every image through every model) | Models abstain outside their world | **FAIL — the ROP model flagged 59 of 59 adult eyes.** ROP cannot occur in an adult | ❌ Fail → routing added |
| 9 | Gradability gate false-rejection on adult fundus images | FRR < 5% | 1.7% on 4,000 EyePACS images (61 not-fundus, 8 poor-exposure) | ✅ Pass |
| 10 | Gradability gate rejects synthetic non-retinal input | All rejected | 6/6 probes rejected (noise, grey, black, white, two dashboard screenshots) | ✅ Pass |
| 11 | ROP operating point selected on val, measured once on test | Sensitivity ≥ 0.90 | thr 0.1933 → **sens 0.990 / spec 0.398** (TTA path, as served) | ✅ Pass |
| 12 | ROP per-infant specificity at the deployed threshold | Some healthy infants cleared | **0.000 — 0 of 22 healthy infants cleared.** A triage filter, not a rule-out test | ⚠️ Stated |
| 13 | Positive predictive value at realistic clinic prevalence | Useful PPV | ROP ≈ **13%** in a real NICU — sensitivity and specificity survive a change of population, PPV does not | ⚠️ Stated |
| 14 | Probability calibration (temperature scaling) | Lower ECE | DR referable 0.141 → 0.111 (T = 0.70); DR 5-class 0.181 → 0.133; ROP 0.059 → 0.041 | ✅ Pass |
| 15 | Subgroup fairness audit for ROP | Report subgroup gaps | **Not assessable** — the available metadata cannot support a fairness claim. That is the finding | ⚠️ Stated |
| 16 | End-to-end manual test set | Correct end-to-end behaviour | 93 labelled images, 84% overall correct | ✅ Pass |
| 17 | Train/serve preprocessing skew for DR | No skew | Investigated and **disproven** — the suspected skew was not real | ✅ Pass |
| 18 | PHI exposure sweep of the repository | No patient data served | **FAIL — six infant patient photographs were being served publicly.** Removed, history swept, egress guard added | ❌ Fail → remediated |

Six of these tests failed. Each failure is listed here rather than removed, because each one
changed the system: test 6 forced a device-stratified re-split, test 8 produced the routing
layer, test 18 produced a PHI egress guard.

---

## Result Images / Videos

All twelve figures are generated by `scripts/make_graphs.py` **from the recorded results**, so
re-running an experiment and re-running the script keeps the figure and the number in step.
Nothing is hand-drawn.

| # | Figure | The question it answers |
|---|---|---|
| 01 | [Confusion matrices](images/01_confusion_matrices.png) | Of N patients, how many did each model get right? |
| 02 | [PPV vs prevalence](images/02_ppv_vs_prevalence.png) | **Of 100 patients flagged, how many really have it?** |
| 03 | [ROC curves](images/03_roc_curves.png) | Ranking ability on each model's own held-out set |
| 04 | [ROP operating curve](images/04_rop_operating_curve.png) | The sensitivity/specificity trade — the deployed point is a choice |
| 05 | [ROP device confound](images/05_rop_device_confound.png) | How much of 0.927 was the model recognising the camera |
| 06 | [Glaucoma source confound](images/06_glaucoma_source_confound.png) | Pooled 0.967 vs within-source 0.878 |
| 07 | [Glaucoma held-out](images/07_glaucoma_heldout.png) | Zero-shot on withheld collections — 0.735 vs 0.936 |
| 08 | [DR external + resolution](images/08_dr_external_and_resolution.png) | DR on IDRiD, and what downscaling costs |
| 09 | [Cross-disease matrix](images/09_cross_disease_matrix.png) | Every image through every model; what routing prevents |
| 10 | [Calibration](images/10_calibration.png) | Is the displayed confidence honest? |
| 11 | [Safety before/after](images/11_safety_before_after.png) | What the gate and routing removed |
| 12 | [ROP subgroups](images/12_rop_subgroups.png) | Can fairness be demonstrated? No — and why that is the finding |

**Start with figure 02.** It carries the result that most changes how the system should be
described: sensitivity and specificity survive a change of population, PPV does not.

**Figures 05, 06 and 07 are the same lesson three times.** A pooled number across cameras or
collections is not purely a measure of the disease — part of it is the model recognising where
the image came from.

The figures use a colourblind-validated palette (DR blue `#3987e5`, ROP orange `#d95926`,
Glaucoma aqua `#199e70`; worst adjacent CVD ΔE 9.4). Colour is never the only cue — every series
is direct-labelled.

**Live demo:** [https://champ610-retinal-ai.hf.space](https://champ610-retinal-ai.hf.space)
**Demo video:** `TO FILL: record a walkthrough and paste the Drive link`

---

## Applications

1. **Diabetic retinopathy screening camps** — triage a day's captures so the ophthalmologist
   reads the flagged subset first, rather than all of them in capture order.
2. **NICU ROP triage** — flag pre-term infants whose retinal images warrant urgent specialist
   review inside the narrow ROP treatment window.
3. **Primary health centres and tele-ophthalmology** — a technician captures; the system
   produces a graded, explained, printable report for a remote specialist to confirm.
4. **Opportunistic glaucoma case-finding** — screen fundus images already captured for another
   reason, since glaucoma is asymptomatic until late.
5. **Medical education** — the Grad-CAM gallery and the confound audits are teaching material on
   both retinal pathology and the ways a medical AI result can be misleading.

---

## Advantages

1. **One capture, three diseases.** A single fundus photograph is screened for DR, ROP and
   Glaucoma in one pass, instead of three separate workflows.
2. **Every number carries its conditions.** Device-controlled, source-controlled and
   prevalence-adjusted figures are reported alongside the pooled ones, so the result can be
   planned around rather than merely quoted.
3. **Explainable, not opaque.** Grad-CAM shows the region driving each finding; confidence is
   temperature-calibrated rather than raw softmax.
4. **Fails safe.** A gradability gate rejects non-retinal input with a stated reason, and
   patient-context routing suppresses clinically impossible findings — the ROP model is not
   allowed to diagnose an adult.
5. **Runs on commodity CPU hardware.** No GPU is needed to serve; a deployment costs a small
   server, not a per-use licence.
6. **Reproducible by construction.** Every run writes the exact config that produced it; every
   figure regenerates from the recorded results.
7. **Reaches the point of care.** The Android client captures, checks quality on-device, and
   exports a PDF report without a desktop.

---

## Limitations

1. **Not a medical device.** No FDA/CE/CDSCO clearance, no IEC 62304 process, no clinical trial,
   and no prospective validation. Every result is retrospective, on curated datasets.
2. **ROP's headline number is confounded.** Every ROP-positive test image comes from a single
   camera. The defensible device-controlled AUC is **0.881**, not 0.927 — and even that is
   measured inside one device.
3. **Glaucoma is exploratory, not settled.** Pooled 0.967 falls to **0.878** once sources are
   controlled, and zero-shot performance on a withheld collection ranges 0.735–0.936 depending on
   which one is withheld. 340 of 506 positives come from sources contributing only one class.
   The earlier headline of 0.973 should not be quoted without this context.
4. **ROP has no external test set.** No public ROP dataset exists, so ROP was only ever validated
   on a held-out split of the same database. Its true generalisation is unmeasured.
5. **Per-infant specificity is 0.000** at the deployed threshold — 0 of 22 healthy infants were
   cleared. The system is a sensitivity-maximising triage filter, not a rule-out test.
6. **PPV collapses at real prevalence.** ROP is right about **13%** of the time it raises a flag
   in a real NICU. Sensitivity and specificity transfer across populations; PPV does not.
7. **Operating points do not transfer.** On Messidor-2, holding sensitivity at 0.90 dropped
   specificity to 0.30. Every new site needs its own threshold re-tuning.
8. **Early DR remains the weak point.** The No-DR ↔ Mild boundary is the largest error source
   (Mild-DR F1 0.30). Early disease is the most valuable to catch and the hardest to detect.
9. **Label noise caps 5-class DR.** EyePACS inter-grader agreement is QWK ≈ 0.83–0.85, so 5-class
   accuracy above ~90% is unachievable — this is why the endpoint was reframed to referable DR.
10. **Fairness is not demonstrated.** The available metadata cannot support a subgroup fairness
    claim for ROP. Stating that is more honest than a fabricated breakdown.
11. **Binary ROP loses clinical detail.** The deployed model does not stage zone, stage or plus
    disease as ICROP grading requires. Six-class staging is in progress.
12. **Grad-CAM is coarse and not causal.** It shows where the last conv layer activated, not
    clinical reasoning. It can look plausible while the model relies on a spurious cue.
13. **No uncertainty or abstention inside the models.** The gate rejects non-retinal input, but a
    model given a gradable image always answers — there is no "I don't know".
14. **No persistence, authentication or audit trail.** Screening history is in-memory and lost on
    restart. Fine for a demo, not for real patient data.
15. **CPU-only latency.** Three models × 5-view TTA + Grad-CAM take a few seconds per image on
    the free tier. Not real-time, and not batched.

---

## Future Scope

1. **Run the significance tests.** McNemar and DeLong are implemented but not yet reported for
   the final architecture choices — the ResNet50-for-ROP recommendation currently rests on point
   metrics alone.
2. **Six-class ICROP ROP staging.** Move beyond binary to zone/stage/plus disease. The staging
   corpus is consolidated and a five-backbone benchmark is running.
3. **A genuinely held-out glaucoma source.** Validate zero-shot on a collection never seen in
   training, and report that number as the headline instead of the pooled one.
4. **A second ROP source.** Partner with a clinic or curate a second database so ROP finally has
   an external validation set and a device-independent number.
5. **Wire calibration fully into serving** so displayed confidence is calibrated everywhere, and
   add per-site threshold re-tuning as a first-class deployment step.
6. **Uncertainty and out-of-distribution rejection** — an abstain path for images that are
   gradable but outside the training distribution.
7. **Both-eye fusion and longitudinal comparison** — accept OD/OS as a pair and compare against a
   patient's previous screening.
8. **Patient management and persistence** — a database, records and an audit trail; the design
   document exists, the implementation does not.
9. **Fairness audit with adequate metadata** — collect the demographic and device fields needed
   to make a subgroup claim that is actually supportable.
10. **Prospective evaluation.** Every number in this project is retrospective. A real reading of
    this system's worth requires deploying it alongside clinicians and measuring what changes.

---

## Research Paper / Publication

| Item | Details |
|---|---|
| Paper Title | *Confound-Controlled Evaluation of Multi-Disease Retinal Screening: Device, Source and Prevalence Effects* (working title) |
| Conference / Journal Name | `TO FILL` |
| Paper Status | Drafting |
| Submission Date | `TO FILL` |
| Paper Link | `TO FILL` |

The paper's contribution is the audit methodology rather than the classifier: the device- and
source-stratification protocol, the cross-disease admissibility matrix, and the argument that a
pooled AUC quoted without its confounds is not a usable screening number. Figures and tables are
collected in the development repository's `thesis_assets/`.

---

## References

```text
[1] V. Gulshan et al., "Development and Validation of a Deep Learning Algorithm for Detection of
    Diabetic Retinopathy in Retinal Fundus Photographs," JAMA, vol. 316, no. 22, pp. 2402-2410, 2016.
[2] International Committee for the Classification of Retinopathy of Prematurity, "The
    International Classification of Retinopathy of Prematurity Revisited," Archives of
    Ophthalmology, vol. 123, no. 7, pp. 991-999, 2005.
[3] M. Tan and Q. V. Le, "EfficientNetV2: Smaller Models and Faster Training," Proc. 38th
    International Conference on Machine Learning (ICML), pp. 10096-10106, 2021.
[4] K. He, X. Zhang, S. Ren and J. Sun, "Deep Residual Learning for Image Recognition," Proc.
    IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pp. 770-778, 2016.
[5] R. R. Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks via Gradient-Based
    Localization," Proc. IEEE International Conference on Computer Vision (ICCV), pp. 618-626, 2017.
[6] T.-Y. Lin, P. Goyal, R. Girshick, K. He and P. Dollar, "Focal Loss for Dense Object
    Detection," Proc. IEEE International Conference on Computer Vision (ICCV), pp. 2980-2988, 2017.
[7] Y. Cui, M. Jia, T.-Y. Lin, Y. Song and S. Belongie, "Class-Balanced Loss Based on Effective
    Number of Samples," Proc. IEEE/CVF Conference on Computer Vision and Pattern Recognition
    (CVPR), pp. 9268-9277, 2019.
[8] C. Guo, G. Pleiss, Y. Sun and K. Q. Weinberger, "On Calibration of Modern Neural Networks,"
    Proc. 34th International Conference on Machine Learning (ICML), pp. 1321-1330, 2017.
[9] X. Sun and W. Xu, "Fast Implementation of DeLong's Algorithm for Comparing the Areas Under
    Correlated Receiver Operating Characteristic Curves," IEEE Signal Processing Letters,
    vol. 21, no. 11, pp. 1389-1393, 2014.
[10] E. Decenciere et al., "Feedback on a Publicly Distributed Image Database: The Messidor
    Database," Image Analysis & Stereology, vol. 33, no. 3, pp. 231-234, 2014.
[11] P. Porwal et al., "Indian Diabetic Retinopathy Image Dataset (IDRiD): A Database for
    Diabetic Retinopathy Screening Research," Data, vol. 3, no. 3, art. 25, 2018.
[12] A. Bajwa, M. I. Malik et al., "G1020: A Benchmark Retinal Fundus Image Dataset for Computer-
    Aided Glaucoma Detection," Proc. International Joint Conference on Neural Networks (IJCNN),
    pp. 1-7, 2020.
[13] J. I. Orlando et al., "REFUGE Challenge: A Unified Framework for Evaluating Automated
    Methods for Glaucoma Assessment from Fundus Photographs," Medical Image Analysis, vol. 59,
    art. 101570, 2020.
[14] Z. Zhang et al., "ORIGA-light: An Online Retinal Fundus Image Database for Glaucoma Analysis
    and Research," Proc. Annual International Conference of the IEEE Engineering in Medicine and
    Biology Society (EMBC), pp. 3065-3068, 2010.
[15] A. J. DeGrave, J. D. Janizek and S.-I. Lee, "AI for Radiographic COVID-19 Detection Selects
    Shortcuts Over Signal," Nature Machine Intelligence, vol. 3, pp. 610-619, 2021.
[16] Kaggle, "Diabetic Retinopathy Detection (EyePACS)," 2015. [Online].
     Available: https://www.kaggle.com/c/diabetic-retinopathy-detection
[17] Kaggle, "APTOS 2019 Blindness Detection," 2019. [Online].
     Available: https://www.kaggle.com/c/aptos2019-blindness-detection
```

---

## Repository Update Guidelines

This repository is maintained per the guide's requirement of an update at least every two weeks.

* README updated with each milestone.
* The [Weekly Progress Updates](#weekly-progress-updates) table is extended every cycle.
* Figures, results and documentation are pushed as they are produced.
* Each update carries a meaningful commit message describing what changed and why.
* Model weights, datasets, patient images and temporary artefacts are **not** committed —
  no patient data of any kind appears in this repository.

---

## Declaration

We declare that this project work is carried out by our team as part of the BE Capstone Project.
The work will be regularly updated on GitHub and all references used will be properly cited.

We further declare that the results reported here are stated with the conditions under which they
were measured. Where an audit reduced a headline number — the ROP device confound, the glaucoma
source confound — the reduced figure is reported as the defensible one and the pooled figure is
labelled as such. Tests that failed are listed alongside those that passed.

---

## License

This project is for academic use only.

**Not a medical device.** No regulatory clearance. Not for clinical use.
