# Software

Implementation notes for the RetinAI codebase — what each part does and why it is built that way.
Headline metrics are in the [README](../README.md); the figures those metrics come from are in
[../images/](../images/).

This directory also holds [`make_diagrams.py`](make_diagrams.py), which generates the
architecture and flowchart images the README references:

```bash
python software/make_diagrams.py
```

---

## 1. The config-driven training framework

The central decision: **an experiment is a YAML file, not a code change.**

```bash
python train.py --config configs/rop.yaml
python train.py --config configs/dr.yaml --set train.epochs=2 model.arch=efficientnet_b0
```

`train.py` runs a fixed sequence: load config → set seed → pick device → build the pooled
manifest → build dataloaders → build the model from config → train → reload the best checkpoint →
evaluate on val and test → compute metrics → write plots → log.

Every run writes an auto-incrementing `results/experiment_NNN/` containing **the exact config that
produced it**, plus `meta.json` with the git hash. A result that cannot be traced back to the
configuration that produced it is not a result.

### `configs/`

| Config | Purpose |
|---|---|
| `dr.yaml` · `rop.yaml` · `glaucoma.yaml` | The three production models |
| `glaucoma_multisource.yaml` | SMDG-19 + G1020 pooled, stratified by source |
| `sweep.yaml` · `sweep_rop.yaml` | Architecture sweeps |
| `rop_staging.yaml` | Six-class ICROP staging (in progress) |
| `ablation_dr.yaml` | Cumulative ablation over preprocessing/loss components |
| `external_dr_messidor.yaml` · `external_glaucoma_refuge.yaml` | External validation, `test_split: 1.0` |

---

## 2. `models/common/` — the shared library

| Module | Responsibility |
|---|---|
| `config.py` | `Config` with recursive dot-access; `load_config(path, overrides)`; CLI `key=value` coercion; records `_source` |
| `datasets.py` | `FundusDataset` over a uniform `image_path,label,split` manifest. Returns a blank tensor on an unreadable file so one bad image never kills a batch |
| `data_prep.py` | `build_manifest` pools heterogeneous sources; `_stratified_split` for deterministic held-outs; `use_sources` filtering for the ablation |
| `preprocessing.py` | **The single source of truth.** `circle_crop(tol=7)`, `clahe(clip=2.0, grid=8)` on the LAB L-channel, `ben_graham(sigma)`, `build_transforms`, `build_tta_transforms` (5 views) |
| `architectures.py` | Factory over `efficientnet_b0/b3/v2_s`, `mobilenet_v3_large`, `resnet50`, `densenet121`, `convnext_tiny`; swaps each head for `Dropout → Linear`; `get_target_layer` returns the last conv for Grad-CAM |
| `losses.py` | `class_balanced_weights` (β = 0.9999), `FocalLoss`, `OrdinalRegressionLoss`. `build_loss` returns `(loss_fn, decode_fn, needs_float_target)` so the trainer stays head-agnostic |
| `metrics.py` | Accuracy, macro/weighted P/R/F1, QWK, confusion, sensitivity/specificity, AUC; `best_threshold_for_sensitivity`, `bootstrap_auc_ci(n_boot=2000)`, `referable_metrics` |
| `train_utils.py` | The one shared loop: AdamW + OneCycleLR + AMP + gradient accumulation + grad-clip + early stopping + optional Mixup + resume-safe per-epoch checkpointing. Plus `evaluate`, `predict_with_tta`, `plot_run` |
| `experiment_logger.py` | `set_seed`, `git_hash`, `pick_device`; the `results/experiment_NNN/` writer |
| `data_quality.py` | Pre-training QA — corrupt/label/resolution checks on all images; blur (variance-of-Laplacian) and perceptual-hash duplicate detection on a sample; emits a PDF report and a cleaned manifest |

**Why preprocessing is one module.** It is imported by both the trainer and the web application.
A screening system whose training and serving pipelines diverge measures one thing and deploys
another. Keeping it in one file makes that class of bug structurally difficult.

---

## 3. `models/evaluation/`, `validation/`, `comparison/`, `experiments/`

- **`evaluation/calibration.py`** — ECE, MCE, Brier, reliability diagrams, and post-hoc
  temperature scaling (LBFGS on one scalar).
- **`evaluation/statistical_tests.py`** — bootstrap CIs, McNemar (exact and chi-square), DeLong
  (fast implementation, Sun & Xu 2014).
- **`evaluation/error_analysis.py`** — per-class errors, top confused pairs, confidence quadrants;
  writes the offending images under `errors/<bucket>/` so failures can be looked at.
- **`validation/external_validation.py`** — evaluate a checkpoint on an external dataset with **no
  retraining**, report the internal→external drop with a PASS/REVIEW verdict.
- **`comparison/make_table.py`** — turns a sweep CSV into a ranked table plus a recommendation,
  and refuses to state a ranking the numbers do not support.
- **`experiments/ablation_study.py`** — applies the ablation config's steps cumulatively.

---

## 4. The training recipe

Common to all three diseases: seed 42, 384×384 input, AdamW (lr 2e-4, weight decay 1e-4),
OneCycleLR, AMP mixed precision, batch 16 × gradient accumulation 2 (effective 32), focal loss
γ = 2.0 with class-balanced weights and a weighted sampler, 5-view TTA, early stopping on the
disease's primary metric.

| | DR | ROP | Glaucoma |
|---|---|---|---|
| Architecture | EfficientNetV2-S | ResNet50 | EfficientNetV2-S |
| Classes | 5 | 2 | 2 |
| Primary metric | QWK (endpoint: referable ≥ 2) | recall | AUC |
| Preprocessing | circle-crop + Ben-Graham + CLAHE | circle-crop + CLAHE + strong aug | circle-crop + CLAHE |
| Epochs / patience | 25 / 7 | 30 / 8 | 25 / 7 |
| Operating point | argmax | threshold 0.1933 | threshold 0.5 |

**Patient-level splitting.** ROP images are grouped by patient (the first token of the filename)
and **patients** are split 70/15/15, not images. An infant contributes both eyes and repeat
exams; a naive image split would place the same infant in train and test and inflate every score.

---

## 5. The Kaggle-first workflow

Nothing trains on a laptop. `kaggle_kernels/` holds thin scripts that pin a P100-safe torch
build, discover mounted data by glob, extract archives, patch the YAML config to the discovered
paths, and call `prepare_data.py` + `train.py`.

Gotchas that are now baked into the kernels:

| Problem | Fix |
|---|---|
| Kaggle's default torch dropped the P100's `sm_60` arch — every CUDA op fails | Pin `torch==2.5.1+cu121` |
| 384px at batch 32 does not fit a 16 GB T4 | AMP + gradient accumulation (16 × 2) |
| EyePACS ships as split 7-zip archives | Install `p7zip-full`, extract, unzip the label CSV |
| SMDG images live under a doubled `full-fundus/full-fundus/` path; label column is `types` with −1 meaning unlabelled | Encoded in the config; the kernel also globs to find the real directory |
| Mount paths differ between datasets and competitions | Discover by glob, then patch the config |
| A local `kaggle/` directory shadowed the `kaggle` PyPI package | Renamed to `kaggle_kernels/` |
| 12-hour session cap kills long runs | Checkpoint every epoch to `last_checkpoint.pth` |

---

## 6. Serving — `webapp/`

Flask, `python webapp/app.py` (port 5002 local, 7860 in Docker).

**`webapp/inference.py`** holds a `Registry` that maps each disease to a config + weights path
from `registry.yaml` and loads all three (device cuda → mps → cpu). It **degrades gracefully**:
a missing checkpoint marks that disease unavailable and the other two still screen.

- `DiseaseModel.predict()` — 5-view TTA, apply the config's threshold (binary) or argmax
  (multiclass), return grade + confidence + risk + recommendation + per-class scores.
- `DiseaseModel.gradcam()` — generic across backbones via forward/backward hooks on the last
  conv layer; JET colormap overlay returned as a base64 PNG.
- `Registry.analyze(image)` — runs the admissible models and attaches one Grad-CAM.

### Routes

| Route | Does |
|---|---|
| `GET /` | Dashboard — model cards, pipeline animation |
| `GET /screen` | The screening tool |
| `GET /gallery` | Explainability gallery |
| `GET /diseases` | Disease education pages |
| `GET /history` | Recent screenings (in-memory) |
| `GET /about` | Model Card & Methodology |
| `POST /predict` | Run the admissible models, return JSON |
| `POST /report` | Generate and download the PDF |

**`reports/report_generator.py`** builds a one-page A4 clinical report with ReportLab: header,
patient block, per-disease findings with risk and recommendation, the fundus image and Grad-CAM
side by side, and the disclaimer.

---

## 7. The safety layer

Added **after** measurement, not designed in advance.

**Gradability gate.** Six cheap image statistics — redness ratio (R/B), saturation, red-hue
fraction, blur variance, luminance, field-of-view ratio — with thresholds calibrated on 400
images from the ROP **train** split only. The test split, HRF and EyePACS/APTOS were held out, so
the false-rejection rates are measurements rather than fits: **1.7% on 4,000 EyePACS images**,
and 6 of 6 synthetic non-retinal probes rejected.

Thresholds are set permissively on purpose. The gate's job is to reject obviously non-retinal
input, not to judge clinical image quality — and the positive side of its calibration is
dominated by neonatal imaging, which differs from adult fundus cameras.

**Patient-context router.** ROP is a disease of prematurity and cannot occur in an adult; DR and
glaucoma are not screened in a neonate. The cross-disease audit ran every test image through
every model and found the ROP model flagging **59 of 59** adult eyes. The router suppresses
findings outside the patient's world; without patient context, everything runs but is marked
unrouted.

---

## 8. Android client — `android/`

Kotlin, Jetpack Compose, Material 3.

- **Capture** — CameraX with focus, torch, an alignment overlay and a live quality meter.
- **On-device quality** — an advisory check calibrated at a fixed scale, so the user gets
  feedback before the upload rather than after.
- **Mandatory patient context** — scanning is gated on entering the age band, because the server
  needs it to route.
- **Result** — honours the server's routing, calibration and gate states rather than
  re-interpreting them client-side.
- **Report** — on-device PDF export, share and print; a file-backed report store.
- **Tests** — unit tests parse fixtures captured from live server responses, so a server contract
  change fails a test instead of silently breaking the app.

---

## 9. Deployment

`retinal-ai/` is a HuggingFace Space (Docker SDK): `python:3.11-slim`, CPU-only torch 2.1.0,
OpenCV system libs, non-root uid 1000, port 7860. Weights ship via Git LFS.

CPU-only is a deliberate constraint, not a limitation to apologise for — it demonstrates the
system runs where a clinic can actually afford to run it. The cost is latency: three models ×
5-view TTA + Grad-CAM take a few seconds per image.

---

## 10. Testing

- Unit tests over the gate, presentation rules, DTO parsing, URL and upload rules.
- Instrumented Android tests against captured fixtures.
- An end-to-end manual set of 93 labelled images — 84% overall correct.
- `scripts/make_graphs.py` regenerates all twelve evidence figures from `results/`, so a figure
  can never drift away from the number it depicts.
