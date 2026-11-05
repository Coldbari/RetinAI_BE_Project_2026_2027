# Project Overview — What We Built

A complete technical tour of RetinAI: the config-driven training framework, the three
dedicated disease models, the evaluation/validation rigor, the Kaggle-first workflow, the
web application, and deployment. For headline numbers see [RESULTS.md](RESULTS.md); for the
problems behind these choices see [CHALLENGES.md](CHALLENGES.md); for what is *not* solid see
[LIMITATIONS.md](LIMITATIONS.md).

---

## 1. The idea and the approach

**Goal:** screen one colour fundus photograph for three sight-threatening diseases —
Diabetic Retinopathy, Retinopathy of Prematurity, and Glaucoma — and return an
*explainable, graded, reportable* result a clinician can act on.

**Key design decision — dedicated models, not one shared network.** An earlier version of
this project used a single multi-task EfficientNet-B3 with two heads (ROP + DR). It worked
but plateaued at weak numbers (ROP test acc 73%, DR macro-F1 50%). The current system trains
**one dedicated model per disease**, each free to use the architecture, resolution,
preprocessing, and operating point that suits it. That decoupling is what unlocked the jump
to ROP AUC 0.927, DR referable AUC 0.906, and Glaucoma AUC 0.973. The evolution is detailed
in [§8](#8-project-evolution).

**Framing:** a **decision-support screening assistant**, never an autonomous diagnostic
device. Every output carries that disclaimer end-to-end.

---

## 2. System architecture

```
                             ┌───────────────────────── TRAINING (Kaggle GPU) ─────────────────────────┐
  Kaggle datasets ─► kaggle_kernels/train_*.py ─► prepare_data.py ─► train.py ──► results/<disease>/
  (EyePACS, APTOS,     (pin torch, glob mounts,     build_manifest      config-driven      weights.pth
   SMDG, G1020, ROP)    extract archives)           (pool sources)      trainer loop       metrics.json
                                                                                            plots/
                             └────────────────────────────────────┬──────────────────────────────────┘
                                                                   │ pull weights back (git LFS)
                             ┌─────────────────────────────────────▼───────────── SERVING (CPU) ───────┐
  Fundus image ─► webapp/app.py ─► webapp/inference.py Registry ─► 3 DiseaseModels (shared preprocess + TTA)
   (upload)         Flask routes      loads registry.yaml            ├─ predict() → grade + risk + confidence
                                                                     └─ gradcam() → heatmap overlay
                                                             ─► reports/report_generator.py → PDF
                             └──────────────────────────────────────────────────────────────────────────┘
```

The same preprocessing module (`models/common/preprocessing.py`) runs in both training and
serving — a single source of truth so there is no train/serve skew.

---

## 3. The config-driven training framework

> *"Everything (disease, dataset, architecture, preprocessing, hyperparameters, resolution)
> is driven by a single YAML file so a new experiment is a new config, not a code edit."*
> — `models/common/config.py`

### Entry points (root scripts that import the `models` package)
| Script | Purpose |
|---|---|
| `train.py` | Train one disease from a config. `--config`, `--set key.subkey=value` overrides, `--tag`. |
| `prepare_data.py` | Build the pooled manifest; `--dqa` runs Dataset-QA and emits a PDF report. |
| `evaluate.py` | Score a `--weights` checkpoint and print an Acceptable/Target/Stretch tier. |
| `benchmark.py` | Single-model latency **and** the architecture-sweep runner. |
| `predict.py` | One-off CLI prediction. |

`train.py` flow: `load_config` → `set_seed` → `pick_device` → `get_manifest` →
`build_dataloaders` → `build_from_cfg` → `ExperimentLogger` → `train_model` → reload best →
`evaluate` on val (+ test) → `compute_metrics` → `plot_run` → `log_metrics`.

### `configs/` — one YAML per experiment
Ten configs: three disease configs (`dr.yaml`, `rop.yaml`, `glaucoma.yaml` +
`glaucoma_multisource.yaml`), two architecture sweeps (`sweep.yaml`, `sweep_rop.yaml`), one
cumulative ablation (`ablation_dr.yaml`), and two external-validation configs
(`external_dr_messidor.yaml`, `external_glaucoma_refuge.yaml`).

**Common recipe (all diseases):** seed 42, image size **384**, AdamW (`lr 2e-4`,
`weight_decay 1e-4`), **OneCycleLR**, **AMP** mixed precision, batch 16 × `grad_accum 2`
(effective 32), **focal loss** (γ = 2.0) + **class-balanced weights** + **weighted sampler**,
5-view **TTA**, early stopping on the disease's primary metric.

**Per-disease specifics:**

| | DR | ROP | Glaucoma |
|---|---|---|---|
| Architecture | EfficientNetV2-S | **ResNet50** (won the sweep) | EfficientNetV2-S |
| Classes | 5 (No/Mild/Moderate/Severe/Proliferative) | 2 (No ROP / ROP) | 2 (Non-glaucoma / Glaucoma) |
| Primary metric | QWK (referable grade ≥ 2) | recall | AUC |
| Preprocess | circle-crop + Ben-Graham + CLAHE | circle-crop + CLAHE + `strong_aug` | circle-crop + CLAHE (+ optic-disc ROI single-source) |
| Epochs / patience | 25 / 7 | 30 / 8 | 25 / 7 |
| Operating point | argmax (5-class); referable_grade 2 | thr **0.1933** (sens 0.990 / spec 0.398, TTA) | thr 0.5 (argmax) |

### `models/common/` — the shared library
- **`config.py`** — `Config` (recursive dot-access), `load_config(path, overrides)`, CLI `key=value` coercion, records `_source` for reproducibility.
- **`datasets.py`** — `FundusDataset` over a uniform manifest (`image_path,label,split`); returns a blank tensor on unreadable files so a batch never crashes.
- **`data_prep.py`** — `build_manifest` pools sources into `results/prepared/<disease>/manifest.csv`. Handles `csv` sources (EyePACS/APTOS/SMDG/G1020) and `imagefolder_split` (ROP); `_stratified_split` for deterministic held-outs; `use_sources` filters (used by the ablation).
- **`preprocessing.py`** — the shared fundus pipeline: `circle_crop(tol=7)`, `clahe(clip=2.0, grid=8)` on the LAB L-channel, `ben_graham(sigma)` (`addWeighted(img,4,blur,-4,128)`). `build_transforms` adds ImageNet normalize + train aug (flips, rotation, jitter; `strong_aug` adds affine/perspective/blur/erasing). `build_tta_transforms` = **5 views** (original + H-flip + V-flip + ±10° rotation), averaged.
- **`architectures.py`** — factory over `efficientnet_b0/b3/v2_s, mobilenet_v3_large, resnet50, densenet121, convnext_tiny`; swaps each head for `Dropout → Linear`; `get_target_layer` returns the last conv for Grad-CAM.
- **`losses.py`** — `class_balanced_weights` (Cui et al. 2019, β = 0.9999), `FocalLoss`, `OrdinalRegressionLoss`; `build_loss` returns `(loss_fn, decode_fn, needs_float_target)` so the trainer is head-agnostic.
- **`metrics.py`** — `compute_metrics` (acc, macro/weighted P/R/F1, QWK, confusion, sensitivity/specificity, AUC), `best_threshold_for_sensitivity(0.90)`, `bootstrap_auc_ci(n_boot=2000)`, `referable_metrics` (collapse ordinal DR to binary grade ≥ k with AUC + CI), `tier()`.
- **`train_utils.py`** — the single shared loop: AdamW + OneCycleLR + AMP GradScaler + gradient accumulation + grad-clip + early stopping + optional **Mixup** + resume-safe per-epoch `last_checkpoint.pth`. Plus `evaluate`, `predict_with_tta`, `plot_run` (a 3×3 diagnostic panel).
- **`experiment_logger.py`** — `set_seed`, `git_hash`, `pick_device`; `ExperimentLogger` auto-increments `results/experiment_NNN/` and writes `config.yaml`, `meta.json`, `weights.pth`, `metrics.json`, `plots/`, `errors/`. **Any run is reproducible from its saved `config.yaml`.**
- **`data_quality.py`** — pre-training QA: corrupt-file/label/resolution checks (all images), blur (variance-of-Laplacian) + perceptual-hash duplicate detection (sampled `deep_sample=2000` to stay tractable on 35k EyePACS images); emits `dataset_report.pdf` + a cleaned manifest.

### `models/evaluation/`, `models/validation/`, `models/comparison/`, `models/experiments/`
- **evaluation/calibration.py** — ECE/MCE/Brier, reliability diagrams, post-hoc **temperature scaling** (LBFGS on one scalar T).
- **evaluation/statistical_tests.py** — bootstrap CIs, **McNemar** (exact/chi-square), **DeLong** (fast, Sun & Xu 2014) for comparing AUCs.
- **evaluation/error_analysis.py** — per-class errors, top-confused pairs, confidence quadrants, saves offending images under `errors/<bucket>/`.
- **validation/external_validation.py** — evaluate a checkpoint on an external dataset with **no retraining** (`test_split: 1.0`), report internal→external **drop** with a PASS/REVIEW verdict (target ≤ 5–10%).
- **comparison/make_table.py** — turns a sweep CSV into a ranked markdown/CSV comparison table + a recommendation.
- **experiments/ablation_study.py** — applies the ablation config's steps **cumulatively**, retraining a short model per step to measure each component's contribution.

### `scripts/`
- **`generate_thesis_assets.py`** — collects every figure/table across runs into `thesis_assets/` as `Figure_NN.png` / `Table_NN.csv` with a source manifest (copy-only).

---

## 4. The three disease models

### Diabetic Retinopathy (DR) — EfficientNetV2-S, 5-class
- **Data:** EyePACS (`diabetic-retinopathy-detection`, ~35,126) + APTOS 2019 (`aptos2019-blindness-detection`, ~3,662) ≈ **38,788** images, stratified 15% val (val set ≈ 5,815).
- **Endpoint:** the honest screening decision is **referable DR (grade ≥ 2)**, reported as an AUC; 5-class grading is reported too but is capped by label noise (see [CHALLENGES §DR label-noise ceiling](CHALLENGES.md)).
- **External:** Messidor-2 (1,744 imgs), no retraining.

### Retinopathy of Prematurity (ROP) — ResNet50, binary
- **Data:** an infant retinal database, **6,004** images / **188** patients, split **by patient** (train 3,574 / val 928 / test 1,502) to prevent leakage. Binary collapse of the raw diagnosis code: `DG0` = No ROP, `DG1` = ROP.
- **Architecture:** ResNet50 won a 5-architecture sweep on the small ROP set (AUC 0.964 vs EfficientNetV2-S 0.87) — it generalizes far better on limited data.
- **Operating point:** threshold **0.1933**, selected on the clean **val** split for sensitivity ≥ 0.90 — screening favours catching disease over avoiding false alarms. Measured once on test through the served 5-view TTA path: **sensitivity 0.990 / specificity 0.398**. The earlier 0.371 (sens 0.90 / spec 0.83) is withdrawn — it was selected by reading the test ROC, which overstated specificity by +0.41.
- **External:** none (no public ROP dataset exists) — a known limitation.

### Glaucoma — EfficientNetV2-S, binary, multi-source
- **Data:** SMDG-19 (`deathtrooper/multichannel-glaucoma-benchmark-dataset`, ~12,449) **+** G1020 (1,020) = **13,336** images, stratified so both sources appear in train/val/test.
- **Why multi-source:** the SMDG-only model scored AUC 0.957 internally but collapsed to **0.589 zero-shot on G1020** — a genuine domain shift. Pooling G1020 into training recovered it to **0.957** while SMDG held at 0.974. This is the project's flagship rigor story (see [CHALLENGES §Domain shift](CHALLENGES.md)).

Full metric tables in [RESULTS.md](RESULTS.md).

---

## 5. The Kaggle-first training workflow

All training runs on **Kaggle GPU** (P100/T4); the desktop only does inference, reporting, and
pulling weights back. `kaggle_kernels/` holds thin scripts (`train_rop.py`, `train_dr.py`,
`train_glaucoma.py`) that pin a P100-safe torch build, discover mounted data by glob, extract
archives, patch the YAML config to the discovered paths, and call `prepare_data.py` + `train.py`.
The workflow (upload code as a private Kaggle dataset → `kaggle kernels push/status/output` →
drop `weights.pth` into `results/<disease>/`) and its hard-won gotchas are documented in
[kaggle_kernels/README.md](../kaggle_kernels/README.md) and [CHALLENGES.md](CHALLENGES.md).

---

## 6. The web application (`webapp/`)

A Flask app titled **RetinAI**, run with `python webapp/app.py` (port 5002 local; 7860 in Docker).

### Model serving — `webapp/inference.py` + `webapp/registry.yaml`
- `registry.yaml` maps each disease to a `config` + `weights` path (`results/<disease>/weights.pth`).
- `Registry` loads all three into `DiseaseModel`s (device cuda → mps → cpu). **Graceful
  degradation:** a missing `.pth` → that disease shows "model not loaded" and the app still runs.
- `DiseaseModel.predict()` runs 5-view TTA, applies the config `screen_threshold` (binary) or
  argmax (multiclass), and returns grade + confidence + risk + recommendation + per-class scores.
- `DiseaseModel.gradcam()` — generic Grad-CAM for any backbone via forward/backward hooks on
  the last conv layer → JET colormap overlay (base64 PNG, 320²).
- `Registry.analyze(image)` runs all three models and attaches **one** Grad-CAM (the first
  abnormal finding).

### Routes (`webapp/app.py`)
| Route | Does |
|---|---|
| `GET /` | Dashboard (model cards, "How it works" animated pipeline, journey, team) |
| `GET /screen` | The screening tool (upload + patient fields) |
| `GET /gallery` | Explainability gallery (pre-computed Grad-CAM examples) |
| `GET /diseases` | Disease education pages |
| `GET /history` | Recent screenings (in-memory `deque(maxlen=24)`) |
| `GET /about` | **Model Card & Methodology** (datasets, preprocessing, architectures, metrics, validation) |
| `POST /predict` | Run all three models, return JSON, push a history record |
| `POST /report` | Re-run inference + `generate_pdf` → download a PDF |
| `GET/POST /history/item·delete·report/<hid>` | View / delete / re-report a stored screening |

### PDF reports — `reports/report_generator.py`
`generate_pdf(...)` (ReportLab, A4) builds a one-page clinical report: dark header, patient
block, per-disease AI findings with risk + recommendation (`RISK_RULES`), the fundus image
and Grad-CAM side by side, and the "not a diagnostic device" disclaimer. Written to
`results/reports/report_*.pdf`.

---

## 7. Deployment

- **`retinal-ai/`** is a nested git repo tracked in the parent as a **gitlink** — the
  HuggingFace **Space** (`huggingface.co/spaces/Champ610/retinal-ai`). Its `webapp/` is a
  byte-identical mirror of the top-level `webapp/`.
- **`retinal-ai/Dockerfile`** (the live one): `python:3.11-slim`, CPU-only torch
  (`2.1.0+cpu`), OpenCV libs, `pip install -r requirements.txt`, copies `webapp/ models/
  configs/ reports/ results/`, runs as non-root uid 1000, `CMD python webapp/app.py`, port 7860.
- **HF config:** `retinal-ai/README.md` front-matter (`sdk: docker`, `app_port: 7860`).
  Weights ship via **Git LFS** (`.gitattributes` routes `*.pth`).
- **CPU-only by design** — a few seconds per image on the free tier.
- ⚠️ The **root** `Dockerfile` still deploys the *legacy* `unified_model/` single-model app —
  do not use it. The real product is `retinal-ai/Dockerfile`. (Noted in [LIMITATIONS.md](LIMITATIONS.md).)

To deploy: sync changed `webapp/` files into `retinal-ai/webapp/`, commit in `retinal-ai/`,
`git push origin main` → HF rebuilds.

---

## 8. Project evolution

| Generation | What | Where | Status |
|---|---|---|---|
| **v0** — single-disease ROP | EfficientNet-B2 ROP-only Flask app | `legacy/app_unified_v1.py` | quarantined (footgun) |
| **v1** — unified multi-task | one EfficientNet-B3, ROP + DR heads, shared backbone | `unified_model/` | superseded (ROP 73% / DR F1 50%) |
| **v2** — config-driven framework | dedicated per-disease models, YAML configs, shared library | `configs/`, `models/`, `train.py` | **current** |
| **v2** — rigor | external validation, calibration, statistical tests, arch sweep, glaucoma multi-source fix | `models/evaluation`, `validation` | **current** |
| **v2** — product | clinic-grade multi-page web app + HF deployment | `webapp/`, `retinal-ai/` | **current** |

The pivots between generations were driven by concrete failures — a single shared backbone
plateauing, EyePACS label noise capping 5-class accuracy, a glaucoma model that didn't
transfer across sources. Each is written up with its evidence in [CHALLENGES.md](CHALLENGES.md).

---

## See also
- [RESULTS.md](RESULTS.md) — every metric, grounded in `results/*.json`
- [CHALLENGES.md](CHALLENGES.md) — the problems behind these design choices
- [LIMITATIONS.md](LIMITATIONS.md) — what is not yet solid, incl. implementation gaps
- [ROADMAP.md](ROADMAP.md) — what comes next
- [future/](future/) — design docs for patient management, camera integration, real-time pipeline
