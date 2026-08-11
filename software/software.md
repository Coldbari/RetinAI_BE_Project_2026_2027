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

## 8. ROP 6-class ICROP staging (in progress)

The deployed ROP model is binary. The staging work extends it to the six ICROP classes —
Normal, Stage 1, Stage 2, Stage 3, Stage 4/5, AP-ROP.

### Corpus and splits

`scripts/build_rop_staging_manifest.py` consolidates four sources; `scripts/make_staging_folds.py`
builds patient-grouped 5-fold splits. One site (`multiview`) is held out entirely and divided into
a **dev** set that may be scored freely and a **locked** set that is scored once, at the end.

| | Images | Infants |
|---|---|---|
| Train pool | 3,112 | 1,528 |
| Held-out site — dev | 663 | 116 |
| Held-out site — locked | 661 | 116 |

Per-class training counts: Normal 1,109 · Stage 1 515 · Stage 2 696 · Stage 3 535 ·
**Stage 4/5 89** · AP-ROP 168. Imbalance ratio 12.46.

`scripts/anonymise_ostrava_tree.py` de-identifies the Ostrava source before anything reads it.
One patient-grouping bug was found and fixed there: IDs 40 and 41 were the same infant, which
would have split one baby across folds.

### The five-gate shortcut audit — `scripts/rop_shortcut_audit.py`

No staging number is trusted until all five gates run. This exists because of the device confound
found earlier in the binary ROP work — the lesson was that the audit has to come *before* the
result, not after it.

| Gate | Asks | Result |
|---|---|---|
| **G1** patient leakage | Is any infant in two folds? | Pass — 0 |
| **G2** duplicate leakage | Any exact or near-duplicate images across folds? | Pass — 0 exact MD5, 0 near-dupes, closest pHash distance 15 |
| **G3** site decodability | Can a model tell which site an image came from? | **Balanced accuracy 0.9973** vs 0.3333 chance. Stage 4/5 comes from a single source |
| **G4** metadata-only baseline | What score is achievable with **no pixels decoded**? | **0.187 macro-F1** — the bar every image model must clear |
| **G5** disease-controlled site probe | Does training *add* site information to the embedding? | Trained 0.882 vs untrained 0.894 on normal-only images — pass |

G4 is the gate that makes the rest meaningful. A 6-class macro-F1 of 0.50 sounds mediocre in
isolation; against a metadata-only bar of 0.19 it is evidence the network is reading retinas.

G5 reads the **normal-only** probe deliberately. Running the probe over all classes lets
site-varying disease prevalence inflate the score, so it would measure prevalence rather than
appearance.

### The benchmark — `kaggle_kernels/rop_staging_benchmark.py`

Five backbones spanning 4.0M–27.8M parameters and three families (CNN, ViT, hybrid):
EfficientNet-B0, EfficientNetV2-S, ConvNeXt-T (in12k), DeiT3-S, CAFormer-S18.

`models/comparison/make_table.py` **refuses to state a ranking the numbers do not support**. On
one fold and one seed the spread is 0.028 macro-F1, so the reported conclusion is "no model is
separable from another", not "EfficientNetV2-S wins". Full 5-fold CV is running.

### The structured head — `configs/rop_staging_structured.yaml`

Three changes to the flat 6-way softmax, each motivated by something clinical:

1. **Ordinal CORN head** — ROP stages are ordered. A flat softmax treats Stage 1 and Stage 4/5 as
   equally distant from Stage 2, discarding the ordering.
2. **Separate AP-ROP branch** — AP-ROP is not a point on the stage ladder; it is an aggressive
   posterior form that co-occurs with stages. Forcing it into the same ordinal axis is wrong.
3. **Site adversary (DANN)** — a gradient-reversal branch predicting the source site, to
   discourage the encoder from representing provenance.

Result on fold 0: macro-F1 **0.554** (vs 0.520 flat), accuracy 0.684, QWK 0.632, AUC 0.868.

### Two findings worth carrying out of this project

**The label unit, not the model.** AP-ROP scored recall 0.311 per image. The cause is annotation
granularity: AP-ROP is labelled per examination *session*, and a session has a median of 18 frames
of which roughly 47% score below the model's 10th percentile — those frames do not show the
pathology, but they carry the session's label. Scored at the session level, recall is **0.905**
(19 of 21) with a **0.000** false-positive rate on normals and precision 0.980.
`scripts/rop_session_level_analysis.py` does this.

**Naive probes overstate adversarial debiasing.** `results/rop/dann_probe_ablation.json`:

| Model | Naive probe | Disease-controlled probe |
|---|---|---|
| Untrained | 0.890 | 0.894 |
| Flat, no DANN | 0.859 | 0.882 |
| Structured + DANN (w_site 0.5) | 0.820 | 0.885 |

The naive probe drops 0.039 and looks like success. The disease-controlled probe does not move.
DANN removed the disease–site *correlation*, not site *appearance*. An ablation at w_site = 2.0
is running to see whether a stronger adversary moves the controlled probe, and what it costs the
clinical metrics.

---

## 9. Android client — `android/`

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

## 10. Deployment

`retinal-ai/` is a HuggingFace Space (Docker SDK): `python:3.11-slim`, CPU-only torch 2.1.0,
OpenCV system libs, non-root uid 1000, port 7860. Weights ship via Git LFS.

CPU-only is a deliberate constraint, not a limitation to apologise for — it demonstrates the
system runs where a clinic can actually afford to run it. The cost is latency: three models ×
5-view TTA + Grad-CAM take a few seconds per image.

---

## 11. Testing

- Unit tests over the gate, presentation rules, DTO parsing, URL and upload rules.
- Instrumented Android tests against captured fixtures.
- An end-to-end manual set of 93 labelled images — 84% overall correct.
- `scripts/make_graphs.py` regenerates all twelve evidence figures from `results/`, so a figure
  can never drift away from the number it depicts.
