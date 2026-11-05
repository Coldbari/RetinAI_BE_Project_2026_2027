# Challenges — Problems We Hit and How We Solved Them

The real engineering and modelling problems encountered building RetinAI, each with the fix
and where it is evidenced in the repo. These are the "war stories" — the parts that don't show
up in a results table but shaped every design decision. For what's still unsolved see
[LIMITATIONS.md](LIMITATIONS.md).

---

## Modelling & data-science problems

### 1. A single shared model plateaued → dedicated per-disease models
**Problem.** The first serious attempt was a **unified multi-task EfficientNet-B3** with one
shared backbone and two heads (ROP + DR) — `unified_model/multitask_training.py`. It trained,
but plateaued: ROP test accuracy 73.3% (ROP-positive recall only 0.585) and DR macro-F1 49.7%.
One backbone forced a compromise representation across very different diseases.
**Fix.** Rebuilt as a **config-driven framework with one dedicated model per disease**, each
free to pick its own architecture, resolution, preprocessing, and operating point. Result: ROP
AUC 0.927, DR referable AUC 0.906, Glaucoma AUC 0.973.
**Evidence.** `unified_model/` (superseded) vs `configs/` + `models/` + `webapp/registry.yaml`.

### 2. Data leakage risk → patient-level split
**Problem.** The ROP database has multiple images per patient (both eyes, repeat exams). A
naive random image split would put a patient's images in both train and test, inflating scores.
**Fix.** `data_prep/Splitting_data_with_patient_ID.py` groups images by patient ID (first token
of the filename) and splits **patients**, not images, 70/15/15 (seed 42). All of a patient's
images stay in one split. Verified counts: train 3,574 (131 patients) / val 928 (28) /
test 1,502 (29).
**Evidence.** `data_prep/Splitting_data_with_patient_ID.py`; `configs/rop.yaml` ("no leakage");
`dataset_split/`.

### 3. Class imbalance → focal loss + class-balanced weights + weighted sampler
**Problem.** DR is dominated by "No DR" (~73%); Mild/Severe grades are rare. The legacy model
used a crude `ROP_WEIGHT = 1.5` upweight.
**Fix.** Three coordinated mechanisms: **focal loss** (γ = 2.0) to focus on hard/minority
examples, **class-balanced weights** (Cui et al. 2019, effective number of samples), and a
**`WeightedRandomSampler`**. All disease configs enable `loss: focal`, `class_balanced: true`,
`weighted_sampler: true`.
**Evidence.** `models/common/losses.py`, `models/common/train_utils.py::make_weighted_sampler`.

### 4. Overfitting on the small ROP set → ResNet50 + mixup + strong aug
**Problem.** EfficientNetV2-S (the default) overfit the ~6k-image ROP set (test AUC 0.915 but
unstable). Small data punishes big, under-regularized backbones.
**Fix.** (a) A 5-architecture **sweep** picked **ResNet50** (AUC 0.964 vs V2-S 0.87) — it
generalizes far better on limited data. (b) Added a **Mixup** regularizer (classification heads
only). (c) `strong_aug: true` (affine/perspective/blur/erasing).
**Evidence.** `results/sweep_rop.md`, `configs/rop.yaml` (`arch: resnet50`, `strong_aug: true`),
`models/common/train_utils.py` (Mixup); commit "Add mixup regularizer (anti-overfit, esp. small ROP set)".

### 5. Domain shift — the flagship story (Glaucoma SMDG → G1020)
**Problem.** The Glaucoma model trained on SMDG-19 scored a strong internal AUC 0.957, but on
an **independent** source (G1020) it collapsed to **AUC 0.589 zero-shot** (−36.8%, a clear
FAIL). Investigation confirmed a **genuine domain shift** (G1020 imaging differs; the model
over-predicted glaucoma on the new source) — not a label flip.
**Fix.** **Multi-source training** — pool SMDG-19 + G1020 (13,336 imgs) with a stratified split
so both domains appear in train/val/test, and report **per-source** test AUC. G1020 recovered
from **0.589 → 0.957**; SMDG held at 0.974; overall 0.973.
**Honest caveat.** G1020 is now in-distribution, so a fresh zero-shot claim needs a *new*
held-out source (noted as future work).
**Evidence.** `configs/glaucoma_multisource.yaml`, `results/glaucoma/multisource_persource.json`,
`results/external_glaucoma.json`; commit "Add multi-source glaucoma config".

### 6. DR label-noise ceiling → referable-DR reframing
**Problem.** EyePACS 5-class labels have inter-grader agreement of only QWK ≈ 0.83–0.85, so a
5-class accuracy > 90% target is literally unachievable — you can't beat the labels. An early
"About" page even showed a misleading multiclass AUC 0.79.
**Fix.** Reframe the clinical endpoint to a binary **referable DR (grade ≥ 2)** screening
decision, reported as an AUC with a bootstrap CI. `referable_metrics()` collapses the ordinal
grades; the honest result is **referable AUC 0.906** (95% CI 0.89–0.92).
**Evidence.** `models/common/metrics.py::referable_metrics`, `bootstrap_auc_ci`; commits
"referable-DR screening eval" and "Final polish: honest web-app About".

### 7. External generalization + operating-point transfer
**Problem.** Internal numbers can be optimistic. Does the model hold on data it never saw?
**Fix.** A no-retraining external-validation harness (`test_split: 1.0`) with a PASS/REVIEW
verdict on the internal→external drop (target ≤ 10%). DR → **Messidor-2**: referable AUC
0.906 → 0.837 (−6.9%, **PASS**). It also revealed that the **operating point does not
transfer** (at sens 0.90 the external spec fell to 0.30) — documented as needing per-site
tuning.
**Evidence.** `models/validation/external_validation.py`, `results/external_dr.json`.

### 8. Over/under-confident probabilities → temperature-scaling calibration
**Problem.** Raw softmax scores are poorly calibrated (DR 5-class ECE 0.181).
**Fix.** Post-hoc **temperature scaling** (single scalar T via LBFGS) with reliability diagrams
and ECE/MCE/Brier reporting. DR 5-class ECE 0.181 → 0.133 (T = 0.78); DR referable 0.141 → 0.111
(T = 0.70); ROP 0.059 → 0.041.
**Evidence.** `models/evaluation/calibration.py`, `results/dr/analysis_summary.json`.
**Caveat.** This is computed offline and **not yet wired into the live app** — see
[LIMITATIONS.md](LIMITATIONS.md).

### 9. Data-quality problems before training
**Problem.** 35k+ scraped fundus images contain corrupt files, duplicates, blurry captures, and
bad labels.
**Fix.** A Dataset-QA pass (`models/common/data_quality.py`): corrupt/label/resolution checks on
all images; blur (variance-of-Laplacian) and perceptual-hash duplicate detection on a random
`deep_sample=2000` (to stay tractable on EyePACS); emits `dataset_report.pdf` + a cleaned
manifest.
**Evidence.** `models/common/data_quality.py`, `prepare_data.py --dqa`.

---

## Infrastructure & Kaggle problems

The training is **Kaggle-first**; these gotchas are baked into `kaggle_kernels/` and documented
in [kaggle_kernels/README.md](../kaggle_kernels/README.md).

### 10. torch / P100 "no kernel image" (`sm_60`)
**Problem.** Kaggle's default image ships `torch 2.x+cu128`, which **dropped the P100 GPU's
`sm_60` arch** — every CUDA op fails with *"no kernel image is available"*.
**Fix.** Every kernel pins **`torch==2.5.1+cu121`** (works on both P100 and T4, and makes runs
deterministic).
**Evidence.** `kaggle_kernels/README.md` §gotchas; `train_rop.py`/`train_dr.py`/`train_glaucoma.py`.

### 11. GPU out-of-memory at 384px → AMP + gradient accumulation
**Problem.** 384px training at batch 32 doesn't fit a 16 GB T4.
**Fix.** **AMP** mixed precision (`torch.amp.autocast` + `GradScaler`, auto-disabled off CUDA)
plus **gradient accumulation** (`grad_accum: 2`, batch 16 → effective 32).
**Evidence.** `models/common/train_utils.py`, `configs/dr.yaml` ("effective batch 32 at 384px on a T4").

### 12. EyePACS split 7-zip archives
**Problem.** EyePACS images ship as split archives `train.zip.001..005` and a zipped label CSV.
**Fix.** `train_dr.py` installs `p7zip-full`, extracts to `/tmp/train/train` (~20 min), and
unzips `trainLabels.csv.zip`.
**Evidence.** `kaggle_kernels/README.md`, `train_dr.py`.

### 13. SMDG doubled path + non-obvious label column
**Problem.** SMDG images live under a **doubled** `full-fundus/full-fundus/<name>.png`, and the
label column is `types` (0/1, with −1 = unlabelled to drop) — easy to miswire.
**Fix.** `configs/glaucoma_multisource.yaml` encodes `images_dir: .../full-fundus/full-fundus`,
`label_col: types`, `drop_label: -1`; the kernel also **globs** to discover the real image dir.
**Evidence.** `kaggle_kernels/README.md`, `configs/glaucoma_multisource.yaml`, `train_glaucoma.py`.

### 14. Fragile mount paths → glob discovery
**Problem.** Kaggle mounts datasets at `/kaggle/input/datasets/<owner>/<slug>/...` and
competitions at `/kaggle/input/competitions/<slug>/...`; hard-coding paths breaks constantly.
**Fix.** Kernels **discover paths by glob** rather than hard-coding, then patch the YAML config.
**Evidence.** `kaggle_kernels/*` , `kaggle_kernels/README.md`.

### 15. `import kaggle` shadowed by a local `kaggle/` folder
**Problem.** A local directory named `kaggle` shadowed the Kaggle PyPI package, breaking
`import kaggle`.
**Fix.** Renamed the folder `kaggle/` → **`kaggle_kernels/`**.
**Evidence.** commit "rename kaggle/ -> kaggle_kernels/ to avoid shadowing the kaggle PyPI package".

### 16. 12-hour session cap + 2-session concurrency → resume-safe checkpointing
**Problem.** Kaggle kills GPU sessions at 12 h and allows only 2 concurrent GPU sessions.
**Fix.** `train.py` checkpoints **every epoch** (`last_checkpoint.pth`) so a killed session
never loses progress; the workflow runs at most two models at once.
**Evidence.** `kaggle_kernels/README.md`, `models/common/train_utils.py`.

### 17. Inference cold-start pulling ImageNet weights
**Problem.** Loading a trained checkpoint still triggered a torchvision ImageNet download
(slow, and pointless since the weights are overwritten).
**Fix.** Build the net with `pretrained=False` at inference and skip the download.
**Evidence.** `webapp/inference.py` (`build_from_cfg(cfg, pretrained=False)`); commit "Skip
ImageNet download when loading trained weights for inference".

---

## Product & repo footguns

### 18. Ambiguous root entry point launched the OLD app
**Problem.** A stale `./app.py` at the repo root launched the *old single-model* app; running
`python app.py` from root silently started the wrong product instead of `webapp/app.py`.
**Fix.** Moved it to `legacy/app_unified_v1.py` (a pure rename) so the only real entry point is
`python webapp/app.py`.
**Evidence.** commit "Move stale root app.py -> legacy/ (footgun: it launched the OLD single-model app)".

### 19. Deployment mirror drift
**Problem.** The HuggingFace Space (`retinal-ai/`) is a separate git repo mirroring `webapp/`;
it drifted behind (e.g. was missing `journey.js`), so the deployed dashboard differed from local.
**Fix.** A disciplined sync-then-push flow (copy changed `webapp/` files into `retinal-ai/webapp/`,
verify parity with `diff -rq`, commit + push to the HF remote). Documented in
[PROJECT_OVERVIEW.md § Deployment](PROJECT_OVERVIEW.md).

### 20. Honest reporting over flattering numbers
**Problem.** Early web copy showed flattering-but-misleading metrics (multiclass AUC 0.79, no
external number).
**Fix.** Replaced with honest internal **and** external numbers, per-grade tables, calibration,
and explicit limitations — the same honesty this documentation set continues.
**Evidence.** commits "Final polish: honest web-app About", "full Model Card & Methodology About page".

---

**Takeaway.** Most of the project's quality came not from a bigger model but from *catching
problems*: leakage, imbalance, domain shift, label noise, and a dozen infrastructure traps.
The ones still open are in [LIMITATIONS.md](LIMITATIONS.md); the next steps are in
[ROADMAP.md](ROADMAP.md).
