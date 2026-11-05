# Model Card — RetinAI ROP Staging (structured EfficientNetV2-S)

*All figures recomputed 2026-09-21 from raw prediction artifacts at commit `f0d4508`.
Uncertainty conventions are stated explicitly wherever a ± appears.*

## 1. Model description

| | |
|---|---|
| Task | 6-class ICROP-3 staging of retinopathy of prematurity from a single fundus photograph |
| Type | Multi-head image classification with an ordinal component |
| Backbone | EfficientNetV2-S, ImageNet-pretrained, classifier stripped → 1280-d embedding |
| Heads | CORN ordinal stage (4 conditional logits) · AP-ROP binary branch · DANN site adversary (training only) |
| Decode | AP-ROP if σ(arop) > 0.5, else CORN rank. `prob6 = [(1−pₐ)·P(rank=k)…, pₐ]` |
| Parameters | 20,512,600, all trainable; the contributed heads are **6,405 params (0.031%)** |
| Checkpoint | fold-0 checkpoint of the seed-42 run, SHA-256 `11b4396086…` (see `CURRENT_MODEL_ID.txt`) |
| Licence constraint | the external test hospital's dataset is **CC BY-NC** |
| Status | research prototype; **not a medical device** |

## 2. Dataset

3,112 training images / 1,528 infants / 3 hospitals, after per-infant caps (normal capped at 2 per infant). Class counts: normal 1,109 · stage_1 515 · stage_2 696 · stage_3 535 · **stage_4_5 89** · arop 168 (**12.46:1** imbalance).

Internal evaluation: 6,700 images across 5 infant-grouped folds (val folds carry natural prevalence and exactly partition the corpus). External: 663 dev + **661 locked** images, 116 infants each, one hospital never used in training.

**Class × source confounding is real and partly under-declared upstream:** stage_4_5 is 100% ROP-VL, AP-ROP has **zero** Shenzhen images, and Stage 3 is only 7% Ostrava. Hospital identity is linearly decodable from file metadata at 0.997 balanced accuracy. A metadata-only classifier scores macro-F1 0.187, which is the floor the image model must beat — it does (0.692). See `02_DATASET/class_by_source.png`.

## 3. Preprocessing

`circle_crop → letterbox_square → CLAHE → Resize(384,384) → ImageNet normalise`. The letterbox is deliberate anti-shortcut engineering: without it, resizing squashes each hospital by a different factor, and this project previously measured AUC 0.911 from image dimensions alone.

Training adds flips, ±15° rotation, colour jitter, hue/saturation jitter, RandomGrayscale, resolution jitter and RandomResizedCrop(0.6–1.0). **Verified: training-val, the committed evaluation artifacts, and the served web app use byte-identical preprocessing.** The served path runs **one forward pass — no TTA**.

## 4. Architecture rationale and its measured pay-off

The ordinal head encodes that stages are ordered; the separate AP-ROP branch encodes that ICROP-3 treats the aggressive form as a *form*, not a rank; the adversary attacks hospital identity in the embedding. The design is principled. Its measured pay-off over a plain 6-way softmax on the same backbone, folds and metrics is **nil on every aggregate metric** and **negative on calibration** (§10, §12). The one clear win is taxonomy fidelity: the model emits an independent AP-ROP probability and monotone ordinal ranks.

## 5. Training

AdamW · lr 3e-4 · weight decay 0.05 · OneCycleLR (pct_start 0.1, cosine) · batch 16 · 20 epochs · AMP · grad-clip 1.0 · **natural-prevalence sampling** · last-epoch weights (`early_stopping: false`).

Training loss falls smoothly 1.83 → ~0.45 with no divergence. Validation macro-F1 is **highly unstable**: mean absolute change between consecutive epochs 0.040–0.065, maximum 0.31. SD of validation macro-F1 over the **last five epochs** is 0.007–0.056. That end-of-schedule noise is **larger than the entire structured-vs-flat difference (0.0055)**, and it is not propagated into any reported figure. `05_TRAINING/`.

**No validation loss and no learning-rate trace were recorded.** The LR curve in this package is reconstructed from the trainer's hardcoded schedule and is labelled as such.

## 6. Validation — internal, 5-fold out-of-fold (n = 6,700 images, 1,528 infants)

| metric | value | uncertainty |
|---|---|---|
| macro-F1 (6 classes) | 0.6923 | ± 0.0861 **fold-to-fold SD**; pooled 0.6517 [0.572, 0.760] patient-clustered 95% CI |
| accuracy | 0.7861 | [0.697, 0.857] patient-clustered 95% CI |
| balanced accuracy | 0.6524 | [0.616, 0.757] |
| **QWK (corrected, 5 ordinal classes)** | **0.8964** | ± 0.0288 fold SD; pooled 0.8969 [0.869, 0.923] |
| QWK (legacy 6-class — superseded) | 0.8255 | ± 0.1454 — see §16 |
| unweighted Cohen's κ | 0.6549 | |
| "any ROP" AUC / AP | 0.9695 / 0.9662 | prevalence 0.427 |

Per class (pooled, precision / recall / support): Normal 0.929 / 0.951 / 3,837 · **Stage 1 0.376 / 0.499 / 577** · Stage 2 0.559 / 0.631 / 959 · Stage 3 0.758 / 0.713 / 620 · Stage 4/5 0.663 / 0.775 / 89 · AP-ROP 0.899 / 0.345 / 618.

## 7. Testing — external held-out hospital, locked half, opened once (n = 661, 116 infants)

Served fold-0 checkpoint, **image level** (the pre-registered primary unit):

accuracy **0.546** · balanced accuracy 0.520 · macro-F1 0.476 · **QWK (corrected) 0.493** · κ 0.355 · any-ROP AUC 0.835.

Per-class recall: Normal **0.996** · Stage 1 **0.276** · Stage 2 **0.305** · Stage 3 **0.292** · Stage 4/5 0.750 *(n=4)* · AP-ROP 0.500 *(n=12)*.

The last two cells are **pre-registered as not externally validated** at those counts and must not be quoted. Across the five fold-checkpoints, image-level QWK is 0.473 ± 0.014 — that ± is **checkpoint spread, not a confidence interval**.

## 8. Eye-level evaluation (230 eyes, 116 infants)

Under the protocol's max-over-views rule: **QWK 0.839 [0.768 – 0.890]** (patient-clustered bootstrap, 2,000 reps) · accuracy 0.678 [0.604 – 0.747] · macro-F1 (present classes) 0.616 [0.528 – 0.718] · **17.7% [11.1 – 25.5] of diseased eyes are called Normal**.

**This number is operator-dependent and that must be stated with it.** Six aggregation rules on identical predictions:

| rule | QWK (corrected) |
|---|---|
| no aggregation (image level) | 0.473 |
| one random view per eye | 0.482 |
| mean probability over views | 0.396 |
| max probability per class | 0.351 |
| majority vote | 0.398 |
| **max over views (protocol rule)** | **0.813** |
| **temporal view only, image level, no aggregation** | **0.851** |

## 9. Error analysis (1,433 errors of 6,700 predictions)

| category | n | % of errors | mean confidence | n at conf > 0.90 |
|---|---|---|---|---|
| adjacent-stage confusion | 603 | 42.1% | 0.726 | 91 |
| AP-ROP ↔ stage confusion | 338 | 23.6% | 0.662 | 45 |
| false positive (Normal → disease) | 187 | 13.0% | 0.730 | 39 |
| **⚠ disease stage → Normal** | **186** | 13.0% | **0.934** | **141** |
| **⚠ AP-ROP → Normal** | **89** | 6.2% | 0.820 | 39 |
| non-adjacent stage error | 27 | 1.9% | 0.661 | 2 |
| ⚠ Stage 4/5 → Normal | 3 | 0.2% | 0.912 | 2 |

The dominant confusion pair is Stage 1 ↔ Stage 2 (373 errors). **The most clinically dangerous error category carries the highest mean confidence of all.**

## 10. Confidence and calibration

| | current model, internal | current model, **external** | flat baseline, internal |
|---|---|---|---|
| accuracy | 0.786 | 0.546 | 0.789 |
| ECE (15 bins) | **0.1093** | **0.3339** | **0.0346** |
| Brier (multiclass) | 0.3226 | 0.7853 | 0.3304 |
| mean confidence | 0.895 | 0.880 | 0.811 |
| confidence − accuracy | +0.109 | +0.334 | +0.022 |
| conf. when right − when wrong | 0.193 | **0.041** | 0.116 |
| predictions above 0.90 | 68.6% | 65.4% | 13.1% |
| confident wrong predictions | 359 | 178 | 181 |

**Verdict: UNCALIBRATED MODEL SCORE.** The output is a usable ranking and is not a probability of disease. Externally the confidence signal is close to uninformative. No temperature scaling has been fitted.

A further internal inconsistency: `argmax(prob6)` disagrees with the decoded label in **1.19%** of predictions (2.4% on fold 0), because the label uses a 0.5 threshold on the AP-ROP branch while probabilities compose differently. AUC and F1 therefore rest on different decision rules. The flat baseline's disagreement is 0.00%.

## 11. Explainability

Fresh Grad-CAM audit of the served checkpoint on the served objective, 88 stratified external-dev images:

- attention inside the retina **56.0%**, on **synthetic letterbox padding 28.7%**, on real camera surround 15.3%
- **30.7% of CAM peaks fall outside the retina**; 19.3% land on invented padding
- retina occupies 53.8% of the frame, so attention **lift = 1.04 — indistinguishable from chance**
- on **Normal** images only 39.5% of attention is on retina and **43.9% on padding**
- confidence is uncorrelated with whether the model looked at the retina

**Grad-CAM must not be presented as evidence that the model localises pathology.** See `14_EXPLAINABILITY/explainability_limitations.md`.

## 12. External validation

One hospital, retrospective, 116 infants, opened once. What transfers: detecting normal, and coarse "any ROP" screening (AUC 0.835). What does not: per-stage discrimination (recall ≈ 0.28) and calibration (ECE 0.334). At image level the **flat softmax baseline is ahead of the current model at both seeds** (0.492 ± 0.021 / 0.486 ± 0.022 vs 0.473 ± 0.014 / 0.456 ± 0.023) — unlike cross-validation, the external sign does **not** flip.

## 13. Statistical uncertainty

Three distinct quantities are kept separate throughout and never substituted for one another:

- **fold-to-fold SD** — spread of a metric across the 5 CV folds (the `±` in §6);
- **checkpoint spread** — SD across the 5 fold-checkpoints scoring the same external set (never called a CI);
- **95% CI** — patient-clustered percentile bootstrap, 2,000 replicates, seed 42, resampling *infants* so an infant's 470 frames move together.

Paired tests, current model vs flat baseline over 5 folds: macro-F1 p = 0.589 (seed 42) / 0.494 (seed 1337); corrected QWK p = 0.977 / 0.490. Seed-to-seed spread within an arm (0.018 structured, 0.043 flat) is 3–8× the between-arm difference.

## 14. Reproducibility

Strong: committed folds and raw predictions; the training log for this exact run exists; every number here recomputed by committed scripts; the served checkpoint is now hash-stamped and its fold identified.

Broken: the committed config `weighted_sampler: true` reproduces the **documented failure arm**, not this model; `set_seed` omits cuDNN determinism, `PYTHONHASHSEED` and DataLoader generators; the s42 run summary does not record its own sampler setting; validation loss and LR were never logged; `requirements.txt` is unpinned. Full report: `20_AUDIT/reproducibility_report.md`.

## 15. Intended use and out-of-scope use

**In scope:** methods research on ROP evaluation units, labelling and site bias; a reference implementation of an ordinal + off-scale-class head.

**Out of scope:** any diagnosis, triage, referral or treatment decision; any per-photograph clinical reading; use on adults or term infants (the routing table blocks this — the model flagged 41 of 45 adult eyes as ROP-positive); any deployment at a site without local re-validation and re-thresholding.

## 16. Correction recorded in this package

`eval.qwk_classes` in the config declares QWK over the five ordinal stages, but no code ever read the key: `models/common/metrics.py` computed `cohen_kappa_score(weights="quadratic")` over all six classes, placing AP-ROP at ordinal rank 5 — the exact modelling error the architecture argues against. Correcting it moves fold-0 QWK from 0.569 to **0.908** and cuts the fold-to-fold SD 5× (0.145 → 0.029). Both values are reported side by side in `07_TEST_RESULTS/qwk_analysis/` and nothing was silently overwritten.

## 17. Deployment assessment

82.96 MB checkpoint · 20.5M parameters · CPU end-to-end **112 ms/image** (27 ms preprocessing + 85 ms forward). Practical to run locally. **Not deployment-ready** for the reasons in §7, §10 and §11.

## 18. Highest-value next experiments

1. Temperature-scale the CORN cascade on out-of-fold logits (fixes §10, no retraining).
2. Repair the reference config and stamp checkpoint provenance (fixes §14).
3. Promote the temporal-view control to the primary instrument for the label-unit result (§8).
4. Run the ablation arms on folds 1–4 and both seeds — every current ablation sits on fold 0, the atypical fold.
5. Independent paediatric-ophthalmologist adjudication of ~300 stratified labels.
