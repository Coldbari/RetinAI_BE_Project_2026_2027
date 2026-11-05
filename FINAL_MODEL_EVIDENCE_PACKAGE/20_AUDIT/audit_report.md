# Complete Technical Audit Report — CURRENT MODEL

*Commit `f0d4508` · built 2026-09-21 · every number recomputed from raw per-sample prediction
artifacts by `20_AUDIT/scripts/s01`–`s12`. Verification: [`verification_report.md`](verification_report.md).
Corrections ledger: [`metric_recalculation_log.md`](metric_recalculation_log.md).*

---

## 1. Model description

Structured EfficientNetV2-S for 6-class ICROP staging of retinopathy of prematurity. A shared
ImageNet-pretrained backbone (20,177,488 params) feeds a 1280-d embedding into three heads:
a **CORN ordinal stage head** (4 conditional logits, 5,124 params) with AP-ROP samples masked
out of its loss; an **AP-ROP binary branch** (1,281 params) decoded at a threshold fixed a
priori at 0.5; and a **DANN site adversary** (328,707 params) that is training-time machinery
only. Total 20,512,600 parameters, all trainable.

Served artifact: `results/rop_staging/weights.pth`, SHA-256 `11b4396086…`, identified in this
audit as the **fold-0 checkpoint of the seed-42 run** (100% prediction agreement over 663
images, max probability difference 3 × 10⁻⁶).

**The architectural contribution is 6,405 parameters — 0.031% of the model.**

## 2. Dataset

3,112 training images / 1,528 infants / 3 hospitals, imbalance 12.46 : 1 (normal 1,109 …
stage_4_5 89). Internal evaluation 6,700 images across 5 infant-grouped folds that **exactly
partition** the corpus. External: 663 dev + 661 locked images, 116 infants each, from a fourth
hospital never used in training.

Class × source confounding is real: stage_4_5 is **100% ROP-VL**, AP-ROP has **zero** Shenzhen
images, Stage 3 is only 7% Ostrava, and hospital identity is decodable from file metadata at
0.997 balanced accuracy. Only stage_4_5 is declared unvalidated upstream; on this evidence
AP-ROP and Stage 3 warrant the same declaration.

Fold 0 is atypical by construction: its validation set holds 476 AP-ROP frames (**470 from a
single infant**) against 30–41 in every other fold.

## 3. Preprocessing

`circle_crop → letterbox_square → CLAHE → Resize(384) → ImageNet normalise`. The letterbox is
load-bearing anti-shortcut engineering (image dimensions alone previously scored AUC 0.911).
Training adds flips, rotation, colour and hue/saturation jitter, RandomGrayscale, resolution
jitter and RandomResizedCrop.

**Train/serve parity verified empirically**, not asserted: the served weights re-run through the
web app's own transform path reproduced the committed evaluation predictions to 3 × 10⁻⁶.

## 4. Architecture assessment

The design is principled — ICROP-3 does treat AP-ROP as a form rather than a rank, and stages
are ordered. Its measured pay-off over a plain softmax on the same backbone, folds and metrics
is **nil on every aggregate metric** and **negative on calibration**. The single clear win is
representational: an independent AP-ROP probability and monotone ordinal ranks.

**Deviation found:** `w_site` multiplies the entire site loss, damping the adversary's own
learning, rather than scaling only the reversed gradient into the backbone as canonical DANN
does. Consistent across arms, so ablations remain internally valid.

## 5. Training

AdamW, lr 3e-4, OneCycleLR (pct_start 0.1, cosine), weight decay 0.05, batch 16, 20 epochs, AMP,
grad-clip 1.0, **natural-prevalence sampling**, last-epoch weights.

Training loss falls smoothly 1.83 → ≈0.45 in all 6 runs with no divergence; the site-adversary
loss stays pinned near ln 3, which is the intended DANN behaviour. **Validation macro-F1 is
highly unstable**: mean absolute change between consecutive epochs 0.040–0.065, maximum 0.31;
SD over the last five epochs 0.007–0.056. **That end-of-schedule noise exceeds the entire
structured-vs-flat difference of 0.0055.**

**No validation loss and no realised learning rate were ever logged.** A classical overfitting
diagnosis is therefore *not available in the provided evidence*; this is stated rather than
substituted for. The LR figure in this package is a labelled reconstruction.

> ⚠️ The committed config sets `weighted_sampler: true`, which is the documented v1 collapse
> (QWK 0.121). Every headline run overrode it to `false` from the Kaggle kernel.

## 6. Validation — internal, 5-fold, infant-grouped (n = 6,700)

macro-F1 **0.6923 ± 0.0861** (fold SD); pooled 0.6517 [0.572, 0.760]. Accuracy 0.7861
[0.697, 0.857]. Balanced accuracy 0.6524 [0.616, 0.757]. **Corrected QWK 0.8964 ± 0.0288**;
pooled 0.8969 **[0.869, 0.923]** — the most stable metric on this corpus. Unweighted κ 0.6549.
Any-ROP AUC 0.9695 / AP 0.9662.

Per class (precision / recall / n): Normal 0.929 / 0.951 / 3,837 · **Stage 1 0.376 / 0.499 / 577**
· Stage 2 0.559 / 0.631 / 959 · Stage 3 0.758 / 0.713 / 620 · Stage 4/5 0.663 / 0.775 / 89 ·
AP-ROP 0.899 / 0.345 / 618.

## 7. Testing — external held-out hospital, locked half, opened once (n = 661)

Image level (pre-registered primary): accuracy **0.546**, balanced accuracy 0.520, macro-F1
0.476, **corrected QWK 0.493**, κ 0.355, any-ROP AUC 0.835.

Per-class recall: Normal **0.996** · Stage 1 **0.276** · Stage 2 **0.305** · Stage 3 **0.292** ·
Stage 4/5 0.750 *(n=4, not validated)* · AP-ROP 0.500 *(n=12, not validated)*.

**At a new hospital the model becomes a near-perfect normal detector and a poor stager.** Normal
*precision* there is only 0.530 — it says Normal far too often.

## 8. Eye-level evaluation (230 eyes, 116 infants)

Protocol max-over-views rule: QWK **0.8394 [0.7683, 0.8903]**, accuracy 0.6783 [0.604, 0.747],
macro-F1 (present) 0.6160 [0.528, 0.718], **17.7% [11.1, 25.5] of diseased eyes called Normal**.
Patient-clustered intervals.

**The figure is operator-dependent and this audit measured that:** max 0.813 · one random view
0.482 · image level 0.473 · majority 0.398 · mean-probability 0.396 · max-probability-per-class
0.351. Only `max` produces the gap.

**The decisive control:** temporal-periphery frames scored at image level **with no aggregation
at all** reach **0.841** (nasal 0.352, posterior pole 0.229; same pattern in the baseline). The
ROP ridge grows temporally; frames that cannot contain it inherit the eye's label. The
label-unit explanation is therefore supported by evidence that no aggregation argument can
touch, and the region analysis is the stronger instrument.

## 9. Error analysis (1,433 errors / 6,700)

Adjacent-stage 603 (42.1%) · AP-ROP ↔ stage 338 · false-positive Normal→disease 187 ·
**disease→Normal 186** · **AP-ROP→Normal 89** · non-adjacent 27 · Stage 4/5→Normal 3.

Only 1.9% of errors are non-adjacent: **the ordinal head delivers error structure**, which is a
real benefit no aggregate metric captures.

> **The most clinically dangerous error category carries the HIGHEST mean confidence of all
> (0.934), and 141 of its 186 cases were made above 0.90 confidence.** No confidence threshold
> can filter these out.

Dominant pair: Stage 1 ↔ Stage 2, 373 errors. **Stage 1 precision 0.376 [0.273, 0.487]** is the
weakest cell in the package and is undiscussed upstream.

## 10. Confidence · 11. Calibration

| | internal | **external** | flat baseline |
|---|---|---|---|
| accuracy | 0.786 | 0.546 | 0.789 |
| ECE | 0.1093 | **0.3339** | **0.0346** |
| Brier | 0.3226 | 0.7853 | 0.3304 |
| conf. right − wrong | 0.193 | **0.041** | 0.116 |
| predictions > 0.90 | 68.6% | 65.4% | 13.1% |
| confident errors | 359 | 178 | 181 |

**Verdict: UNCALIBRATED MODEL SCORE.** Externally the confidence signal is nearly uninformative.
The mechanism is the CORN cascade's cumulative product of sigmoids. **The flat baseline is 3.2×
better calibrated at identical accuracy** — a measured disadvantage of the contributed head that
no prior artifact compared.

Secondary defect: `argmax(prob6)` disagrees with the decoded label in **1.19%** of predictions,
so F1/accuracy and AUC rest on different decision rules (baseline: 0.00%).

## 12. Explainability

Fresh Grad-CAM audit of the served checkpoint on the served objective, 88 stratified images:
attention on retina **56.0%** against a retina area share of **53.8%** → **lift 1.04, i.e.
chance**. 30.7% of peaks fall outside the retina; **19.3% land on synthetic letterbox padding**.
On Normal images, 39.5% retina vs **43.9% padding**. Attention quality is uncorrelated with
confidence. Independently corroborated by the project's own `cam_leakage.json` (55.5% / 27.8%).

**Grad-CAM must not be presented as evidence that the model localises pathology.**

## 13. External validation

One hospital, retrospective, opened once under a pre-registered protocol. **Transfers:**
detecting normal, coarse any-ROP screening (AUC 0.835). **Does not transfer:** per-stage
discrimination (≈0.28 recall) and calibration (ECE 0.334). At the pre-registered image-level
unit the **flat baseline is ahead at both seeds** (0.492 / 0.486 vs 0.473 / 0.456) — unlike
cross-validation, the external sign does **not** flip.

## 14. Statistical uncertainty

Four quantities kept rigorously distinct: fold-to-fold SD, checkpoint spread, patient-clustered
bootstrap 95% CI (2,000 reps, resampling **infants**), and seed variation. Checkpoint spread is
**never** presented as a confidence interval.

Paired over 5 folds: macro-F1 p = 0.589 / 0.494; corrected QWK p = 0.977 / 0.490. Three noise
floors each exceed the effect being tested: seed change 0.018–0.043, last-5-epoch 0.007–0.056,
fold spread 0.086–0.099.

**AP-ROP recall CI is [0.254, 0.836]** — width 0.58. That interval *is* the label-unit finding
expressed as statistics, and the point estimate 0.345 should never be quoted without it.

## 15. Reproducibility

**Evidence: fully reproducible.** Raw predictions, split files, metric implementation and build
scripts are all committed; four previously published values reproduced exactly and two were
confirmed by a second independent route.

**Model: not yet.** Four breaks — the reference config reproduces the retired failure arm; the
seed-42 summary omits its own sampler setting; `set_seed` lacks cuDNN determinism,
`PYTHONHASHSEED` and DataLoader generators; `requirements.txt` is unpinned. All four are about
an hour's work. Full report: [`reproducibility_report.md`](reproducibility_report.md).

## 16. Limitations

39 enumerated in [`../18_RESEARCH_REVIEW/limitations.md`](../18_RESEARCH_REVIEW/limitations.md).
The binding ones: staging does not transfer; probabilities are uncalibrated and worst where it
matters; the most dangerous error is the most confident; Grad-CAM is unreliable; two classes are
externally unvalidated at n = 4 and n = 12; one external site; no independent grader.

## 17. Deployment assessment

82.96 MB, 20.5M params, **112 ms/image CPU end-to-end**. Compute is not the constraint.
**Not deployment-ready** — see [`../15_DEPLOYMENT/deployment_assessment.md`](../15_DEPLOYMENT/deployment_assessment.md) for the eight blockers, and for the five things the product already does right (context routing, an honest calibration badge, a committed threshold artifact, retiring a served model on its own negative audit, graceful degradation).

## 18. Future experiments

E1 temperature-scale the CORN cascade · E2 repair the config and stamp the checkpoint ·
E3 promote the anatomical control · E4 re-derive the served operating point · E5 ablations off
fold 0 · E6 instrument the training loop · E7 a minimal hyper-parameter sweep · E8 clinician
adjudication · E9 a second external hospital. E1–E4 are CPU-only and could be done in a day.
Specifications: [`../18_RESEARCH_REVIEW/recommended_future_experiments.md`](../18_RESEARCH_REVIEW/recommended_future_experiments.md).

---

## Overall verdict

**Model learning — proven.** Smooth convergence, far above the 0.187 metadata floor.

**Generalization — partially demonstrated, honestly negative where it fails.** Internal
generalisation is solid and leakage-free. Cross-hospital generalisation holds for detecting
normal and fails for staging, and a large part of the per-photograph failure is explained —
with evidence — by the label's native unit and the anatomy of the lesion.

**Reliability of the metrics — high, after one correction.** Every headline reproduced exactly
from raw predictions. QWK was mis-specified against its own config and is now corrected in both
directions, with the legacy values preserved.

**Dataset quality — careful engineering, irreducible limits.** Leakage-free and well-documented;
limited by 89 stage 4/5 images from one source, one infant dominating AP-ROP evaluation, and
labels whose native unit is not the image.

**Evaluation quality — the strongest part of the project, with one blind spot now closed.** A
pre-registered locked protocol, patient-clustered bootstraps, a metadata floor, and negative
results reported as results. The blind spot was that the two instruments carrying the headline
— QWK and eye-level max — had never themselves been audited. Both now have been.

**Deployment readiness — not ready, and correctly self-described as such.**

**The single most useful thing in this package** is not a model number. It is the temporal-view
control: evidence that per-image ICROP metrics understate agreement by more than 0.35 QWK for a
reason that is anatomical and measurable, and that this holds across two architectures and two
seeds. That finding is more solid than the model it came from, and it is what the project should
lead with.
