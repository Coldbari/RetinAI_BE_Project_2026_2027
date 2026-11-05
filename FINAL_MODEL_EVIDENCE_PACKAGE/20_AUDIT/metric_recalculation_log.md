# Metric Recalculation Log — every correction, with an audit trail

**Rule observed: no silent corrections.** Each entry gives the previous value, the corrected
value, the reason, the source, and the impact. Legacy values remain published in
[`../19_COMPARISONS/retired_models/legacy_qwk_values.csv`](../19_COMPARISONS/retired_models/legacy_qwk_values.csv).

---

## Correction 1 — QWK computed over the wrong class set

**Previous result.** CV mean QWK **0.8255 ± 0.1454**; fold 0 **0.5685**; pooled 0.7610; external locked (5-checkpoint mean) **0.5232 ± 0.0179**; eye-level max 0.8012.

**Corrected result.** CV mean QWK **0.8964 ± 0.0288**; fold 0 **0.9077**; pooled 0.8969; external locked **0.4732 ± 0.0140**; eye-level max 0.8127.

**Reason.** `configs/rop_staging*.yaml` declare `eval.qwk_classes: ["Normal","Stage 1","Stage 2","Stage 3","Stage 4/5"]` with the comment *"QWK is computed over the 5 ORDINAL classes only. AP-ROP is not a point on the stage scale."* A repository-wide grep finds **zero** Python references to the key. `models/common/metrics.py::compute_metrics` ran `cohen_kappa_score(y_true, y_pred, weights="quadratic")` over all six integer labels, placing AP-ROP (index 5) one step beyond Stage 4/5 on the severity ladder — charging an AP-ROP→Normal error 25 units against 16 for Stage 4/5→Normal.

**Source.** Raw per-sample predictions: `results/rop/staging_runs/structured_5fold/fold_{0..4}.npz`, `results/rop/locked_eval/structured_s42_ckpt{0..4}.npz`. Recomputation: `scripts/common.py::qwk_ordinal`; legacy reimplemented as `qwk_6class_legacy` and verified to reproduce every published value to 4 dp.

**Impact.**
- Fold-to-fold SD falls **5×** (0.145 → 0.029); QWK becomes the most stable metric on this corpus (CI width 0.054).
- **The claim "fold 0 shows a QWK signature of the label problem" is WITHDRAWN.** Under the declared metric, fold 0 is the *second-best* fold. Fold 0's macro-F1 anomaly (0.553 vs 0.67–0.76) and calibration anomaly (ECE 0.201 vs 0.058–0.093) survive.
- The equivalence claim **strengthens**: structured 0.8964 vs flat 0.8962, p = 0.977.
- The external number moves **down**, 0.523 → 0.473. This is not a flattering correction.
- The image→eye gap **widens** slightly, 0.473 → 0.813.

**Scope applied.** All 5 CV folds × 4 arms; external dev and locked × 4 arms × 5 checkpoints; all 6 eye-level aggregation rules; all 3 anatomical regions. **Not applied** to the 7-arm fold-0 ablation table or the 5-backbone benchmark — raw `.npz` files do not exist for every arm. Those remain on the legacy metric in `19_COMPARISONS/` with an explicit warning.

---

## Correction 2 — the served checkpoint's identity was unknown

**Previous result.** `results/rop_staging/weights.pth` had no recorded fold, seed, run tag or hash anywhere in the repository. A prior audit flagged this as an unresolvable reproducibility gap.

**Corrected result.** The served checkpoint **is the fold-0 checkpoint of the structured seed-42 natural-sampling run.** SHA-256 `11b439608690024700cdc927dece45d51ebc78b648aef81090cb6be3e7dc747c`.

**Reason / method.** The served weights were loaded and run over all 663 external dev images through the served preprocessing path, then compared against the five committed per-checkpoint dev predictions:

| candidate | prediction agreement | max \|Δ prob\| |
|---|---|---|
| **fold 0, seed 42** | **1.000 (663/663)** | **3.0 × 10⁻⁶** |
| fold 1, seed 42 | 0.885 | 0.94 |
| fold 2, seed 42 | 0.888 | 0.85 |
| fold 3, seed 42 | 0.872 | 0.91 |
| fold 4, seed 42 | 0.879 | 0.98 |
| all 5, seed 1337 | 0.875 – 0.893 | 0.87 – 0.98 |

**Source.** `scripts/s01_identity.py`; output `served_checkpoint_identification.json`; predictions `served_checkpoint_dev_rerun.npz`.

**Impact.** The served model was trained on **4/5** of the internal corpus, so internal operating points derived from *pooled out-of-fold* scores (sensitivity 0.958 / specificity 0.781, as displayed in the app) **do not describe it**. Also confirms train/serve preprocessing parity to 3e-06.

---

## Correction 3 — training history for the current model was recorded as missing

**Previous result.** An earlier audit of this project concluded that no per-epoch training history existed for the structured 5-fold runs, and listed validation curves among the missing evidence.

**Corrected result.** **The history exists.** `kaggle_cache/struct42/rop-structured-effnetv2s-s42.log` contains **120 epoch records** across 6 runs (5 CV folds + leave-site), with total loss and the three component losses, the realised DANN λ, validation macro-F1 and validation QWK per epoch.

**Reason.** The earlier audit searched `results/rop/staging_runs/`, which holds only the five *flat* five-backbone benchmark logs. The structured logs live under `kaggle_cache/`.

**Source.** Extracted by `scripts/s07_training.py` into [`../05_TRAINING/training_history.csv`](../05_TRAINING/training_history.csv).

**Impact.** [`../05_TRAINING/`](../05_TRAINING/) is now evidence-based rather than a gap. **Cross-verification:** the final validation macro-F1 in the log matches the value recomputed from the committed `.npz` files exactly for all five folds (0.5527 / 0.7573 / 0.6712 / 0.7184 / 0.7616) — two artifacts produced months apart, agreeing.

**What remains genuinely missing:** validation **loss** and the realised **learning rate** were never logged. The LR figure in this package is a labelled reconstruction.

---

## Correction 4 — eye-level performance reported without its operator

**Previous result.** "Eye-level QWK 0.79–0.80 across all four arm-seed cells", quoted as a headline without naming the aggregation rule or reporting alternatives.

**Corrected result.** The figure is reproduced (0.813 corrected, current model) **and** placed in context:

| rule | corrected QWK |
|---|---|
| max over views (the protocol rule) | 0.813 |
| one random view | 0.482 |
| no aggregation (image level) | 0.473 |
| majority vote | 0.398 |
| mean probability | 0.396 |
| max probability per class | 0.351 |
| **temporal frames only, image level, no aggregation** | **0.851** |

**Reason.** `max` is the most optimistic ordinal aggregator, and it was adopted after inspecting dev-half results (disclosed in the protocol's Amendment 1). A headline that depends on an operator must name it.

**Source.** `scripts/s04_external_eye_region.py`; [`../11_AGGREGATION_AND_VIEW_ANALYSIS/aggregation_results.csv`](../11_AGGREGATION_AND_VIEW_ANALYSIS/aggregation_results.csv).

**Impact.** The scientific claim **survives and is strengthened** — the temporal-only control reproduces the eye-level value with no aggregation at all — but the reporting changes: the region analysis becomes the primary instrument and the eye-level figure a secondary with its rule named.

---

## Correction 5 — calibration was never compared against the baseline

**Previous result.** `results/rop/structured_bootstrap_calibration.json` reported pooled ECE 0.107 for the structured model. No comparison against the flat baseline existed in any artifact, and no external calibration figure existed at all.

**Corrected result.**

| | structured (current) | flat baseline | structured, external |
|---|---|---|---|
| ECE | 0.1093 | **0.0346** | **0.3339** |
| accuracy | 0.7861 | 0.7888 | 0.5461 |
| predictions > 0.90 confidence | 68.6% | 13.1% | 65.4% |
| confident wrong predictions | 359 | 181 | 178 |

*(this package's 0.1093 corroborates the project's own 0.107 — the difference is binning.)*

**Reason.** A calibration number without a comparator cannot be judged. At identical accuracy the baseline is **3.2× better calibrated**.

**Source.** `scripts/s05_calibration.py`; [`../10_CONFIDENCE_CALIBRATION/calibration_metrics.csv`](../10_CONFIDENCE_CALIBRATION/calibration_metrics.csv).

**Impact.** A new, measured **disadvantage** of the contributed architecture, and the discovery that the confidence signal nearly vanishes externally (right-vs-wrong separation 0.041).

---

## Values re-derived and CONFIRMED unchanged

| quantity | previously published | recomputed here | verdict |
|---|---|---|---|
| CV macro-F1, structured s42 | 0.692 ± 0.086 | **0.6923 ± 0.0861** | ✅ exact |
| CV macro-F1, structured s1337 | 0.708 ± 0.097 | **0.7079 ± 0.0965** | ✅ exact |
| CV macro-F1, flat s42 | 0.698 ± 0.099 | **0.6977 ± 0.0986** | ✅ exact |
| CV macro-F1, flat s1337 | 0.686 ± 0.109 | **0.6855 ± 0.1084** | ✅ exact |
| paired difference, seed 42 / 1337 | −0.0055 / +0.0224 | **−0.0055 / +0.0224** | ✅ exact, sign flip confirmed |
| patient-level leakage | 0 overlaps | **0 overlaps** | ✅ independently confirmed |
| train pool size | 3,112 images / 1,528 infants | **3,112 / 1,528** | ✅ |
| Grad-CAM retina attention | 55.5% | **56.0%** (fresh run, n=88) | ✅ corroborated |
| Grad-CAM attention on padding | 27.8% | **28.7%** | ✅ corroborated |
| structured pooled ECE | 0.107 | **0.1093** | ✅ (binning difference) |
