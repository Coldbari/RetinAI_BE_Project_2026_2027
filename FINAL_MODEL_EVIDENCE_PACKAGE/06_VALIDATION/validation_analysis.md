# Internal Validation Analysis — 5-fold, infant-grouped

*All values recomputed from the committed per-sample predictions by `20_AUDIT/scripts/s03_cv_metrics.py`.
Tables: [`fold_results.csv`](fold_results.csv), [`validation_metrics.csv`](validation_metrics.csv),
[`seed_comparison.csv`](seed_comparison.csv). Figure: [`cross_validation_summary.png`](cross_validation_summary.png).*

## Design

Five folds, grouped by infant, over the three training hospitals. The five validation sets are **disjoint and exactly partition 6,700 images**, so pooling them gives a genuine out-of-fold prediction for every image in the corpus. Training sets carry per-infant caps; **validation sets carry natural prevalence**, because capping the validation set would mean the metric measures the cap.

## Headline table (CURRENT MODEL, seed 42)

| fold | n | macro-F1 | accuracy | balanced acc. | **QWK (corrected)** | QWK (legacy 6-class) |
|---|---|---|---|---|---|---|
| 0 | 1,691 | 0.5527 | 0.6505 | 0.6430 | **0.9077** | 0.5685 |
| 1 | 1,252 | 0.7573 | 0.8211 | 0.7463 | **0.8467** | 0.8642 |
| 2 | 1,252 | 0.6712 | 0.8243 | 0.7171 | **0.9073** | 0.8731 |
| 3 | 1,252 | 0.7184 | 0.8195 | 0.7091 | **0.8996** | 0.9020 |
| 4 | 1,253 | 0.7616 | 0.8627 | 0.7102 | **0.9206** | 0.9197 |
| **mean ± SD** | | **0.6923 ± 0.0861** | 0.7956 ± 0.0831 | 0.7051 ± 0.0440 | **0.8964 ± 0.0288** | 0.8255 ± 0.1454 |
| **pooled** | 6,700 | 0.6517 | 0.7861 | 0.6524 | **0.8969** | 0.7610 |

`±` is **fold-to-fold standard deviation**, not a confidence interval. Patient-clustered 95% CIs on the pooled values are in [`../07_TEST_RESULTS/statistical_uncertainty/uncertainty_image_level.csv`](../07_TEST_RESULTS/statistical_uncertainty/uncertainty_image_level.csv).

## Fold 0 is the atypical fold — and the QWK correction shows why

Under the **legacy** metric, fold 0 looks catastrophic (0.5685 against 0.86–0.92 elsewhere). Under the **corrected** metric it is 0.9077 — the second-best fold. The apparent collapse was AP-ROP being scored as ordinal rank 5, in a fold whose validation set holds 476 AP-ROP frames (470 from one infant) against 30–41 elsewhere.

What remains genuinely anomalous about fold 0 after the correction:

- macro-F1 0.553 vs 0.67–0.76 — **real**, driven by AP-ROP recall 0.212 on n=476
- calibration error 0.201 vs 0.058–0.093 — **real**
- decode/argmax disagreement 2.4% vs 0.6–1.0% — **real**

So the fold-0 story survives; the *QWK leg of it does not*. See [`../07_TEST_RESULTS/qwk_analysis/`](../07_TEST_RESULTS/qwk_analysis/).

## Seed sensitivity

| | fold 0 | fold 1 | fold 2 | fold 3 | fold 4 | mean \|Δ\| |
|---|---|---|---|---|---|---|
| structured, \|seed 42 − seed 1337\| | 0.006 | 0.011 | 0.020 | 0.037 | 0.017 | **0.018** |
| flat baseline, same | 0.032 | 0.002 | 0.096 | 0.043 | 0.041 | **0.043** |

Changing only the random seed moves macro-F1 by 0.018–0.043 on average — **3 to 8 times the 0.0055 difference between the two architectures.**

## Comparison with the flat-softmax baseline

| arm | macro-F1 (mean ± fold SD) | QWK corrected | accuracy |
|---|---|---|---|
| **CURRENT structured, seed 42** | **0.6923 ± 0.0861** | **0.8964 ± 0.0288** | 0.7956 |
| structured, seed 1337 | 0.7079 ± 0.0965 | 0.9045 ± 0.0326 | 0.8122 |
| flat softmax, seed 42 | 0.6977 ± 0.0986 | 0.8962 ± 0.0285 | 0.7973 |
| flat softmax, seed 1337 | 0.6855 ± 0.1084 | 0.8967 ± 0.0338 | 0.7937 |

Paired over the 5 folds ([`../16_STATISTICAL_ANALYSIS/paired_tests.csv`](../16_STATISTICAL_ANALYSIS/paired_tests.csv)):

| metric | seed | mean paired difference | paired *t* | Wilcoxon |
|---|---|---|---|---|
| macro-F1 | 42 | **−0.0055** (flat ahead) | p = 0.589 | p = 0.813 |
| macro-F1 | 1337 | **+0.0224** (structured ahead) | p = 0.494 | p = 1.000 |
| QWK (corrected) | 42 | **+0.0002** | p = 0.977 | p = 1.000 |
| QWK (corrected) | 1337 | +0.0077 | p = 0.490 | p = 1.000 |

**The sign of the macro-F1 difference reverses between seeds, no test approaches significance, and under the corrected QWK the two heads differ by 0.0002.** This is an equivalence result, and the corrected metric makes it stronger than the uncorrected one did.

*Caveat: five folds is a small n for a paired test, and the folds are not independent (they share training data). These p-values should be read as "nothing to see", not as evidence of equivalence at a specified power.*

## Per-class, pooled out-of-fold (n = 6,700)

| class | precision | recall | F1 | support | recall 95% CI (patient-clustered) |
|---|---|---|---|---|---|
| Normal | 0.929 | 0.951 | 0.940 | 3,837 | — |
| **Stage 1** | **0.376** | 0.499 | 0.429 | 577 | see `per_class_metrics.csv` |
| Stage 2 | 0.559 | 0.631 | 0.593 | 959 | |
| Stage 3 | 0.758 | 0.713 | 0.735 | 620 | |
| Stage 4/5 | 0.663 | 0.775 | 0.715 | 89 | wide — 89 images from one source |
| AP-ROP | 0.899 | **0.345** | 0.499 | 618 | wide — 476 from one infant |

Full table with intervals: [`../07_TEST_RESULTS/per_class_metrics/per_class_metrics.csv`](../07_TEST_RESULTS/per_class_metrics/per_class_metrics.csv).

**Stage 1 precision 0.376 is the weakest number in this package** and is not discussed anywhere in the parent project's documentation. The model over-predicts Stage 1: 169 Normal and 193 Stage 2 images are labelled Stage 1.
