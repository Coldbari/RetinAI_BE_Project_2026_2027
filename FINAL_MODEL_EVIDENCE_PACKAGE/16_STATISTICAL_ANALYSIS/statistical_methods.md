# Statistical Methods

*Every interval, spread and test in this package is produced by `20_AUDIT/scripts/common.py`.
The four quantities below are kept rigorously distinct and are **never** substituted for one another.*

## 1. Fold-to-fold standard deviation

Written `mean ± SD` in cross-validation tables. It describes how much a metric moves **between the five cross-validation folds**, which on this corpus is driven largely by how the rare classes happen to distribute (fold 0's validation set holds 476 AP-ROP frames against 30–41 elsewhere).

**It is not a confidence interval and is not divided by √n anywhere.** Where a standard error is genuinely wanted, `fold_sem = fold_sd/√5` is given explicitly in [`confidence_intervals.csv`](confidence_intervals.csv).

## 2. Checkpoint spread

Written `mean ± SD over 5 checkpoints` in external tables. Five fold-checkpoints score the **same fixed 661-image test set**; the spread describes model-to-model variation, not sampling uncertainty about a population.

> **Rule observed throughout this package: checkpoint variation is never presented as a confidence interval.** Figures using it say so in the axis label or the title.

## 3. Patient-clustered bootstrap 95% confidence interval

The only quantity in this package called a confidence interval.

```
resample INFANTS with replacement (not images, not eyes)
  → all rows belonging to a resampled infant enter together
  → recompute the metric
  → 2,000 replicates, numpy default_rng(seed=42)
  → percentile interval [2.5th, 97.5th]
```

Clustering is mandatory here: one infant contributes up to **470** frames, and both eyes of a baby are correlated. An image-level bootstrap on this corpus would produce intervals several times too narrow.

Reported for: pooled out-of-fold accuracy, balanced accuracy, macro-F1 and corrected QWK ([`confidence_intervals.csv`](confidence_intervals.csv)); per-class precision and recall ([`../07_TEST_RESULTS/per_class_metrics/per_class_metrics.csv`](../07_TEST_RESULTS/per_class_metrics/per_class_metrics.csv)); and all eye-level metrics ([`../08_EYE_LEVEL_ANALYSIS/eye_confidence_intervals.csv`](../08_EYE_LEVEL_ANALYSIS/eye_confidence_intervals.csv)).

## 4. Seed variation

The difference in a metric when **only the random seed changes** (42 → 1337), everything else identical. Reported in [`../06_VALIDATION/seed_comparison.csv`](../06_VALIDATION/seed_comparison.csv).

This is the single most useful uncertainty quantity for the architecture comparison, because it bounds what any single-seed comparison can legitimately claim: mean |Δ| is **0.018** (structured) and **0.043** (flat), against an architecture difference of **0.0055**.

## Paired significance tests

[`paired_tests.csv`](paired_tests.csv). Structured vs flat, paired over the five folds, per seed, for macro-F1, corrected QWK and accuracy. Both a paired *t*-test and a Wilcoxon signed-rank test are reported.

**Stated limitation:** n = 5 folds, and folds are **not independent** — they share training data by construction. These p-values are evidence that *nothing is detectable*, not evidence of equivalence at a specified power. They are reported alongside the seed-flip, which is the stronger argument.

## Metric definitions used

| metric | definition |
|---|---|
| macro-F1 (6 classes) | unweighted mean of per-class F1 over all six labels, `zero_division=0` |
| macro-F1 (present classes) | same, restricted to classes with non-zero support — reported **alongside** the 6-label value, never instead of it, because the external set has structural zeros |
| balanced accuracy | unweighted mean of per-class recall over present classes |
| **QWK** | quadratic-weighted Cohen's κ over the **five ordinal stage classes only**; pairs where either truth or prediction is AP-ROP are excluded. See [`../07_TEST_RESULTS/qwk_analysis/`](../07_TEST_RESULTS/qwk_analysis/) |
| unweighted κ | Cohen's κ over all six labels, no weighting — reported as a floor |
| "any ROP" AUC / AP | binary ROC-AUC and average precision of `Σ prob6[1:]` against `label ≥ 1` |
| ECE | 15 equal-width bins on max-class probability, weighted mean absolute gap between bin confidence and bin accuracy |
| Brier | multiclass, `mean Σ_k (p_k − y_k)²` |

## Deliberate omissions

| not computed | why |
|---|---|
| DeLong tests on multiclass AUC | the decode rule and the probability argmax disagree in 1.19% of cases, so AUC does not describe the reported decision rule |
| per-class ROC/PR at the external site for stage 4/5 and AP-ROP | n = 4 and n = 12 |
| McNemar on image-level pairs | already shown upstream to find "significance" from seed noise alone at the image level; only patient-clustered comparisons are trustworthy on this corpus |
| a power analysis | would require an assumed effect size; the seed-flip makes the point without one |
