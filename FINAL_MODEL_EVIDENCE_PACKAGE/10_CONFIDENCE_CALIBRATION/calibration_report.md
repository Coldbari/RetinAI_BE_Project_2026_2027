# Confidence and Calibration Report

*Recomputed by `20_AUDIT/scripts/s05_calibration.py`. ECE uses 15 equal-width bins.
Tables: [`calibration_metrics.csv`](calibration_metrics.csv), [`reliability_bins.csv`](reliability_bins.csv),
[`high_confidence_error_analysis.csv`](high_confidence_error_analysis.csv)*

---

# ⚠️ UNCALIBRATED MODEL SCORE

**The six percentages this model outputs are a ranking score. They are NOT probabilities of disease and must not be read, displayed or documented as such.** No temperature scaling or other calibration has been fitted for this model.

---

## The numbers

| | CURRENT model, internal OOF | **CURRENT model, external locked** | flat baseline, internal OOF |
|---|---|---|---|
| n | 6,700 | 661 | 6,700 |
| accuracy | 0.7861 | 0.5461 | 0.7888 |
| **ECE (15 bins)** | **0.1093** | **0.3339** | **0.0346** |
| Brier score (multiclass) | 0.3226 | 0.7853 | 0.3304 |
| mean stated confidence | 0.8954 | 0.8799 | 0.8113 |
| **over-confidence (confidence − accuracy)** | **+0.1093** | **+0.3337** | +0.0225 |
| mean confidence when **right** | 0.9367 | 0.8983 | 0.8357 |
| mean confidence when **wrong** | 0.7436 | 0.8577 | 0.7200 |
| **separation (right − wrong)** | 0.1931 | **0.0406** | 0.1157 |
| fraction of predictions above 0.90 | 68.6% | 65.4% | 13.1% |
| confident (>0.90) **wrong** predictions | **359** | 178 | 181 |
| accuracy among confident predictions | 0.922 | 0.588 | 0.794 |

Per fold: ECE 0.201 (fold 0) · 0.093 · 0.082 · 0.091 · 0.058. Fold 0 is 2–3× the others — the third independent signature of that fold's label problem (the other two being macro-F1 and AP-ROP recall; the QWK signature turned out to be a metric artifact, see [`../07_TEST_RESULTS/qwk_analysis/`](../07_TEST_RESULTS/qwk_analysis/)).

## Reading the reliability diagram

[`reliability_diagram.png`](reliability_diagram.png)

Every bin below 0.93 confidence is **badly** over-confident: in the 0.87–0.93 bin the model claims 0.904 and is right 0.691 of the time. The mass then piles into the top bin — 4,255 of 6,700 predictions at mean confidence 0.99 and accuracy 0.94. The overall ECE is modest only because that top bin is nearly calibrated and holds most of the weight.

**At the external hospital the picture collapses.** ECE triples to 0.334, and the separation between confidence-when-right and confidence-when-wrong falls to **0.041** — the model is essentially *equally confident* whether it is right or wrong. A confidence-based triage rule would be worthless there.

## Why this model is over-confident — mechanism

The probability vector is `[(1 − pₐ) · P(rank = k) …, pₐ]`, where `P(rank = k)` comes from differencing a **cumulative product of sigmoids**. Multiplying 1–4 near-saturated sigmoids drives mass into a single rank. A plain softmax over six logits does not have this geometry, which is why the flat baseline — the same backbone, the same folds, the same accuracy — is 3.2× better calibrated.

**This is a measured disadvantage of the contributed architecture that no previous artifact in the project compared.**

## Are the confident errors filterable?

No — and this is the key safety finding.

| | count | mean confidence |
|---|---|---|
| all errors | 1,433 | 0.735 |
| **"disease stage called Normal"** | **186** | **0.934** |
| "AP-ROP called Normal" | 89 | 0.820 |
| adjacent-stage confusion | 603 | 0.726 |

The single most dangerous error category is the *most* confident one. A rule of the form "only auto-report predictions above 0.90 confidence" would **retain 141 of the 186 missed-disease cases** while discarding a large share of the harmless adjacent-stage errors. See [`high_confidence_error_analysis.csv`](high_confidence_error_analysis.csv) and the gallery in [`../09_ERROR_ANALYSIS/high_confidence_errors/`](../09_ERROR_ANALYSIS/high_confidence_errors/).

## A second, smaller inconsistency

`argmax(prob6)` disagrees with the decoded label `pred6` in **1.19%** of internal predictions (2.4% on fold 0, 1.2% externally), because the label is decided by a hard 0.5 threshold on the AP-ROP branch while the probability vector composes multiplicatively. Accuracy and F1 therefore describe one decision rule and AUC describes another. The flat baseline's disagreement is exactly 0.00%.

## Recommendation

1. **Immediately:** label the displayed percentages in the product as an uncalibrated ranking score. (The web app already does this correctly for the binary screening score — `"calibration": {"status": "unverified"}` — but the staging card shows six percentages from the worse-calibrated head without that qualifier.)
2. **Cheap fix, no retraining:** fit temperature scaling on the CORN conditional logits using out-of-fold predictions only. Acceptance criterion: pooled ECE below the flat baseline's 0.035 with macro-F1 moving by less than 0.01. Protocol in [`../18_RESEARCH_REVIEW/recommended_future_experiments.md`](../18_RESEARCH_REVIEW/recommended_future_experiments.md) (experiment 1).
3. **Report calibration at the external site whenever accuracy is reported there.** A model that is 88% confident and 55% accurate is a different object from one that is 55% confident and 55% accurate.
