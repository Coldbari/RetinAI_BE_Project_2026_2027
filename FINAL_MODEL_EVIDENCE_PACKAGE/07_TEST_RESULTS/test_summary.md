# Test Results Summary — CURRENT MODEL

Two evaluations are reported and must not be conflated.

| | **internal** | **external** |
|---|---|---|
| set | 5-fold out-of-fold, 3 training hospitals | locked half of a **held-out hospital**, opened once |
| n | 6,700 images / 1,528 infants | 661 images / 230 eyes / 116 infants |
| relationship to training | no infant overlap, but the *hospitals* were trained on | no infant and **no hospital** overlap |
| model | the 5 fold checkpoints, each scoring its own held-out fold | the **served** fold-0 checkpoint (and 4 siblings for spread) |

## Internal, image level (the model at its best)

| metric | value | 95% CI (patient-clustered bootstrap, 2,000 reps) |
|---|---|---|
| accuracy | 0.7861 | [0.697, 0.857] |
| balanced accuracy | 0.6524 | [0.616, 0.757] |
| macro-F1 (6 classes) | 0.6517 | [0.572, 0.760] |
| **QWK (corrected, 5 ordinal)** | **0.8969** | **[0.869, 0.923]** |
| unweighted Cohen's κ | 0.6549 | — |
| "any ROP" AUC | 0.9695 | — |
| "any ROP" average precision | 0.9662 | prevalence 0.427 |

## External locked, image level (the pre-registered primary unit)

| metric | served fold-0 checkpoint | 5-checkpoint mean ± SD* |
|---|---|---|
| accuracy | 0.5461 | 0.5204 ± 0.0150 |
| balanced accuracy | 0.5197 | — |
| macro-F1 (6 classes) | 0.4756 | 0.4116 ± 0.0491 |
| **QWK (corrected, 5 ordinal)** | **0.4926** | **0.4732 ± 0.0140** |
| QWK (legacy 6-class) | 0.5389 | 0.5232 ± 0.0179 |
| unweighted Cohen's κ | 0.3554 | — |
| "any ROP" AUC | 0.8346 | — |

*\* `±` here is **spread across five checkpoints**, not a confidence interval. It describes checkpoint-to-checkpoint variation on a fixed test set and must never be reported as uncertainty about the population.*

Per-class recall, external locked: Normal **0.996** (n=232) · Stage 1 **0.276** (n=105) · Stage 2 **0.305** (n=164) · Stage 3 **0.292** (n=144) · Stage 4/5 0.750 *(n=4 — not validated)* · AP-ROP 0.500 *(n=12 — not validated)*.

## External locked, eye level (co-primary, max-over-views rule)

QWK **0.8394** [0.768 – 0.890] · accuracy 0.6783 [0.604 – 0.747] · macro-F1 (present classes) 0.6160 [0.528 – 0.718] · **17.7% [11.1 – 25.5] of diseased eyes called Normal**. 230 eyes, patient-clustered intervals.

**Read this number with its operator attached.** Five other aggregation rules on the same predictions give 0.35–0.48. See [`../11_AGGREGATION_AND_VIEW_ANALYSIS/`](../11_AGGREGATION_AND_VIEW_ANALYSIS/).

## The binary screening endpoint

The product's actual decision is `P(any ROP) = Σ prob6[1:] > 0.0155`. Full threshold sweep in [`roc_pr_analysis/any_rop_threshold_table.csv`](roc_pr_analysis/any_rop_threshold_table.csv); curves in [`roc_pr_analysis/any_rop_roc_pr.png`](roc_pr_analysis/any_rop_roc_pr.png).

Internal AUC 0.970 vs external 0.835 is the honest summary: **coarse screening transfers; staging does not.**

⚠️ The served threshold 0.0155 was selected on pooled **out-of-fold** internal scores plus the external **dev** half. The served model is a single fold's checkpoint that saw 4/5 of the internal data, and the external dev half was used in the selection. Neither advertised operating point is an unbiased estimate for this checkpoint. See [`../18_RESEARCH_REVIEW/critical_findings.md`](../18_RESEARCH_REVIEW/critical_findings.md).

## Metrics deliberately NOT reported

| metric | why not |
|---|---|
| per-class ROC/PR curves for stage_4_5 and AP-ROP externally | n = 4 and n = 12. Any curve would be noise dressed as evidence. |
| multiclass one-vs-rest AUC as a headline | it rewards a probability ranking the decode rule does not use (`pred6` and `argmax(prob6)` disagree in 1.19% of cases) |
| "sensitivity / specificity" for the 6-class task | undefined without collapsing to a binary endpoint; the binary endpoint is reported separately |
| F1 macro over all 6 labels at the external site as the *only* macro number | the locked set has structural zeros; both 6-label and present-class macro-F1 are given together |
