# Error Analysis — CURRENT MODEL

*1,433 errors out of 6,700 pooled out-of-fold predictions (error rate 21.4%).
Full atlas: [`error_atlas_all_predictions.csv`](error_atlas_all_predictions.csv) (every prediction, right or wrong).
Errors only: [`error_summary.csv`](error_summary.csv) · Categories: [`error_categories.csv`](error_categories.csv)*

**All case identifiers are salted SHA-256 hashes.** No filename, infant ID, session code, sex, gestational age or birth weight appears anywhere in this folder. See [`../20_AUDIT/privacy_audit.md`](../20_AUDIT/privacy_audit.md).

## Error taxonomy

| category | n | % of errors | mean confidence | n at confidence > 0.90 |
|---|---|---|---|---|
| adjacent-stage confusion | 603 | 42.1% | 0.726 | 91 |
| AP-ROP ↔ stage confusion | 338 | 23.6% | 0.662 | 45 |
| false positive — Normal called diseased | 187 | 13.0% | 0.730 | 39 |
| **⚠ CLINICALLY CRITICAL — disease stage called Normal** | **186** | **13.0%** | **0.934** | **141** |
| **⚠ CLINICALLY CRITICAL — AP-ROP called Normal** | **89** | 6.2% | 0.820 | 39 |
| non-adjacent stage error | 27 | 1.9% | 0.661 | 2 |
| ⚠ CLINICALLY CRITICAL — Stage 4/5 called Normal | 3 | 0.2% | 0.912 | 2 |

Figure: [`confusion_error_matrix.png`](confusion_error_matrix.png)

## The single most important finding in this section

> **The model's average confidence is HIGHEST on the error type that matters most.**
> "Disease stage called Normal" carries mean confidence **0.934** — higher than any other
> error category and higher than the model's overall accuracy (0.786). **141 of those 186
> errors (76%) were made above 0.90 confidence.**

A confidence threshold cannot be used to filter out these errors, because they sit in the top confidence band. This rules out the most obvious safety mitigation ("only act on low-confidence cases") and is the strongest single argument against deploying the current probability output.

## Good news: the error structure is ordinal

603 of 1,433 errors (42%) are one stage away, and only 27 (1.9%) are non-adjacent stage errors. The ordinal head is doing what it was designed to do — when the model is wrong, it is usually *nearly* right, and the severity ranking is preserved. This is the strongest measured argument for the CORN head, and it is a structural benefit that no aggregate metric captures.

## Dominant confusion pairs (pooled OOF)

| pair | count | reading |
|---|---|---|
| Stage 1 → Stage 2 | 180 | the demarcation-line / ridge boundary is genuinely hard |
| Stage 2 → Stage 1 | 193 | …and the confusion is symmetric |
| AP-ROP → Stage 2 | 142 | |
| Normal → Stage 1 | 169 | the source of Stage 1's 0.376 precision |
| Stage 2 → Stage 3 | 69 | |
| AP-ROP → Stage 1 | 107 | |

**Stage 1 is the model's structural weak point.** With precision 0.376, "Stage 1" as an output carries almost no information: it is right about a third of the time. The most likely explanations, in order:
1. genuine clinical ambiguity at the Stage 1/2 boundary (a human ceiling issue);
2. label noise — Stage 1 is the grade most likely to be inconsistently assigned;
3. the class-balanced loss plus the ordinal cascade pushing borderline Normals up one rank.

Only (1) and (2) can be separated by an independent grader. This is the strongest scientific argument for the clinician-adjudication experiment.

## AP-ROP misses, and the fold-0 caveat

AP-ROP pooled recall is 0.345 — the model misses two thirds of aggressive-form frames while being 90% precise when it does fire. **476 of the 618 AP-ROP validation frames come from fold 0, and 470 of those from a single infant** whose diagnosis is a per-session code stamped on every frame of the session. Per-fold AP-ROP recall is 0.212 (fold 0) against 0.694–0.854 on folds 1–4. The pooled 0.345 is therefore dominated by one infant and should not be read as the model's AP-ROP capability.

## Image galleries (de-identified)

| folder | contents |
|---|---|
| [`false_negative_cases/`](false_negative_cases/) | diseased eyes confidently called Normal — the clinically critical failures |
| [`false_positive_cases/`](false_positive_cases/) | healthy eyes called diseased, highest confidence first |
| [`high_confidence_errors/`](high_confidence_errors/) | the >0.90-confidence mistakes |
| [`correct_predictions/`](correct_predictions/) | high-confidence correct predictions, for visual contrast |
| [`incorrect_predictions/`](incorrect_predictions/) | low-confidence mistakes — what "the model was unsure" looks like |

Each folder contains a `_contact_sheet.png`, individual PNGs named by case hash, and a `manifest.csv` mapping case hash → true class, predicted class, confidence, error category and source site.

## Caveats on this analysis

- Errors are scored against the corpora's **own labels**, which for Ostrava are per-session and for Multi-View per-eye. A share of the "errors" here — especially "disease called Normal" on non-temporal views — are the model correctly describing a photograph that does not contain the finding. The external region analysis in [`../11_AGGREGATION_AND_VIEW_ANALYSIS/`](../11_AGGREGATION_AND_VIEW_ANALYSIS/) quantifies this; no equivalent view metadata exists for the internal corpus, so the internal error counts cannot be decomposed the same way.
- The confidence figures inherit the calibration defect in [`../10_CONFIDENCE_CALIBRATION/`](../10_CONFIDENCE_CALIBRATION/); "0.934 mean confidence" means the model's *stated* certainty, which is systematically inflated.
