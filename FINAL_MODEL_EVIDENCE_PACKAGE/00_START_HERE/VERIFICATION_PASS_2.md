# Verification pass 2 — read this before the rest of the package

This package was built in two independent passes over the same raw artifacts.

**Pass 1** (`20_AUDIT/scripts/s01…s11`) built the package you see: 267 files, all recomputed
from raw per-sample predictions.

**Pass 2** (this one, `20_AUDIT/verification_pass_2/`) re-derived pass 1's conclusions with a
separately written pipeline, and added five analyses pass 1 did not contain. It was run as a
hostile review, not a confirmation.

## The two passes agree on everything they both measured

Same conclusions, independently derived: the QWK class-definition defect, the `max`-operator
dependence of the eye-level headline, the temporal-view control that reproduces it, the
structured-vs-flat calibration disadvantage, confident misses as the dangerous error class,
the served checkpoint being fold-0, zero patient/path leakage across split boundaries, and
every dataset count. Numeric differences between the two are convention only — pass 1 quotes
QWK under the 5-class *drop* rule on the served checkpoint alone, pass 2 quotes 6-class and
5-class *map* averaged over 5 checkpoints. Both are tabulated; the recommendation is in
[`../06_VALIDATION/QWK_DEFINITION_CORRECTION.md`](../06_VALIDATION/QWK_DEFINITION_CORRECTION.md).

## What pass 2 adds

| Analysis | Why it was missing | Where |
|---|---|---|
| **Screening at image / eye / infant on the locked half** | never computed by anyone; the deployed endpoint had no out-of-sample evaluation at the unit of decision | [`07_TEST_RESULTS/LOCKED_SCREENING_EVALUATION.md`](../07_TEST_RESULTS/LOCKED_SCREENING_EVALUATION.md) |
| **Full-corpus duplicate + near-duplicate audit, and `patient_id` semantics** | no prior audit hashed image content across folds | [`03_DATA_LEAKAGE/DUPLICATE_AND_PATIENT_ID_AUDIT.md`](../03_DATA_LEAKAGE/DUPLICATE_AND_PATIENT_ID_AUDIT.md) |
| **Case-mix standardised site calibration** | the committed "case-mix" conclusion controlled on the easiest class | [`10_CONFIDENCE_CALIBRATION/SITE_CALIBRATION_CORRECTION.md`](../10_CONFIDENCE_CALIBRATION/SITE_CALIBRATION_CORRECTION.md) |
| **Label granularity from the source annotation table** | the label-unit thesis rested on a statistical argument | [`02_DATASET/LABEL_GRANULARITY_ANALYSIS.md`](../02_DATASET/LABEL_GRANULARITY_ANALYSIS.md) |
| **Eye-level clustered CIs + eight aggregation rules on both halves** | the headline had no interval in any artifact | [`11_AGGREGATION_AND_VIEW_ANALYSIS/aggregation_mechanism_analysis.md`](../11_AGGREGATION_AND_VIEW_ANALYSIS/aggregation_mechanism_analysis.md) |

## The one finding that outranks everything else

On the locked test, at the deployed threshold, with the served checkpoint and the any-frame
rule a screening service uses:

| Unit | Sensitivity | Specificity |
|---|---:|---:|
| image | 0.911 [0.879, 0.942] | 0.323 [0.252, 0.392] |
| eye | 0.994 [0.979, 1.000] | 0.052 [0.011, 0.110] |
| **infant** | **1.000** | **0.000 — all 39 healthy infants flagged** |

It replicates on the dev half and across all five checkpoints. It is a decision-rule problem,
not a proven model problem, and it is fixable without retraining.

## Two housekeeping notes

- Pass 2 initially wrote 33 files over pass-1 names. All pass-1 outputs were regenerated from
  their own scripts; pass-2 outputs live in `20_AUDIT/verification_pass_2/` mirroring the
  package layout, with a `_v2` suffix where a name genuinely collides. Nothing was lost.
- **`20_AUDIT/PHI_PRIVACY_AUDIT.md` contains an open release blocker.** Read it before any push.
