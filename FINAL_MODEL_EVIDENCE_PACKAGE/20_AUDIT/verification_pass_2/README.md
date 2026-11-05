# Verification pass 2 — outputs

Independent second pass. Loader: `FINAL_MODEL_EVIDENCE_PACKAGE/_build/common.py`;
scripts `p1_data.py` (dataset, leakage), `p2_evaluation.py` (QWK definitions),
`p3_units.py` (aggregation, views, clustered CIs, per-class, confusion),
`p4_screening_calib.py` (screening, calibration, errors), `p5_deploy.py`, `p6_figs.py`.

Every script asserts row alignment between each `.npz` and its manifest CSV before computing
anything (`label == y_true` for all six folds and both test halves); a misalignment aborts.

| Path | Contents |
|---|---|
| `02_DATASET/` | reconciliation (10 stages, claimed vs recomputed), class distribution, source×class matrix, label granularity |
| `03_DATA_LEAKAGE/` | exact duplicates (groups + members), label conflicts, split leakage matrix, full-corpus perceptual duplicates (7,680 pairs with per-pair provenance), patient-ID semantics |
| `06_VALIDATION/` | QWK under three class definitions × 2 halves × 2 arms × 2 seeds × 2 units + 6 CV rows per arm-seed (40 rows) |
| `07_TEST_RESULTS/` | per-class (image + eye), confusion (counts + row-normalised, both units), screening at 3 units, screening clustered CIs |
| `08_EYE_LEVEL_ANALYSIS/` | 8 aggregation rules × 2 halves; clustered CIs for 4 metrics × 2 units × 2 halves |
| `09_ERROR_ANALYSIS/` | current-model error summary, most-confused pairs, high-confidence errors, errors by view |
| `10_CONFIDENCE_CALIBRATION/` | structured vs flat calibration, per-class calibration, per-class gap by site, standardised site calibration |
| `11_AGGREGATION_AND_VIEW_ANALYSIS/` | view-stratified performance |
| `16_STATISTICAL_ANALYSIS/` | consolidated confidence intervals |

Figures added to the package proper (not duplicated here):
`07_TEST_RESULTS/screening_performance.png`, `10_CONFIDENCE_CALIBRATION/site_calibration_corrected.png`.
