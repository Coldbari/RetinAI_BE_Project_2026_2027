# Final Verification Report

*Run 2026-09-21 19:11 · commit `f0d45087` · script `20_AUDIT/scripts/s12_final_validation.py`*

## 23 of 25 checks PASS

| # | check | result | detail |
|---|---|---|---|
| 1 | PHI: no sex+GA+birth-weight identifier pattern (this build's files) | ✅ **PASS** | 0 hits |
| 2 | PHI: no original Ostrava filenames in tables/docs (this build's files) | ✅ **PASS** | 0 hits |
| 3 | PHI: no raw image paths in any CSV (this build's files) | ✅ **PASS** | 0 hits |
| 4 | PHI: no raw identifiers in files written by OTHER processes | ❌ **FAIL** | 5 foreign files contain raw image paths - NOT produced by this build; remove or hash before sharing. See 20_AUDIT/privacy_audit.md section 6.2. Examples: ['03_DATA_LEAKAGE/exact_duplicate_members.csv', '03_DATA_LEAKAGE/perceptual_duplicate_audit.csv', '20_AUDIT/verification_pass_2/03_DATA_LEAKAGE/exact_duplicate_members.csv'] |
| 5 | De-identification: all case ids are salted hashes | ✅ **PASS** | 6788 unique hashed case ids |
| 6 | Separation: no retired-model files outside 19_COMPARISONS | ✅ **PASS** | 0 files |
| 7 | Separation: every retired file carries a STATUS block | ✅ **PASS** | missing: [] |
| 8 | Figures: generated | ✅ **PASS** | 140 png files |
| 9 | Metric: QWK excludes AP-ROP pairs | ✅ **PASS** | pairs kept=5 (expected 5), value=1.000000 |
| 10 | Metric: legacy 6-class QWK differs from corrected | ✅ **PASS** |  |
| 11 | Metric: class index matches the ordinal axis | ✅ **PASS** | {'normal': 0, 'stage_1': 1, 'stage_2': 2, 'stage_3': 3, 'stage_4_5': 4, 'arop': 5} |
| 12 | Numbers: CV macro-F1 == 0.6923 ± 0.0861 | ✅ **PASS** | 0.6923 ± 0.0861 |
| 13 | Numbers: corrected CV QWK == 0.8964 | ✅ **PASS** | 0.8964 |
| 14 | Numbers: external served-checkpoint accuracy == 0.5461 | ✅ **PASS** |  |
| 15 | Checkpoint: served weights uniquely identified | ✅ **PASS** | fold0_ckpt agreement=1.0 |
| 16 | Checkpoint: sha256 recorded in CURRENT_MODEL_ID.txt | ✅ **PASS** |  |
| 17 | Denominators: per-class support column present and non-zero | ✅ **PASS** | supports=[3837, 577, 959, 620, 89, 618] |
| 18 | Denominators: tiny external classes flagged | ✅ **PASS** | stage_4_5 n=4, arop n=12 — flagged unvalidated everywhere |
| 19 | Uncertainty: every CI resamples infants | ✅ **PASS** |  |
| 20 | Uncertainty: checkpoint spread never labelled a CI | ✅ **PASS** |  |
| 21 | Leakage: zero patient overlaps | ✅ **PASS** |  |
| 22 | Leakage: zero image overlaps | ✅ **PASS** |  |
| 23 | Leakage: val folds exactly partition the corpus | ✅ **PASS** |  |
| 24 | Structure: all required documents present | ✅ **PASS** | missing: [] |
| 25 | Repository test suite passes | ❌ **FAIL** | 1 failed, 151 passed, 4 skipped in 54.10s - the single failure is the PRE-EXISTING DEMO.md PHI guard (finding C-1), not caused by this package |
