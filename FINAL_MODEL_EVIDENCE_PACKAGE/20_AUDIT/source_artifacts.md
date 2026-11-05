# Source Artifacts — provenance for every major result

*Common to every row unless overridden:*

```
MODEL             : structured EfficientNetV2-S (CORN + AP-ROP branch + DANN), variant "full"
MODEL CHECKPOINT  : seed 42, natural sampling, 20 epochs, last-epoch weights
                    served file = fold-0 checkpoint, sha256 11b439608690024700cdc927dece45d51ebc78b648aef81090cb6be3e7dc747c
DATASET           : results/prepared/rop_staging/ (committed split CSVs)
GIT COMMIT        : f0d4508707970697280f63943ef4f4496e54bafe
DATE              : 2026-09-21
ENVIRONMENT       : torch 2.12.0 · python 3.14.6 · numpy 2.4.6 · scikit-learn 1.8.0 · macOS-27.0-arm64
```

---

## Internal cross-validation

```
DATASET SPLIT     : fold_0.csv … fold_4.csv, split=="val" (6,700 images, 1,528 infants)
EVALUATION UNIT   : image
METRIC            : accuracy · balanced accuracy · macro-F1 · weighted-F1 · per-class P/R/F1
                    · QWK (5 ordinal classes) · legacy 6-class QWK · unweighted κ · any-ROP AUC/AP
CALCULATION SCRIPT: 20_AUDIT/scripts/s03_cv_metrics.py
SOURCE ARTIFACT   : results/rop/staging_runs/structured_5fold/fold_{0..4}.npz
                    (y_true, y_pred, prob6 — written by train_structured.py at training time)
OUTPUTS           : 06_VALIDATION/fold_results.csv · validation_metrics.csv
                    07_TEST_RESULTS/metrics_summary.csv · current_model_oof_predictions.npz
INTEGRITY CHECK   : y_true asserted equal to the manifest's label column, row by row, all 5 folds
```

## Corrected QWK and the correction ledger

```
EVALUATION UNIT   : image / eye / region (all scopes)
METRIC            : QWK, both definitions, with the pair count after excluding AP-ROP
CALCULATION SCRIPT: 20_AUDIT/scripts/common.py::qwk_ordinal and ::qwk_6class_legacy
SOURCE ARTIFACT   : the same .npz files as above
OUTPUTS           : 07_TEST_RESULTS/qwk_analysis/qwk_correction_ledger.csv · qwk_correction.png
                    19_COMPARISONS/retired_models/legacy_qwk_values.csv
```

## External held-out hospital

```
DATASET SPLIT     : test_locked.csv (661 images, 230 eyes, 116 infants) — opened once, 2026-08-18
                    test_dev.csv (663 images) for the served-checkpoint re-run and Grad-CAM only
EVALUATION UNIT   : image (pre-registered primary) and eye (co-primary, max-over-views)
CALCULATION SCRIPT: 20_AUDIT/scripts/s04_external_eye_region.py
SOURCE ARTIFACT   : results/rop/locked_eval/structured_s42_ckpt{0..4}.npz
                    data/rop/multiview/ROP-DATASET/Dataset Acquisition View Table.xlsx  (eye + region join)
PROTOCOL          : docs/paper/LOCKED_EVAL_PROTOCOL.md (committed before the set was opened)
OUTPUTS           : 13_EXTERNAL_GENERALIZATION/* · 08_EYE_LEVEL_ANALYSIS/* · 11_AGGREGATION_AND_VIEW_ANALYSIS/*
INTEGRITY CHECK   : view-table join asserted complete (661/661) before any aggregation
```

## Served-checkpoint identification

```
EVALUATION UNIT   : image
METRIC            : prediction agreement and max |Δ probability| vs each committed checkpoint
CALCULATION SCRIPT: 20_AUDIT/scripts/s01_identity.py
SOURCE ARTIFACT   : results/rop_staging/weights.pth  (re-run over test_dev.csv)
                    compared against results/rop/dev_eval/structured_s{42,1337}_ckpt{0..4}.npz
OUTPUTS           : 20_AUDIT/served_checkpoint_identification.json · served_checkpoint_dev_rerun.npz
```

## Training behaviour

```
EVALUATION UNIT   : epoch
METRIC            : total/stage/AP-ROP/site training loss · DANN λ · validation macro-F1 · validation QWK
CALCULATION SCRIPT: 20_AUDIT/scripts/s07_training.py
SOURCE ARTIFACT   : kaggle_cache/struct42/rop-structured-effnetv2s-s42.log (120 epoch records, 6 runs)
OUTPUTS           : 05_TRAINING/training_history.csv · last_epoch_noise.csv · training_curves/*
NOT MEASURED      : validation loss, realised learning rate. The LR figure is a labelled
                    RECONSTRUCTION from the trainer's hardcoded OneCycleLR settings.
CROSS-CHECK       : final val macro-F1 in the log == value recomputed from the .npz, all 5 folds
```

## Calibration and confidence

```
EVALUATION UNIT   : image
METRIC            : ECE (15 equal-width bins) · multiclass Brier · confidence distributions
                    · high-confidence error counts · decode-vs-argmax disagreement rate
CALCULATION SCRIPT: 20_AUDIT/scripts/s05_calibration.py
SOURCE ARTIFACT   : structured_5fold/*.npz · flat_natural_5fold/*.npz · locked_eval/structured_s42_ckpt0.npz
OUTPUTS           : 10_CONFIDENCE_CALIBRATION/*
```

## Error analysis and galleries

```
EVALUATION UNIT   : image
METRIC            : error taxonomy · per-category confidence · high-confidence error confusion
CALCULATION SCRIPT: 20_AUDIT/scripts/s06_errors.py
SOURCE ARTIFACT   : 07_TEST_RESULTS/current_model_oof_predictions.npz + the fold manifests
OUTPUTS           : 09_ERROR_ANALYSIS/* · 17_USER_FACING/07_error_examples/
DE-IDENTIFICATION : case_id = "CASE-" + sha256(salt || image_path)[:10]. No filename, infant ID,
                    session code, sex, gestational age or birth weight is written anywhere.
```

## Explainability

```
EVALUATION UNIT   : image
METRIC            : attention fraction on retina / synthetic padding / camera surround · peak location
CALCULATION SCRIPT: 20_AUDIT/scripts/s08_gradcam.py
SOURCE ARTIFACT   : results/rop_staging/weights.pth (the SERVED checkpoint), run fresh over an
                    88-image class-stratified sample of test_dev.csv
TARGET LAYER      : model.backbone.features[-1]
OBJECTIVE         : log Σ prob6[1:]  — the SERVED screening objective
MASKS             : models/common/data_quality.py::retina_mask, plus an explicit
                    synthetic-letterbox-padding mask so invented pixels are counted separately
OUTPUTS           : 14_EXPLAINABILITY/*
CORROBORATION     : agrees with results/rop/cam_leakage.json (n=663) to within 1 percentage point
```

## Dataset and leakage

```
EVALUATION UNIT   : image / infant
CALCULATION SCRIPT: 20_AUDIT/scripts/s02_dataset_leakage.py
SOURCE ARTIFACT   : results/prepared/rop_staging/{fold_*,test_dev,test_locked}.csv
OUTPUTS           : 02_DATASET/* · 03_DATA_LEAKAGE/*
NOT RECOMPUTED    : perceptual near-duplicate sweep (inherited from the parent project's gate G2);
                    site-decodability probe (gate G3); metadata-only baseline (gate G4)
```

## Statistical uncertainty

```
CALCULATION SCRIPT: 20_AUDIT/scripts/common.py::cluster_bootstrap  (+ s11_bootstrap_consolidate.py)
METHOD            : patient-clustered percentile bootstrap, 2,000 replicates (1,000 for per-class),
                    numpy default_rng(42), resampling INFANTS so all of an infant's rows move together
OUTPUTS           : 16_STATISTICAL_ANALYSIS/confidence_intervals.csv · bootstrap_results.csv
                    · paired_tests.csv
```

## Model, deployment, comparison

```
CALCULATION SCRIPT: 20_AUDIT/scripts/s09_comparison_prep_deploy.py
SOURCE ARTIFACT   : results/rop_staging/weights.pth · configs/rop_staging_structured.yaml
OUTPUTS           : 01_MODEL/architecture.txt · model_parameters.txt · model_configuration.yaml
                    04_PREPROCESSING/* · 12_MODEL_COMPARISON/* · 15_DEPLOYMENT/*
LATENCY           : measured on CPU (Apple Silicon), single image, no batching, 10 repeats
```

## User-facing figures

```
CALCULATION SCRIPT: 20_AUDIT/scripts/s10_userfacing.py
SOURCE            : derived entirely from the CSVs and .npz files listed above — no new computation
OUTPUTS           : 17_USER_FACING/01..10
```

---

## Artifacts deliberately NOT used

| artifact | why excluded |
|---|---|
| `results/rop/weights.pth` (94 MB) | the **retired ResNet50** binary screener — not the current model |
| `results/rop/preds_*_overfit_*.npz`, `tta_*_preds.npz`, `devsplit_*` | belong to the retired binary screening pipeline |
| `results/rop/staging_runs/ablations/*` | fold-0-only ablation arms; legacy QWK, not re-scored — quarantined in `19_COMPARISONS/` |
| `results/rop/staging_runs/staging_*_s42.json` + kernel logs | the five-backbone flat benchmark — provisional, quarantined |
| `graphs/`, `thesis_assets/`, `reports/` | pre-existing figures from mixed model generations; **none is copied into this package** |
| `junk/**` | parked DR and glaucoma work, out of scope |
