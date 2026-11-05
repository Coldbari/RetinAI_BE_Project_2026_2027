# Results — Consolidated Metrics & Validation

Every number here is grounded in an on-disk artifact under `results/` (or
[thesis_assets/RESULTS_SUMMARY.md](../thesis_assets/RESULTS_SUMMARY.md)). Internal = held-out
test/val from the same data source; External = an independent public dataset with **no
retraining**.

> **Decision-support prototype — not a diagnostic device.** Operating points are
> population-specific and were *not* prospectively validated.

---

## Headline per disease

| Disease | Arch | Internal (primary) | External | Calibration (ECE) | Source |
|---|---|---|---|---|---|
| **ROP** | ResNet50 | **AUC 0.927**, sens 0.990 / spec 0.398 @ deployed thr 0.1933 | — (no public ROP set) | 0.059 → 0.041 | `results/rop/metrics.json` |
| **DR (referable, grade ≥ 2)** | EfficientNetV2-S | **AUC 0.906** (95% CI 0.89–0.92); argmax sens 0.84 / spec 0.89 | Messidor-2 **AUC 0.837** (−6.9%, **PASS**) | 0.141 → 0.111 (T = 0.70) | `results/dr/analysis_summary.json` |
| DR (5-class) | EfficientNetV2-S | QWK 0.728, macro-F1 0.586, acc 74.4%, AUC 0.788 | Messidor-2 QWK 0.654, AUC 0.600 | 0.181 → 0.133 (T = 0.78) | `results/dr/metrics.json` |
| **Glaucoma (multi-source)** | EfficientNetV2-S | **AUC 0.973**, acc 92.4%, QWK 0.84 | per-source **G1020 0.957** | — | `results/glaucoma/metrics.json` |
| Glaucoma (SMDG-only, superseded) | EfficientNetV2-S | AUC 0.957 | G1020 **0.589** zero-shot (FAIL) | — | `results/external_glaucoma.json` |

---

## ROP — ResNet50 (binary; primary = recall)

`results/rop/metrics.json`, trained 12 epochs, `best_val_primary` (macro-recall) 0.8416.

| Split | Acc | Macro-F1 | QWK | **AUC** | Sensitivity | Specificity |
|---|---|---|---|---|---|---|
| Val (928) | 0.747 | 0.740 | 0.499 | 0.842 | 0.670 | 0.907 |
| **Test (1,502)** | **0.844** | 0.843 | 0.688 | **0.927** | 0.799 | 0.898 |

- **Deployable operating point** (`configs/rop.yaml`, `results/rop/operating_point.json`): threshold **0.1933**, selected on the clean val split, measured once on test with the **5-view TTA path the app actually serves** → **sensitivity 0.990 [0.981, 0.996] / specificity 0.398 [0.361, 0.436]**. The table's 0.799/0.898 are at the default argmax point.
  - **Quote TTA numbers or single-view numbers, never one of each.** `results/rop_statistics.md` scores the same threshold on single-view predictions and gets sens 0.982 / spec 0.456. Neither is wrong; they are different scoring paths. Serving uses TTA, so TTA is the deployed reality.
  - The previously published pair — thr 0.371 → sens 0.90 / spec 0.83 — is **withdrawn**. That threshold was read off the *test* ROC, so quoting test performance at it is circular. Re-selecting on val gives 0.182, and applying it once to test yields **spec 0.414** against the 0.827 that test-selection implied — an overstatement of **+0.41**.
  - **Per infant, specificity is 0.000** — an infant counts as flagged if *any* of their images is flagged, and 0 of 22 healthy infants were cleared. At this operating point the model is a sensitivity-maximising triage filter, not a rule-out test.
- Test confusion `[[609, 69], [166, 658]]`; per-class F1: No ROP 0.838, ROP 0.848.
- Error analysis (`results/rop/errors/error_summary.json`): 73 false positives, 177 false negatives.
- **Architecture choice justified by a sweep** (see below) — ResNet50 beat EfficientNetV2-S (superseded variant `metrics_overfit.json`: test AUC 0.915 vs 0.927).

### ROP architecture sweep (`results/sweep_rop.csv`, 320px / 12 epochs)

| arch | acc | macro-F1 | QWK | **AUC** | params (M) | CPU ms | GPU fps |
|---|---|---|---|---|---|---|---|
| efficientnet_b0 | 0.758 | 0.757 | 0.525 | 0.867 | 4.0 | 56 | 138 |
| efficientnet_v2_s | 0.706 | 0.701 | 0.431 | 0.870 | 20.2 | 130 | 59 |
| mobilenet_v3_large | 0.744 | 0.743 | 0.500 | 0.857 | 4.2 | 34 | 163 |
| **resnet50** ✓ | **0.886** | **0.882** | **0.766** | **0.964** | 23.5 | 157 | 137 |
| densenet121 | 0.764 | 0.764 | 0.537 | 0.887 | 7.0 | 136 | 70 |

**ResNet50** wins clearly on the small ROP set — best AUC with competitive latency.

---

## DR — EfficientNetV2-S (5-class; primary = QWK; referable_grade = 2)

`results/dr/metrics.json`, 25 epochs, `best_val_primary` (QWK) 0.7281.

**5-class validation (5,815 imgs):** acc 0.744, macro-P 0.595, macro-R 0.598, macro-F1 0.586,
weighted-F1 0.764, **QWK 0.728**, AUC 0.788.

### Per-grade (internal val)

| Grade | Precision | Recall | F1 | n |
|---|---|---|---|---|
| No DR | 0.926 | 0.798 | 0.857 | 4,142 |
| Mild | 0.241 | 0.406 | 0.302 | 421 |
| Moderate | 0.539 | 0.731 | 0.620 | 943 |
| Severe | 0.531 | 0.428 | 0.474 | 159 |
| Proliferative | 0.740 | 0.627 | 0.679 | 150 |

- **Weakest class: Mild (F1 0.30, error rate 0.59).** The No-DR ↔ Mild boundary is the
  early-disease challenge — top confusions are No DR → Mild (×443) and No DR → Moderate (×384).
- **Referable-DR (grade ≥ 2), the honest screening endpoint:** internal **AUC 0.906**
  (95% CI 0.89–0.92); argmax sens 0.84 / spec 0.89.

### DR external — Messidor-2 (1,744 imgs, no retrain) — `results/external_dr.json`
- 5-class: acc 0.679, QWK 0.654, AUC 0.600 (internal 0.788 → **drop 0.188**).
- **Referable: AUC 0.837** (95% CI 0.809–0.863); internal 0.906 → external 0.837 = **6.9% drop = PASS** (≤ 10% target).
- Operating point does **not** transfer: at target sens 0.90 the external threshold gives spec 0.30 — per-site threshold tuning is required.

---

## Glaucoma — EfficientNetV2-S (binary; primary = AUC; multi-source)

`results/glaucoma/metrics.json`, 25 epochs, `best_val_primary` (AUC) 0.9643.

| Split | Acc | Macro-F1 | QWK | **AUC** | Sensitivity | Specificity |
|---|---|---|---|---|---|---|
| Val (1,333) | 0.914 | 0.909 | 0.817 | 0.964 | 0.889 | 0.929 |
| **Test (1,999)** | **0.924** | 0.920 | 0.839 | **0.973** | 0.908 | 0.934 |

### The domain-shift story (`results/glaucoma/multisource_persource.json`)
- SMDG-only model: internal AUC 0.957 → **G1020 zero-shot AUC 0.589** (−36.8%, **FAIL**).
- Fix = pool SMDG-19 + G1020 (13,336 imgs), stratified so both sources are in each split.
- Result — **per-source test AUC:** overall **0.973**, SMDG **0.974**, **G1020 0.957** (recovered from 0.589).
- Honest caveat: G1020 is now *in-distribution*; a fresh zero-shot claim needs a new held-out source.

---

## Calibration (temperature scaling) — `results/dr/analysis_summary.json`

Post-hoc temperature scaling reduces over/under-confidence (lower ECE = better calibrated):

| Model | Temperature | ECE before → after | Brier before → after |
|---|---|---|---|
| DR 5-class | 0.782 | 0.181 → 0.133 | 0.447 → 0.422 |
| DR referable | 0.697 | 0.141 → 0.111 | — |
| ROP | — | 0.059 → 0.041 | (reliability PNGs only) |

⚠️ **Implementation gap:** this calibration is computed **offline**; the live web app does
**not** apply temperature scaling at serving time — displayed confidence is the raw
TTA-averaged softmax. See [LIMITATIONS.md](LIMITATIONS.md).

---

## Success tiers (`evaluate.py`)

| Disease | Metric | Acceptable / Target / Stretch | Achieved |
|---|---|---|---|
| DR | QWK | 0.75 / 0.80 / 0.85 | 0.728 (below Acceptable on 5-class; referable AUC 0.906 is the real target) |
| ROP | macro-recall | 0.85 / 0.90 / 0.93 | 0.842 val / operating sens 0.90 |
| Glaucoma | AUC | 0.90 / 0.92 / 0.95 | **0.973 (Stretch)** |

---

## Artifacts on disk

- **Weights:** `results/{dr,rop,glaucoma}/weights.pth` (+ superseded `weights_smdg_only.pth`, `weights_overfit_v2s.pth`).
- **Metrics:** `results/<disease>/metrics.json`, `analysis_summary.json`, `multisource_persource.json`, `external_{dr,glaucoma}.json`.
- **Plots:** DR training curves (`results/dr/plots/training_eval.png`); reliability diagrams for DR 5-class, DR referable, and ROP (calibrated + uncalibrated).
- **Thesis package:** `thesis_assets/` has 7 Figures, 2 Tables, `RESULTS_SUMMARY.md`, and `assets_manifest.json` mapping each asset to its source.

### Two honest measurement notes
1. `results/dr/metrics.json` has only a **val** block, so the **dashboard's DR card shows
   5-class val AUC 0.788**, while the About page cites the **referable-DR AUC 0.906** — two
   different measurements of the same model.
2. The persisted DR run's `results/dr/config.yaml` recorded `circle_crop/ben_graham/clahe =
   false`, even though the curated `configs/dr.yaml` template enables them — the DR checkpoint
   on disk may have trained on minimally-processed images. Both are tracked in
   [LIMITATIONS.md](LIMITATIONS.md).
