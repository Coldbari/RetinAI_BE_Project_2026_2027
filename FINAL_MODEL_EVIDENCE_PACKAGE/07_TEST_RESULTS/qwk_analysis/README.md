# QWK Correction — what changed, why, and the full before/after

**Nothing was silently overwritten. Both values are reported everywhere.**

Ledger: [`qwk_correction_ledger.csv`](qwk_correction_ledger.csv) · Figure: [`qwk_correction.png`](qwk_correction.png)

---

## The defect

Both staging configs declare:

```yaml
eval:
  qwk_classes: ["Normal", "Stage 1", "Stage 2", "Stage 3", "Stage 4/5"]
  # QWK is computed over the 5 ORDINAL classes only. AP-ROP is not a point on the
  # stage scale, so including it makes the kappa weights arbitrary.
```

A repository-wide grep finds **zero** Python references to `qwk_classes`. The metric that actually ran was:

```python
# models/common/metrics.py
"qwk": cohen_kappa_score(y_true, y_pred, weights="quadratic")   # over all 6 integer labels
```

With AP-ROP encoded as class index **5**, quadratic weighting treats it as one step beyond Stage 4/5 on the severity ladder. An AP-ROP image called Normal is charged (5−0)² = 25, versus 16 for Stage 4/5 → Normal — **the exact modelling error the structured head exists to avoid.**

## The correction

```python
def qwk_ordinal(y, p):
    m = (y != AROP) & (p != AROP)          # AP-ROP is a FORM, not a rank
    return cohen_kappa_score(y[m], p[m], weights="quadratic")
```

Pairs where either the truth or the prediction is AP-ROP are excluded, and the number of pairs actually used is recorded alongside every value.

## Verification performed before adopting the corrected metric

| check | result |
|---|---|
| class index mapping matches `results/prepared/rop_staging/class_index.json` | ✅ `normal 0, stage_1 1, stage_2 2, stage_3 3, stage_4_5 4, arop 5` |
| ordinal ordering is Normal < S1 < S2 < S3 < S4/5 | ✅ matches ICROP-3 and the CORN head's rank axis |
| AP-ROP excluded, not remapped | ✅ remapping AP-ROP to any rank would re-introduce the defect |
| weight matrix is quadratic | ✅ `weights="quadratic"` — `w[i,j] = (i−j)²/(k−1)²` |
| legacy value reproducible | ✅ the legacy implementation is reimplemented in `common.py::qwk_6class_legacy` and reproduces every previously published value to 4 dp |
| label order in the `.npz` matches the manifest | ✅ asserted row-by-row for all 5 folds |
| degenerate cases handled | ✅ returns NaN, never 0, when fewer than 2 classes survive the mask |

## Impact — CURRENT MODEL, cross-validation

| scope | previously reported (6-class) | **corrected (5 ordinal)** | Δ | pairs used |
|---|---|---|---|---|
| fold 0 | 0.5685 | **0.9077** | **+0.339** | 1,213 / 1,691 |
| fold 1 | 0.8642 | 0.8467 | −0.018 | 1,216 |
| fold 2 | 0.8731 | 0.9073 | +0.034 | 1,212 |
| fold 3 | 0.9020 | 0.8996 | −0.002 | 1,203 |
| fold 4 | 0.9197 | 0.9206 | +0.001 | 1,213 |
| **mean ± SD** | 0.8255 ± 0.1454 | **0.8964 ± 0.0288** | +0.071 | — |
| **pooled** | 0.7610 | **0.8969** | +0.136 | 6,057 / 6,700 |

## Impact — external locked set

| arm | legacy 6-class | **corrected** |
|---|---|---|
| CURRENT structured s42 (5 ckpt mean) | 0.5232 ± 0.0179 | **0.4732 ± 0.0140** |
| structured s1337 | 0.5085 ± 0.0158 | **0.4556 ± 0.0232** |
| flat baseline s42 | 0.5606 ± 0.0225 | **0.4924 ± 0.0209** |
| flat baseline s1337 | 0.5527 ± 0.0184 | **0.4856 ± 0.0218** |
| CURRENT structured, **eye level (max rule)** | 0.8012 | **0.8127** |
| CURRENT structured, **temporal frames only, image level** | — | **0.8406** |

Note the correction moves the external number **down**, not up. It is not a flattering correction; it is the correct one.

## What this changes about the project's conclusions

| claim | before | after |
|---|---|---|
| "fold 0 is anomalous, and QWK is one of three independent signatures" | supported | **the QWK leg is withdrawn** — under the declared metric fold 0 is the second-best fold. The macro-F1 and calibration signatures survive. |
| "structured ≈ flat" | supported (QWK 0.826 vs 0.834) | **more strongly supported** — 0.8964 vs 0.8962, p = 0.977 |
| "the evaluation unit dominates" | image 0.52 → eye 0.80 | **image 0.47 → eye 0.81** — the gap is slightly *larger* under the corrected metric |
| fold-to-fold stability of QWK | SD 0.145 | **SD 0.029** — 5× more stable; the metric was mostly measuring AP-ROP prevalence per fold |

## Scope of the correction

Applied to: all 5 CV folds × 4 arms; the external dev and locked sets × 4 arms × 5 checkpoints; all six eye-level aggregation rules; all three anatomical regions. **Not** re-run: the seven-arm fold-0 ablation table and the five-backbone benchmark, whose raw `.npz` files exist for only some arms — those remain in [`../19_COMPARISONS/`](../19_COMPARISONS/) with legacy values and a warning label.
