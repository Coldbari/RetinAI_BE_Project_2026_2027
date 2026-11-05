# QWK definition — the declared endpoint was never the computed one

*Verification pass 2. Recomputed from `results/rop/locked_eval/*.npz`, `results/rop/dev_eval/*.npz`
and `results/rop/staging_runs/*/fold_*.npz`. Table:
[`20_AUDIT/verification_pass_2/06_VALIDATION/qwk_5class_vs_6class_recalculation.csv`](../20_AUDIT/verification_pass_2/06_VALIDATION/qwk_5class_vs_6class_recalculation.csv)
(40 rows: 2 test halves × 2 arms × 2 seeds × 2 units, plus 6 CV rows per arm-seed).*

## The defect

`configs/rop_staging_structured.yaml` and `configs/rop_staging.yaml` both declare:

```yaml
eval:
  qwk_classes: ["Normal", "Stage 1", "Stage 2", "Stage 3", "Stage 4/5"]   # five ordinal levels
  unvalidated_classes: ["Stage 4/5", "AP-ROP"]
```

A repository-wide grep for `qwk_classes` returns **zero Python references**. The key is a
no-op. Every QWK in this project is `cohen_kappa_score(y, p, weights="quadratic")` over all
six labels, which places **AP-ROP at ordinal rank 5** — one step beyond Stage 4/5 on a
severity ladder it does not belong to. AP-ROP is a distinct phenotype, which is exactly why
the architecture gives it its own sigmoid branch.

The same config file carries the comment *"Do not add keys the code does not consult — that
mistake produced silent no-ops twice in this project already."* This is the third.

## Three defensible numbers, one recommendation

| Variant | Rule | What it does to errors |
|---|---|---|
| `qwk_6class` | all six labels, AP-ROP = rank 5 | **what every existing artifact reports**; penalises an AP-ROP↔Normal confusion as a 5-step ordinal error |
| `qwk_5class_map` | AP-ROP folded onto Stage 4/5 on both sides | keeps every row; treats AP-ROP as severity ≥ Stage 4/5 |
| `qwk_5class_drop` | rows with true **or predicted** AP-ROP removed | **optimistically biased** — it discards rows the model got wrong by predicting AP-ROP |

**Recommendation: report `qwk_5class_map` as the primary ordinal endpoint**, with
`qwk_6class` shown for continuity with prior reports, and AP-ROP scored separately as the
binary branch metric the architecture actually provides. `qwk_5class_drop` should not be a
headline: on CV fold 0 it removes 476 AP-ROP rows *and* every row the model wrongly called
AP-ROP, which is why it produces the largest jump in the table below.

## Impact (structured, seed 42)

| Dataset | Unit | n | 6-class (reported) | 5-class map | 5-class drop | Δ map |
|---|---|---:|---:|---:|---:|---:|
| CV fold 0 | image | 1,691 | 0.5685 | 0.6437 | 0.9077 | **+0.075** |
| CV fold 1 | image | 1,252 | 0.8642 | 0.8595 | 0.8467 | −0.005 |
| CV fold 2 | image | 1,252 | 0.8731 | 0.8856 | 0.9073 | +0.013 |
| CV fold 3 | image | 1,252 | 0.9020 | 0.9049 | 0.8996 | +0.003 |
| CV fold 4 | image | 1,253 | 0.9197 | 0.9225 | 0.9206 | +0.003 |
| **CV pooled OOF** | image | 6,700 | **0.7610** | **0.8067** | 0.8969 | **+0.046** |
| **Locked** | image | 661 | **0.5232** | **0.5165** | 0.4732 | −0.007 |
| **Locked** | eye (max) | 230 | **0.8012** | **0.8225** | 0.8127 | **+0.021** |
| Dev | image | 663 | 0.4049 | 0.4068 | 0.3952 | +0.002 |
| Dev | eye (max) | 230 | 0.7206 | 0.7548 | 0.7793 | +0.034 |

**The fold-0 anomaly is largely an artifact of the definition.** Fold 0 holds 476 AP-ROP
frames, 470 of them from a single infant; scoring AP-ROP as ordinal rank 5 converts that
infant's confusions into the largest possible ordinal penalties. Under the declared 5-level
scale, fold 0 stops being the outlier. The *macro-F1* and *calibration* signatures of fold 0
are unaffected and remain real.

## What must change upstream

1. Implement `qwk_classes` in `models/common/metrics.py::compute_metrics` (or delete the key).
2. Re-issue every QWK in `docs/`, the deck, `results/rop/staging_runs/*.json`,
   `results/rop/locked_eval/locked_table.json` and `results/rop/matched_equivalence.json`.
3. Never quote a QWK again without naming the class set. This document is the correction
   record; the old values are preserved in the table above and in the CSV.

**Classification: evaluation fix. No retraining is implied — the predictions are unchanged.**
