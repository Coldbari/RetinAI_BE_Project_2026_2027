# Served checkpoint provenance

| Field | Value | How established |
|---|---|---|
| Served file | `results/rop_staging/weights.pth` | `webapp/inference.py` loads this path literally |
| SHA-256 | `11b439608690024700cdc927dece45d51ebc78b648aef81090cb6be3e7dc747c` | recomputed in pass 2 |
| Size / mtime | 82,964,554 B · 2026-08-26T21:12:53 | filesystem |
| Identity | **fold-0 checkpoint of the structured, seed-42, natural-sampling 5-fold run** | empirically: the served weights were re-run over all 663 dev images and agreed with the committed fold-0 predictions at **1.000**, max \|Δprob\| 3e-06; folds 1–4 agreed at 0.872–0.888 (`20_AUDIT/served_checkpoint_identification.json`) |
| Config | `configs/rop_staging_structured.yaml` | loaded by the app; `n_sites` read from the checkpoint |
| Architecture | EfficientNetV2-S + CORN stage head + AP-ROP branch + GRL site head | strict `load_state_dict` succeeds |
| Screening threshold | **0.0155**, read at runtime from `results/rop/screening_rebase.json` | `webapp/inference.py`; never hardcoded |
| AP-ROP threshold | 0.5, fixed a priori | `configs/rop_staging_structured.yaml` |
| TTA | none (1 forward pass) | code inspection + timing |

## Two provenance defects

**1. The fold checkpoints no longer exist on this machine.** `scripts/rop_locked_eval.py`
reads 20 checkpoints from `kaggle_cache/<run>/code/results/experiment_00k_…/weights.pth`.
`find kaggle_cache -name "*.pth"` returns nothing — only logs, code and JSON survive. The
locked evaluation **cannot be re-run**; its `.npz` outputs are the only record. The served
file's identity therefore rests on the prediction-agreement test above, which is strong, but
the checkpoint set itself is unrecoverable. Any future re-run is a different experiment.

**2. The served checkpoint saw 4/5 of the internal corpus.** Fold-0's checkpoint trained on
folds 1–4. Consequently:

- The pooled out-of-fold calibration figures (accuracy 0.786, ECE 0.109) describe an
  *ensemble of five models on held-out data*, **not this checkpoint**.
- The internal operating point quoted by the product (sensitivity 0.958 / specificity 0.781)
  was derived from those pooled out-of-fold scores, which this checkpoint does not produce.
- The external half of the threshold selection (dev) was also used to *choose* 0.0155.

The only unbiased estimates for the served checkpoint are those computed on the **locked
half**, which it never saw — see
[`07_TEST_RESULTS/LOCKED_SCREENING_EVALUATION.md`](../07_TEST_RESULTS/LOCKED_SCREENING_EVALUATION.md).

**Required:** write `results/rop_staging/WEIGHTS.json` carrying fold, seed, run tag, config
hash and SHA-256, and assert the hash at application boot, so an anonymous checkpoint can
never again be the production model.
