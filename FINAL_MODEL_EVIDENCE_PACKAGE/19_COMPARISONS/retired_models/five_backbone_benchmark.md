STATUS:
RETIRED / HISTORICAL / PROVISIONAL

MODEL:
Five flat-softmax backbones, fold 0 + leave-site fold only, seed 42, 20-30 epochs.
This sweep SELECTED EfficientNetV2-S as the backbone of the current model.

DATE:
2026-08-07 to 2026-08-09.

NOT CURRENT MODEL RESULT
================================================================================

| backbone | params | macro-F1 (fold 0) | cross-site macro-F1 | AP-ROP recall |
|---|---|---|---|---|
| **EfficientNetV2-S** | 20.2M | **0.520** | **0.641** | 0.183 |
| DeiT3-S | 21.8M | 0.517 | 0.507 | 0.174 |
| ConvNeXt-T (in12k) | 27.8M | 0.509 | 0.530 | 0.162 |
| CAFormer-S18 | 24.3M | 0.498 | 0.507 | 0.221 |
| EfficientNet-B0 | 4.0M | 0.492 | 0.536 | (pre-metric-fix) |

# How to read this table now

**The in-fold column should not be used to rank the backbones.** Its total spread is
**0.028**, while the SD of a *single* run's validation macro-F1 over its **last five epochs**
is 0.007–0.056 (measured for the current model in
[`../../05_TRAINING/last_epoch_noise.csv`](../../05_TRAINING/last_epoch_noise.csv)). The entire
between-architecture spread is inside the end-of-schedule noise of one training run.

This *supports* the project's conclusion that "architecture is not the constraint" far more
strongly than the argument currently made for it — but it also means the in-fold ranking is
meaningless.

**The cross-site column is a different matter.** EfficientNetV2-S at 0.641 against 0.507–0.536
for the other four is an 11-point gap, well outside that noise. The backbone selection is
therefore defensible — but it rests on **one fold and one seed**, and no multi-seed
replication exists.

# Other caveats

- QWK values from this sweep use the **legacy 6-class definition** and have **not** been
  recomputed, because raw `.npz` predictions were not saved for every arm.
- The AP-ROP recall column is a fold-0 artifact: fold 0 holds 476 of 618 AP-ROP validation
  frames, 470 from one infant. On folds 1–4 flat softmax reaches AP-ROP recall 0.83–0.97.
- All figures under `graphs/rop_staging/` from this sweep are stamped PROVISIONAL upstream.

Source: `results/rop/staging_runs/staging_*_s42.json` and the kernel logs beside them.
