# Preprocessing Consistency — training vs evaluation vs serving

**Verdict: consistent. No train/serve skew in this model.**

| context | code path | transform |
|---|---|---|
| Training (train split) | `build_transforms(cfg, "train")` | preprocess + RandomResizedCrop + augmentation + normalise |
| Training (val split, the numbers in `06_VALIDATION/`) | `build_transforms(cfg, "val")` | preprocess + Resize(384) + normalise |
| External dev / locked evaluation | `build_transforms(cfg, "val")` (`scripts/rop_locked_eval.py`) | identical to val |
| **Served web app** | `build_transforms(cfg, "val")` (`webapp/inference.py`, line ~380) | **identical to val** |
| This package's fresh re-run and Grad-CAM | `build_transforms(cfg, "val")` on the same config | identical |

All five load the **same** `configs/rop_staging_structured.yaml`, so `image_size: 384`, `circle_crop: true`, `letterbox: true`, `clahe: true`, `ben_graham: false` are guaranteed to agree — there is no second copy of these constants anywhere in the serving path.

## How this was verified rather than asserted

The served checkpoint was loaded from `results/rop_staging/weights.pth`, run through
`build_transforms(cfg, "val")` over all **663** external dev images, and compared against the
committed `dev_eval/structured_s42_ckpt0.npz` predictions produced months earlier by a
different script on different hardware:

- prediction agreement: **1.000 (663/663)**
- maximum absolute probability difference: **3.0 × 10⁻⁶** (float32 rounding)

If preprocessing differed anywhere along that chain, this comparison would not close.
Evidence: [`../20_AUDIT/served_checkpoint_identification.json`](../20_AUDIT/served_checkpoint_identification.json).

## The one documented inconsistency

The repository README states that each model runs with **5-view test-time augmentation**. The served structured path (`ROPStructuredModel.probs`) performs **a single forward pass**. Since the committed evaluation artifacts are also single-pass, the numbers in this package correctly describe the served behaviour — but the README is stale and should be corrected.

## Residual risk

`FundusDataset.__getitem__` silently substitutes a **black image with the true label** for any file it cannot read. No such substitution occurred in the runs behind this package (all 5,489 relevant paths were verified present on disk), but the failure is silent by design and would degrade metrics without warning if paths ever broke — for example under Kaggle's path remapping.
