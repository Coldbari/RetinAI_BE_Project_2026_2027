# Reproducibility Report

*Can another researcher reproduce the current results? Split into (A) reproducing the
**numbers** in this package from the committed artifacts, and (B) reproducing the **model**
by retraining.*

---

## A. Reproducing this package's numbers — ✅ YES, fully

| requirement | status |
|---|---|
| Raw per-sample predictions committed | ✅ `structured_5fold/*.npz`, `locked_eval/*.npz`, `dev_eval/*.npz` (y_true, y_pred, prob6) |
| Split files committed | ✅ `results/prepared/rop_staging/fold_*.csv`, `test_dev.csv`, `test_locked.csv`, repo-relative and PHI-free |
| Metric implementation committed | ✅ `20_AUDIT/scripts/common.py`, written from scratch against scikit-learn primitives |
| Build scripts committed | ✅ `20_AUDIT/scripts/s01…s11` |
| Deterministic | ✅ every bootstrap uses `default_rng(42)`; every sampling step is seeded |
| Integrity checks | ✅ label order asserted against the manifests; view-table join asserted complete |
| Environment recorded | ✅ torch 2.12.0, python 3.14.6, numpy 2.4.6, sklearn 1.8.0, macOS-27.0-arm64 |
| Git commit recorded | ✅ `f0d4508` |

**Command to reproduce everything:**

```bash
cd /Users/praharsh/Project/RetinAI
for s in 01 02 03 04 05 06 07 08 09 10 11; do
  python3 FINAL_MODEL_EVIDENCE_PACKAGE/20_AUDIT/scripts/s${s}_*.py
done
```

`s01` and `s08` require the image files (`data/rop/`) and the served checkpoint; the rest need
only the committed `.npz` and `.csv` artifacts.

**Independent verification achieved:** three previously published quantities were reproduced
*exactly* from raw predictions (CV macro-F1 for all four arms, the paired seed-flip, the
leakage audit), and two were reproduced by a *second, independent* route (final validation
macro-F1 from the training log vs from the `.npz`; Grad-CAM statistics from a fresh run vs the
project's own `cam_leakage.json`).

---

## B. Reproducing the model by retraining — ⚠️ PARTIALLY, with four known breaks

| requirement | status | detail |
|---|---|---|
| Configuration | ❌ **BROKEN** | `configs/rop_staging_structured.yaml` sets `weighted_sampler: true`. Every headline run overrode it to `false` from the Kaggle kernel. Running the documented command reproduces the **retired v1 failure arm** (QWK 0.121), not this model. |
| Checkpoint identity | ✅ **FIXED IN THIS PACKAGE** | the served file is the fold-0, seed-42 checkpoint; sha256 `11b4396086…`. Previously unrecorded anywhere. |
| Run provenance in the artifact | ⚠️ partial | the seed-42 5-fold summary JSON records model, arch, seed, epochs and fold metrics — but **not** `weighted_sampler` or `w_site`. The seed-1337 summary does. |
| Dataset version | ✅ | the fold CSVs **are** the experiment and are committed |
| Data files | ⚠️ | images are not in the repository (correctly — they are PHI-adjacent and 15 GB). Provenance is traced in `data/rop/PROVENANCE.json`; the Multi-View test set is **CC BY-NC** |
| Random seed | ⚠️ **INCOMPLETE** | `set_seed` sets `random`, `numpy`, `torch`, `torch.cuda` — but **not** `torch.backends.cudnn.deterministic/benchmark`, `PYTHONHASHSEED`, or a DataLoader `generator=`. Runs are approximately, not bit-, reproducible. This matters because **seed noise is itself a headline finding** |
| Dependencies | ❌ | `requirements.txt` is entirely `>=`; no per-run freeze beyond the torch version string in `meta.json` |
| Hardware | ⚠️ | Kaggle P100/T4; the kernel self-detects sm_60 and installs a cu121 build. Reproducible in principle, but Kaggle images drift |
| Training logs | ✅ | 120 epoch records for this model in `kaggle_cache/struct42/` |
| Validation loss / LR traces | ❌ | never logged for any run |
| Experiment logger | ✅ | writes `config.yaml`, `meta.json` (git hash, seed, torch version, platform, GPU) per run — but those directories live on Kaggle, not in the repo |

---

## What a third party would have to be given

1. The four source datasets, obtained from their original registries (never Kaggle mirrors).
2. The committed `results/prepared/rop_staging/` split CSVs.
3. `configs/rop_staging_structured.yaml` **with `weighted_sampler: false`**.
4. `train_structured.py` and `models/common/` at commit `f0d4508`.
5. The corrected metric implementation (`20_AUDIT/scripts/common.py`) — the repository's own `compute_metrics` still computes the legacy 6-class QWK.

## Ranked fixes

| # | fix | cost | effect |
|---|---|---|---|
| 1 | `weighted_sampler: false` in the committed config | 1 line | the repository stops shipping a command that reproduces a known-broken model |
| 2 | Write `results/rop_staging/WEIGHTS.json` (fold, seed, sampler, epochs, sha256, git hash); assert at app boot | ~20 lines | the served artifact becomes self-identifying |
| 3 | Record every override in the run summary JSON | ~5 lines in the kernel | each artifact documents its own configuration |
| 4 | Complete `set_seed` (cuDNN determinism, `PYTHONHASHSEED`, DataLoader generator) | ~6 lines | bit-level reproducibility, which the seed-noise finding deserves |
| 5 | Have the Kaggle summary copy the `history` block it already builds; add validation loss and realised LR | ~10 lines | closes the last genuine evidence gap in `05_TRAINING/` |
| 6 | Pin `requirements.txt`; emit `pip freeze` into each run directory | trivial | environment reproducibility |
| 7 | Implement `eval.qwk_classes` in `compute_metrics`, or delete the key | ~4 lines | future runs report the metric they declare |

## Honest verdict

> **The evidence is fully reproducible; the model is not yet.**
>
> Committing raw per-sample predictions and the split files was the decision that makes this
> package possible — every headline number was recomputed from them, and several previously
> published values were confirmed to four decimal places. That is better practice than most
> published work in this area.
>
> Retraining, by contrast, currently fails at the first step because the reference config
> describes a different (and known-broken) model. Fixes 1–4 are an hour's work and would
> close the gap.
