# Kaggle training workflow (validated)

All model training runs on **Kaggle GPU** (nothing trains on the desktop). These thin
kernel scripts install deps, attach the data + this repo (as a Kaggle dataset), and call
the config-driven CLI. The desktop only does inference, reporting, and pulling weights back.

## One-time setup
1. **Auth** — Kaggle's newer token (`KGAT_...`) goes in `~/.kaggle/access_token` (the old
   `kaggle.json` is no longer issued). Then `pip install -U kaggle` (>=2.2).
   The account is **`praharshnagpure`** — every `<user>` below is that. Guessing the owner
   slug returns `Permission 'kernels.get' was denied`, which reads like an auth failure but
   is really a 404; `kaggle kernels list --mine -v` lists the real slugs.
2. **Upload the code** as a private Kaggle dataset so kernels can import it:
   ```bash
   # bundle code only (no data/venv/weights), then:
   kaggle datasets create -p <code_bundle_dir>     # id: praharshnagpure/retinal-framework-code
   ```
3. **Accept competition rules** (for DR) on the EyePACS and APTOS competition pages.

## Run a model (via the Kaggle API)
Each kernel folder has a `kernel-metadata.json` + a `train_*.py`. Example metadata:
```json
{
  "id": "praharshnagpure/retinal-train-rop", "code_file": "train_rop.py",
  "language": "python", "kernel_type": "script",
  "enable_gpu": true, "enable_internet": true, "is_private": true,
  "dataset_sources": ["praharshnagpure/retinal-framework-code", "praharshnagpure/rop-retinal-dataset"],
  "competition_sources": []
}
```
DR additionally uses `"competition_sources": ["diabetic-retinopathy-detection","aptos2019-blindness-detection"]`.
```bash
kaggle kernels push   -p <kernel_dir>
kaggle kernels status praharshnagpure/retinal-train-rop
kaggle kernels output praharshnagpure/retinal-train-rop -p ./out   # weights.pth + metrics.json + plots
```
Then drop `weights.pth` into `results/<disease>/weights.pth` so `webapp/registry.yaml` serves it.

## Hard-won gotchas (baked into the scripts)
- **Invoke the CLI as `.venv/bin/python -m kaggle`, not `.venv/bin/kaggle`.** The console
  script's shebang points at a venv path that no longer exists, so `.venv/bin/kaggle` dies
  with *bad interpreter* — which looks like a broken install rather than a stale shebang.
- **The kernel bundle ships no `results/`.** Anything writing under `results/` must
  `mkdir(parents=True, exist_ok=True)` first, or the kernel dies *after* the expensive work
  is already done (this cost one full T9 split build).
- **torch/P100:** Kaggle's image ships `torch 2.x+cu128`, which dropped the **P100** GPU's
  `sm_60` arch -> any CUDA op fails with *no kernel image*. The scripts pin
  `torch==2.5.1+cu121` (works on P100 and T4). Also makes runs deterministic.
- **Mount layout:** datasets mount at `/kaggle/input/datasets/<owner>/<slug>/...` and
  competitions at `/kaggle/input/competitions/<slug>/...` — the scripts **discover** paths
  by glob rather than hard-coding.
- **EyePACS** images are split 7-zip archives (`train.zip.001..005`); `train_dr.py`
  extracts them to `/tmp/train/train` (~20 min) and unzips `trainLabels.csv.zip`.
- **SMDG (glaucoma):** label column is `types` (0/1, -1 = drop); images live under a
  **doubled** `full-fundus/full-fundus/<names>.png` — discovered automatically.
- **Concurrency:** Kaggle allows **2 simultaneous GPU sessions** — run at most two at once.
- **Resume-safe:** `train.py` checkpoints every epoch, so the 12-hr session cap never loses
  progress.

The legacy GitHub-clone notebook (`train.ipynb`) is kept as an alternative; the dataset-based
`train_*.py` scripts above are the validated path.
