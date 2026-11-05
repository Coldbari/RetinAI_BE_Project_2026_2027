"""T9 — retrain ROP on a DEVICE-STRATIFIED, duplicate-clean split.

This is an ADDITIVE experiment. It does not replace the deployed checkpoint; it answers a
question the current test split cannot: how well does the model do when device is not
confounded with the label?

The old split (results/rop_device_audit.json):
  * ROP prevalence per device 48.1% / 21.3% / 17.4% — device predicts the label
  * EVERY ROP-positive test image came from D1; D2/D3 test images were all negative
  * so discrimination was measurable in only ONE device, and the pooled AUC 0.9271 was
    partly an easy device-vs-device separation. Device-controlled AUC was 0.8808.

The new split (scripts/build_device_stratified_split.py) guarantees every device carries
BOTH classes in val and test, merges patient IDs that are the same infant, and drops the
19 duplicate groups whose identical pixels carried contradictory labels.

Reports per-device test AUC — now measurable on all three devices — and a device-controlled
mean, which is the honest headline.
"""
import glob
import json
import os
import shutil
import subprocess
import sys
import time

import yaml

t0 = time.time()
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "torch==2.5.1", "torchvision==0.20.1",
                "--index-url", "https://download.pytorch.org/whl/cu121"], check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "imagehash"], check=False)
print("deps %.0fs" % (time.time() - t0), flush=True)

CODE = os.path.dirname(glob.glob("/kaggle/input/**/evaluate.py", recursive=True)[0])
ROP = glob.glob("/kaggle/input/**/dataset_split", recursive=True)
assert ROP, "rop dataset_split not found"
ROP = ROP[0]
print("CODE:", CODE, "| ROP:", ROP, flush=True)

WORK = "/kaggle/working/code"
shutil.copytree(CODE, WORK, dirs_exist_ok=True)
os.chdir(WORK)
sys.path.insert(0, WORK)
# the split builder and the trainer both address images relative to CWD
if not os.path.exists("dataset_split"):
    os.symlink(ROP, "dataset_split")

# rebuild the ORIGINAL manifest first (the builder reads it), then the device-stratified one
cfg_path = "configs/rop.yaml"
subprocess.run([sys.executable, "prepare_data.py", "--config", cfg_path], check=True)
subprocess.run([sys.executable, "scripts/build_device_stratified_split.py"], check=True)

DEV_MANIFEST = "results/prepared/rop/manifest_devsplit.csv"
assert os.path.exists(DEV_MANIFEST), "device-stratified manifest was not written"

d = yaml.safe_load(open(cfg_path))
d["data"]["manifest"] = DEV_MANIFEST
d["train"]["num_workers"] = 2
yaml.safe_dump(d, open(cfg_path, "w"), sort_keys=False)

t = time.time()
subprocess.run([sys.executable, "train.py", "--config", cfg_path, "--tag", "devsplit"],
               check=True)
print(f"training {time.time()-t:.0f}s", flush=True)

# ── evaluate on the new test split, per device ───────────────────────────────
import re                                                       # noqa: E402
import numpy as np                                              # noqa: E402
import pandas as pd                                             # noqa: E402
import torch                                                    # noqa: E402
from pathlib import Path                                        # noqa: E402
from sklearn.metrics import roc_auc_score                       # noqa: E402
from models.common.config import load_config                    # noqa: E402
from models.common.architectures import build_from_cfg          # noqa: E402
from models.common.losses import build_loss                     # noqa: E402
from models.common.metrics import compute_metrics               # noqa: E402
from models.common.train_utils import build_dataloaders, evaluate  # noqa: E402

runs = sorted(glob.glob("results/experiment_*/weights.pth"), key=os.path.getmtime)
assert runs, "no checkpoint produced"
W = runs[-1]
print("checkpoint:", W, flush=True)

cfg = load_config(cfg_path)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
manifest = pd.read_csv(DEV_MANIFEST)
loaders = build_dataloaders(cfg, manifest)
model = build_from_cfg(cfg, pretrained=False).to(device)
model.load_state_dict(torch.load(W, map_location=device))
_, decode_fn, _ = build_loss(cfg, loaders["counts"])
yt, yp, pr = evaluate(model, loaders["test"], decode_fn, 2, device, "classification")
m = compute_metrics(yt, yp, pr, 2, list(cfg.data.class_names))

paths = manifest[manifest.split == "test"]["image_path"].tolist()
assert len(paths) == len(yt), "prediction/manifest misalignment"
dev = [re.search(r"_D(\d+)_S", Path(p).name).group(1) for p in paths]
df = pd.DataFrame({"y": yt, "p": pr[:, 1], "dev": dev})

overall = roc_auc_score(df.y, df.p)
print("\n" + "=" * 70)
print("T9 — ROP ON A DEVICE-STRATIFIED SPLIT")
print("=" * 70)
print(f"test: {len(df)} images   accuracy {m['accuracy']:.4f}   macroF1 {m['macro_f1']:.4f}")
print(f"overall test AUC {overall:.4f}")
print(f"\n{'device':<9}{'images':>8}{'ROP+':>7}{'AUC':>9}")
per_dev = {}
for dv, g in df.groupby("dev"):
    if g.y.nunique() < 2:
        print(f"D{dv:<8}{len(g):>8}{int(g.y.sum()):>7}{'n/a':>9}  SINGLE CLASS")
        per_dev[dv] = None
        continue
    a = roc_auc_score(g.y, g.p)
    per_dev[dv] = float(a)
    print(f"D{dv:<8}{len(g):>8}{int(g.y.sum()):>7}{a:>9.4f}")

measurable = [v for v in per_dev.values() if v is not None]
ctrl = float(np.mean(measurable)) if measurable else None
print(f"\nmeasurable devices: {len(measurable)}/{len(per_dev)}")
if ctrl is not None:
    print(f"device-controlled AUC (mean over devices) {ctrl:.4f}")
    print(f"pooled AUC                                {overall:.4f}   "
          f"({ctrl - overall:+.4f})")
print("\nFor comparison, the OLD split gave pooled 0.9271 with only ONE measurable device "
      "(device-controlled 0.8808).")

json.dump({"split": "device-stratified", "n_test": int(len(df)),
           "accuracy": m["accuracy"], "macro_f1": m["macro_f1"], "qwk": m["qwk"],
           "pooled_auc": float(overall), "per_device_auc": per_dev,
           "measurable_devices": len(measurable), "device_controlled_auc": ctrl,
           "old_split": {"pooled_auc": 0.9271, "device_controlled_auc": 0.8808,
                         "measurable_devices": 1}},
          open("/kaggle/working/rop_devsplit_result.json", "w"), indent=2, default=float)
shutil.copy(W, "/kaggle/working/rop_devsplit_weights.pth")
print("\nDONE", flush=True)
