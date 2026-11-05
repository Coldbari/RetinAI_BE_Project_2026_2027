"""C2/C3 (DR) — serving-pipeline calibration for the last `unverified` model.

DR is the only disease still reporting calibration status `unverified` in the app. The S8
kernel dumped SINGLE-VIEW predictions (it used evaluate(), not predict_with_tta), so it
could not answer this: serving averages 5 TTA views, and a temperature fitted on single-view
logits does not apply to an averaged probability vector.

Same method as ROP and Glaucoma: fit temperature on log(TTA-averaged probs), which is
monotone and therefore cannot change any prediction — only the confidence.

Also reports the referable (grade>=2) endpoint's calibration separately, since that is the
clinically meaningful decision and the one the 0.906 AUC measures.
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
print("torch %.0fs" % (time.time() - t0), flush=True)

CODE = os.path.dirname(glob.glob("/kaggle/input/**/evaluate.py", recursive=True)[0])
DRC = glob.glob("/kaggle/input/**/diabetic-retinopathy-detection", recursive=True)[0]
aptos_csv = glob.glob("/kaggle/input/**/aptos*/train.csv", recursive=True)[0]
aptos_img = glob.glob("/kaggle/input/**/aptos*/train_images", recursive=True)[0]
cands = glob.glob("/kaggle/input/**/dr_weights.pth", recursive=True)
assert cands, "dr_weights.pth not found — refusing to glob for some other .pth"
WEIGHTS = cands[0]
print("CODE:", CODE, "| weights:", WEIGHTS, flush=True)

if not os.path.exists("/tmp/train/train"):
    if not (shutil.which("7z") or shutil.which("7za")):
        os.system("apt-get install -y p7zip-full >/dev/null 2>&1")
    sevenz = shutil.which("7z") or shutil.which("7za") or "7z"
    os.makedirs("/tmp/train", exist_ok=True)
    print("extracting EyePACS (~20 min)...", flush=True)
    os.system(f"{sevenz} x {DRC}/train.zip.001 -o/tmp/train/ -y >/dev/null")
if not os.path.exists("/tmp/trainLabels.csv"):
    os.system(f"unzip -o -q {DRC}/trainLabels.csv.zip -d /tmp/")

WORK = "/kaggle/working/code"
shutil.copytree(CODE, WORK, dirs_exist_ok=True)
os.chdir(WORK)
sys.path.insert(0, WORK)

cfg_path = "configs/dr.yaml"
d = yaml.safe_load(open(cfg_path))
for s in d["data"]["sources"]:
    if s["name"] == "eyepacs":
        s["images_dir"] = "/tmp/train/train"; s["csv"] = "/tmp/trainLabels.csv"
    elif s["name"] == "aptos":
        s["images_dir"] = aptos_img; s["csv"] = aptos_csv
yaml.safe_dump(d, open(cfg_path, "w"), sort_keys=False)
subprocess.run([sys.executable, "prepare_data.py", "--config", cfg_path], check=True)

import numpy as np                                            # noqa: E402
import pandas as pd                                           # noqa: E402
import torch                                                  # noqa: E402
from PIL import Image                                         # noqa: E402
from sklearn.metrics import roc_auc_score                     # noqa: E402
from models.common.config import load_config                  # noqa: E402
from models.common.architectures import build_from_cfg        # noqa: E402
from models.common.preprocessing import build_tta_transforms  # noqa: E402
from models.common.train_utils import predict_with_tta        # noqa: E402
from models.evaluation.calibration import (                   # noqa: E402
    brier_score, expected_calibration_error, fit_temperature, softmax_np)

cfg = load_config(cfg_path)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = build_from_cfg(cfg, pretrained=False).to(device)
model.load_state_dict(torch.load(WEIGHTS, map_location=device))
model.eval()
tta = build_tta_transforms(cfg)

man = pd.read_csv(cfg.data.manifest)
val = man[man.split == "val"].reset_index(drop=True)
print(f"val: {len(val)} images x 5 TTA views", flush=True)

P, Y = [], []
t = time.time()
for i, r in enumerate(val.itertuples()):
    P.append(predict_with_tta(model, Image.open(r.image_path).convert("RGB"),
                              tta, device, "classification").numpy())
    Y.append(int(r.label))
    if (i + 1) % 500 == 0:
        print(f"  {i+1}/{len(val)} ({time.time()-t:.0f}s)", flush=True)
P = np.asarray(P, dtype=np.float64); Y = np.asarray(Y)


def decide(p, y, ncls, name):
    z = np.log(np.clip(p, 1e-12, 1.0))
    e0, m0 = expected_calibration_error(p, y)
    b0 = brier_score(p, y, ncls)
    T = fit_temperature(z, y)
    pc = softmax_np(z / T)
    e1, m1 = expected_calibration_error(pc, y)
    b1 = brier_score(pc, y, ncls)
    changed = int((p.argmax(1) != pc.argmax(1)).sum())
    rel = (e0 - e1) / e0 if e0 > 0 else 0.0
    absg = e0 - e1
    # relative AND absolute AND non-degrading MCE -- a relative threshold alone fires on
    # noise (glaucoma showed "26.1% better" that was 0.0044 absolute while MCE worsened)
    ap = bool(rel >= 0.10 and absg >= 0.01 and m1 <= m0)
    reason = (f"ECE improved {rel:.1%} ({e0:.4f} -> {e1:.4f}, absolute {absg:.4f}) "
              f"without degrading MCE" if ap else
              f"NOT applied: rel {rel:+.1%}, absolute {absg:+.4f} ({e0:.4f} -> {e1:.4f}), "
              f"MCE {m0:.4f} -> {m1:.4f}")
    print(f"\n{name}:  T={T:.4f}")
    print(f"  ECE   {e0:.4f} -> {e1:.4f}   MCE {m0:.4f} -> {m1:.4f}   "
          f"Brier {b0:.4f} -> {b1:.4f}")
    print(f"  argmax changed {changed}/{len(y)}")
    print(f"  DECISION: {'APPLY' if ap else 'DO NOT APPLY'} — {reason}")
    return {"temperature": float(T), "apply": ap, "decision_reason": reason,
            "uncalibrated": {"ece": e0, "mce": m0, "brier": b0},
            "calibrated": {"ece": e1, "mce": m1, "brier": b1},
            "argmax_changed": changed, "n": int(len(y))}

print("\n" + "=" * 70)
five = decide(P, Y, 5, "DR 5-class")

ref = int(cfg.eval.get("referable_grade", 2))
pr2 = np.column_stack([P[:, :ref].sum(1), P[:, ref:].sum(1)])
y2 = (Y >= ref).astype(int)
binr = decide(pr2, y2, 2, f"DR referable (grade>={ref})")
binr["referable_auc"] = float(roc_auc_score(y2, pr2[:, 1]))
print(f"  referable AUC (TTA serving path) {binr['referable_auc']:.4f}")

out = {"disease": "dr", "method": "temperature on log(TTA-averaged probs)",
       "split": "val", "n_classes": 5, **five, "referable": binr,
       "note": "Fitted against predict_with_tta output (5-view probability average). "
               "NOT interchangeable with results/dr/calibration/, fitted on single-view logits."}
json.dump(out, open("/kaggle/working/calibration_serving.json", "w"), indent=2, default=float)
np.savez("/kaggle/working/dr_tta_val_preds.npz", prob=P, y_true=Y)
print("\nDONE", flush=True)
