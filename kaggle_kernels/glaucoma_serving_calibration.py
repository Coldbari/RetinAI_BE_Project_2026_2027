"""C2/C3 (Glaucoma) — serving-pipeline calibration + per-source contamination check (S11).

Glaucoma currently ships as calibration `unverified`: no val predictions exist at all, so
nothing could be fitted. This scores the deployed multi-source checkpoint on its val split
through the EXACT serving path (5-view TTA probability averaging) and fits temperature on
log(TTA-averaged probs) — the same method used for ROP, which cannot change a prediction.

Also emits per-source (SMDG vs G1020) metrics, which is the S11 contamination question:
G1020 is IN-distribution for this checkpoint because it was pooled into training, and any
honest reporting has to label it as such.
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
print("torch install %.0fs" % (time.time() - t0), flush=True)

CODE = os.path.dirname(glob.glob("/kaggle/input/**/evaluate.py", recursive=True)[0])
smdg_csv = glob.glob("/kaggle/input/**/metadata*.csv", recursive=True)[0]
smdg_img = glob.glob("/kaggle/input/**/full-fundus/full-fundus", recursive=True)[0]
g1020_csv = glob.glob("/kaggle/input/**/G1020/G1020.csv", recursive=True)[0]
g1020_img = glob.glob("/kaggle/input/**/G1020/Images", recursive=True)[0]
# Load the DEPLOYED multi-source checkpoint by EXACT name. A previous run globbed for
# "*.pth" and silently picked up weights.pth from a dataset version uploaded 73 minutes
# BEFORE the multi-source training finished -- i.e. the SMDG-only model. Its G1020 AUC came
# back 0.566, matching the documented zero-shot failure (0.589), and the calibration fitted
# to it was invalid. Never glob for a checkpoint.
cands = glob.glob("/kaggle/input/**/weights_multisource.pth", recursive=True)
assert cands, ("weights_multisource.pth not found. Refusing to fall back to any other .pth "
               "-- loading the wrong glaucoma checkpoint silently produces a plausible but "
               "meaningless calibration.")
WEIGHTS = cands[0]
print("CODE:", CODE, "\nSMDG:", smdg_img, "\nG1020:", g1020_img, "\nweights:", WEIGHTS, flush=True)

WORK = "/kaggle/working/code"
shutil.copytree(CODE, WORK, dirs_exist_ok=True)
os.chdir(WORK)
sys.path.insert(0, WORK)

cfg_path = "configs/glaucoma_multisource.yaml"
d = yaml.safe_load(open(cfg_path))
for s in d["data"]["sources"]:
    if s["name"] == "smdg":
        s["images_dir"] = smdg_img; s["csv"] = smdg_csv
    elif s["name"] == "g1020":
        s["images_dir"] = g1020_img; s["csv"] = g1020_csv
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
print("cuda:", torch.cuda.is_available(), flush=True)

model = build_from_cfg(cfg, pretrained=False).to(device)
model.load_state_dict(torch.load(WEIGHTS, map_location=device))
model.eval()
tta = build_tta_transforms(cfg)

man = pd.read_csv(cfg.data.manifest)
val = man[man.split == "val"].reset_index(drop=True)
print(f"val: {len(val)} images", flush=True)

P, Y, SRC = [], [], []
t = time.time()
for i, r in enumerate(val.itertuples()):
    img = Image.open(r.image_path).convert("RGB")
    P.append(predict_with_tta(model, img, tta, device, "classification").numpy())
    Y.append(int(r.label))
    SRC.append(getattr(r, "source", "unknown"))
    if (i + 1) % 300 == 0:
        print(f"  {i+1}/{len(val)} ({time.time()-t:.0f}s)", flush=True)
P = np.asarray(P, dtype=np.float64); Y = np.asarray(Y); SRC = np.asarray(SRC)

z = np.log(np.clip(P, 1e-12, 1.0))
ece0, mce0 = expected_calibration_error(P, Y)
brier0 = brier_score(P, Y, P.shape[1])
T = fit_temperature(z, Y)
Pc = softmax_np(z / T)
ece1, mce1 = expected_calibration_error(Pc, Y)
brier1 = brier_score(Pc, Y, P.shape[1])
changed = int((P.argmax(1) != Pc.argmax(1)).sum())

rel = (ece0 - ece1) / ece0 if ece0 > 0 else 0.0
apply = bool(rel >= 0.10)
reason = (f"ECE improved {rel:.1%} ({ece0:.4f} -> {ece1:.4f})" if apply else
          f"NOT applied: ECE change {rel:+.1%} ({ece0:.4f} -> {ece1:.4f}) is not material; "
          f"5-view TTA averaging already acts as an ensemble")

print("\n" + "=" * 70)
print(f"GLAUCOMA serving calibration   T={T:.4f}")
print(f"  ECE   {ece0:.4f} -> {ece1:.4f}")
print(f"  MCE   {mce0:.4f} -> {mce1:.4f}")
print(f"  Brier {brier0:.4f} -> {brier1:.4f}")
print(f"  argmax changed {changed}/{len(Y)} (must be 0)")
print(f"  DECISION: {'APPLY' if apply else 'DO NOT APPLY'} — {reason}")

per_source = {}
print("\nPER-SOURCE (S11 contamination view)")
for s in sorted(set(SRC.tolist())):
    m = SRC == s
    if len(set(Y[m].tolist())) < 2:
        per_source[s] = {"n": int(m.sum()), "auc": None, "note": "single-class"}
        print(f"  {s:<10} n={int(m.sum()):<6} single-class"); continue
    a = roc_auc_score(Y[m], P[m][:, 1])
    per_source[s] = {"n": int(m.sum()), "auc": float(a),
                     "in_distribution": True,
                     "note": "pooled into training — NOT a held-out source"}
    print(f"  {s:<10} n={int(m.sum()):<6} AUC {a:.4f}   (in-distribution: trained on)")

out = {"disease": "glaucoma", "method": "temperature on log(TTA-averaged probs)",
       "apply": apply, "decision_reason": reason, "temperature": float(T),
       "n": int(len(Y)), "n_classes": int(P.shape[1]), "split": "val",
       "uncalibrated": {"ece": ece0, "mce": mce0, "brier": brier0},
       "calibrated": {"ece": ece1, "mce": mce1, "brier": brier1},
       "argmax_changed": changed, "threshold_changed": None,
       "per_source": per_source,
       "note": "Fitted against predict_with_tta output (5-view probability average). "
               "Every source here was pooled into training; none is held out."}
json.dump(out, open("/kaggle/working/calibration_serving.json", "w"), indent=2, default=float)
np.savez("/kaggle/working/glaucoma_tta_val_preds.npz", prob=P, y_true=Y, source=SRC)
print("\nDONE", flush=True)
