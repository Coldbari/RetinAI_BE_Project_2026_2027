"""S8 — resolve the DR train/serve preprocessing ambiguity, empirically.

The ambiguity: results/dr/config.yaml (written by the training run) records
circle_crop / ben_graham / clahe = FALSE, but configs/dr.yaml — the config the web app
actually serves from — has all three TRUE. Local spot-testing showed the *recorded* config
performing WORSE, which suggests the recording may be stale rather than authoritative.

The test: score the SAME checkpoint on the SAME val split under BOTH preprocessing
configurations. Whichever reproduces the recorded QWK 0.7281 / accuracy 0.7443 is the
configuration the checkpoint was actually trained with. Everything else follows from that.

Also dumps TTA-averaged val probabilities so serving calibration (C3) can be fitted for DR
against the real 5-view pipeline.
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
# Kaggle's image ships torch 2.x+cu128, which dropped the P100's sm_60 arch -> every CUDA
# op fails with "no kernel image is available". 2.5.1+cu121 works on both P100 and T4.
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "torch==2.5.1", "torchvision==0.20.1",
                "--index-url", "https://download.pytorch.org/whl/cu121"], check=True)
print("torch install %.0fs" % (time.time() - t0), flush=True)

CODE = os.path.dirname(glob.glob("/kaggle/input/**/evaluate.py", recursive=True)[0])
DRC = glob.glob("/kaggle/input/**/diabetic-retinopathy-detection", recursive=True)[0]
aptos_csv = glob.glob("/kaggle/input/**/aptos*/train.csv", recursive=True)[0]
aptos_img = glob.glob("/kaggle/input/**/aptos*/train_images", recursive=True)[0]
WEIGHTS = glob.glob("/kaggle/input/**/dr_weights.pth", recursive=True)[0]
print("CODE:", CODE, "| EyePACS:", DRC, "| weights:", WEIGHTS, flush=True)

if not os.path.exists("/tmp/train/train"):
    if not (shutil.which("7z") or shutil.which("7za")):
        os.system("apt-get install -y p7zip-full >/dev/null 2>&1")
    sevenz = shutil.which("7z") or shutil.which("7za") or "7z"
    os.makedirs("/tmp/train", exist_ok=True)
    print("extracting EyePACS (~20 min)...", flush=True)
    os.system(f"{sevenz} x {DRC}/train.zip.001 -o/tmp/train/ -y >/dev/null")
if not os.path.exists("/tmp/trainLabels.csv"):
    os.system(f"unzip -o -q {DRC}/trainLabels.csv.zip -d /tmp/")
print("EyePACS images:", len(glob.glob('/tmp/train/train/*')), flush=True)

WORK = "/kaggle/working/code"
shutil.copytree(CODE, WORK, dirs_exist_ok=True)
os.chdir(WORK)
sys.path.insert(0, WORK)

# point the config at the mounted data, keeping seed/val_split identical so the stratified
# split reproduces exactly the one the checkpoint was validated on
base = yaml.safe_load(open("configs/dr.yaml"))
for s in base["data"]["sources"]:
    if s["name"] == "eyepacs":
        s["images_dir"] = "/tmp/train/train"; s["csv"] = "/tmp/trainLabels.csv"
    elif s["name"] == "aptos":
        s["images_dir"] = aptos_img; s["csv"] = aptos_csv
yaml.safe_dump(base, open("configs/dr.yaml", "w"), sort_keys=False)
subprocess.run([sys.executable, "prepare_data.py", "--config", "configs/dr.yaml"], check=True)

RECORDED = {"qwk": 0.7281019162522491, "accuracy": 0.7442820292347377,
            "macro_f1": 0.5864430519204475, "auc": 0.7884222234403359}

import numpy as np                                            # noqa: E402
import pandas as pd                                           # noqa: E402
import torch                                                  # noqa: E402
from models.common.config import load_config                  # noqa: E402
from models.common.architectures import build_from_cfg        # noqa: E402
from models.common.losses import build_loss                   # noqa: E402
from models.common.metrics import compute_metrics             # noqa: E402
from models.common.train_utils import build_dataloaders, evaluate  # noqa: E402

print("cuda:", torch.cuda.is_available(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "", flush=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
manifest = pd.read_csv("results/prepared/dr/manifest.csv")
print("manifest:", len(manifest), "val:", int((manifest.split == 'val').sum()), flush=True)

VARIANTS = {
    # what the web app serves from
    "served_configs_dr_yaml": {"circle_crop": True, "ben_graham": True, "clahe": True},
    # what the training run RECORDED for this checkpoint
    "recorded_results_dr_config": {"circle_crop": False, "ben_graham": False, "clahe": False},
}

results = {}
for name, pre in VARIANTS.items():
    cfg = load_config("configs/dr.yaml")
    for k, v in pre.items():
        cfg.preprocess[k] = v
    cfg.train["num_workers"] = 2
    loaders = build_dataloaders(cfg, manifest)
    model = build_from_cfg(cfg, pretrained=False).to(device)
    model.load_state_dict(torch.load(WEIGHTS, map_location=device))
    _, decode_fn, _ = build_loss(cfg, loaders["counts"])
    t = time.time()
    yt, yp, pr = evaluate(model, loaders["val"], decode_fn, 5, device, "classification")
    m = compute_metrics(yt, yp, pr, 5, list(cfg.data.class_names))
    dq = abs(m["qwk"] - RECORDED["qwk"])
    results[name] = {"preprocess": pre, "qwk": m["qwk"], "accuracy": m["accuracy"],
                     "macro_f1": m["macro_f1"], "auc": m.get("auc"),
                     "abs_qwk_delta_vs_recorded": dq, "seconds": time.time() - t}
    print(f"\n[{name}] {pre}", flush=True)
    print(f"  QWK {m['qwk']:.4f}  (recorded {RECORDED['qwk']:.4f}, delta {dq:+.4f})", flush=True)
    print(f"  acc {m['accuracy']:.4f}  macroF1 {m['macro_f1']:.4f}  "
          f"auc {m.get('auc', float('nan')):.4f}  [{time.time()-t:.0f}s]", flush=True)
    np.savez(f"/kaggle/working/dr_val_{name}.npz", y_true=yt, y_pred=yp, y_prob=pr)

winner = min(results, key=lambda k: results[k]["abs_qwk_delta_vs_recorded"])
verdict = {
    "recorded_metrics": RECORDED,
    "variants": results,
    "reproduces_recorded_run": winner,
    "delta": results[winner]["abs_qwk_delta_vs_recorded"],
    "conclusive": bool(results[winner]["abs_qwk_delta_vs_recorded"] < 0.01),
}
print("\n" + "=" * 70)
print("VERDICT")
print("=" * 70)
for k, v in results.items():
    print(f"  {k:<32} QWK {v['qwk']:.4f}  |delta| {v['abs_qwk_delta_vs_recorded']:.4f}")
print(f"\n  reproduces the recorded run -> {winner}")
if verdict["conclusive"]:
    print(f"  CONCLUSIVE (delta {verdict['delta']:.4f} < 0.01): the checkpoint was trained")
    print(f"  with preprocessing {results[winner]['preprocess']}")
else:
    print(f"  INCONCLUSIVE: closest delta {verdict['delta']:.4f}. Neither configuration")
    print(f"  reproduces the recorded metrics — checkpoint provenance is unknown.")
json.dump(verdict, open("/kaggle/working/dr_config_verdict.json", "w"), indent=2, default=float)
print("\nDONE", flush=True)
