"""G8 — false-rejection rate of the gradability gate on EyePACS + APTOS.

The gate's thresholds were calibrated on the infant TRAIN split only, because that is the
only training data available locally. Adult evidence so far is 45 HRF images from a single
camera — thin. EyePACS is ~35k images from many cameras and operators, including genuinely
poor captures, so it is the real test of whether the gate rejects real patients.

CPU only: gradability() is classical OpenCV, no model involved. Deliberately does not
request a GPU so it costs no GPU quota.

Reports FRR overall, per source, and per verdict — plus the metric distributions, so a bad
threshold can be diagnosed rather than merely observed.
"""
import glob
import json
import os
import random
import shutil
import subprocess
import sys
import time

import numpy as np

t0 = time.time()
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "opencv-python-headless"],
               check=False)

CODE = os.path.dirname(glob.glob("/kaggle/input/**/evaluate.py", recursive=True)[0])
DRC = glob.glob("/kaggle/input/**/diabetic-retinopathy-detection", recursive=True)[0]
aptos_img = glob.glob("/kaggle/input/**/aptos*/train_images", recursive=True)[0]
print("CODE:", CODE, "| EyePACS:", DRC, "| APTOS:", aptos_img, flush=True)

SAMPLE = 4000          # per source; enough to bound FRR tightly without a 35k decode
if not os.path.exists("/tmp/train/train"):
    if not (shutil.which("7z") or shutil.which("7za")):
        os.system("apt-get install -y p7zip-full >/dev/null 2>&1")
    sevenz = shutil.which("7z") or shutil.which("7za") or "7z"
    os.makedirs("/tmp/train", exist_ok=True)
    print("extracting EyePACS (~20 min)...", flush=True)
    os.system(f"{sevenz} x {DRC}/train.zip.001 -o/tmp/train/ -y >/dev/null")

WORK = "/kaggle/working/code"
shutil.copytree(CODE, WORK, dirs_exist_ok=True)
os.chdir(WORK)
sys.path.insert(0, WORK)

from PIL import Image                                              # noqa: E402
from models.common.data_quality import gradability, load_gate_thresholds  # noqa: E402

thr = load_gate_thresholds("results/gate_thresholds.json")
print("thresholds:", json.dumps(thr), flush=True)

rng = random.Random(42)
sources = {}
ep = sorted(glob.glob("/tmp/train/train/*.jpeg"))
rng.shuffle(ep)
sources["eyepacs"] = ep[:SAMPLE]
ap = sorted(glob.glob(os.path.join(aptos_img, "*")))
rng.shuffle(ap)
sources["aptos"] = ap[:SAMPLE]

report = {"thresholds": thr, "sample_per_source": SAMPLE, "sources": {}}
for name, paths in sources.items():
    if not paths:
        print(f"{name}: no images found"); continue
    verdicts, metrics, rejected = {}, [], []
    t = time.time()
    for i, p in enumerate(paths):
        try:
            r = gradability(Image.open(p).convert("RGB"), thr)
        except Exception:
            continue
        verdicts[r["verdict"]] = verdicts.get(r["verdict"], 0) + 1
        metrics.append(r["metrics"])
        if not r["gradable"]:
            rejected.append((os.path.basename(p), r["verdict"]))
        if (i + 1) % 1000 == 0:
            print(f"  {name} {i+1}/{len(paths)} ({time.time()-t:.0f}s)", flush=True)
    n = sum(verdicts.values())
    frr = 1.0 - verdicts.get("gradable", 0) / max(n, 1)
    q = {k: [float(np.percentile([m[k] for m in metrics], p)) for p in (0.5, 5, 50, 95)]
         for k in ("redness", "saturation", "red_hue_frac", "blur_var", "luminance")}
    report["sources"][name] = {"n": n, "false_rejection_rate": frr,
                               "verdicts": verdicts, "percentiles_p0.5_p5_p50_p95": q,
                               "examples_rejected": rejected[:15]}
    print(f"\n{name}: n={n}  FRR={frr:.3%}  verdicts={verdicts}", flush=True)
    for k, v in q.items():
        print(f"    {k:<15} p0.5={v[0]:.3f}  p5={v[1]:.3f}  p50={v[2]:.3f}  p95={v[3]:.3f}",
              flush=True)

tot_n = sum(s["n"] for s in report["sources"].values())
tot_rej = sum(s["n"] * s["false_rejection_rate"] for s in report["sources"].values())
overall = tot_rej / max(tot_n, 1)
report["overall"] = {"n": tot_n, "false_rejection_rate": overall, "budget": 0.02,
                     "pass": bool(overall <= 0.02)}
print("\n" + "=" * 70)
print(f"OVERALL adult FRR: {overall:.3%} on {tot_n} images   budget 2%   "
      f"{'PASS' if overall <= 0.02 else 'FAIL'}")
print("=" * 70)
json.dump(report, open("/kaggle/working/gate_frr_adult.json", "w"), indent=2, default=float)
print("DONE", flush=True)
