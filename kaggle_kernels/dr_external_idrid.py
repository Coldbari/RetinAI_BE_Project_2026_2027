"""T13 — DR external validation on IDRiD (never seen in training).

DR's only external evidence is Messidor-2 (referable AUC 0.837). IDRiD is a genuinely
different population (Indian, Nidek AFC-210, Kasturba Medical College) graded 0-4 on the
same ICDR scale, so it tests the same claim against different eyes and different optics.

The claim under test is NOT "accuracy holds". It is:
  * does RANKING transfer             -> referable AUC
  * does the OPERATING POINT transfer -> sens/spec under the rule production actually applies

Four traps, all found by adversarial review of the first draft of this script before it ran:

1. **Production does NOT threshold the referable probability mass.** `webapp/inference.py`
   sets `is_referable = bool(argmax >= 2)`. A separate code path, `_decisive_score`, returns
   `probs[2:].sum()` with 0.5 — that feeds the displayed score and the abstention band, not
   the referral flag. The two rules can disagree, so BOTH are reported here and the argmax
   rule is labelled as the deployed one.
2. **IDRiD ships two label CSVs** (Training ~413 rows, Testing ~103) and two image
   directories. Taking the first match evaluates a subset and silently calls it the whole set.
3. **The two image directories reuse filenames** (`IDRiD_001.jpg` exists in both). A flat
   stem-keyed index therefore matches training labels to testing images. The index here is
   keyed by (subset, stem) and each label row is resolved only within its own subset.
4. **Ben Graham runs at native resolution** (`circle_crop -> clahe -> ben_graham(sigma=10)`
   then resize to 384). IDRiD is 4288x2848; APTOS is ~3000px; EyePACS varies. A fixed pixel
   blur radius therefore means IDRiD images are processed differently from anything the model
   trained on. That is a real confound for this experiment, so the model is scored at native
   resolution AND at a size-matched downscale, and both are reported.
"""
import glob
import json
import os
import re
import shutil
import subprocess
import sys
import time

t0 = time.time()
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "torch==2.5.1", "torchvision==0.20.1",
                "--index-url", "https://download.pytorch.org/whl/cu121"], check=True)
print("torch %.0fs" % (time.time() - t0), flush=True)


def one(pattern, what):
    hits = sorted(glob.glob(pattern, recursive=True))
    assert hits, f"{what} not found for pattern {pattern}"
    return hits[0]


CODE = os.path.dirname(one("/kaggle/input/**/evaluate.py", "framework code"))
WEIGHTS = one("/kaggle/input/**/dr_weights.pth", "dr_weights.pth")
print("CODE:", CODE, "| weights:", WEIGHTS, flush=True)

import pandas as pd  # noqa: E402

# ── IDRiD label CSVs: require IDRiD in the path, take ALL of them ─────────────
label_csvs = [p for p in glob.glob("/kaggle/input/**/*.csv", recursive=True)
              if "idrid" in p.lower() and "framework-code" not in p
              and re.search(r"grad", p, re.I)]
if not label_csvs:
    label_csvs = [p for p in glob.glob("/kaggle/input/**/*.csv", recursive=True)
                  if "idrid" in p.lower() and "framework-code" not in p]
assert label_csvs, "no IDRiD grading CSV found — refusing to score an unlabelled set"
print("\nIDRiD label CSVs:")
for p in label_csvs:
    print("   ", p)


def subset_of(path):
    """Training vs Testing — the two halves reuse filenames, so this must be tracked."""
    s = path.lower()
    return "test" if ("test" in s and "train" not in s.split("test")[-1]) else "train"


frames = []
for p in label_csvs:
    df = pd.read_csv(p)
    cols = {c.lower().strip(): c for c in df.columns}
    gc = next((cols[k] for k in cols
               if "retinopathy" in k or k in ("grade", "dr_grade", "diagnosis", "level")), None)
    ic = next((cols[k] for k in cols if "image" in k or k in ("id", "name", "id_code")), None)
    if not (gc and ic):
        print(f"  skipping {os.path.basename(p)} — no grade/id column ({list(df.columns)[:4]})")
        continue
    df = df[[ic, gc]].dropna()
    df.columns = ["id", "grade"]
    df["grade"] = df["grade"].astype(int)
    if not df.grade.between(0, 4).all():
        print(f"  skipping {os.path.basename(p)} — grades outside 0-4")
        continue
    df["subset"] = subset_of(p)
    frames.append(df)
    print(f"  {os.path.basename(p)}: {len(df)} rows, subset={df.subset.iloc[0]}, "
          f"grades={df.grade.value_counts().sort_index().to_dict()}")

assert frames, "no usable IDRiD grading CSV"
labels = pd.concat(frames, ignore_index=True)
print(f"\ntotal labelled rows: {len(labels)}  "
      f"({labels.subset.value_counts().to_dict()})")

# ── image index keyed by (subset, stem) — never a bare stem ──────────────────
img_files = [p for p in glob.glob("/kaggle/input/**/*", recursive=True)
             if p.lower().endswith((".jpg", ".jpeg", ".png"))
             and "idrid" in p.lower() and "framework-code" not in p]
index = {}
collisions = 0
for p in img_files:
    stem = os.path.splitext(os.path.basename(p))[0]
    key = (subset_of(os.path.dirname(p)), stem)
    if key in index:
        collisions += 1
        continue
    index[key] = p
print(f"indexed {len(index)} images from {len(img_files)} files "
      f"({collisions} same-subset duplicates ignored)")
by_sub = {}
for (sub, _), _p in index.items():
    by_sub[sub] = by_sub.get(sub, 0) + 1
print("  per subset:", by_sub)

WORK = "/kaggle/working/code"
shutil.copytree(CODE, WORK, dirs_exist_ok=True)
os.chdir(WORK)
sys.path.insert(0, WORK)

import numpy as np                                            # noqa: E402
import torch                                                  # noqa: E402
from PIL import Image                                         # noqa: E402
from sklearn.metrics import roc_auc_score, cohen_kappa_score  # noqa: E402
from models.common.config import load_config                  # noqa: E402
from models.common.architectures import build_from_cfg        # noqa: E402
from models.common.preprocessing import build_tta_transforms  # noqa: E402
from models.common.train_utils import predict_with_tta        # noqa: E402

cfg = load_config("configs/dr.yaml")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = build_from_cfg(cfg, pretrained=False).to(device)
model.load_state_dict(torch.load(WEIGHTS, map_location=device))
model.eval()
tta = build_tta_transforms(cfg)

rows = []
for _, r in labels.iterrows():
    stem = os.path.splitext(str(r["id"]).strip())[0]
    p = index.get((r["subset"], stem))
    if p:
        rows.append((p, int(r["grade"]), r["subset"]))
print(f"\nresolved {len(rows)}/{len(labels)} labelled images within their own subset")
assert len(rows) >= 100, "too few resolved — id column probably does not match filenames"
print("sample resolution size:", Image.open(rows[0][0]).size)

MATCH_PX = 1024   # APTOS-like scale, so ben_graham's fixed sigma is comparable to training


def run(rescale):
    P, Y = [], []
    t = time.time()
    for i, (p, g, _s) in enumerate(rows):
        im = Image.open(p).convert("RGB")
        if rescale and max(im.size) > rescale:
            im.thumbnail((rescale, rescale))
        P.append(predict_with_tta(model, im, tta, device, "classification").numpy())
        Y.append(g)
        if (i + 1) % 100 == 0:
            print(f"    {i+1}/{len(rows)} ({time.time()-t:.0f}s)", flush=True)
    return np.asarray(P, dtype=np.float64), np.asarray(Y)


REF = int(cfg.eval.get("referable_grade", 2))


def evaluate(P, Y, tag):
    s = P[:, REF:].sum(1)
    y2 = (Y >= REF).astype(int)
    auc = float(roc_auc_score(y2, s))
    res = {"n": int(len(Y)), "referable_auc": auc, "prevalence": float(y2.mean()),
           "qwk": float(cohen_kappa_score(Y, P.argmax(1), weights="quadratic"))}
    print(f"\n--- {tag}")
    print(f"  n={len(Y)}  referable prevalence {y2.mean():.3f}")
    print(f"  referable AUC {auc:.4f}   5-class QWK {res['qwk']:.4f}")
    for name, pred in (("DEPLOYED rule: argmax >= 2", P.argmax(1) >= REF),
                       ("secondary: sum(P[2:]) >= 0.5", s >= 0.5)):
        tp = int((pred & (y2 == 1)).sum()); fn = int((~pred & (y2 == 1)).sum())
        tn = int((~pred & (y2 == 0)).sum()); fp = int((pred & (y2 == 0)).sum())
        sens = tp / max(tp + fn, 1); spec = tn / max(tn + fp, 1)
        key = "deployed_argmax" if "argmax" in name else "sum_ge_0.5"
        res[key] = {"sensitivity": sens, "specificity": spec,
                    "tp": tp, "fn": fn, "tn": tn, "fp": fp}
        print(f"  {name:<30} sens {sens:.3f}  spec {spec:.3f}  "
              f"(TP {tp} FN {fn} TN {tn} FP {fp})")
    return res


print("\n" + "=" * 78)
print("T13 — DR ON IDRiD (EXTERNAL, NEVER TRAINED ON)")
print("=" * 78)
P_nat, Y = run(None)
out = {"dataset": "IDRiD", "csvs": label_csvs, "referable_grade": REF,
       "native": evaluate(P_nat, Y, "native resolution (what a user would upload)")}
P_scl, _ = run(MATCH_PX)
out["rescaled"] = evaluate(P_scl, Y, f"downscaled to <={MATCH_PX}px (ben_graham scale-matched)")
out["rescale_px"] = MATCH_PX

print("\n" + "-" * 78)
print("REFERENCE POINTS")
print("  internal val  referable AUC 0.9124")
print("  Messidor-2    referable AUC 0.837, specificity collapsed 0.89 -> 0.30")
print("\nIf IDRiD also shows AUC holding while specificity drops, 'ranking transfers but")
print("the operating point does not' has two independent confirmations and becomes a")
print("finding rather than an anecdote.")
print("\nThe native/rescaled gap is itself a result: ben_graham uses a fixed pixel sigma")
print("at native resolution, so it measures how much this model's output depends on the")
print("size of the file a clinician happens to upload.")

json.dump(out, open("/kaggle/working/dr_external_idrid.json", "w"), indent=2, default=float)
np.savez("/kaggle/working/idrid_preds.npz", prob_native=P_nat, prob_rescaled=P_scl, y_true=Y)
print("\nDONE", flush=True)
