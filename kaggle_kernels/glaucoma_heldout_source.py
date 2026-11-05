"""T12 — glaucoma with two source collections genuinely held OUT of training.

Glaucoma is the one headline with no held-out evidence at all: 0.973 pooled, and G1020 —
once its external test — was pooled into training and stopped being independent.

The recorded blocker was "no usable REFUGE source found", which assumed a held-out set had
to be sourced externally. T11a showed that is wrong: SMDG-19 is a meta-dataset of 15
constituent collections with dataset-prefixed names, so an entire collection can simply be
withheld.

Two are withheld here rather than one, because a single training run then buys two
INDEPENDENT external numbers:

  REFUGE1   800 images,  80 glaucoma   the canonical benchmark; closes the recorded blocker
  ORIGA     650 images, 168 glaucoma   more positives, so a tighter interval

Held out at the MANIFEST level before any dataloader is built, so no image, no augmentation
and no validation batch can see them. The script asserts that afterwards instead of trusting
it — a leak here would produce a beautiful and completely worthless number.

Also reports the within-source AUC of the SAME architecture on the sources it did train on,
so the external drop is measured against a like-for-like baseline rather than against the
0.967 pooled figure that T11b already showed is inflated.
"""
import glob
import json
import os
import shutil
import subprocess
import sys
import time

import yaml

HOLDOUT = ["REFUGE1", "ORIGA"]

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
smdg_csv = one("/kaggle/input/**/metadata*.csv", "SMDG metadata csv")
smdg_img = one("/kaggle/input/**/full-fundus/full-fundus", "SMDG image dir")
print("CODE:", CODE, "\nSMDG:", smdg_img, flush=True)

WORK = "/kaggle/working/code"
shutil.copytree(CODE, WORK, dirs_exist_ok=True)
os.chdir(WORK)
sys.path.insert(0, WORK)

# SMDG-only config: G1020 is excluded here so the experiment has ONE data source whose
# constituent collections we control, rather than two overlapping ones.
cfg_path = "configs/glaucoma.yaml"
d = yaml.safe_load(open(cfg_path))
for s in d["data"]["sources"]:
    if s["name"] == "smdg":
        s["images_dir"] = smdg_img
        s["csv"] = smdg_csv
d["train"]["num_workers"] = 2
yaml.safe_dump(d, open(cfg_path, "w"), sort_keys=False)
subprocess.run([sys.executable, "prepare_data.py", "--config", cfg_path], check=True)

import numpy as np                                            # noqa: E402
import pandas as pd                                           # noqa: E402
import re                                                     # noqa: E402
from pathlib import Path                                      # noqa: E402

MAN = "results/prepared/glaucoma/manifest.csv"
man = pd.read_csv(MAN)


def src_of(p):
    """Constituent collection from the filename.

    Must stay byte-identical in behaviour to scripts/glaucoma_source_ci.py — if this
    kernel holds out rows under one parsing and the audit evaluates under another, the
    two experiments are silently about different images. The earlier draft used a GREEDY
    `[A-Za-z0-9]*` here against a LAZY `[A-Za-z0-9]*?(?=[_\\d])` there, and had no None
    guard on the fallback.
    """
    stem = Path(p).stem
    if "-" in stem:
        return stem.split("-")[0]
    m = re.match(r"^([A-Za-z][A-Za-z0-9]*?)(?=[_\d])", stem)
    return m.group(1) if m else stem


man["src"] = man.image_path.map(src_of)
counts = man.src.value_counts().to_dict()
print("\nsources in manifest:", counts)

is_out = man.src.isin(HOLDOUT)

# The leak assertion below is only meaningful if the hold-out actually MATCHED something.
# `man.src.isin(HOLDOUT)` selecting zero rows makes every downstream check vacuously true
# and would train on the full dataset while reporting a held-out experiment. Refuse first.
missing = [s for s in HOLDOUT if counts.get(s, 0) == 0]
assert not missing, (
    f"hold-out sources {missing} matched NO manifest rows. Prefixes present: "
    f"{sorted(counts)}. Refusing to run — a hold-out that selects nothing would train on "
    f"everything and still report a zero-shot number.")
EXPECT = {"REFUGE1": 400, "ORIGA": 300}          # generous floors on the known sizes
for s in HOLDOUT:
    assert counts[s] >= EXPECT.get(s, 50), (
        f"{s} matched only {counts[s]} rows, expected >= {EXPECT.get(s, 50)} — "
        f"the prefix parsing has drifted")

man.loc[is_out, "split"] = "heldout"          # not train/val/test -> invisible to dataloaders
man.to_csv(MAN, index=False)

print(f"\nheld out {int(is_out.sum())} images from {HOLDOUT}")
for s in HOLDOUT:
    sub = man[man.src == s]
    print(f"  {s:<10} n={len(sub):<5} glaucoma={int((sub.label == 1).sum()):<5} "
          f"non={int((sub.label == 0).sum())}")
print("\nremaining split sizes:", man[~is_out].split.value_counts().to_dict())

# assert the hold-out is real BEFORE spending two GPU-hours on it
leaked = man[(man.src.isin(HOLDOUT)) & (man.split.isin(["train", "val", "test"]))]
assert leaked.empty, f"LEAK: {len(leaked)} held-out images still in a training split"
assert (man[~is_out].split == "train").sum() > 1000, "training set collapsed"
print("hold-out verified: no REFUGE1/ORIGA image is in train, val or test")

t = time.time()
subprocess.run([sys.executable, "train.py", "--config", cfg_path, "--tag", "heldout"],
               check=True)
print(f"training {time.time()-t:.0f}s", flush=True)

import torch                                                  # noqa: E402
from PIL import Image                                         # noqa: E402
from sklearn.metrics import roc_auc_score                     # noqa: E402
from models.common.config import load_config                  # noqa: E402
from models.common.architectures import build_from_cfg        # noqa: E402
from models.common.preprocessing import build_tta_transforms  # noqa: E402
from models.common.train_utils import predict_with_tta        # noqa: E402

runs = sorted(glob.glob("results/experiment_*/weights.pth"), key=os.path.getmtime)
assert runs, "no checkpoint produced"
W = runs[-1]
print("checkpoint:", W, flush=True)

cfg = load_config(cfg_path)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = build_from_cfg(cfg, pretrained=False).to(device)
model.load_state_dict(torch.load(W, map_location=device))
model.eval()
tta = build_tta_transforms(cfg)


def score(rows, label):
    P, Y = [], []
    t0 = time.time()
    for i, r in enumerate(rows.itertuples()):
        P.append(predict_with_tta(model, Image.open(r.image_path).convert("RGB"),
                                  tta, device, "classification").numpy()[1])
        Y.append(int(r.label))
        if (i + 1) % 300 == 0:
            print(f"    {label} {i+1}/{len(rows)} ({time.time()-t0:.0f}s)", flush=True)
    return np.asarray(P), np.asarray(Y)


print("\n" + "=" * 78)
print("T12 — GLAUCOMA, ZERO-SHOT ON COLLECTIONS NEVER SEEN IN TRAINING")
print("=" * 78)

out = {"held_out": HOLDOUT, "zero_shot": {}, "internal": {}}
for s in HOLDOUT:
    rows = man[man.src == s]
    P, Y = score(rows, s)
    a = float(roc_auc_score(Y, P))
    out["zero_shot"][s] = {"n": int(len(Y)), "positives": int(Y.sum()), "auc": a,
                           "prevalence": float(Y.mean())}
    print(f"\n{s:<10} ZERO-SHOT   n={len(Y)}  glaucoma={int(Y.sum())}  AUC {a:.4f}")

# Like-for-like baseline. The zero-shot numbers above are WITHIN a single collection, so
# comparing them to a POOLED val AUC would compare a within-source figure against a
# cross-source one — exactly the inflation T11b measured (0.967 pooled vs 0.878
# within-source). The baseline must therefore also be per-source, averaged over the
# collections large enough to support an estimate.
val = man[(man.split == "val") & (~man.src.isin(HOLDOUT))]
if len(val) > 50:
    P, Y = score(val, "val")
    val = val.assign(p=P, y=Y)
    pooled = float(roc_auc_score(Y, P))
    per_src, usable = {}, []
    for s, g in val.groupby("src"):
        pos, neg = int((g.y == 1).sum()), int((g.y == 0).sum())
        if pos < 10 or neg < 10 or len(g) < 30:
            per_src[s] = {"n": int(len(g)), "positives": pos, "auc": None,
                          "why": "too few of one class"}
            continue
        a = float(roc_auc_score(g.y, g.p))
        per_src[s] = {"n": int(len(g)), "positives": pos, "auc": a}
        usable.append(a)
    within = float(np.mean(usable)) if usable else None
    out["internal"] = {"n": int(len(Y)), "pooled_val_auc": pooled,
                       "within_source_auc": within, "n_usable_sources": len(usable),
                       "per_source": per_src}
    print(f"\n{'in-dist':<12} pooled val AUC        {pooled:.4f}  (n={len(Y)})")
    for s, v in sorted(per_src.items()):
        if v["auc"] is not None:
            print(f"             {s:<12} n={v['n']:<5} AUC {v['auc']:.4f}")
    if within is not None:
        print(f"{'in-dist':<12} WITHIN-SOURCE mean    {within:.4f}  "
              f"({len(usable)} usable collections)  <- the like-for-like baseline")
        print("\nZERO-SHOT vs LIKE-FOR-LIKE (both are within-a-single-collection figures):")
        for s in HOLDOUT:
            z = out["zero_shot"][s]["auc"]
            print(f"   {s:<10} {z:.4f}  vs  {within:.4f}   "
                  f"{z - within:+.4f} ({100*(z - within)/within:+.1f}%)")

print("\nReference points:")
print("  pooled multi-source AUC (T11b)        0.9669   — inflated by source mixing")
print("  within-source AUC (T11b, 6 sources)   0.8784   — the like-for-like comparison")
print("  SMDG-only model zero-shot on G1020    0.589    — the documented domain-shift failure")
print("\nA zero-shot figure near 0.88 would mean the multi-source model genuinely")
print("generalises. Near 0.59 would mean the earlier 'fix' only worked because the")
print("test set had been absorbed into training.")

json.dump(out, open("/kaggle/working/glaucoma_heldout.json", "w"), indent=2, default=float)
shutil.copy(W, "/kaggle/working/glaucoma_heldout_weights.pth")
print("\nDONE", flush=True)
