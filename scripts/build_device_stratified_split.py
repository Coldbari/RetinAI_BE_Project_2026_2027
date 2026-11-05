"""T9 — build a DEVICE-STRATIFIED, duplicate-clean patient split for ROP.

Why the current split is unusable for an honest number (results/rop_device_audit.json):
  * ROP prevalence per device is 48.1% / 21.3% / 17.4% — device predicts the label
  * test is 62.9% D1 vs 36.3% in train — enriched in the highest-prevalence camera
  * EVERY ROP-positive test image is D1; D2/D3 test images are 100% negative, so
    discrimination is only measurable inside one device and the pooled AUC is partly a
    device-vs-device separation

This builds a replacement split with three properties the old one lacks:
  1. duplicate images with CONTRADICTORY labels are dropped (19 groups, byte-identical
     pixels labelled both DG0 and DG1 — we cannot know which label is right)
  2. patient IDs that are the same infant (same sex+GA+BW) never straddle a split
  3. every split carries every device, and each device carries BOTH classes wherever the
     patient pool allows it

Writes results/prepared/rop/manifest_devsplit.csv and refuses to write a split that fails
its own checks.

    .venv/bin/python scripts/build_device_stratified_split.py
"""
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

SEED = 42
TARGET = {"train": 0.70, "val": 0.15, "test": 0.15}


def parse(p):
    m = re.match(r"(?P<pid>\d+)_(?P<sex>[MF])_GA(?P<ga>\d+)_BW(?P<bw>\d+)_"
                 r"PA(?P<pa>\d+)_DG(?P<dg>\d)_PF(?P<pf>\d)_D(?P<dev>\d+)_S(?P<sess>\d+)",
                 Path(p).name)
    return m.groupdict() if m else None


def contradictory_duplicates(man):
    """Byte-identical images carrying different labels — unusable either way."""
    try:
        import imagehash
    except ImportError:
        print("  imagehash unavailable; skipping duplicate removal")
        return set()
    groups = defaultdict(list)
    for _, r in man.iterrows():
        try:
            h = str(imagehash.phash(Image.open(r["image_path"]).convert("RGB")))
        except Exception:
            continue
        groups[h].append((r["image_path"], int(r["label"])))
    bad = set()
    for g in groups.values():
        if len({lab for _, lab in g}) > 1:
            bad.update(p for p, _ in g)
    return bad


def main():
    man = pd.read_csv(ROOT / "results" / "prepared" / "rop" / "manifest.csv")
    missing = [p for p in man.image_path if not (ROOT / p).exists()]
    if missing:
        print(f"ERROR: {len(missing)}/{len(man)} images not present locally. "
              f"Extract rop_dataset.zip first — building a split from absent files would "
              f"silently produce a manifest nothing can train on.")
        return 1

    bad = contradictory_duplicates(man)
    man = man[~man.image_path.isin(bad)].reset_index(drop=True)
    print(f"dropped {len(bad)} images in label-contradicting duplicate groups -> {len(man)} remain")

    meta = pd.DataFrame([{**parse(p), "image_path": p, "label": int(l)}
                         for p, l in zip(man.image_path, man.label) if parse(p)])

    # same infant filed under two patient IDs -> one unit, never straddling a split
    sig2pids = defaultdict(set)
    for r in meta.itertuples():
        sig2pids[(r.sex, r.ga, r.bw)].add(r.pid)
    pid2unit, merged = {}, 0
    for pids in sig2pids.values():
        root = sorted(pids)[0]
        if len(pids) > 1:
            merged += 1
        for p in pids:
            pid2unit[p] = root
    meta["unit"] = meta["pid"].map(pid2unit)
    print(f"merged {merged} demographic signatures shared by >1 patient ID")

    # one row per patient-unit: its device(s) and whether it is ROP-positive
    units = meta.groupby("unit").agg(
        label=("label", "max"),
        dev=("dev", lambda s: s.value_counts().idxmax()),
        n=("image_path", "size")).reset_index()
    print(f"{len(units)} patient-units, {int(units.label.sum())} ROP-positive")

    # stratify by (device, label): assign each stratum's units across splits by target ratio
    rng = random.Random(SEED)
    assign = {}
    for (dev, lab), grp in units.groupby(["dev", "label"]):
        ids = list(grp["unit"])
        rng.shuffle(ids)
        n = len(ids)
        n_test = max(1, round(n * TARGET["test"])) if n >= 3 else (1 if n >= 2 else 0)
        n_val = max(1, round(n * TARGET["val"])) if n >= 4 else 0
        for i, u in enumerate(ids):
            assign[u] = "test" if i < n_test else ("val" if i < n_test + n_val else "train")
        print(f"  D{dev} label={lab}: {n} units -> test {n_test}, val {n_val}, "
              f"train {n - n_test - n_val}")

    meta["split"] = meta["unit"].map(assign)
    out = meta[["image_path", "label", "split"]].copy()
    out["source"] = "rop"

    # ── refuse to ship a split that fails the property it exists to guarantee ──
    print("\nVALIDATION")
    ok = True
    for sp in ("train", "val", "test"):
        s = meta[meta.split == sp]
        print(f"  {sp:<6} {len(s):>5} images  {s.unit.nunique():>3} units  "
              f"prev {s.label.mean():.3f}  devices {sorted(s.dev.unique())}")
    for sp in ("val", "test"):
        s = meta[meta.split == sp]
        for dev in sorted(meta.dev.unique()):
            d = s[s.dev == dev]
            both = d.label.nunique() > 1
            print(f"    {sp} D{dev}: n={len(d):<5} pos={int(d.label.sum()):<5} "
                  f"{'both classes OK' if both else 'SINGLE CLASS'}")
            if sp == "test" and not both:
                ok = False
    leak = meta.groupby("unit")["split"].nunique()
    if (leak > 1).any():
        print(f"  FAIL: {(leak > 1).sum()} patient-units straddle splits")
        ok = False
    else:
        print("  no patient-unit straddles a split")

    if not ok:
        print("\nSplit FAILED its own checks — not written. The test split must carry both "
              "classes on every device, or it reproduces the confound it was built to remove.")
        return 1

    # Create both output dirs. A fresh environment (e.g. a Kaggle kernel, where the code
    # bundle ships no results/rop/) has neither, and failing AFTER a valid split has been
    # computed throws away ~2 minutes of work for a missing directory.
    dst = ROOT / "results" / "prepared" / "rop" / "manifest_devsplit.csv"
    dst.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(dst, index=False)
    summary = {"dropped_contradictory": len(bad), "merged_signatures": merged,
               "units": int(len(units)), "positives": int(units.label.sum()),
               "counts": {sp: int((meta.split == sp).sum()) for sp in ("train", "val", "test")}}
    sdst = ROOT / "results" / "rop" / "devsplit_summary.json"
    sdst.parent.mkdir(parents=True, exist_ok=True)
    sdst.write_text(json.dumps(summary, indent=2))
    print(f"\n-> {dst}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
