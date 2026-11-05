"""Emit per-fold manifests for the 6-class ROP staging benchmark.

    .venv/bin/python scripts/make_staging_folds.py

Reads  results/prepared/rop_staging/splits.csv   (from build_rop_splits.py)
Writes results/prepared/rop_staging/fold_{0..4}.csv
       results/prepared/rop_staging/fold_site.csv     (leave-one-site-out: Shenzhen as val)
       results/prepared/rop_staging/test_locked.csv   (the held-out hospital, opened ONCE)
       results/prepared/rop_staging/test_dev.csv      (audit/debug half of the held-out site)
       results/prepared/rop_staging/class_index.json

Each fold manifest is in the schema models/common/datasets.py already reads —
``image_path, label, split`` with split in {train, val} — so no loader changes are needed and
the existing training loop runs against these directly.

The per-infant caps live in splits.csv as the `kept` flag and are applied to TRAIN ONLY. The
val side of every fold is uncapped and at natural prevalence, because a capped val set would
make the validation metric measure the cap rather than the model.
"""
import json
import os
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
D = ROOT / "results" / "prepared" / "rop_staging"
SPLITS = D / "splits.csv"
CLASSES = ["normal", "stage_1", "stage_2", "stage_3", "stage_4_5", "arop"]
IDX = {c: i for i, c in enumerate(CLASSES)}


def emit(df, path, note):
    out = df[["image_path", "label", "split", "patient_id", "source", "cls6"]].copy()
    # Paths are stored RELATIVE TO THE REPO ROOT, not absolute. These CSVs are the experiment
    # contract — they get committed, and they get read on Kaggle where the tree mounts at a
    # different prefix. An absolute /Users/... path makes the fold assignment unreproducible
    # on any other machine and forces the Kaggle kernel to string-patch every row.
    # train.py runs with cwd=repo root, so relative paths resolve unchanged locally.
    out["image_path"] = out.image_path.map(
        lambda p: os.path.relpath(p, ROOT) if os.path.isabs(p) else p)
    out.to_csv(path, index=False)
    n = out.groupby("split").size().to_dict()
    per = pd.crosstab(out.split, out.cls6).reindex(columns=CLASSES, fill_value=0)
    print(f"  {path.name:<18} {n}  {note}")
    return per


def main():
    if not SPLITS.exists():
        print(f"missing {SPLITS} — run scripts/build_rop_splits.py first", file=sys.stderr)
        return 1
    df = pd.read_csv(SPLITS)
    df["label"] = df.cls6.map(IDX)
    assert df.label.notna().all(), "unmapped class in splits.csv"
    df["label"] = df.label.astype(int)

    train_all = df[df.split == "train_cv"]
    print(f"training pool: {len(train_all):,} rows ({int(train_all.kept.sum()):,} kept after caps)")
    print("\nemitting fold manifests:")

    # ---- 5 patient-grouped CV folds -------------------------------------------------
    for f in sorted(train_all.fold.unique()):
        tr = train_all[(train_all.fold != f) & train_all.kept].assign(split="train")
        va = train_all[train_all.fold == f].assign(split="val")   # uncapped, natural prevalence
        emit(pd.concat([tr, va]), D / f"fold_{f}.csv", f"val=fold{f}")

    # ---- leave-one-site-out: train on Ostrava+ROP-VL, validate on Shenzhen ----------
    tr = train_all[(~train_all.site_val) & train_all.kept].assign(split="train")
    va = train_all[train_all.site_val].assign(split="val")
    per = emit(pd.concat([tr, va]), D / "fold_site.csv", "val=Shenzhen (cross-site)")
    print("     ^ cross-site estimate that does NOT touch the held-out hospital")
    if per.loc["val", "stage_4_5"] == 0:
        print("     NOTE: Shenzhen has no stage_4_5 or arop — this fold scores 4 classes only")

    # ---- the held-out hospital ------------------------------------------------------
    for name, split_name in (("test_dev", "multiview_dev"), ("test_locked", "multiview_locked")):
        sub = df[df.split == split_name].assign(split="test")
        emit(sub, D / f"{name}.csv", "held-out hospital")

    (D / "class_index.json").write_text(json.dumps(
        {"classes": CLASSES, "index": IDX,
         "note": "stage_4 and stage_5 merged into stage_4_5; stage_0 and treated out of scope"},
        indent=2) + "\n")

    print("\nclass index:", IDX)
    print(f"\n-> {D}/fold_*.csv, test_*.csv, class_index.json")
    print("\nUSE: fold_0..4 for model selection (pooled out-of-fold metrics, no early stopping).")
    print("     fold_site for the cross-site sanity check.")
    print("     test_dev for shortcut audits. test_locked ONCE, at the very end.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
