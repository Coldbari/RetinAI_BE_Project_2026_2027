"""Build a uniform manifest (image_path,label,split) from the config's data sources.

Supports two source types:
  - ``csv``              : a labels CSV + an images dir (EyePACS, APTOS, SMDG)
  - ``imagefolder_split``: pre-split class-folders (ROP dataset_split/{train,val,test})

CSV sources are pooled and given a deterministic stratified train/val(/test) split.
The ``data.use_sources`` list (optional) filters which named sources are included —
used by the DR ablation to toggle EyePACS-only vs +APTOS.
"""
from __future__ import annotations

import random
from collections import defaultdict
from pathlib import Path

import pandas as pd


def _find_image(images_dir: Path, stem: str, exts) -> Path | None:
    for ext in exts:
        p = images_dir / f"{stem}{ext}"
        if p.exists():
            return p
    # stem may already include an extension
    direct = images_dir / stem
    return direct if direct.exists() else None


def _rows_from_csv(src) -> list[dict]:
    images_dir = Path(src.images_dir)
    df = pd.read_csv(src.csv)
    exts = list(src.get("ext", [".jpeg", ".jpg", ".png"]))
    drop_label = src.get("drop_label", None)
    rows = []
    missing = 0
    for _, r in df.iterrows():
        label = int(r[src.label_col])
        if drop_label is not None and label == int(drop_label):
            continue
        path = _find_image(images_dir, str(r[src.image_col]), exts)
        if path is None:
            missing += 1
            continue
        rows.append({"image_path": str(path), "label": label,
                     "split": None, "source": src.name})
    if missing:
        print(f"[data_prep] {src.name}: {missing} images referenced in CSV not found on disk")
    return rows


def _rows_from_imagefolder_split(src) -> list[dict]:
    rows = []
    split_dirs = {"train": src.get("train_dir"), "val": src.get("val_dir"),
                  "test": src.get("test_dir")}
    for split, d in split_dirs.items():
        if not d:
            continue
        base = Path(d)
        if not base.exists():
            print(f"[data_prep] {src.name}: split dir missing: {base}")
            continue
        classes = sorted([c.name for c in base.iterdir() if c.is_dir()])
        for label, cname in enumerate(classes):
            for img in (base / cname).rglob("*"):
                if img.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                    rows.append({"image_path": str(img), "label": label,
                                 "split": split, "source": src.name})
    return rows


def _stratified_split(rows, val_split, test_split, seed):
    """Assign splits to rows that don't already have one (stratified by label)."""
    by_label = defaultdict(list)
    for i, row in enumerate(rows):
        if row["split"] is None:
            by_label[row["label"]].append(i)
    rng = random.Random(seed)
    for label, idxs in by_label.items():
        rng.shuffle(idxs)
        n = len(idxs)
        n_val = max(1, int(n * val_split)) if val_split else 0
        n_test = max(1, int(n * test_split)) if test_split else 0
        for j, idx in enumerate(idxs):
            if j < n_test:
                rows[idx]["split"] = "test"
            elif j < n_test + n_val:
                rows[idx]["split"] = "val"
            else:
                rows[idx]["split"] = "train"
    return rows


def build_manifest(cfg) -> pd.DataFrame:
    use = cfg.data.get("use_sources", None)
    use = set(use) if use else None

    rows: list[dict] = []
    for src in cfg.data.sources:
        if use is not None and src.name not in use:
            continue
        if src.type == "csv":
            rows += _rows_from_csv(src)
        elif src.type == "imagefolder_split":
            rows += _rows_from_imagefolder_split(src)
        else:
            raise ValueError(f"Unknown source type '{src.type}'")

    if not rows:
        raise RuntimeError("No images found — check the data source paths in the config.")

    rows = _stratified_split(
        rows,
        float(cfg.data.get("val_split", 0.15)),
        float(cfg.data.get("test_split", 0.0)),
        int(cfg.seed),
    )

    df = pd.DataFrame(rows)[["image_path", "label", "split", "source"]]
    out = Path(cfg.data.manifest)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    counts = df.groupby(["split", "label"]).size().unstack(fill_value=0)
    print(f"[data_prep] wrote {len(df)} rows -> {out}")
    print(counts)
    return df
