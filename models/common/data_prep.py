"""Build a uniform manifest (image_path,label,split,source,group) from the config's data sources.

Supports two source types:
  - ``csv``              : a labels CSV + an images dir (EyePACS, APTOS, SMDG)
  - ``imagefolder_split``: pre-split class-folders (ROP dataset_split/{train,val,test})

CSV sources are pooled and given a deterministic stratified train/val(/test) split.
The ``data.use_sources`` list (optional) filters which named sources are included —
used by the DR ablation to toggle EyePACS-only vs +APTOS.

GROUPING (why this exists). Splitting per image leaks when several images share a patient:
the model sees one eye in train and is scored on the fellow eye in val, which is nearly the
same picture of the same disease. Measured on this repo's own data, EVERY one of EyePACS's
17,563 patients contributes both eyes (``<id>_left`` / ``<id>_right``), so an image-level
split put a sibling of ~85% of val images into train. A source may therefore declare
``group_pattern``: a regex applied to the image filename whose first capture group is the
patient key. Rows sharing a key are kept on the same side of every split. Sources without
``group_pattern`` behave exactly as before — each image is its own group — so adding this
changes nothing until a config opts in.
"""
from __future__ import annotations

import random
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd

# Emitted for rows whose source declares no grouping rule. Each such row is its own group,
# which reproduces the old per-image behaviour exactly.
UNGROUPED = ""


def _find_image(images_dir: Path, stem: str, exts) -> Path | None:
    for ext in exts:
        p = images_dir / f"{stem}{ext}"
        if p.exists():
            return p
    # stem may already include an extension
    direct = images_dir / stem
    return direct if direct.exists() else None


def _group_of(src, path: Path) -> str:
    """Patient key for one image, or UNGROUPED when the source declares no rule.

    The regex is matched against the filename STEM (no directory, no extension) so a pattern
    stays valid when the same dataset is mounted at a different path. A pattern that compiles
    but does not match is a silent-wrongness risk — every unmatched row would become its own
    group and leak — so it raises instead.
    """
    pattern = src.get("group_pattern", None)
    if not pattern:
        return UNGROUPED
    m = re.match(pattern, path.stem)
    if not m or not m.groups():
        raise ValueError(
            f"[data_prep] {src.name}: group_pattern {pattern!r} did not match '{path.stem}'. "
            f"A non-matching pattern would silently disable grouping and leak siblings across "
            f"splits, so this is fatal. Fix the pattern or remove it."
        )
    return f"{src.name}:{m.group(1)}"


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
        rows.append({"image_path": str(path), "label": label, "split": None,
                     "source": src.name, "group": _group_of(src, path)})
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
                    rows.append({"image_path": str(img), "label": label, "split": split,
                                 "source": src.name, "group": _group_of(src, img)})
    return rows


def _stratified_split(rows, val_split, test_split, seed):
    """Assign splits to rows that don't already have one, stratified by label.

    Rows sharing a ``group`` (patient key) always land on the same side. Grouping and
    stratification pull against each other — a group can hold several labels — so a group is
    stratified by its WORST label (``max``), which for an ordinal grade is the sick eye and for
    a binary label is "this patient has the disease". Quotas are then filled by IMAGE count,
    not group count, because group sizes are wildly unequal: one infant in the ROP set supplies
    470 of 3,024 positive images. Filling by group count would let that single patient swing
    the val fraction by 15%.
    """
    # group key -> row indices. Ungrouped rows each get a unique key, so they behave as before.
    members: dict[str, list[int]] = defaultdict(list)
    for i, row in enumerate(rows):
        if row["split"] is None:
            g = row.get("group") or UNGROUPED
            members[g if g != UNGROUPED else f"\0row{i}"].append(i)

    by_stratum: dict[int, list[str]] = defaultdict(list)
    for g, idxs in members.items():
        by_stratum[max(rows[i]["label"] for i in idxs)].append(g)

    rng = random.Random(seed)
    for stratum in sorted(by_stratum):
        groups = sorted(by_stratum[stratum])          # sort first: dict order must not leak in
        rng.shuffle(groups)
        n_img = sum(len(members[g]) for g in groups)
        # Keep the old guarantee that a non-zero fraction yields a non-empty split.
        want = {"test": max(1, int(n_img * test_split)) if test_split else 0,
                "val": max(1, int(n_img * val_split)) if val_split else 0}
        want["train"] = n_img - want["test"] - want["val"]
        placed = {"test": 0, "val": 0, "train": 0}

        # Largest group first, then give each to whichever split is furthest below quota.
        # Taking them in shuffled order instead lets one oversized group land in a split whose
        # quota is far smaller than the group — with a 200-image patient and a 100-image val
        # quota that produced a 54% val fraction. Big-first bounds the overshoot by the
        # largest single group rather than by however big the group at the boundary happened
        # to be. `sort` is stable, so the shuffle still breaks ties between equal-sized groups.
        for g in sorted(groups, key=lambda g: -len(members[g])):
            split = max(("test", "val", "train"), key=lambda s: want[s] - placed[s])
            placed[split] += len(members[g])
            for i in members[g]:
                rows[i]["split"] = split
    return rows


def audit_group_leakage(df: pd.DataFrame) -> dict:
    """Count images whose group also appears in another split. Zero is the only good answer."""
    if "group" not in df.columns:
        return {"checked": False, "reason": "manifest has no group column"}
    real = df[df["group"].astype(str) != UNGROUPED]
    if real.empty:
        return {"checked": True, "grouped_rows": 0, "leaked_images": 0, "leaked_groups": 0}
    spans = real.groupby("group")["split"].nunique()
    bad = set(spans[spans > 1].index)
    return {"checked": True,
            "grouped_rows": int(len(real)),
            "leaked_groups": int(len(bad)),
            "leaked_images": int(real["group"].isin(bad).sum())}


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

    df = pd.DataFrame(rows)[["image_path", "label", "split", "source", "group"]]
    out = Path(cfg.data.manifest)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    counts = df.groupby(["split", "label"]).size().unstack(fill_value=0)
    print(f"[data_prep] wrote {len(df)} rows -> {out}")
    print(counts)

    leak = audit_group_leakage(df)
    if leak.get("grouped_rows"):
        n_groups = df.loc[df["group"].astype(str) != UNGROUPED, "group"].nunique()
        print(f"[data_prep] grouping: {leak['grouped_rows']} rows in {n_groups} patient groups")
        # A leak here means the split silently inflates every metric downstream, so refuse.
        if leak["leaked_images"]:
            raise RuntimeError(
                f"[data_prep] {leak['leaked_images']} images in {leak['leaked_groups']} groups "
                f"span more than one split. A grouped source must never leak; refusing to write "
                f"a manifest that would inflate val/test scores."
            )
        print("[data_prep] grouping: 0 images leak across splits")
    ungrouped = int((df["group"].astype(str) == UNGROUPED).sum())
    if ungrouped:
        print(f"[data_prep] NOTE: {ungrouped} rows have no patient key (no group_pattern on "
              f"their source) and were split per image.")
    return df
