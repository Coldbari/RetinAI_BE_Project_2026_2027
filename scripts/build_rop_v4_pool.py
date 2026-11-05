"""v4 training pool: PROTOCOL_V2 Amendment 5a, items 3-5, as revised after the audit.

    .venv/bin/python scripts/build_rop_v4_pool.py

Reads  results/prepared/rop_staging/splits.csv          (train_cv rows only, paths made relative)
       results/rop/v4/audit/{quality.csv, thresholds.json, near_duplicates.csv}
Writes results/prepared/rop_staging/v4/
       fold_{0..4}.csv, fold_site.csv            same validation rows as v3, byte for byte
       fold_loso_ostrava.csv, fold_loso_ropvl.csv leave-one-source-out
       exclude_near_duplicates.csv, v4_summary.json
   and results/prepared/rop_staging/fold_loso_{ostrava,ropvl}.csv: the same two LOSO folds on the
       v3 pool (train = v3 `kept` rows) for arm B; val sides byte-identical to the v4 files

Only TRAINING rows change. A pool row may train (`kept_v4`) when:
  1. it is not a member of a cross-fold true-duplicate pair (dHash <= 4 and pixel MAD <= 5.0);
     both members are dropped from training in every fold, and stay in validation;
  2. it is an Ostrava frame that passes the per-device quality rule (`rop_v4_audit.ostrava_pass`),
     or a ROP-VL / Shenzhen photo (never quality-filtered: their haze is often the disease);
  3. it is among the top-cap rows of its (infant, class) cell by `blur_rank_in_device`
     (descending, ties by image_path). Caps are v3's: normal 2, every other class 20.
Floor: an Ostrava (infant, class) cell with fewer than 2 passing frames keeps its 2
best-ranked frames regardless of quality (all of them if it has fewer), so no infant loses a
class. Near-duplicate members are removed before ranking, so they never take a cap slot.

Validation is never filtered or capped: fold_k val = every pool row of fold k, in splits.csv
order, exactly as scripts/make_staging_folds.py emits it. The 14 exact duplicates
(exclude_duplicates.csv) are dropped at kernel run time, as in v3, not here. No Multi-View row
and neither test CSV is read.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.build_rop_splits import CAPS, DEFAULT_CAP  # noqa: E402
from scripts.make_staging_folds import CLASSES, IDX, emit  # noqa: E402
from scripts.rop_v4_audit import FLOOR, load_pool, ostrava_pass, rel  # noqa: E402

V3_DIR = ROOT / "results" / "prepared" / "rop_staging"
OUT_DIR = V3_DIR / "v4"
AUDIT = ROOT / "results" / "rop" / "v4" / "audit"
QUALITY = AUDIT / "quality.csv"
THRESHOLDS = AUDIT / "thresholds.json"
NEAR = AUDIT / "near_duplicates.csv"

CELL = ["patient_id", "cls6"]
SOURCES = ["ostrava", "ropvl", "shenzhen"]
LOSO = {"fold_loso_ostrava": "ostrava", "fold_loso_ropvl": "ropvl"}
EXCLUDE_COLS = ["image_path", "pair_with", "distance", "pixel_mad"]
RANKING = "blur_rank_in_device descending, ties by image_path"


def near_duplicate_exclusions(near: pd.DataFrame) -> pd.DataFrame:
    """Both members of every cross-fold true-duplicate pair, one row per member."""
    x = near[near.cross_fold.astype(bool) & near.true_duplicate.astype(bool)]
    a = x.rename(columns={"image_a": "image_path", "image_b": "pair_with"})
    b = x.rename(columns={"image_b": "image_path", "image_a": "pair_with"})
    return (pd.concat([a[EXCLUDE_COLS], b[EXCLUDE_COLS]])
              .sort_values(["image_path", "pair_with"]).reset_index(drop=True))


def select_training_rows(pool: pd.DataFrame, quality: pd.DataFrame, thr: dict,
                         near: pd.DataFrame, caps: dict = CAPS,
                         default_cap: int = DEFAULT_CAP) -> pd.DataFrame:
    """`pool` (same order) plus `kept_v4` and the reasons behind it: `quality_pass`,
    `near_dup`, `floor_cell` (an Ostrava cell with fewer than FLOOR passing frames)."""
    q = quality[["image_path", "empty", "blur_var", "luminance", "blur_rank_in_device"]]
    df = pool.merge(q, on="image_path", how="left", validate="one_to_one")
    missing = int(df.blur_rank_in_device.isna().sum())
    if missing:
        raise ValueError(f"{missing} pool rows have no row in quality.csv")
    df["empty"] = df["empty"].astype(bool)

    ost = df.source.eq("ostrava")
    df["quality_pass"] = True
    df.loc[ost, "quality_pass"] = ostrava_pass(df[ost], thr).to_numpy()
    df["near_dup"] = df.image_path.isin(set(near_duplicate_exclusions(near).image_path))

    c = (df[~df.near_dup]
         .sort_values(["blur_rank_in_device", "image_path"], ascending=[False, True]))
    cap = c.cls6.map(lambda k: caps.get(k, default_cap))
    floor = ost.loc[c.index] & (c.groupby(CELL).quality_pass.transform("sum") < FLOOR)
    rank_all = c.groupby(CELL).cumcount()
    rank_pass = c[c.quality_pass].groupby(CELL).cumcount().reindex(c.index)  # NaN = failing
    keep = (~floor & (rank_pass < cap)) | (floor & (rank_all < FLOOR))

    df["floor_cell"] = df.set_index(CELL).index.isin(c[floor].set_index(CELL).index)
    df["kept_v4"] = df.index.isin(c.index[keep])
    return df


def load_inputs():
    pool = load_pool()
    return (pool, pd.read_csv(QUALITY), json.loads(THRESHOLDS.read_text()), pd.read_csv(NEAR))


def _val_lines(path: Path) -> list[str]:
    return [ln for ln in path.read_text().splitlines()[1:] if ln.split(",")[2] == "val"]


def _by_class(d: pd.DataFrame) -> dict:
    return {c: int((d.cls6 == c).sum()) for c in CLASSES}


def summarise(sel: pd.DataFrame, per: dict, excl: pd.DataFrame, thr: dict) -> dict:
    tr, v3 = sel[sel.kept_v4], sel[sel.kept]
    o = sel[sel.source == "ostrava"]
    fail = o[~o.quality_pass]
    removed = fail[~fail.kept_v4]
    floor_cells = [{"patient_id": pid, "cls6": cls, "frames": len(g),
                    "passing": int(g.quality_pass.sum()),
                    "kept": sorted(g[g.kept_v4].image_path),
                    "kept_failing": int((g.kept_v4 & ~g.quality_pass).sum())}
                   for (pid, cls), g in o[o.floor_cell].groupby(CELL)]
    devices = sorted(o.device.unique())
    return {
        "rule": thr["rule"],
        "thresholds_by_device": thr["by_device"],
        "caps": {**CAPS, "_default": DEFAULT_CAP}, "floor": FLOOR, "ranking": RANKING,
        "pool_rows": len(sel),
        "training_rows": {"v3_kept": len(v3), "v4": len(tr)},
        "training_by_class": {"v3_kept": _by_class(v3), "v4": _by_class(tr)},
        "training_infants_by_class": {c: int(tr[tr.cls6 == c].patient_id.nunique())
                                      for c in CLASSES},
        "training_infants": int(tr.patient_id.nunique()),
        "training_by_source": {s: {"v3_kept": _by_class(v3[v3.source == s]),
                                   "v4": _by_class(tr[tr.source == s])} for s in SOURCES},
        "selection_changed_vs_v3": {
            s: {"v3_kept_only": int((sel.kept & ~sel.kept_v4 & (sel.source == s)).sum()),
                "v4_only": int((~sel.kept & sel.kept_v4 & (sel.source == s)).sum())}
            for s in SOURCES},
        "training_by_fold_assignment": {int(f): _by_class(g)
                                        for f, g in tr.groupby("fold")},
        "ostrava_quality": {
            "frames": len(o), "failing": len(fail),
            "failing_kept_by_floor": int(fail.kept_v4.sum()),
            "removed_by_quality": len(removed),
            "v3_kept_removed": int(removed.kept.sum()),
            "failing_by_class": _by_class(fail),
            "removed_by_class": _by_class(removed),
            "failing_by_device": {d: int((fail.device == d).sum()) for d in devices},
            "removed_by_device": {d: int((removed.device == d).sum()) for d in devices},
        },
        "floor_cells": floor_cells,
        "near_duplicate_exclusions": {
            "images": int(excl.image_path.nunique()),
            "pairs": len(excl) // 2,
            "by_source": {s: int(sel[sel.near_dup & (sel.source == s)].shape[0])
                          for s in SOURCES},
            "v3_kept_among_them": int(sel[sel.near_dup].kept.sum()),
            "rows": excl.to_dict("records"),
        },
        "folds": {name: {split: {c: int(ct.loc[split, c]) for c in CLASSES}
                         for split in ("train", "val")} for name, ct in per.items()},
    }


def loso_folds(pool: pd.DataFrame, train: pd.Series) -> dict:
    """The two leave-one-source-out folds, {name: (train rows, val rows, note)}: val = every pool
    row of the held-out source, in pool order; train = the rows `train` marks among the other two
    sources (`kept_v4` for the v4 pool, v3's `kept` for the v3 pool)."""
    tr = pool[train]
    return {name: (tr[tr.source != src], pool[pool.source == src], f"val=all {src} (LOSO)")
            for name, src in LOSO.items()}


def write_folds(out_dir) -> dict:
    """Build the v4 pool and write every v4 fold file plus the exclusions and the summary."""
    pool, quality, thr, near = load_inputs()
    sel = select_training_rows(pool, quality, thr, near)
    sel["label"] = sel.cls6.map(IDX).astype(int)
    tr = sel[sel.kept_v4]
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    folds = {f"fold_{f}": (tr[tr.fold != f], sel[sel.fold == f], f"val=fold{f}")
             for f in sorted(sel.fold.unique())}
    folds["fold_site"] = (tr[~tr.site_val], sel[sel.site_val], "val=Shenzhen (cross-site)")
    folds.update(loso_folds(sel, sel.kept_v4))

    per = {}
    for name, (t, v, note) in folds.items():
        per[name] = emit(pd.concat([t.assign(split="train"), v.assign(split="val")]),
                         out / f"{name}.csv", note)
        if (V3_DIR / f"{name}.csv").exists():
            assert _val_lines(out / f"{name}.csv") == _val_lines(V3_DIR / f"{name}.csv"), \
                f"{name}: validation rows differ from v3"

    excl = near_duplicate_exclusions(near)
    excl.to_csv(out / "exclude_near_duplicates.csv", index=False)
    summary = summarise(sel, per, excl, thr)
    (out / "v4_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def write_v3_loso_folds(out_dir=V3_DIR, v4_dir=OUT_DIR) -> dict:
    """The LOSO folds on the v3 pool, for arm B: training rows are splits.csv's v3 `kept` flag
    (no quality rule, no re-ranking, no near-duplicate exclusion). Val sides must be byte-identical
    to the v4 LOSO files in `v4_dir`."""
    pool = load_pool()
    pool["label"] = pool.cls6.map(IDX).astype(int)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    per = {}
    for name, (t, v, note) in loso_folds(pool, pool.kept.astype(bool)).items():
        per[name] = emit(pd.concat([t.assign(split="train"), v.assign(split="val")]),
                         out / f"{name}.csv", note + ", v3 pool")
        assert _val_lines(out / f"{name}.csv") == _val_lines(Path(v4_dir) / f"{name}.csv"), \
            f"{name}: v3-pool validation rows differ from the v4 file"
    return per


def _print_tables(s: dict):
    tc, inf = s["training_by_class"], s["training_infants_by_class"]
    t = pd.DataFrame({"v3 images": tc["v3_kept"], "v4 images": tc["v4"],
                      "v4 infants": inf}).reindex(CLASSES)
    t["v4 %"] = (100 * t["v4 images"] / t["v4 images"].sum()).round(1)
    print("\n" + "=" * 78)
    print("V4 TRAINING POOL (quality, caps re-ranked, floor, near-duplicates out)")
    print("=" * 78)
    print(t.to_string())
    v4 = t["v4 images"]
    print(f"  total {int(v4.sum()):,} images / {s['training_infants']:,} infants  "
          f"| imbalance {v4.max() / v4[v4 > 0].min():.1f}:1  "
          f"(v3 kept {s['training_rows']['v3_kept']:,} of {s['pool_rows']:,})")

    print("\nPER-SOURCE training images, v3 kept -> v4:")
    src = pd.DataFrame({f"{c.replace('stage_', 's')}": [
        f"{s['training_by_source'][x]['v3_kept'][c]}->{s['training_by_source'][x]['v4'][c]}"
        for x in SOURCES] for c in CLASSES}, index=SOURCES)
    print(src.to_string())
    print("  selection changed vs v3:", s["selection_changed_vs_v3"])

    print("\nPER-FOLD class counts of v4 training rows (by fold assignment):")
    print(pd.DataFrame(s["training_by_fold_assignment"]).T.reindex(columns=CLASSES).to_string())

    print("\nFOLD FILES (train / val images):")
    for name, v in s["folds"].items():
        print(f"  {name:<18} train {sum(v['train'].values()):>5,}  val {sum(v['val'].values()):>5,}"
              f"   train s4_5 {v['train']['stage_4_5']:>3}  val s4_5 {v['val']['stage_4_5']:>3}")

    oq = s["ostrava_quality"]
    print(f"\nOSTRAVA QUALITY: {oq['failing']:,}/{oq['frames']:,} frames fail; "
          f"{oq['failing_kept_by_floor']} kept by the floor; "
          f"{oq['removed_by_quality']:,} removed from training "
          f"({oq['v3_kept_removed']} of them were v3-kept)")
    print("  removed by class: ", oq["removed_by_class"])
    print("  removed by device:", oq["removed_by_device"])
    for f in s["floor_cells"]:
        print(f"  floor: {f['patient_id']} {f['cls6']}: {f['passing']}/{f['frames']} pass, "
              f"kept {len(f['kept'])} ({f['kept_failing']} failing)")
    nd = s["near_duplicate_exclusions"]
    print(f"\nNEAR-DUPLICATE exclusions (cross-fold true duplicates): {nd['images']} images / "
          f"{nd['pairs']} pairs, by source {nd['by_source']}")


def main():
    missing = [rel(str(p)) for p in (THRESHOLDS, QUALITY, NEAR) if not p.exists()]
    if missing:
        print(f"missing {', '.join(missing)} — run scripts/rop_v4_audit.py first",
              file=sys.stderr)
        return 1
    print("emitting v4 fold manifests:")
    s = write_folds(OUT_DIR)
    _print_tables(s)
    print(f"\n-> {rel(str(OUT_DIR))}/")
    print("\nemitting the v3-pool LOSO manifests (arm B):")
    write_v3_loso_folds(V3_DIR, OUT_DIR)
    print(f"-> {rel(str(V3_DIR))}/fold_loso_*.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
