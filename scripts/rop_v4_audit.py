"""v4 data audit: PROTOCOL_V2 Amendment 5a, items 1-2, as revised after the first audit.

Two measurements over the 6,700-photo training pool (`splits.csv` rows with
`split == "train_cv"`), written before any v4 pool is built:

  1. Near-duplicate sweep. 256-bit dHash (`build_rop_staging_manifest.dhash256`) of every
     pool photo, every pair at Hamming distance <= 4 (the threshold calibrated on Shenzhen's
     byte-duplicates). Each pair is tagged with its two folds, two labels, whether both frames
     come from one Ostrava exam session (`<infant>_D<device>_S<session>_<frame>.jpg`), and its
     pixel MAD (mean absolute grey-level difference at 256x256). dHash <= 4 alone collides
     across distinct Ostrava retinas, so a pair is a `true_duplicate` only if it also has
     pixel MAD <= 5.0. Every pair stays on the clinician list.
  2. Frame quality. `data_quality.gradability_metrics` (work size 512) on every pool photo.
     The first rule took its thresholds from the two single-photo hospitals and measured the
     camera instead of the frame (the field of view is an aperture constant). Revised rule:
     quality is judged WITHIN each Ostrava device, over that device's non-empty Ostrava
     frames. min_blur = 10th percentile of blur_var, lum_lo / lum_hi = 2.5th / 97.5th
     percentiles of luminance, and no field-of-view criterion. The superseded hospital
     thresholds and their counts are kept in the outputs for the record only.

Nothing here filters or relabels: the pool builder (scripts/build_rop_v4_pool.py) applies the
rules. Only pool rows are read; no Multi-View image and neither test CSV is ever opened.
Every path written is relative to the repo root.

    .venv/bin/python scripts/rop_v4_audit.py

Writes results/rop/v4/audit/: near_duplicates.csv, quality.csv, thresholds.json,
audit_summary.json, AUDIT.md.
"""
from __future__ import annotations

import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from models.common import data_quality  # noqa: E402
from models.common.data_quality import gradability_metrics  # noqa: E402
from scripts.build_rop_staging_manifest import KEY_RE, POP, dhash256  # noqa: E402

SPLITS = ROOT / "results" / "prepared" / "rop_staging" / "splits.csv"
CLASS_INDEX = ROOT / "results" / "prepared" / "rop_staging" / "class_index.json"
OUT_DIR = ROOT / "results" / "rop" / "v4" / "audit"

DHASH_THR = 4
MAX_MAD, MAD_SIZE = 5.0, 256
WORK_SIZE = 512
FLOOR = 2                                # (infant, class) floor applied by the pool builder
DEVICE_PCT = {"min_blur": ("blur_var", 10.0), "lum_lo": ("luminance", 2.5),
              "lum_hi": ("luminance", 97.5)}
RULE = ("An Ostrava frame passes if it is not empty and, against the thresholds of its own "
        "device (percentiles over that device's non-empty Ostrava frames), blur_var >= "
        "min_blur (10th percentile) and lum_lo <= luminance <= lum_hi (2.5th / 97.5th "
        "percentiles); there is no field-of-view criterion.")
# Superseded (Amendment 5a.2 as first registered): ROP-VL + Shenzhen percentiles. Record only.
REFERENCE = ("ropvl", "shenzhen")
REFERENCE_PCT = {"min_blur": ("blur_var", 5.0), "lum_lo": ("luminance", 2.5),
                 "lum_hi": ("luminance", 97.5), "min_fov": ("fov_ratio", 5.0)}

QUALITY_COLS = ["image_path", "source", "device", "blur_var", "luminance", "fov_ratio", "empty",
                "blur_rank_in_device"]
PAIR_COLS = ["image_a", "image_b", "distance", "fold_a", "fold_b", "cls6_a", "cls6_b",
             "same_session", "cross_fold", "same_label"]
REASONS = ("empty", "blur", "dark", "bright")
OLD_REASONS = REASONS + ("fov",)
WORKERS, BLOCK, PROGRESS = 8, 128, 500


def resolve(p: str) -> str:
    return p if os.path.isabs(p) else str(ROOT / p)


def rel(p: str) -> str:
    return os.path.relpath(p, ROOT) if os.path.isabs(p) else p


def _map(fn, items, what):
    """`fn` over `items` on 8 threads, in order, printing progress every 500."""
    out = []
    with ThreadPoolExecutor(WORKERS) as ex:
        for i, r in enumerate(ex.map(fn, items), 1):
            out.append(r)
            if i % PROGRESS == 0 or i == len(items):
                print(f"  {what}: {i:,}/{len(items):,}", flush=True)
    return out


# ── 1. near-duplicate sweep ──────────────────────────────────────────────────
def near_duplicate_pairs(paths: list[str], thr: int = DHASH_THR) -> list[tuple[int, int, int]]:
    """Every index pair (a < b) whose 256-bit dHashes lie within Hamming distance `thr`."""
    H = np.stack(_map(lambda p: dhash256(resolve(p)), list(paths), "dHash"))
    pairs = []
    for i in range(0, len(H), BLOCK):
        d = POP[np.bitwise_xor(H[i:i + BLOCK, None, :], H[None, i:, :])].sum(-1)
        ra, rb = np.nonzero(d <= thr)              # rb is offset by i, like ra
        keep = rb > ra                             # upper triangle: a < b, no self-pairs
        pairs += [(i + int(x), i + int(y), int(d[x, y])) for x, y in zip(ra[keep], rb[keep])]
    return sorted(pairs)


def _session(path: str, source: str):
    """(infant, session) of an Ostrava frame; None for any other source."""
    if source != "ostrava":
        return None
    m = KEY_RE.match(os.path.basename(path))
    if m is None:
        raise ValueError(f"Ostrava filename without <infant>_D<d>_S<nn>_: {path}")
    return m.group(1), m.group(3)


def pair_table(pool: pd.DataFrame, pairs: list[tuple[int, int, int]]) -> pd.DataFrame:
    """One row per near-duplicate pair, with folds, labels and the session/fold/label flags."""
    pool = pool.reset_index(drop=True)
    a = pool.iloc[[p[0] for p in pairs]].reset_index(drop=True)
    b = pool.iloc[[p[1] for p in pairs]].reset_index(drop=True)
    sa = [_session(p, s) for p, s in zip(a.image_path, a.source)]
    sb = [_session(p, s) for p, s in zip(b.image_path, b.source)]
    t = pd.DataFrame({
        "image_a": a.image_path, "image_b": b.image_path,
        "distance": pd.Series([p[2] for p in pairs], dtype=int),
        "fold_a": a.fold, "fold_b": b.fold, "cls6_a": a.cls6, "cls6_b": b.cls6,
        "same_session": pd.Series([x is not None and x == y for x, y in zip(sa, sb)],
                                  dtype=bool),
    })
    t["cross_fold"] = t.fold_a != t.fold_b
    t["same_label"] = t.cls6_a == t.cls6_b
    return t[PAIR_COLS]


def pixel_mad(path_a: str, path_b: str, size: int = MAD_SIZE) -> float:
    """Mean absolute grey-level difference, both images as L resized to size x size bilinear."""
    def grey(p):
        with Image.open(resolve(p)) as im:
            return np.asarray(im.convert("L").resize((size, size), Image.BILINEAR),
                              dtype=np.float32)
    return float(np.abs(grey(path_a) - grey(path_b)).mean())


def confirm_duplicates(near: pd.DataFrame, max_mad: float = MAX_MAD) -> pd.DataFrame:
    """Add `pixel_mad` and `true_duplicate` (distance <= 4 and pixel_mad <= max_mad)."""
    near = near.copy()
    near["pixel_mad"] = pd.Series(
        _map(lambda ab: pixel_mad(*ab), list(zip(near.image_a, near.image_b)), "pixel MAD"),
        index=near.index, dtype=float)
    near["true_duplicate"] = (near.distance <= DHASH_THR) & (near.pixel_mad <= max_mad)
    return near


# ── 2. frame quality ─────────────────────────────────────────────────────────
def _metrics(path: str) -> dict:
    with Image.open(resolve(path)) as im:
        m = gradability_metrics(im, work_size=WORK_SIZE)
    return {k: m[k] for k in ("blur_var", "luminance", "fov_ratio", "empty")}


def blur_rank_in_device(q: pd.DataFrame) -> pd.Series:
    """Percentile rank (0-1] of blur_var within the row's source x device group."""
    return q.groupby(["source", "device"]).blur_var.rank(pct=True).rename("blur_rank_in_device")


def quality_table(df: pd.DataFrame) -> pd.DataFrame:
    """blur_var, luminance, fov_ratio, empty, blur_rank_in_device per row of `df` (same order)."""
    m = pd.DataFrame(_map(_metrics, df.image_path.tolist(), "quality"),
                     columns=["blur_var", "luminance", "fov_ratio", "empty"])
    q = pd.concat([df[["image_path", "source", "device"]].reset_index(drop=True), m], axis=1)
    q["empty"] = q["empty"].astype(bool)
    q["blur_rank_in_device"] = blur_rank_in_device(q)
    return q[QUALITY_COLS]


def thresholds_by_device(q: pd.DataFrame) -> dict[str, dict]:
    """{device: {min_blur, lum_lo, lum_hi, n}} from each device's non-empty Ostrava frames."""
    o = q[(q.source == "ostrava") & ~q["empty"].astype(bool)]
    return {dev: {**{k: float(np.percentile(g[col], pct))
                     for k, (col, pct) in DEVICE_PCT.items()}, "n": len(g)}
            for dev, g in o.groupby("device", sort=True)}


def _row_thresholds(q: pd.DataFrame, thr: dict) -> pd.DataFrame:
    """Each row's own device thresholds. Accepts {device: ...} or thresholds.json's dict."""
    thr = thr.get("by_device", thr)
    missing = sorted(set(q.device) - set(thr))
    if missing:
        raise KeyError(f"no quality thresholds for device(s) {missing}")
    return pd.DataFrame([thr[d] for d in q.device], index=q.index,
                        columns=["min_blur", "lum_lo", "lum_hi"])


def _fail_reasons(q: pd.DataFrame, thr: dict) -> dict:
    """Which condition of the per-device rule each row breaks (not exclusive)."""
    t = _row_thresholds(q, thr)
    return {"empty": q["empty"].astype(bool),
            "blur": q.blur_var < t.min_blur,
            "dark": q.luminance < t.lum_lo,
            "bright": q.luminance > t.lum_hi}


def ostrava_pass(q: pd.DataFrame, thr: dict) -> pd.Series:
    """The per-device Ostrava frame rule; `q` must carry `device`, `thr` cover every device."""
    r = _fail_reasons(q, thr)
    return ~(r["empty"] | r["blur"] | r["dark"] | r["bright"]).astype(bool).rename("pass")


def _reference_thresholds(q: pd.DataFrame) -> dict:
    """SUPERSEDED 5a.2 thresholds from non-empty ROP-VL and Shenzhen rows. Record only."""
    ref = q[q.source.isin(REFERENCE) & ~q["empty"].astype(bool)]
    return {k: float(np.percentile(ref[col], pct)) for k, (col, pct) in REFERENCE_PCT.items()}


def _reference_fail_reasons(q: pd.DataFrame, thr: dict) -> dict:
    """The superseded rule's conditions, for the record subsection."""
    return {"empty": q["empty"].astype(bool),
            "blur": q.blur_var < thr["min_blur"],
            "dark": q.luminance < thr["lum_lo"],
            "bright": q.luminance > thr["lum_hi"],
            "fov": q.fov_ratio < thr["min_fov"]}


# ── pool and report ──────────────────────────────────────────────────────────
def load_pool() -> pd.DataFrame:
    """The 6,700 train_cv rows of splits.csv, paths made repo-relative. Nothing else."""
    df = pd.read_csv(SPLITS)
    pool = df[df.split == "train_cv"].reset_index(drop=True)
    pool["image_path"] = pool.image_path.map(rel)
    assert not pool.image_path.str.contains("multiview").any(), \
        "a Multi-View path reached the audit pool"
    return pool


def _pid_key(pid: str):
    tail = pid.split(":")[-1]
    return (0, int(tail), pid) if tail.isdigit() else (1, 0, pid)


def _images_in(pairs: pd.DataFrame, pool: pd.DataFrame) -> dict:
    imgs = pool[pool.image_path.isin(set(pairs.image_a) | set(pairs.image_b))]
    return {"images": len(imgs),
            "by_source": {s: int((imgs.source == s).sum()) for s in sorted(pool.source.unique())}}


def _counts(pas: pd.Series, reasons: dict, m: pd.Series) -> dict:
    return {"frames": int(m.sum()), "failing": int((m & ~pas).sum()),
            "by_reason": {r: int((m & v).sum()) for r, v in reasons.items()}}


def _nd_summary(pool, near):
    n = len(pool)
    pidx = {p: i for i, p in enumerate(pool.image_path)}
    ia, ib = near.image_a.map(pidx).values, near.image_b.map(pidx).values
    src_a, src_b = pool.source.values[ia], pool.source.values[ib]
    same_infant = pool.patient_id.values[ia] == pool.patient_id.values[ib]
    src_pair = pd.Series(["–".join(sorted(x)) for x in zip(src_a, src_b)], dtype=object)
    kind = np.where(near.same_session, "same Ostrava session",
                    np.where(same_infant, "same infant, other session", "different infants"))
    td = near[near.true_duplicate]
    return {
        "threshold": DHASH_THR, "photos": n, "pairs_compared": n * (n - 1) // 2,
        "pairs_total": len(near),
        "cross_fold": int(near.cross_fold.sum()),
        "different_label": int((~near.same_label).sum()),
        "same_session": int(near.same_session.sum()),
        "cross_fold_different_label": int((near.cross_fold & ~near.same_label).sum()),
        "different_infant": int((~same_infant).sum()),
        "cross_source": int((src_a != src_b).sum()),
        "by_distance": {str(d): int((near.distance == d).sum()) for d in range(DHASH_THR + 1)},
        "by_source_pair": {k: int(v) for k, v in src_pair.value_counts().sort_index().items()},
        "max_pixel_mad": MAX_MAD, "pixel_mad_size": MAD_SIZE,
        "pixel_mad_by_kind": {
            k: {"pairs": int((kind == k).sum()),
                "true_duplicate": int(near.true_duplicate[kind == k].sum()),
                "min": float(near.pixel_mad[kind == k].min()),
                "median": float(near.pixel_mad[kind == k].median()),
                "max": float(near.pixel_mad[kind == k].max())}
            for k in ("same Ostrava session", "same infant, other session", "different infants")
            if (kind == k).any()},
        "true_duplicate": {
            "total": len(td), "cross_fold": int(td.cross_fold.sum()),
            "different_label": int((~td.same_label).sum()),
            "same_session": int(td.same_session.sum())},
        # The revised 5a.1 exclusion: both members of every cross-fold TRUE duplicate.
        "excluded_from_training": _images_in(td[td.cross_fold], pool),
        # What the first rule (all distance <= 4 cross-fold pairs) would have excluded.
        "superseded_exclusion": _images_in(near[near.cross_fold], pool),
    }


def summarise(pool, near, q, thr, ref_thr, classes):
    """Every count the report states, as plain data."""
    pq = pd.concat([pool[["patient_id", "cls6", "kept"]], q], axis=1)
    o = pq[pq.source == "ostrava"].copy()
    o["pass"] = ostrava_pass(o, thr)
    reasons = _fail_reasons(o, thr)

    by_class = {}
    for c in classes:
        m = o.cls6 == c
        cells = o[m].groupby("patient_id")["pass"].sum()
        by_class[c] = {**_counts(o["pass"], reasons, m), "infants": int(len(cells)),
                       "infants_with_no_passing_frame": int((cells == 0).sum()),
                       f"infants_below_floor_{FLOOR}": int((cells < FLOOR).sum())}
    by_device = {}
    for dev in sorted(o.device.unique()):
        m = o.device == dev
        by_device[dev] = {**_counts(o["pass"], reasons, m),
                          "by_class": {c: {"frames": int((m & (o.cls6 == c)).sum()),
                                           "failing": int((m & (o.cls6 == c) & ~o["pass"]).sum())}
                                       for c in classes if (m & (o.cls6 == c)).any()}}
    by_infant = {}
    for pid in sorted(o.patient_id.unique(), key=_pid_key):
        g = o[o.patient_id == pid]
        n_pass = g.groupby("cls6")["pass"].sum()
        by_infant[pid] = {"frames": len(g), "failing": int((~g["pass"]).sum()),
                          "devices": sorted(g.device.unique()),
                          "classes": sorted(g.cls6.unique(), key=classes.index),
                          f"classes_below_floor_{FLOOR}": sorted(
                              (c for c, k in n_pass.items() if k < FLOOR), key=classes.index)}
    oq = {"rule": RULE, **_counts(o["pass"], reasons, pd.Series(True, index=o.index)),
          "v3_kept_frames": int(o.kept.sum()),
          "v3_kept_failing": int((o.kept & ~o["pass"]).sum()),
          "infants": len(by_infant),
          "infants_with_a_failing_frame": sum(v["failing"] > 0 for v in by_infant.values()),
          "by_class": by_class, "by_device": by_device, "by_infant": by_infant}

    old_reasons = _reference_fail_reasons(o, ref_thr)
    old_pass = ~(old_reasons["empty"] | old_reasons["blur"] | old_reasons["dark"]
                 | old_reasons["bright"] | old_reasons["fov"])
    ref = q[q.source.isin(REFERENCE) & ~q["empty"]]
    superseded = {
        "thresholds": {**ref_thr, "reference_sources": list(REFERENCE),
                       "reference_rows": len(ref),
                       "percentiles": {k: pct for k, (_, pct) in REFERENCE_PCT.items()}},
        **_counts(old_pass, old_reasons, pd.Series(True, index=o.index)),
        "by_class": {c: _counts(old_pass, old_reasons, o.cls6 == c) for c in classes},
        "by_device": {d: _counts(old_pass, old_reasons, o.device == d)
                      for d in sorted(o.device.unique())},
        "infant_classes_with_no_passing_frame":
            int((o.assign(p=old_pass).groupby(["patient_id", "cls6"]).p.sum() == 0).sum()),
    }

    return {"protocol": "PROTOCOL_V2 Amendment 5a, items 1-2, with the 5a revision",
            "pool": {"rows": len(pool), "by_source": {
                s: int(v) for s, v in pool.source.value_counts().sort_index().items()}},
            "near_duplicates": _nd_summary(pool, near),
            "thresholds": {"rule": RULE, "by_device": thr["by_device"],
                           "percentiles": {k: pct for k, (_, pct) in DEVICE_PCT.items()},
                           "work_size": WORK_SIZE},
            "ostrava_quality": oq, "superseded_hospital_rule": superseded}


def _table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def _pct(k, n):
    return f"{100 * k / n:.1f}%" if n else "–"


def _fail_cells(v, reasons=REASONS):
    """Frames, failing, % and the per-reason counts of one class / device / total."""
    return [f"{v['frames']:,}", f"{v['failing']:,}", _pct(v["failing"], v["frames"]),
            *(f"{v['by_reason'][r]:,}" for r in reasons)]


def _yn(b):
    return "yes" if b else "no"


def render_md(s: dict, near: pd.DataFrame, classes: list[str]) -> str:
    nd, th, oq, old = (s["near_duplicates"], s["thresholds"], s["ostrava_quality"],
                       s["superseded_hospital_rule"])
    td, ex, old_ex = nd["true_duplicate"], nd["excluded_from_training"], nd["superseded_exclusion"]
    by_src = ", ".join(f"{s['pool']['by_source'][k]:,} {k}" for k in s["pool"]["by_source"])
    fail_head = ["Frames", "Failing", "%", "Empty", "Blur", "Dark", "Bright"]

    def src_list(d):
        return ", ".join(f"{v:,} {k}" for k, v in d["by_source"].items())

    L = ["# v4 data audit (PROTOCOL_V2 Amendment 5a, items 1–2, with the 5a revision)", "",
         "Generated by `scripts/rop_v4_audit.py`; every number below is computed, none typed. "
         f"Input: the {s['pool']['rows']:,} `train_cv` rows of `splits.csv` ({by_src}). "
         "No Multi-View image and neither test CSV was opened. Paths are repo-relative.", "",
         "## 1. Near-duplicate sweep", "",
         f"Every pool photo was hashed with the 256-bit dHash and all "
         f"{nd['pairs_compared']:,} pairs were compared. **{nd['pairs_total']:,} pairs** lie at "
         f"Hamming distance ≤ {nd['threshold']}. Of these, **{nd['cross_fold']:,} cross CV "
         f"folds**, **{nd['different_label']:,} carry different labels**, and "
         f"**{nd['same_session']:,} lie within one Ostrava exam session**. "
         f"{nd['cross_fold_different_label']:,} pairs both cross folds and disagree on the "
         f"label; {nd['different_infant']:,} pairs join two different infants and "
         f"{nd['cross_source']:,} join two different sources.", "",
         f"dHash alone collides across distinct retinas, so each pair's pixel MAD (mean absolute "
         f"grey-level difference, both images greyscale at {nd['pixel_mad_size']}×"
         f"{nd['pixel_mad_size']} bilinear) was measured; a pair is a **true duplicate** if its "
         f"MAD is ≤ {nd['max_pixel_mad']}. **{td['total']:,} pairs are true duplicates**: "
         f"{td['cross_fold']:,} cross CV folds, {td['different_label']:,} carry different "
         f"labels and {td['same_session']:,} lie within one Ostrava session.", "",
         f"The revised 5a.1 removes both members of every cross-fold true duplicate from "
         f"training rows: **{ex['images']:,} distinct images** ({src_list(ex)}). The first rule "
         f"(every cross-fold pair at distance ≤ 4) would have removed {old_ex['images']:,} "
         f"({src_list(old_ex)}). Validation rows are untouched.", "",
         _table(["Pairs at distance ≤ 4", "All", "True duplicates (MAD ≤ 5)"], [
             ["Total", f"{nd['pairs_total']:,}", f"{td['total']:,}"],
             ["Crossing CV folds", f"{nd['cross_fold']:,}", f"{td['cross_fold']:,}"],
             ["Different labels (clinician list)", f"{nd['different_label']:,}",
              f"{td['different_label']:,}"],
             ["Within one Ostrava session", f"{nd['same_session']:,}",
              f"{td['same_session']:,}"],
             ["Cross-fold and different labels", f"{nd['cross_fold_different_label']:,}", ""],
             ["Different infants", f"{nd['different_infant']:,}", ""],
             ["Different sources", f"{nd['cross_source']:,}", ""]]), "",
         _table(["Pair kind", "Pairs", "True duplicates", "MAD min", "MAD median", "MAD max"],
                [[k, v["pairs"], v["true_duplicate"], f"{v['min']:.2f}", f"{v['median']:.2f}",
                  f"{v['max']:.2f}"] for k, v in nd["pixel_mad_by_kind"].items()]), "",
         _table(["Hamming distance", "Pairs"],
                [[d, f"{v:,}"] for d, v in nd["by_distance"].items()]), "",
         _table(["Source pair", "Pairs"],
                [[k, f"{v:,}"] for k, v in nd["by_source_pair"].items()]), "",
         "## 2. Frame-quality thresholds (per Ostrava device)", "",
         f"`gradability_metrics` (work size {th['work_size']}) ran on all "
         f"{s['pool']['rows']:,} pool photos. {th['rule']} `fov_ratio` stays in `quality.csv` "
         "for the record only: it is a camera constant (the aperture), not a frame property. "
         "ROP-VL and Shenzhen rows are not filtered. `blur_rank_in_device` in `quality.csv` "
         "is the blur_var percentile within the row's source and device.", "",
         _table(["Device", "Frames (non-empty)", "min_blur (p10 blur_var)",
                 "lum_lo (p2.5 luminance)", "lum_hi (p97.5 luminance)"],
                [[d, f"{v['n']:,}", f"{v['min_blur']:.4g}", f"{v['lum_lo']:.4g}",
                  f"{v['lum_hi']:.4g}"] for d, v in th["by_device"].items()]), "",
         "## 3. Ostrava frames failing the per-device rule", "",
         f"**{oq['failing']:,} of {oq['frames']:,} Ostrava frames fail** "
         f"({_pct(oq['failing'], oq['frames'])}), spread over "
         f"{oq['infants_with_a_failing_frame']} of {oq['infants']} infants. Of the "
         f"{oq['v3_kept_frames']:,} Ostrava frames the v3 cap kept for training, "
         f"{oq['v3_kept_failing']:,} fail. Passing frames per class: "
         + "; ".join(f"{c} {v['frames'] - v['failing']:,} of {v['frames']:,}"
                     for c, v in oq["by_class"].items() if v["frames"])
         + f". The pool builder keeps the {FLOOR} best-ranked frames of any (infant, class) "
         f"with fewer than {FLOOR} passing frames; the last column counts those cells. Reasons "
         "overlap, so the reason columns need not sum to the failing count.", "",
         _table(["Class", *fail_head, "Infants", "Infants with no passing frame",
                 f"Infants below the floor of {FLOOR}"],
                [[c, *_fail_cells(v), v["infants"], v["infants_with_no_passing_frame"],
                  v[f"infants_below_floor_{FLOOR}"]] for c, v in oq["by_class"].items()]
                + [["**all**", *_fail_cells(oq), oq["infants"], "", ""]]), "",
         "Per device: " + "; ".join(
             f"{d} {v['failing']:,} of {v['frames']:,} fail ({_pct(v['failing'], v['frames'])})"
             for d, v in oq["by_device"].items()) + ".", "",
         _table(["Device", *fail_head], [[d, *_fail_cells(v)]
                                         for d, v in oq["by_device"].items()]), "",
         "Failing / frames per device and class:", "",
         _table(["Device", *classes],
                [[d, *(f"{v['by_class'][c]['failing']:,} / {v['by_class'][c]['frames']:,}"
                       if c in v["by_class"] else "–" for c in classes)]
                 for d, v in oq["by_device"].items()]), "",
         "Per infant (every Ostrava infant in the pool):", "",
         _table(["Infant", "Device", "Classes", "Frames", "Failing", "%",
                 f"Classes below the floor of {FLOOR}"],
                [[pid, ", ".join(v["devices"]), ", ".join(v["classes"]), v["frames"],
                  v["failing"], _pct(v["failing"], v["frames"]),
                  ", ".join(v[f"classes_below_floor_{FLOOR}"]) or "–"]
                 for pid, v in oq["by_infant"].items()]), "",
         "### Superseded rule (for the record)", "",
         "The first 5a.2 rule took its thresholds from the "
         f"{old['thresholds']['reference_rows']:,} non-empty ROP-VL and Shenzhen photos and "
         "added a field-of-view criterion. It is replaced because it measured the camera: "
         f"**{old['failing']:,} of {old['frames']:,} Ostrava frames failed** "
         f"({_pct(old['failing'], old['frames'])}), "
         + "; ".join(f"{d} {v['failing']:,} of {v['frames']:,} ({v['by_reason']['fov']:,} "
                     "below min_fov)" for d, v in old["by_device"].items())
         + f", and {old['infant_classes_with_no_passing_frame']} (infant, class) cells had no "
         "passing frame.", "",
         _table(["Threshold", "Metric", "Percentile", "Value"], [
             [k, REFERENCE_PCT[k][0], f"{REFERENCE_PCT[k][1]:g}th",
              f"{old['thresholds'][k]:.4g}"] for k in REFERENCE_PCT]), "",
         _table(["Class", *fail_head, "FOV"],
                [[c, *_fail_cells(v, OLD_REASONS)] for c, v in old["by_class"].items()]
                + [["**all**", *_fail_cells(old, OLD_REASONS)]]), "",
         _table(["Device", *fail_head, "FOV"],
                [[d, *_fail_cells(v, OLD_REASONS)] for d, v in old["by_device"].items()]), "",
         "## 4. Clinician list: near-duplicate pairs with different labels", "",
         f"{nd['different_label']:,} pairs at distance ≤ {nd['threshold']}, of which "
         f"{td['different_label']:,} are true duplicates (pixel MAD ≤ {nd['max_pixel_mad']}). "
         "All stay on the list. Nothing is relabelled (Amendment 5a.1).", ""]
    dl = near[~near.same_label]
    if len(dl):
        L.append(_table(["#", "image_a", "cls6_a", "image_b", "cls6_b", "Distance",
                         "Pixel MAD", "True duplicate", "Cross-fold", "Same session"],
                        [[i + 1, f"`{r.image_a}`", r.cls6_a, f"`{r.image_b}`", r.cls6_b,
                          r.distance, f"{r.pixel_mad:.2f}", _yn(r.true_duplicate),
                          _yn(r.cross_fold), _yn(r.same_session)]
                         for i, r in enumerate(dl.itertuples(index=False))]))
    else:
        L.append("None.")
    return "\n".join(L) + "\n"


def write_audit(out_dir, pool: pd.DataFrame | None = None) -> dict:
    pool = (load_pool() if pool is None else pool).reset_index(drop=True)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    classes = json.loads(CLASS_INDEX.read_text())["classes"]

    print(f"near-duplicate sweep over {len(pool):,} pool photos "
          f"(dHash-256, distance <= {DHASH_THR}, pixel MAD <= {MAX_MAD})", flush=True)
    near = pair_table(pool, near_duplicate_pairs(pool.image_path.tolist(), DHASH_THR))
    near = confirm_duplicates(near)
    print(f"frame quality (gradability_metrics, work size {WORK_SIZE})", flush=True)
    q = quality_table(pool)
    ref_thr = _reference_thresholds(q)
    thr = {"rule": RULE, "by_device": thresholds_by_device(q),
           "reference_single_photo_hospitals": {
               **ref_thr, "status": "superseded by the Amendment 5a revision; record only",
               "sources": list(REFERENCE),
               "percentiles": {k: pct for k, (_, pct) in REFERENCE_PCT.items()}}}
    summary = summarise(pool, near, q, thr, ref_thr, classes)

    near.to_csv(out / "near_duplicates.csv", index=False)
    q.to_csv(out / "quality.csv", index=False)
    (out / "thresholds.json").write_text(json.dumps(thr, indent=2) + "\n")
    (out / "audit_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (out / "AUDIT.md").write_text(render_md(summary, near, classes))
    return summary


def main():
    assert data_quality._HAS_CV2, "gradability_metrics only resizes to the work size with cv2"
    pool = load_pool()
    assert len(pool) and not pool.image_path.str.contains("multiview").any()
    s = write_audit(OUT_DIR, pool)
    nd, oq = s["near_duplicates"], s["ostrava_quality"]
    td = nd["true_duplicate"]
    print(f"pairs {nd['pairs_total']:,} | cross-fold {nd['cross_fold']:,} | "
          f"different-label {nd['different_label']:,} | same-session {nd['same_session']:,}")
    print(f"true duplicates {td['total']:,} | cross-fold {td['cross_fold']:,} | "
          f"different-label {td['different_label']:,} | excluded from training "
          f"{nd['excluded_from_training']['images']:,}")
    for d, v in s["thresholds"]["by_device"].items():
        print(f"thresholds {d}:", {k: round(x, 4) for k, x in v.items() if k != "n"})
    print(f"Ostrava failing {oq['failing']:,}/{oq['frames']:,} by class:",
          {c: v["failing"] for c, v in oq["by_class"].items()})
    print("by device:", {d: v["failing"] for d, v in oq["by_device"].items()})
    print(f"wrote {OUT_DIR.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
