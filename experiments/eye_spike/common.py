"""Shared helpers for the eye/examination-level spike (throwaway; see RESULTS.md).

Never reads test_locked. Never writes filenames, exam dates or patient ids to outputs:
bags and patients are written as salted-HMAC hashes; the salt lives in cache/ (untracked).
"""
from __future__ import annotations

import hashlib
import hmac
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
CACHE = HERE / "cache"
PREP = ROOT / "results" / "prepared" / "rop_staging"
CLASSES = ["normal", "stage_1", "stage_2", "stage_3", "stage_4_5", "arop"]
sys.path.insert(0, str(ROOT))


def _salt() -> bytes:
    CACHE.mkdir(parents=True, exist_ok=True)
    f = CACHE / "salt.bin"
    if not f.exists():
        f.write_bytes(os.urandom(32))
    return f.read_bytes()


def opaque(key: str, n: int = 12) -> str:
    return hmac.new(_salt(), key.encode(), hashlib.sha256).hexdigest()[:n]


def load_pool() -> pd.DataFrame:
    """The 6,700-image CV pool = fold_k val rows, in fold order then committed row order.
    `row_in_fold` is the row index inside fold_k's val rows = row index in the committed
    structured_5fold/fold_k.npz."""
    parts = []
    for k in range(5):
        d = pd.read_csv(PREP / f"fold_{k}.csv")
        d = d[d.split == "val"].reset_index(drop=True)
        d["fold"] = k
        d["row_in_fold"] = np.arange(len(d))
        parts.append(d)
    pool = pd.concat(parts, ignore_index=True)
    assert len(pool) == 6700 and pool.image_path.is_unique
    return pool


def load_dev() -> pd.DataFrame:
    """Multi-View DEV half only, with the repo's own eye definition (rop_locked_eval.eye_frame)."""
    from scripts.rop_locked_eval import eye_frame
    df = pd.read_csv(PREP / "test_dev.csv").reset_index(drop=True)
    e = eye_frame(df)
    assert len(e) == len(df) and (e.image_path.values == df.image_path.values).all()
    df["eye"] = e.eye.values
    df["fold"] = -1
    return df


_OST = re.compile(r"/(\d+)_D(\d)_S(\d+)_(\d+)\.jpg$")
_RVL = re.compile(r"/(\d+)_(\d{4}-\d{2}-\d{2})_([LR])_(\d+)\.jpg$")


def bag_key(row, sz_eye: dict) -> tuple[str, int]:
    """(raw bag key — NEVER written out, frame order int for deterministic sub-sampling)."""
    p = row.image_path
    if row.source == "ostrava":
        m = _OST.search(p)
        return f"ostrava|{row.patient_id}|D{m.group(2)}|S{m.group(3)}", int(m.group(4))
    if row.source == "ropvl":
        m = _RVL.search(p)
        return f"ropvl|{row.patient_id}|{m.group(2)}|{m.group(3)}", int(m.group(4))
    if row.source == "shenzhen":
        name = p.rsplit("/", 1)[1]
        eye = sz_eye[name]
        num = int(re.search(r"(\d+)\.jpg$", name).group(1))
        return f"shenzhen|{row.patient_id}|{eye}", num
    if row.source == "multiview":
        num = int(re.search(r"\.(\d+)\.jpg$", p).group(1))
        return f"multiview|{row.patient_id}|{row.eye}", num
    raise ValueError(row.source)


# ── metrics ──────────────────────────────────────────────────────────────────
def bag_metrics(y, p) -> dict:
    from sklearn.metrics import cohen_kappa_score, f1_score
    y, p = np.asarray(y), np.asarray(p)
    present = sorted(set(y.tolist()))
    return {
        "n": int(len(y)),
        "qwk": float(cohen_kappa_score(y, p, weights="quadratic", labels=list(range(6)))),
        "macro_f1": float(f1_score(y, p, average="macro", labels=present, zero_division=0)),
        "acc": float((y == p).mean()),
        "recall": {CLASSES[c]: float((p[y == c] == c).mean()) for c in present},
        "n_per_class": {CLASSES[c]: int((y == c).sum()) for c in present},
    }


def qwk(y, p):
    from sklearn.metrics import cohen_kappa_score
    return float(cohen_kappa_score(y, p, weights="quadratic", labels=list(range(6))))


def mf1(y, p):
    from sklearn.metrics import f1_score
    return float(f1_score(y, p, average="macro", labels=sorted(set(np.asarray(y).tolist())),
                          zero_division=0))


def paired_cluster_boot(y, PA, PB, groups, fn, n=1000, seed=0):
    """Patient-clustered paired bootstrap of mean_s fn(y, PA[s]) - mean_s fn(y, PB[s]).
    PA / PB are lists of prediction vectors (seeds or checkpoints), averaged inside every
    replicate. Bags (rows) are resampled as whole patients."""
    rng = np.random.default_rng(seed)
    y, groups = np.asarray(y), np.asarray(groups)
    uniq = np.unique(groups)
    idx = {g: np.where(groups == g)[0] for g in uniq}
    F = lambda yy, Ps, t: float(np.mean([fn(yy[t], p[t]) for p in Ps]))  # noqa: E731
    da, db, dd = [], [], []
    for _ in range(n):
        t = np.concatenate([idx[g] for g in rng.choice(uniq, len(uniq), True)])
        a, b = F(y, PA, t), F(y, PB, t)
        da.append(a); db.append(b); dd.append(a - b)
    allr = np.arange(len(y))
    a0, b0 = F(y, PA, allr), F(y, PB, allr)
    q = lambda v: [round(float(x), 4) for x in np.percentile(v, [2.5, 97.5])]  # noqa: E731
    return {"a": round(a0, 4), "a_ci": q(da), "b": round(b0, 4), "b_ci": q(db),
            "diff": round(a0 - b0, 4), "diff_ci": q(dd),
            "frac_reps_diff_le_0": round(float(np.mean(np.array(dd) <= 0)), 4),
            "n_bags": int(len(y)), "n_patients": int(len(uniq)), "reps": n}
