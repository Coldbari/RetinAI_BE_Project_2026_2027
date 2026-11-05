"""Session-level AP-ROP scoring for every fold-0 ablation arm.

    .venv/bin/python scripts/rop_ablation_session_level.py

The component table (ablations_fold0.json) showed every arm hitting the same image-level
AP-ROP ceiling, and we argued the branch's value should appear at the session level — the
unit the label actually describes. This script measures that instead of arguing it: for
each arm's committed fold-0 predictions, aggregate the AP-ROP probability per imaging
session (max over frames, threshold 0.5 fixed a priori — same protocol as
rop_session_level_analysis.py) and report session recall plus the false-positive rate on
normal sessions.

Caveat that travels with every number: fold-0 val holds 21 AP-ROP sessions from 6 infants.
These are comparisons on a small, shared denominator — direction is meaningful, precision
is not.

Output: results/rop/ablation_session_level.json
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
ARMS = ROOT / "results" / "rop" / "staging_runs" / "ablations"
FOLD0 = ROOT / "results" / "prepared" / "rop_staging" / "fold_0.csv"
MAIN = ROOT / "results" / "rop" / "staging_runs" / "structured_5fold" / "fold_0.npz"
OUT = ROOT / "results" / "rop" / "ablation_session_level.json"

AROP = 5
THR = 0.5


def session_key(row) -> str:
    """Ostrava filenames encode device+serie (…/183_D2_S01_17.jpg) = one imaging session.

    Other sources have no session structure, so they fall back to the PATIENT — matching
    scripts/rop_session_level_analysis.py. An earlier version of this file fell back to the
    filename, which made every ROP-VL image its own "session" and inflated the unit count
    (fold 0: 22 units instead of 21). Session numbers published before 2026-08-14 came from
    that broken key.
    """
    m = re.search(r"/(\d+)_D(\d)_S(\d+)_\d+\.jpg$", row.image_path)
    return f"{m.group(1)}_D{m.group(2)}_S{m.group(3)}" if m else "pat:" + row.patient_id


def exact_ci(k, n, alpha=0.05):
    from scipy.stats import beta
    lo = beta.ppf(alpha / 2, k, n - k + 1) if k > 0 else 0.0
    hi = beta.ppf(1 - alpha / 2, k + 1, n - k) if k < n else 1.0
    return round(float(lo), 3), round(float(hi), 3)


def score(npz_path, val):
    d = np.load(npz_path)
    key = "prob6" if "prob6" in d else "prob"
    if len(d["y_true"]) != len(val) or not (val.label.values == d["y_true"]).all():
        raise SystemExit(f"{npz_path.name}: rows do not align with fold_0 val")
    df = val.copy()
    df["p_arop"] = d[key][:, AROP]
    df["pred"] = d["y_pred"]
    ar, no = df[df.label == AROP], df[df.label != AROP]
    sess = ar.groupby("session").p_arop.max()
    no_sess = no.groupby("session").p_arop.max()
    k, n = int((sess > THR).sum()), len(sess)
    img_recall = float((ar.pred == AROP).mean())
    # DECOMPOSITION. The image -> session jump mixes two distinct effects, and the audit
    # showed re-weighting is the larger one: a single Ostrava infant supplies 76% of AP-ROP
    # images, so giving each unit equal weight moves the number even with NO aggregation.
    # reweight_only = mean over units of that unit's own image-level recall.
    reweight_only = float(ar.groupby("session").p_arop.apply(lambda g: (g > THR).mean()).mean())
    return {
        "image_recall": round(img_recall, 3),
        "reweight_only_recall": round(reweight_only, 3),
        "session_recall": round(k / n, 3),
        "sessions_detected": f"{k}/{n}",
        "session_ci95": exact_ci(k, n),
        "aggregation_effect": round(k / n - reweight_only, 3),
        "reweighting_effect": round(reweight_only - img_recall, 3),
        "normal_session_fp": round(float((no_sess > THR).mean()), 4),
        "n_normal_sessions": int(len(no_sess)),
    }


def main():
    val = pd.read_csv(FOLD0).query("split == 'val'").reset_index(drop=True)
    val["session"] = val.apply(session_key, axis=1)

    arms = {"full": MAIN}
    for f in sorted(ARMS.glob("*_fold0.npz")):
        arms[f.stem.replace("_fold0", "")] = f

    out = {"protocol": "max AP-ROP probability over each session's frames, threshold 0.5 "
                       "fixed a priori; fold-0 val only — direction meaningful, precision not. "
                       "Ostrava keyed by device+serie; other sources fall back to patient.",
           "decomposition_note":
               "image_recall -> reweight_only_recall -> session_recall separates the two "
               "effects that the raw image->session jump conflates. reweighting_effect is "
               "what you get from weighting each unit equally (one Ostrava infant supplies "
               "76% of AP-ROP images); aggregation_effect is what max-over-frames adds on "
               "top. Report both — the pooled 5-fold split is +0.348 reweighting vs +0.147 "
               "aggregation, i.e. ~70% of the headline jump is population, not unit.",
           "arms": {}}
    for name, path in arms.items():
        out["arms"][name] = score(path, val)
        a = out["arms"][name]
        print(f"  {name:<12} image {a['image_recall']:.3f} -> reweight "
              f"{a['reweight_only_recall']:.3f} -> session {a['session_recall']:.3f} "
              f"({a['sessions_detected']}, CI {a['session_ci95']})  "
              f"normal-FP {a['normal_session_fp']:.4f}")

    OUT.write_text(json.dumps(out, indent=2) + "\n")
    print(f"-> {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
