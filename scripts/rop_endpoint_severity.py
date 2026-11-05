"""What does the 0.90-sensitivity floor actually protect — and would a treatment-relevant
endpoint choose a different operating point?

    .venv/bin/python scripts/rop_endpoint_severity.py

The served threshold (0.0155) was selected as: largest threshold with sensitivity >= 0.90
for ANY ROP at both sites. But "any ROP" includes stage 1, which regresses spontaneously
and needs no treatment — so the floor may be spending specificity on the endpoint that
matters least. This measures, at the SERVED threshold, per-stage sensitivity at both sites,
and then re-runs the identical selection rule with the floor applied to a severe-ROP proxy
instead (stage >= 3: stage 3, stage 4/5, AP-ROP — a PROXY: the labels carry no zone or
plus-disease, which the real Type 1 / treatment criteria are built from).

This script changes nothing about the app. It writes the numbers the endpoint decision
needs into results/rop/endpoint_severity.json.
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results/rop/endpoint_severity.json"
SERVED = 0.0155
FLOOR = 0.90
GRID = np.round(np.arange(0.0005, 0.2000, 0.0005), 4)
SEVERE = 3            # label >= 3  <=>  stage_3, stage_4_5, arop
NAMES = {1: "stage_1", 2: "stage_2", 3: "stage_3", 4: "stage_4_5", 5: "arop"}


def _internal():
    y6, s = [], []
    for f in sorted(glob.glob(str(ROOT / "results/rop/staging_runs/structured_5fold/fold_[0-4].npz"))):
        z = np.load(f, allow_pickle=True)
        y6.append(z["y_true"])
        s.append(z["prob6"][:, 1:].sum(1))
    assert len(y6) == 5
    return np.concatenate(y6), np.concatenate(s)


def _external():
    z = np.load(ROOT / "results/rop/screening_external_spec.npz", allow_pickle=True)
    dev = pd.read_csv(ROOT / "results/prepared/rop_staging/test_dev.csv")
    lab = dict(zip(dev.image_path, dev.label))
    y6 = np.array([lab[p] for p in z["image_path"]])
    assert ((y6 > 0).astype(int) == z["y"]).all(), "labels must agree with the spec's binary y"
    return y6, z["staging_fold0"]


def _profile(y6, s, thr):
    flag = s > thr
    out = {"threshold": float(thr),
           "specificity": round(float((~flag[y6 == 0]).mean()), 4),
           "healthy_flagged": f"{int(flag[y6 == 0].sum())}/{int((y6 == 0).sum())}"}
    for v, name in NAMES.items():
        n = int((y6 == v).sum())
        out[f"sens_{name}"] = round(float(flag[y6 == v].mean()), 4) if n else None
    out["sens_any_rop"] = round(float(flag[y6 > 0].mean()), 4)
    out["sens_severe_stage3plus"] = round(float(flag[y6 >= SEVERE].mean()), 4)
    out["sens_mild_stage12"] = round(float(flag[(y6 > 0) & (y6 < SEVERE)].mean()), 4)
    return out


def _select(yi6, si, ye6, se, positive):
    """The rebase's selection rule verbatim, with a pluggable definition of positive."""
    best = None
    for t in GRID:
        if (si[positive(yi6)] > t).mean() >= FLOOR and (se[positive(ye6)] > t).mean() >= FLOOR:
            best = float(t)
    return best


def main() -> None:
    yi6, si = _internal()
    ye6, se = _external()

    severe_thr = _select(yi6, si, ye6, se, lambda y: y >= SEVERE)
    res = {
        "question": ("The served floor guarantees sens>=0.90 on ANY ROP, stage 1 included. "
                     "Would a floor on severe ROP (stage>=3 proxy) buy meaningful "
                     "specificity, and what mild-ROP sensitivity would it cost?"),
        "severe_proxy": ("label >= 3 (stage_3, stage_4_5, arop). A PROXY ONLY: no zone or "
                         "plus-disease labels exist in this data, and real treatment "
                         "criteria (ETROP Type 1) require both."),
        "served_threshold_profile": {
            "internal_oof_n%d" % len(yi6): _profile(yi6, si, SERVED),
            "external_dev_n%d" % len(ye6): _profile(ye6, se, SERVED),
        },
        "counterfactual_severe_floor": None,
        "caveats": [
            "external dev half was used to select BOTH thresholds -> its numbers are in-sample",
            "class counts at the external site are small for stage_4_5 (14) and arop (6)",
            "nothing in the served app changes unless the endpoint decision is made explicitly",
        ],
    }
    if severe_thr is not None:
        res["counterfactual_severe_floor"] = {
            "selection_rule": "identical rule, floor applied to stage>=3 only",
            "threshold": severe_thr,
            "internal_oof": _profile(yi6, si, severe_thr),
            "external_dev": _profile(ye6, se, severe_thr),
        }
    OUT.write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))
    print(f"\nwrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
