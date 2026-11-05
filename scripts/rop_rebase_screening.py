"""Re-base the binary ROP screening decision onto the structured staging model.

WHY. The served ResNet50 screening head was audited against a hospital it never trained on
(scripts/rop_screening_external_spec.py) and found degenerate there: at its 0.1933 threshold
it flagged 663 of 663 images, every healthy eye included, because the lowest score anywhere
on that site (0.2788) sits ABOVE the cut-off. External AUC 0.691. On the same images the
structured staging model's any-ROP probability scored AUC 0.821. The better detector was
already in the app, labelled "research preview".

THE SELECTION RULE, fixed before looking at the numbers:

    Take the LARGEST threshold at which sensitivity >= 0.90 holds on BOTH the training
    hospitals (5-fold out-of-fold) AND the held-out hospital (dev half).

Largest, because specificity rises with the threshold and the false-alarm rate is the
problem being fixed. Both sites, because a threshold that satisfies only one is exactly the
failure being replaced — the old cut-off met its target on Ostrava and collapsed elsewhere.
0.90 because that is the sensitivity target this project pre-registered for screening.

HONESTY CAVEAT, recorded in the output and shown in the app: the held-out hospital's DEV
half is used for SELECTION, so the external numbers reported here are in-sample for that
choice and should be read as optimistic. The locked half is not touched — it was opened
once under a pre-registered protocol and stays closed. A clean confirmation needs a site
neither half has seen.

Writes results/rop/screening_rebase.json.

Run:  .venv/bin/python scripts/rop_rebase_screening.py
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

OUT = ROOT / "results/rop/screening_rebase.json"
SENS_FLOOR = 0.90
GRID = np.round(np.arange(0.0005, 0.2000, 0.0005), 4)


def _internal() -> tuple[np.ndarray, np.ndarray]:
    """Out-of-fold any-ROP scores across the 5 training folds."""
    ys, ss = [], []
    for f in sorted(glob.glob(str(ROOT / "results/rop/staging_runs/structured_5fold/fold_[0-4].npz"))):
        z = np.load(f, allow_pickle=True)
        ys.append((z["y_true"] > 0).astype(int))
        ss.append(z["prob6"][:, 1:].sum(1))
    assert len(ys) == 5, f"expected 5 folds, found {len(ys)}"
    return np.concatenate(ys), np.concatenate(ss)


def _external() -> tuple[np.ndarray, np.ndarray]:
    z = np.load(ROOT / "results/rop/screening_external_spec.npz", allow_pickle=True)
    return z["y"], z["staging_fold0"]


def _cell(y: np.ndarray, s: np.ndarray, t: float) -> dict:
    pos, neg = y == 1, y == 0
    tp, fp = int((s[pos] > t).sum()), int((s[neg] > t).sum())
    return {"sensitivity": round(float((s[pos] > t).mean()), 4),
            "specificity": round(float((s[neg] <= t).mean()), 4),
            "healthy_flagged": fp, "healthy_total": int(neg.sum()),
            "ppv": round(tp / max(tp + fp, 1), 4), "n": int(len(y))}


def main() -> None:
    from sklearn.metrics import roc_auc_score

    yi, si = _internal()
    ye, se = _external()

    ok = [t for t in GRID
          if (si[yi == 1] > t).mean() >= SENS_FLOOR and (se[ye == 1] > t).mean() >= SENS_FLOOR]
    if not ok:
        raise SystemExit(f"No threshold reaches {SENS_FLOOR} sensitivity at both sites.")
    thr = float(max(ok))

    # what the model being replaced does at the same two sites
    old = json.loads((ROOT / "results/rop/screening_external_spec.json").read_text())
    old_op = json.loads((ROOT / "results/rop/operating_point.json").read_text())

    res = {
        "decision": "binary ROP screening moves from the ResNet50 head to the structured "
                    "staging model's any-ROP probability",
        "selection_rule": (f"largest threshold with sensitivity >= {SENS_FLOOR} at BOTH the "
                           "training hospitals (5-fold out-of-fold) and the held-out "
                           "hospital (dev half)"),
        "threshold": thr,
        "sensitivity_floor": SENS_FLOOR,
        "new_model": {
            "internal_oof": _cell(yi, si, thr),
            "external_dev": _cell(ye, se, thr),
            "auc_internal_oof": round(float(roc_auc_score(yi, si)), 4),
            "auc_external_dev": round(float(roc_auc_score(ye, se)), 4),
        },
        "retired_model": {
            "threshold": old_op["threshold"],
            "internal_test": {"sensitivity": round(old_op["test"]["sensitivity"], 4),
                              "specificity": round(old_op["test"]["specificity"], 4),
                              "n": old_op["test"]["n"]},
            "external_dev": old["served_operating_point"]["screening"],
            "auc_external_dev": old["auc"]["screening"],
        },
        "caveats": [
            "The external DEV half was used to SELECT this threshold, so its numbers here "
            "are in-sample for that choice and are optimistic.",
            "The locked half is untouched and stays closed; a clean confirmation needs a "
            "site neither half has seen.",
            "Sensitivity is traded down (internal 0.990 -> {:.3f}) to buy specificity that "
            "exists outside the training hospital at all.".format(
                _cell(yi, si, thr)["sensitivity"]),
            "'Any ROP' includes stage 1, which does not itself require treatment.",
        ],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))
    print(f"\nwrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
