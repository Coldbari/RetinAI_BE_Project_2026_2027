"""T15 — the ROP operating characteristic, instead of one point presented as inevitable.

The deployed threshold 0.1933 gives sensitivity 0.990 / specificity 0.398: about 60% of
healthy infants are flagged. That is a real cost, and the end-to-end test reproduced it
almost exactly (6 of 9 healthy babies falsely flagged).

But 0.398 is not a property of the model — it is the consequence of ONE choice, made by
targeting sensitivity >= 0.90 on validation. Presenting a single point invites the reader to
treat it as fixed. It is not: it is a dial, and the person who should turn it is a
neonatologist who knows what a missed ROP case costs versus what an unnecessary referral
costs in their unit.

So this reports the whole curve, with:
  * exact (Clopper-Pearson) intervals at every candidate point, because 909 val images come
    from few infants and the intervals are wide
  * the threshold DERIVED on val and the sens/spec it then achieves on the untouched TEST
    split — derived once, applied once, no peeking
  * a concrete translation: per 100 screened infants at this prevalence, how many missed
    cases and how many unnecessary referrals

Both splits are the deployed model's own (tta_val_preds.npz / tta_test_preds.npz), scored
through the real 5-view serving path.

    .venv/bin/python scripts/rop_operating_curve.py
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import beta

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

TARGETS = [0.999, 0.99, 0.98, 0.95, 0.90, 0.85, 0.80]
PREVALENCE = 0.10   # ~10% of screened preterm infants develop treatment-relevant ROP


def cp(k, n, alpha=0.05):
    if n == 0:
        return (float("nan"), float("nan"))
    lo = beta.ppf(alpha / 2, k, n - k + 1) if k > 0 else 0.0
    hi = beta.ppf(1 - alpha / 2, k + 1, n - k) if k < n else 1.0
    return float(lo), float(hi)


def at(thr, y, s):
    pred = s >= thr
    tp = int((pred & (y == 1)).sum()); fn = int((~pred & (y == 1)).sum())
    tn = int((~pred & (y == 0)).sum()); fp = int((pred & (y == 0)).sum())
    sens = tp / max(tp + fn, 1); spec = tn / max(tn + fp, 1)
    return {"threshold": float(thr), "sens": sens, "sens_ci": cp(tp, tp + fn),
            "spec": spec, "spec_ci": cp(tn, tn + fp),
            "tp": tp, "fn": fn, "tn": tn, "fp": fp}


def load(name):
    z = np.load(ROOT / "results" / "rop" / name, allow_pickle=True)
    return z["prob"][:, 1], z["y_true"].astype(int)


def main():
    sv, yv = load("tta_val_preds.npz")
    st, yt = load("tta_test_preds.npz")
    print(f"val  n={len(yv)}  ROP prevalence {yv.mean():.3f}")
    print(f"test n={len(yt)}  ROP prevalence {yt.mean():.3f}")

    print("\n" + "=" * 104)
    print("THRESHOLD DERIVED ON VAL AT EACH TARGET SENSITIVITY, THEN APPLIED ONCE TO TEST")
    print("=" * 104)
    print(f"{'target':>7}{'thr':>9}   {'TEST sens':<24}{'TEST spec':<24}"
          f"{'missed/100':>11}{'false alarms/100':>18}")

    rows = []
    order = np.argsort(sv)
    for tgt in TARGETS:
        # smallest threshold on val whose sensitivity still meets the target
        best = None
        for thr in np.unique(np.concatenate([[0.0], sv[order]])):
            r = at(thr, yv, sv)
            if r["sens"] >= tgt:
                best = thr                       # keep raising while the target still holds
        if best is None:
            continue
        te = at(best, yt, st)
        # per 100 screened infants at realistic prevalence
        missed = (1 - te["sens"]) * PREVALENCE * 100
        false_alarms = (1 - te["spec"]) * (1 - PREVALENCE) * 100
        rows.append({"target_sensitivity": tgt, "val_threshold": float(best), "test": te,
                     "per100_missed": missed, "per100_false_alarms": false_alarms})
        print(f"{tgt:>7.3f}{best:>9.4f}   "
              f"{te['sens']:.3f} [{te['sens_ci'][0]:.3f},{te['sens_ci'][1]:.3f}]   "
              f"{te['spec']:.3f} [{te['spec_ci'][0]:.3f},{te['spec_ci'][1]:.3f}]   "
              f"{missed:>10.1f}{false_alarms:>18.1f}")

    dep = at(0.1933, yt, st)
    print(f"\nDEPLOYED threshold 0.1933 on test: "
          f"sens {dep['sens']:.3f} [{dep['sens_ci'][0]:.3f}, {dep['sens_ci'][1]:.3f}]   "
          f"spec {dep['spec']:.3f} [{dep['spec_ci'][0]:.3f}, {dep['spec_ci'][1]:.3f}]")

    print("\nHOW TO READ THE LAST TWO COLUMNS")
    print(f"  Assuming {PREVALENCE:.0%} of screened infants have treatment-relevant ROP, out of")
    print("  every 100 babies screened: how many cases the model MISSES, and how many healthy")
    print("  babies get sent for an unnecessary exam. Those are the two harms, in the same units.")
    print("\n  The deployed setting is deliberately at the extreme sensitivity end. For a")
    print("  condition that blinds and is treatable, missing a case is far worse than an extra")
    print("  exam — but this is a clinical judgement, not a modelling one, and the table is here")
    print("  so a neonatologist can make it rather than inherit it.")
    print("\n  Every interval is wide because 909 val / 1502 test images come from 29 infants.")
    print("  The dial is real; the precision of any single setting is not.")

    dst = ROOT / "results" / "rop" / "operating_curve.json"
    dst.write_text(json.dumps({"prevalence_assumed": PREVALENCE, "points": rows,
                               "deployed": dep,
                               "note": "thresholds derived on val, applied once to test; "
                                       "Clopper-Pearson intervals; images cluster within "
                                       "infants so these are the optimistic end"}, indent=2))
    print(f"\n-> {dst}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
