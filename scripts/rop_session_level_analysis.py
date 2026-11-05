"""Session-level AP-ROP evaluation: the clinically-honest read of the AP-ROP branch.

    .venv/bin/python scripts/rop_session_level_analysis.py <val_predictions.npz> [fold_csv]

WHY. Ostrava's AP-ROP labels are per-SESSION diagnosis codes stamped onto every frame of an
imaging series (median 18, up to 58 frames per session in fold-0 val). Many frames in a
confirmed AP-ROP session are peripheral or poor views that do not show the disease — but
they carry its label. Image-level recall therefore has a structural ceiling that no model
can cross, which is exactly what the five-backbone benchmark showed: every architecture
stuck at 0.16-0.22.

Screening decisions are made per eye per visit, not per frame: the clinical question is
"did any frame of this session show the aggressive form". So the honest evaluation
aggregates by session (max of the branch probability over the session's frames) — the
standard screening read — and reports the false-alarm side under the SAME aggregation,
because pooling that multiplied false positives on healthy sessions would be a trick, not
a result.

Measured on the structured model's fold-0 val predictions (threshold 0.5, fixed a priori):

    image level    0.311      (476 images)
    session level  0.905      (21 sessions; 19 detected)
    patient level  0.833      (6 infants; 5 detected)
    normal false-positive rate: 0.000 at BOTH levels

and the label-noise signature: the median AP-ROP session has 47% of its frames scored
below p=0.1 — half the frames carry the label but (per the model) not the appearance.

CAVEATS, always reported with the numbers: fold-0 val holds only 21 AP-ROP sessions from
6 infants, so the session-level estimate is 19/21 with a wide exact CI; this is evidence
for the evaluation-unit argument, not a precise recall claim. Pool across all five folds
before headlining.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent


def session_key(row) -> str:
    """Ostrava filenames encode device+serie (…/183_D2_S01_17.jpg) — one imaging session.
    Other sources have no session structure; fall back to the patient."""
    m = re.search(r"/(\d+)_D(\d)_S(\d+)_\d+\.jpg$", row.image_path)
    return f"{row.patient_id}|D{m.group(2)}S{m.group(3)}" if m else row.patient_id


def exact_ci(k: int, n: int, alpha: float = 0.05):
    """Clopper-Pearson, so tiny denominators are not dressed up as precision."""
    from scipy.stats import beta
    lo = beta.ppf(alpha / 2, k, n - k + 1) if k > 0 else 0.0
    hi = beta.ppf(1 - alpha / 2, k + 1, n - k) if k < n else 1.0
    return float(lo), float(hi)


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 1
    npz = np.load(sys.argv[1])
    fold_csv = sys.argv[2] if len(sys.argv) > 2 else \
        ROOT / "results" / "prepared" / "rop_staging" / "fold_0.csv"

    val = pd.read_csv(fold_csv)
    val = val[val.split == "val"].reset_index(drop=True)
    yt, p6 = npz["y_true"], npz["prob6"]
    assert len(val) == len(yt) and (val.label.values == yt).all(), \
        "prediction rows do not align with the manifest — refusing to analyse"

    val["p_arop"] = p6[:, 5]
    val["session"] = val.apply(session_key, axis=1)

    ar = val[val.cls6 == "arop"]
    no = val[val.cls6 == "normal"]
    sess = ar.groupby("session").p_arop.max()
    pat = ar.groupby("patient_id").p_arop.max()
    no_sess = no.groupby("session").p_arop.max()

    k_img, n_img = int((ar.p_arop > 0.5).sum()), len(ar)
    k_s, n_s = int((sess > 0.5).sum()), len(sess)
    k_p, n_p = int((pat > 0.5).sum()), len(pat)
    frac_low = ar.groupby("session").p_arop.apply(lambda g: float((g < 0.1).mean()))

    out = {
        "threshold": 0.5,
        "image_level":   {"recall": k_img / n_img, "n": n_img, "ci95": exact_ci(k_img, n_img)},
        "session_level": {"recall": k_s / n_s, "n": n_s, "detected": k_s,
                          "ci95": exact_ci(k_s, n_s)},
        "patient_level": {"recall": k_p / n_p, "n": n_p, "ci95": exact_ci(k_p, n_p)},
        "normal_fp_rate": {"image": float((no.p_arop > 0.5).mean()),
                           "session": float((no_sess > 0.5).mean())},
        "median_frac_frames_below_p0.1_in_arop_sessions": float(frac_low.median()),
        "frames_per_arop_session": {"median": float(ar.groupby('session').size().median()),
                                    "max": int(ar.groupby('session').size().max())},
    }
    for lvl in ("image_level", "session_level", "patient_level"):
        r, (lo, hi) = out[lvl]["recall"], out[lvl]["ci95"]
        print(f"  AP-ROP recall, {lvl:<14} {r:.3f}   (n={out[lvl]['n']}, 95% CI {lo:.2f}-{hi:.2f})")
    print(f"  normal FP rate: image {out['normal_fp_rate']['image']:.3f}, "
          f"session {out['normal_fp_rate']['session']:.3f}")
    print(f"  median fraction of AP-ROP-session frames at p<0.1: "
          f"{out['median_frac_frames_below_p0.1_in_arop_sessions']:.2f}")

    dest = ROOT / "results" / "rop" / "session_level_arop.json"
    dest.write_text(json.dumps(out, indent=2) + "\n")
    print(f"-> {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
