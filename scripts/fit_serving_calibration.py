"""Phase 2 (C1/C2/C3) — fit calibration against the REAL serving pipeline.

Why this exists instead of reusing results/<disease>/calibration/:
those temperatures were fitted on SINGLE-VIEW logits, but serving runs
`predict_with_tta`, which averages PROBABILITIES over 5 views. A temperature fitted on
single-view logits is not applicable to an averaged probability vector, so wiring the old
constants into the app would have produced confident-looking but wrong calibration.

Method. Serving yields an averaged probability vector p. We calibrate with

    p_cal = softmax(log(p) / T)          (equivalently  p^(1/T) normalised)

i.e. treat log(p) as pseudo-logits. This is well defined, monotone in every component and
therefore **cannot change any prediction** — only the confidence, which is exactly what
calibration is allowed to do.

The val split is DE-DUPLICATED first: 19 val images are byte-identical to train images and
carry contradictory labels (see docs/IMPLEMENTATION_LOG.md U4). Fitting on those would
calibrate against known-bad labels.

    .venv/bin/python scripts/fit_serving_calibration.py --disease rop
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from models.common.config import load_config  # noqa: E402
from models.evaluation.calibration import (  # noqa: E402
    brier_score, expected_calibration_error, fit_temperature, softmax_np)
from webapp.inference import DiseaseModel, _device  # noqa: E402

EPS = 1e-12


def log_pseudologits(p):
    """log of an averaged probability vector, safe against zeros."""
    return np.log(np.clip(np.asarray(p, dtype=np.float64), EPS, 1.0))


def contaminated_val_images(manifest):
    """Val images that are byte-identical duplicates of images in another split."""
    try:
        import imagehash
    except ImportError:
        print("  (imagehash unavailable — skipping de-duplication)")
        return set()
    groups = defaultdict(list)
    for _, r in manifest.iterrows():
        try:
            h = str(imagehash.phash(Image.open(r["image_path"]).convert("RGB")))
        except Exception:
            continue
        groups[h].append((r["split"], r["image_path"]))
    bad = set()
    for g in groups.values():
        if len({s for s, _ in g}) > 1:                 # spans >1 split
            bad.update(p for s, p in g if s == "val")
    return bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--disease", required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--weights", default=None)
    ap.add_argument("--split", default="val")
    args = ap.parse_args()

    cfg_path = args.config or f"configs/{args.disease}.yaml"
    weights = args.weights or f"results/{args.disease}/weights.pth"
    cfg = load_config(cfg_path)

    manifest = pd.read_csv(cfg.data.manifest)
    rows = manifest[manifest["split"] == args.split].reset_index(drop=True)
    missing = [p for p in rows["image_path"] if not Path(p).exists()]
    if missing:
        print(f"ERROR: {len(missing)}/{len(rows)} {args.split} images are not present locally "
              f"(e.g. {missing[0]}).")
        print("Calibration on missing files would silently score blank images. Aborting.")
        return 1

    bad = contaminated_val_images(manifest)
    keep = ~rows["image_path"].isin(bad)
    print(f"{args.split}: {len(rows)} images, {int((~keep).sum())} removed as cross-split "
          f"duplicates -> {int(keep.sum())} used for calibration")
    rows = rows[keep].reset_index(drop=True)

    dm = DiseaseModel(args.disease.upper(), cfg, weights, _device())
    if not dm.loaded:
        print(f"ERROR: weights not found at {weights}")
        return 1

    # EXACT serving path: same TTA transforms, same probability averaging.
    print(f"running serving-path TTA inference on {len(rows)} images…")
    P, Y = [], []
    for i, r in rows.iterrows():
        img = Image.open(r["image_path"]).convert("RGB")
        from models.common.train_utils import predict_with_tta
        p = predict_with_tta(dm.model, img, dm.tta, dm.device,
                             cfg.model.get("head", "classification")).numpy()
        P.append(p); Y.append(int(r["label"]))
        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{len(rows)}")
    P = np.asarray(P, dtype=np.float64)
    Y = np.asarray(Y, dtype=int)
    n_cls = P.shape[1]

    z = log_pseudologits(P)
    ece0, mce0 = expected_calibration_error(P, Y)
    brier0 = brier_score(P, Y, n_cls)

    T = fit_temperature(z, Y)
    P_cal = softmax_np(z / T)
    ece1, mce1 = expected_calibration_error(P_cal, Y)
    brier1 = brier_score(P_cal, Y, n_cls)

    # Calibration must not move any decision.
    changed = int((P.argmax(1) != P_cal.argmax(1)).sum())
    thr = float(cfg.eval.get("screen_threshold", 0.5)) if n_cls == 2 else None
    thr_changed = (int(((P[:, 1] >= thr) != (P_cal[:, 1] >= thr)).sum())
                   if thr is not None else None)

    print(f"\nTTA-serving calibration for {args.disease.upper()}")
    print(f"  temperature T          {T:.4f}")
    print(f"  ECE   {ece0:.4f} -> {ece1:.4f}   ({'improved' if ece1 < ece0 else 'WORSE'})")
    print(f"  MCE   {mce0:.4f} -> {mce1:.4f}")
    print(f"  Brier {brier0:.4f} -> {brier1:.4f}")
    print(f"  argmax decisions changed: {changed}/{len(Y)} (must be 0)")
    if thr is not None:
        print(f"  threshold@{thr} decisions changed: {thr_changed}/{len(Y)}")

    # Decide whether this calibration is worth APPLYING. Temperature minimises NLL, not
    # ECE, so it can leave ECE flat or slightly worse — and applying it shifts the
    # probability scale, which invalidates any threshold tuned on the raw scale. Only apply
    # when there is a real, material improvement to pay for that churn.
    rel_gain = (ece0 - ece1) / ece0 if ece0 > 0 else 0.0
    abs_gain = ece0 - ece1
    # A relative threshold ALONE fires on noise: glaucoma showed "26.1% better" that was
    # 0.0044 in absolute terms on an already-tiny ECE of 0.017, while MCE got WORSE.
    # Require a materially sized absolute gain, and refuse to trade tail calibration (MCE)
    # for a fractional average gain.
    apply = bool(rel_gain >= 0.10 and abs_gain >= 0.01 and mce1 <= mce0)
    if apply:
        reason = (f"ECE improved {rel_gain:.1%} ({ece0:.4f} -> {ece1:.4f}, "
                  f"absolute {abs_gain:.4f}) without degrading MCE")
    elif rel_gain >= 0.10 and abs_gain < 0.01:
        reason = (f"NOT applied: ECE gain is {rel_gain:.1%} but only {abs_gain:.4f} in "
                  f"absolute terms ({ece0:.4f} -> {ece1:.4f}) — the model is already well "
                  f"calibrated by 5-view TTA averaging"
                  + (f", and MCE degrades {mce0:.4f} -> {mce1:.4f}" if mce1 > mce0 else ""))
    else:
        reason = (f"NOT applied: ECE change {rel_gain:+.1%} ({ece0:.4f} -> {ece1:.4f}) is not "
                  f"material. 5-view TTA averaging already acts as an ensemble and calibrates "
                  f"this model; applying T={T:.3f} would shift the probability scale and "
                  f"invalidate the tuned operating threshold for no measurable gain")
    print(f"  DECISION: {'APPLY' if apply else 'DO NOT APPLY'} — {reason}")

    out = {
        "disease": args.disease, "method": "temperature on log(TTA-averaged probs)",
        "apply": apply, "decision_reason": reason,
        "temperature": T, "n": int(len(Y)), "n_classes": n_cls,
        "split": args.split, "removed_duplicates": int((~keep).sum()),
        "uncalibrated": {"ece": ece0, "mce": mce0, "brier": brier0},
        "calibrated": {"ece": ece1, "mce": mce1, "brier": brier1},
        "argmax_changed": changed, "threshold_changed": thr_changed,
        "note": "Fitted against predict_with_tta output (5-view probability average). "
                "NOT interchangeable with results/<disease>/calibration/ which was fitted "
                "on single-view logits.",
    }
    # keep the serving-path predictions: the abstention band (C6) must be derived from the
    # SAME distribution the app produces, not from evaluate.py's non-TTA output
    np.savez(ROOT / "results" / args.disease / f"tta_{args.split}_preds.npz",
             prob=P, y_true=Y, image_path=np.array(list(rows["image_path"]), dtype=object))

    dst = ROOT / "results" / args.disease / "calibration_serving.json"
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(out, indent=2))
    print(f"\n-> {dst}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
