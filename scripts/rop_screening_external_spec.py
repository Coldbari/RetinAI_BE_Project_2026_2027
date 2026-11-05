"""Is "ROP Detected" on a healthy eye a bug, or the operating point?

A user screened a visibly healthy retina and got "ROP Detected (48.4%)". The served
threshold is 0.1933, chosen on validation for >=0.90 sensitivity; on the internal test split
it measured 0.990 sensitivity at 0.398 SPECIFICITY. Six healthy eyes in ten are flagged by
design. This script asks the question that matters for the product: what does that look
like OUTSIDE the training hospital, and can the staging head — which the field report shows
calling the same eye Normal at 97.6% — do the same job better?

Evaluation set: the dev half of the held-out hospital (663 multiview images, 150 normal /
513 with ROP). The screening model never saw this hospital: it trained on dataset_split/,
which is Ostrava only and contains zero multiview images (asserted below). The LOCKED half
is not touched — it was opened once, under a pre-registered protocol, and stays closed.

Both detectors answer the same binary question ("any ROP?"), so they are directly
comparable:
  screening : the served ResNet50's P(ROP Detected), 5-view TTA, serving calibration
  staging   : the structured model's P(stage >= 1) = prob6[:, 1:].sum()

Writes results/rop/screening_external_spec.json.

Run:  .venv/bin/python scripts/rop_screening_external_spec.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DEV = ROOT / "results/prepared/rop_staging/test_dev.csv"
OUT = ROOT / "results/rop/screening_external_spec.json"
SERVED_THRESHOLD = 0.1933


def _assert_disjoint(dev: pd.DataFrame) -> None:
    """The screening model's training corpus must not contain this hospital."""
    z = np.load(ROOT / "results/rop/devsplit_test_preds.npz", allow_pickle=True)
    trained_on = {str(p) for p in z["image_path"]}
    assert not any("multiview" in p.lower() for p in trained_on), \
        "screening corpus contains multiview — this is not an external test"
    assert not (set(dev.image_path) & trained_on), "dev images leaked into the screening split"


def _roc(y: np.ndarray, s: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    from sklearn.metrics import roc_curve
    fpr, tpr, thr = roc_curve(y, s)
    return fpr, tpr, thr


def _at_threshold(y: np.ndarray, s: np.ndarray, t: float) -> dict:
    pos, neg = y == 1, y == 0
    sens = float((s[pos] > t).mean())
    spec = float((s[neg] <= t).mean())
    flagged = float((s > t).mean())
    tp, fp = int((s[pos] > t).sum()), int((s[neg] > t).sum())
    return {"threshold": round(float(t), 4), "sensitivity": round(sens, 4),
            "specificity": round(spec, 4), "flagged_frac": round(flagged, 4),
            "ppv": round(tp / max(tp + fp, 1), 4),
            "healthy_flagged": fp, "healthy_total": int(neg.sum())}


def _spec_at_sens(y: np.ndarray, s: np.ndarray, target: float) -> dict:
    """Specificity at the lowest threshold that still reaches `target` sensitivity."""
    fpr, tpr, thr = _roc(y, s)
    ok = np.nonzero(tpr >= target)[0]
    if not len(ok):
        return {"target_sensitivity": target, "achievable": False}
    i = int(ok[np.argmin(fpr[ok])])
    out = _at_threshold(y, s, thr[i])
    out["target_sensitivity"] = target
    out["achievable"] = True
    return out


def main() -> None:
    from sklearn.metrics import roc_auc_score

    dev = pd.read_csv(DEV)
    _assert_disjoint(dev)
    y = (dev.label.values > 0).astype(int)          # any ROP vs normal
    print(f"dev half: {len(dev)} images, {int((y == 0).sum())} normal, {int(y.sum())} ROP")

    # ── the staging head, straight from the committed dev predictions ────────
    folds = [np.load(ROOT / f"results/rop/dev_eval/structured_s42_ckpt{i}.npz",
                     allow_pickle=True) for i in range(5)]
    assert all(np.array_equal(f["y_true"], folds[0]["y_true"]) for f in folds)
    stage_fold0 = folds[0]["prob6"][:, 1:].sum(1)              # what the app serves today
    stage_mean = np.mean([f["prob6"] for f in folds], 0)[:, 1:].sum(1)   # 5-fold ensemble
    assert len(stage_fold0) == len(dev)

    # ── the screening head, through the exact serving path ───────────────────
    import webapp.inference as I
    reg = I.Registry()
    dm = next(m for m in reg.models if m.disease == "ROP" and m.loaded)
    from models.common.train_utils import predict_with_tta

    scr = np.zeros(len(dev), dtype=np.float64)
    for i, p in enumerate(dev.image_path):
        img = Image.open(ROOT / p).convert("RGB")
        probs = predict_with_tta(dm.model, img, dm.tta, dm.device,
                                 dm.cfg.model.get("head", "classification")).numpy()
        scr[i] = float(dm._apply_calibration(probs)[1])
        if (i + 1) % 100 == 0:
            print(f"  screening {i + 1}/{len(dev)}")

    # Save the raw scores: a claim as strong as "specificity 0.000 at the served threshold"
    # has to be recomputable by anyone who doubts it.
    np.savez_compressed(OUT.with_suffix(".npz"), screening=scr, staging_fold0=stage_fold0,
                        staging_5fold=stage_mean, y=y,
                        image_path=np.array(dev.image_path, dtype=object))

    def _dist(s, m):
        q = np.percentile(s[m], [0, 5, 50, 95, 100])
        return {k: round(float(v), 4) for k, v in
                zip(("min", "p5", "median", "p95", "max"), q)}

    res = {
        "score_distribution": {
            "screening_normal": _dist(scr, y == 0),
            "screening_rop": _dist(scr, y == 1),
            "served_threshold": SERVED_THRESHOLD,
            "interpretation": (
                "If screening_normal.min exceeds served_threshold, the served operating "
                "point calls EVERY image at this hospital positive — the threshold was "
                "tuned on Ostrava and does not transfer."),
        },
        "eval_set": {"csv": str(DEV.relative_to(ROOT)), "n": len(dev),
                     "n_normal": int((y == 0).sum()), "n_rop": int(y.sum()),
                     "site": "multiview (held-out hospital), DEV half only",
                     "note": "the locked half is not touched by this script"},
        "auc": {
            "screening": round(float(roc_auc_score(y, scr)), 4),
            "staging_any_rop_fold0": round(float(roc_auc_score(y, stage_fold0)), 4),
            "staging_any_rop_5fold": round(float(roc_auc_score(y, stage_mean)), 4),
        },
        "served_operating_point": {
            "screening": _at_threshold(y, scr, SERVED_THRESHOLD),
        },
        "matched_sensitivity": {
            f"{t:.2f}": {"screening": _spec_at_sens(y, scr, t),
                         "staging_fold0": _spec_at_sens(y, stage_fold0, t),
                         "staging_5fold": _spec_at_sens(y, stage_mean, t)}
            for t in (0.99, 0.95, 0.90)
        },
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))
    print(f"\nwrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
