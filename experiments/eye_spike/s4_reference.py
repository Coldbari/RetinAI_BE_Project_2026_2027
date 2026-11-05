"""Step 4 — fine-tuned EfficientNetV2-S structured head (seed 42), from COMMITTED predictions only.

    .venv/bin/python experiments/eye_spike/s4_reference.py

CV:  results/rop/staging_runs/structured_5fold/fold_k.npz (out-of-fold, row order = fold_k val
     rows) -> per-bag max of the predicted class.
DEV: results/rop/dev_eval/structured_s42_ckpt{0..4}.npz (row order = test_dev.csv) -> per-eye
     max, metric = mean over the 5 checkpoints (as rop_locked_eval / preprint_stats B do).
Alignment is verified by y_true == committed label, row by row, and the dev eye-level QWK must
reproduce preprint_stats.json B (structured_s42 eye_qwk mean 0.7206).
Writes cache/ref_preds.npz (per-image predictions, no paths) and ref_results.json.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from common import CACHE, CLASSES, HERE, ROOT, bag_metrics, load_dev, load_pool, mf1, qwk


def main():
    pool, dev = load_pool(), load_dev()
    mem = np.load(CACHE / "membership.npz")
    bags = pd.read_csv(HERE / "bags.csv")
    pb, db = bags[bags.fold >= 0].reset_index(drop=True), bags[bags.fold < 0].reset_index(drop=True)

    # ---- CV out-of-fold
    pred = np.full(len(pool), -1)
    for k in range(5):
        z = np.load(ROOT / f"results/rop/staging_runs/structured_5fold/fold_{k}.npz")
        rows = pool.index[pool.fold == k].values
        assert len(z["y_true"]) == len(rows)
        assert (z["y_true"] == pool.label.values[rows]).all(), f"fold {k} misaligned"
        assert (pool.row_in_fold.values[rows] == np.arange(len(rows))).all()
        pred[rows] = z["y_pred"]
    assert (pred >= 0).all()
    ib = mem["pool_img_bag"]
    bag_pred = pd.Series(pred).groupby(ib).max().sort_index().values
    bag_y = pd.Series(pool.label.values).groupby(ib).max().sort_index().values
    assert (bag_y == pb.label.values).all()

    res = {"cv": {"ALL": bag_metrics(bag_y, bag_pred)}}
    for s in ["ostrava", "ropvl", "shenzhen"]:
        m = (pb.source == s).values
        res["cv"][s] = bag_metrics(bag_y[m], bag_pred[m])
    m = (pb.source != "ostrava").values
    res["cv"]["non_ostrava"] = bag_metrics(bag_y[m], bag_pred[m])
    m = (pb.n_images > 1).values
    res["cv"]["multi_image_bags"] = bag_metrics(bag_y[m], bag_pred[m])
    res["cv"]["per_fold"] = {k: {"qwk": qwk(bag_y[pb.fold == k], bag_pred[pb.fold == k]),
                                 "macro_f1": mf1(bag_y[pb.fold == k], bag_pred[pb.fold == k])}
                             for k in range(5)}
    res["cv"]["image_level_pooled"] = bag_metrics(pool.label.values, pred)

    # ---- DEV, 5 checkpoints
    dib = mem["dev_img_bag"]
    dev_y = pd.Series(dev.label.values).groupby(dib).max().sort_index().values
    assert (dev_y == db.label.values).all()
    per_ck, dev_preds = [], []
    for c in range(5):
        z = np.load(ROOT / f"results/rop/dev_eval/structured_s42_ckpt{c}.npz")
        assert (z["y_true"] == dev.label.values).all(), f"dev ckpt {c} misaligned"
        dev_preds.append(z["y_pred"])
        bp = pd.Series(z["y_pred"]).groupby(dib).max().sort_index().values
        per_ck.append(bag_metrics(dev_y, bp))
    eye_qwk_mean = float(np.mean([m["qwk"] for m in per_ck]))
    ref = json.loads((ROOT / "results/rop/preprint_stats.json").read_text())
    want = ref["B_dev_image_vs_eye_qwk"]["cells"]["structured_s42"]["eye_qwk"]["mean"]
    assert round(eye_qwk_mean, 4) == want, (eye_qwk_mean, want)
    res["dev"] = {"qwk_mean": eye_qwk_mean, "qwk_sd": float(np.std([m["qwk"] for m in per_ck], ddof=1)),
                  "macro_f1_mean": float(np.mean([m["macro_f1"] for m in per_ck])),
                  "macro_f1_sd": float(np.std([m["macro_f1"] for m in per_ck], ddof=1)),
                  "acc_mean": float(np.mean([m["acc"] for m in per_ck])),
                  "recall_mean": {c: float(np.mean([m["recall"][c] for m in per_ck]))
                                  for c in per_ck[0]["recall"]},
                  "per_checkpoint": per_ck,
                  "reproduction_check": {"eye_qwk_mean": round(eye_qwk_mean, 4),
                                         "preprint_stats_B": want, "match": True}}
    np.savez(CACHE / "ref_preds.npz", pool_pred=pred, dev_pred=np.stack(dev_preds))
    (HERE / "ref_results.json").write_text(json.dumps(res, indent=2) + "\n")
    print("CV bag:", {k: round(v, 4) for k, v in res["cv"]["ALL"].items() if isinstance(v, float)})
    print("CV image pooled:", {k: round(v, 4) for k, v in res["cv"]["image_level_pooled"].items()
                               if isinstance(v, float)})
    print("dev eye: qwk %.4f (preprint %.4f)  mF1 %.4f" % (eye_qwk_mean, want, res["dev"]["macro_f1_mean"]))
    for s in ["ostrava", "ropvl", "shenzhen", "multi_image_bags"]:
        print(s, round(res["cv"][s]["qwk"], 4), round(res["cv"][s]["macro_f1"], 4), res["cv"][s]["n"])


if __name__ == "__main__":
    main()
