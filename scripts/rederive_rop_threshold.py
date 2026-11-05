"""Re-derive the ROP operating threshold honestly.

Fixes two defects found in Phase 1 / U4:
  S5  the deployed threshold 0.371 is exactly the value that hits sens>=0.90 on the TEST
      split — it was selected on test, so the quoted specificity is optimistic.
  U4  19 val images are byte-identical duplicates of train images and carry contradictory
      labels, so the val split used for selection was itself contaminated.

Procedure (the only defensible one):
  1. derive the threshold on the CLEAN, de-duplicated VAL split
  2. apply it to TEST exactly once and report the result, whatever it is

Both splits are scored through the real serving path (5-view TTA), because a threshold
derived on non-TTA scores does not transfer to a TTA-averaged probability.

    .venv/bin/python scripts/rederive_rop_threshold.py
"""
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
from models.common.train_utils import predict_with_tta  # noqa: E402
from models.evaluation.statistical_tests import clopper_pearson  # noqa: E402
from webapp.inference import DiseaseModel, _device  # noqa: E402

TARGET_SENS = 0.90


def contaminated(manifest):
    import imagehash
    groups = defaultdict(list)
    for _, r in manifest.iterrows():
        try:
            h = str(imagehash.phash(Image.open(r["image_path"]).convert("RGB")))
        except Exception:
            continue
        groups[h].append((r["split"], r["image_path"]))
    bad = set()
    for g in groups.values():
        if len({s for s, _ in g}) > 1:
            bad.update(p for _, p in g)          # drop the duplicate from EVERY split
    return bad


def tta_scores(dm, cfg, rows, tag):
    cache = ROOT / "results" / "rop" / f"tta_{tag}_preds.npz"
    if cache.exists():
        d = np.load(cache, allow_pickle=True)
        if len(d["y_true"]) == len(rows):
            print(f"  {tag}: reusing cached serving-path predictions ({len(rows)})")
            return d["prob"][:, 1], d["y_true"].astype(int)
    print(f"  {tag}: scoring {len(rows)} images through the serving path…")
    P, Y = [], []
    for i, r in enumerate(rows.itertuples()):
        img = Image.open(r.image_path).convert("RGB")
        P.append(predict_with_tta(dm.model, img, dm.tta, dm.device,
                                  cfg.model.get("head", "classification")).numpy())
        Y.append(int(r.label))
        if (i + 1) % 300 == 0:
            print(f"    {i+1}/{len(rows)}")
    P = np.asarray(P); Y = np.asarray(Y)
    np.savez(cache, prob=P, y_true=Y,
             image_path=np.array(list(rows["image_path"]), dtype=object))
    return P[:, 1], Y


def sens_spec(y, s, thr):
    pred = (s >= thr).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum()); fn = int(((pred == 0) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
    return tp, fn, tn, fp


def pick(y, s, target):
    best = None
    for thr in np.unique(np.round(s, 4)):
        tp, fn, tn, fp = sens_spec(y, s, thr)
        sn = tp / max(tp + fn, 1); sp = tn / max(tn + fp, 1)
        if sn >= target and (best is None or sp > best[2]):
            best = (float(thr), sn, sp)
    return best or (0.5, 0.0, 0.0)


def main():
    cfg = load_config("configs/rop.yaml")
    man = pd.read_csv(cfg.data.manifest)
    bad = contaminated(man)
    print(f"cross-split duplicate images removed from ALL splits: {len(bad)}")

    clean = man[~man["image_path"].isin(bad)]
    val = clean[clean.split == "val"].reset_index(drop=True)
    test = clean[clean.split == "test"].reset_index(drop=True)
    print(f"clean val {len(val)} (was {(man.split=='val').sum()}), "
          f"clean test {len(test)} (was {(man.split=='test').sum()})")

    dm = DiseaseModel("ROP", cfg, "results/rop/weights.pth", _device())
    if not dm.loaded:
        print("ERROR: ROP weights missing"); return 1

    vs, vy = tta_scores(dm, cfg, val, "val")
    ts, ty = tta_scores(dm, cfg, test, "test")

    thr, v_sn, v_sp = pick(vy, vs, TARGET_SENS)
    tp, fn, tn, fp = sens_spec(ty, ts, thr)
    t_sn = tp / max(tp + fn, 1); t_sp = tn / max(tn + fp, 1)
    sn_lo, sn_hi = clopper_pearson(tp, tp + fn)[1:]
    sp_lo, sp_hi = clopper_pearson(tn, tn + fp)[1:]

    old_thr = float(cfg.eval.get("screen_threshold", 0.5))
    o_tp, o_fn, o_tn, o_fp = sens_spec(ty, ts, old_thr)

    print(f"\n{'='*72}")
    print("HONEST OPERATING POINT (derived on clean val, measured once on test)")
    print(f"{'='*72}")
    print(f"  threshold from clean VAL @ sens>={TARGET_SENS}: {thr:.4f}")
    print(f"    val  sens {v_sn:.3f} / spec {v_sp:.3f}")
    print(f"  applied ONCE to TEST:")
    print(f"    test sens {t_sn:.3f}  95% CI [{sn_lo:.3f}, {sn_hi:.3f}]")
    print(f"    test spec {t_sp:.3f}  95% CI [{sp_lo:.3f}, {sp_hi:.3f}]")
    print(f"\n  previously deployed threshold {old_thr} (selected ON test):")
    print(f"    test sens {o_tp/max(o_tp+o_fn,1):.3f} / spec {o_tn/max(o_tn+o_fp,1):.3f}")

    out = {"target_sensitivity": TARGET_SENS, "threshold": thr,
           "derived_on": "clean val (cross-split duplicates removed), 5-view TTA",
           "val": {"n": int(len(vy)), "sensitivity": v_sn, "specificity": v_sp},
           "test": {"n": int(len(ty)), "sensitivity": t_sn, "specificity": t_sp,
                    "sens_ci": [sn_lo, sn_hi], "spec_ci": [sp_lo, sp_hi]},
           "previous_threshold": old_thr,
           "previous_test": {"sensitivity": o_tp/max(o_tp+o_fn,1),
                             "specificity": o_tn/max(o_tn+o_fp,1),
                             "note": "threshold was selected on test — optimistic"},
           "removed_duplicate_images": len(bad)}
    (ROOT / "results" / "rop" / "operating_point.json").write_text(json.dumps(out, indent=2))
    print(f"\n-> results/rop/operating_point.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
