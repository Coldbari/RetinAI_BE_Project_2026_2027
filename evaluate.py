#!/usr/bin/env python
"""Evaluate a trained model on a split and report the tier reached (Acceptable /
Target / Stretch) per the plan's success-criteria, against the documented baselines.

    python evaluate.py --config configs/dr.yaml --weights results/experiment_001/weights.pth --split val
"""
import argparse

import pandas as pd
import torch

from models.common.config import load_config
from models.common.architectures import build_from_cfg
from models.common.losses import build_loss
from models.common.metrics import (best_threshold_for_sensitivity, compute_metrics,
                                    referable_metrics, tier)
from models.common.train_utils import build_dataloaders, evaluate

# Success-criteria thresholds from the approved plan (Acceptable / Target / Stretch).
TIERS = {
    "dr":       {"metric": "qwk",          "acc": 0.75, "tgt": 0.80, "str": 0.85, "baseline": "macro F1 49.7%"},
    "rop":      {"metric": "macro_recall", "acc": 0.85, "tgt": 0.90, "str": 0.93, "baseline": "recall 58.5%"},
    "glaucoma": {"metric": "auc",          "acc": 0.90, "tgt": 0.92, "str": 0.95, "baseline": "n/a (new)"},
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--dump-preds", default=None,
                    help="save (y_true, y_pred, y_prob) to this .npz for local re-analysis")
    args = ap.parse_args()

    cfg = load_config(args.config, overrides=args.set)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")       # local Apple-silicon evaluation
    else:
        device = torch.device("cpu")

    manifest = pd.read_csv(cfg.data.manifest)
    loaders = build_dataloaders(cfg, manifest)
    if args.split not in loaders:
        raise SystemExit(f"No '{args.split}' split in the manifest.")

    model = build_from_cfg(cfg, pretrained=False).to(device)
    model.load_state_dict(torch.load(args.weights, map_location=device))

    _, decode_fn, _ = build_loss(cfg, loaders["counts"])
    head = cfg.model.get("head", "classification")
    num_classes = int(cfg.data.num_classes)
    class_names = list(cfg.data.class_names)

    yt, yp, pr = evaluate(model, loaders[args.split], decode_fn, num_classes, device, head)
    m = compute_metrics(yt, yp, pr, num_classes, class_names)

    print(f"\n=== {cfg.disease.upper()} — {args.split} ===")
    print(f"accuracy        {m['accuracy']*100:.2f}%")
    print(f"macro P/R/F1    {m['macro_precision']*100:.1f} / {m['macro_recall']*100:.1f} / {m['macro_f1']*100:.1f}")
    print(f"QWK             {m['qwk']:.4f}")
    print(f"AUC             {m.get('auc', float('nan')):.4f}")
    if num_classes == 2 and pr is not None:
        thr, sens, spec = best_threshold_for_sensitivity(
            yt, pr[:, 1], float(cfg.eval.get("target_sensitivity", 0.90)))
        print(f"@>=sens op-point thr={thr:.3f} sens={sens:.3f} spec={spec:.3f}")

    # Referable screening view for ordinal multi-grade diseases (e.g. DR grade>=2):
    # the clinically actionable binary endpoint where a >90% AUC legitimately lives.
    ref_grade = cfg.eval.get("referable_grade")
    if pr is not None and num_classes > 2 and ref_grade is not None:
        rm = referable_metrics(yt, pr, int(ref_grade),
                               float(cfg.eval.get("target_sensitivity", 0.90)))
        lo, hi = rm["auc_ci"]
        print(f"\n--- Referable-{cfg.disease.upper()} (grade>={rm['referable_grade']}) "
              f"binary screening view ---")
        print(f"prevalence       {rm['n_referable']}/{rm['n_total']} referable")
        print(f"referable AUC    {rm['auc']:.4f}  (95% CI {lo:.3f}-{hi:.3f})")
        print(f"argmax point     sens={rm['argmax_sensitivity']:.3f} "
              f"spec={rm['argmax_specificity']:.3f}")
        print(f"@>=sens{rm['target_sensitivity']:.2f} op-point  thr={rm['op_threshold']:.3f} "
              f"sens={rm['op_sensitivity']:.3f} spec={rm['op_specificity']:.3f} "
              f"acc={rm['op_accuracy']:.3f}")

    if args.dump_preds:
        import numpy as np
        # image_path is REQUIRED for patient-level analysis: the ROP test split is 1,502
        # images from only 29 infants, so image-level resampling badly overstates precision.
        # The patient id is the first underscore-token of the filename.
        paths = manifest[manifest["split"] == args.split]["image_path"].tolist()
        if len(paths) != len(yt):
            raise SystemExit(
                f"manifest/{args.split} has {len(paths)} rows but evaluation returned "
                f"{len(yt)} predictions — order or filtering mismatch, refusing to dump "
                f"misaligned paths.")
        np.savez(args.dump_preds, y_true=yt, y_pred=yp,
                 y_prob=(pr if pr is not None else []),
                 image_path=np.array(paths, dtype=object))
        print(f"\n[dump] predictions + image_path -> {args.dump_preds}")

    t = TIERS.get(cfg.disease)
    if t:
        val = m.get(t["metric"], 0.0)
        reached = tier(val, t["acc"], t["tgt"], t["str"])
        print(f"\nPrimary ({t['metric']}) = {val:.4f} -> {reached.upper()} tier"
              f"  (baseline {t['baseline']}; acc/tgt/str = {t['acc']}/{t['tgt']}/{t['str']})")


if __name__ == "__main__":
    main()
