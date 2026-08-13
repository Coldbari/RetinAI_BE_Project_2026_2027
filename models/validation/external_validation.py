#!/usr/bin/env python
"""External validation (W8) — prove generalization on an INDEPENDENT dataset.

The external config must reuse the SAME ``preprocess`` and ``model`` blocks as the
training config (that's the point) and point ``data`` at the external set with
``test_split: 1.0`` so every external image is evaluated. No retraining.

    python -m models.validation.external_validation \\
        --config configs/external_dr_messidor.yaml \\
        --weights results/experiment_003/weights.pth \\
        --internal 0.94            # internal AUC, to report the drop

Report internal-vs-external drop (e.g. AUC 0.94 -> 0.91 = 3%). Success: drop <= 5-10%.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from models.common.config import load_config
from models.common.architectures import build_from_cfg
from models.common.data_prep import build_manifest
from models.common.losses import build_loss
from models.common.metrics import compute_metrics
from models.common.train_utils import evaluate
from models.validation.dataset_loader import build_eval_loader


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="external data config (matching preprocess/model)")
    ap.add_argument("--weights", required=True)
    ap.add_argument("--internal", type=float, default=None,
                    help="internal-test value of the primary metric, to report the drop")
    ap.add_argument("--internal-referable", type=float, default=None,
                    help="internal referable AUC, to report the referable drop (DR)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--set", nargs="*", default=[])
    args = ap.parse_args()

    cfg = load_config(args.config, overrides=args.set)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    manifest_path = Path(cfg.data.manifest)
    manifest = build_manifest(cfg) if not manifest_path.exists() else \
        __import__("pandas").read_csv(manifest_path)

    loader, ds = build_eval_loader(cfg, manifest, "test")
    model = build_from_cfg(cfg).to(device)
    model.load_state_dict(torch.load(args.weights, map_location=device))

    _, decode_fn, _ = build_loss(cfg, [1] * int(cfg.data.num_classes))
    num_classes = int(cfg.data.num_classes)
    yt, yp, pr = evaluate(model, loader, decode_fn, num_classes, device,
                          cfg.model.get("head", "classification"))
    m = compute_metrics(yt, yp, pr, num_classes, list(cfg.data.class_names))

    primary = cfg.eval.get("primary_metric", "auc")
    ext_val = m.get("auc" if primary == "auc" else primary, m.get("macro_f1"))
    print(f"\n=== EXTERNAL {cfg.disease.upper()} ({len(ds)} images) ===")
    print(f"accuracy {m['accuracy']*100:.2f}%  macroF1 {m['macro_f1']*100:.2f}%  "
          f"AUC {m.get('auc', float('nan')):.3f}  QWK {m['qwk']:.3f}")

    report = {"external": m, "primary_metric": primary, "external_value": ext_val}
    if args.internal is not None:
        drop = args.internal - ext_val
        report["internal_value"] = args.internal
        report["drop"] = drop
        verdict = "PASS" if drop <= 0.10 else "REVIEW"
        print(f"{primary}: internal {args.internal:.3f} -> external {ext_val:.3f}  "
              f"(drop {drop*100:.1f}%)  [{verdict}; target <= 5-10%]")

    # Referable screening view (ordinal diseases, e.g. DR grade>=2) — the clinically
    # actionable endpoint. Reported on the external set as the honest generalization test.
    ref_grade = cfg.eval.get("referable_grade")
    if pr is not None and num_classes > 2 and ref_grade is not None:
        from models.common.metrics import referable_metrics
        rm = referable_metrics(yt, pr, int(ref_grade),
                               float(cfg.eval.get("target_sensitivity", 0.90)))
        lo, hi = rm["auc_ci"]
        print(f"referable(grade>={rm['referable_grade']})  AUC {rm['auc']:.4f} "
              f"(95% CI {lo:.3f}-{hi:.3f})  | argmax sens {rm['argmax_sensitivity']:.3f} "
              f"spec {rm['argmax_specificity']:.3f} | @sens{rm['target_sensitivity']:.2f} "
              f"thr {rm['op_threshold']:.3f} sens {rm['op_sensitivity']:.3f} "
              f"spec {rm['op_specificity']:.3f}")
        report["referable"] = rm
        if args.internal_referable is not None:
            d = args.internal_referable - rm["auc"]
            print(f"referable AUC: internal {args.internal_referable:.3f} -> external "
                  f"{rm['auc']:.3f}  (drop {d*100:.1f}%)")
            report["internal_referable"] = args.internal_referable
            report["referable_drop"] = d

    if pr is not None:
        np.savez(f"results/external_{cfg.disease}_preds.npz",
                 y_true=yt, y_pred=yp, y_prob=pr)

    out = Path(args.out or f"results/external_{cfg.disease}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(f"[external] -> {out}")


if __name__ == "__main__":
    main()
