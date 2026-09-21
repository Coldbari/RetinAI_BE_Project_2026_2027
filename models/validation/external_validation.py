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
    ap.add_argument("--source", default=None,
                    help="name of the external source (default: derived from the config "
                         "filename). Output files are keyed by it so two external sets for "
                         "the SAME disease cannot overwrite each other.")
    ap.add_argument("--force", action="store_true",
                    help="overwrite an existing report for this disease+source")
    ap.add_argument("--set", nargs="*", default=[])
    args = ap.parse_args()

    cfg = load_config(args.config, overrides=args.set)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    # Outputs used to be keyed by disease alone, so evaluating glaucoma on G1020 and then on
    # REFUGE silently destroyed the first result. Key by disease + source.
    source = args.source or Path(args.config).stem
    for prefix in ("external_", f"{cfg.disease}_"):
        if source.startswith(prefix):
            source = source[len(prefix):]
    tag = f"{cfg.disease}_{source}" if source else str(cfg.disease)

    manifest_path = Path(cfg.data.manifest)
    manifest = build_manifest(cfg) if not manifest_path.exists() else \
        __import__("pandas").read_csv(manifest_path)

    out = Path(args.out or f"results/external_{tag}.json")
    if out.exists() and not args.force:
        raise SystemExit(
            f"{out} already exists. Refusing to overwrite an external-validation report — "
            f"silently replacing one is how a prior result disappears. Pass --force, or "
            f"--source <name> to write a separate report.")

    loader, ds = build_eval_loader(cfg, manifest, "test")
    model = build_from_cfg(cfg, pretrained=False).to(device)
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

    # Binary diseases (glaucoma, ROP) previously got no operating point and no interval at
    # all — only multiclass got the referable block. A bare AUC hides the thing that
    # actually breaks across populations: the decision threshold.
    if pr is not None and num_classes == 2:
        from models.common.metrics import best_threshold_for_sensitivity
        from models.evaluation.statistical_tests import bootstrap_ci, clopper_pearson
        from sklearn.metrics import roc_auc_score

        score = pr[:, 1]
        auc, lo, hi = bootstrap_ci(yt, score, roc_auc_score, n_boot=2000)
        target = float(cfg.eval.get("target_sensitivity", 0.90))
        thr, sn, sp = best_threshold_for_sensitivity(yt, score, target)
        deployed = cfg.eval.get("screen_threshold")
        binr = {"auc": auc, "auc_ci": [lo, hi], "target_sensitivity": target,
                "op_threshold": float(thr), "op_sensitivity": float(sn),
                "op_specificity": float(sp)}
        print(f"binary  AUC {auc:.4f} (95% CI {lo:.3f}-{hi:.3f})  | @sens{target:.2f} "
              f"thr {thr:.3f} sens {sn:.3f} spec {sp:.3f}")
        if deployed is not None:
            pred = (score >= float(deployed)).astype(int)
            tp = int(((pred == 1) & (yt == 1)).sum()); fn = int(((pred == 0) & (yt == 1)).sum())
            tn = int(((pred == 0) & (yt == 0)).sum()); fp = int(((pred == 1) & (yt == 0)).sum())
            d_sn, sn_lo, sn_hi = clopper_pearson(tp, tp + fn)
            d_sp, sp_lo, sp_hi = clopper_pearson(tn, tn + fp)
            binr["deployed_threshold"] = float(deployed)
            binr["deployed_sensitivity"] = [d_sn, sn_lo, sn_hi]
            binr["deployed_specificity"] = [d_sp, sp_lo, sp_hi]
            print(f"        at the DEPLOYED threshold {float(deployed):.4f}: "
                  f"sens {d_sn:.3f} [{sn_lo:.3f},{sn_hi:.3f}]  "
                  f"spec {d_sp:.3f} [{sp_lo:.3f},{sp_hi:.3f}]")
        report["binary"] = binr

    report["source"] = source
    report["config"] = args.config
    report["weights"] = args.weights

    if pr is not None:
        np.savez(f"results/external_{tag}_preds.npz", y_true=yt, y_pred=yp, y_prob=pr)

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(f"[external] -> {out}")


if __name__ == "__main__":
    main()
