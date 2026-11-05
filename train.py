#!/usr/bin/env python
"""Train one disease model from a config. Everything is config-driven.

    python train.py --config configs/dr.yaml
    python train.py --config configs/rop.yaml --set train.epochs=2 model.arch=efficientnet_b0
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from models.common.config import load_config
from models.common.architectures import build_from_cfg
from models.common.data_prep import build_manifest
from models.common.experiment_logger import ExperimentLogger, pick_device, set_seed
from models.common.metrics import compute_metrics
from models.common.train_utils import build_dataloaders, evaluate, plot_run, train_model
from models.common.losses import build_loss


def get_manifest(cfg):
    path = Path(cfg.data.manifest)
    if path.exists():
        print(f"[train] using existing manifest {path}")
        return pd.read_csv(path)
    print("[train] manifest not found — building it now")
    return build_manifest(cfg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config, overrides=args.set)
    set_seed(int(cfg.seed))
    device = pick_device()

    manifest = get_manifest(cfg)
    loaders = build_dataloaders(cfg, manifest)
    model = build_from_cfg(cfg).to(device)

    logger = ExperimentLogger(cfg, tag=args.tag or cfg.model.arch)
    history, best = train_model(cfg, model, loaders, logger, device)

    # Reload best weights and run the final evaluation on val (+ test if present).
    model.load_state_dict(__import__("torch").load(logger.weights_path, map_location=device))
    _, decode_fn, _ = build_loss(cfg, loaders["counts"])
    head = cfg.model.get("head", "classification")
    num_classes = int(cfg.data.num_classes)
    class_names = list(cfg.data.class_names)

    results = {"best_val_primary": best, "history": history}
    for split in ["val", "test"]:
        if split not in loaders:
            continue
        yt, yp, pr = evaluate(model, loaders[split], decode_fn, num_classes, device, head)
        m = compute_metrics(yt, yp, pr, num_classes, class_names)
        results[split] = m
        print(f"[{split}] acc {m['accuracy']*100:.2f}%  macroF1 {m['macro_f1']*100:.2f}%  "
              f"QWK {m['qwk']:.3f}  AUC {m.get('auc', float('nan')):.3f}")
        if split == "val":
            # Per-sample probabilities are the raw material for everything beyond a point
            # estimate — bootstrap CIs, calibration, per-session aggregation. The flat ROP
            # runs that predate this line kept no predictions, and the session-level
            # comparison could not be computed for them retroactively.
            np.savez_compressed(logger.dir / "val_predictions.npz",
                                y_true=yt, y_pred=yp, prob=pr)
            plot_run(history, {"y_true": yt, "y_pred": yp, "y_prob": pr}, cfg,
                     logger.plot_path("training_eval.png"))

    logger.log_metrics(results)
    print(f"[train] done -> {logger.dir}")


if __name__ == "__main__":
    main()
