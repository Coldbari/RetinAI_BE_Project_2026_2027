#!/usr/bin/env python
"""Cumulative ablation study (W7). Starts from the base config and applies each
step's ``set`` block CUMULATIVELY, training a (short) model per step and recording the
running metric table — turning "CLAHE helped" into measurable evidence.

    python -m models.experiments.ablation_study --config configs/ablation_dr.yaml

Outputs results/ablation_<disease>.csv and an improvement plot. Run on Kaggle.
"""
import argparse
import copy
from pathlib import Path

import pandas as pd
import yaml

from models.common.config import Config, _apply_override, load_config
from models.common.architectures import build_from_cfg
from models.common.data_prep import build_manifest
from models.common.experiment_logger import ExperimentLogger, pick_device, set_seed
from models.common.losses import build_loss
from models.common.metrics import compute_metrics
from models.common.train_utils import build_dataloaders, evaluate, train_model


def _flatten(d, prefix=""):
    """Flatten a (possibly nested) dict into dotted-key -> value pairs."""
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flatten(v, key + "."))
        else:
            out[key] = v
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="configs/ablation_<disease>.yaml")
    ap.add_argument("--set", nargs="*", default=[])
    args = ap.parse_args()

    meta = load_config(args.config, overrides=args.set)
    base_path = meta.base
    with open(base_path) as fh:
        base_dict = yaml.safe_load(fh)

    common = _flatten(meta.get("common_overrides", Config()).to_dict())
    accumulated: dict = {}
    device = pick_device()

    rows = []
    for step in meta.steps:
        accumulated.update(_flatten(step.set.to_dict()))
        merged = copy.deepcopy(base_dict)
        for k, v in {**common, **accumulated}.items():
            _apply_override(merged, k, v)
        cfg = Config(merged)
        set_seed(int(cfg.seed))

        print(f"\n===== ablation step: {step.name} =====")
        manifest = build_manifest(cfg)                 # rebuilt (use_sources may change)
        loaders = build_dataloaders(cfg, manifest)
        model = build_from_cfg(cfg).to(device)
        logger = ExperimentLogger(cfg, tag=f"ablation_{step.name}".replace("+", "p"))
        train_model(cfg, model, loaders, logger, device)

        model.load_state_dict(__import__("torch").load(logger.weights_path,
                                                       map_location=device))
        _, decode_fn, _ = build_loss(cfg, loaders["counts"])
        split = "test" if "test" in loaders else "val"
        yt, yp, pr = evaluate(model, loaders[split], decode_fn,
                              int(cfg.data.num_classes), device,
                              cfg.model.get("head", "classification"))
        m = compute_metrics(yt, yp, pr, int(cfg.data.num_classes),
                            list(cfg.data.class_names))
        rows.append({"step": step.name, "image_size": cfg.preprocess.image_size,
                     "loss": cfg.train.loss, "accuracy": round(m["accuracy"], 4),
                     "macro_f1": round(m["macro_f1"], 4), "qwk": round(m["qwk"], 4),
                     "auc": round(m.get("auc", float("nan")), 4),
                     "macro_recall": round(m["macro_recall"], 4)})

    df = pd.DataFrame(rows)
    out = Path(f"results/ablation_{meta.get('disease', Config(base_dict).disease)}.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    _plot(df, out.with_suffix(".png"))
    print("\n" + df.to_string(index=False))
    print(f"[ablation] -> {out}")


def _plot(df, save_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(11, 5))
    for col, color in [("macro_f1", "#60A5FA"), ("qwk", "#34D399"),
                       ("auc", "#F472B6"), ("macro_recall", "#FBBF24")]:
        if df[col].notna().any():
            ax.plot(df["step"], df[col], marker="o", label=col, color=color)
    ax.set_title("Ablation — cumulative improvement"); ax.set_ylabel("score")
    ax.tick_params(axis="x", rotation=45); ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(save_path, dpi=130); plt.close(fig)


if __name__ == "__main__":
    main()
