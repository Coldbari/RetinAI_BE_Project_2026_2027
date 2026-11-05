#!/usr/bin/env python
"""Deployment benchmarking + architecture sweep (W6).

Single model (latency / memory / size):
    python benchmark.py --config configs/dr.yaml --weights results/experiment_001/weights.pth

Sweep (train each architecture short, then benchmark) -> results/sweep_<disease>.csv:
    python benchmark.py --sweep configs/sweep.yaml
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from models.common.config import load_config
from models.common.architectures import build_from_cfg, build_model
from models.common.experiment_logger import ExperimentLogger, pick_device, set_seed
from models.common.losses import build_loss
from models.common.metrics import compute_metrics
from models.common.train_utils import build_dataloaders, evaluate, train_model


def model_size_mb(model) -> float:
    bytes_ = sum(p.numel() * p.element_size() for p in model.parameters())
    bytes_ += sum(b.numel() * b.element_size() for b in model.buffers())
    return bytes_ / (1024 ** 2)


@torch.no_grad()
def latency_ms(model, image_size, device, batch=1, n=30, warmup=5):
    model.eval().to(device)
    x = torch.randn(batch, 3, image_size, image_size, device=device)
    for _ in range(warmup):
        model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n * 1000.0


def benchmark_model(model, image_size, num_classes):
    params = sum(p.numel() for p in model.parameters())
    row = {"params_M": round(params / 1e6, 2), "size_MB": round(model_size_mb(model), 1)}
    cpu = torch.device("cpu")
    row["cpu_ms"] = round(latency_ms(model, image_size, cpu, batch=1), 1)
    row["cpu_fps"] = round(1000.0 / row["cpu_ms"], 1)
    if torch.cuda.is_available():
        gpu = torch.device("cuda")
        row["gpu_ms"] = round(latency_ms(model, image_size, gpu, batch=1), 2)
        row["gpu_batch16_ms"] = round(latency_ms(model, image_size, gpu, batch=16), 1)
        row["gpu_fps"] = round(1000.0 / row["gpu_ms"], 1)
    return row


def run_single(args):
    cfg = load_config(args.config, overrides=args.set)
    model = build_from_cfg(cfg)
    if args.weights:
        model.load_state_dict(torch.load(args.weights, map_location="cpu"))
    row = benchmark_model(model, int(cfg.preprocess.image_size), int(cfg.data.num_classes))
    print(f"{cfg.model.arch}: {row}")


def run_sweep(args):
    meta = load_config(args.sweep)
    base = load_config(meta.base)
    overrides = meta.get("overrides", None)
    over_list = [f"{k}={v}" for k, v in (overrides.items() if overrides else [])]
    base = load_config(meta.base, overrides=over_list)

    set_seed(int(base.seed))
    device = pick_device()
    manifest = pd.read_csv(base.data.manifest)

    rows = []
    for arch in meta.architectures:
        print(f"\n===== sweep: {arch} =====")
        cfg = load_config(meta.base, overrides=over_list + [f"model.arch={arch}"])
        loaders = build_dataloaders(cfg, manifest)
        model = build_model(arch, int(cfg.data.num_classes),
                            pretrained=bool(cfg.model.get("pretrained", True))).to(device)
        logger = ExperimentLogger(cfg, tag=f"sweep_{arch}")
        _, best = train_model(cfg, model, loaders, logger, device)

        model.load_state_dict(torch.load(logger.weights_path, map_location=device))
        _, decode_fn, _ = build_loss(cfg, loaders["counts"])
        split = "test" if "test" in loaders else "val"
        yt, yp, pr = evaluate(model, loaders[split], decode_fn,
                              int(cfg.data.num_classes), device,
                              cfg.model.get("head", "classification"))
        m = compute_metrics(yt, yp, pr, int(cfg.data.num_classes), list(cfg.data.class_names))
        bench = benchmark_model(model.cpu(), int(cfg.preprocess.image_size),
                                int(cfg.data.num_classes))
        rows.append({"arch": arch, "accuracy": round(m["accuracy"], 4),
                     "macro_f1": round(m["macro_f1"], 4), "qwk": round(m["qwk"], 4),
                     "auc": round(m.get("auc", float("nan")), 4), **bench})

    out = Path(f"results/sweep_{base.disease}.csv")
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\n[sweep] -> {out}")
    print(pd.DataFrame(rows).to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--weights")
    ap.add_argument("--sweep")
    ap.add_argument("--set", nargs="*", default=[])
    args = ap.parse_args()
    if args.sweep:
        run_sweep(args)
    elif args.config:
        run_single(args)
    else:
        ap.error("pass --config (single) or --sweep (architecture comparison)")


if __name__ == "__main__":
    main()
