"""Reproducible experiment logging. Every run writes a self-contained
``results/experiment_NNN/`` with the exact config, git commit, seed, environment and
metrics — so any run can be reproduced from its saved ``config.yaml``.
"""
from __future__ import annotations

import json
import os
import platform
import random
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import yaml


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def git_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


def device_info() -> dict:
    info = {"torch": torch.__version__, "python": sys.version.split()[0],
            "platform": platform.platform()}
    if torch.cuda.is_available():
        info["device"] = "cuda"
        info["gpu"] = torch.cuda.get_device_name(0)
    elif torch.backends.mps.is_available():
        info["device"] = "mps"
    else:
        info["device"] = "cpu"
    return info


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class ExperimentLogger:
    def __init__(self, cfg, results_root: str = "results", tag: str | None = None):
        self.cfg = cfg
        root = Path(results_root)
        root.mkdir(parents=True, exist_ok=True)

        existing = [int(p.name.split("_")[1]) for p in root.glob("experiment_*")
                    if p.name.split("_")[1].isdigit()]
        n = (max(existing) + 1) if existing else 1
        name = f"experiment_{n:03d}"
        if tag:
            name += f"_{tag}"
        self.dir = root / name
        self.plots = self.dir / "plots"
        self.errors = self.dir / "errors"
        for d in (self.dir, self.plots, self.errors):
            d.mkdir(parents=True, exist_ok=True)

        self.meta = {
            "experiment": name,
            "created": datetime.now().isoformat(timespec="seconds"),
            "git_hash": git_hash(),
            "seed": int(cfg.seed),
            "disease": cfg.get("disease", "unknown"),
            "env": device_info(),
        }
        with open(self.dir / "config.yaml", "w") as fh:
            yaml.safe_dump(cfg.to_dict(), fh, sort_keys=False)
        with open(self.dir / "meta.json", "w") as fh:
            json.dump(self.meta, fh, indent=2)
        print(f"[logger] {self.dir}  (git {self.meta['git_hash'][:8]}, "
              f"device {self.meta['env']['device']})")

    @property
    def weights_path(self) -> Path:
        return self.dir / "weights.pth"

    def save_weights(self, state_dict):
        torch.save(state_dict, self.weights_path)

    def log_metrics(self, metrics: dict, filename: str = "metrics.json"):
        with open(self.dir / filename, "w") as fh:
            json.dump(metrics, fh, indent=2)

    def plot_path(self, name: str) -> Path:
        return self.plots / name
