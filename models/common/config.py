"""Config loading for the retinal screening pipeline.

Everything (disease, dataset, architecture, preprocessing, hyperparameters,
resolution) is driven by a single YAML file so a new experiment is a new config,
not a code edit. Access is dot-style: ``config.train.image_size``.

Usage:
    from models.common.config import load_config
    cfg = load_config("configs/dr.yaml", overrides=["train.epochs=2"])
    print(cfg.train.image_size)        # attribute access
    cfg.to_dict()                      # plain dict (for logging/saving)
"""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Iterable

import yaml


class Config:
    """Recursive dot-access wrapper around a (possibly nested) dict."""

    def __init__(self, data: dict | None = None):
        object.__setattr__(self, "_data", {})
        for key, value in (data or {}).items():
            self[key] = value

    # -- mapping-style access -------------------------------------------------
    def __setitem__(self, key: str, value: Any):
        self._data[key] = self._wrap(value)

    @staticmethod
    def _wrap(value: Any) -> Any:
        if isinstance(value, dict):
            return Config(value)
        if isinstance(value, list):
            return [Config._wrap(v) for v in value]
        return value

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def __contains__(self, key: str) -> bool:
        return key in self._data

    # -- attribute-style access ----------------------------------------------
    def __getattr__(self, key: str) -> Any:
        try:
            return self._data[key]
        except KeyError as exc:
            raise AttributeError(
                f"Config has no key '{key}'. Available: {list(self._data)}"
            ) from exc

    def __setattr__(self, key: str, value: Any):
        self[key] = value

    # -- helpers --------------------------------------------------------------
    def get(self, key: str, default: Any = None) -> Any:
        cur = self._data.get(key, default)
        return cur

    def keys(self):
        return self._data.keys()

    def items(self):
        return self._data.items()

    @staticmethod
    def _unwrap(value: Any) -> Any:
        if isinstance(value, Config):
            return value.to_dict()
        if isinstance(value, list):
            return [Config._unwrap(v) for v in value]
        return value

    def to_dict(self) -> dict:
        return {key: self._unwrap(value) for key, value in self._data.items()}

    def __repr__(self) -> str:
        return f"Config({self.to_dict()!r})"


def _coerce(value: str) -> Any:
    """Turn a CLI override string into an int/float/bool/None when possible."""
    low = value.lower()
    if low in {"true", "false"}:
        return low == "true"
    if low in {"null", "none"}:
        return None
    for cast in (int, float):
        try:
            return cast(value)
        except ValueError:
            continue
    return value


def _apply_override(data: dict, dotted_key: str, value: Any) -> None:
    keys = dotted_key.split(".")
    cur = data
    for k in keys[:-1]:
        cur = cur.setdefault(k, {})
        if not isinstance(cur, dict):
            raise ValueError(f"Cannot set '{dotted_key}': '{k}' is not a mapping")
    cur[keys[-1]] = value


def load_config(path: str | Path, overrides: Iterable[str] | None = None) -> Config:
    """Load a YAML config, optionally applying ``key.subkey=value`` overrides.

    Overrides let Kaggle notebooks tweak a run without editing files, e.g.
    ``--set train.epochs=2 train.image_size=320``.
    """
    path = Path(path)
    with open(path, "r") as fh:
        data = yaml.safe_load(fh) or {}

    data = copy.deepcopy(data)
    for override in overrides or []:
        if "=" not in override:
            raise ValueError(f"Override '{override}' must be key=value")
        key, _, raw = override.partition("=")
        _apply_override(data, key.strip(), _coerce(raw.strip()))

    # Record where it came from (handy for logging/reproducibility).
    data.setdefault("_source", str(path))
    return Config(data)
