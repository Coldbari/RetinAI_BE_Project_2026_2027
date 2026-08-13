"""Loader for an external (independent) evaluation set — applies the SAME
preprocessing as training so the comparison is fair.
"""
from __future__ import annotations

from torch.utils.data import DataLoader

from models.common.datasets import FundusDataset
from models.common.preprocessing import build_transforms


def build_eval_loader(cfg, manifest, split="test", batch_size=None):
    """A plain (no-shuffle, no-sampler) loader for one split, using the val/test
    transform — identical fundus preprocessing to training."""
    size = int(cfg.preprocess.image_size)
    bs = int(batch_size or cfg.train.batch_size)
    ds = FundusDataset(manifest, split, build_transforms(cfg, "val"), size)
    if len(ds) == 0:
        raise RuntimeError(f"External '{split}' split is empty — check the data config.")
    loader = DataLoader(ds, batch_size=bs, shuffle=False,
                        num_workers=int(cfg.train.get("num_workers", 4)))
    return loader, ds
