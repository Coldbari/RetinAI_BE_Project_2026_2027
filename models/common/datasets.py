"""Dataset objects. Every disease is reduced to a uniform manifest CSV with columns
``image_path,label,split`` (built by ``prepare_data.py``), so training/eval code is
identical across ROP, DR and Glaucoma.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


class FundusDataset(Dataset):
    """Reads (image_path, label) rows for one split from a manifest."""

    def __init__(self, manifest, split: str, transform=None, image_size: int = 384):
        df = manifest if isinstance(manifest, pd.DataFrame) else pd.read_csv(manifest)
        if split is not None and "split" in df.columns:
            df = df[df["split"] == split]
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.image_size = image_size
        self.targets = self.df["label"].astype(int).tolist()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        label = int(row["label"])
        try:
            img = Image.open(row["image_path"]).convert("RGB")
        except Exception:
            # Missing/corrupt file: return a blank image so a batch never crashes.
            return torch.zeros(3, self.image_size, self.image_size), label
        if self.transform:
            img = self.transform(img)
        return img, label


def class_counts(targets, num_classes: int):
    import numpy as np
    return np.bincount(np.asarray(targets, dtype=int), minlength=num_classes).tolist()
