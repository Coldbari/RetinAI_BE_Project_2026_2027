"""Dataset objects. Every disease is reduced to a uniform manifest CSV with columns
``image_path,label,split`` (built by ``prepare_data.py``), so training/eval code is
identical across ROP, DR and Glaucoma.

``FundusDataset(cache_root=...)`` reads each image from a tree of pre-processed copies instead
of the original (``cached_path`` says where); see its docstring for what that changes.
"""
from __future__ import annotations

from pathlib import Path, PurePosixPath

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

# The image sources under data/rop/, and so under <cache_root>/rop/. The same list the Kaggle
# kernels use to find the image dataset's mount.
ROP_SOURCES = ("ostrava", "shenzhen", "rop_vl", "multiview")


def cached_path(image_path: str, cache_root: str) -> Path:
    """Where the pre-processed copy of ``image_path`` lives: ``<cache_root>/rop/<rest>.jpg``.

    A manifest spells the same image differently on every machine that reads it, so the tree is
    found by an anchor, never by a prefix: repo-relative (``data/rop/ostrava/...``, as in the
    committed fold CSVs), absolute on a laptop (``/Users/.../data/rop/...``) or remapped onto a
    Kaggle mount (``/kaggle/input/<dataset>/rop/<source>/...``, where the ``data/`` level is
    gone). The LAST ``data/rop/`` wins; failing that, the LAST ``rop/<source>/`` with ``<source>``
    one of ``ROP_SOURCES``. Anything else raises ``ValueError``: guessing would point the Dataset
    at the wrong file.

    The cache stores JPEG, so the suffix is ``.jpg`` whatever the source had (a ``.png`` source
    is cached under the same stem). Every path in the rop_staging manifests is already ``.jpg``,
    so on them the mapping is one-to-one.
    """
    parts = PurePosixPath(str(image_path)).parts
    last = range(len(parts) - 3, -1, -1)       # from the last index that a file name still follows
    after_data_rop = next((parts[i + 2:] for i in last if parts[i:i + 2] == ("data", "rop")), None)
    after_source = next((parts[i + 1:] for i in last
                         if parts[i] == "rop" and parts[i + 1] in ROP_SOURCES), None)
    rest = after_data_rop or after_source
    if rest is None:
        raise ValueError(f"cannot place {str(image_path)!r} in the cache: no 'data/rop/' and no "
                         f"'rop/<source>/' directory in it (sources: {', '.join(ROP_SOURCES)})")
    return Path(cache_root, "rop", *rest).with_suffix(".jpg")


class FundusDataset(Dataset):
    """Reads (image_path, label) rows for one split from a manifest.

    With ``cache_root`` set (``preprocess.cached``), each image is read from
    ``cached_path(image_path, cache_root)``: a copy that has already been through the fundus
    preprocessing, which the transform pipeline then leaves out. A cache file that is missing or
    unreadable RAISES. The blank-tensor fallback is for an unreadable photograph in an uncached
    read only: a cache is a build product, so a hole in it is a broken build, and a black image
    with a real label would silently train the model on nothing.
    """

    def __init__(self, manifest, split: str, transform=None, image_size: int = 384,
                 cache_root: str | None = None):
        df = manifest if isinstance(manifest, pd.DataFrame) else pd.read_csv(manifest)
        if split is not None and "split" in df.columns:
            df = df[df["split"] == split]
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.image_size = image_size
        self.cache_root = cache_root
        self.targets = self.df["label"].astype(int).tolist()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        label = int(row["label"])
        if self.cache_root is not None:
            img = Image.open(cached_path(row["image_path"], self.cache_root)).convert("RGB")
        else:
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
