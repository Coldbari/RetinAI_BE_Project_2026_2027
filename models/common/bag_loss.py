"""Arm D (PROTOCOL_V2 Amendment 3c): an exam-level log-sum-exp bag loss on one source's sessions.

With ``train.bag_loss.enabled`` an epoch alternates two kinds of batch:

- IMAGE batches: every training row outside a multi-frame session (the other sources, plus the
  bag source's singleton sessions), ``batch_size`` per batch, shuffled; per-image loss as before.
- BAG batches: the multi-frame sessions of ``source`` (key = ``units.unit_keys``),
  ``batch_size // k`` bags per batch; each bag is ``k`` frames drawn without replacement, afresh
  every epoch (all frames if the session has <= k). A bag's logit is
  ``tau * logsumexp(frame_logits / tau)``, scored by the SAME loss function on the session label.

Every image batch and every bag is visited once per epoch; the two kinds are interleaved in
proportion (``interleave``). A batch carries one group id per row: in an image batch every row is
its own group, so the pooling is the identity there and the loss is exactly the per-image loss.
Evaluation is unchanged (per image).
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, Sampler


def bag_settings(cfg) -> dict | None:
    """``{"source", "k", "tau"}`` when ``train.bag_loss.enabled`` is true, else None."""
    b = cfg.train.get("bag_loss")
    if not b:
        return None
    on = b.get("enabled", False)
    if not isinstance(on, bool):    # "--set ...enabled=yes" would otherwise read as true
        raise ValueError(f"train.bag_loss.enabled must be true or false, got {on!r}")
    if not on:
        return None
    k, tau = int(b.get("k")), float(b.get("tau"))
    if k < 1 or not tau > 0:
        raise ValueError(f"train.bag_loss needs k >= 1 and tau > 0, got k={k}, tau={tau}")
    return {"source": str(b.get("source")), "k": k, "tau": tau}


def session_plan(df: pd.DataFrame, source: str) -> tuple[np.ndarray, list[np.ndarray]]:
    """Row positions (in ``df``'s order) of the image rows, and of each multi-frame session.

    Raises if a session holds two labels: its bag would have no single label to be scored on.
    """
    from models.common.units import unit_keys
    df = df.reset_index(drop=True)
    pos = np.flatnonzero((df["source"] == source).values)
    keys = unit_keys(df.iloc[pos].reset_index(drop=True))
    groups = pd.Series(pos).groupby(keys, sort=True)
    labels = df["label"].astype(int).values
    bags, singles = [], []
    for key, rows in groups:
        rows = rows.values
        if len(set(labels[rows])) > 1:
            raise ValueError(f"session {key} has mixed labels {sorted(set(labels[rows]))}")
        (bags if len(rows) > 1 else singles).append(rows)
    in_bag = np.zeros(len(df), bool)
    for b in bags:
        in_bag[b] = True
    return np.flatnonzero(~in_bag), bags


def interleave(n_img: int, n_bag: int) -> list[tuple[str, int]]:
    """``("I", i)`` / ``("B", j)`` in proportional order: batch i of n sits at (i + 0.5) / n."""
    keys = [((i + 0.5) / n_img, 0, "I", i) for i in range(n_img)]
    keys += [((j + 0.5) / n_bag, 1, "B", j) for j in range(n_bag)]
    return [(kind, j) for _, _, kind, j in sorted(keys)]


class BagBatchSampler(Sampler):
    """Yields batches of ``(row, group)``; see the module docstring. One RNG, advanced per epoch."""

    def __init__(self, image_rows, bags, batch_size: int, k: int, seed: int = 0):
        self.image_rows = np.asarray(image_rows, dtype=int)
        self.bags = [np.asarray(b, dtype=int) for b in bags]
        self.bs, self.k = int(batch_size), int(k)
        self.per_batch = max(1, self.bs // self.k)
        self.rng = np.random.default_rng(seed)

    def n_batches(self) -> tuple[int, int]:
        return math.ceil(len(self.image_rows) / self.bs), math.ceil(len(self.bags) / self.per_batch)

    def __len__(self) -> int:
        return sum(self.n_batches())

    def __iter__(self):
        rng = self.rng
        img = rng.permutation(self.image_rows)
        img_batches = [img[i:i + self.bs] for i in range(0, len(img), self.bs)]
        drawn = [rng.choice(self.bags[j], size=min(self.k, len(self.bags[j])), replace=False)
                 for j in rng.permutation(len(self.bags))]
        bag_batches = [drawn[i:i + self.per_batch] for i in range(0, len(drawn), self.per_batch)]
        for kind, j in interleave(len(img_batches), len(bag_batches)):
            if kind == "I":
                yield [(int(r), g) for g, r in enumerate(img_batches[j])]
            else:
                yield [(int(r), g) for g, bag in enumerate(bag_batches[j]) for r in bag]


class GroupedDataset(Dataset):
    """``dataset[(row, group)] -> (image, label, group)``, for the batches above."""

    def __init__(self, base):
        self.base = base

    def __len__(self):
        return len(self.base)

    def __getitem__(self, item):
        row, group = item
        img, label = self.base[row]
        return img, label, group


def bag_loader(train_ds, batch_size: int, bag: dict, seed: int, num_workers: int,
               pin_memory: bool) -> DataLoader:
    image_rows, bags = session_plan(train_ds.df, bag["source"])
    sampler = BagBatchSampler(image_rows, bags, batch_size, bag["k"], seed=seed)
    n_img, n_bag = sampler.n_batches()
    print(f"[bag] plan: {len(image_rows)} image rows -> {n_img} image batches; {len(bags)} "
          f"multi-frame {bag['source']} sessions ({sum(map(len, bags))} frames) -> {n_bag} bag "
          f"batches of {sampler.per_batch}; k={bag['k']} tau={bag['tau']}", flush=True)
    return DataLoader(GroupedDataset(train_ds), batch_sampler=sampler, num_workers=num_workers,
                      pin_memory=pin_memory)


def pool_bags(logits: torch.Tensor, labels: torch.Tensor, groups: torch.Tensor,
              tau: float = 1.0) -> tuple[torch.Tensor, torch.Tensor]:
    """One ``tau * logsumexp(logits / tau)`` row per group (ids 0..G-1), and the group's label.

    When every row is its own group (an image batch) the logits come back unchanged.
    """
    n = int(groups.max()) + 1 if len(groups) else 0
    if n == logits.shape[0]:
        return logits, labels
    out, lab = [], []
    for g in range(n):
        m = groups == g
        y = labels[m]
        if bool((y != y[0]).any()):
            raise ValueError(f"bag {g} holds more than one label: {y.tolist()}")
        out.append(tau * torch.logsumexp(logits[m].float() / tau, dim=0))
        lab.append(y[0])
    return torch.stack(out), torch.stack(lab)
