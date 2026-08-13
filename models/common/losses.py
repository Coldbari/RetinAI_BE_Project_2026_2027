"""Loss functions and prediction decoders.

- Cross-entropy (with label smoothing + optional class-balanced weights)
- Focal loss (focuses on hard / minority examples)
- Ordinal regression (single scalar output, rounded to a grade) for DR

``build_loss(cfg, class_counts)`` returns ``(loss_fn, decode_fn)`` where ``decode_fn``
turns raw model outputs into integer class predictions, so the training/eval loop is
agnostic to the head type.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def class_balanced_weights(class_counts, beta: float = 0.9999) -> torch.Tensor:
    """Cui et al. (2019) class-balanced weights from per-class counts."""
    counts = np.asarray(class_counts, dtype=np.float64)
    counts = np.maximum(counts, 1.0)
    eff = 1.0 - np.power(beta, counts)
    w = (1.0 - beta) / eff
    w = w / w.sum() * len(counts)
    return torch.tensor(w, dtype=torch.float32)


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, weight: torch.Tensor | None = None,
                 label_smoothing: float = 0.0):
        super().__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        # Register as a buffer so .to(device) moves the class weights with the module.
        self.register_buffer("weight", weight)

    def forward(self, logits, target):
        ce = F.cross_entropy(logits, target, weight=self.weight,
                             label_smoothing=self.label_smoothing, reduction="none")
        pt = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce).mean()


class OrdinalRegressionLoss(nn.Module):
    """Treat ordered grades as a scalar regression target (MSE on normalized grade)."""

    def __init__(self, num_classes: int):
        super().__init__()
        self.num_classes = num_classes

    def forward(self, output, target):
        pred = output.squeeze(1).float()
        return F.mse_loss(pred, target.float())


def build_loss(cfg, class_counts):
    """Return ``(loss_fn, decode_fn, needs_float_target)`` from the config."""
    name = cfg.train.get("loss", "ce")
    num_classes = int(cfg.data.num_classes)
    label_smoothing = float(cfg.train.get("label_smoothing", 0.0))

    weight = None
    if bool(cfg.train.get("class_balanced", False)) and name != "ordinal":
        weight = class_balanced_weights(class_counts)

    if name == "ce":
        loss_fn = nn.CrossEntropyLoss(weight=weight, label_smoothing=label_smoothing)
        decode = lambda out: out.argmax(1)
        return loss_fn, decode, False

    if name == "focal":
        loss_fn = FocalLoss(gamma=float(cfg.train.get("focal_gamma", 2.0)),
                            weight=weight, label_smoothing=label_smoothing)
        decode = lambda out: out.argmax(1)
        return loss_fn, decode, False

    if name == "ordinal":
        loss_fn = OrdinalRegressionLoss(num_classes)
        hi = num_classes - 1
        decode = lambda out: out.squeeze(1).round().clamp(0, hi).long()
        return loss_fn, decode, True

    raise ValueError(f"Unknown loss '{name}' (ce | focal | ordinal)")


def probabilities(output, head: str = "classification"):
    """Softmax probabilities for classification heads (used by TTA / calibration)."""
    if head == "regression":
        # No calibrated probability for a scalar head; return the normalized score.
        return output.squeeze(1)
    return torch.softmax(output, dim=1)
