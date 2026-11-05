"""Arm F (PROTOCOL_V2 Amendment 4b): self-ensembling (French et al., ICLR 2018) on unlabelled photos.

With ``train.uda.enabled`` every source batch of 16 labelled photos is joined by
``target_batch`` UNLABELLED photos of the target manifest (the Multi-View dev half):

- the TEACHER is an exponential moving average of the student (``ema_decay``), a copy of the
  student taken after its head is built, updated after each optimizer step, kept in eval mode
  with no gradient;
- each target photo gives a WEAK view (resize + horizontal/vertical flips) to the teacher and a
  STRONG view (the source training transform, B's augmentation) to the student;
- target loss = mean over the target photos of ``mask * sum_c (p_student - p_teacher)^2``, where
  ``mask`` = teacher max probability >= ``conf_threshold``;
- total = source loss + w(t) * target loss, w rising linearly from 0 to ``weight`` over the
  first ``ramp_epochs`` epochs, by step;
- the target loader cycles (663 dev photos against ~150 source batches per epoch).

The target manifest is read with ``usecols=TARGET_COLUMNS``: its label columns are never parsed, so
no training code can see a dev label. Validation, val_predictions.npz and dev_predictions.npz come
from the teacher (``train_utils.train_model`` saves the teacher's weights as weights.pth).
"""
from __future__ import annotations

import copy
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from .datasets import cached_path
from .preprocessing import (_fundus_stage, build_transforms, cache_root_from_cfg,
                            normalize_from_cfg)

TARGET_COLUMNS = ["image_path"]          # the only column of the target manifest training reads
TARGET_NAME = "test_dev.csv"             # Amendment 4(b): the target is the Multi-View dev half


def uda_settings(cfg) -> dict | None:
    """The ``train.uda`` block, checked, when ``enabled`` is true; else None."""
    u = cfg.train.get("uda")
    if not u:
        return None
    on = u.get("enabled", False)
    if not isinstance(on, bool):        # "--set ...enabled=yes" would otherwise read as true
        raise ValueError(f"train.uda.enabled must be true or false, got {on!r}")
    if not on:
        return None
    if u.get("method") != "self_ensembling":
        raise ValueError(f"train.uda.method must be self_ensembling, got {u.get('method')!r}")
    s = {"method": "self_ensembling", "ema_decay": float(u.get("ema_decay")),
         "conf_threshold": float(u.get("conf_threshold")), "weight": float(u.get("weight")),
         "ramp_epochs": int(u.get("ramp_epochs")), "target_manifest": str(u.get("target_manifest")),
         "target_batch": int(u.get("target_batch"))}
    if not 0.0 < s["ema_decay"] < 1.0:
        raise ValueError(f"train.uda.ema_decay must lie in (0, 1), got {s['ema_decay']}")
    if not 0.0 <= s["conf_threshold"] <= 1.0:
        raise ValueError(f"train.uda.conf_threshold must lie in [0, 1], got {s['conf_threshold']}")
    if s["weight"] < 0 or s["ramp_epochs"] < 0 or s["target_batch"] < 1:
        raise ValueError(f"train.uda needs weight >= 0, ramp_epochs >= 0, target_batch >= 1: {s}")
    if Path(s["target_manifest"]).name != TARGET_NAME:
        raise ValueError(f"train.uda.target_manifest must be the dev half ({TARGET_NAME}), "
                         f"got {s['target_manifest']!r}")
    return s


def load_target_frame(path) -> pd.DataFrame:
    """The target manifest's image paths ONLY: ``usecols`` keeps the label columns unparsed."""
    return pd.read_csv(path, usecols=TARGET_COLUMNS)


def weak_transform(cfg):
    """The teacher's view: the same fundus stage, resize and normalisation as validation, plus flips."""
    size = int(cfg.preprocess.image_size)
    return transforms.Compose([*_fundus_stage(cfg), transforms.Resize((size, size)),
                               transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(),
                               transforms.ToTensor(), normalize_from_cfg(cfg)])


class TargetDataset(Dataset):
    """Unlabelled photos: ``dataset[i] -> (weak view, strong view)`` of one image, and no label."""

    def __init__(self, frame: pd.DataFrame, weak, strong, cache_root: str | None = None):
        extra = sorted(set(frame.columns) - set(TARGET_COLUMNS))
        if extra:
            raise ValueError(f"the target frame must carry image paths only; it has {extra}")
        self.paths = frame["image_path"].tolist()
        self.weak, self.strong, self.cache_root = weak, strong, cache_root

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        p = self.paths[i]
        # A missing or unreadable file raises (no blank fallback): a black target photo would be
        # an easy, confident teacher prediction that teaches nothing.
        img = Image.open(cached_path(p, self.cache_root) if self.cache_root else p).convert("RGB")
        return self.weak(img), self.strong(img)


def target_loader(cfg, settings: dict, num_workers: int, pin_memory: bool) -> DataLoader:
    frame = load_target_frame(settings["target_manifest"])
    ds = TargetDataset(frame, weak_transform(cfg), build_transforms(cfg, "train"),
                       cache_root_from_cfg(cfg))
    dl = DataLoader(ds, batch_size=settings["target_batch"], shuffle=True, drop_last=True,
                    num_workers=num_workers, pin_memory=pin_memory,
                    persistent_workers=num_workers > 0)
    print(f"[uda] target: {len(ds)} unlabelled photos from {settings['target_manifest']} "
          f"({TARGET_COLUMNS} only) -> {len(dl)} batches of {settings['target_batch']}, cycled; "
          f"ema_decay={settings['ema_decay']} conf_threshold={settings['conf_threshold']} "
          f"weight={settings['weight']} ramp_epochs={settings['ramp_epochs']}", flush=True)
    return dl


def cycle(loader):
    """Batches of ``loader`` without end: a new pass (a new shuffle) whenever one is exhausted."""
    while True:
        yield from loader


class EMATeacher:
    """``model`` = a frozen copy of the student, moved towards it by ``update`` after each step."""

    def __init__(self, student: torch.nn.Module, decay: float):
        self.model = copy.deepcopy(student)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.decay = float(decay)

    @torch.no_grad()
    def update(self, student: torch.nn.Module) -> None:
        for t, s in zip(self.model.parameters(), student.parameters()):
            t.mul_(self.decay).add_(s.detach(), alpha=1.0 - self.decay)
        for t, s in zip(self.model.buffers(), student.buffers()):
            t.copy_(s)


def consistency_loss(student_logits: torch.Tensor, teacher_logits: torch.Tensor,
                     threshold: float) -> tuple[torch.Tensor, torch.Tensor]:
    """(mean over target rows of mask * squared softmax difference summed over classes, mask rate).

    The teacher side is a fixed target (detached); both softmaxes are computed in fp32.
    """
    ps = torch.softmax(student_logits.float(), dim=1)
    pt = torch.softmax(teacher_logits.detach().float(), dim=1)
    mask = (pt.max(dim=1).values >= threshold).float()
    return (mask * ((ps - pt) ** 2).sum(dim=1)).mean(), mask.mean()


def ramp_weight(step: int, ramp_steps: int, weight: float) -> float:
    """``weight`` * min(1, step / ramp_steps): 0 at the first step, ``weight`` from ``ramp_steps`` on."""
    if ramp_steps <= 0:
        return float(weight)
    return float(weight) * min(1.0, step / ramp_steps)
