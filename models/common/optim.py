"""AdamW parameter groups with layer-wise learning-rate decay, and the OneCycleLR that honours them.

OneCycleLR rewrites every group's lr each step from ``max_lr``; a scalar ``max_lr`` would flatten
the decay on step 0, so it is passed as a list with one entry per group. (timm's ``lr_scale``
key is only read by timm's own schedulers, never by torch's.)
"""
from __future__ import annotations

import re

import torch

_EMBED = ("cls_token", "reg_token", "pos_embed", "patch_embed")


def _layer_id(name: str, depth: int) -> int:
    """0 = embeddings, i+1 = blocks.i, depth+1 = everything after the blocks (norm, head)."""
    m = re.match(r"blocks\.(\d+)\.", name)
    if m:
        return int(m.group(1)) + 1
    if name.split(".")[0] in _EMBED:
        return 0
    return depth + 1


def _group_name(layer: int, depth: int) -> str:
    if layer == 0:
        return "embed"
    return "head" if layer == depth + 1 else f"blocks.{layer - 1}"


def build_param_groups(model, base_lr: float, weight_decay: float,
                       layer_decay: float | None) -> list[dict]:
    """One group per layer id (lr = base_lr * layer_decay ** (depth+1 - id)); weight decay on all.

    With ``layer_decay`` absent / 1.0, or a model without ``.blocks``, returns a single group.
    """
    blocks = getattr(model, "blocks", None)
    if layer_decay is None or float(layer_decay) == 1.0 or blocks is None:
        return [{"params": list(model.parameters()), "lr": base_lr,
                 "weight_decay": weight_decay, "name": "all"}]
    depth = len(blocks)
    by_layer: dict[int, list] = {}
    for n, p in model.named_parameters():
        by_layer.setdefault(_layer_id(n, depth), []).append(p)
    return [{"params": by_layer[i], "lr": base_lr * float(layer_decay) ** (depth + 1 - i),
             "weight_decay": weight_decay, "name": _group_name(i, depth)}
            for i in sorted(by_layer)]


_CLASSIFIER = ("head", "fc", "classifier")   # the classifier attribute of a model without .blocks


def head_param_names(model) -> set[str]:
    """The parameters head warm-up trains: ``build_param_groups``' "head" group.

    With ``.blocks``: every name whose layer id is depth + 1 (after the blocks: norm, head). Without
    (one "all" group there, no layer ids): the names under the top-level classifier module,
    ``head``, ``fc`` or ``classifier``. Empty when a model has neither.
    """
    blocks = getattr(model, "blocks", None)
    names = [n for n, _ in model.named_parameters()]
    if blocks is None:
        return {n for n in names if n.split(".")[0] in _CLASSIFIER}
    depth = len(blocks)
    return {n for n in names if _layer_id(n, depth) == depth + 1}


def build_optimizer_and_scheduler(cfg, model, steps_per_epoch: int):
    groups = build_param_groups(model, float(cfg.train.lr),
                                float(cfg.train.get("weight_decay", 1e-4)),
                                cfg.train.get("layer_decay", None))
    optimizer = torch.optim.AdamW(groups)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=[g["lr"] for g in groups], steps_per_epoch=steps_per_epoch,
        epochs=int(cfg.train.epochs), pct_start=0.1, anneal_strategy="cos")
    return optimizer, scheduler


def format_group_lrs(optimizer) -> str:
    """``[lr] head=<v> blocks.0=<v>`` for layer-decay runs, ``[lr] all=<v>`` for one group."""
    by = {g.get("name", "all"): g["lr"] for g in optimizer.param_groups}
    if len(by) == 1:
        return f"[lr] all={next(iter(by.values())):.3e}"
    return f"[lr] head={by['head']:.3e} blocks.0={by['blocks.0']:.3e}"
