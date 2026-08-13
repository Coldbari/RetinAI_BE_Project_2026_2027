"""Model factory — one ``build_model(name, num_classes)`` for every architecture in
the comparison sweep. Each returns a standard torchvision backbone with its classifier
replaced by ``Dropout -> Linear(in_features, num_classes)`` (num_classes=1 for the DR
ordinal/regression head).

``get_target_layer`` returns the last conv layer per architecture for Grad-CAM.
"""
from __future__ import annotations

import torch.nn as nn
from torchvision import models

SUPPORTED = [
    "efficientnet_b0",
    "efficientnet_b3",
    "efficientnet_v2_s",
    "mobilenet_v3_large",
    "resnet50",
    "densenet121",
    "convnext_tiny",
]

# ── timm backbones ───────────────────────────────────────────────────────────
# A name is routed to timm if it carries an explicit "timm:" prefix OR contains a "." —
# the latter is timm's own pretrained-tag convention ("convnext_tiny.in12k_ft_in1k_384"),
# and no torchvision name contains a dot, so the two namespaces cannot collide.
#
# timm is an OPTIONAL dependency: it is imported lazily inside _build_timm, so every
# existing torchvision path (and the webapp, which imports this module) still works on a
# machine without timm installed.
#
# KAGGLE / OFFLINE NOTE: timm downloads weights from HuggingFace on first use. Kaggle
# kernels may run without internet, so upload the checkpoints as a Kaggle dataset and point
# HF_HOME at it, or pass pretrained=False and load_state_dict() the file directly.

# The five backbones of the ROP staging comparison. Chosen to span architecture FAMILIES —
# running five variants of one family would teach us nothing — at a capacity appropriate to
# ~3.1k training images. All Apache-2.0 or torchvision; no CC-BY-NC checkpoints, which would
# block any commercial clinical use.
STAGING_BENCHMARK = {
    # modern CNN. Best <30M backbone AND best OOD family in "Battle of the Backbones".
    "convnext_in12k": "timm:convnext_tiny.in12k_ft_in1k_384",
    # conv+attention hybrid. Best accuracy/param; +1.1 ImageNet-A, +1.5 ImageNet-R vs ConvNeXt-T.
    "caformer": "timm:caformer_s18.sail_in22k_ft_in1k_384",
    # the anchor: the architecture already used in this repo, so results tie to prior work.
    "effnetv2s": "efficientnet_v2_s",
    # pure ViT arm — does transformer inductive bias help or hurt at n=3.1k?
    "deit3": "timm:deit3_small_patch16_384.fb_in22k_ft_in1k",
    # capacity floor at 4.0M with a 6-class head (5.3M is the 1000-class ImageNet figure;
    # measured counts are 27.8 / 24.3 / 20.2 / 21.8 / 4.0M for the five below). If this WINS
    # cross-site, every 20M+ model is memorising —
    # which with 89 images in the rarest class is a live hypothesis, not a formality.
    "effnetb0": "efficientnet_b0",
}


def _is_timm(name: str) -> bool:
    return name.startswith("timm:") or "." in name


def _build_timm(name: str, num_classes: int, pretrained: bool, dropout: float) -> nn.Module:
    """Create a timm backbone with a fresh head. timm sizes the head from num_classes."""
    try:
        import timm
    except ImportError as e:  # pragma: no cover - depends on the environment
        raise ImportError(
            f"arch '{name}' needs timm, which is not installed. "
            "Add `timm>=1.0` to requirements.txt (it is preinstalled on Kaggle)."
        ) from e
    model_id = name[5:] if name.startswith("timm:") else name
    return timm.create_model(model_id, pretrained=pretrained,
                             num_classes=num_classes, drop_rate=dropout)


def _weights(name: str, pretrained: bool):
    if not pretrained:
        return None
    table = {
        "efficientnet_b0": models.EfficientNet_B0_Weights.DEFAULT,
        "efficientnet_b3": models.EfficientNet_B3_Weights.DEFAULT,
        "efficientnet_v2_s": models.EfficientNet_V2_S_Weights.DEFAULT,
        "mobilenet_v3_large": models.MobileNet_V3_Large_Weights.DEFAULT,
        "resnet50": models.ResNet50_Weights.DEFAULT,
        "densenet121": models.DenseNet121_Weights.DEFAULT,
        "convnext_tiny": models.ConvNeXt_Tiny_Weights.DEFAULT,
    }
    return table[name]


def build_model(name: str, num_classes: int, pretrained: bool = True,
                dropout: float = 0.3) -> nn.Module:
    """Build a classifier with a fresh ``Dropout -> Linear`` head."""
    name = name.lower()
    # Friendly aliases for the five-model staging comparison ("caformer" -> the full timm id).
    name = STAGING_BENCHMARK.get(name, name)
    if _is_timm(name):
        return _build_timm(name, num_classes, pretrained, dropout)
    if name not in SUPPORTED:
        raise ValueError(
            f"Unsupported arch '{name}'. Choose from {SUPPORTED}, "
            f"a staging alias {sorted(STAGING_BENCHMARK)}, or any timm id "
            "(prefix 'timm:' or use a tagged name like 'convnext_tiny.in12k_ft_in1k_384')."
        )

    weights = _weights(name, pretrained)

    if name.startswith("efficientnet"):
        net = getattr(models, name)(weights=weights)
        in_f = net.classifier[-1].in_features
        net.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_f, num_classes))
    elif name == "mobilenet_v3_large":
        net = models.mobilenet_v3_large(weights=weights)
        in_f = net.classifier[-1].in_features
        net.classifier[-1] = nn.Linear(in_f, num_classes)
    elif name == "resnet50":
        net = models.resnet50(weights=weights)
        in_f = net.fc.in_features
        net.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_f, num_classes))
    elif name == "densenet121":
        net = models.densenet121(weights=weights)
        in_f = net.classifier.in_features
        net.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_f, num_classes))
    elif name == "convnext_tiny":
        net = models.convnext_tiny(weights=weights)
        in_f = net.classifier[-1].in_features
        net.classifier[-1] = nn.Linear(in_f, num_classes)

    return net


def build_from_cfg(cfg, num_classes: int | None = None,
                   pretrained: bool | None = None) -> nn.Module:
    n = num_classes if num_classes is not None else int(cfg.data.num_classes)
    if cfg.model.get("head", "classification") == "regression":
        n = 1
    # When loading trained weights for inference, pass pretrained=False to skip the
    # ImageNet download (the weights get overwritten by load_state_dict anyway).
    pt = pretrained if pretrained is not None else bool(cfg.model.get("pretrained", True))
    return build_model(cfg.model.arch, n, pretrained=pt,
                       dropout=float(cfg.model.get("dropout", 0.3)))


def get_target_layer(model: nn.Module, arch: str) -> nn.Module:
    """Last conv layer for Grad-CAM, per architecture."""
    arch = STAGING_BENCHMARK.get(arch.lower(), arch.lower())
    if _is_timm(arch):
        # timm models have no single shared layout, so resolve by structure rather than by
        # name. ConvNeXt/CAFormer expose .stages; ViT-family models expose .blocks — note a
        # ViT target gives token-grid attention maps, which need reshaping before they look
        # like a CAM, so treat DeiT3 saliency as indicative rather than directly comparable.
        for attr in ("stages", "blocks", "layers"):
            block = getattr(model, attr, None)
            if block is not None and len(block):
                return block[-1]
        norm = getattr(model, "norm_pre", None) or getattr(model, "norm", None)
        if norm is not None:
            return norm
        raise ValueError(f"Could not resolve a Grad-CAM target layer for timm arch '{arch}'")
    if arch.startswith("efficientnet"):
        return model.features[-1]
    if arch == "mobilenet_v3_large":
        return model.features[-1]
    if arch == "resnet50":
        return model.layer4[-1]
    if arch == "densenet121":
        return model.features[-1]
    if arch == "convnext_tiny":
        return model.features[-1]
    raise ValueError(f"No Grad-CAM target layer registered for '{arch}'")
