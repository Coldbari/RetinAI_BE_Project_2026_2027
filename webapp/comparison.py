"""The demo's COMPARISON model: a second opinion shown beside arm B, never a decision.

A teammate's `rop_foundation` package (Nov 2026): RETFound-DINOv2 ViT-L/14 (Moorfields
retina-pretrained, 304 M weights, frozen) with LoRA adapters of rank 16 / alpha 32 on the
attention q-k-v and projection and both MLP layers of all 24 blocks (6.3 M trainable weights
incl. the head), a linear 4-class head (Normal, Stage 1, Stage 2, Stage 3), three seeds whose
softmax vectors are averaged. Its own preprocessing is reproduced here exactly — black-border
crop, resize to 320 px, CLAHE on the L channel, resize to 280 px, ImageNet statistics — and its
4-view flip test-time augmentation is optional (`comparison.tta` in the registry).

Why it is only a comparison: it was trained on a photo-level split of 3,141 of our photos, so
its test numbers carry infant leakage, and it has seen 390 development-half and 331
sealed-half photos. It has no validated screening line and no Stage 4/5 or AP-ROP class.

Deployment: scripts/deploy_comparison_weights.py writes ONE fp16 backbone file (the three
seeds' backbones are bit-identical, which the script verifies) and one small adapter file per
seed; this loads the backbone once and swaps the adapters in before each member's forward pass.
Checkpoints are read with `weights_only=True`: no code from the source package runs here.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

try:
    import cv2
except Exception:                                   # pragma: no cover - optional at import
    cv2 = None

ARCH = "vit_large_patch14_dinov2"
LORA_TARGETS = ("attn.qkv", "attn.proj", "mlp.fc1", "mlp.fc2")
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
SOURCE_CLASSES = ["NORMAL", "STAGE1", "STAGE2", "STAGE3"]
CLASSES = ["Normal", "Stage 1", "Stage 2", "Stage 3"]        # our names for the same classes
NOT_COVERED = ("Stage 4/5", "AP-ROP")


class LoRALinear(nn.Module):
    """base(x) + (x A^T B^T) * alpha/r — the source package's adapter, dropout omitted (eval)."""

    def __init__(self, base: nn.Linear, r: int, alpha: int):
        super().__init__()
        self.base, self.scale = base, alpha / r
        self.lora_A = nn.Parameter(torch.zeros(r, base.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, r))

    def forward(self, x):
        return self.base(x) + (x @ self.lora_A.t() @ self.lora_B.t()) * self.scale


class _Net(nn.Module):
    """Parameter names match the source checkpoints: backbone.*, head.*, ord_head.*."""

    def __init__(self, backbone, n_cls: int):
        super().__init__()
        self.backbone = backbone
        d = backbone.num_features
        self.head, self.ord_head = nn.Linear(d, n_cls), nn.Linear(d, n_cls - 1)

    def forward(self, x):
        return self.head(self.backbone(x))              # ord_head is an unused training aid


def build_network(cfg: dict) -> _Net:
    import timm
    bb = timm.create_model(ARCH, pretrained=False, img_size=int(cfg["img"]), num_classes=0,
                           global_pool="token")
    targets = LORA_TARGETS if cfg.get("lora_mlp") else LORA_TARGETS[:2]
    for blk in bb.blocks:
        for t in targets:
            parent, leaf = t.split(".")
            mod = getattr(blk, parent)
            setattr(mod, leaf, LoRALinear(getattr(mod, leaf), int(cfg["lora_r"]),
                                          int(cfg["lora_alpha"])))
    return _Net(bb, int(cfg["n_cls"]))


def source_preprocess(image: Image.Image, store: int) -> np.ndarray:
    """The source package's `preprocess`: black-border crop -> resize -> CLAHE (L of LAB). RGB uint8."""
    if cv2 is None:
        raise RuntimeError("opencv is required for the comparison model's preprocessing")
    bgr = np.ascontiguousarray(np.asarray(image.convert("RGB"))[..., ::-1])
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    ys, xs = np.where(gray > 10)
    if len(ys) >= 100:
        bgr = bgr[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    img = cv2.resize(bgr, (store, store), interpolation=cv2.INTER_AREA)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    lab[..., 0] = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(lab[..., 0])
    return cv2.cvtColor(cv2.cvtColor(lab, cv2.COLOR_LAB2BGR), cv2.COLOR_BGR2RGB)


def to_input(image: Image.Image, store: int, img: int) -> torch.Tensor:
    """(1, 3, img, img) float tensor, exactly the source package's eval transform."""
    import torchvision.transforms.v2.functional as F
    x = torch.from_numpy(np.ascontiguousarray(source_preprocess(image, store))).permute(2, 0, 1)
    x = F.resize(x, [img, img], antialias=True)
    x = F.normalize(F.to_dtype(x, torch.float32, scale=True), list(MEAN), list(STD))
    return x.unsqueeze(0)


def compare(ours: str | None, theirs: str) -> dict:
    """How the comparison model's stage relates to arm B's: agree / disagree / not covered."""
    if ours is None:
        return {"status": "no_reference", "ours": None, "theirs": theirs,
                "message": "Our model produced no stage for this photo."}
    if ours in NOT_COVERED:
        return {"status": "not_covered", "ours": ours, "theirs": theirs,
                "message": f"Our model says {ours}. The comparison model only knows Normal to "
                           f"Stage 3, so it cannot agree or disagree; it answered {theirs}."}
    if ours == theirs:
        return {"status": "agree", "ours": ours, "theirs": theirs,
                "message": f"Both models say {ours}."}
    gap = abs(CLASSES.index(ours) - CLASSES.index(theirs))
    return {"status": "disagree", "ours": ours, "theirs": theirs, "adjacent": gap == 1,
            "message": f"Our model says {ours}; the comparison model says {theirs}"
                       + (" (one stage apart)." if gap == 1 else f" ({gap} stages apart).")}


def eye_from_comparisons(per_photo: list[dict]) -> dict | None:
    """The worst-view rule on the comparison model's per-photo stages (same rule as ours)."""
    cands = []
    for i, p in enumerate(per_photo):
        c = p.get("comparison") if p.get("gradable", True) else None
        if not c or not c.get("prediction"):
            continue
        idx = CLASSES.index(c["prediction"])
        cands.append((idx, float(c["scores"][c["prediction"]]), i))
    if not cands:
        return None
    top = max(c[0] for c in cands)
    idx, _, drv = max((c for c in cands if c[0] == top), key=lambda c: (c[1], -c[2]))
    return {"stage_index": idx, "stage": CLASSES[idx], "driving_photo": drv}


class ComparisonModel:
    """Loads once, fails soft: a missing file leaves `loaded` False and the app still boots."""

    CLASSES = CLASSES
    NOTE = ("Second opinion only. This model never decides the verdict. Trained by a teammate on "
            "a photo-level split of our photos; its numbers carry infant leakage and it has seen "
            "part of our test data, so use new photos.")

    def __init__(self, device, spec: dict | None):
        self.loaded, self.card, self.load_seconds, self.spec = False, None, None, spec or {}
        if not spec:
            return
        try:
            t0 = time.perf_counter()
            bb_path = Path(spec["backbone"])
            adapter_paths = [Path(a["file"]) for a in spec["adapters"]]
            missing = [str(p) for p in [bb_path, *adapter_paths] if not p.exists()]
            if missing:
                print(f"[webapp] comparison model unavailable: missing {missing} "
                      f"(run scripts/deploy_comparison_weights.py)")
                return
            bb = torch.load(bb_path, map_location="cpu", weights_only=True)
            self.cfg, self.img, self.store = dict(bb["cfg"]), int(bb["img"]), int(bb["store"])
            if list(bb["classes"]) != SOURCE_CLASSES:
                raise ValueError(f"unexpected classes {bb['classes']}")
            self.device, self.tta = device, bool(spec.get("tta", True))
            self.net = build_network(self.cfg)
            state = {k: v.float() for k, v in bb["state_dict"].items()}
            self.members = []
            for p in adapter_paths:
                a = torch.load(p, map_location="cpu", weights_only=True)
                sd = {k: v.float() for k, v in a["state_dict"].items()}
                if not self.members:                       # first: load everything, strictly
                    self.net.load_state_dict({**state, **sd}, strict=True)
                self.members.append({"seed": int(a["seed"]),
                                     "state": {k: v.to(device) for k, v in sd.items()},
                                     "source_test": (a.get("source_meta") or {}).get("test")})
            self.net = self.net.to(device).eval()
            self.card = self._card()
            self.probs(Image.new("RGB", (self.store, self.store), (120, 40, 20)))   # warm up
            self.load_seconds = round(time.perf_counter() - t0, 2)
            self.loaded = True
            print(f"[webapp] comparison model loaded: {self.card['name']} on {device} in "
                  f"{self.load_seconds}s | {len(self.members)} adapter sets, tta={self.tta}")
        except Exception as e:                           # pragma: no cover - defensive boot
            print(f"[webapp] comparison model unavailable: {e}")

    def _card(self) -> dict:
        s, cfg = dict(self.spec.get("scores") or {}), self.cfg
        return {"name": self.spec["name"], "credit": self.spec.get("credit", ""),
                "role": "comparison only — never decides",
                "architecture": f"RETFound-DINOv2 ViT-L/14, frozen, LoRA rank {cfg['lora_r']} "
                                f"alpha {cfg['lora_alpha']} on attention + MLP layers, "
                                f"{self.img} px input, 4-way head",
                "preprocessing": f"black-border crop, resize {self.store}, CLAHE, resize {self.img}, "
                                 f"ImageNet normalisation" + (", 4-view flip TTA" if self.tta else ""),
                "classes": list(CLASSES), "n_members": len(self.members),
                "seeds": [m["seed"] for m in self.members], "scores": s,
                "caveat": (f"{s.get('accuracy', 0) * 100:.1f}% accuracy, QWK {s.get('qwk', 0):.3f} on "
                           f"its own {s.get('n_test_photos', '?')}-photo test split, 4 classes. "
                           f"{s.get('split', '')}; {s.get('overlap_with_our_tests', '')}. Not "
                           f"externally validated."),
                "note": self.NOTE}

    @torch.no_grad()
    def member_probs(self, image: Image.Image) -> np.ndarray:
        """(n_members, 4) softmax vectors; each member = the shared backbone + its adapters."""
        x = to_input(image, self.store, self.img).to(self.device)
        if self.tta:
            x = torch.cat([x, x.flip(3), x.flip(2), x.flip((2, 3))])
        out = []
        for m in self.members:
            res = self.net.load_state_dict(m["state"], strict=False)
            assert not res.unexpected_keys, res.unexpected_keys
            out.append(self.net(x).float().softmax(1).mean(0, keepdim=True))
        return torch.cat(out).cpu().numpy()

    def probs(self, image: Image.Image):
        p = self.member_probs(image).mean(0)
        return p, int(p.argmax())

    def predict(self, image: Image.Image, ours: str | None, probs=None, idx=None) -> dict | None:
        if not self.loaded:
            return None
        if probs is None:
            probs, idx = self.probs(image)
        return {"model": self.card["name"], "credit": self.card["credit"],
                "prediction": CLASSES[idx],
                "scores": {c: round(float(probs[i]) * 100, 1) for i, c in enumerate(CLASSES)},
                "any_rop_probability": round(float(1.0 - probs[0]) * 100, 1),
                "n_members": len(self.members), "classes": list(CLASSES),
                "not_covered": list(NOT_COVERED), "agreement": compare(ours, CLASSES[idx]),
                "note": self.NOTE}
