"""Fundus-image preprocessing — the single source of truth for BOTH training and
inference. Train-time and inference-time pipelines MUST be identical, so the web app
imports ``build_transforms(cfg, "test")`` from here too.

Steps (all toggled from the YAML ``preprocess`` block):
  - circle_crop : remove black borders around the fundus disc
  - ben_graham  : local-average colour subtraction (exposes DR microaneurysms)
  - clahe       : contrast-limited adaptive histogram equalisation on luminance

Implemented as a top-level picklable callable (``FundusPreprocess``) so it survives
DataLoader worker spawning.
"""
from __future__ import annotations

import cv2
import numpy as np
from PIL import Image
from torchvision import transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# ── individual operations (operate on RGB uint8 numpy arrays) ────────────────
def circle_crop(rgb: np.ndarray, tol: int = 7) -> np.ndarray:
    """Crop to the bounding box of the non-black fundus region."""
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    mask = gray > tol
    if not mask.any():
        return rgb
    coords = np.argwhere(mask)
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1
    return rgb[y0:y1, x0:x1]


def letterbox_square(rgb: np.ndarray, pad_value: int = 0) -> np.ndarray:
    """Pad to a square with black bars, WITHOUT changing the aspect ratio.

    Why this exists: ``transforms.Resize((size, size))`` squashes a non-square image, and the
    amount of squash is a function of the source camera's aspect ratio — 1.16-1.21 across our
    three training sites versus 1.333 at the held-out hospital. That makes "how distorted the
    vessels look" a site cue, i.e. exactly the shortcut this project already got burned by
    (a classifier on width/height alone scored AUC 0.911). Letterboxing applies the *same*
    geometric transform to every source: pad to square, then resize once, isotropically.
    """
    h, w = rgb.shape[:2]
    if h == w:
        return rgb
    side = max(h, w)
    out = np.full((side, side, rgb.shape[2]), pad_value, dtype=rgb.dtype)
    y0, x0 = (side - h) // 2, (side - w) // 2
    out[y0:y0 + h, x0:x0 + w] = rgb
    return out


def ben_graham(rgb: np.ndarray, sigma: float = 10.0) -> np.ndarray:
    """Subtract local average colour: addWeighted(img,4,blur,-4,128)."""
    blur = cv2.GaussianBlur(rgb, (0, 0), sigmaX=sigma)
    out = cv2.addWeighted(rgb, 4, blur, -4, 128)
    return np.clip(out, 0, 255).astype(np.uint8)


def clahe(rgb: np.ndarray, clip: float = 2.0, grid: int = 8) -> np.ndarray:
    """CLAHE on the L channel of LAB."""
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=clip, tileGridSize=(grid, grid)).apply(l)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2RGB)


class FundusPreprocess:
    """Configurable fundus preprocessing: PIL.Image -> PIL.Image."""

    def __init__(self, circle: bool = True, ben: bool = False,
                 ben_sigma: float = 10.0, do_clahe: bool = True,
                 letterbox: bool = False):
        self.circle = circle
        self.ben = ben
        self.ben_sigma = ben_sigma
        self.do_clahe = do_clahe
        # letterbox defaults OFF so every already-trained checkpoint (DR, glaucoma, ROP-binary)
        # keeps the exact preprocessing it was fitted with. The staging model opts in.
        self.letterbox = letterbox

    def __call__(self, img: Image.Image) -> Image.Image:
        rgb = np.asarray(img.convert("RGB"))
        if self.circle:
            rgb = circle_crop(rgb)
        if self.letterbox:
            rgb = letterbox_square(rgb)
        if self.do_clahe:
            rgb = clahe(rgb)
        if self.ben:
            rgb = ben_graham(rgb, self.ben_sigma)
        return Image.fromarray(rgb)

    def __repr__(self) -> str:
        return (f"FundusPreprocess(circle={self.circle}, ben={self.ben}, "
                f"ben_sigma={self.ben_sigma}, clahe={self.do_clahe}, "
                f"letterbox={self.letterbox})")


class ResolutionJitter:
    """Randomly destroy resolution, then restore the size — a sharpness-domain randomiser.

    Our four sites are separable by native resolution (Shenzhen 512x512, ROP-VL ~1450 square,
    Ostrava 640x480/12402/1440x1080, Multi-View 1600x1200). After a common resize, the residue
    of that is SHARPNESS: an upsampled 512px image and a downsampled 1450px image look different
    even at identical pixel dimensions. Gaussian blur only perturbs sharpness in one direction;
    this samples a random intermediate resolution and round-trips through it, so neither
    sharpness nor the resample signature is a stable class cue.

    Top-level class (not a lambda) so DataLoader workers can pickle it.
    """

    def __init__(self, p: float = 0.5, min_frac: float = 0.33):
        self.p = p
        self.min_frac = min_frac

    def __call__(self, img: Image.Image) -> Image.Image:
        import random
        if random.random() >= self.p:
            return img
        w, h = img.size
        f = random.uniform(self.min_frac, 1.0)
        small = img.resize((max(8, int(w * f)), max(8, int(h * f))), Image.BILINEAR)
        return small.resize((w, h), Image.BILINEAR)

    def __repr__(self) -> str:
        return f"ResolutionJitter(p={self.p}, min_frac={self.min_frac})"


# ── transform builders ───────────────────────────────────────────────────────
def _fundus_from_cfg(cfg) -> FundusPreprocess:
    p = cfg.preprocess
    return FundusPreprocess(
        circle=bool(p.get("circle_crop", True)),
        ben=bool(p.get("ben_graham", False)),
        ben_sigma=float(p.get("ben_graham_sigma", 10.0)),
        do_clahe=bool(p.get("clahe", True)),
        letterbox=bool(p.get("letterbox", False)),
    )


def build_transforms(cfg, split: str = "train"):
    """Return a torchvision transform for ``split`` ("train" | "val"/"test").

    The fundus preprocessing + resize + normalise are identical across splits;
    only augmentation differs (train only).
    """
    size = int(cfg.preprocess.image_size)
    pre = _fundus_from_cfg(cfg)
    resize = transforms.Resize((size, size))
    normalize = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)

    if split == "train":
        aug = [
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
        ]
        # SITE RANDOMISATION — opt-in, for cross-hospital training only.
        #
        # The default jitter above is luminance-only (brightness/contrast/saturation), so it
        # leaves the per-channel R/B ratio untouched — and that ratio is the single strongest
        # surviving site signature in this corpus (post-CLAHE mean blue is 53.6 / 73.3 / 74.5
        # across the three training sites versus 87.6 at the held-out hospital). A model can
        # read "which hospital" straight off colour and, because class mixture differs by
        # site, get the label for free. These three ops randomise the channels that carry site
        # identity while leaving vessel/ridge STRUCTURE — the actual disease signal — intact.
        if bool(cfg.train.get("site_randomization", False)):
            aug += [
                transforms.ColorJitter(hue=0.05, saturation=0.3),   # attacks the R/B ratio
                transforms.RandomGrayscale(p=0.10),                 # forces structural features
                ResolutionJitter(p=0.5, min_frac=0.33),             # attacks sharpness/FOV scale
            ]
        if bool(cfg.train.get("strong_aug", False)):
            aug += [
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1),
                                        scale=(0.88, 1.12), shear=8),
                transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
                transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
            ]
        tail = [transforms.ToTensor(), normalize]
        if bool(cfg.train.get("strong_aug", False)):
            tail.append(transforms.RandomErasing(p=0.35, scale=(0.02, 0.15)))
        # With site randomisation on, the fixed Resize becomes a RandomResizedCrop so that the
        # apparent field-of-view diameter — another per-camera constant — varies within a site
        # as much as it varies between sites. ratio is kept near 1.0 because the letterbox has
        # already made every input square; the crop is for scale, not for aspect distortion.
        head = resize
        if bool(cfg.train.get("site_randomization", False)):
            head = transforms.RandomResizedCrop(size, scale=(0.6, 1.0), ratio=(0.95, 1.05))
        return transforms.Compose([pre, head, *aug, *tail])

    return transforms.Compose([pre, resize, transforms.ToTensor(), normalize])


def build_tta_transforms(cfg):
    """5-view TTA (original + H-flip + V-flip + ±10° rotation), sharing the same
    fundus preprocessing as training so inference matches."""
    size = int(cfg.preprocess.image_size)
    pre = _fundus_from_cfg(cfg)
    normalize = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    resize = transforms.Resize((size, size))

    def make(extra):
        return transforms.Compose([pre, resize, *extra,
                                   transforms.ToTensor(), normalize])

    return [
        make([]),
        make([transforms.RandomHorizontalFlip(p=1.0)]),
        make([transforms.RandomVerticalFlip(p=1.0)]),
        make([transforms.RandomRotation(degrees=(10, 10))]),
        make([transforms.RandomRotation(degrees=(-10, -10))]),
    ]
