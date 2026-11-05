"""Fundus-image preprocessing — the single source of truth for BOTH training and
inference. Train-time and inference-time pipelines MUST be identical, so the web app
imports ``build_transforms(cfg, "test")`` from here too.

Steps (all toggled from the YAML ``preprocess`` block):
  - circle_crop : remove black borders around the fundus disc
  - ben_graham  : local-average colour subtraction (exposes DR microaneurysms)
  - clahe       : contrast-limited adaptive histogram equalisation on luminance
  - normalize   : tensor statistics, "imagenet" (default) or "half" (mean = std = 0.5);
                  read by BOTH build_transforms and build_tta_transforms
  - cached      : (default false) the Dataset reads images that already went through
                  circle_crop -> letterbox -> clahe (``preprocess.cache_root`` holds them), so the
                  builders leave that stage out. The flags above are left exactly as configured
                  (true for ROP): a consumer that reads ORIGINAL images with the same config (the
                  web app, the locked-set scorer) runs with ``cached`` off and still applies
                  them. See ``cache_root_from_cfg``.

Implemented as a top-level picklable callable (``FundusPreprocess``) so it survives
DataLoader worker spawning.
"""
from __future__ import annotations

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# What ``preprocess.normalize`` may select. "imagenet" is the default, and what every checkpoint
# trained before v3 was fitted with. "half" is mean = std = 0.5 (pixels mapped to [-1, 1]), the
# scaling the RETFound-Green ViT was pretrained with.
NORMALIZATIONS = {
    "imagenet": (IMAGENET_MEAN, IMAGENET_STD),
    "half": ([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
}


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


class GammaJitter:
    """With probability ``p``, ``img ** gamma`` on [0, 1] intensities, gamma ~ U(lo, hi), the same
    gamma for every channel. PIL -> PIL (a 256-entry lookup table per band).

    ColorJitter's brightness and contrast are linear in pixel value; a gamma bends the tone curve,
    lifting or crushing mid-tones while black and white stay put. Arm I's recipe (Amendment 5b).
    Top-level class so DataLoader workers can pickle it.
    """

    def __init__(self, lo: float, hi: float, p: float = 0.5):
        self.lo, self.hi, self.p = lo, hi, p

    def __call__(self, img: Image.Image) -> Image.Image:
        if torch.rand(1).item() >= self.p:
            return img
        g = torch.empty(1).uniform_(self.lo, self.hi).item()
        lut = [round(255 * (v / 255) ** g) for v in range(256)]
        return img.point(lut * len(img.getbands()))

    def __repr__(self) -> str:
        return f"GammaJitter(lo={self.lo}, hi={self.hi}, p={self.p})"


class GaussianNoise:
    """With probability ``p``, add N(0, sigma^2) to every pixel, then clamp to [0, 1].

    A TENSOR transform: it sits after ``ToTensor`` and before ``Normalize``, so ``sigma`` is a
    fraction of the full intensity range whatever normalisation the config selects. Arm I's recipe
    (Amendment 5b). Top-level class so DataLoader workers can pickle it.
    """

    def __init__(self, sigma: float, p: float = 0.5):
        self.sigma, self.p = sigma, p

    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        if torch.rand(1).item() >= self.p:
            return t
        return (t + torch.randn_like(t) * self.sigma).clamp_(0.0, 1.0)

    def __repr__(self) -> str:
        return f"GaussianNoise(sigma={self.sigma}, p={self.p})"


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


def normalize_from_cfg(cfg) -> transforms.Normalize:
    """The ``Normalize`` layer selected by ``preprocess.normalize`` (default "imagenet").

    Read by BOTH ``build_transforms`` and ``build_tta_transforms``, so a model is trained,
    validated and served under the same statistics. Anything but a known name raises: falling
    back to the default would train a model under statistics its config did not ask for.
    """
    name = cfg.preprocess.get("normalize", "imagenet")
    if not isinstance(name, str) or name not in NORMALIZATIONS:
        raise ValueError(f"preprocess.normalize must be one of {sorted(NORMALIZATIONS)}, "
                         f"got {name!r}")
    mean, std = NORMALIZATIONS[name]
    return transforms.Normalize(mean, std)


def cache_root_from_cfg(cfg) -> str | None:
    """Where the Dataset reads pre-processed images from: ``preprocess.cache_root``, or ``None``
    when ``preprocess.cached`` is off (the default).

    ``cached: true`` is a promise that the files the Dataset opens have ALREADY been through the
    fundus steps. Two halves keep it: the transform builders leave that stage out, and the
    Dataset is handed this root. Both read the key through this one function, so neither can be
    on without the other, and a root that is missing is an error here, never a quiet fall-back to
    the original images, which would reach the model un-preprocessed.

    ``cached`` must be a real boolean. ``--set preprocess.cached=no`` arrives as the string
    "no", which ``bool()`` calls true: the cache would switch on for the person who meant it off.
    """
    p = cfg.preprocess
    cached = p.get("cached", False)
    if not isinstance(cached, bool):
        raise ValueError(f"preprocess.cached must be true or false, got {cached!r}")
    if not cached:
        return None
    root = p.get("cache_root")
    if not root:
        raise ValueError("preprocess.cached is true but preprocess.cache_root is not set: there "
                         "is nowhere to read the pre-processed images from")
    return str(root)


def _fundus_stage(cfg) -> list:
    """The pipeline's fundus stage: ``[FundusPreprocess]``, or ``[]`` when the images are cached."""
    return [] if cache_root_from_cfg(cfg) is not None else [_fundus_from_cfg(cfg)]


# Arm I's augmentation block (protocol Amendment 5b). Every key is required and no other is allowed:
# the values are fixed by the protocol, so a missing key must not fall back to a default and a
# misspelt one must not become a key nothing reads.
AUG_V4_KEYS = ("enabled", "rotation", "min_crop_scale", "gamma", "gamma_p", "noise_sigma",
               "noise_p", "fourier_block", "fourier_p")


def aug_v4_settings(cfg) -> dict | None:
    """The ``train.aug_v4`` block, checked, when ``enabled`` is true; else None (arm B, v3).

    ``rotation`` (degrees, +-) and ``min_crop_scale`` replace v3's 15 and 0.6; ``gamma`` is a
    ``[lo, hi]`` pair applied with ``gamma_p``; ``noise_sigma`` with ``noise_p``; ``fourier_block``
    (odd, see ``fourier_aug``) and ``fourier_p`` are read by the trainer for the batch-level swap.
    """
    a = cfg.train.get("aug_v4")
    if not a:
        return None
    on = a.get("enabled", False)
    if not isinstance(on, bool):        # "--set ...enabled=yes" would otherwise read as true
        raise ValueError(f"train.aug_v4.enabled must be true or false, got {on!r}")
    if not on:
        return None
    have, want = set(a.keys()), set(AUG_V4_KEYS)
    if have != want:
        raise ValueError(f"train.aug_v4 needs exactly the keys {list(AUG_V4_KEYS)}: missing "
                         f"{sorted(want - have)}, unknown {sorted(have - want)}")
    gamma, block = a.get("gamma"), a.get("fourier_block")
    if not (isinstance(gamma, (list, tuple)) and len(gamma) == 2):
        raise ValueError(f"train.aug_v4.gamma must be a [lo, hi] pair, got {gamma!r}")
    if isinstance(block, bool) or not isinstance(block, int) or block < 1 or block % 2 == 0:
        raise ValueError(f"train.aug_v4.fourier_block must be a positive odd int, got {block!r}")
    s = {"enabled": True, "rotation": float(a.get("rotation")),
         "min_crop_scale": float(a.get("min_crop_scale")),
         "gamma": [float(gamma[0]), float(gamma[1])], "gamma_p": float(a.get("gamma_p")),
         "noise_sigma": float(a.get("noise_sigma")), "noise_p": float(a.get("noise_p")),
         "fourier_block": block, "fourier_p": float(a.get("fourier_p"))}
    if not (0.0 <= s["rotation"] <= 180.0 and 0.0 < s["min_crop_scale"] <= 1.0
            and 0.0 < s["gamma"][0] <= s["gamma"][1] and s["noise_sigma"] >= 0.0
            and all(0.0 <= s[k] <= 1.0 for k in ("gamma_p", "noise_p", "fourier_p"))):
        raise ValueError("train.aug_v4 needs 0 <= rotation <= 180, 0 < min_crop_scale <= 1, "
                         f"0 < gamma lo <= hi, noise_sigma >= 0 and probabilities in [0, 1]: {s}")
    return s


def build_transforms(cfg, split: str = "train"):
    """Return a torchvision transform for ``split`` ("train" | "val"/"test").

    The fundus preprocessing + resize + normalise are identical across splits;
    only augmentation differs (train only). With ``preprocess.cached`` the fundus stage is
    left out: the Dataset reads images that already went through it. With ``train.aug_v4``
    enabled (arm I) the train pipeline takes that block's rotation and crop scale, plus a gamma
    jitter and pixel noise; without it (arm B) it is exactly v3's.
    """
    size = int(cfg.preprocess.image_size)
    pre = _fundus_stage(cfg)
    resize = transforms.Resize((size, size))
    normalize = normalize_from_cfg(cfg)

    if split == "train":
        v4 = aug_v4_settings(cfg)
        site = bool(cfg.train.get("site_randomization", False))
        if v4 and not site:
            raise ValueError("train.aug_v4.min_crop_scale sets site randomisation's "
                             "RandomResizedCrop: train.aug_v4 needs train.site_randomization true")
        aug = [
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(v4["rotation"] if v4 else 15),
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
        if site:
            aug += [
                transforms.ColorJitter(hue=0.05, saturation=0.3),   # attacks the R/B ratio
                transforms.RandomGrayscale(p=0.10),                 # forces structural features
                ResolutionJitter(p=0.5, min_frac=0.33),             # attacks sharpness/FOV scale
            ]
        if v4:  # gamma right after the colour-jitter ops, still on the PIL image
            last = max(i for i, t in enumerate(aug) if isinstance(t, transforms.ColorJitter))
            aug.insert(last + 1, GammaJitter(*v4["gamma"], p=v4["gamma_p"]))
        if bool(cfg.train.get("strong_aug", False)):
            aug += [
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1),
                                        scale=(0.88, 1.12), shear=8),
                transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
                transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
            ]
        tail = [transforms.ToTensor(), normalize]
        if v4:  # pixel noise on the [0, 1] tensor, before Normalize
            tail.insert(1, GaussianNoise(v4["noise_sigma"], p=v4["noise_p"]))
        if bool(cfg.train.get("strong_aug", False)):
            tail.append(transforms.RandomErasing(p=0.35, scale=(0.02, 0.15)))
        # With site randomisation on, the fixed Resize becomes a RandomResizedCrop so that the
        # apparent field-of-view diameter — another per-camera constant — varies within a site
        # as much as it varies between sites. ratio is kept near 1.0 because the letterbox has
        # already made every input square; the crop is for scale, not for aspect distortion.
        head = resize
        if site:
            min_scale = v4["min_crop_scale"] if v4 else 0.6
            head = transforms.RandomResizedCrop(size, scale=(min_scale, 1.0), ratio=(0.95, 1.05))
        return transforms.Compose([*pre, head, *aug, *tail])

    return transforms.Compose([*pre, resize, transforms.ToTensor(), normalize])


def build_tta_transforms(cfg):
    """5-view TTA (original + H-flip + V-flip + ±10° rotation), sharing the same
    fundus preprocessing and normalisation statistics as training so inference matches
    (including leaving the fundus stage out under ``preprocess.cached``)."""
    size = int(cfg.preprocess.image_size)
    pre = _fundus_stage(cfg)
    normalize = normalize_from_cfg(cfg)
    resize = transforms.Resize((size, size))

    def make(extra):
        return transforms.Compose([*pre, resize, *extra,
                                   transforms.ToTensor(), normalize])

    return [
        make([]),
        make([transforms.RandomHorizontalFlip(p=1.0)]),
        make([transforms.RandomVerticalFlip(p=1.0)]),
        make([transforms.RandomRotation(degrees=(10, 10))]),
        make([transforms.RandomRotation(degrees=(-10, -10))]),
    ]
