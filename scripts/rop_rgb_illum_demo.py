"""Show, in full colour, what illumination correction would actually hand the model.

Everything in rop_enhance_demo.py ran on the green channel and displayed grey, which answers
"can a human see the ridge" but NOT "what tensor does the network get". This renders the RGB
the model would receive.

There are two ways to divide out a vignette, and they are not interchangeable:

  SHARED GAIN   Estimate the falloff once from luminance, then multiply R, G and B by the
                SAME field. Multiplying all three channels by one scalar leaves the ratios
                between them untouched, so hue is preserved exactly and only brightness
                geometry changes. The eye still sees a retina.

  PER CHANNEL   Divide each channel by its own blur. This also flattens the COLOUR, because
                each channel is independently pushed to its own local average — a red retina
                and an orange one both come out grey. It removes more device signature, and
                it destroys real information: choroidal pigmentation and haemorrhage are
                colour, not brightness.

Shared gain is the one to train on. Per-channel is shown so the difference is visible rather
than asserted — look at how much of the disease colour survives in each.

    .venv/bin/python scripts/rop_rgb_illum_demo.py
"""
import importlib.util
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

_spec = importlib.util.spec_from_file_location(
    "enh", Path(__file__).resolve().parent / "rop_enhance_demo.py")
enh = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(enh)

RELEASE = Path.home() / "Downloads" / "RetinAI_ROP_v1"


def masked_blur(x: np.ndarray, mask: np.ndarray, sigma: float) -> np.ndarray:
    """Blur that ignores the black surround — blur(x*m)/blur(m). Without the normalisation
    the estimate collapses toward zero at the rim and the division explodes exactly there."""
    m = mask.astype(np.float32)
    return (cv2.GaussianBlur(x * m, (0, 0), sigma) /
            (cv2.GaussianBlur(m, (0, 0), sigma) + 1e-6))


def illum_shared(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """One gain field from luminance, applied to all three channels. Hue is preserved."""
    lum = rgb @ np.array([0.299, 0.587, 0.114], np.float32)
    sigma = max(rgb.shape[:2]) / 30.0
    bg = masked_blur(lum, mask, sigma)
    gain = np.median(lum[mask]) / (bg + 1e-3)
    return np.clip(rgb * gain[..., None], 0, 1) * mask[..., None]


def illum_per_channel(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Each channel divided by its own blur. Flattens colour as well as brightness."""
    sigma = max(rgb.shape[:2]) / 30.0
    out = np.zeros_like(rgb)
    for c in range(3):
        ch = rgb[..., c]
        bg = masked_blur(ch, mask, sigma)
        out[..., c] = ch * (np.median(ch[mask]) / (bg + 1e-3))
    return np.clip(out, 0, 1) * mask[..., None]


def current_pipeline(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """What configs/rop*.yaml does today: CLAHE, on the L channel so colour survives."""
    lab = cv2.cvtColor((rgb * 255).astype(np.uint8), cv2.COLOR_RGB2LAB)
    lab[:, :, 0] = enh.clahe(lab[:, :, 0], clip=2.0)
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB) / 255.0 * mask[..., None]


def main() -> None:
    df = pd.read_csv(RELEASE / "classification/labels.csv")
    df = df[df.width >= 1000]
    want = ["normal", "stage_2", "stage_3", "arop"]
    rng = np.random.default_rng(42)
    picks = [df[df.icrop == c].iloc[rng.integers((df.icrop == c).sum())] for c in want]

    out = Path.home() / "Downloads" / "rop_enhancement"
    out.mkdir(parents=True, exist_ok=True)

    cols = ["1. ORIGINAL RGB\n(what the model gets today, pre-CLAHE)",
            "2. current pipeline\n(CLAHE on L, colour kept)",
            "3. ILLUMINATION - SHARED GAIN\n(recommended: hue preserved)",
            "4. illumination - per channel\n(colour flattened too)"]
    fig, axes = plt.subplots(len(picks), 4, figsize=(16.5, 4.15 * len(picks)),
                             constrained_layout=True)
    axes = np.atleast_2d(axes)

    for r, row in enumerate(picks):
        bgr = cv2.imread(str(RELEASE / row.path), cv2.IMREAD_COLOR)
        h, w = bgr.shape[:2]
        if max(h, w) > 1024:
            s = 1024 / max(h, w)
            bgr = cv2.resize(bgr, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
        mask = enh.retina_mask(bgr)
        rgb = (cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB) / 255.0).astype(np.float32)

        for c, im in enumerate([rgb * mask[..., None], current_pipeline(rgb, mask),
                                illum_shared(rgb, mask), illum_per_channel(rgb, mask)]):
            ax = axes[r, c]
            ax.imshow(np.clip(im, 0, 1))
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(cols[c], fontsize=10)
            if c == 0:
                ax.set_ylabel(f"{row.icrop}\n{row.source}", fontsize=10, rotation=0,
                              ha="right", va="center", labelpad=42)
    fig.suptitle("What illumination correction hands the model, in RGB", fontsize=14)
    p = out / "00_rgb_illumination.png"
    fig.savefig(p, dpi=105, facecolor="white")
    print(p)


if __name__ == "__main__":
    main()
