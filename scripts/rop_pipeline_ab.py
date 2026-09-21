"""Current preprocessing vs a proposed one, on the same five images, one per source.

WHY THESE FIVE. One image from each source, not five of the same eye: the proposal is about
cross-site behaviour, so the comparison only means anything if you can see all five sites at
once and ask "do these still look like five different cameras".

WHAT THE PROPOSAL CHANGES, and why each step is here rather than something else.

`models/common/preprocessing.py` already records the finding that matters: post-CLAHE mean
blue is 53.6 / 73.3 / 74.5 across the three training sites against 87.6 at the held-out
hospital, and the R/B ratio is called "the single strongest surviving site signature in this
corpus". The site fingerprint is in COLOUR. An earlier attempt of mine to remove it via the
illumination vignette moved device separability only 46.7% -> 45.0% against 20% chance,
because the vignette is not where the signature lives. So:

  GREY-WORLD        Scale each channel so its in-mask mean matches the image's overall mean.
                    This targets the R/B ratio directly — the thing that was actually
                    measured — instead of the brightness falloff that was not.
  DENOISE           Bilateral, before any contrast step. CLAHE amplifies whatever is in the
                    dim periphery, and in a JPEG that is compression noise.
  ILLUMINATION      Shared-gain, so hue is untouched. Retained NOT for site invariance (that
                    claim failed its test) but because the ridge lives in the periphery and
                    this is what makes the periphery readable.
  CLAHE             Kept, but gentler, since the two steps above have already done part of it.

Everything else — circle crop, letterbox, ImageNet normalisation — is unchanged, and the
repo's own functions are imported rather than reimplemented so the "current" column is the
real thing and not my reading of it.

    .venv/bin/python scripts/rop_pipeline_ab.py
"""
import sys
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from models.common.preprocessing import circle_crop, letterbox_square, clahe  # noqa: E402

RELEASE = Path.home() / "Downloads" / "RetinAI_ROP_v1"
SIZE = 384          # what configs/rop*.yaml actually feed the network


def fundus_mask(rgb: np.ndarray) -> np.ndarray:
    grey = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    return grey > max(8, grey.mean() * 0.18)


def core_mask(mask: np.ndarray) -> np.ndarray:
    """The mask pulled back from its own edge, for MEASURING statistics only.

    The rim is dim, partly vignetted and partly specular, so its pixels skew both the chroma
    mean and the illumination background. Statistics come from the core; the corrections are
    still applied to the full retina, so no real periphery is thrown away — which matters
    here more than anywhere, because the ridge lives at the edge.
    """
    d = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
    core = d > 0.10 * d.max()
    return core if core.sum() > 0.05 * mask.sum() else mask


def grey_world(rgb: np.ndarray, mask: np.ndarray,
               strength: float = 0.8, gain_cap: float = 28.0) -> np.ndarray:
    """PARTIAL, BOUNDED colour-cast removal: pull chroma toward neutral without flattening it.

    Full RGB grey-world (equalise the channel means) collapses the cross-site R/B spread
    beautifully — SD 1.465 -> 0.063 — and ruins the pictures. A retina returns almost no blue
    light, so an image at R/B = 5.0 needs a ~5x gain on blue, and what gets amplified 5x is
    the blue channel's NOISE: Ostrava came out magenta-and-green blotched, HVDROPDB split
    blue-to-orange. That is optimising the proxy and destroying the signal.

    Bounding that gain helped but did not fix it, because the real culprit is CLIPPING.
    HVDROPDB's red channel is pinned at 255 across the bright pole, so any multiplicative gain
    lifts G and B there while R cannot move, and the hue slides as a function of position.

    So this works in LAB on the CHROMA channels only: shift a and b toward neutral by a
    bounded amount and leave L untouched. The correction is additive, cannot interact with a
    clipped channel, and changes no luminance — so vessel and ridge contrast is preserved
    exactly while the cast comes off. `strength` is the fraction of the cast removed and
    `gain_cap` bounds the shift in LAB units.

    The shift is applied INSIDE THE MASK ONLY. Applied to the whole frame it also moves the
    letterbox padding, whose neutral black (a=b=128) becomes vivid blue or green on the way
    back to RGB — and the padded area is a function of aspect ratio, which is the camera. That
    would paint a brand-new site cue into the input while removing an old one.
    """
    core = core_mask(mask)
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    for ch in (1, 2):                       # a = green-red, b = blue-yellow; L is left alone
        m = lab[..., ch][core].mean()
        shift = np.clip(strength * (m - 128.0), -gain_cap, gain_cap)
        lab[..., ch] = np.where(mask, lab[..., ch] - shift, lab[..., ch])
    out = cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB)
    return out * mask[..., None].astype(np.uint8)


def illum_shared(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Flatten the vignette with ONE gain field for all channels, so hue is preserved."""
    f = rgb.astype(np.float32) / 255.0
    lum = f @ np.array([0.299, 0.587, 0.114], np.float32)
    m = mask.astype(np.float32)
    sigma = max(rgb.shape[:2]) / 30.0
    bg = (cv2.GaussianBlur(lum * m, (0, 0), sigma) /
          (cv2.GaussianBlur(m, (0, 0), sigma) + 1e-6))
    # CLAMPED. At the dim, partly-vignetted rim the background estimate is near zero and the
    # raw gain runs to double figures, which multiplies that rim's chroma noise into the blue
    # and green fringes visible in the previous revision. Bounding the gain keeps the
    # correction where it is meaningful and leaves the rim approximately alone.
    gain = np.clip(np.median(lum[core_mask(mask)]) / (bg + 1e-3), 0.5, 2.2)
    out = np.clip(f * gain[..., None] * 255, 0, 255).astype(np.uint8)
    # Same rule as the chroma step: never let a correction leak into the letterbox padding.
    return out * mask[..., None].astype(np.uint8)


def pipeline_current(rgb: np.ndarray) -> np.ndarray:
    """configs/rop_staging.yaml: circle_crop -> letterbox -> CLAHE -> resize."""
    rgb = circle_crop(rgb)
    rgb = letterbox_square(rgb)
    rgb = clahe(rgb)
    return cv2.resize(rgb, (SIZE, SIZE), interpolation=cv2.INTER_AREA)


def pipeline_proposed(rgb: np.ndarray) -> np.ndarray:
    rgb = circle_crop(rgb)
    rgb = letterbox_square(rgb)
    mask = fundus_mask(rgb)
    rgb = grey_world(rgb, mask)
    rgb = cv2.bilateralFilter(rgb, d=7, sigmaColor=35, sigmaSpace=7)
    rgb = illum_shared(rgb, mask)
    rgb = clahe(rgb, clip=1.5)
    return cv2.resize(rgb, (SIZE, SIZE), interpolation=cv2.INTER_AREA)


def rb_ratio(rgb: np.ndarray) -> float:
    m = fundus_mask(rgb)
    if m.sum() < 50:
        return float("nan")
    return float(rgb[..., 0][m].mean() / (rgb[..., 2][m].mean() + 1e-6))


def main() -> None:
    df = pd.read_csv(RELEASE / "classification/labels.csv")
    rng = np.random.default_rng(7)
    # One per source, preferring a diseased eye so the comparison is not five normals.
    picks = []
    for src in ["ostrava", "ropvl", "multiview", "shenzhen", "hvdropdb"]:
        pool = df[(df.source == src) & (df.icrop != "normal")]
        if pool.empty:
            pool = df[df.source == src]
        picks.append(pool.iloc[rng.integers(len(pool))])

    fig, axes = plt.subplots(len(picks), 3, figsize=(11.5, 3.7 * len(picks)),
                             constrained_layout=True)
    axes = np.atleast_2d(axes)
    stats = []
    for r, row in enumerate(picks):
        bgr = cv2.imread(str(RELEASE / row.path), cv2.IMREAD_COLOR)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        cur, new = pipeline_current(rgb), pipeline_proposed(rgb)
        stats.append((row.source, rb_ratio(cur), rb_ratio(new)))

        for c, (im, ttl) in enumerate((
                (cv2.resize(rgb, (SIZE, SIZE), interpolation=cv2.INTER_AREA), "raw input"),
                (cur, "CURRENT\ncircle+letterbox+CLAHE"),
                (new, "PROPOSED\n+grey-world +denoise +illum"))):
            ax = axes[r, c]
            ax.imshow(im)
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(ttl, fontsize=11)
            if c == 0:
                ax.set_ylabel(f"{row.source}\n{row.icrop}", fontsize=10, rotation=0,
                              ha="right", va="center", labelpad=40)
            if c:
                ax.set_xlabel(f"R/B = {rb_ratio(im):.2f}", fontsize=9)

    cur_sd = np.std([s[1] for s in stats])
    new_sd = np.std([s[2] for s in stats])
    fig.suptitle(f"Same 5 images, one per source, at the {SIZE}px the model actually sees\n"
                 f"R/B spread across sites:  current SD={cur_sd:.3f}   "
                 f"proposed SD={new_sd:.3f}", fontsize=13)
    out = Path.home() / "Downloads" / "rop_enhancement" / "00_pipeline_ab.png"
    fig.savefig(out, dpi=110, facecolor="white")

    print(f"{'source':<11}{'R/B current':>13}{'R/B proposed':>14}")
    for s, a, b in stats:
        print(f"{s:<11}{a:>13.3f}{b:>14.3f}")
    print(f"{'SD across sites':<11}{cur_sd:>13.3f}{new_sd:>14.3f}")
    print(f"\n{out}")


if __name__ == "__main__":
    main()
