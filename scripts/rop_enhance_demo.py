"""Render one ROP photograph nine ways, to see which rendering makes the disease visible.

ROP is graded on three things a camera captures badly: the RIDGE (the raised demarcation line
at the vascular/avascular junction, which is what separates stage 1 from 2 from 3), PLUS
DISEASE (dilation and tortuosity of the posterior vessels), and the AVASCULAR PERIPHERY. All
three are structural, and all three sit in a photograph whose illumination falls off toward
the edge — which is exactly where the ridge lives.

So the enhancements here are not decoration. Each one targets a specific failure:

  green / red-free    Haemoglobin absorbs green, so vessels carry their highest contrast in
                      the green channel. The red channel is mostly choroidal glow and buries
                      them. Red-free is standard clinical practice, not a filter trick.
  illumination        Divide out a heavy blur to kill the centre-bright/edge-dark gradient,
                      so the periphery becomes readable at the same exposure as the pole.
  CLAHE               Local contrast, tile by tile, so the dim periphery gets stretched as
                      hard as the bright pole.
  vesselness          Hessian eigenvalues at several scales: responds to locally tubular
                      structure and ignores blobs and edges. This is the ridge/vessel map.
  black-hat           Morphological: what a closing added. Picks out dark curvilinear
                      structure thinner than the kernel — the demarcation line.
  blue-red LUT        The requested one. Intensity to colour, dark to blue and bright to red,
                      so a flat grey gradient becomes a hue change the eye reads instantly.

MASKING is load-bearing. A fundus photograph is a bright disc on a black surround, and every
local-contrast method will happily amplify sensor noise in that black surround into something
that looks like pathology. Everything below is computed inside the retinal mask only.

    .venv/bin/python scripts/rop_enhance_demo.py [--n 10] [--seed 42] [--out DIR]
"""
import argparse
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy import ndimage as ndi

RELEASE = Path.home() / "Downloads" / "RetinAI_ROP_v1"

# Dark -> blue, mid -> white, bright -> red. Applied to a contrast-equalised channel, this
# turns "slightly brighter than its surround" — which is all a ridge is — into a hue flip.
BLUE_RED = LinearSegmentedColormap.from_list(
    "blue_red", ["#08103a", "#1b4fd8", "#69a8ff", "#f4f4f4", "#ff9d5c", "#e02b1d", "#5c0700"])


def retina_mask(bgr: np.ndarray) -> np.ndarray:
    """The illuminated disc, as a boolean, ERODED away from its own edge.

    The erosion is the whole point. A contact camera leaves a bright specular rim right at
    the aperture, and that rim is a high-contrast curved line — so every method here reads it
    as anatomy. Un-eroded, the vesselness filter paints a perfect red circle around the image
    and the false positive is more prominent than any real vessel. Pulling the mask in by ~4%
    of the radius costs a sliver of true periphery and removes the artefact entirely.
    """
    grey = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    m = grey > max(8, grey.mean() * 0.18)
    m = ndi.binary_fill_holes(ndi.binary_opening(m, np.ones((7, 7))))
    if m.sum() < 0.05 * m.size:          # thresholding failed; use the whole frame
        m = np.ones_like(m, bool)
    # Erode via the distance transform rather than a fixed kernel. These discs are often
    # CLIPPED by the frame — Multi-View is 1600x1200, so the retina runs off the left and
    # right edges and the bright rim there is a straight line, not an arc. A distance
    # threshold pulls back from whatever the true boundary is, curved or straight, and scales
    # with the disc rather than the image.
    d = cv2.distanceTransform(m.astype(np.uint8), cv2.DIST_L2, 5)
    return d > 0.08 * d.max()


def denoise(green: np.ndarray) -> np.ndarray:
    """Edge-preserving smoothing BEFORE any local-contrast step.

    CLAHE and the illumination division both multiply up whatever is in the dim periphery,
    and in a JPEG that is compression noise. Without this the blue-red LUT renders as static
    and the black-hat map is pure speckle. A bilateral filter removes the grain while leaving
    vessel and ridge edges intact, which is exactly the trade we want.
    """
    return cv2.bilateralFilter(green, d=7, sigmaColor=35, sigmaSpace=7)


def norm(x: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Scale to 0..1 using percentiles INSIDE the mask, so the black surround cannot set the
    range (it otherwise pins the low end and flattens everything else)."""
    v = x[mask]
    if v.size == 0:
        return np.zeros_like(x, np.float32)
    lo, hi = np.percentile(v, [1, 99])
    if hi <= lo:
        lo, hi = float(v.min()), float(v.max() + 1e-6)
    return np.clip((x - lo) / (hi - lo), 0, 1).astype(np.float32)


def clahe(u8: np.ndarray, clip=3.0, grid=8) -> np.ndarray:
    return cv2.createCLAHE(clipLimit=clip, tileGridSize=(grid, grid)).apply(u8)


def illumination_corrected(green: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Divide by a heavy blur of itself. Removes the vignette that hides the periphery.

    The blur is NORMALISED BY THE MASK — blur(image*mask) / blur(mask) — rather than taken
    over the raw frame. A plain Gaussian pulls the black surround into the background estimate
    near the edge, so the background there goes to nearly zero and the division explodes into
    a bright ring exactly where the ridge lives. Dividing out the blurred mask cancels that
    fall-off and keeps the periphery on the same scale as the pole.
    """
    g = green.astype(np.float32)
    m = mask.astype(np.float32)
    sigma = max(green.shape) / 30.0
    bg = cv2.GaussianBlur(g * m, (0, 0), sigma) / (cv2.GaussianBlur(m, (0, 0), sigma) + 1e-6)
    out = np.where(mask, g / (bg + 1e-3), 0)
    return norm(out, mask)


def vesselness(green: np.ndarray, mask: np.ndarray,
               scales=(1.5, 2.5, 4.0, 6.0)) -> np.ndarray:
    """Multiscale Frangi vesselness, written out because scikit-image is not installed.

    At each scale the Hessian's eigenvalues say what the local structure is: a tube has one
    small eigenvalue along it and one large across it. Vessels and the ridge are DARK on the
    green channel, so the across-vessel eigenvalue is positive and negative responses drop.
    """
    g = norm(green.astype(np.float32), mask)
    best = np.zeros_like(g)
    for s in scales:
        # Scale-normalised second derivatives (the s**2 keeps scales comparable).
        gxx = ndi.gaussian_filter(g, s, order=(0, 2)) * s ** 2
        gyy = ndi.gaussian_filter(g, s, order=(2, 0)) * s ** 2
        gxy = ndi.gaussian_filter(g, s, order=(1, 1)) * s ** 2
        tmp = np.sqrt(np.maximum((gxx - gyy) ** 2 + 4 * gxy ** 2, 0))
        l1, l2 = (gxx + gyy + tmp) / 2, (gxx + gyy - tmp) / 2
        # order by magnitude: |a| <= |b|
        swap = np.abs(l1) > np.abs(l2)
        a = np.where(swap, l2, l1)
        b = np.where(swap, l1, l2)
        rb2 = (a / (b + 1e-10)) ** 2                 # blob-ness: 1 for a blob, 0 for a tube
        s2 = a ** 2 + b ** 2                         # structure-ness: 0 in flat background
        c = 0.5 * max(np.sqrt(s2[mask]).max(), 1e-6)
        v = np.exp(-rb2 / 0.5) * (1 - np.exp(-s2 / (2 * c ** 2)))
        v[b <= 0] = 0                                # keep dark-on-light only
        best = np.maximum(best, v)
    best[~mask] = 0
    return norm(best, mask)


def blackhat(green: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Dark curvilinear structure thinner than the kernel — the demarcation line."""
    k = max(7, int(max(green.shape) / 60) | 1)
    bh = cv2.morphologyEx(green, cv2.MORPH_BLACKHAT,
                          cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    bh[~mask] = 0
    return norm(bh.astype(np.float32), mask)


def to_rgb(x, cmap=None, mask=None):
    """Float 0..1 (optionally colour-mapped) to display RGB, with the surround forced black."""
    if cmap is None:
        rgb = np.dstack([x] * 3)
    else:
        rgb = plt.get_cmap(cmap)(x)[..., :3] if isinstance(cmap, str) else cmap(x)[..., :3]
    if mask is not None:
        rgb = rgb * mask[..., None]
    return np.clip(rgb, 0, 1)


def render(path: Path):
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        return None
    # Work at a bounded size: the methods are scale-sensitive and a 2040px image and a 640px
    # image should get comparable kernels.
    h, w = bgr.shape[:2]
    if max(h, w) > 1024:
        s = 1024 / max(h, w)
        bgr = cv2.resize(bgr, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)

    mask = retina_mask(bgr)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB) / 255.0
    green = denoise(bgr[:, :, 1])

    g_clahe = clahe(green, clip=2.0)
    illum = illumination_corrected(green, mask)
    # The colour-mapped panels need a SMOOTHER base than the greyscale ones. A LUT turns every
    # small intensity step into a hue step, so residual JPEG grain that is invisible in grey
    # becomes red-on-blue confetti and buries the structure the map exists to show. Smooth
    # first, then stretch gently.
    illum_smooth = cv2.bilateralFilter((illum * 255).astype(np.uint8), 9, 45, 9)
    lut_base = norm(clahe(illum_smooth, clip=1.5, grid=6).astype(np.float32), mask)
    ves = vesselness(green, mask)
    bh = blackhat(green, mask)


    # Vesselness painted over the original in red — the one panel a clinician can check
    # against the photograph, because the anatomy stays visible underneath.
    overlay = rgb.copy()
    a = np.clip(ves * 1.6, 0, 1)[..., None]
    overlay = overlay * (1 - a) + np.array([1.0, 0.15, 0.1]) * a

    return [
        ("1. original", rgb),
        ("2. green channel / red-free\n(vessel contrast)",
         to_rgb(norm(green.astype(np.float32), mask), None, mask)),
        ("3. CLAHE on green\n(local contrast)", to_rgb(norm(g_clahe.astype(np.float32), mask),
                                                       None, mask)),
        ("4. illumination corrected\n(vignette removed)", to_rgb(illum, None, mask)),
        ("5. LUT base\n(smoothed + gentle CLAHE)", to_rgb(lut_base, None, mask)),
        ("6. BLUE->RED LUT\n(dark=blue, bright=red)", to_rgb(lut_base, BLUE_RED, mask)),
        ("7. turbo LUT\n(on corrected green)", to_rgb(lut_base, "turbo", mask)),
        ("8. vesselness\n(ridge + vessels)", to_rgb(ves, None, mask)),
        ("9. black-hat\n(demarcation line)", to_rgb(bh, None, mask)),
        ("10. vesselness on original\n(red overlay)", overlay * mask[..., None]),
    ]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--min-width", type=int, default=1000,
                    help="skip low-resolution sources; enhancement is pointless at 512px")
    ap.add_argument("--out", default=str(Path.home() / "Downloads" / "rop_enhancement"))
    args = ap.parse_args()

    df = pd.read_csv(RELEASE / "classification/labels.csv")
    df = df[(df.width >= args.min_width) & df.icrop.isin(
        ["normal", "stage_1", "stage_2", "stage_3", "stage_4", "stage_5", "arop", "treated"])]

    # Stratified, not uniform: a uniform draw over this corpus returns mostly `normal`, and a
    # sheet of ten healthy eyes says nothing about whether a method reveals disease.
    rng = np.random.default_rng(args.seed)
    order = ["stage_3", "arop", "stage_2", "stage_1", "stage_4", "stage_5", "treated", "normal"]
    picks = []
    while len(picks) < args.n:
        for cls in order:
            pool = df[df.icrop == cls]
            if pool.empty:
                continue
            picks.append(pool.iloc[rng.integers(len(pool))])
            if len(picks) == args.n:
                break

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    grid_rows = []
    for i, row in enumerate(picks, 1):
        panels = render(RELEASE / row.path)
        if panels is None:
            print(f"  unreadable: {row.path}")
            continue
        # Keep a thumbnail of every panel for the master grid. Holding the full-size panels
        # would be ~30GB across ten images; 256px is plenty for comparing methods at a glance.
        grid_rows.append((row, [cv2.resize(np.clip(im, 0, 1).astype(np.float32), (256, 256),
                                           interpolation=cv2.INTER_AREA)
                                for _, im in panels], [t for t, _ in panels]))
        # constrained_layout, not tight_layout: the two-line panel titles on the bottom row
        # get clipped by tight_layout and only their second line survives.
        fig, axes = plt.subplots(2, 5, figsize=(22, 10.4), constrained_layout=True)
        for ax, (title, img) in zip(axes.ravel(), panels):
            ax.imshow(np.clip(img, 0, 1))
            ax.set_title(title, fontsize=10)
            ax.axis("off")
        fig.suptitle(f"{row.icrop.upper()}  |  {row.source}  |  {row.width}x{row.height}  "
                     f"|  {Path(row.path).name}", fontsize=14)
        f = out / f"{i:02d}_{row.icrop}_{row.source}.png"
        fig.savefig(f, dpi=95, facecolor="white")
        plt.close(fig)
        print(f"  {f.name}  ({row.width}x{row.height})")
    # One master grid: methods across, cases down. This is the view that answers "which
    # method should I use" — a single method's column can be read top to bottom across every
    # stage, which the per-image sheets cannot show.
    if grid_rows:
        n_r, n_c = len(grid_rows), len(grid_rows[0][1])
        fig, axes = plt.subplots(n_r, n_c, figsize=(2.05 * n_c, 2.15 * n_r + 0.8),
                                 constrained_layout=True)
        axes = np.atleast_2d(axes)
        for r, (row, imgs, titles) in enumerate(grid_rows):
            for c, im in enumerate(imgs):
                ax = axes[r, c]
                ax.imshow(im)
                ax.set_xticks([]); ax.set_yticks([])
                for sp in ax.spines.values():
                    sp.set_visible(False)
                if r == 0:
                    ax.set_title(titles[c].replace("\n", " "), fontsize=7.5, pad=4)
                if c == 0:
                    ax.set_ylabel(f"{row.icrop}\n{row.source}", fontsize=8, rotation=0,
                                  ha="right", va="center", labelpad=34)
        fig.suptitle("ROP enhancement methods (columns) x sampled cases (rows)", fontsize=13)
        gp = out / "00_master_grid.png"
        fig.savefig(gp, dpi=115, facecolor="white")
        plt.close(fig)
        print(f"  {gp.name}")

    print(f"\n{len(picks)} sheets -> {out}")


if __name__ == "__main__":
    main()
