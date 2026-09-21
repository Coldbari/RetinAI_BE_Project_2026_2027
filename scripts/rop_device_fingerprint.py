"""Does illumination correction actually erase the device fingerprint? Measure, don't assert.

The case for training on illumination-corrected input was never "it looks nicer" — on one
image in RGB it barely looks like anything. The case is that each camera has its own
brightness falloff, that falloff is a per-device signature, and the corpus already has a
documented shortcut where device predicts label (image dimensions alone score AUC 0.911).

So the question is a distribution question, not a picture question: do the five sources look
LESS like five different sources after correction?

Two measurements:

  RADIAL PROFILE  Mean brightness as a function of normalised distance from the centre of the
                  retinal disc. This IS the vignette, plotted. If the sources' curves lie on
                  top of each other after correction, the photometric signature is gone.

  SEPARABILITY    Fit a small logistic regression to predict WHICH SOURCE an image came from,
                  using only the radial profile as features, and score it by cross-validated
                  accuracy. Chance is 1/n_sources. Before correction this should be near
                  perfect; how far it falls afterwards is the actual result.

The second is the honest one, because two curves can look close on a plot and still be
trivially separable.

    .venv/bin/python scripts/rop_device_fingerprint.py [--per-source 80]
"""
import argparse
import importlib.util
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

_spec = importlib.util.spec_from_file_location(
    "enh", Path(__file__).resolve().parent / "rop_enhance_demo.py")
enh = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(enh)
_spec2 = importlib.util.spec_from_file_location(
    "rgbd", Path(__file__).resolve().parent / "rop_rgb_illum_demo.py")
rgbd = importlib.util.module_from_spec(_spec2)
_spec2.loader.exec_module(rgbd)

RELEASE = Path.home() / "Downloads" / "RetinAI_ROP_v1"
NBINS = 24


def radial_profile(lum: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Mean brightness in NBINS rings, from disc centre to disc edge.

    Radius is normalised by the disc's own extent, not the frame, so a 640x480 RetCam and a
    2040x2040 Neo are compared on the same axis — otherwise this would just re-measure the
    resolution confound instead of the illumination one.
    """
    ys, xs = np.nonzero(mask)
    cy, cx = ys.mean(), xs.mean()
    yy, xx = np.mgrid[:lum.shape[0], :lum.shape[1]]
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    rmax = r[mask].max()
    if rmax <= 0:
        return np.full(NBINS, np.nan)
    rn = r / rmax
    out = np.full(NBINS, np.nan)
    for i in range(NBINS):
        sel = mask & (rn >= i / NBINS) & (rn < (i + 1) / NBINS)
        if sel.sum() > 20:
            out[i] = lum[sel].mean()
    # Interpolate any empty ring so every image yields a complete feature vector.
    idx = np.arange(NBINS)
    ok = ~np.isnan(out)
    return np.interp(idx, idx[ok], out[ok]) if ok.sum() > 2 else np.full(NBINS, np.nan)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-source", type=int, default=80)
    args = ap.parse_args()

    df = pd.read_csv(RELEASE / "classification/labels.csv")
    rng = np.random.default_rng(0)
    rows = []
    for src, grp in df.groupby("source"):
        take = grp.iloc[rng.permutation(len(grp))[:args.per_source]]
        for _, r in take.iterrows():
            bgr = cv2.imread(str(RELEASE / r.path), cv2.IMREAD_COLOR)
            if bgr is None:
                continue
            h, w = bgr.shape[:2]
            if max(h, w) > 640:
                s = 640 / max(h, w)
                bgr = cv2.resize(bgr, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
            mask = enh.retina_mask(bgr)
            if mask.sum() < 500:
                continue
            rgb = (cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB) / 255.0).astype(np.float32)
            w3 = np.array([0.299, 0.587, 0.114], np.float32)
            before = radial_profile(rgb @ w3, mask)
            corr = rgbd.illum_shared(rgb, mask)
            after = radial_profile(corr @ w3, mask)
            # Third arm. The AFTER curves come out flat but at DIFFERENT HEIGHTS per source,
            # so the fingerprint simply moves from falloff-shape into overall brightness and
            # the classifier reads the offset instead. Standardising each image to zero mean /
            # unit variance inside its own mask removes that offset as well. (ImageNet
            # normalisation does NOT do this: it subtracts one fixed constant for every image,
            # which preserves per-image level differences exactly.)
            lum = corr @ w3
            v = lum[mask]
            zs = np.zeros_like(lum)
            zs[mask] = (v - v.mean()) / (v.std() + 1e-6)
            after_z = radial_profile(zs, mask)
            if np.isnan(before).any() or np.isnan(after).any() or np.isnan(after_z).any():
                continue
            rows.append((src, before, after, r.patient_id, after_z))
        print(f"  {src}: {sum(1 for x in rows if x[0] == src)} images")

    srcs = sorted({r[0] for r in rows})
    y = np.array([srcs.index(r[0]) for r in rows])
    Xb = np.stack([r[1] for r in rows])
    Xa = np.stack([r[2] for r in rows])
    Xz = np.stack([r[4] for r in rows])

    # GROUPED by patient. Ungrouped, two frames of the same infant land either side of the
    # split and the classifier can recognise the eye rather than the camera, which inflates
    # "device separability" into something closer to "patient separability".
    groups = np.array([r[3] for r in rows])

    def sep(X):
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
        return cross_val_score(clf, X, y, groups=groups, cv=GroupKFold(n_splits=5),
                               scoring="accuracy").mean()

    acc_b, acc_a, acc_z = sep(Xb), sep(Xa), sep(Xz)
    chance = 1.0 / len(srcs)

    print("\n=== DEVICE SEPARABILITY from the radial brightness profile alone ===")
    print(f"  sources           : {len(srcs)} ({', '.join(srcs)})")
    print(f"  images            : {len(rows)}")
    print(f"  chance            : {chance:.3f}")
    print(f"  BEFORE correction : {acc_b:.3f}")
    print(f"  AFTER  shared gain: {acc_a:.3f}   "
          f"({(acc_b - acc_a) / max(acc_b - chance, 1e-9) * 100:5.1f}% of gap closed)")
    print(f"  AFTER  + per-image z-score: {acc_z:.3f}   "
          f"({(acc_b - acc_z) / max(acc_b - chance, 1e-9) * 100:5.1f}% of gap closed)")

    fig, axes = plt.subplots(1, 3, figsize=(18, 4.8), constrained_layout=True)
    x = (np.arange(NBINS) + 0.5) / NBINS
    for X, ax, title in ((Xb, axes[0], f"BEFORE  (source predicted {acc_b:.0%})"),
                         (Xa, axes[1], f"shared-gain  ({acc_a:.0%})"),
                         (Xz, axes[2], f"shared-gain + z-score  ({acc_z:.0%})")):
        for i, s in enumerate(srcs):
            m = X[y == i]
            mu, sd = m.mean(0), m.std(0)
            ax.plot(x, mu, label=f"{s} (n={len(m)})", lw=2)
            ax.fill_between(x, mu - sd, mu + sd, alpha=0.13)
        ax.set_title(title)
        ax.set_xlabel("normalised radius  (0 = disc centre, 1 = disc edge)")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("mean luminance")
    axes[0].legend(fontsize=8)
    fig.suptitle(f"Per-device illumination fingerprint  (chance = {chance:.0%})", fontsize=13)
    p = Path.home() / "Downloads" / "rop_enhancement" / "00_device_fingerprint.png"
    fig.savefig(p, dpi=110, facecolor="white")
    print(f"\n{p}")


if __name__ == "__main__":
    main()
