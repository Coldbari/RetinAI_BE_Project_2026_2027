"""Dataset Quality Assurance (W2) — run BEFORE training so the thesis can say
"we first cleaned and verified the data".

Checks: corrupted/unreadable files, duplicates (perceptual hash), blurry images
(variance of Laplacian), label validity, resolution + class distributions. Emits a
``dataset_report.pdf`` and returns a cleaned manifest (corrupted rows dropped,
duplicate/blurry rows flagged).

Header-only checks (corruption, resolution, class balance) run over every image;
the heavier decode-based checks (blur, perceptual hash) run over a random sample
(``deep_sample``) to stay tractable on the 35k-image EyePACS set.
"""
from __future__ import annotations

import random
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

try:
    import cv2
    _HAS_CV2 = True
except Exception:
    _HAS_CV2 = False

try:
    import imagehash
    _HAS_IMAGEHASH = True
except Exception:
    _HAS_IMAGEHASH = False


def _is_readable(path) -> bool:
    try:
        with Image.open(path) as im:
            im.verify()
        return True
    except Exception:
        return False


def _resolution(path):
    try:
        with Image.open(path) as im:
            return im.size  # (w, h) from header — fast
    except Exception:
        return None


def _blur_score(path) -> float:
    if not _HAS_CV2:
        return float("nan")
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return float("nan")
    return float(cv2.Laplacian(img, cv2.CV_64F).var())


# ── serving-time gradability gate (Phase 0B) ─────────────────────────────────
# Before this gate existed, all three models returned a positive finding on 100% of
# non-retinal input — random noise, a solid grey square, and two of the project's own
# matplotlib charts. See docs/REMEDIATION_PLAN.md Phase 0B.
#
# Deliberately classical, not learned: a trained fundus-vs-not classifier would put a new
# unvalidated model underneath three models whose failure modes we are containing. These
# thresholds are calibrated on TRAIN splits only (never on test or on the HRF cohort) by
# scripts/calibrate_gate.py, which writes results/gate_thresholds.json.

# Three colour signals are needed, because different non-fundus inputs fail differently and
# NO SINGLE ONE catches them all (measured on the four probes plus 300 infant train images):
#   * grey / white / chart images  -> near-zero saturation, but redness and hue look fine
#   * random RGB noise             -> saturation 99.9 (ABOVE the infant minimum of 38.1!),
#                                     caught only by its flat hue distribution
#   * achromatic frames            -> hue is undefined and reads as red, caught by saturation
# An image is rejected if ANY signal says non-retinal.
GATE_DEFAULTS = {
    # Mean luminance of the four 12% corner patches. The circular fundus aperture leaves
    # corners near-black in every cohort (p99 <= 19.8); real-world photographs fill them
    # (measured warm indoor scenes: >130). PRIMARY non-fundus check as of 2026-08-19 -
    # found by a user handing the app a photo of furniture and getting "ROP Detected 91.5%".
    "max_corner_lum": 40.0,
    # Fraction of pixels with a near-zero Laplacian. Graphics have big exactly-flat areas
    # (charts p50 = 0.84); photographs never do (fundus max = 0.39 across all cohorts).
    "max_flat_frac": 0.6,
    # mean(R)/mean(B) over the field of view. Retinal tissue is red-dominant - but pale
    # premature retina under some cameras dips below 1.0 (ROP-VL/Shenzhen p1 = 0.81), so
    # this bound must stay permissive; the corner check carries the structural burden.
    "min_redness": 0.75,
    # mean HSV saturation. Kills greyscale, white and chart-like input (charts measure ~11;
    # the palest real cohort's p1 is 25).
    "min_saturation": 20.0,
    # fraction of FOV pixels whose hue is red/orange. DISABLED (0.0): calibrated on Ostrava
    # it sat at 0.34, silently rejecting real Shenzhen/Multi-View fundus whose p1 is
    # 0.03-0.09. Kept as a metric for reporting only.
    "min_red_hue_frac": 0.0,
    # R > G > B ordering. Held on only 95% of infant training images, so OFF by default —
    # requiring it would falsely reject 1 in 20 real patients.
    "require_rgb_order": False,
    # Laplacian variance over the field of view.
    "min_blur_var": 5.0,
    # Mean luminance inside the field of view (0-255).
    "min_luminance": 18.0,
    "max_luminance": 235.0,
    # Fraction of the frame that is not black border. A tiny FOV means a bad capture.
    "min_fov_ratio": 0.05,
}

_GATE_CACHE = {}


def load_gate_thresholds(path="results/gate_thresholds.json"):
    """Calibrated thresholds if present, else the conservative defaults."""
    if path in _GATE_CACHE:
        return _GATE_CACHE[path]
    import json
    thr = dict(GATE_DEFAULTS)
    p = Path(path)
    if p.exists():
        try:
            thr.update(json.loads(p.read_text()).get("thresholds", {}))
        except Exception:
            pass
    _GATE_CACHE[path] = thr
    return thr


def gradability_metrics(image, work_size=512):
    """Cheap classical descriptors of an RGB PIL image, computed over its field of view."""
    rgb = np.asarray(image.convert("RGB"))
    h, w = rgb.shape[:2]
    if max(h, w) > work_size and _HAS_CV2:
        s = work_size / max(h, w)
        rgb = cv2.resize(rgb, (max(1, int(w * s)), max(1, int(h * s))),
                         interpolation=cv2.INTER_AREA)

    if _HAS_CV2:
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    else:
        gray = rgb.mean(axis=2).astype(np.uint8)

    mask = gray > 7                      # same tol as preprocessing.circle_crop
    fov_ratio = float(mask.mean())
    if mask.sum() < 64:                  # essentially an empty / black frame
        return {"fov_ratio": fov_ratio, "redness": 0.0, "rgb_ordered": False,
                "saturation": 0.0, "red_hue_frac": 0.0, "corner_lum": float(gray.mean()),
                "flat_frac": 1.0,
                "blur_var": 0.0, "luminance": float(gray.mean()), "empty": True}

    px = rgb[mask].astype(np.float32)
    r, g, b = px[:, 0].mean(), px[:, 1].mean(), px[:, 2].mean()
    if _HAS_CV2:
        blur = float(cv2.Laplacian(gray, cv2.CV_64F)[mask].var())
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        saturation = float(hsv[:, :, 1][mask].astype(np.float32).mean())
        hue = hsv[:, :, 0][mask]          # OpenCV hue is 0-179
        red_hue_frac = float(((hue < 25) | (hue > 160)).mean())
    else:
        blur = float(np.gradient(gray.astype(np.float64))[0][mask].var())
        mx, mn = px.max(axis=1), px.min(axis=1)
        saturation = float((255.0 * (mx - mn) / np.maximum(mx, 1.0)).mean())
        red_hue_frac = float((px[:, 0] == mx).mean())

    # Fundus photography exposes the retina through a circular aperture: whatever the
    # camera or cohort, the frame CORNERS are near-black (measured p99 across Ostrava /
    # ROP-VL / Shenzhen / Multi-View: 0.4-19.8). A real-world photograph fills its corners.
    # This is the structural check the colour heuristics cannot provide - a warm-lit indoor
    # scene passes every colour test (measured: a furniture photo scored redness 1.2-1.4,
    # saturation >40, red_hue >0.7) but its corners average >130.
    ch, cw = max(1, int(gray.shape[0] * 0.12)), max(1, int(gray.shape[1] * 0.12))
    corner_lum = float(np.mean([gray[:ch, :cw].mean(), gray[:ch, -cw:].mean(),
                                gray[-ch:, :cw].mean(), gray[-ch:, -cw:].mean()]))

    # Graphics (charts, screenshots, renders) contain large PERFECTLY flat regions;
    # a photograph of anything - retina included - never does, because sensor noise and
    # texture keep the Laplacian nonzero almost everywhere. Measured: fundus flat_frac
    # max 0.39 across all four cohorts, charts p50 0.84.
    if _HAS_CV2:
        lap_abs = np.abs(cv2.Laplacian(gray.astype(np.float64), cv2.CV_64F))
    else:
        gy, gx = np.gradient(gray.astype(np.float64))
        lap_abs = np.abs(gx) + np.abs(gy)
    flat_frac = float((lap_abs < 0.5).mean())

    return {
        "fov_ratio": fov_ratio,
        "redness": float(r / max(b, 1.0)),
        "rgb_ordered": bool(r > g > b),
        "saturation": saturation,
        "red_hue_frac": red_hue_frac,
        "corner_lum": corner_lum,
        "flat_frac": flat_frac,
        "blur_var": blur,
        "luminance": float(gray[mask].mean()),
        "empty": False,
    }


def gradability(image, thresholds=None):
    """Decide whether an uploaded image may be shown to the disease models.

    Returns {gradable, verdict, reason, metrics}. Verdicts:
      gradable | not_fundus | too_blurry | poor_exposure
    """
    t = thresholds or load_gate_thresholds()
    m = gradability_metrics(image)

    if m["empty"] or m["fov_ratio"] < t["min_fov_ratio"]:
        return {"gradable": False, "verdict": "not_fundus", "metrics": m,
                "reason": "No retinal field of view detected — the frame is almost entirely dark."}

    # Graphics signal: large perfectly-flat regions -> a chart or screenshot, not a photo.
    if m["flat_frac"] > t.get("max_flat_frac", 0.6):
        return {"gradable": False, "verdict": "not_fundus", "metrics": m,
                "reason": "This looks like a chart, screenshot or rendered graphic, "
                          "not a photograph."}

    # Structural signal: no circular aperture -> not a fundus photograph, whatever the hue.
    if m["corner_lum"] > t.get("max_corner_lum", 40.0):
        return {"gradable": False, "verdict": "not_fundus", "metrics": m,
                "reason": "This does not look like a fundus photograph — retinal images "
                          "are captured through a circular aperture, leaving the frame "
                          "corners black; this image fills its corners."}

    # Colour signals — reject if ANY says non-retinal (they catch different failures).
    if (m["redness"] < t["min_redness"]
            or m["saturation"] < t["min_saturation"]
            or m["red_hue_frac"] < t["min_red_hue_frac"]
            or (t.get("require_rgb_order") and not m["rgb_ordered"])):
        return {"gradable": False, "verdict": "not_fundus", "metrics": m,
                "reason": "This does not look like a colour fundus photograph "
                          "(retinal images are red-dominant and strongly saturated)."}

    if m["luminance"] < t["min_luminance"]:
        return {"gradable": False, "verdict": "poor_exposure", "metrics": m,
                "reason": "Image is too dark to grade — increase illumination and retake."}
    if m["luminance"] > t["max_luminance"]:
        return {"gradable": False, "verdict": "poor_exposure", "metrics": m,
                "reason": "Image is over-exposed — reduce flash and retake."}

    if m["blur_var"] < t["min_blur_var"]:
        return {"gradable": False, "verdict": "too_blurry", "metrics": m,
                "reason": "Image is too blurred or out of focus to grade — retake."}

    return {"gradable": True, "verdict": "gradable", "metrics": m, "reason": ""}


def run_dqa(manifest, output_pdf="thesis_assets/dataset_report.pdf",
            num_classes=2, class_names=None, deep_sample=2000,
            blur_threshold=20.0, seed=42, cleaned_manifest=None) -> pd.DataFrame:
    df = manifest if isinstance(manifest, pd.DataFrame) else pd.read_csv(manifest)
    df = df.copy().reset_index(drop=True)
    class_names = class_names or [str(i) for i in range(num_classes)]
    rng = random.Random(seed)

    total = len(df)
    readable, resolutions = [], []
    for p in df["image_path"]:
        ok = _is_readable(p)
        readable.append(ok)
        resolutions.append(_resolution(p) if ok else None)
    df["readable"] = readable
    corrupted = int((~df["readable"]).sum())

    # label validity
    valid_labels = df["label"].between(0, num_classes - 1)
    invalid_labels = int((~valid_labels).sum())

    # deep checks on a sample of readable images
    sample_idx = [i for i in df.index if df.at[i, "readable"]]
    rng.shuffle(sample_idx)
    sample_idx = sample_idx[:deep_sample]

    blur_scores, hashes = {}, {}
    for i in sample_idx:
        path = df.at[i, "image_path"]
        blur_scores[i] = _blur_score(path)
        if _HAS_IMAGEHASH:
            try:
                with Image.open(path) as im:
                    hashes[i] = str(imagehash.phash(im.convert("RGB")))
            except Exception:
                pass

    blurry = sum(1 for v in blur_scores.values()
                 if not np.isnan(v) and v < blur_threshold)
    dup_groups = [g for g in Counter(hashes.values()).values() if g > 1]
    duplicates = sum(g - 1 for g in dup_groups)

    df["flag_blurry"] = df.index.map(
        lambda i: blur_scores.get(i, float("nan")) < blur_threshold
        if i in blur_scores and not np.isnan(blur_scores.get(i, float("nan"))) else False)

    cleaned = df[df["readable"] & valid_labels].reset_index(drop=True)

    _write_report(output_pdf, df, cleaned, resolutions, num_classes, class_names,
                  total, corrupted, invalid_labels, blurry, duplicates,
                  len(sample_idx), blur_scores)

    if cleaned_manifest:
        Path(cleaned_manifest).parent.mkdir(parents=True, exist_ok=True)
        # Carry every column the input manifest had. A fixed list here silently DROPPED the
        # patient-group key, so a cleaned manifest could no longer be checked for sibling
        # leakage — the split would look fine because the evidence had been thrown away.
        keep = [c for c in ("image_path", "label", "split", "source", "group")
                if c in cleaned.columns]
        cleaned[keep].to_csv(cleaned_manifest, index=False)
        print(f"[dqa] cleaned manifest -> {cleaned_manifest} ({len(cleaned)} rows, "
              f"columns: {', '.join(keep)})")

    print(f"[dqa] total={total} corrupted={corrupted} invalid_labels={invalid_labels} "
          f"blurry~{blurry}/{len(sample_idx)} duplicates~{duplicates}")
    return cleaned


def _write_report(output_pdf, df, cleaned, resolutions, num_classes, class_names,
                  total, corrupted, invalid_labels, blurry, duplicates,
                  sampled, blur_scores):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    Path(output_pdf).parent.mkdir(parents=True, exist_ok=True)
    res = [r for r in resolutions if r]
    widths = [w for (w, h) in res]
    heights = [h for (w, h) in res]
    class_dist = cleaned["label"].value_counts().sort_index()

    with PdfPages(output_pdf) as pdf:
        # page 1 — summary
        fig = plt.figure(figsize=(8.27, 11.69))
        fig.suptitle("Dataset Quality Assurance Report", fontsize=16, fontweight="bold")
        lines = [
            f"Total images referenced : {total}",
            f"Readable                : {total - corrupted}",
            f"Corrupted / unreadable  : {corrupted}",
            f"Invalid labels          : {invalid_labels}",
            f"Deep-checked sample     : {sampled}",
            f"Blurry (in sample)      : {blurry}",
            f"Duplicate pairs (sample): {duplicates}",
            f"Clean images retained   : {len(cleaned)}",
            "",
            "Per-class (clean):",
        ]
        for i in range(num_classes):
            lines.append(f"  {class_names[i]:<20} {int(class_dist.get(i, 0))}")
        fig.text(0.1, 0.86, "\n".join(lines), fontsize=11, family="monospace", va="top")
        pdf.savefig(fig); plt.close(fig)

        # page 2 — distributions
        fig, ax = plt.subplots(2, 2, figsize=(8.27, 11.69))
        fig.suptitle("Distributions", fontsize=14, fontweight="bold")
        ax[0, 0].bar([class_names[i] for i in class_dist.index], class_dist.values,
                     color="#60A5FA")
        ax[0, 0].set_title("Class distribution"); ax[0, 0].tick_params(axis="x", rotation=45)
        if widths:
            ax[0, 1].hist(widths, bins=30, color="#34D399"); ax[0, 1].set_title("Width (px)")
            ax[1, 0].hist(heights, bins=30, color="#F472B6"); ax[1, 0].set_title("Height (px)")
        valid_blur = [v for v in blur_scores.values() if not np.isnan(v)]
        if valid_blur:
            ax[1, 1].hist(valid_blur, bins=30, color="#FBBF24")
            ax[1, 1].set_title("Blur score (Laplacian var)")
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        pdf.savefig(fig); plt.close(fig)
    print(f"[dqa] report -> {output_pdf}")
