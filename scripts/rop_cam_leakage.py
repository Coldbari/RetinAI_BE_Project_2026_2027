"""How much Grad-CAM attention lands OUTSIDE the retina, measured over a whole cohort.

    .venv/bin/python scripts/rop_cam_leakage.py

A user's upload produced a heatmap whose brightest blob sat in the black surround. One image
is an anecdote. This runs the same measurement over all 663 held-out-hospital dev images, for
BOTH the served structured model and the retired ResNet50 head, so "does this model read the
frame instead of the eye?" becomes a distribution rather than a story.

Why both models: the retired head was disqualified for reading site rather than disease
(external AUC 0.691, and it flagged 663 of 663 images). If that diagnosis is right, its
attention should sit off-retina more often than the model that replaced it. That is a
falsifiable prediction, and this script is the test of it.

Writes results/rop/cam_leakage.json (summary) and cam_leakage.npz (per-image, no paths —
the dev set is public-cohort multiview, but the artifact stays path-free by policy).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DEV = ROOT / "results/prepared/rop_staging/test_dev.csv"
OUT = ROOT / "results/rop/cam_leakage.json"


def main() -> None:
    import webapp.inference as I

    dev = pd.read_csv(DEV)
    reg = I.Registry()
    structured = reg.staging
    resnet = next((m for m in reg.models if m.disease == "ROP" and m.loaded), None)
    if not (structured.loaded and resnet):
        raise SystemExit("both models must be loaded to compare them")

    rows: dict[str, list] = {f"{t}_{k}": [] for t in ("s", "r") for k in
                             ("off", "pad", "lift", "retfrac", "conc", "peak_in")}
    y = (dev.label.values > 0).astype(int)

    for i, rel in enumerate(dev.image_path):
        img = Image.open(ROOT / rel).convert("RGB")
        for tag, model in (("s", structured), ("r", resnet)):
            _, ev = model.gradcam(img, 1)
            rows[f"{tag}_off"].append(ev.get("off_retina_pct", np.nan))
            rows[f"{tag}_pad"].append(ev.get("on_padding_pct", 0.0))
            rows[f"{tag}_lift"].append(ev.get("off_retina_lift", np.nan))
            rows[f"{tag}_retfrac"].append(ev.get("retina_frac_pct", np.nan))
            rows[f"{tag}_conc"].append(ev.get("concentration_pct", np.nan))
            rows[f"{tag}_peak_in"].append(bool(ev.get("peak_inside_retina", True)))
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(dev)}", flush=True)

    arr = {k: np.asarray(v) for k, v in rows.items()}
    np.savez_compressed(OUT.with_suffix(".npz"), y=y, **arr)

    def summarise(tag: str, name: str) -> dict:
        off, conc, pin = arr[f"{tag}_off"], arr[f"{tag}_conc"], arr[f"{tag}_peak_in"]
        q = np.nanpercentile(off, [5, 25, 50, 75, 95])
        return {
            "model": name,
            "retina_frac_pct_mean": round(float(np.nanmean(arr[f"{tag}_retfrac"])), 2),
            "on_synthetic_padding_pct_mean": round(float(np.nanmean(arr[f"{tag}_pad"])), 2),
            "off_retina_lift_mean": round(float(np.nanmean(arr[f"{tag}_lift"])), 3),
            "off_retina_pct_camera_surround_only": {"mean": round(float(np.nanmean(off)), 2),
                               **{k: round(float(v), 2) for k, v in
                                  zip(("p5", "p25", "median", "p75", "p95"), q)}},
            "frac_images_over_15pct_off_retina": round(float(np.nanmean(off > 15)), 4),
            "frac_peak_outside_retina": round(float(1 - np.mean(pin)), 4),
            "concentration_pct_median": round(float(np.nanmedian(conc)), 2),
        }

    res = {
        "what": "Grad-CAM attention falling outside the retinal field of view, per image.",
        "eval_set": {"csv": str(DEV.relative_to(ROOT)), "n": len(dev),
                     "site": "held-out hospital, dev half"},
        "method": ("attention measured on the RAW map before any masking, against a fitted "
                   "elliptical aperture (models.common.data_quality.retina_mask). The "
                   "displayed overlay is masked; this measurement is not."),
        "served_structured": summarise("s", "structured EfficientNetV2-S, P(any ROP)"),
        "retired_resnet50": summarise("r", "retired ResNet50 screening head"),
    }
    # the falsifiable bit, stated as a comparison rather than left to the reader
    d = (res["retired_resnet50"]["off_retina_lift_mean"]
         - res["served_structured"]["off_retina_lift_mean"])
    res["comparison"] = {
        "retired_minus_served_off_retina_LIFT": round(float(d), 3),
        "why_lift_not_raw_pct": ("the two frames devote different areas to non-retina "
                                "(structured letterboxes), so raw percentages are not comparable"),
        "prediction_was": ("the disqualified head should read the frame MORE than the model "
                           "that replaced it"),
        "prediction_held": bool(d > 0),
    }
    OUT.write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))
    print(f"\nwrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
