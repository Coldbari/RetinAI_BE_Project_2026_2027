"""Does the peak-outside-retina gap survive when both models face the SAME objective?

    .venv/bin/python scripts/rop_cam_matched_objective.py

The cohort audit (rop_cam_leakage.py) found 35.8% of the served model's Grad-CAM peaks
outside the retina against 2.9% for the retired ResNet50 — but it measured the served
model under log P(any ROP) and the retired one under a raw class logit, and the objective
control showed the objective alone moves off-retina attention by 26 points. So that gap
is unattributable: model, or objective?

This crosses them. Both models, both objective FORMS, all 663 held-out dev images:

  log-prob form:  structured -> log P(any ROP)      (what the app serves)
                  resnet50   -> log softmax P(ROP)
  logit form:     structured -> first CORN logit    (the logit of P(stage>=1) --
                                the stage branch's own any-ROP logit; the composite
                                probability also carries the AP-ROP branch, which has
                                no single-logit form, so this is the nearest native one)
                  resnet50   -> raw ROP class logit (what the audit measured)

If the peak gap holds in BOTH matched columns it is a property of the models. If it
collapses or flips between columns, it was the measurement.

Writes results/rop/cam_matched_objective.json (+ .npz, per-image, path-free by policy).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DEV = ROOT / "results/prepared/rop_staging/test_dev.csv"
OUT = ROOT / "results/rop/cam_matched_objective.json"


def _cam_for(model, target, tensor, objective):
    grads, acts = [], []
    bw = target.register_full_backward_hook(lambda m, gi, go: grads.append(go[0].detach().cpu()))
    fw = target.register_forward_hook(lambda m, i, o: acts.append(o.detach().cpu()))
    model.zero_grad()
    objective(model(tensor)).backward()
    bw.remove(); fw.remove()
    import webapp.inference as I
    w = grads[0].mean(dim=[2, 3], keepdim=True)
    cam = torch.relu((w * acts[0]).sum(1)).squeeze().numpy().astype(np.float32)
    return np.clip(I._resize_f32(cam, I.DISPLAY), 0.0, None)


def main() -> None:
    import webapp.inference as I
    from models.common.architectures import get_target_layer
    from models.common.data_quality import retina_mask

    dev = pd.read_csv(DEV)
    reg = I.Registry()
    st = reg.staging
    rn = next((m for m in reg.models if m.disease == "ROP" and m.loaded), None)
    if not (st.loaded and rn):
        raise SystemExit("both models must be loaded")

    setups = {
        "structured": {
            "model": st.model, "tfm": st.base_tfm, "device": st.device,
            "target": get_target_layer(st.model.backbone, st.arch),
            "logprob": lambda o: torch.log(st.decode(o, arop_threshold=st.thr)[1][0, 1:].sum() + 1e-9),
            "logit":   lambda o: o["stage"].flatten()[0],
        },
        "resnet50": {
            "model": rn.model, "tfm": rn.base_tfm, "device": rn.device,
            "target": get_target_layer(rn.model, rn.arch),
            "logprob": lambda o: torch.log_softmax(o, dim=1)[0, 1],
            "logit":   lambda o: o[0, 1],
        },
    }

    keys = [f"{m}_{f}_{k}" for m in setups for f in ("logprob", "logit")
            for k in ("peak_in", "lift", "off", "pad")]
    rows: dict[str, list] = {k: [] for k in keys}

    for i, rel in enumerate(dev.image_path):
        img = Image.open(ROOT / rel).convert("RGB")
        for mname, s in setups.items():
            frame = I._model_frame(s["tfm"], img)
            rm = retina_mask(frame)
            if rm is None:
                for f in ("logprob", "logit"):
                    for k in ("peak_in", "lift", "off", "pad"):
                        rows[f"{mname}_{f}_{k}"].append(np.nan)
                continue
            pad = I._padding_mask(s["tfm"], img)
            tensor = s["tfm"](img).unsqueeze(0).to(s["device"])
            for f in ("logprob", "logit"):
                cam = _cam_for(s["model"], s["target"], tensor, s[f])
                ev = I._cam_evidence(cam, rm, pad)
                dead = not ev.get("measurable", False)   # an all-zero map has no peak
                rows[f"{mname}_{f}_peak_in"].append(
                    np.nan if dead else float(bool(ev["peak_inside_retina"])))
                rows[f"{mname}_{f}_lift"].append(np.nan if dead else ev["off_retina_lift"])
                rows[f"{mname}_{f}_off"].append(np.nan if dead else ev["off_retina_pct"])
                rows[f"{mname}_{f}_pad"].append(np.nan if dead else ev.get("on_padding_pct", 0.0))
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(dev)}", flush=True)

    arr = {k: np.asarray(v, dtype=np.float64) for k, v in rows.items()}
    np.savez_compressed(OUT.with_suffix(".npz"), **arr)

    def cell(m, f):
        return {
            "n_dead_maps": int(np.isnan(arr[f"{m}_{f}_peak_in"]).sum()),
            "frac_peak_outside_retina": round(float(1 - np.nanmean(arr[f"{m}_{f}_peak_in"])), 4),
            "off_retina_lift_mean": round(float(np.nanmean(arr[f"{m}_{f}_lift"])), 3),
            "camera_surround_pct_mean": round(float(np.nanmean(arr[f"{m}_{f}_off"])), 2),
            "padding_pct_mean": round(float(np.nanmean(arr[f"{m}_{f}_pad"])), 2),
        }

    res = {
        "what": ("2x2: both ROP models under both Grad-CAM objective forms, so the "
                 "peak-outside-retina comparison is finally like-for-like."),
        "n": len(dev),
        "cells": {m: {f: cell(m, f) for f in ("logprob", "logit")} for m in setups},
    }
    gaps = {f: round(res["cells"]["structured"][f]["frac_peak_outside_retina"]
                     - res["cells"]["resnet50"][f]["frac_peak_outside_retina"], 4)
            for f in ("logprob", "logit")}
    res["matched_peak_gap_structured_minus_resnet"] = gaps
    same_sign = (gaps["logprob"] > 0) == (gaps["logit"] > 0)
    res["reading"] = (
        "Gap persists with the same sign under both matched forms -> a property of the "
        "models, reportable." if same_sign and min(abs(g) for g in gaps.values()) > 0.05
        else "Gap does not survive matching -> it was the measurement; the 35.8%-vs-2.9% "
             "claim must not appear in the paper.")
    OUT.write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))
    print(f"\nwrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
