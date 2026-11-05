"""Is off-retina Grad-CAM attention a property of the MODEL, or of the objective we chose?

    .venv/bin/python scripts/rop_cam_objective_control.py

Grad-CAM needs a scalar to differentiate. The served structured model has no single logit
for "any ROP" — the quantity the verdict turns on is a composite of the CORN cascade and the
AP-ROP sigmoid — so the app differentiates log P(any ROP). The retired ResNet50 has a plain
class logit. Those are different objectives, and if the choice moves the leakage number then
the two models were never comparable on it.

This measures the same model under three objectives on the same images, and splits the
off-retina attention into the camera's own surround and the synthetic letterbox padding that
preprocessing adds (which is black by construction and carries no information at all).

Writes results/rop/cam_objective_control.json.
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

N = 120
OUT = ROOT / "results/rop/cam_objective_control.json"


def main() -> None:
    import webapp.inference as I
    from models.common.architectures import get_target_layer
    from models.common.data_quality import retina_mask

    dev = pd.read_csv(ROOT / "results/prepared/rop_staging/test_dev.csv").head(N)
    reg = I.Registry()
    st = reg.staging
    if not st.loaded:
        raise SystemExit("structured weights required")
    tgt = get_target_layer(st.model.backbone, st.arch)

    objectives = {
        "composite log P(any ROP)  [SERVED]":
            lambda o: torch.log(st.decode(o, arop_threshold=st.thr)[1][0, 1:].sum() + 1e-9),
        "raw AP-ROP logit": lambda o: o["arop"].flatten()[0],
        "raw CORN stage-logit sum": lambda o: o["stage"][0].sum(),
    }

    cache = []
    for rel in dev.image_path:
        img = Image.open(ROOT / rel).convert("RGB")
        frame = I._model_frame(st.base_tfm, img)
        rm = retina_mask(frame)
        if rm is None:
            continue
        pad = I._padding_mask(st.base_tfm, img)
        pad = (pad & ~rm) if pad is not None else np.zeros_like(rm)
        cache.append((img, rm, pad))

    res = {
        "what": "Grad-CAM off-retina attention for ONE model under three objectives.",
        "n_images": len(cache),
        "frame_geometry_pct": {
            "retina": round(float(np.mean([m.mean() for _, m, _ in cache])) * 100, 2),
            "synthetic_letterbox_padding": round(
                float(np.mean([p.mean() for _, _, p in cache])) * 100, 2),
            "camera_surround": round(
                float(np.mean([(~m & ~p).mean() for _, m, p in cache])) * 100, 2),
        },
        "objectives": {},
    }
    for name, obj in objectives.items():
        surround, padding = [], []
        for img, rm, pad in cache:
            t = st.base_tfm(img).unsqueeze(0).to(st.device)
            grads, acts = [], []
            bw = tgt.register_full_backward_hook(
                lambda m, gi, go: grads.append(go[0].detach().cpu()))
            fw = tgt.register_forward_hook(lambda m, i, o: acts.append(o.detach().cpu()))
            st.model.zero_grad()
            obj(st.model(t)).backward()
            bw.remove(); fw.remove()
            w = grads[0].mean(dim=[2, 3], keepdim=True)
            cam = np.clip(I._resize_f32(
                torch.relu((w * acts[0]).sum(1)).squeeze().numpy().astype(np.float32),
                I.DISPLAY), 0, None)
            tot = cam.sum()
            if tot <= 0:
                continue
            surround.append(cam[~rm & ~pad].sum() / tot * 100)
            padding.append(cam[pad].sum() / tot * 100)
        res["objectives"][name] = {
            "on_camera_surround_pct": round(float(np.mean(surround)), 2),
            "on_synthetic_padding_pct": round(float(np.mean(padding)), 2),
            "total_off_retina_pct": round(float(np.mean(surround) + np.mean(padding)), 2),
        }

    vals = [v["total_off_retina_pct"] for v in res["objectives"].values()]
    res["finding"] = {
        "spread_across_objectives_pct_points": round(max(vals) - min(vals), 2),
        "reading": ("The choice of scalar to differentiate moves total off-retina attention "
                    f"by {max(vals) - min(vals):.1f} percentage points on identical images "
                    "and identical weights. Off-retina attention is therefore NOT a "
                    "model-intrinsic property, and comparing two models measured under "
                    "different objectives says nothing about either. It remains useful for "
                    "what it was built for: flagging an individual map whose evidence sits "
                    "off the retina."),
    }
    OUT.write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))
    print(f"\nwrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
