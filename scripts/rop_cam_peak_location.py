"""WHERE do the structured model's off-retina Grad-CAM peaks actually land?

    .venv/bin/python scripts/rop_cam_peak_location.py

The matched-objective audit (cam_matched_objective.json) established that the served
structured model puts its Grad-CAM peak outside the retina in 21.6-35.8% of images
(objective-form dependent) against 2-3% for the retired ResNet50. But its frame contains
synthetic letterbox bars the ResNet50's frame does not, so "outside" conflates two places:
the camera's own surround (a real frame-reading signal) and padding that preprocessing
added (a frame-geometry artifact). This classifies every off-retina peak into one or the
other. Writes results/rop/cam_peak_location.json.
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

OUT = ROOT / "results/rop/cam_peak_location.json"


def main() -> None:
    import webapp.inference as I
    from models.common.architectures import get_target_layer
    from models.common.data_quality import retina_mask
    from scripts.rop_cam_matched_objective import _cam_for

    dev = pd.read_csv(ROOT / "results/prepared/rop_staging/test_dev.csv")
    st = I.Registry().staging
    if not st.loaded:
        raise SystemExit("structured weights required")
    tgt = get_target_layer(st.model.backbone, st.arch)
    objectives = {
        "logprob": lambda o: torch.log(st.decode(o, arop_threshold=st.thr)[1][0, 1:].sum() + 1e-9),
        "logit":   lambda o: o["stage"].flatten()[0],
    }

    counts = {f: {"retina": 0, "camera_surround": 0, "synthetic_padding": 0, "dead": 0}
              for f in objectives}
    for i, rel in enumerate(dev.image_path):
        img = Image.open(ROOT / rel).convert("RGB")
        frame = I._model_frame(st.base_tfm, img)
        rm = retina_mask(frame)
        if rm is None:
            continue
        pad = I._padding_mask(st.base_tfm, img)
        pad = pad if pad is not None else np.zeros_like(rm)
        tensor = st.base_tfm(img).unsqueeze(0).to(st.device)
        for f, obj in objectives.items():
            cam = _cam_for(st.model, tgt, tensor, obj)
            if cam.max() <= 0:
                counts[f]["dead"] += 1
                continue
            r, c = np.unravel_index(int(cam.argmax()), cam.shape)
            counts[f]["retina" if rm[r, c] else
                      "synthetic_padding" if pad[r, c] else "camera_surround"] += 1
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(dev)}", flush=True)

    res = {"what": ("location of the structured model's raw Grad-CAM argmax, split so "
                    "synthetic letterbox padding cannot masquerade as frame-reading"),
           "n": len(dev), "counts": counts}
    for f, c in counts.items():
        live = sum(v for k, v in c.items() if k != "dead")
        res[f"{f}_pct_of_live_maps"] = {k: round(v / live * 100, 1)
                                        for k, v in c.items() if k != "dead"}
    OUT.write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))
    print(f"\nwrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
