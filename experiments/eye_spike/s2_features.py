"""Step 2 — frozen embeddings, one image-decoding pass feeding every encoder.

    .venv/bin/python experiments/eye_spike/s2_features.py

Preprocessing = repo circle_crop + letterbox_square (models/common/preprocessing.py), then
  rfg        RETFound-Green  vit_small_patch14_reg4_dinov2 @392, NO CLAHE, mean/std 0.5
  rfg_clahe  RETFound-Green  same, WITH repo CLAHE (ablation)
  dino       DINOv2 ViT-S/14 reg4 (timm .lvd142m) @392, NO CLAHE, ImageNet mean/std
  effnet     torchvision EfficientNetV2-S ImageNet (the repo backbone's init) @384, WITH
             CLAHE (the repo staging pipeline), ImageNet mean/std
ViTs store CLS (post-norm) and mean of patch tokens (post-norm, registers excluded);
EffNet stores the global-avg-pooled 1280-d vector. float16 on disk.
Only the CV pool and Multi-View DEV are read. test_locked is never touched.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from common import CACHE, HERE, ROOT, load_dev, load_pool

RFG_W = HERE / "weights" / "retfound_green" / "retfoundgreen_statedict.pth"
EFF_W = Path.home() / ".cache/torch/hub/checkpoints/efficientnet_v2_s-dd5fe13b.pth"
IMNET = (torch.tensor([0.485, 0.456, 0.406]), torch.tensor([0.229, 0.224, 0.225]))


class Imgs(Dataset):
    def __init__(self, paths):
        self.paths = list(paths)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        from models.common.preprocessing import circle_crop, clahe, letterbox_square
        rgb = np.asarray(Image.open(ROOT / self.paths[i]).convert("RGB"))
        sq = letterbox_square(circle_crop(rgb))
        cl = clahe(sq)
        r = lambda a, s: torch.from_numpy(np.asarray(  # noqa: E731
            Image.fromarray(a).resize((s, s), Image.BILINEAR))).permute(2, 0, 1).contiguous()
        return r(sq, 392), r(cl, 392), r(cl, 384)


def build(dev):
    import timm
    import torchvision
    rfg = timm.create_model("vit_small_patch14_reg4_dinov2", img_size=(392, 392),
                            num_classes=0, pretrained=False)
    rfg.load_state_dict(torch.load(RFG_W, map_location="cpu", weights_only=True), strict=True)
    dino = timm.create_model("vit_small_patch14_reg4_dinov2.lvd142m", img_size=(392, 392),
                             num_classes=0, pretrained=True)
    eff = torchvision.models.efficientnet_v2_s(weights=None)
    eff.load_state_dict(torch.load(EFF_W, map_location="cpu", weights_only=True), strict=True)
    eff.classifier = torch.nn.Identity()
    return {k: m.to(dev).eval() for k, m in (("rfg", rfg), ("dino", dino), ("effnet", eff))}


def vit_feats(m, x):
    t = m.forward_features(x)                      # (B, prefix + patches, D), post-norm
    return t[:, 0], t[:, m.num_prefix_tokens:].mean(1)


def main():
    dev = "mps"
    pool, devdf = load_pool(), load_dev()
    paths = list(pool.image_path) + list(devdf.image_path)
    n_pool = len(pool)
    assert not any("test_locked" in p for p in paths)
    models = build(dev)
    norm = {"rfg": (torch.tensor([.5] * 3), torch.tensor([.5] * 3)), "dino": IMNET, "effnet": IMNET}
    norm = {k: (a.view(1, 3, 1, 1).to(dev), b.view(1, 3, 1, 1).to(dev)) for k, (a, b) in norm.items()}
    out = {k: {"cls": [], "mean": []} for k in ("rfg", "rfg_clahe", "dino")}
    out["effnet"] = {"pool": []}
    gpu_t = {k: 0.0 for k in out}
    fp16_check = {}
    dl = DataLoader(Imgs(paths), batch_size=32, num_workers=8, shuffle=False)
    t0 = time.time()
    ac = lambda: torch.autocast("mps", dtype=torch.float16)  # noqa: E731
    with torch.inference_mode():
        for bi, (sq392, cl392, cl384) in enumerate(dl):
            f = lambda u8, k: ((u8.to(dev).float() / 255.0) - norm[k][0]) / norm[k][1]  # noqa: E731
            for name, mk, x in (("rfg", "rfg", f(sq392, "rfg")), ("rfg_clahe", "rfg", f(cl392, "rfg")),
                                ("dino", "dino", f(sq392, "dino")), ("effnet", "effnet", f(cl384, "effnet"))):
                ts = time.time()
                with ac():
                    if mk == "effnet":
                        e = models[mk](x).float()
                        out[name]["pool"].append(e.half().cpu().numpy())
                    else:
                        c, mn = vit_feats(models[mk], x)
                        e = torch.cat([c, mn], 1).float()
                        out[name]["cls"].append(c.half().cpu().numpy())
                        out[name]["mean"].append(mn.half().cpu().numpy())
                torch.mps.synchronize()
                gpu_t[name] += time.time() - ts
                if bi == 0:   # fp16-autocast vs fp32 agreement on the first real batch
                    e32 = (models[mk](x) if mk == "effnet" else torch.cat(vit_feats(models[mk], x), 1))
                    fp16_check[name] = round(float(torch.nn.functional.cosine_similarity(
                        e, e32.float(), dim=1).min()), 5)
                    print(f"  {name}: min cos(fp16, fp32) = {fp16_check[name]}", flush=True)
            if bi % 20 == 0:
                print(f"  batch {bi}/{len(dl)}  {time.time() - t0:.0f}s", flush=True)
    wall = time.time() - t0
    for name, d in out.items():
        arrs = {k: np.concatenate(v) for k, v in d.items()}
        assert all(np.isfinite(a.astype(np.float32)).all() for a in arrs.values())
        np.savez(CACHE / f"feat_{name}.npz",
                 **{f"pool_{k}": a[:n_pool] for k, a in arrs.items()},
                 **{f"dev_{k}": a[n_pool:] for k, a in arrs.items()})
    timing = {"n_images": len(paths), "wall_s_total_incl_decode": round(wall, 1),
              "gpu_forward_s": {k: round(v, 1) for k, v in gpu_t.items()},
              "imgs_per_s_forward": {k: round(len(paths) / v, 1) for k, v in gpu_t.items()},
              "device": dev, "precision": "fp16 autocast", "min_cos_fp16_vs_fp32_batch0": fp16_check,
              "torch": torch.__version__}
    (HERE / "timing_features.json").write_text(json.dumps(timing, indent=2) + "\n")
    print(json.dumps(timing, indent=1))


if __name__ == "__main__":
    main()
