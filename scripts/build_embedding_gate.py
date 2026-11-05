"""Build the embedding gate: 'does this image even live near the fundus manifold?'

    .venv/bin/python scripts/build_embedding_gate.py

Third layer of the gradability gate (2026-08-19). The first two are hand-crafted
(colour + structure), and field testing keeps producing image classes that defeat one
rule at a time: a furniture photo beat the colour checks, a night-time interior beat the
corner check. Rather than add a fourth heuristic, this measures distance to the fundus
distribution in the SCREENING MODEL'S OWN feature space - the strongest available notion
of "looks like a retina", and free at runtime because the backbone is already loaded.

Method: embed TRAIN-side fundus from all four cohorts through an IMAGENET-pretrained
ResNet50 (NOT the fine-tuned screening model - measured first: the fine-tuned backbones
collapse scene information and cannot separate a warm room from a retina; generic
ImageNet features separate them by a wide margin, because they still know what rooms,
lamps and furniture look like). The gate statistic is cosine distance to the K-th
nearest fundus embedding (kNN, multimodal-friendly). The threshold sits at fundus-max
plus a quarter of the gap to the nearest negative; the build REFUSES to publish if the
absolute gap is under 0.15.

Output: results/rop/gate_embedding.npz  (centroid, threshold, provenance)
"""
import json
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

SEED = 42
PER_GROUP = 150


def main():
    import torch
    from models.common.config import load_config
    from models.common.preprocessing import build_transforms
    from models.common.architectures import build_model

    sys.path.insert(0, str(ROOT / "scripts"))
    from calibrate_gate import positives, negatives   # same samples as the heuristic gate

    import torchvision
    cfg = load_config(ROOT / "configs" / "rop.yaml")
    device = ("mps" if torch.backends.mps.is_available()
              else "cuda" if torch.cuda.is_available() else "cpu")
    model = torchvision.models.resnet50(weights="IMAGENET1K_V2")
    model.fc = torch.nn.Identity()
    model.to(device).eval()
    tfm = build_transforms(cfg, "val")

    def embed(img):
        x = tfm(img.convert("RGB")).unsqueeze(0).to(device)
        with torch.no_grad():
            z = model(x).flatten(1).float().cpu().numpy()[0]
        return z / (np.linalg.norm(z) + 1e-9)

    pos_d, embs = {}, []
    for name, files in positives().items():
        for f in files:
            try:
                embs.append(embed(Image.open(f)))
            except Exception:
                pass
        pos_d[name] = len(embs)
    E = np.stack(embs).astype(np.float32)          # the memory bank, L2-normalised
    K = 5

    def knn_dist(z, bank):
        sims = bank @ z
        return float(1.0 - np.sort(sims)[-K])       # distance to the K-th nearest fundus

    # leave-one-out on the bank itself (drop self-similarity = the top hit)
    sims_all = E @ E.T
    np.fill_diagonal(sims_all, -1.0)
    d_pos = 1.0 - np.sort(sims_all, axis=1)[:, -K]
    print(f"fundus (LOO kNN, K={K}): n={len(E)}  p50={np.percentile(d_pos,50):.4f} "
          f"p99={np.percentile(d_pos,99):.4f}  max={d_pos.max():.4f}")

    neg_min = np.inf
    for name, imgs in negatives():
        d = np.array([knn_dist(embed(im), E) for im in imgs])
        neg_min = min(neg_min, float(d.min()))
        print(f"neg {name:<16} n={len(d):>3}  dist min={d.min():.4f}  p50={np.percentile(d,50):.4f}")

    gap = neg_min - float(d_pos.max())
    spread = float(d_pos.max() - np.percentile(d_pos, 50))
    print(f"\ngap fundus-max -> nearest negative: {gap:.4f}  (fundus spread {spread:.4f})")
    if gap < 0.15:
        raise SystemExit("REFUSING to publish: the gap to the nearest negative is under "
                         "0.15 - this feature space does not separate.")
    threshold = float(d_pos.max() + 0.25 * gap)
    np.savez_compressed(ROOT / "results/rop/gate_embedding.npz",
                        bank=E.astype(np.float16),   # 600 x 2048 fp16 ~ 2.4 MB
                        k=np.int32(K),
                        threshold=np.float32(threshold),
                        meta=json.dumps({"built": "2026-08-19", "seed": SEED,
                                         "arch": "torchvision resnet50 IMAGENET1K_V2",
                                         "metric": "cosine kNN",
                                         "n_fundus": int(len(E)), "per_group": pos_d,
                                         "fundus_loo_max": float(d_pos.max()),
                                         "nearest_negative": neg_min}))
    print(f"threshold = {threshold:.4f}  -> results/rop/gate_embedding.npz")


if __name__ == "__main__":
    main()
