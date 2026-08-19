"""Fetch the trained model weights this app serves.

    python scripts/get_weights.py

The two checkpoints are too large for a GitHub repository, so they live in a HuggingFace
model repository and this script pulls them from there (~180 MB total, one-time):

  results/rop_staging/weights.pth  THE PRODUCT MODEL (EfficientNetV2-S, structured head).
                                   Produces BOTH the binary screening verdict and the ICROP
                                   stage since the 19 Aug 2026 re-basing.
  results/rop/weights.pth          ROP binary screening (ResNet50) — RETIRED from deciding
                                   on 19 Aug 2026 after an external audit; kept as the
                                   routing anchor and as a fallback.

Run it once after cloning; the web app will then load both models. Without the weights the
app still boots, but every result shows "model not loaded".

(History note: these used to be fetched from our HuggingFace Space. The Space was deleted
on 14 Aug 2026 to purge patient images that survived in its git history, and the weights
moved to a dedicated model repo: https://huggingface.co/Champ610/retinai-rop-weights)
"""
import sys
import urllib.request
from pathlib import Path

BASE = "https://huggingface.co/Champ610/retinai-rop-weights/resolve/main"
WEIGHTS = [
    ("results/rop_staging/weights.pth", "rop_staging_effnetv2s_structured_fold0.pth",
     "ROP screening + ICROP staging (EfficientNetV2-S, structured) — the product model"),
    ("results/rop/weights.pth", "rop_screening_resnet50.pth",
     "ROP screening (ResNet50) — retired, fallback only"),
]

ROOT = Path(__file__).resolve().parent.parent


def fetch(rel, remote_name, label):
    dst = ROOT / rel
    if dst.exists() and dst.stat().st_size > 1_000_000:
        print(f"already present: {rel}")
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    url = f"{BASE}/{remote_name}"
    print(f"downloading {label}\n  {url}")

    def hook(blocks, bs, total):
        done = blocks * bs
        if total > 0:
            sys.stdout.write(f"\r  {done / 1e6:6.1f} / {total / 1e6:.1f} MB")
            sys.stdout.flush()

    urllib.request.urlretrieve(url, dst, reporthook=hook)
    print(f"\n  -> {rel} ({dst.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    for rel, remote_name, label in WEIGHTS:
        fetch(rel, remote_name, label)
    print("done — run:  python webapp/app.py")
