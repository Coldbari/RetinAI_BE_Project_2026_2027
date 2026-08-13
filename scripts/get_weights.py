"""Fetch the trained model weights this app serves.

    python scripts/get_weights.py

The two checkpoints are too large for a GitHub repository, so they live with the deployed
HuggingFace Space and this script pulls them from there (~180 MB total, one-time):

  results/rop/weights.pth          ROP binary screening model (ResNet50)
  results/rop_staging/weights.pth  ICROP staging research preview (EfficientNetV2-S)

Run it once after cloning; the web app will then load both models. Without the weights the
app still boots, but every result shows "model not loaded".
"""
import sys
import urllib.request
from pathlib import Path

BASE = "https://huggingface.co/spaces/Champ610/retinal-ai/resolve/main"
WEIGHTS = [
    ("results/rop/weights.pth", "ROP screening (ResNet50)"),
    ("results/rop_staging/weights.pth", "ICROP staging research preview (EfficientNetV2-S)"),
]

ROOT = Path(__file__).resolve().parent.parent


def fetch(rel, label):
    dst = ROOT / rel
    if dst.exists() and dst.stat().st_size > 1_000_000:
        print(f"already present: {rel}")
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    url = f"{BASE}/{rel}"
    print(f"downloading {label}\n  {url}")

    def hook(blocks, bs, total):
        done = blocks * bs
        if total > 0:
            sys.stdout.write(f"\r  {done / 1e6:6.1f} / {total / 1e6:.1f} MB")
            sys.stdout.flush()

    urllib.request.urlretrieve(url, dst, reporthook=hook)
    print(f"\n  -> {rel} ({dst.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    for rel, label in WEIGHTS:
        fetch(rel, label)
    print("done — run:  python webapp/app.py")
