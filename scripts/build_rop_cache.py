"""Pre-compute the ROP fundus preprocessing once, so training does not redo it every epoch.

For every image in the fold CSVs and test_dev.csv: circle-crop -> letterbox -> CLAHE at NATIVE
resolution (``FundusPreprocess``), then resize to ``max_side`` px, saved as an EXIF-free 4:4:4
JPEG at ``cached_path(image_path, out_root)`` so the training Dataset finds it. The tree is
stamped with CACHE_INFO.json, which ``scripts/rop_cache_gate.py`` checks against the config.
test_locked.csv is never read: a manifest with "locked" in its name is refused.
"""
from __future__ import annotations

import argparse
import datetime
import json
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from models.common.datasets import cached_path  # noqa: E402
from models.common.preprocessing import FundusPreprocess  # noqa: E402

STAGING = ROOT / "results" / "prepared" / "rop_staging"
FLAGS = dict(circle_crop=True, letterbox=True, clahe=True, ben_graham=False)


def default_manifests() -> list[Path]:
    return [STAGING / f"fold_{i}.csv" for i in range(5)] + [STAGING / "test_dev.csv"]


def unique_paths(manifests) -> list[str]:
    """Every distinct image_path in the manifests, refusing any 'locked' manifest by name."""
    for m in manifests:
        if "locked" in Path(m).name.lower():
            raise ValueError(f"refusing {Path(m).name}: the locked test set is never cached")
    seen: dict[str, None] = {}
    for m in manifests:
        for p in pd.read_csv(m)["image_path"]:
            seen.setdefault(str(p))
    return list(seen)


def cache_one(src: str, dst: Path, max_side: int = 768, quality: int = 95,
              fmt: str = "PNG") -> None:
    pre = FundusPreprocess(circle=FLAGS["circle_crop"], ben=FLAGS["ben_graham"],
                           do_clahe=FLAGS["clahe"], letterbox=FLAGS["letterbox"])
    img = pre(Image.open(src).convert("RGB"))
    w, h = img.size
    s = max_side / max(w, h)
    img = img.resize((max(1, round(w * s)), max(1, round(h * s))), Image.LANCZOS)
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".part")
    # Default PNG: lossless, so the only loss vs the original pipeline is the 768 resize. JPEG
    # q95 4:4:4 flipped 19/1691 near-tie predictions (agreement 0.9888 < 0.99); PNG flips 14.
    # The file keeps the .jpg NAME that cached_path() expects; PIL sniffs content, not suffix.
    if fmt == "JPEG":
        img.save(tmp, format="JPEG", quality=quality, subsampling=0)   # no exif= -> none written
    else:
        img.save(tmp, format=fmt)       # no exif= -> none written
    tmp.replace(dst)


def _work(a):
    cache_one(*a)
    return 1


def build_cache(manifests, out_root, max_side: int = 768, quality: int = 95,
                workers: int = 8, fmt: str = "PNG") -> dict:
    paths = unique_paths(manifests)
    out_root = Path(out_root)
    todo, skipped = [], 0
    for p in paths:
        dst = cached_path(p, str(out_root))
        if dst.exists():
            skipped += 1
        else:
            todo.append((p, dst, max_side, quality, fmt))
    if workers <= 1:
        done = sum(_work(a) for a in todo)
    else:
        with ProcessPoolExecutor(workers) as ex:
            done = 0
            for done, _ in enumerate(ex.map(_work, todo, chunksize=16), 1):
                if done % 500 == 0:
                    print(f"{done}/{len(todo)}", flush=True)
    out_root.mkdir(parents=True, exist_ok=True)
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                          text=True).stdout.strip()
    (out_root / "CACHE_INFO.json").write_text(json.dumps({
        **FLAGS, "max_side": max_side, "encoding": fmt,
        "encoding_note": "bytes are %s inside files named .jpg (cached_path's suffix)" % fmt,
        **({"jpeg_quality": quality, "subsampling": "4:4:4"} if fmt == "JPEG" else {}),
        "git_head": head, "manifests": [Path(m).name for m in manifests],
        "image_count": len(paths), "built": datetime.date.today().isoformat(),
    }, indent=2))
    return {"written": done, "skipped_existing": skipped}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    import os
    os.chdir(ROOT)                      # manifests hold repo-relative paths
    print(build_cache(default_manifests(), Path(a.out), workers=a.workers))


if __name__ == "__main__":
    main()
