"""Copy the served ROP staging checkpoints into results/rop_staging_v3/ and write their MANIFEST.

The served model is arm B of docs/paper/PROTOCOL_V2.md (DINOv2 ViT-S/14) as the probability-
averaged ensemble of the five CV-fold checkpoints of seed 42. The destination paths and the
scores come from configs/rop_staging_deploy.yaml (`deploy:`), so the app, this script and the
manifest cannot disagree about which files are served.

Refuses to copy anything if a single source checkpoint is missing. Idempotent: a destination
whose sha256 already equals its source is left alone, and the manifest is rewritten with the
same content (it changes only when HEAD or a file changes). The .pth files are gitignored
(results/**/*.pth); MANIFEST.json is matched by `results/**` too, so committing it needs
`git add -f`.

    .venv/bin/python scripts/deploy_rop_v3_weights.py
"""
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
CONFIG = "configs/rop_staging_deploy.yaml"
RUN = "kaggle_cache/v3/rop-v3-dinov2-s-s42-lr1e-4/code/results"


def source_dir(fold: int, seed: int = 42) -> str:
    return f"{RUN}/experiment_{fold + 1:03d}_v3_dinov2_s_fold{fold}_s{seed}"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def deploy(root: Path = ROOT) -> dict:
    dep = yaml.safe_load((root / CONFIG).read_text())["deploy"]
    seed, dests = int(dep["seed"]), [Path(p) for p in dep["weights"]]
    srcs = [Path(source_dir(k, seed)) / "weights.pth" for k in range(len(dests))]
    missing = [str(s) for s in srcs if not (root / s).is_file()]
    if missing:
        sys.exit(f"refusing to deploy: source checkpoint(s) missing: {missing}")

    folds = []
    for k, (src, dst) in enumerate(zip(srcs, dests)):
        digest = sha256(root / src)
        out = root / dst
        if out.is_file() and sha256(out) == digest:
            action = "unchanged"
        else:
            out.parent.mkdir(parents=True, exist_ok=True)
            tmp = out.with_name(out.name + ".part")
            shutil.copyfile(root / src, tmp)
            if sha256(tmp) != digest:
                tmp.unlink()
                sys.exit(f"copy of {src} does not match its source sha256")
            tmp.replace(out)
            action = "copied"
        folds.append({"fold": k, "file": str(dst), "source_experiment": str(src.parent),
                      "sha256": digest, "bytes": out.stat().st_size, "action": action})

    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=root,
                            capture_output=True, text=True).stdout.strip()
    manifest = {
        "arm": dep["arm"], "protocol": dep["protocol"], "model": dep["model_name"],
        "arch": "dinov2_s (timm vit_small_patch14_reg4_dinov2.lvd142m, 392 px, 6-class head)",
        "seed": seed, "ensemble": dep["ensemble"],
        "config": CONFIG, "training_config": dep["training_config"], "commit": commit,
        "folds": [{k: v for k, v in f.items() if k != "action"} for f in folds],
        "deployed_configuration_scores": dep["scores"],
        "operating_point": dep["operating_point"],
    }
    mpath = root / Path(dests[0]).parent / "MANIFEST.json"
    mpath.write_text(json.dumps(manifest, indent=2) + "\n")
    return {"manifest": str(mpath.relative_to(root)),
            "folds": [(f["fold"], f["action"]) for f in folds]}


if __name__ == "__main__":
    print(json.dumps(deploy(), indent=2))
