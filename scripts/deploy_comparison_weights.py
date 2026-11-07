"""Deploy the demo's COMPARISON model: RETFound-DINOv2 ViT-L/14 + LoRA, three seeds.

The comparison model is a teammate's `rop_foundation` package (Nov 2026): a frozen RETFound-DINOv2
ViT-L/14 backbone with LoRA adapters (rank 16, alpha 32, on attention q/k/v, projection and both
MLP layers) and a linear 4-class head (Normal, Stage 1, Stage 2, Stage 3), trained three times with
seeds 42, 1 and 2. The webapp shows its answer BESIDE arm B's as a second opinion; it never
decides anything.

Each of the three `best.pt` files is 620 MB because it stores the whole backbone. The backbone is
frozen, so the three copies are bit-identical; this script verifies that and writes it ONCE
(fp16, as saved) plus one small adapter file per seed, so the app loads one backbone and swaps
adapters. It also writes:

  results/comparison/MANIFEST.json      sha256 of every source and written file, the source
                                        meta.json scores, and the data-overlap audit (how many
                                        of the comparison model's training photos are in OUR
                                        training pool / development half / sealed half)
  results/comparison/seen_photos.json   md5 -> tags for every photo in results/prepared/
                                        rop_staging/manifest.csv, so the app can warn when an
                                        uploaded photo is one that a model trained or tested on

Nothing here executes code from the source package: the checkpoints are read with
`torch.load(weights_only=True)` and the data lists with numpy.

    .venv/bin/python scripts/deploy_comparison_weights.py \
        --source "~/Downloads/rop_foundation - Copy (2)/outputs"
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

ROOT = Path(__file__).resolve().parent.parent
REGISTRY = ROOT / "webapp" / "registry.yaml"
RUN = "retfound_dinov2_lora_final_mlp_img280_s{seed}"
EXPECTED_CFG = {"name": "retfound_dinov2", "img": 280, "n_cls": 4, "strategy": "lora", "pool": None,
                "lora_r": 16, "lora_alpha": 32, "lora_mlp": True}
EXPECTED_CLASSES = ["NORMAL", "STAGE1", "STAGE2", "STAGE3"]
OUR_MANIFEST = ROOT / "results/prepared/rop_staging/manifest.csv"
OUR_SPLITS = ROOT / "results/prepared/rop_staging/splits.csv"
SPLIT_TAG = {"train_cv": "our training/validation pool",
             "multiview_dev": "our development test half",
             "multiview_locked": "our sealed test half"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def is_adapter(key: str) -> bool:
    return "lora_" in key or key.startswith(("head.", "ord_head."))


def split_checkpoint(path: Path):
    """(backbone fp16 tensors, adapter fp32 tensors, metadata) from one teammate best.pt."""
    ck = torch.load(path, map_location="cpu", weights_only=True)
    cfg = dict(ck["cfg"])
    bad = {k: cfg.get(k) for k, v in EXPECTED_CFG.items() if cfg.get(k) != v}
    if bad or list(ck["classes"]) != EXPECTED_CLASSES or int(ck["img"]) != 280:
        sys.exit(f"{path}: not the expected configuration (differs: {bad}, classes "
                 f"{ck['classes']}, img {ck['img']})")
    sd = ck["state_dict"]
    backbone = {k: v for k, v in sd.items() if not is_adapter(k)}
    adapters = {k: v.float() for k, v in sd.items() if is_adapter(k)}
    meta = {"cfg": cfg, "classes": list(ck["classes"]), "img": int(ck["img"]),
            "store": int(ck["store"]), "preproc": str(ck["preproc"])}
    return backbone, adapters, meta


def overlap_audit(source: Path) -> dict | None:
    """How the comparison model's photos relate to ours (by file name, then by our split)."""
    npz = source / "data_clahe.npz"
    if not (npz.exists() and OUR_SPLITS.exists() and OUR_MANIFEST.exists()):
        return None
    z = np.load(npz, allow_pickle=False)
    theirs = pd.DataFrame({"name": [p.replace("\\", "/").rsplit("/", 1)[-1] for p in z["paths"]],
                           "their_split": z["split"]})
    ours = pd.read_csv(OUR_SPLITS)
    ours["name"] = ours.image_path.str.rsplit("/", n=1).str[1]
    j = theirs.merge(ours[["name", "split", "source"]], on="name", how="left")
    tab = pd.crosstab(j.their_split, j.split.fillna("not in our splits"))
    return {"n_their_photos": int(len(theirs)),
            "matched_to_our_data_by_name": int(j.split.notna().sum()),
            "their_split_vs_our_split": {r: {c: int(v) for c, v in row.items()}
                                         for r, row in tab.to_dict("index").items()},
            "sources_matched": {k: int(v) for k, v in j.source.value_counts().items()}}


def seen_photos(source: Path) -> dict:
    """md5 -> [tags] for every photo in our manifest; the comparison model's lists join by name."""
    m = pd.read_csv(OUR_MANIFEST)
    m["name"] = m.image_path.str.rsplit("/", n=1).str[1]
    s = pd.read_csv(OUR_SPLITS) if OUR_SPLITS.exists() else None
    tag_by_name = {}
    if s is not None:
        for p, sp in zip(s.image_path, s.split):
            tag_by_name.setdefault(p.rsplit("/", 1)[1], []).append(SPLIT_TAG.get(sp, sp))
    npz = source / "data_clahe.npz"
    if npz.exists():
        z = np.load(npz, allow_pickle=False)
        for p, sp in zip(z["paths"], z["split"]):
            name = p.replace("\\", "/").rsplit("/", 1)[-1]
            tag_by_name.setdefault(name, []).append(f"comparison model: {sp}")
    out = {}
    for p, name in zip(m.image_path, m.name):
        path = Path(p) if Path(p).is_absolute() else ROOT / p
        if not path.is_file():
            continue
        tags = tag_by_name.get(name) or ["our data, outside the staging splits (treated / other)"]
        out.setdefault(md5(path), sorted(set(tags)))
    return out


def deploy(source: Path, root: Path = ROOT) -> dict:
    spec = yaml.safe_load((root / REGISTRY.relative_to(ROOT)).read_text())["comparison"]
    dest = root / Path(spec["backbone"]).parent
    seeds = [int(a["seed"]) for a in spec["adapters"]]
    srcs = {s: source / RUN.format(seed=s) / "best.pt" for s in seeds}
    missing = [str(p) for p in srcs.values() if not p.is_file()]
    if missing:
        sys.exit(f"refusing to deploy: source checkpoint(s) missing: {missing}")
    dest.mkdir(parents=True, exist_ok=True)

    backbone_ref, meta_ref, files = None, None, []
    for a in spec["adapters"]:
        seed, src = int(a["seed"]), srcs[int(a["seed"])]
        backbone, adapters, meta = split_checkpoint(src)
        if backbone_ref is None:
            backbone_ref, meta_ref = backbone, meta
        else:
            same = backbone.keys() == backbone_ref.keys() and all(
                torch.equal(backbone[k], backbone_ref[k]) for k in backbone)
            if not same:
                sys.exit(f"seed {seed}: frozen backbone differs from seed {seeds[0]}'s; "
                         f"the one-backbone deployment is not valid")
        source_meta = json.loads((src.parent / "meta.json").read_text())
        out = root / a["file"]
        torch.save({"state_dict": adapters, "seed": seed, "source_file": str(src),
                    "source_sha256": sha256(src), "source_meta": source_meta}, out)
        files.append({"file": a["file"], "seed": seed, "sha256": sha256(out),
                      "bytes": out.stat().st_size, "source_file": str(src),
                      "source_sha256": sha256(src), "n_tensors": len(adapters),
                      "n_params": int(sum(v.numel() for v in adapters.values())),
                      "source_test": source_meta.get("test"), "source_val": source_meta.get("val")})
    bb_out = root / spec["backbone"]
    torch.save({"state_dict": backbone_ref, **meta_ref,
                "note": "frozen RETFound-DINOv2 ViT-L/14 backbone, identical in all three seeds, "
                        "saved fp16 as in the source checkpoints"}, bb_out)
    files.insert(0, {"file": spec["backbone"], "sha256": sha256(bb_out), "bytes": bb_out.stat().st_size,
                     "n_tensors": len(backbone_ref),
                     "n_params": int(sum(v.numel() for v in backbone_ref.values()))})

    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=root,
                            capture_output=True, text=True).stdout.strip()
    manifest = {"model": spec["name"], "role": "comparison only — never decides",
                "architecture": meta_ref, "commit": commit, "source_dir": str(source),
                "files": files, "overlap_audit": overlap_audit(source)}
    (dest / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
    seen = seen_photos(source)
    (dest / "seen_photos.json").write_text(json.dumps(seen, separators=(",", ":")) + "\n")
    return {"manifest": str((dest / "MANIFEST.json").relative_to(root)),
            "files": [(f["file"], f["bytes"]) for f in files], "seen_photos": len(seen)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default=str(Path.home() / "Downloads" / "rop_foundation - Copy (2)"
                                            / "outputs"))
    a = ap.parse_args()
    print(json.dumps(deploy(Path(a.source).expanduser()), indent=2))
