"""Agreement gate: the cache must not change what the committed fold-0 checkpoint predicts.

Checks (all must hold, else exit 1): CACHE_INFO.json flags equal the config's; every manifest
image is cached and no test_locked path is; argmax agreement >= 0.99 between scoring fold-0 val
from originals and from the cache; |dQWK| < 0.005; median PSNR > 40 dB on n_psnr images (tensors
at the model size, before normalisation).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from models.common.datasets import cached_path  # noqa: E402

STAGING = ROOT / "results" / "prepared" / "rop_staging"
FLAG_KEYS = ("circle_crop", "letterbox", "clahe", "ben_graham")


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    return float("inf") if mse == 0 else 10 * np.log10(255.0 ** 2 / mse)


def check_stamp(cache_root, expected: dict) -> dict:
    stamp = json.loads((Path(cache_root) / "CACHE_INFO.json").read_text())
    bad = {k: (stamp.get(k), expected[k]) for k in FLAG_KEYS if stamp.get(k) != expected[k]}
    if bad:
        sys.exit(f"CACHE_INFO.json disagrees with the config (stamp, config): {bad}")
    return stamp


def completeness(manifests, cache_root, locked) -> dict:
    paths = {str(p) for m in manifests for p in pd.read_csv(m)["image_path"]}
    missing = sum(not cached_path(p, str(cache_root)).exists() for p in paths)
    lp = {str(p) for p in pd.read_csv(locked)["image_path"]} if locked else set()
    locked_present = sum(cached_path(p, str(cache_root)).exists() for p in lp - paths)
    # a locked path also in a pool manifest would be a manifest bug; count it too
    locked_present += len(lp & paths) if lp else 0
    return {"unique_paths": len(paths), "missing": missing, "locked_present": locked_present}


def _scores(model, ds, device, bs=32):
    import torch
    from torch.utils.data import DataLoader
    out = []
    with torch.no_grad():
        for x, _ in DataLoader(ds, batch_size=bs, num_workers=4):
            out.append(model(x.to(device)).float().cpu())
    return torch.cat(out).argmax(1).numpy()


def gate(ckpt: Path, manifest: Path, cache_root: Path, config: Path | None = None,
         n_psnr: int = 200) -> dict:
    import torch
    from sklearn.metrics import cohen_kappa_score
    from models.common.architectures import build_from_cfg
    from models.common.config import load_config
    from models.common.datasets import FundusDataset
    from models.common.preprocessing import build_transforms

    cache_root = Path(cache_root)
    config = Path(config or Path(ckpt).parent / "config.yaml")
    cfg_o = load_config(config, ["preprocess.cached=false"])
    cfg_c = load_config(config, ["preprocess.cached=true", f"preprocess.cache_root={cache_root}"])
    model = build_from_cfg(cfg_o, pretrained=False)
    model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model.to(device).eval()

    df = pd.read_csv(manifest)
    df = df[df["split"] == "val"].reset_index(drop=True)
    ds_o = FundusDataset(df, "val", build_transforms(cfg_o, "val"))
    ds_c = FundusDataset(df, "val", build_transforms(cfg_c, "val"), cache_root=str(cache_root))
    p_o, p_c = _scores(model, ds_o, device), _scores(model, ds_c, device)
    y = df["label"].astype(int).to_numpy()
    q = lambda p: cohen_kappa_score(y, p, weights="quadratic")  # noqa: E731

    import torchvision.transforms as T
    size = int(cfg_o.preprocess.image_size)
    to_u8 = lambda t: (t.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)  # noqa: E731
    pre = [T.Resize((size, size)), T.ToTensor()]
    tr_o = T.Compose([_pre(cfg_o), *pre])
    tr_c = T.Compose(pre)
    idx = np.random.default_rng(0).choice(len(df), min(n_psnr, len(df)), replace=False)
    ps = []
    for i in idx:
        src = df["image_path"][i]
        a = to_u8(tr_o(Image.open(src).convert("RGB")))
        b = to_u8(tr_c(Image.open(cached_path(src, str(cache_root))).convert("RGB")))
        ps.append(psnr(a, b))
    return {"n_images": int(len(df)), "n_psnr": int(len(idx)),
            "argmax_agreement": float((p_o == p_c).mean()),
            "qwk_original": float(q(p_o)), "qwk_cached": float(q(p_c)),
            "delta_qwk": float(q(p_c) - q(p_o)), "psnr_median": float(np.median(ps)),
            "device": device, "ckpt": str(ckpt)}


def _pre(cfg):
    from models.common.preprocessing import _fundus_from_cfg
    return _fundus_from_cfg(cfg)


def passes(r: dict) -> bool:
    return r["argmax_agreement"] >= 0.99 and abs(r["delta_qwk"]) < 0.005 and r["psnr_median"] > 40


def main():
    import os
    from models.common.config import load_config
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--cache-root", required=True)
    ap.add_argument("--manifest", default=str(STAGING / "fold_0.csv"))
    ap.add_argument("--out", default=str(ROOT / "results/rop/v3/cache_gate.json"))
    a = ap.parse_args()
    os.chdir(ROOT)
    cache = Path(a.cache_root)
    canon = load_config(ROOT / "configs/rop_staging_rfg.yaml").preprocess
    stamp = check_stamp(cache, {k: bool(canon[k]) for k in FLAG_KEYS})
    manifests = [STAGING / f"fold_{i}.csv" for i in range(5)] + [STAGING / "test_dev.csv"]
    comp = completeness(manifests, cache, STAGING / "test_locked.csv")
    print(comp)
    if comp["missing"] or comp["locked_present"] or comp["unique_paths"] != stamp["image_count"]:
        sys.exit(f"GATE FAIL (completeness): {comp}")
    r = gate(Path(a.ckpt), Path(a.manifest), cache, Path(a.config) if a.config else None)
    r.update(completeness=comp, stamp=stamp, passed=passes(r))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(r, indent=2))
    print({k: r[k] for k in ("argmax_agreement", "delta_qwk", "psnr_median")})
    print("GATE PASS" if r["passed"] else "GATE FAIL")
    sys.exit(0 if r["passed"] else 1)


if __name__ == "__main__":
    main()
