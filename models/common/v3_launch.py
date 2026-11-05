"""Helpers the v3 Kaggle kernel runs before and after training (kept here so they are testable).

The kernel file itself is a script that is rewritten by the pusher and cannot be imported.
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd

from models.common.datasets import cached_path

FLAGS = ("circle_crop", "letterbox", "clahe", "ben_graham")


def inner_probe_manifest(fold0: pd.DataFrame, seed: int = 0, val_frac: float = 0.2) -> pd.DataFrame:
    """Fold-0 TRAIN rows only, split by patient into train / val (val = the inner 20 %).

    Fold-0's own val rows are dropped, never used, so a learning-rate choice made on this
    split cannot leak the fold-0 evaluation set.
    """
    from sklearn.model_selection import GroupShuffleSplit
    df = fold0[fold0["split"] == "train"].reset_index(drop=True).copy()
    gss = GroupShuffleSplit(n_splits=1, test_size=val_frac, random_state=seed)
    _, va = next(gss.split(df, groups=df["patient_id"]))
    df["split"] = "train"
    df.loc[va, "split"] = "val"
    return df


def drop_duplicate_train_rows(df: pd.DataFrame, exclude: pd.DataFrame) -> pd.DataFrame:
    """Protocol v2 "Deduplication": the in-pool exact duplicates are dropped from TRAINING.

    Val rows stay, so val_predictions.npz still lines up row-for-row with the fold's val rows;
    the scorer drops the duplicates from every CV metric.
    """
    drop = (df["split"] == "train") & df["image_path"].isin(set(exclude["image_path"]))
    return df[~drop].reset_index(drop=True)


def check_cache_flags(info: dict, preprocess: dict) -> None:
    """The cache was built with some preprocessing flags; the run config must agree."""
    bad = [f"{k}: cache={info.get(k)!r} config={preprocess.get(k)!r}" for k in FLAGS
           if info.get(k) != preprocess.get(k)]
    if bad:
        raise SystemExit("cache/config preprocessing mismatch -> " + "; ".join(bad))


# "" = arms A/B; "bag" = arm D (Amendment 3c); "r518" = arm G and "uda" = arm F (Amendment 4).
VARIANTS = ("", "bag", "r518", "uda", "v4data", "v4recipe")
FIVE_FOLD_ONLY = ("r518", "uda")        # Amendment 4: the five CV folds, no site fold
LOSO_VARIANTS = ("v4data", "v4recipe")  # Amendment 5b: arms H and I (and B) may run LOSO folds


def _describe(cfg: dict) -> str:
    train = cfg.get("train") or {}
    return (f"train.bag_loss.enabled={(train.get('bag_loss') or {}).get('enabled', False)}, "
            f"train.uda.enabled={(train.get('uda') or {}).get('enabled', False)}, "
            f"preprocess.image_size={(cfg.get('preprocess') or {}).get('image_size')}")


def config_variant(cfg: dict) -> str:
    """The variant a config (a plain dict) describes: "bag" iff train.bag_loss.enabled is true,
    "uda" iff train.uda.enabled is true, "r518" iff preprocess.image_size is 518, else "".
    A config that is two variants at once is no registered arm: ValueError.
    A ``train.arm_tag`` (Amendment 5b: "v4data" / "v4recipe") names the variant outright, and a
    tagged config that also looks like bag/uda/r518 is a ValueError."""
    train = cfg.get("train") or {}
    tag = train.get("arm_tag")
    on = {"bag": (train.get("bag_loss") or {}).get("enabled", False) is True,
          "uda": (train.get("uda") or {}).get("enabled", False) is True,
          "r518": (cfg.get("preprocess") or {}).get("image_size") == 518}
    hits = [v for v, x in on.items() if x]
    if tag is not None:
        if tag not in LOSO_VARIANTS:
            raise ValueError(f"unknown train.arm_tag {tag!r}; known: {LOSO_VARIANTS}")
        if hits:
            raise ValueError(f"config is tagged {tag!r} but also looks like {hits}: {_describe(cfg)}")
        return tag
    if len(hits) > 1:
        raise ValueError(f"config is more than one variant {hits}: {_describe(cfg)}")
    return hits[0] if hits else ""


def check_variant(variant: str, cfg) -> None:
    """The kernel's VARIANT and its config must agree (``config_variant``)."""
    if variant not in VARIANTS:
        raise SystemExit(f"unknown variant {variant!r}; known: {VARIANTS}")
    d = cfg.to_dict()
    try:
        got = config_variant(d)
    except ValueError as e:
        raise SystemExit(str(e)) from None
    if got != variant:
        raise SystemExit(f"variant {variant!r} but the config is variant {got!r} ({_describe(d)})")


def check_site_fold(variant: str, also_site_fold: bool) -> None:
    """Amendment 4: arms G and F train the five CV folds only."""
    if variant in FIVE_FOLD_ONLY and also_site_fold:
        raise SystemExit(f"variant {variant!r} trains the five CV folds only (PROTOCOL_V2 "
                         f"Amendment 4): no site fold")


def check_loso(variant: str, loso: bool) -> None:
    """Amendment 5b: leave-one-site-out folds run for B and the two v4 arms only."""
    if loso and variant not in ("",) + LOSO_VARIANTS:
        raise SystemExit(f"variant {variant!r} may not run LOSO folds; allowed: "
                         f"{('',) + LOSO_VARIANTS}")


POOLS = ("v3", "v4")                  # Amendment 5a: the v4 pool's fold files live in <prep>/v4
LOSO_SOURCES = ("ostrava", "ropvl")   # Amendment 5a.5: the two leave-one-source-out folds


def check_pool(pool: str) -> None:
    """The kernel's POOL names the fold directory: "v3" (<prep>) or "v4" (<prep>/v4)."""
    if pool not in POOLS:
        raise SystemExit(f"unknown pool {pool!r}; known: {POOLS}")


def job_list(folds: list[int], also_site_fold: bool, loso: bool) -> list[str]:
    """The jobs of a main (non-probe) launch, in run order: the CV folds, the site fold, then
    the LOSO folds. Empty ``folds`` with ``loso`` only is arm B's LOSO-only launch."""
    return ([f"fold{f}" for f in folds] + (["foldsite"] if also_site_fold else [])
            + ([f"foldloso_{s}" for s in LOSO_SOURCES] if loso else []))


def fold_file(name: str, prep: str, pool: str) -> str:
    """The fold CSV a job reads, in the pool's directory (``prep`` for "v3", ``prep/v4`` for
    "v4"): ``fold<k>`` -> fold_<k>.csv, ``foldsite`` -> fold_site.csv and ``foldloso_<source>`` ->
    fold_loso_<source>.csv."""
    if pool not in POOLS:
        raise ValueError(f"unknown pool {pool!r}; known: {POOLS}")
    pool_dir = prep if pool == "v3" else f"{prep}/v4"
    if re.fullmatch(r"fold[0-9]+", name):
        return f"{pool_dir}/fold_{name[len('fold'):]}.csv"
    if name == "foldsite":
        return f"{pool_dir}/fold_site.csv"
    src = name.removeprefix("foldloso_")
    if src != name and src in LOSO_SOURCES:
        return f"{pool_dir}/fold_loso_{src}.csv"
    raise ValueError(f"unknown job {name!r}; known: fold<k>, foldsite, "
                     f"{', '.join('foldloso_' + s for s in LOSO_SOURCES)}")


def check_cached_files(df: pd.DataFrame, cache_root: str) -> None:
    missing = [p for p in df["image_path"] if not cached_path(p, cache_root).is_file()]
    if missing:
        raise SystemExit(f"{len(missing)} images have no cached file under {cache_root}; "
                         f"e.g. {missing[:3]}")


def score_manifest(run_dir, manifest: pd.DataFrame, cache_root: str, out_name: str,
                   device=None, batch_size: int = 32, num_workers: int = 4) -> Path:
    """Score every row of ``manifest`` with the run's saved last-epoch model, from the cache.

    Uses the run's own config.yaml and ``build_transforms(cfg, "val")``. Saves
    ``prob``, ``y_true``, ``image_path`` (manifest order) to ``<run_dir>/<out_name>``.
    """
    import torch
    from torch.utils.data import DataLoader
    from models.common.architectures import build_from_cfg
    from models.common.config import load_config
    from models.common.datasets import FundusDataset
    from models.common.preprocessing import build_transforms

    run_dir = Path(run_dir)
    cfg = load_config(run_dir / "config.yaml")
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_from_cfg(cfg).to(device)
    model.load_state_dict(torch.load(run_dir / "weights.pth", map_location=device))
    model.eval()
    ds = FundusDataset(manifest, None, build_transforms(cfg, "val"),
                       int(cfg.preprocess.image_size), cache_root=cache_root)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    probs = []
    with torch.no_grad():
        for imgs, _ in dl:
            probs.append(torch.softmax(model(imgs.to(device)), 1).cpu().numpy())
    out = run_dir / out_name
    np.savez_compressed(out, prob=np.concatenate(probs), y_true=np.asarray(ds.targets),
                        image_path=np.asarray(manifest["image_path"].tolist()))
    return out
