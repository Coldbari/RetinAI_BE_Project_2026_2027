"""Copy one v3 training kernel's outputs into results/rop/v3/runs/<arch>_s<seed>/ (plan Task 9 step 3);
a variant kernel (summary "variant", e.g. arm D's "bag") goes to <arch>_<variant>_s<seed>/.
Arms G ("r518") and F ("uda") train the five CV folds only, over two kernels (a = folds 0-2,
b = 3-4): pass both. Arms H ("v4data") and I ("v4recipe", PROTOCOL_V2 Amendment 5) train on the
v4 pool (summary "pool": "v4", fold files in <prep>/v4/) and need the five CV folds, the site fold
and both leave-one-source-out folds (a = folds 0-4; b = site + LOSO): pass both. Arm B's LOSO-only
kernel (v3 pool, "loso": true, only the two LOSO folds) is added to B's already collected
dinov2_s_s<seed>/ as fold_loso_ostrava/ and fold_loso_ropvl/; nothing there is overwritten.
No other arm has LOSO folds.

    python scripts/collect_v3_run.py kaggle_cache/v3/rop-v3-retfound-green-s42 [more kernels of it]

Per fold: val_predictions.npz, dev_predictions.npz, metrics.json, config.yaml -> fold_<k>/
(the site fold -> fold_site/), plus a merged summary.json. Weights stay in kaggle_cache.
Every file is checked before anything is written: the kernel identity (arch, seed, not a probe,
20 epochs, the lr the probe fixed in lr_probe.json), the five CV folds and the site fold all
present (the five CV folds only, never a site fold, for a variant in FIVE_FOLD_ONLY), the pool
the variant trains on, each fold trained on its own de-duplicated manifest whose training rows are
those of the pool's fold file, val labels aligned with the pool's committed fold manifest (the v3
and v4 val sides are identical, so the training rows are what tell the pools apart),
dev predictions in test_dev.csv order, and each fold's config describing exactly the kernel's
variant (v3_launch.config_variant: train.bag_loss, train.uda, preprocess.image_size). For arms H
and I the as-run config's train, preprocess and model sections must also equal the registered
YAML's (the pusher's VARIANT_CONFIGS), per-run keys aside (V4_PER_RUN_KEYS). test_locked is never
touched.

If one fold's dev scoring failed in the kernel, do not retrain it: score that fold's downloaded
weights.pth locally with models.common.v3_launch.score_manifest against
kaggle_cache/rop-staging-cache768, then collect.
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from models.common.v3_launch import (FIVE_FOLD_ONLY, LOSO_SOURCES, LOSO_VARIANTS,  # noqa: E402
                                     POOLS, VARIANTS, config_variant, fold_file)
from scripts.push_v3_kernels import VARIANT_CONFIGS  # noqa: E402

PREP = ROOT / "results/prepared/rop_staging"
RUNS = ROOT / "results/rop/v3/runs"
LR_PROBE = ROOT / "results/rop/v3/lr_probe.json"
FILES = ("val_predictions.npz", "dev_predictions.npz", "metrics.json", "config.yaml")
FOLDS = ({f"fold{k}": f"fold_{k}" for k in range(5)} | {"foldsite": "fold_site"}
         | {f"foldloso_{s}": f"fold_loso_{s}" for s in LOSO_SOURCES})
LOSO_FOLDS = tuple(f"foldloso_{s}" for s in LOSO_SOURCES)
# Arms H and I: the as-run config must be the registered YAML in these sections, except for the
# keys the kernel sets per run (kaggle_kernels/rop_v3_finetune.py's --set list; train.lr is written
# from a string and is checked against the summary and the lr probe instead).
V4_CHECKED_SECTIONS = ("train", "preprocess", "model")
V4_PER_RUN_KEYS = {"data.manifest", "seed", "train.epochs", "train.num_workers", "preprocess.cached",
                   "preprocess.cache_root", "model.weights_path", "train.lr"}


def may_run_loso(arch: str, variant: str) -> bool:
    """Amendment 5b: arms H and I (their LOSO folds are required) and arm B (a LOSO-only kernel)."""
    return variant in LOSO_VARIANTS or (arch, variant) == ("dinov2_s", "")


def _flat(d: dict, prefix: str = "") -> dict:
    """{"a.b.c": leaf} for a nested dict; lists and scalars are leaves."""
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict) and v:
            out |= _flat(v, key + ".")
        else:
            out[key] = v
    return out


def v4_config_drift(as_run: dict, registered: dict) -> list:
    """Dotted keys of the V4_CHECKED_SECTIONS where the as-run config and the registered YAML
    differ (a value differs, or a key is in one only), V4_PER_RUN_KEYS aside; sorted."""
    a, r = ({k: v for sec in V4_CHECKED_SECTIONS for k, v in _flat(c.get(sec) or {}, sec + ".").items()
             if k not in V4_PER_RUN_KEYS} for c in (as_run, registered))
    return sorted(k for k in a.keys() | r.keys() if k not in a or k not in r or a[k] != r[k])


def check_v4_config(cfg: dict, arch: str, variant: str, where) -> None:
    """Arms H and I only: ValueError naming every key where the fold's as-run config leaves the
    registered YAML (VARIANT_CONFIGS). Every other arm is checked by config_variant alone."""
    if variant not in LOSO_VARIANTS:
        return
    path = VARIANT_CONFIGS.get((arch, variant))
    if path is None:
        raise ValueError(f"{where}: no registered config for {arch} variant {variant!r}")
    drift = v4_config_drift(cfg, yaml.safe_load((ROOT / path).read_text()))
    if drift:
        raise ValueError(f"{where}: the as-run config differs from {path} at {', '.join(drift)}")


def check(src: Path, prep: Path = PREP) -> tuple[dict, dict]:
    """Validate one downloaded kernel output; return (summary, {fold name: experiment dir})
    for the folds that kernel completed (a kernel stopped by its time budget completes fewer)."""
    s = json.loads((src / "summary.json").read_text())
    if s.get("probe") or s.get("epochs") != 20:
        raise ValueError(f"{src}: not a main run (probe={s.get('probe')!r}, epochs={s.get('epochs')})")
    variant = s.get("variant", "")          # kernels before arm D wrote no variant key
    if variant not in VARIANTS:
        raise ValueError(f"{src}: unknown variant {variant!r}")
    pool, loso = s.get("pool", "v3"), s.get("loso", False)   # kernels before v4 wrote neither
    if pool not in POOLS:
        raise ValueError(f"{src}: unknown pool {pool!r}; known: {POOLS}")
    if (pool == "v4") != (variant in LOSO_VARIANTS):
        raise ValueError(f"{src}: variant {variant!r} on pool {pool!r}; arms H and I "
                         f"{LOSO_VARIANTS} train on the v4 pool, every other arm on v3 (Amendment 5a)")
    dev_paths = pd.read_csv(prep / "test_dev.csv").image_path.tolist()
    dups = set(pd.read_csv(prep / "exclude_duplicates.csv").image_path)
    dirs = {}
    for name, v in s["folds"].items():
        if v.get("macro_f1") is None:
            continue
        exp = src / "code" / v["dir"]
        for f in FILES:
            if not (exp / f).is_file():
                raise ValueError(f"{exp}: {f} missing")
        cfg = yaml.safe_load((exp / "config.yaml").read_text())
        got = (cfg["model"]["arch"], cfg["seed"], float(cfg["train"]["lr"]))
        if got != (s["arch"], s["seed"], float(s["lr"])):
            raise ValueError(f"{exp}: config {got} disagrees with summary")
        try:
            got_variant = config_variant(cfg)
        except ValueError as e:
            raise ValueError(f"{exp}: {e}") from None
        if got_variant != variant:
            raise ValueError(f"{exp}: the config is variant {got_variant!r} (train.bag_loss, "
                             f"train.uda, preprocess.image_size) but the kernel variant is {variant!r}")
        check_v4_config(cfg, s["arch"], variant, exp)
        if name == "foldsite" and variant in FIVE_FOLD_ONLY:
            raise ValueError(f"{exp}: a site fold in variant {variant!r}, which trains the five CV "
                             f"folds only (PROTOCOL_V2 Amendment 4)")
        if name in LOSO_FOLDS and not may_run_loso(s["arch"], variant):
            raise ValueError(f"{exp}: a LOSO fold in {s['arch']} variant {variant!r}; only arm B "
                             f"and arms H, I {LOSO_VARIANTS} run LOSO folds (Amendment 5b)")
        if name in LOSO_FOLDS and loso is not True:
            raise ValueError(f"{exp}: a LOSO fold in a kernel whose summary says loso={loso!r}")
        trained_on = Path(cfg["data"]["manifest"]).name
        if trained_on != f"_v3_{name}.csv":
            raise ValueError(f"{exp}: trained on {trained_on}, expected _v3_{name}.csv")
        used = pd.read_csv(src / "code" / cfg["data"]["manifest"])
        if dups & set(used[used.split == "train"].image_path):
            raise ValueError(f"{exp}: trained on excluded duplicate images (stale kernel source?)")
        man = pd.read_csv(fold_file(name, str(prep), pool))
        if set(used[used.split == "train"].image_path) != set(man[man.split == "train"].image_path) - dups:
            raise ValueError(f"{exp}: training rows are not those of pool {pool!r}'s "
                             f"{FOLDS[name]}.csv (less the excluded duplicates)")
        val = np.load(exp / "val_predictions.npz")
        if not np.array_equal(val["y_true"], man[man.split == "val"].label.values):
            raise ValueError(f"{exp}: val labels not aligned with pool {pool!r}'s {FOLDS[name]}.csv")
        dev = np.load(exp / "dev_predictions.npz")
        if dev["image_path"].tolist() != dev_paths:
            raise ValueError(f"{exp}: dev predictions not in test_dev.csv order")
        dirs[name] = exp
    return s, dirs


def collect(srcs, runs: Path = RUNS, prep: Path = PREP, lr_probe: Path = LR_PROBE) -> Path:
    """One arm-seed from one kernel, or from several when a kernel stopped early and the
    remaining folds were relaunched: every fold must come from exactly one kernel."""
    ident, dirs, folds = None, {}, {}
    for src in map(Path, srcs):
        s, d = check(src, prep)
        this = (s["arch"], s["seed"], float(s["lr"]), s.get("variant", ""))
        if ident not in (None, this):
            raise ValueError(f"{src}: {this} is a different arm-seed-lr-variant from {ident}")
        ident = this
        for name in d:
            if name in dirs:
                raise ValueError(f"{name} completed in two kernels ({dirs[name]}, {d[name]})")
            dirs[name] = d[name]
            folds[name] = {**s["folds"][name], "kernel": src.name, "gpu": s.get("gpu")}
    chosen = json.loads(lr_probe.read_text())["arms"][ident[0]]["chosen_lr"]
    if ident[2] != float(chosen):
        raise ValueError(f"lr {ident[2]} is not the probe's choice {chosen} for {ident[0]}")
    arch, seed, _, variant = ident
    out = runs / (f"{arch}_{variant}_s{seed}" if variant else f"{arch}_s{seed}")
    if not variant and any(n in LOSO_FOLDS for n in dirs):      # check(): arm B only
        return add_loso_folds(out, dirs, folds, ident)
    missing = [n for n in FOLDS if n not in dirs
               and not (n == "foldsite" and variant in FIVE_FOLD_ONLY)
               and not (n in LOSO_FOLDS and variant not in LOSO_VARIANTS)]
    if missing:
        raise ValueError(f"folds missing or failed: {missing}")
    if out.exists():
        raise FileExistsError(f"{out} exists; refusing to overwrite a collected run")
    tmp = out.with_name(out.name + ".partial")       # a failed copy never leaves a half run
    shutil.rmtree(tmp, ignore_errors=True)
    for name, exp in dirs.items():
        d = tmp / FOLDS[name]
        d.mkdir(parents=True)
        for f in FILES:
            shutil.copy2(exp / f, d / f)
    summary = {"arch": arch, "seed": seed, "lr": s["lr"], "epochs": 20, "folds": folds}
    if variant:
        summary["variant"] = variant
    if s.get("pool", "v3") != "v3":
        summary["pool"] = s["pool"]
    (tmp / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    tmp.rename(out)
    return out


def add_loso_folds(out: Path, dirs: dict, folds: dict, ident: tuple) -> Path:
    """Arm B's LOSO-only kernel (Amendment 5b): its two LOSO folds join B's collected run as two
    new fold dirs, and summary.json gains their two entries (the rest kept). Nothing already in
    the run is overwritten: a LOSO fold dir or summary entry already there is a FileExistsError."""
    if set(dirs) != set(LOSO_FOLDS):
        raise ValueError(f"arm B's LOSO folds come from a LOSO-only kernel (both LOSO folds, "
                         f"nothing else; its CV and site folds are collected already): got {sorted(dirs)}")
    if not (out / "summary.json").is_file():
        raise FileNotFoundError(f"{out}: collect arm B's run before adding its LOSO folds")
    summary = json.loads((out / "summary.json").read_text())
    have = (summary["arch"], summary["seed"], float(summary["lr"]), summary.get("variant", ""))
    if have != ident:
        raise ValueError(f"{out}: the collected run is {have}, the LOSO kernel {ident}")
    clash = [n for n in LOSO_FOLDS if n in summary["folds"] or (out / FOLDS[n]).exists()]
    if clash:
        raise FileExistsError(f"{out}: {clash} already collected; refusing to overwrite")
    tmp = out.with_name(out.name + ".loso.partial")    # a failed copy never touches the run
    shutil.rmtree(tmp, ignore_errors=True)
    for name, exp in dirs.items():
        d = tmp / FOLDS[name]
        d.mkdir(parents=True)
        for f in FILES:
            shutil.copy2(exp / f, d / f)
    summary["folds"] |= {n: folds[n] for n in LOSO_FOLDS}
    (tmp / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    for n in LOSO_FOLDS:
        (tmp / FOLDS[n]).rename(out / FOLDS[n])
    (tmp / "summary.json").replace(out / "summary.json")
    tmp.rmdir()
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("src", type=Path, nargs="+",
                    help="kaggle_cache/v3/<slug> (`kaggle kernels output` folders of one arm-seed)")
    a = ap.parse_args()
    out = collect(a.src)
    s = json.loads((out / "summary.json").read_text())
    print(f"{out.relative_to(ROOT)}: " + "  ".join(
        f"{n} mF1 {v['macro_f1']:.3f} QWK {v['qwk']:.3f}" for n, v in s["folds"].items()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
