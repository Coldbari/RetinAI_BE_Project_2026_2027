"""Push one v3 ROP fine-tuning kernel (a ViT arm x one seed), or an inner lr probe.

    .venv/bin/python scripts/push_v3_kernels.py --arch retfound_green --seed 42
    .venv/bin/python scripts/push_v3_kernels.py --arch retfound_green --seed 42 --folds 3 \
        --no-site-fold --slug-suffix r1        # relaunch of one fold: suffix, no second site fold
    .venv/bin/python scripts/push_v3_kernels.py --arch dinov2_s --probe inner --lr 3e-5
    .venv/bin/python scripts/push_v3_kernels.py --arch dinov2_s --variant bag --seed 42
        # arm D (PROTOCOL_V2 Amendment 3c): the bag-loss config; the arch's protocol lr, even
        # for a --probe (a smoke run), since D's lr is fixed by the amendment
    .venv/bin/python scripts/push_v3_kernels.py --arch dinov2_s --variant r518 --seed 42 \
        --folds 0,1,2 --no-site-fold --slug-suffix a      # arm G (Amendment 4a), then 3,4 as "b";
        # G trains the five CV folds only, so a site fold is refused; arm F (Amendment 4b) is
        # --variant uda, launched the same way
    .venv/bin/python scripts/push_v3_kernels.py --arch dinov2_s --variant v4data --pool v4 \
        --folds 0,1,2,3,4 --no-site-fold --slug-suffix a   # arm H (Amendment 5b), slug rop-v4-...;
        # then --folds "" --loso --slug-suffix b: the site fold and both LOSO folds. Arm I is
        # --variant v4recipe. H and I train on the v4 pool only, every other arm on v3.
    .venv/bin/python scripts/push_v3_kernels.py --arch dinov2_s --folds "" --no-site-fold \
        --loso --slug-suffix loso       # arm B's LOSO-only kernel (v3 pool): the two LOSO folds

Knobs are rewritten by line-anchored substitution that must match EXACTLY ONE line (a plain
replace once hit a docstring and every "different" backbone trained the same model). Kaggle
silently drops a third concurrent push: re-list with `kaggle kernels list --mine`.
"""
import argparse
import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from models.common.v3_launch import LOSO_VARIANTS, POOLS, check_loso, check_pool  # noqa: E402

KERNEL = ROOT / "kaggle_kernels" / "rop_v3_finetune.py"
USER = "praharshnagpure"
DATASETS = [f"{USER}/retinal-framework-code", f"{USER}/rop-staging-cache768",
            f"{USER}/rop-v3-weights"]
CONFIGS = {"retfound_green": "configs/rop_staging_rfg.yaml",
           "dinov2_s": "configs/rop_staging_dinov2.yaml"}
# (arch, variant) -> config of a variant arm; the variant joins the slug after the arch.
VARIANT_CONFIGS = {("dinov2_s", "bag"): "configs/rop_staging_dinov2_bag.yaml",
                   ("dinov2_s", "r518"): "configs/rop_staging_dinov2_518.yaml",
                   ("dinov2_s", "uda"): "configs/rop_staging_dinov2_uda.yaml",
                   ("dinov2_s", "v4data"): "configs/rop_staging_dinov2_v4data.yaml",       # arm H
                   ("dinov2_s", "v4recipe"): "configs/rop_staging_dinov2_v4recipe.yaml"}   # arm I
FIVE_FOLD_ONLY = ("r518", "uda")  # PROTOCOL_V2 Amendment 4: these variants train no site fold
LR_PROBE = ROOT / "results/rop/v3/lr_probe.json"


def protocol_lr(arch: str, lr: str | None, probe_json: Path = LR_PROBE) -> str:
    """A main (non-probe) run trains at the peak lr the probe fixed (PROTOCOL_V2 Amendment 2):
    filled in when --lr is omitted, refused when --lr says otherwise."""
    if not probe_json.exists():
        raise SystemExit(f"{probe_json} missing: run the lr probe before any main run")
    chosen = json.loads(probe_json.read_text())["arms"][arch]["chosen_lr"]
    if lr is not None and float(lr) != float(chosen):
        raise SystemExit(f"--lr {lr} contradicts the probe's choice {chosen} for {arch}")
    return chosen


def build(arch: str, seed: int, out: Path, folds: str | None = None, no_site_fold: bool = False,
          lr: str | None = None, epochs: int | None = None, probe: str = "",
          suffix: str = "", variant: str = "", pool: str = "v3", loso: bool = False) -> str:
    if arch not in CONFIGS:
        raise SystemExit(f"unknown arch {arch!r}; choose from {sorted(CONFIGS)}")
    if variant and (arch, variant) not in VARIANT_CONFIGS:
        raise SystemExit(f"no variant {variant!r} for {arch!r}; known: {sorted(VARIANT_CONFIGS)}")
    config = VARIANT_CONFIGS[(arch, variant)] if variant else CONFIGS[arch]
    if probe not in ("", "inner"):
        raise SystemExit("--probe must be 'inner'")
    if probe == "inner":
        folds, no_site_fold, loso = "0", True, False
    if variant in FIVE_FOLD_ONLY and not no_site_fold:
        raise SystemExit(f"variant {variant!r} trains the five CV folds only (PROTOCOL_V2 "
                         f"Amendment 4): no site fold, pass --no-site-fold")
    check_pool(pool)
    if (pool == "v4") != (variant in LOSO_VARIANTS):
        raise SystemExit(f"variant {variant!r} on pool {pool!r}: arms H and I {LOSO_VARIANTS} train "
                         f"on the v4 pool (--pool v4), every other arm on v3 (PROTOCOL_V2 Amendment 5a)")
    check_loso(variant, loso)
    if loso and not variant and (arch != "dinov2_s" or folds != "" or not no_site_fold):
        raise SystemExit("arm B's LOSO kernel is LOSO-only (--folds '' --no-site-fold: its CV and "
                         "site folds are collected already); arm A runs no LOSO folds")
    src = KERNEL.read_text()
    subs = [(r'^ARCH = ".*?"', f'ARCH = "{arch}"'), (r"^SEED = \d+", f"SEED = {seed}"),
            (r'^CONFIG = ".*?"', f'CONFIG = "{config}"')]
    if variant:
        subs.append((r'^VARIANT = ".*?"', f'VARIANT = "{variant}"'))
    if folds is not None:                               # "" -> FOLDS = [] (site / LOSO folds only)
        subs.append((r"^FOLDS = \[.*?\]", f"FOLDS = [{folds}]"))
    if no_site_fold:
        subs.append((r"^ALSO_SITE_FOLD = (?:True|False)", "ALSO_SITE_FOLD = False"))
    if lr:
        subs.append((r'^LR = ".*?"', f'LR = "{lr}"'))
    if epochs is not None:
        subs.append((r"^EPOCHS = \d+", f"EPOCHS = {epochs}"))
    if probe:
        subs.append((r'^PROBE = ".*?"', f'PROBE = "{probe}"'))
    if pool != "v3":
        subs.append((r'^POOL = ".*?"', f'POOL = "{pool}"'))
    if loso:
        subs.append((r"^LOSO = (?:True|False)", f"LOSO = {loso}"))
    for pat, new in subs:
        rx = re.compile(pat, re.M)
        n = len(rx.findall(src))
        if n != 1:
            raise SystemExit(f"pattern {pat!r} matched {n} module-level assignments in the "
                             f"kernel (need exactly 1) - refusing to push")
        src = rx.sub(lambda _: new, src, count=1)
    got = re.search(r'^ARCH = "(.*?)"', src, re.M)
    if not got or got.group(1) != arch:
        raise SystemExit(f"generated kernel has ARCH={got and got.group(1)!r}, expected {arch!r}")

    out.mkdir(parents=True, exist_ok=True)
    (out / KERNEL.name).write_text(src)
    slug = (f"rop-{pool}-{arch.replace('_', '-')}" + (f"-{variant}" if variant else "") + f"-s{seed}"
            + (f"-probe-lr{lr}" if probe else (f"-lr{lr}" if lr else ""))
            + (f"-{suffix}" if suffix else ""))
    meta = {"id": f"{USER}/{slug}", "title": slug[:50].replace("-", " "),
            "code_file": KERNEL.name, "language": "python", "kernel_type": "script",
            "is_private": True, "enable_gpu": True, "enable_tpu": False,
            # internet only for the cu121 torch fallback on a P100; weights come from datasets
            "enable_internet": True, "dataset_sources": DATASETS,
            "competition_sources": [], "kernel_sources": []}
    (out / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    return slug


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", required=True, choices=sorted(CONFIGS))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--folds", help="comma-separated, e.g. '0,1,2'; '' for none (site/LOSO only)")
    ap.add_argument("--no-site-fold", action="store_true")
    ap.add_argument("--lr")
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--probe", default="", choices=["", "inner"])
    ap.add_argument("--variant", default="", choices=[""] + sorted({v for _, v in VARIANT_CONFIGS}))
    ap.add_argument("--pool", default="v3", choices=POOLS, help="v4 for arms H and I only")
    ap.add_argument("--loso", action="store_true", help="also the two leave-one-source-out folds")
    ap.add_argument("--slug-suffix", default="")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    if not a.probe or a.variant:
        a.lr = protocol_lr(a.arch, a.lr)
    tmp = Path(tempfile.mkdtemp(prefix="kernel_v3_"))
    try:
        slug = build(a.arch, a.seed, tmp, folds=a.folds, no_site_fold=a.no_site_fold, lr=a.lr,
                     epochs=a.epochs, probe=a.probe, suffix=a.slug_suffix, variant=a.variant,
                     pool=a.pool, loso=a.loso)
        if a.dry_run:
            print(f"[dry-run] {slug}")
            return 0
        r = subprocess.run([sys.executable, "-m", "kaggle", "kernels", "push", "-p", str(tmp)],
                           capture_output=True, text=True)
        print(f"{'OK  ' if r.returncode == 0 else 'FAIL'} {slug}\n"
              + (r.stdout or r.stderr).strip())
        return r.returncode
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
