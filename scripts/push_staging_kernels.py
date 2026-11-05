"""Push one Kaggle kernel per backbone for the ROP staging comparison.

    .venv/bin/python scripts/push_staging_kernels.py --dry-run
    .venv/bin/python scripts/push_staging_kernels.py --arch convnext_in12k
    .venv/bin/python scripts/push_staging_kernels.py --all --seed 42

Each kernel is kaggle_kernels/rop_staging_benchmark.py with ARCH and SEED substituted, run
against the three attached datasets. One kernel per backbone, because five backbones x five
folds in a single session runs 12-18h against Kaggle's ~9h cap.

INTERNET IS ON, for one reason only. Kaggle allocates either a T4 (sm_75) or a P100 (sm_60)
and exposes no way to ask for one; the preinstalled torch carries no sm_60 kernels, so a P100
run dies on its first forward pass. The kernel detects that and installs a cu121 build, which
needs the network. Pretrained weights are still read from the attached dataset via HF_HOME and
TORCH_HOME — no backbone is ever fetched over the wire, which is the part that would otherwise
show up as an unexplained gap between two arms of the comparison.

GPU BUDGET. Kaggle allows roughly 30 GPU-hours per week. One backbone x five folds is ~3h, so
a full five-backbone sweep at one seed is ~15h — half the weekly budget. The 3-seed repeat
that makes the ranking meaningful is ~45h, i.e. two more weeks. Plan the seeds accordingly;
do not start all fifteen and discover the quota mid-sweep.
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
KERNEL = ROOT / "kaggle_kernels" / "rop_staging_benchmark.py"
USER = "praharshnagpure"
DATASETS = [
    f"{USER}/retinal-framework-code",
    f"{USER}/rop-staging-corpus",
    f"{USER}/rop-staging-backbones",
]
ARCHES = ["convnext_in12k", "caformer", "effnetv2s", "deit3", "effnetb0"]


def build(arch: str, seed: int, folds: str | None, out: Path,
          sampler: str = "true", no_site_fold: bool = False, suffix: str = ""):
    src = KERNEL.read_text()
    # The kernel declares its knobs as plain module-level assignments precisely so they can be
    # rewritten here without a config layer.
    #
    # SUBSTITUTE ON LINE-ANCHORED ASSIGNMENTS, and require EXACTLY ONE match. A plain
    # str.replace(old, new, 1) is not safe here and this is not hypothetical: the kernel's
    # module docstring also contains the text `ARCH = "convnext_in12k"` while listing the
    # options, so the first replacement landed in the docstring and the real assignment was
    # left untouched. Every backbone would have trained ConvNeXt, and the output would have
    # looked like a finished five-model comparison. The `old in src` check passed, because the
    # string genuinely was there — just not where it mattered.
    subs = [(r'^ARCH = ".*?"', f'ARCH = "{arch}"'),
            (r"^SEED = \d+", f"SEED = {seed}"),
            (r'^WEIGHTED_SAMPLER = ".*?"', f'WEIGHTED_SAMPLER = "{sampler}"')]
    if folds:
        subs.append((r"^FOLDS = \[.*?\]", f"FOLDS = [{folds}]"))
    if no_site_fold:
        subs.append((r"^ALSO_SITE_FOLD = (?:True|False)", "ALSO_SITE_FOLD = False"))
    for pat, new in subs:
        rx = re.compile(pat, re.M)
        n = len(rx.findall(src))
        if n != 1:
            raise SystemExit(f"pattern {pat!r} matched {n} module-level assignments in the "
                             f"kernel (need exactly 1) — refusing to push a job whose knobs "
                             f"may not have been set")
        src = rx.sub(lambda _: new, src, count=1)

    # Belt and braces: read the value back out of the generated file.
    got = re.search(r'^ARCH = "(.*?)"', src, re.M)
    if not got or got.group(1) != arch:
        raise SystemExit(f"generated kernel has ARCH={got and got.group(1)!r}, expected {arch!r}")

    out.mkdir(parents=True, exist_ok=True)
    (out / "rop_staging_benchmark.py").write_text(src)
    slug = (f"rop-staging-{arch.replace('_', '-')}-s{seed}"
            + ("" if sampler == "true" else "-natural")
            + (f"-{suffix}" if suffix else ""))
    meta = {
        "id": f"{USER}/{slug}",
        "title": (f"ROP staging {arch} s{seed}"
                  + ("" if sampler == "true" else " natural")
                  + (f" {suffix}" if suffix else ""))[:50],
        "code_file": "rop_staging_benchmark.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        # Internet is ON, reluctantly. Kaggle hands out a T4 (sm_75) or a P100 (sm_60) at
        # random with no API field to choose, and the preinstalled torch has no sm_60 kernels
        # — on a P100 nothing runs at all. The kernel's fallback installs a cu121 build, and
        # that needs the network. Determinism is preserved where it matters: HF_HOME and
        # TORCH_HOME point at the attached weights dataset, so no backbone is ever fetched
        # over the wire, and which GPU a run landed on is printed in its log.
        "enable_internet": True,
        "dataset_sources": DATASETS,
        "competition_sources": [],
        "kernel_sources": [],
    }
    (out / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    return slug


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", choices=ARCHES)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--folds", help="comma-separated fold list, e.g. '0' for a single-fold probe")
    ap.add_argument("--sampler", default="true", choices=["true", "false"],
                    help="'false' = natural-sampling ablation arm (slug gets -natural)")
    ap.add_argument("--no-site-fold", action="store_true")
    ap.add_argument("--slug-suffix", default="",
                    help="distinguishes split runs of the same config (e.g. 'a'/'b' — six "
                         "folds in one session trips the kernel's 8h budget guard)")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    targets = ARCHES if a.all else ([a.arch] if a.arch else [])
    if not targets:
        return ap.error("pass --arch <name> or --all")

    for arch in targets:
        tmp = Path(tempfile.mkdtemp(prefix=f"kernel_{arch}_"))
        try:
            slug = build(arch, a.seed, a.folds, tmp,
                         sampler=a.sampler, no_site_fold=a.no_site_fold,
                         suffix=a.slug_suffix)
            if a.dry_run:
                print(f"  [dry-run] {slug}")
                print(f"            {json.loads((tmp/'kernel-metadata.json').read_text())}")
                continue
            r = subprocess.run([sys.executable, "-m", "kaggle", "kernels", "push", "-p", str(tmp)],
                               capture_output=True, text=True)
            ok = r.returncode == 0
            print(f"  {'OK  ' if ok else 'FAIL'} {slug}")
            print("      " + (r.stdout or r.stderr).strip().replace("\n", "\n      "))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
    if not a.dry_run:
        print(f"\nWatch: https://www.kaggle.com/{USER}/code")
        print("Rank on CV mean +- SD across 3 seeds. A gap smaller than the SD is not a result.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
