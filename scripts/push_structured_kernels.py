"""Push Kaggle kernels for the structured-model ablation arms.

    .venv/bin/python scripts/push_structured_kernels.py --dry-run \
        --slug wsite0 --w-site 0.0 --folds 0
    .venv/bin/python scripts/push_structured_kernels.py \
        --slug sampler --sampler true --folds 0 --no-site-fold

Same substitution discipline as push_staging_kernels.py: line-anchored regex on the
module-level knob assignments, exactly one match required, value read back out of the
generated file before anything is pushed. The staging pusher exists because a plain
str.replace once landed in the kernel's docstring and would have trained the wrong model
five times; this script inherits the fix rather than the bug.

Ablation arms this parameterizes (all fold 0 unless --folds says otherwise):
  wsite0     W_SITE=0.0              — true -DANN: same head, adversary off entirely
  sampler    WEIGHTED_SAMPLER=true   — closes the flat-vs-structured sampling confound
                                       (flat trained weighted, structured natural)
  noordinal  VARIANT=no_ordinal      — softmax stages + AP-ROP branch (what CORN adds)
  nobranch   VARIANT=no_branch       — CORN over 6 forced ranks (what the branch adds)
The variant arms need the code dataset on Kaggle to carry variant-aware
train_structured.py / models/common/structured.py — rebuild and version the bundle first.
Each fold is ~70 min on a T4/P100; Kaggle runs at most 2 kernels concurrently and
SILENTLY drops pushes beyond that — verify with `kaggle kernels status` after pushing.
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
KERNEL = ROOT / "kaggle_kernels" / "rop_structured_benchmark.py"
USER = "praharshnagpure"
DATASETS = [
    f"{USER}/retinal-framework-code",
    f"{USER}/rop-staging-corpus",
    f"{USER}/rop-staging-backbones",
]


def build(a, out: Path) -> str:
    src = KERNEL.read_text()
    subs = [
        (r'^W_SITE = ".*?"', f'W_SITE = "{a.w_site}"'),
        (r'^WEIGHTED_SAMPLER = ".*?"', f'WEIGHTED_SAMPLER = "{a.sampler}"'),
        (r'^VARIANT = ".*?"', f'VARIANT = "{a.variant}"'),
        (r"^SEED = \d+", f"SEED = {a.seed}"),
        (r"^FOLDS = \[.*?\]", f"FOLDS = [{a.folds}]"),
        (r"^ALSO_SITE_FOLD = (?:True|False)",
         f"ALSO_SITE_FOLD = {not a.no_site_fold}"),
    ]
    for pat, new in subs:
        rx = re.compile(pat, re.M)
        n = len(rx.findall(src))
        if n != 1:
            raise SystemExit(f"pattern {pat!r} matched {n} module-level assignments "
                             f"(need exactly 1) — refusing to push")
        src = rx.sub(lambda _: new, src, count=1)

    # Belt and braces: read every knob back out of the generated file.
    checks = {
        r'^W_SITE = "(.*?)"': a.w_site,
        r'^WEIGHTED_SAMPLER = "(.*?)"': a.sampler,
        r'^VARIANT = "(.*?)"': a.variant,
        r"^SEED = (\d+)": str(a.seed),
        r"^FOLDS = \[(.*?)\]": a.folds,
    }
    for pat, want in checks.items():
        got = re.search(pat, src, re.M)
        if not got or got.group(1) != want:
            raise SystemExit(f"generated kernel: {pat} -> {got and got.group(1)!r}, "
                             f"expected {want!r}")

    out.mkdir(parents=True, exist_ok=True)
    (out / "rop_structured_benchmark.py").write_text(src)
    # slug "main" = a full seed-repeat run, named like the original CV kernel rather than
    # as an ablation arm.
    if a.slug == "main":
        slug = f"rop-structured-effnetv2s-s{a.seed}"
        title = f"ROP structured effnetv2s s{a.seed}"
    else:
        slug = f"rop-structured-ablation-{a.slug}-s{a.seed}"
        title = f"ROP structured ablation {a.slug} s{a.seed}"
    meta = {
        "id": f"{USER}/{slug}",
        "title": title[:50],
        "code_file": "rop_structured_benchmark.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        # Internet ON only for the sm_60/cu121 torch fallback; weights come from the
        # attached dataset via HF_HOME/TORCH_HOME (see push_staging_kernels.py).
        "enable_internet": True,
        "dataset_sources": DATASETS,
        "competition_sources": [],
        "kernel_sources": [],
    }
    (out / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    return slug


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slug", required=True,
                    help="arm name, becomes rop-structured-ablation-<slug>-s<seed>")
    ap.add_argument("--w-site", default="0.5")
    ap.add_argument("--sampler", default="false", choices=["true", "false"])
    ap.add_argument("--variant", default="full",
                    choices=["full", "no_ordinal", "no_branch"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--folds", default="0", help="comma-separated, e.g. '0'")
    ap.add_argument("--no-site-fold", action="store_true",
                    help="skip the cross-site fold (default runs it)")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    tmp = Path(tempfile.mkdtemp(prefix=f"kernel_structured_{a.slug}_"))
    try:
        slug = build(a, tmp)
        if a.dry_run:
            print(f"  [dry-run] {slug}")
            print(f"            {json.loads((tmp / 'kernel-metadata.json').read_text())}")
            return 0
        r = subprocess.run([sys.executable, "-m", "kaggle", "kernels", "push",
                            "-p", str(tmp)], capture_output=True, text=True)
        ok = r.returncode == 0
        print(f"  {'OK  ' if ok else 'FAIL'} {slug}")
        print("      " + (r.stdout or r.stderr).strip().replace("\n", "\n      "))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("\nVerify it actually exists (Kaggle silently drops pushes beyond 2 running):")
    print(f"  .venv/bin/python -m kaggle kernels status {USER}/{slug}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
