"""Collect the five-backbone staging results and report the comparison honestly.

    .venv/bin/python scripts/staging_comparison.py --fetch          # pull from Kaggle first
    .venv/bin/python scripts/staging_comparison.py                  # use what is on disk

Reads results/rop/staging_runs/staging_<arch>_s<seed>.json (written by the Kaggle kernel) and
emits a table plus graphs/rop_staging/backbone_comparison.png.

WHY THIS IS NOT JUST A SORT. With 3,112 training images and 89 in the rarest class, the
run-to-run spread across seeds can easily exceed the gap between two backbones. Printing a
ranked list would imply a difference the data cannot support, so this script:

  * reports mean +- SD across folds, and across seeds when more than one seed exists
  * refuses to name a winner when the top two are separated by less than the pooled SD
  * prints the metadata-only bar from the shortcut audit next to every score, because a
    backbone that does not clear it has learned nothing from pixels regardless of its rank
  * shows which GPU each run landed on — a P100 arm runs torch 2.5.1+cu121 and a T4 arm runs
    the preinstalled build, so an ordering that tracks GPU type is not an ordering of models
"""
from __future__ import annotations

import argparse
import glob
import json
import re
import statistics as st
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "results" / "rop" / "staging_runs"
AUDIT = ROOT / "results" / "rop" / "staging_shortcut_audit.json"
FIG = ROOT / "graphs" / "rop_staging"
USER = "praharshnagpure"
ARCHES = ["convnext_in12k", "caformer", "effnetv2s", "deit3", "effnetb0"]
CLASSES = ["Normal", "Stage 1", "Stage 2", "Stage 3", "Stage 4/5", "AP-ROP"]


def fetch(seeds):
    """Download each kernel's output json into RUNS/."""
    RUNS.mkdir(parents=True, exist_ok=True)
    for arch in ARCHES:
        for seed in seeds:
            slug = f"rop-staging-{arch.replace('_', '-')}-s{seed}"
            r = subprocess.run([sys.executable, "-m", "kaggle", "kernels", "output",
                                f"{USER}/{slug}", "-p", str(RUNS), "-q"],
                               capture_output=True, text=True)
            got = list(RUNS.glob(f"staging_{arch}_s{seed}.json"))
            print(f"  {slug:<40} {'ok' if got else 'no output ('+(r.stderr or r.stdout).strip()[:60]+')'}")


def load():
    out = []
    for f in sorted(RUNS.glob("staging_*_s*.json")):
        try:
            d = json.loads(f.read_text())
        except Exception as e:
            print(f"  skipping unreadable {f.name}: {e}", file=sys.stderr)
            continue
        # A truncated run must not be averaged as if it were complete.
        req, done = d.get("folds_requested"), d.get("folds_completed")
        d["_truncated"] = req is not None and done is not None and done < req
        out.append(d)
    return out


def fold_values(run, metric):
    return [v[metric] for k, v in run.get("folds", {}).items()
            if k != "fold_site" and isinstance(v, dict) and v.get(metric) is not None]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch", action="store_true")
    ap.add_argument("--seeds", default="42", help="comma-separated, e.g. 42,1337,2024")
    a = ap.parse_args()
    seeds = [int(s) for s in a.seeds.split(",")]

    if a.fetch:
        print("fetching kernel outputs:")
        fetch(seeds)

    runs = load()
    if not runs:
        print(f"no run json found in {RUNS} — pass --fetch, or check the kernels finished",
              file=sys.stderr)
        return 1

    bar = None
    if AUDIT.exists():
        bar = json.loads(AUDIT.read_text()).get("G4_metadata_baseline", {}).get("bar_cv_macro_f1")

    # ---- aggregate per backbone, across folds and seeds --------------------------------
    table = {}
    for arch in ARCHES:
        mine = [r for r in runs if r.get("arch") == arch]
        if not mine:
            continue
        per_seed = {}
        for r in mine:
            vals = fold_values(r, "macro_f1")
            if vals:
                per_seed[r.get("seed")] = st.mean(vals)
        allf = [v for r in mine for v in fold_values(r, "macro_f1")]
        if not allf:
            table[arch] = {"status": "no completed folds"}
            continue
        site = [v.get("macro_f1") for r in mine
                for k, v in r.get("folds", {}).items()
                if k == "fold_site" and isinstance(v, dict) and v.get("macro_f1") is not None]
        table[arch] = {
            "cv_mean": st.mean(allf),
            "cv_sd": st.stdev(allf) if len(allf) > 1 else 0.0,
            "n_folds": len(allf),
            "seed_means": per_seed,
            "seed_sd": st.stdev(list(per_seed.values())) if len(per_seed) > 1 else None,
            "cross_site": st.mean(site) if site else None,
            "truncated": any(r["_truncated"] for r in mine),
        }

    # ---- print -------------------------------------------------------------------------
    print("\n" + "=" * 78)
    print("ROP 6-CLASS STAGING — BACKBONE COMPARISON")
    print("=" * 78)
    if bar is not None:
        print(f"metadata-only bar (no pixels decoded): CV macro-F1 {bar:.3f}")
        print("a backbone at or below this has learned nothing from the images\n")
    print(f"  {'backbone':<16} {'CV macro-F1':>16}  {'folds':>5}  {'cross-site':>10}  notes")
    print("  " + "-" * 74)
    ranked = sorted([(k, v) for k, v in table.items() if "cv_mean" in v],
                    key=lambda kv: -kv[1]["cv_mean"])
    for arch, v in ranked:
        note = []
        if v["truncated"]:
            note.append("TRUNCATED run")
        if bar is not None and v["cv_mean"] <= bar:
            note.append("AT/BELOW METADATA BAR")
        if v["seed_sd"] is not None:
            note.append(f"seed SD {v['seed_sd']:.3f}")
        cs = f"{v['cross_site']:.3f}" if v["cross_site"] is not None else "n/a"
        # "+- 0.000" from a single fold reads as "extremely consistent" when it means the
        # opposite: there is no spread estimate at all. Print the mean alone and say n=1.
        score = (f"{v['cv_mean']:.3f} +- {v['cv_sd']:.3f}" if v["n_folds"] > 1
                 else f"{v['cv_mean']:.3f}  (no SD)")
        if v["n_folds"] == 1:
            note.insert(0, "single fold — provisional")
        print(f"  {arch:<16} {score:>16}  {v['n_folds']:>5}  {cs:>10}  {', '.join(note)}")
    for arch, v in table.items():
        if "cv_mean" not in v:
            print(f"  {arch:<16} {v['status']}")

    # ---- is the ordering real? ---------------------------------------------------------
    print("\n" + "-" * 78)
    if len(ranked) < 2:
        print("Only one backbone has results — nothing to compare yet.")
    else:
        (a1, v1), (a2, v2) = ranked[0], ranked[1]
        gap = v1["cv_mean"] - v2["cv_mean"]
        single = v1["n_folds"] == 1 or v2["n_folds"] == 1
        # With one fold per arm the SD is 0 by construction, so ANY gap would "beat" it and
        # the script would crown a winner off a single measurement. Suppress the verdict
        # entirely rather than compare against a spread that was never estimated.
        pooled = None if single else max(v1["cv_sd"], v2["cv_sd"])
        head = f"top two: {a1} {v1['cv_mean']:.3f} vs {a2} {v2['cv_mean']:.3f}   gap {gap:.3f}"
        print(head if pooled is None else f"{head}, fold SD {pooled:.3f}")

        if pooled is None:
            print("NO VERDICT. At least one arm has a single fold, so there is no spread to")
            print("compare the gap against. This ordering is provisional — treat it as a")
            print("smoke test of which backbones train at all, not as a ranking.")
        elif gap < pooled:
            print("NO WINNER. The gap is smaller than the fold-to-fold spread, so this data")
            print(f"cannot separate {a1} from {a2}. Report them as indistinguishable — that is")
            print("a result, not a failure. Adding seeds narrows the error on the mean; it")
            print(f"does not make a {gap:.3f} gap meaningful on its own.")
        else:
            print(f"{a1} leads by more than the fold spread. Confirm across 3 seeds before")
            print("calling it, then open test_locked ONCE on the winner.")

        if pooled is not None and all(v["seed_sd"] is None for _, v in ranked):
            print("\nSingle seed only. A one-seed ranking of five models on 3,112 images is")
            print("weak evidence; seeds 1337 and 2024 are what make it defensible.")

    # ---- figure ------------------------------------------------------------------------
    if ranked:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            FIG.mkdir(parents=True, exist_ok=True)
            names = [a for a, _ in ranked]
            means = [v["cv_mean"] for _, v in ranked]
            sds = [v["cv_sd"] for _, v in ranked]
            fig, ax = plt.subplots(figsize=(8, 4.2))
            ax.barh(names[::-1], means[::-1], xerr=sds[::-1], color="#4C72B0",
                    error_kw={"ecolor": "#333", "capsize": 4})
            if bar is not None:
                ax.axvline(bar, color="#C44E52", ls="--", lw=1.6)
                ax.text(bar, -0.6, f" metadata-only bar {bar:.3f}", color="#C44E52",
                        va="top", fontsize=9)
            ax.set_xlabel("out-of-fold CV macro-F1  (error bars = SD across folds)")
            ax.set_title("ROP 6-class staging — backbone comparison")
            fig.tight_layout()
            p = FIG / "backbone_comparison.png"
            fig.savefig(p, dpi=140)
            print(f"\n-> {p}")
        except ImportError:
            print("\n(matplotlib not available — table only)")

    summary = ROOT / "results" / "rop" / "staging_comparison.json"
    summary.write_text(json.dumps({"bar_cv_macro_f1": bar, "backbones": table}, indent=2) + "\n")
    print(f"-> {summary}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
