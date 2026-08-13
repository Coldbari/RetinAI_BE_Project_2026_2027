#!/usr/bin/env python
"""Turn a sweep CSV (from benchmark.py --sweep) into a per-disease comparison table
(markdown + CSV) with a written justification of the recommended model.

    python -m models.comparison.make_table --sweep results/sweep_dr.csv
"""
import argparse
from pathlib import Path

import pandas as pd

# Columns ranked higher-is-better; latency/size are lower-is-better.
HIGHER = ["accuracy", "macro_f1", "qwk", "auc", "cpu_fps", "gpu_fps"]
LOWER = ["cpu_ms", "gpu_ms", "gpu_batch16_ms", "size_MB", "params_M"]


def recommend(df: pd.DataFrame, primary: str) -> str:
    metric = primary if primary in df.columns else "macro_f1"
    df = df.copy()
    df["_score"] = df[metric].rank(ascending=False)
    if "gpu_ms" in df.columns:
        df["_score"] += 0.25 * df["gpu_ms"].rank(ascending=True)
    elif "cpu_ms" in df.columns:
        df["_score"] += 0.25 * df["cpu_ms"].rank(ascending=True)
    best = df.sort_values("_score").iloc[0]
    return (f"Recommended: **{best['arch']}** — best {metric} ({best[metric]:.4f}) "
            f"with competitive latency. Selection should be confirmed with the W9 "
            f"statistical tests (McNemar / DeLong) before finalising.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", required=True, help="results/sweep_<disease>.csv")
    ap.add_argument("--primary", default="macro_f1")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    df = pd.read_csv(args.sweep)
    out = Path(args.out) if args.out else Path(args.sweep).with_suffix(".md")

    lines = ["# Architecture comparison", "", df.to_markdown(index=False), "",
             recommend(df, args.primary)]
    out.write_text("\n".join(lines))
    df.to_csv(out.with_suffix(".table.csv"), index=False)
    print("\n".join(lines))
    print(f"\n[make_table] -> {out}")


if __name__ == "__main__":
    main()
