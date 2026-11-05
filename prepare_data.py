#!/usr/bin/env python
"""Build the uniform manifest for a disease, optionally running Dataset QA.

    python prepare_data.py --config configs/dr.yaml --dqa
    python prepare_data.py --config configs/glaucoma.yaml --set data.val_split=0.2
"""
import argparse

from models.common.config import load_config
from models.common.data_prep import build_manifest
from models.common.data_quality import run_dqa


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", nargs="*", default=[], help="key.subkey=value overrides")
    ap.add_argument("--dqa", action="store_true", help="run dataset QA + write report")
    ap.add_argument("--deep-sample", type=int, default=2000)
    args = ap.parse_args()

    cfg = load_config(args.config, overrides=args.set)
    df = build_manifest(cfg)

    if args.dqa:
        run_dqa(df,
                output_pdf=f"thesis_assets/dataset_report_{cfg.disease}.pdf",
                num_classes=int(cfg.data.num_classes),
                class_names=list(cfg.data.class_names),
                deep_sample=args.deep_sample,
                seed=int(cfg.seed),
                cleaned_manifest=str(cfg.data.manifest).replace(".csv", "_clean.csv"))


if __name__ == "__main__":
    main()
