"""Score the held-out hospital — dev half freely, locked half EXACTLY ONCE.

    .venv/bin/python scripts/rop_locked_eval.py --set dev      # dry-run, repeatable
    .venv/bin/python scripts/rop_locked_eval.py --set locked   # one shot, refuses to rerun

Implements docs/paper/LOCKED_EVAL_PROTOCOL.md, which was committed before any locked
inference existed. Nothing here is configurable beyond --set and --device on purpose:
every analysis decision was pre-registered, so there is nothing to tune.

The interlock: --set locked refuses to run if results/rop/locked_eval/ already exists.
The dev dry-run writes to results/rop/dev_eval/ and may be repeated while debugging.
"""
from __future__ import annotations

import argparse
import datetime
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
PREP = ROOT / "results" / "prepared" / "rop_staging"
KC = ROOT / "kaggle_cache"
AROP = 5
CLASSES = ["normal", "stage_1", "stage_2", "stage_3", "stage_4_5", "arop"]

# Checkpoints by (arm, seed) -> list of 5 fold weight files, recovered from their kernels.
CKPTS = {
    ("structured", 42): [KC / "struct42" / "code" / "results" /
                         f"experiment_00{k+1}_structured_effnetv2s_fold{k}_s42" / "weights.pth"
                         for k in range(5)],
    ("structured", 1337): [KC / "struct1337" / "code" / "results" /
                           f"experiment_00{k+1}_structured_effnetv2s_fold{k}_s1337" / "weights.pth"
                           for k in range(5)],
    ("flat", 42): [KC / ("nat_a" if k < 3 else "nat_b") / "code" / "results" /
                   f"experiment_00{(k % 3) + 1}_staging_effnetv2s_fold{k}_s42_natural" / "weights.pth"
                   for k in range(5)],
    ("flat", 1337): [KC / ("nat1337_a" if k < 3 else "nat1337_b") / "code" / "results" /
                     f"experiment_00{(k % 3) + 1}_staging_effnetv2s_fold{k}_s1337_natural" / "weights.pth"
                     for k in range(5)],
}


def load_model(arm: str, ckpt: Path, device: str):
    import torch
    from models.common.architectures import build_model
    from models.common.structured import StructuredROPModel
    state = torch.load(ckpt, map_location="cpu")
    state = state.get("model", state)
    if arm == "structured":
        n_sites = state["site_head.3.bias"].shape[0]
        m = StructuredROPModel("effnetv2s", n_sites=n_sites, pretrained=False)
        m.load_state_dict(state, strict=True)
    else:
        m = build_model("effnetv2s", 6, pretrained=False)
        m.load_state_dict(state, strict=True)
    return m.to(device).eval()


def infer(arm: str, model, loader, device: str):
    import torch
    import torch.nn.functional as F
    from models.common.structured import decode_6class
    preds, probs = [], []
    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device)
            if arm == "structured":
                out = model(x)
                p6, prob6 = decode_6class(out, arop_threshold=0.5, variant="full")
            else:
                logits = model(x)
                prob6 = F.softmax(logits, dim=1)
                p6 = logits.argmax(dim=1)
            preds.append(p6.cpu().numpy())
            probs.append(prob6.float().cpu().numpy())
    return np.concatenate(preds), np.concatenate(probs)


def metrics(y, p, min_n=30):
    from sklearn.metrics import cohen_kappa_score, f1_score
    present = sorted(set(y.tolist()))
    counts = {c: int((y == c).sum()) for c in present}
    big = [c for c in present if counts[c] >= min_n]
    return {
        "qwk": round(float(cohen_kappa_score(y, p, weights="quadratic")), 4),
        "accuracy": round(float((y == p).mean()), 4),
        "macro_f1_present": round(float(f1_score(y, p, average="macro", labels=present,
                                                 zero_division=0)), 4),
        "macro_f1_6label": round(float(f1_score(y, p, average="macro",
                                                labels=list(range(6)), zero_division=0)), 4),
        f"macro_f1_classes_ge{min_n}": round(float(f1_score(y, p, average="macro",
                                                            labels=big, zero_division=0)), 4),
        "per_class_recall": {CLASSES[c]: {"recall": round(float((p[y == c] == c).mean()), 4),
                                          "n": counts[c]} for c in present},
    }


def eye_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Join the dataset's own Acquisition View Table -> (view, eye, region) per image.
    Protocol Amendment 1: Multi-View labels are per-EYE (217/230 dev eyes carry one label
    across all views), so the eye is a co-primary evaluation unit."""
    av = pd.read_excel(ROOT / "data/rop/multiview/ROP-DATASET/Dataset Acquisition View Table.xlsx")
    vc = ["Left temporal", "Left posterior pole", "Left nasal",
          "Right nasal", "Right posterior pole", "Right temporal"]
    av["view"] = av[vc].notna().idxmax(axis=1)
    av["eye"] = av["view"].str.split().str[0]
    av["region"] = av["view"].str.split(n=1).str[1]
    out = df.copy()
    out["Image_name"] = out.image_path.str.rsplit("/", n=1).str[1].str.replace(
        ".jpg", "", regex=False)
    out = out.merge(av[["Image_name", "eye", "region"]], on="Image_name", how="left")
    assert out.eye.notna().all(), "view-table join incomplete - refusing to aggregate eyes"
    return out


def eye_metrics(edf: pd.DataFrame, pred: np.ndarray):
    """Co-primary unit: eye = (patient, eye); label and prediction = max over views."""
    from sklearn.metrics import cohen_kappa_score, f1_score
    g = edf.assign(pred=pred).groupby(["patient_id", "eye"]).agg(
        y=("label", "max"), p=("pred", "max"))
    y, p = g.y.values, g.p.values
    present = sorted(set(y.tolist()))
    dis = g[g.y.isin([1, 2, 3])]
    return {
        "n_eyes": int(len(g)),
        "qwk": round(float(cohen_kappa_score(y, p, weights="quadratic")), 4),
        "accuracy": round(float((y == p).mean()), 4),
        "macro_f1_present": round(float(f1_score(y, p, average="macro", labels=present,
                                                 zero_division=0)), 4),
        "stage123_eyes_called_normal": (round(float((dis.p == 0).mean()), 4)
                                        if len(dis) else None),
        "n_stage123_eyes": int(len(dis)),
    }


def view_miss_table(edf: pd.DataFrame, pred: np.ndarray):
    """Mechanism check: stage 1-3 images predicted normal, by acquisition region."""
    d = edf.assign(pred=pred)
    d = d[d.label.isin([1, 2, 3])]
    return {r: {"n": int(len(sub)), "called_normal": round(float((sub.pred == 0).mean()), 4)}
            for r, sub in d.groupby("region")}


def cluster_ci(y, ps, groups, fn, n=2000, seed=42):
    """Patient-clustered CI of mean-over-checkpoints of fn on each replicate."""
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups)
    idx = {g: np.where(groups == g)[0] for g in uniq}
    vals = []
    for _ in range(n):
        take = np.concatenate([idx[g] for g in rng.choice(uniq, len(uniq), True)])
        vals.append(float(np.mean([fn(y[take], p[take]) for p in ps])))
    return [round(float(x), 4) for x in np.percentile(vals, [2.5, 97.5])]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", dest="which", required=True, choices=["dev", "locked"])
    ap.add_argument("--device", default=None)
    ap.add_argument("--aggregate-only", action="store_true",
                    help="rebuild the table from already-saved npz (no inference)")
    a = ap.parse_args()

    outdir = ROOT / "results" / "rop" / ("locked_eval" if a.which == "locked" else "dev_eval")
    if a.which == "locked" and outdir.exists() and not a.aggregate_only:
        raise SystemExit("REFUSING: results/rop/locked_eval/ already exists. The locked half "
                         "is scored exactly once; there is no second shot. If this run "
                         "crashed mid-way, the partial outputs are evidence of the opening "
                         "and stay — analyse what was written.")
    missing = [str(c) for cs in CKPTS.values() for c in cs if not c.exists()]
    if missing:
        raise SystemExit("missing checkpoints — refusing to open with a partial 2x2:\n  "
                         + "\n  ".join(missing[:8]))

    import torch
    from torch.utils.data import DataLoader
    from models.common.config import load_config
    from models.common.datasets import FundusDataset
    from models.common.preprocessing import build_transforms
    device = a.device or ("mps" if torch.backends.mps.is_available()
                          else "cuda" if torch.cuda.is_available() else "cpu")

    df = pd.read_csv(PREP / f"test_{a.which}.csv").reset_index(drop=True)
    y = df.label.values
    groups = df.patient_id.values
    print(f"{a.which}: {len(df)} images, {df.patient_id.nunique()} infants, device={device}")
    print(f"  class counts: { {CLASSES[c]: int((y == c).sum()) for c in sorted(set(y.tolist()))} }")

    cfg = load_config(ROOT / "configs" / "rop_staging.yaml")
    ds = FundusDataset(df, split=None, transform=build_transforms(cfg, split="val"),
                       image_size=int(cfg.preprocess.image_size))
    loader = DataLoader(ds, batch_size=32, num_workers=2)

    outdir.mkdir(parents=True, exist_ok=True)
    from sklearn.metrics import cohen_kappa_score, f1_score
    qwk_fn = lambda yy, pp: cohen_kappa_score(yy, pp, weights="quadratic")          # noqa: E731
    mf1_fn = lambda yy, pp: f1_score(yy, pp, average="macro",                        # noqa: E731
                                     labels=sorted(set(yy.tolist())), zero_division=0)

    table = {"protocol": "docs/paper/LOCKED_EVAL_PROTOCOL.md (committed before this ran)",
             "set": a.which, "n_images": int(len(df)),
             "n_infants": int(df.patient_id.nunique()),
             "opened_at": datetime.datetime.now().isoformat(timespec="seconds"),
             "device": device, "cells": {}}

    edf = eye_frame(df)
    for (arm, seed), ckpts in CKPTS.items():
        per_ckpt_preds, per_ckpt_probs, per_ckpt_metrics = [], [], []
        for k, c in enumerate(ckpts):
            f = outdir / f"{arm}_s{seed}_ckpt{k}.npz"
            if a.aggregate_only:
                z = np.load(f)
                assert (z["y_true"] == y).all()
                p, pr = z["y_pred"], z["prob6"]
            else:
                model = load_model(arm, c, device)
                p, pr = infer(arm, model, loader, device)
                del model
                np.savez_compressed(f, y_true=y, y_pred=p, prob6=pr)
            per_ckpt_preds.append(p)
            per_ckpt_probs.append(pr)
            per_ckpt_metrics.append(metrics(y, p))
            print(f"  {arm} s{seed} ckpt{k}: QWK {per_ckpt_metrics[-1]['qwk']:.4f} "
                  f"macroF1(present) {per_ckpt_metrics[-1]['macro_f1_present']:.4f}")

        # PRIMARY: mean +/- sd over the 5 checkpoints.
        def ms(key):
            v = [m[key] for m in per_ckpt_metrics]
            return {"mean": round(float(np.mean(v)), 4), "sd": round(float(np.std(v, ddof=1)), 4)}
        # SECONDARY: probability-mean ensemble.
        ens_prob = np.mean(per_ckpt_probs, axis=0)
        ens_pred = ens_prob.argmax(1)
        eye_per_ckpt = [eye_metrics(edf, p) for p in per_ckpt_preds]
        def ems(key):
            v = [m[key] for m in eye_per_ckpt if m[key] is not None]
            return {"mean": round(float(np.mean(v)), 4), "sd": round(float(np.std(v, ddof=1)), 4)}
        cell = {
            "primary_per_checkpoint": {k: ms(k) for k in
                                       ("qwk", "accuracy", "macro_f1_present",
                                        "macro_f1_6label", "macro_f1_classes_ge30")},
            "coprimary_eye_level": {k: ems(k) for k in
                                    ("qwk", "accuracy", "macro_f1_present",
                                     "stage123_eyes_called_normal")},
            "n_eyes": eye_per_ckpt[0]["n_eyes"],
            "n_stage123_eyes": eye_per_ckpt[0]["n_stage123_eyes"],
            "view_miss_table_ckpt_mean": {
                r: {"n": view_miss_table(edf, per_ckpt_preds[0])[r]["n"],
                    "called_normal_mean": round(float(np.mean(
                        [view_miss_table(edf, p)[r]["called_normal"]
                         for p in per_ckpt_preds])), 4)}
                for r in view_miss_table(edf, per_ckpt_preds[0])},
            "primary_qwk_ci95_clustered": cluster_ci(y, per_ckpt_preds, groups, qwk_fn),
            "primary_macro_f1_present_ci95_clustered": cluster_ci(y, per_ckpt_preds, groups, mf1_fn),
            "secondary_ensemble": metrics(y, ens_pred),
            "per_checkpoint_detail": per_ckpt_metrics,
            "unvalidated_note": "stage_4_5 and arop cells are NOT externally validated at "
                                "these n regardless of their values (pre-registered).",
        }
        table["cells"][f"{arm}_s{seed}"] = cell
        print(f"     eye-level: QWK {cell['coprimary_eye_level']['qwk']['mean']:.4f} "
              f"stage1-3 eyes called normal {cell['coprimary_eye_level']['stage123_eyes_called_normal']['mean']:.4f}")
        print(f"  == {arm} s{seed}: QWK {cell['primary_per_checkpoint']['qwk']['mean']:.4f}"
              f" +/- {cell['primary_per_checkpoint']['qwk']['sd']:.4f}"
              f"  CI {cell['primary_qwk_ci95_clustered']}"
              f"  | macroF1(present) {cell['primary_per_checkpoint']['macro_f1_present']['mean']:.4f}"
              f"  | ensemble QWK {cell['secondary_ensemble']['qwk']:.4f}")

    out = outdir / (f"{a.which}_table.json")
    out.write_text(json.dumps(table, indent=2) + "\n")
    print(f"-> {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
