"""Patient-clustered bootstrap CIs for the DANN dose-response probe cells.

    .venv/bin/python scripts/rop_dann_probe_ci.py \
        --ckpt-w0 <wsite0 weights.pth> --ckpt-w05 <full weights.pth> --ckpt-w2 <wsite2 weights.pth>

The draft's §4.4 claims the disease-controlled site probe RISES monotonically with adversary
strength (0.867 -> 0.885 -> 0.921). Three point estimates cannot carry "monotonically" on
their own: with ~29 normal-class patients per site cell, a 0.018 step could be noise. This
script reruns the exact G5 probe (patient-grouped 5-fold CV, StandardScaler + balanced
logistic regression on penultimate embeddings, fold_0 val) for each w_site cell, keeps the
out-of-fold predictions, and bootstrap-resamples PATIENTS to get a CI per cell and — the
part that actually tests monotonicity — a paired CI on each between-cell difference, using
the same patient resample on both arms.

Honesty box, stated here and in the output: the OOF predictions are held FIXED across
replicates, so the CI covers sampling variability of the score given the fitted probes, not
probe-refit variability. Refitting 5 logistic regressions per replicate x 2000 replicates is
not a price this analysis can pay; the fixed-OOF cluster bootstrap is the standard
compromise and is stated as such wherever these intervals are quoted.

Output: results/rop/dann_probe_ci.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
OUT = ROOT / "results" / "rop" / "dann_probe_ci.json"

# The committed point estimates these runs must reproduce (results/rop/dann_probe_ablation.json).
# A checkpoint that lands far from its committed cell is the WRONG checkpoint, not "drift".
COMMITTED = {0.0: {"naive": 0.788, "controlled": 0.867},
             0.5: {"naive": 0.820, "controlled": 0.885},
             2.0: {"naive": 0.868, "controlled": 0.921}}
TOL = 0.02   # embeddings recomputed on MPS rather than CPU wobble in the 3rd decimal

# The w=0.5 cell DOES exceed that tolerance (controlled 0.852 vs committed 0.885), and the
# explanation was chased down rather than shrugged off: the checkpoint probed here is the
# canonical fold-0 checkpoint of the 5-fold structured kernel — byte-identical (sha256
# 11b43960…) to the HF-served webapp weights — while the committed 0.885 was measured on an
# EARLIER same-config training run whose Kaggle output a later kernel version overwrote.
# That 0.033 sibling-run spread is itself a datum: training-run variability in the
# controlled probe is the same size as the w=0 -> w=0.5 step, which is exactly why that
# step's CI spans zero and why "monotonic" cannot be claimed from three point estimates.
W05_PROVENANCE = (
    "w0.5 probed on the canonical 5-fold-kernel fold-0 checkpoint (= HF-served weights, "
    "sha256 11b43960…). The committed 0.820/0.885 cell came from an earlier same-config "
    "training run, since overwritten on Kaggle and unrecoverable. The 0.033 spread between "
    "sibling runs measures training-run variability in the controlled probe — the same "
    "size as the w=0->0.5 step, which is why that step is not treated as signal.")


def embed_val(ckpt: Path, arch: str, df: pd.DataFrame, device: str):
    import torch
    from torch.utils.data import DataLoader
    from models.common.config import load_config
    from models.common.datasets import FundusDataset
    from models.common.preprocessing import build_transforms
    from models.common.structured import StructuredROPModel

    cfg = load_config(ROOT / "configs" / "rop_staging.yaml")
    state = torch.load(ckpt, map_location="cpu")
    state = state.get("model", state)
    n_sites = state["site_head.3.bias"].shape[0]
    wrapper = StructuredROPModel(arch, n_sites=n_sites, pretrained=False)
    wrapper.load_state_dict(state, strict=True)
    net = wrapper.backbone.to(device).eval()

    ds = FundusDataset(df, split=None, transform=build_transforms(cfg, split="val"),
                       image_size=int(cfg.preprocess.image_size))
    feats = []
    with torch.no_grad():
        for x, _ in DataLoader(ds, batch_size=32, num_workers=2):
            feats.append(net(x.to(device)).flatten(1).float().cpu().numpy())
    return np.concatenate(feats)


def probe_oof(X, src, groups):
    """G5's probe verbatim, but returning the out-of-fold predictions, not just the score."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    from sklearn.preprocessing import StandardScaler
    pred = np.empty_like(src)
    for tr_i, te_i in GroupKFold(n_splits=5).split(X, src, groups=groups):
        sc = StandardScaler().fit(X[tr_i])
        clf = LogisticRegression(max_iter=3000, class_weight="balanced")
        clf.fit(sc.transform(X[tr_i]), src[tr_i])
        pred[te_i] = clf.predict(sc.transform(X[te_i]))
    return pred


def cluster_boot(src, groups, preds_by_arm, n=2000, seed=42):
    """One patient resample per replicate, scored across ALL arms — that pairing is what
    lets the difference CIs subtract out shared patient-composition noise."""
    from sklearn.metrics import balanced_accuracy_score
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups)
    idx_by_g = {g: np.where(groups == g)[0] for g in uniq}
    n_sites = len(set(src))
    scores = {arm: [] for arm in preds_by_arm}
    skipped = 0
    for _ in range(n):
        take = np.concatenate([idx_by_g[g] for g in rng.choice(uniq, len(uniq), True)])
        if len(set(src[take])) < n_sites:   # a replicate missing a site has no comparable bacc
            skipped += 1
            continue
        for arm, pred in preds_by_arm.items():
            scores[arm].append(balanced_accuracy_score(src[take], pred[take]))
    return {arm: np.array(v) for arm, v in scores.items()}, skipped


def ci(v):
    lo, hi = np.percentile(v, [2.5, 97.5])
    return [round(float(lo), 4), round(float(hi), 4)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-w0", required=True, type=Path)
    ap.add_argument("--ckpt-w05", required=True, type=Path)
    ap.add_argument("--ckpt-w2", required=True, type=Path)
    ap.add_argument("--arch", default="effnetv2s")
    ap.add_argument("--device", default=None)
    a = ap.parse_args()

    import torch
    device = a.device or ("mps" if torch.backends.mps.is_available()
                          else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    df = pd.read_csv(ROOT / "results/prepared/rop_staging/fold_0.csv")
    df = df[df.split == "val"].reset_index(drop=True)
    src, groups = df.source.values, df.patient_id.values
    norm = (df.cls6.values == "normal")
    print(f"fold_0 val: {len(df)} images, {len(set(groups))} patients; "
          f"controlled subset: {int(norm.sum())} normal images, "
          f"{len(set(groups[norm]))} patients")

    ckpts = {0.0: a.ckpt_w0, 0.5: a.ckpt_w05, 2.0: a.ckpt_w2}
    oof = {}      # (w, 'naive'|'controlled') -> OOF predictions over the relevant rows
    cells = {}
    for w, ckpt in ckpts.items():
        print(f"w_site={w}: embedding {ckpt.name} ...")
        X = embed_val(ckpt, a.arch, df, device)
        p_all = probe_oof(X, src, groups)
        p_norm = probe_oof(X[norm], src[norm], groups[norm])
        from sklearn.metrics import balanced_accuracy_score
        pt_all = balanced_accuracy_score(src, p_all)
        pt_norm = balanced_accuracy_score(src[norm], p_norm)
        oof[(w, "naive")], oof[(w, "controlled")] = p_all, p_norm
        for kind, pt in (("naive", pt_all), ("controlled", pt_norm)):
            drift = abs(pt - COMMITTED[w][kind])
            flag = "" if drift <= TOL else "  ** exceeds tolerance — wrong checkpoint? **"
            print(f"    {kind:>10}: {pt:.3f}  (committed {COMMITTED[w][kind]:.3f}, "
                  f"drift {drift:.3f}){flag}")
        cells[w] = {"checkpoint": ckpt.name,
                    "checkpoint_sha256": hashlib.sha256(ckpt.read_bytes()).hexdigest()[:16],
                    "naive": {"point": round(float(pt_all), 4)},
                    "controlled": {"point": round(float(pt_norm), 4)},
                    "committed_point": COMMITTED[w]}

    ws = [0.0, 0.5, 2.0]
    out = {"what": "Patient-clustered bootstrap CIs for the G5 site-probe dose-response "
                   "cells (fold_0 val, effnetv2s, seed 42). 2000 replicates resampling "
                   "patients; the same resample scores every arm, so difference CIs are "
                   "paired. OOF probe predictions are held fixed across replicates: the CI "
                   "covers patient-sampling variability given the fitted probes, not "
                   "probe-refit variability.",
           "w05_provenance": W05_PROVENANCE,
           "cells": {}, "paired_diffs": {}}

    for kind, mask in (("naive", None), ("controlled", norm)):
        s = src if mask is None else src[mask]
        g = groups if mask is None else groups[mask]
        boot, skipped = cluster_boot(s, g, {w: oof[(w, kind)] for w in ws})
        for w in ws:
            cells[w][kind]["ci95"] = ci(boot[w])
        diffs = {}
        for wa, wb in [(0.5, 0.0), (2.0, 0.5), (2.0, 0.0)]:
            d = boot[wa] - boot[wb]
            lo, hi = ci(d)
            diffs[f"w{wa}_minus_w{wb}"] = {
                "point": round(float(cells[wa][kind]["point"] - cells[wb][kind]["point"]), 4),
                "ci95": [lo, hi], "excludes_zero": bool(lo > 0 or hi < 0)}
        out["paired_diffs"][kind] = diffs
        out[f"replicates_{kind}"] = {"used": int(len(boot[ws[0]])), "skipped_missing_site": skipped}
        print(f"\n  {kind} probe ({int(len(boot[ws[0]]))} usable replicates):")
        for w in ws:
            print(f"    w={w:<4} {cells[w][kind]['point']:.3f}  CI {cells[w][kind]['ci95']}")
        for name, d in diffs.items():
            print(f"    {name}: {d['point']:+.3f}  CI [{d['ci95'][0]:+.3f}, {d['ci95'][1]:+.3f}]"
                  f"  {'EXCLUDES 0' if d['excludes_zero'] else 'spans 0'}")

    out["cells"] = {f"w{w}": cells[w] for w in ws}
    OUT.write_text(json.dumps(out, indent=2) + "\n")
    print(f"\n-> {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
