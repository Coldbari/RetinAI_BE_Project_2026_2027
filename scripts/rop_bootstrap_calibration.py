"""Patient-clustered bootstrap CIs and calibration for the structured 5-fold CV run.

    .venv/bin/python scripts/rop_bootstrap_calibration.py

Reads the committed raw predictions (results/rop/staging_runs/structured_5fold/fold_*.npz)
and the fold manifests, joins them row-by-row (verified: val loaders are unshuffled, and
the y_true vector must match the manifest's label column exactly or this script refuses),
and reports:

  BOOTSTRAP.  Percentile CIs for macro-F1, QWK, and AP-ROP F1/recall, resampling PATIENTS
  with replacement, not images. Frames from one infant are highly correlated (median ~30
  frames each); an image-level bootstrap would treat them as independent and produce
  intervals roughly sqrt(cluster size) too narrow — flattering and wrong.

  CALIBRATION.  Top-label ECE (15 equal-width bins), Brier score, and the reliability-bin
  table (for figures later). Almost nothing in the ROP literature reports calibration; a
  screening model's probabilities matter as much as its argmax.

Point estimates are computed per fold and on the POOLED CV predictions (each image appears
in exactly one fold's val split, so pooling gives one prediction per image). The bootstrap
runs on the pooled set — per-fold CIs on ~30 patients are too wide to say anything and
would invite over-reading.

Output: results/rop/structured_bootstrap_calibration.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, f1_score, recall_score

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "results" / "rop" / "staging_runs" / "structured_5fold"
PREP = ROOT / "results" / "prepared" / "rop_staging"
OUT = ROOT / "results" / "rop" / "structured_bootstrap_calibration.json"

AROP = 5
LABELS = list(range(6))
B = 2000
SEED = 42
N_BINS = 15


def load_folds():
    frames = []
    for f in range(5):
        d = np.load(RUNS / f"fold_{f}.npz")
        va = pd.read_csv(PREP / f"fold_{f}.csv").query("split == 'val'").reset_index(drop=True)
        if len(va) != len(d["y_true"]) or not (va.label.values == d["y_true"]).all():
            raise SystemExit(f"fold {f}: npz rows do not align with the manifest's val rows "
                             f"— the patient join would be garbage, refusing")
        frames.append(pd.DataFrame({
            "fold": f, "patient_id": va.patient_id, "source": va.source,
            "y_true": d["y_true"], "y_pred": d["y_pred"],
            "conf": d["prob6"].max(axis=1),
            "conf_correct": (d["prob6"].argmax(axis=1) == d["y_true"]).astype(float),
            "brier": ((d["prob6"] - np.eye(6)[d["y_true"]]) ** 2).sum(axis=1),
        }))
    return pd.concat(frames, ignore_index=True)


def metric_block(yt, yp):
    return {
        "macro_f1": float(f1_score(yt, yp, labels=LABELS, average="macro",
                                   zero_division=0)),
        "qwk": float(cohen_kappa_score(yt, yp, labels=LABELS, weights="quadratic")),
        "arop_f1": float(f1_score(yt == AROP, yp == AROP, zero_division=0)),
        "arop_recall": float(recall_score(yt == AROP, yp == AROP, zero_division=0)),
    }


def patient_bootstrap(df, rng):
    """One replicate: resample patients with replacement, keep all their frames."""
    patients = df.patient_id.unique()
    groups = {p: g for p, g in df.groupby("patient_id")}
    picked = rng.choice(patients, size=len(patients), replace=True)
    rep = pd.concat([groups[p] for p in picked], ignore_index=True)
    return metric_block(rep.y_true.values, rep.y_pred.values)


def calibration(df):
    """Top-label ECE over equal-width confidence bins, plus the bin table."""
    edges = np.linspace(0.0, 1.0, N_BINS + 1)
    idx = np.clip(np.digitize(df.conf.values, edges) - 1, 0, N_BINS - 1)
    bins, ece = [], 0.0
    for b in range(N_BINS):
        m = idx == b
        n = int(m.sum())
        if n == 0:
            continue
        conf, acc = float(df.conf.values[m].mean()), float(df.conf_correct.values[m].mean())
        ece += (n / len(df)) * abs(acc - conf)
        bins.append({"bin_low": float(edges[b]), "bin_high": float(edges[b + 1]),
                     "n": n, "mean_confidence": round(conf, 4), "accuracy": round(acc, 4)})
    return {"ece": round(float(ece), 4), "brier": round(float(df.brier.mean()), 4),
            "n_bins": N_BINS, "bins": bins}


def main():
    df = load_folds()
    n_pat = df.patient_id.nunique()
    print(f"pooled CV predictions: {len(df)} images, {n_pat} patients")

    rng = np.random.default_rng(SEED)
    reps = [patient_bootstrap(df, rng) for _ in range(B)]
    ci = {}
    for k in reps[0]:
        v = np.array([r[k] for r in reps])
        ci[k] = {"point": round(metric_block(df.y_true.values, df.y_pred.values)[k], 4),
                 "ci_low": round(float(np.percentile(v, 2.5)), 4),
                 "ci_high": round(float(np.percentile(v, 97.5)), 4)}
        print(f"  {k:<12} {ci[k]['point']:.3f}  [{ci[k]['ci_low']:.3f}, "
              f"{ci[k]['ci_high']:.3f}]  (95% patient-bootstrap CI)")

    out = {
        "run": "structured effnetv2s seed 42, 5-fold CV pooled val predictions",
        "bootstrap": {"replicates": B, "seed": SEED, "unit": "patient",
                      "n_images": int(len(df)), "n_patients": int(n_pat),
                      "why_patient_unit": "frames within an infant are correlated; an "
                                          "image-level bootstrap understates the intervals",
                      "metrics": ci},
        "calibration_pooled": calibration(df),
        "per_fold": {},
    }
    for f, g in df.groupby("fold"):
        blk = metric_block(g.y_true.values, g.y_pred.values)
        out["per_fold"][f"fold_{f}"] = {
            **{k: round(v, 4) for k, v in blk.items()},
            "ece": calibration(g)["ece"], "n_images": int(len(g)),
            "n_patients": int(g.patient_id.nunique()),
        }
        print(f"  fold_{f}: ECE {out['per_fold'][f'fold_{f}']['ece']:.3f}  "
              f"({g.patient_id.nunique()} patients)")
    print(f"  pooled ECE {out['calibration_pooled']['ece']:.3f}  "
          f"Brier {out['calibration_pooled']['brier']:.3f}")

    OUT.write_text(json.dumps(out, indent=2) + "\n")
    print(f"-> {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
