"""Shared helpers for the FINAL_MODEL_EVIDENCE_PACKAGE build.

CURRENT MODEL ONLY unless a function name says otherwise. Every metric here is
implemented from scratch against scikit-learn primitives so the package does not
inherit models/common/metrics.py's 6-class QWK defect.
"""
from __future__ import annotations
import json, os, subprocess
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (cohen_kappa_score, confusion_matrix, f1_score,
                             precision_recall_fscore_support, roc_auc_score,
                             average_precision_score, accuracy_score)

ROOT = Path("/Users/praharsh/Project/RetinAI")
PKG  = ROOT / "FINAL_MODEL_EVIDENCE_PACKAGE"
PREP = ROOT / "results/prepared/rop_staging"

CLASSES  = ["normal", "stage_1", "stage_2", "stage_3", "stage_4_5", "arop"]
NICE     = ["Normal", "Stage 1", "Stage 2", "Stage 3", "Stage 4/5", "AP-ROP"]
ORDINAL  = [0, 1, 2, 3, 4]          # AP-ROP (5) is NOT a point on the stage scale
AROP     = 5

# ── CURRENT MODEL ────────────────────────────────────────────────────────────
CUR = dict(
    name="RetinAI ROP Staging — structured EfficientNetV2-S (CORN + AP-ROP branch + DANN)",
    arch="effnetv2s", seed=42, variant="full", epochs=20, sampling="natural",
    cv_dir=ROOT / "results/rop/staging_runs/structured_5fold",
    locked_glob="structured_s42_ckpt{}.npz", dev_glob="structured_s42_ckpt{}.npz",
    config=ROOT / "configs/rop_staging_structured.yaml",
    served=ROOT / "results/rop_staging/weights.pth",
    log=ROOT / "kaggle_cache/struct42/rop-structured-effnetv2s-s42.log",
)
# Comparators (clearly labelled wherever used; never merged into current-model charts)
SEED_B = ROOT / "results/rop/staging_runs/structured_5fold_s1337"
FLAT   = ROOT / "results/rop/staging_runs/flat_natural_5fold"
FLAT_B = ROOT / "results/rop/staging_runs/flat_natural_5fold_s1337"

# ── metric definitions (the corrected ones) ──────────────────────────────────
def qwk_ordinal(y, p):
    """QWK as DOCUMENTED: the 5 ordinal stage classes only (Normal<S1<S2<S3<S4/5).

    AP-ROP is an aggressive FORM, not a rank, so pairs where either the truth or the
    prediction is AP-ROP are excluded rather than being given a quadratic distance
    from Normal. This is the definition configs/rop_staging*.yaml declares in
    `eval.qwk_classes` and that models/common/metrics.py never implemented.
    """
    y, p = np.asarray(y), np.asarray(p)
    m = (y != AROP) & (p != AROP)
    if m.sum() < 2 or len(set(y[m].tolist())) < 2:
        return float("nan"), int(m.sum())
    return float(cohen_kappa_score(y[m], p[m], weights="quadratic")), int(m.sum())

def qwk_6class_legacy(y, p):
    """The metric models/common/metrics.py ACTUALLY computed: AP-ROP as ordinal rank 5.
    Reproduced here only so the correction can be shown side by side. Do not headline."""
    return float(cohen_kappa_score(y, p, weights="quadratic"))

def kappa_unweighted(y, p):
    return float(cohen_kappa_score(y, p))

def full_metrics(y, p, prob=None, labels=range(6)):
    y, p = np.asarray(y), np.asarray(p)
    labels = list(labels)
    pr, rc, f1, sup = precision_recall_fscore_support(y, p, labels=labels, zero_division=0)
    present = [c for c in labels if (y == c).sum() > 0]
    q, qn = qwk_ordinal(y, p)
    out = dict(
        n=int(len(y)), accuracy=float(accuracy_score(y, p)),
        balanced_accuracy=float(np.mean([rc[labels.index(c)] for c in present])),
        macro_f1_all6=float(f1_score(y, p, labels=labels, average="macro", zero_division=0)),
        macro_f1_present=float(f1_score(y, p, labels=present, average="macro", zero_division=0)),
        weighted_f1=float(f1_score(y, p, labels=labels, average="weighted", zero_division=0)),
        macro_precision=float(np.mean(pr)), macro_recall=float(np.mean(rc)),
        qwk_ordinal_5class=q, qwk_ordinal_n_pairs=qn,
        qwk_6class_legacy=qwk_6class_legacy(y, p),
        kappa_unweighted=kappa_unweighted(y, p),
    )
    out["per_class"] = {CLASSES[c]: dict(precision=float(pr[i]), recall=float(rc[i]),
                                         f1=float(f1[i]), support=int(sup[i]))
                        for i, c in enumerate(labels)}
    if prob is not None:
        prob = np.asarray(prob, dtype=float)
        try:
            out["auc_ovr_macro_present"] = float(roc_auc_score(
                y, prob[:, present] / prob[:, present].sum(1, keepdims=True),
                multi_class="ovr", average="macro", labels=present)) if len(present) > 2 else float("nan")
        except Exception:
            out["auc_ovr_macro_present"] = float("nan")
        yb = (y >= 1).astype(int); s = prob[:, 1:].sum(1)
        if len(set(yb.tolist())) == 2:
            out["any_rop_auc"] = float(roc_auc_score(yb, s))
            out["any_rop_ap"] = float(average_precision_score(yb, s))
    return out

# ── uncertainty ──────────────────────────────────────────────────────────────
def cluster_bootstrap(y, p, groups, fn, n=2000, seed=42, prob=None):
    """Patient-CLUSTERED percentile bootstrap: resample patients, not rows.
    Returns (point, lo, hi). Rows inside a patient move together, which is the only
    honest interval when one infant can contribute 470 frames."""
    y, p, groups = np.asarray(y), np.asarray(p), np.asarray(groups)
    uq = np.unique(groups); idx_of = {g: np.where(groups == g)[0] for g in uq}
    rng = np.random.default_rng(seed); vals = []
    point = fn(y, p) if prob is None else fn(y, p, prob)
    for _ in range(n):
        pick = rng.choice(uq, len(uq), replace=True)
        ii = np.concatenate([idx_of[g] for g in pick])
        try:
            v = fn(y[ii], p[ii]) if prob is None else fn(y[ii], p[ii], prob[ii])
            if v == v: vals.append(v)
        except Exception:
            pass
    if not vals: return point, float("nan"), float("nan")
    return float(point), float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))

def ece_bins(conf, correct, nb=15):
    conf, correct = np.asarray(conf, float), np.asarray(correct, float)
    edges = np.linspace(0, 1, nb + 1); rows = []; e = 0.0; n = len(conf)
    for i in range(nb):
        m = (conf > edges[i]) & (conf <= edges[i + 1]) if i else (conf >= 0) & (conf <= edges[1])
        if m.sum() == 0:
            rows.append(dict(lo=edges[i], hi=edges[i+1], n=0, conf=np.nan, acc=np.nan)); continue
        e += m.sum() / n * abs(correct[m].mean() - conf[m].mean())
        rows.append(dict(lo=edges[i], hi=edges[i+1], n=int(m.sum()),
                         conf=float(conf[m].mean()), acc=float(correct[m].mean())))
    return float(e), pd.DataFrame(rows)

def brier_multiclass(y, prob, K=6):
    y = np.asarray(y); oh = np.zeros((len(y), K)); oh[np.arange(len(y)), y] = 1
    return float(np.mean(np.sum((np.asarray(prob, float) - oh) ** 2, axis=1)))

# ── io ───────────────────────────────────────────────────────────────────────
def load_npz(p):
    z = np.load(p)
    return z["y_true"], z["y_pred"], (z["prob6"] if "prob6" in z else z["prob"])

def cv_folds(d=None):
    d = Path(d or CUR["cv_dir"])
    return {f: load_npz(d / f"fold_{f}.npz") for f in range(5)}

def git_hash():
    try: return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT).decode().strip()
    except Exception: return "unknown"

def save_json(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2, default=float))

def save_csv(df, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)

# ── plotting house style ─────────────────────────────────────────────────────
PROV = "Model: structured EfficientNetV2-S (CORN+AP-ROP+DANN), seed 42 · CURRENT MODEL"
plt.rcParams.update({"figure.dpi": 130, "savefig.dpi": 130, "font.size": 9,
                     "axes.titlesize": 10, "axes.grid": True, "grid.alpha": 0.25,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "figure.facecolor": "white", "savefig.bbox": "tight"})
C_OK, C_BAD, C_ALT, C_MUT = "#2a6f97", "#c1440e", "#7a9e7e", "#8d99ae"

def stamp(fig, unit, split, extra=""):
    """Every figure carries evaluation unit, split and model identity."""
    fig.text(0.005, 0.003, f"{PROV} | unit: {unit} | split: {split}{(' | ' + extra) if extra else ''}",
             fontsize=6, color="#555")

def finish(fig, path, unit, split, extra=""):
    stamp(fig, unit, split, extra)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path); plt.close(fig)
    return str(Path(path).relative_to(PKG))
