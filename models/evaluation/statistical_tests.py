"""Statistical validation (W9) — so model-selection claims are evidence, not vibes.

- bootstrap_ci      : 95% CI for any metric (Accuracy / F1 / AUC ...)
- mcnemar_test      : paired comparison of two classifiers' correctness
- delong_test       : significance of the difference between two correlated ROC AUCs

DeLong uses the fast algorithm (Sun & Xu, 2014).
"""
from __future__ import annotations

import numpy as np
from scipy import stats


# ── bootstrap ────────────────────────────────────────────────────────────────
def bootstrap_ci(y_true, y_score, metric_fn, n_boot=2000, alpha=0.05, seed=42):
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    n = len(y_true)
    point = metric_fn(y_true, y_score)
    stats_ = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        try:
            stats_.append(metric_fn(y_true[idx], y_score[idx]))
        except Exception:
            continue
    lo, hi = np.percentile(stats_, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(point), float(lo), float(hi)


# ── cluster bootstrap ────────────────────────────────────────────────────────
def cluster_bootstrap_ci(y_true, y_score, groups, metric_fn, n_boot=2000,
                         alpha=0.05, seed=42):
    """95% CI resampling GROUPS (e.g. patients) with replacement, not rows.

    Required whenever rows are correlated within a cluster. The ROP test split is 1,502
    images from only 29 infants — several photographs of the same eye on the same day.
    Row-level bootstrap treats those as independent and reports an interval far tighter
    than the data supports.

    Returns (point, lo, hi, n_effective_clusters).
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    groups = np.asarray(groups)
    uniq = np.unique(groups)
    idx_by_group = {g: np.flatnonzero(groups == g) for g in uniq}

    rng = np.random.default_rng(seed)
    point = metric_fn(y_true, y_score)
    stats_ = []
    for _ in range(n_boot):
        picked = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([idx_by_group[g] for g in picked])
        if len(np.unique(y_true[idx])) < 2:
            continue
        try:
            stats_.append(metric_fn(y_true[idx], y_score[idx]))
        except Exception:
            continue
    if not stats_:
        return float(point), float("nan"), float("nan"), len(uniq)
    lo, hi = np.percentile(stats_, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(point), float(lo), float(hi), len(uniq)


def clopper_pearson(k, n, alpha=0.05):
    """Exact binomial CI. Use for sensitivity/specificity on small denominators —
    the ROP test split has only 7 positive patients, where normal approximations lie."""
    k, n = int(k), int(n)
    if n == 0:
        return float("nan"), 0.0, 1.0
    lo = 0.0 if k == 0 else float(stats.beta.ppf(alpha / 2, k, n - k + 1))
    hi = 1.0 if k == n else float(stats.beta.ppf(1 - alpha / 2, k + 1, n - k))
    return k / n, lo, hi


# ── McNemar ──────────────────────────────────────────────────────────────────
def mcnemar_test(y_true, pred_a, pred_b):
    """Paired test on where two models disagree. Exact binomial for small discordance,
    else chi-square with continuity correction."""
    y_true = np.asarray(y_true)
    a_ok = np.asarray(pred_a) == y_true
    b_ok = np.asarray(pred_b) == y_true
    b = int(np.sum(a_ok & ~b_ok))   # A right, B wrong
    c = int(np.sum(~a_ok & b_ok))   # A wrong, B right
    n = b + c
    if n == 0:
        return {"b": b, "c": c, "statistic": 0.0, "p_value": 1.0, "test": "none"}
    if n < 25:
        p = float(stats.binomtest(min(b, c), n, 0.5).pvalue)
        return {"b": b, "c": c, "statistic": float(min(b, c)), "p_value": p,
                "test": "exact-binomial"}
    stat = (abs(b - c) - 1) ** 2 / n
    p = float(stats.chi2.sf(stat, 1))
    return {"b": b, "c": c, "statistic": float(stat), "p_value": p, "test": "chi2-cc"}


# ── DeLong (fast) ────────────────────────────────────────────────────────────
def _compute_midrank(x):
    J = np.argsort(x)
    z = x[J]
    N = len(x)
    t = np.zeros(N)
    i = 0
    while i < N:
        j = i
        while j < N and z[j] == z[i]:
            j += 1
        t[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    out = np.empty(N)
    out[J] = t
    return out


def _fast_delong(predictions_sorted_transposed, label_1_count):
    m = label_1_count
    n = predictions_sorted_transposed.shape[1] - m
    pos = predictions_sorted_transposed[:, :m]
    neg = predictions_sorted_transposed[:, m:]
    k = predictions_sorted_transposed.shape[0]
    tx = np.empty([k, m]); ty = np.empty([k, n]); tz = np.empty([k, m + n])
    for r in range(k):
        tx[r] = _compute_midrank(pos[r])
        ty[r] = _compute_midrank(neg[r])
        tz[r] = _compute_midrank(predictions_sorted_transposed[r])
    aucs = tz[:, :m].sum(axis=1) / m / n - (m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    delongcov = sx / m + sy / n
    return aucs, np.atleast_2d(delongcov)


def delong_test(y_true, score_a, score_b):
    """Two-sided p-value for AUC(A) == AUC(B) on the same samples."""
    y_true = np.asarray(y_true)
    order = (-y_true).argsort(kind="mergesort")
    label_1_count = int(y_true.sum())
    preds = np.vstack((np.asarray(score_a), np.asarray(score_b)))[:, order]
    aucs, cov = _fast_delong(preds, label_1_count)
    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    if var <= 0:
        z, p = 0.0, 1.0
    else:
        z = float((aucs[0] - aucs[1]) / np.sqrt(var))
        p = float(2 * stats.norm.sf(abs(z)))
    return {"auc_a": float(aucs[0]), "auc_b": float(aucs[1]), "z": z, "p_value": p}
