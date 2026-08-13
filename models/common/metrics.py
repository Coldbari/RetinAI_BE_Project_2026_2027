"""Evaluation metrics — one ``compute_metrics`` used by every disease and the sweep.

Covers Accuracy, macro/weighted Precision/Recall/F1, per-class P/R/F1, AUC-ROC,
Quadratic Weighted Kappa (DR), confusion matrix, and binary sensitivity/specificity.
Also ``best_threshold_for_sensitivity`` to pick a high-recall operating point (ROP).
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (accuracy_score, cohen_kappa_score, confusion_matrix,
                             precision_recall_fscore_support, roc_auc_score)


def quadratic_weighted_kappa(y_true, y_pred) -> float:
    try:
        return float(cohen_kappa_score(y_true, y_pred, weights="quadratic"))
    except Exception:
        return float("nan")


def safe_auc(y_true, y_prob, num_classes: int) -> float:
    """Binary or macro one-vs-rest AUC; NaN if a class is absent."""
    y_true = np.asarray(y_true)
    try:
        if num_classes == 2:
            scores = y_prob[:, 1] if np.ndim(y_prob) == 2 else y_prob
            return float(roc_auc_score(y_true, scores))
        return float(roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro"))
    except Exception:
        return float("nan")


def compute_metrics(y_true, y_pred, y_prob=None, num_classes: int = 2,
                    class_names=None) -> dict:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    class_names = class_names or [str(i) for i in range(num_classes)]

    p, r, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(num_classes)), zero_division=0)
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0)
    w_p, w_r, w_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0)

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(macro_p),
        "macro_recall": float(macro_r),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(w_f1),
        "qwk": quadratic_weighted_kappa(y_true, y_pred),
        "confusion_matrix": confusion_matrix(
            y_true, y_pred, labels=list(range(num_classes))).tolist(),
        "per_class": {
            class_names[i]: {
                "precision": float(p[i]), "recall": float(r[i]),
                "f1": float(f1[i]), "support": int(support[i]),
            } for i in range(num_classes)
        },
    }

    if y_prob is not None:
        metrics["auc"] = safe_auc(y_true, np.asarray(y_prob), num_classes)

    if num_classes == 2:
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        metrics["sensitivity"] = float(tp / (tp + fn)) if (tp + fn) else 0.0
        metrics["specificity"] = float(tn / (tn + fp)) if (tn + fp) else 0.0

    return metrics


def best_threshold_for_sensitivity(y_true, y_score, target_sensitivity=0.90):
    """Lowest threshold on the positive-class score that achieves >= target recall,
    maximising specificity. Returns ``(threshold, sensitivity, specificity)``."""
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    best = (0.5, 0.0, 0.0)
    for thr in np.unique(np.concatenate([[0.0, 1.0], np.sort(y_score)])):
        pred = (y_score >= thr).astype(int)
        tp = int(((pred == 1) & (y_true == 1)).sum())
        fn = int(((pred == 0) & (y_true == 1)).sum())
        tn = int(((pred == 0) & (y_true == 0)).sum())
        fp = int(((pred == 1) & (y_true == 0)).sum())
        sens = tp / (tp + fn) if (tp + fn) else 0.0
        spec = tn / (tn + fp) if (tn + fp) else 0.0
        if sens >= target_sensitivity and spec >= best[2]:
            best = (float(thr), float(sens), float(spec))
    return best


def bootstrap_auc_ci(y_true, y_score, n_boot: int = 2000, alpha: float = 0.05,
                     seed: int = 42):
    """Percentile bootstrap 95% CI for a binary AUC. Returns ``(low, high)``."""
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    if len(np.unique(y_true)) < 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    n = len(y_true)
    aucs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yt = y_true[idx]
        if len(np.unique(yt)) < 2:
            continue
        aucs.append(roc_auc_score(yt, y_score[idx]))
    if not aucs:
        return (float("nan"), float("nan"))
    return (float(np.percentile(aucs, 100 * alpha / 2)),
            float(np.percentile(aucs, 100 * (1 - alpha / 2))))


def referable_metrics(y_true, y_prob, referable_grade: int = 2,
                      target_sensitivity: float = 0.90) -> dict:
    """Collapse an ordinal multi-grade problem (e.g. 5-class DR) to a binary
    *referable* screening decision (grade >= ``referable_grade``) and report the
    clinically actionable view: referable AUC + bootstrap CI, the operating point
    at >= ``target_sensitivity``, and the naive-argmax point. This is the endpoint
    where a >90% screening result legitimately lives (vs the noisy 5-class QWK)."""
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    yt = (y_true >= referable_grade).astype(int)
    p_ref = y_prob[:, referable_grade:].sum(axis=1)        # P(refer) = sum of grade>=k
    auc = safe_auc(yt, p_ref, 2)
    lo, hi = bootstrap_auc_ci(yt, p_ref)
    thr, sens, spec = best_threshold_for_sensitivity(yt, p_ref, target_sensitivity)
    pred = (p_ref >= thr).astype(int)
    # naive-argmax operating point (refer if predicted grade >= referable_grade)
    argmax_pred = (y_prob.argmax(axis=1) >= referable_grade).astype(int)
    tp = int(((argmax_pred == 1) & (yt == 1)).sum())
    fn = int(((argmax_pred == 0) & (yt == 1)).sum())
    tn = int(((argmax_pred == 0) & (yt == 0)).sum())
    fp = int(((argmax_pred == 1) & (yt == 0)).sum())
    return {
        "referable_grade": int(referable_grade),
        "n_referable": int(yt.sum()), "n_total": int(len(yt)),
        "auc": auc, "auc_ci": [lo, hi],
        "target_sensitivity": float(target_sensitivity),
        "op_threshold": thr, "op_sensitivity": sens,
        "op_specificity": spec, "op_accuracy": float((pred == yt).mean()),
        "argmax_sensitivity": float(tp / (tp + fn)) if (tp + fn) else 0.0,
        "argmax_specificity": float(tn / (tn + fp)) if (tn + fp) else 0.0,
    }


def tier(value: float, acceptable: float, target: float, stretch: float,
         higher_is_better: bool = True) -> str:
    """Classify a metric against the plan's Acceptable/Target/Stretch thresholds."""
    if higher_is_better:
        if value >= stretch:
            return "Stretch"
        if value >= target:
            return "Target"
        if value >= acceptable:
            return "Acceptable"
        return "Below"
    # lower is better (e.g. external-validation AUC drop)
    if value <= stretch:
        return "Stretch"
    if value <= target:
        return "Target"
    if value <= acceptable:
        return "Acceptable"
    return "Below"
