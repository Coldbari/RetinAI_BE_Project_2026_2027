"""Confidence calibration (W9).

Reliability diagram, Expected/Maximum Calibration Error, Brier score, and post-hoc
**temperature scaling** (fit on a validation set). Reports calibrated vs uncalibrated
so the thesis can show ECE drops after scaling.

Works on logits (preferred — needed for temperature scaling). A CLI gathers logits
from a trained model; the pure functions below are framework-only and unit-testable.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def softmax_np(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def expected_calibration_error(probs, labels, n_bins=15):
    probs = np.asarray(probs)
    labels = np.asarray(labels)
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == labels).astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    ece, mce = 0.0, 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf > lo) & (conf <= hi)
        if not m.any():
            continue
        gap = abs(correct[m].mean() - conf[m].mean())
        ece += (m.mean()) * gap
        mce = max(mce, gap)
    return float(ece), float(mce)


def brier_score(probs, labels, num_classes):
    probs = np.asarray(probs)
    onehot = np.eye(num_classes)[np.asarray(labels)]
    return float(((probs - onehot) ** 2).sum(axis=1).mean())


def fit_temperature(logits, labels, max_iter=200) -> float:
    """Optimise a single scalar T minimising NLL of logits/T."""
    logits_t = torch.tensor(np.asarray(logits), dtype=torch.float32)
    labels_t = torch.tensor(np.asarray(labels), dtype=torch.long)
    log_T = torch.zeros(1, requires_grad=True)  # T = exp(log_T) > 0
    opt = torch.optim.LBFGS([log_T], lr=0.05, max_iter=max_iter)

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(logits_t / log_T.exp(), labels_t)
        loss.backward()
        return loss

    opt.step(closure)
    return float(log_T.exp().item())


def reliability_diagram(probs, labels, save_path, n_bins=15, title="Reliability"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    probs = np.asarray(probs)
    conf = probs.max(axis=1)
    correct = (probs.argmax(axis=1) == np.asarray(labels)).astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    centres, accs = [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf > lo) & (conf <= hi)
        centres.append((lo + hi) / 2)
        accs.append(correct[m].mean() if m.any() else np.nan)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "--", color="gray", label="perfect")
    ax.bar(centres, accs, width=1.0 / n_bins, alpha=0.7, edgecolor="black",
           color="#60A5FA", label="accuracy")
    ax.set_xlabel("Confidence"); ax.set_ylabel("Accuracy"); ax.set_title(title)
    ax.legend()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(); plt.savefig(save_path, dpi=130); plt.close(fig)


def calibrate_and_report(logits, labels, num_classes, outdir):
    """Full pre/post-temperature-scaling calibration report. Returns a metrics dict."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    logits = np.asarray(logits)

    probs = softmax_np(logits)
    ece0, mce0 = expected_calibration_error(probs, labels)
    brier0 = brier_score(probs, labels, num_classes)
    reliability_diagram(probs, labels, outdir / "reliability_uncalibrated.png",
                        title="Before temperature scaling")

    T = fit_temperature(logits, labels)
    probs_t = softmax_np(logits / T)
    ece1, mce1 = expected_calibration_error(probs_t, labels)
    brier1 = brier_score(probs_t, labels, num_classes)
    reliability_diagram(probs_t, labels, outdir / "reliability_calibrated.png",
                        title=f"After temperature scaling (T={T:.2f})")

    report = {
        "temperature": T,
        "uncalibrated": {"ece": ece0, "mce": mce0, "brier": brier0},
        "calibrated": {"ece": ece1, "mce": mce1, "brier": brier1},
        "ece_reduction": ece0 - ece1,
    }
    print(f"[calibration] T={T:.3f}  ECE {ece0:.4f} -> {ece1:.4f}  "
          f"Brier {brier0:.4f} -> {brier1:.4f}")
    return report
