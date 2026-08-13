"""Automatic error analysis (W9) — the viva's best friend.

From (y_true, y_pred, y_prob, image_paths) it produces:
  - false positives / false negatives (binary)
  - most-confused class pairs + per-class error stats
  - four confidence quadrants: high/low-confidence correct/incorrect

Offending images are copied under ``<outdir>/errors/<bucket>/`` and a JSON summary
is written. The high-confidence-wrong / low-confidence-correct exemplars also feed the
explainability gallery.
"""
from __future__ import annotations

import json
import shutil
from collections import Counter
from pathlib import Path

import numpy as np


def _save(images, dst_dir, k=24):
    dst_dir.mkdir(parents=True, exist_ok=True)
    for src in images[:k]:
        try:
            shutil.copy2(src, dst_dir / Path(src).name)
        except Exception:
            pass


def run_error_analysis(y_true, y_pred, y_prob, image_paths, outdir,
                       num_classes, class_names=None, top_k=24):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    image_paths = np.asarray(image_paths, dtype=object)
    class_names = class_names or [str(i) for i in range(num_classes)]
    conf = np.asarray(y_prob).max(axis=1) if y_prob is not None else np.ones(len(y_true))

    errors_dir = Path(outdir) / "errors"
    correct = y_true == y_pred
    wrong = ~correct

    # per-class error stats
    per_class = {}
    for c in range(num_classes):
        mask = y_true == c
        n = int(mask.sum())
        per_class[class_names[c]] = {
            "support": n,
            "errors": int((mask & wrong).sum()),
            "error_rate": float((mask & wrong).sum() / n) if n else 0.0,
        }

    # most-confused ordered pairs (true -> pred)
    confused = Counter()
    for t, p in zip(y_true[wrong], y_pred[wrong]):
        confused[(int(t), int(p))] += 1
    most_confused = [
        {"true": class_names[t], "pred": class_names[p], "count": c}
        for (t, p), c in confused.most_common(10)
    ]

    # confidence quadrants
    order_wrong = np.argsort(-conf)  # high conf first
    order_correct = np.argsort(conf)  # low conf first
    hi_wrong = [image_paths[i] for i in order_wrong if wrong[i]]
    lo_correct = [image_paths[i] for i in order_correct if correct[i]]
    _save(hi_wrong, errors_dir / "high_confidence_wrong", top_k)
    _save(lo_correct, errors_dir / "low_confidence_correct", top_k)
    _save([image_paths[i] for i in np.argsort(conf) if wrong[i]],
          errors_dir / "low_confidence_wrong", top_k)
    _save([image_paths[i] for i in np.argsort(-conf) if correct[i]],
          errors_dir / "high_confidence_correct", top_k)

    summary = {
        "n": int(len(y_true)),
        "accuracy": float(correct.mean()),
        "per_class": per_class,
        "most_confused": most_confused,
    }

    if num_classes == 2:
        fp = [image_paths[i] for i in range(len(y_true))
              if y_pred[i] == 1 and y_true[i] == 0]
        fn = [image_paths[i] for i in range(len(y_true))
              if y_pred[i] == 0 and y_true[i] == 1]
        _save(fp, errors_dir / "false_positive", top_k)
        _save(fn, errors_dir / "false_negative", top_k)
        summary["false_positives"] = len(fp)
        summary["false_negatives"] = len(fn)

    errors_dir.mkdir(parents=True, exist_ok=True)
    with open(errors_dir / "error_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"[error_analysis] acc {summary['accuracy']*100:.2f}%  "
          f"top-confused {most_confused[:3]}  -> {errors_dir}")
    return summary
