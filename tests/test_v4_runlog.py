"""The v4 run log (spec "The run log"): every arm, seed, fold and epoch, generated from data files.

Synthetic inputs only (one arm-B run dir, 2 folds x 3 epochs; 2-class fold files; one duplicate in
fold 0's training rows); every optional input points at a missing file, so the real data never enters.
"""
import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

from scripts.rop_v4_runlog import build_runlog

CLASS_NAMES = ["Normal", "Stage 1", "Stage 2", "Stage 3", "Stage 4/5", "AP-ROP"]


def _fold_csv(path: Path, k: int, train: dict, val: dict) -> list[str]:
    rows, paths = [], []
    for split, counts in (("train", train), ("val", val)):
        for cls6, n in counts.items():
            for i in range(n):
                p = f"data/rop/x/f{k}_{split}_{cls6}_{i}.jpg"
                rows.append(dict(image_path=p, label=0 if cls6 == "normal" else 1, split=split,
                                 patient_id=f"x:{k}{i}", source="ropvl", cls6=cls6))
                paths.append(p)
    pd.DataFrame(rows).to_csv(path, index=False)
    return paths


def _fold_run(d: Path, k: int, epochs: int) -> dict:
    d.mkdir(parents=True)
    hist = {"train_loss": [1.5 - 0.1 * e for e in range(epochs)], "lr": [1e-5 * (e + 1) for e in range(epochs)],
            "val_accuracy": [0.5 + 0.01 * e for e in range(epochs)],
            "val_macro_f1": [0.4 + 0.01 * e for e in range(epochs)],
            "val_macro_precision": [0.45] * epochs, "val_macro_recall": [0.42] * epochs,
            "val_primary": [0.4 + 0.01 * e for e in range(epochs)]}
    (d / "metrics.json").write_text(json.dumps({"history": hist, "best_val_primary": hist["val_primary"][-1],
                                                "val": {"accuracy": 0.52}}))
    cfg = {"data": {"manifest": f"results/prepared/rop_staging/_v3_fold{k}.csv", "class_names": CLASS_NAMES},
           "preprocess": {"image_size": 392},
           "model": {"arch": "dinov2_s"},
           "train": {"epochs": epochs, "batch_size": 2, "lr": 1e-4, "layer_decay": 0.8, "label_smoothing": 0.05,
                     "class_balanced": True, "class_balanced_beta": 0.999}}
    (d / "config.yaml").write_text(yaml.safe_dump(cfg))
    per_class = {c: {"precision": 0.5, "recall": 0.25 * (i % 4), "f1": 0.3, "support": 2}
                 for i, c in enumerate(CLASS_NAMES[:2])}
    return {"macro_f1": 0.42, "accuracy": 0.52, "balanced_accuracy": 0.5, "qwk": 0.3141, "auc": 0.6,
            "per_class": per_class, "kernel": "rop-v3-dinov2-s-s42-lr1e-4", "gpu": "Tesla T4"}


def test_runlog_lists_every_arm_seed_fold_epoch(tmp_path):
    prep, runs = tmp_path / "prep", tmp_path / "runs"
    prep.mkdir()
    f0 = _fold_csv(prep / "fold_0.csv", 0, train={"normal": 5, "stage_1": 3}, val={"normal": 2, "stage_1": 2})
    _fold_csv(prep / "fold_1.csv", 1, train={"normal": 4, "stage_1": 4}, val={"normal": 3, "stage_1": 1})
    dup = f0[0]                                   # a fold-0 training row (normal): dropped at run time
    pd.DataFrame({"image_path": [dup], "md5": ["0"], "group": [0]}).to_csv(prep / "exclude_duplicates.csv",
                                                                           index=False)
    arm = runs / "dinov2_s_s42"
    folds = {f"fold{k}": _fold_run(arm / f"fold_{k}", k, epochs=3) for k in range(2)}
    arm.mkdir(exist_ok=True)
    (arm / "summary.json").write_text(json.dumps({"arch": "dinov2_s", "seed": 42, "lr": "1e-4", "epochs": 3,
                                                  "folds": folds}))
    scores = tmp_path / "scores.json"
    scores.write_text(json.dumps({"arms": {"B": {"status": "present", "seeds_present": [42],
                                                 "pooled_over_seeds": {"n_lists_cv": 1, "cv_image": {
                                                     "qwk_5class": 0.5, "qwk_6label": 0.4,
                                                     "macro_f1_present": 0.3}}}},
                                  "selection_rule": {"table": {}}, "contrasts": {}}))
    missing = tmp_path / "absent"
    out = tmp_path / "out" / "V4_RUN_LOG.md"

    md = build_runlog(runs, prep, scores, out, audit_path=missing / "a.json", pool_summary=missing / "p.json",
                      readme_path=missing / "README.md", splits_summary=missing / "s.json",
                      ledger_path=missing / "progress.md", kaggle_dir=missing / "kaggle")

    assert out.read_text() == md
    # every epoch of every fold, and no more
    assert md.count("| Epoch 3 |") == 2 and "Epoch 4" not in md
    assert md.count("| Epoch 1 |") == 2
    # training rows per class (after the duplicate), validation rows per class, totals
    assert "| training rows | 4 | 3 | 0 | 0 | 0 | 0 | 7 |" in md        # fold 0: 5 + 3 rows, 1 duplicate dropped
    assert "| training rows | 4 | 4 | 0 | 0 | 0 | 0 | 8 |" in md        # fold 1: no duplicate in training
    assert "| validation rows | 2 | 2 | 0 | 0 | 0 | 0 | 4 |" in md
    assert "images per epoch = 7" in md and "images per epoch = 8" in md
    assert "steps per epoch = 4" in md                                  # ceil(7 / 2) = ceil(8 / 2) = 4
    # the final epoch's QWK and per-class recall come from summary.json
    assert "0.3141" in md
    # timing logs absent -> n/a, not invented
    assert "minutes n/a" in md
    # every arm has a section; the absent ones say so
    for lab in "ABCDEFGHI":
        assert f"## Arm {lab}" in md
    section_h = md.split("## Arm H")[1].split("## Arm I")[0]
    assert "not run yet" in section_h
    assert "not run yet" in md.split("## Arm A")[1].split("## Arm B")[0]
    assert "ledger not present" in md


def test_runlog_refuses_test_locked(tmp_path):
    with pytest.raises(ValueError, match="test_locked"):
        build_runlog(tmp_path, tmp_path / "test_locked", tmp_path / "s.json", tmp_path / "o.md")
