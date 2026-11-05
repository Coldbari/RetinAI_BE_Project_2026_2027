import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.rop_v3_lr_probe import choose, delete_dev_predictions, read_summary, slug  # noqa: E402


def test_default_kept_unless_beaten_by_more_than_margin():
    assert choose({"3e-5": 0.70, "1e-4": 0.75, "2e-4": 0.78}) == "1e-4"      # +0.03 exactly: not more
    assert choose({"3e-5": 0.70, "1e-4": 0.75, "2e-4": 0.7801}) == "2e-4"
    assert choose({"3e-5": 0.79, "1e-4": 0.75, "2e-4": 0.81}) == "2e-4"      # several beat it: best
    assert choose({"3e-5": 0.90, "1e-4": 0.95, "2e-4": 0.60}) == "1e-4"


def test_choose_needs_the_whole_grid():
    with pytest.raises(ValueError):
        choose({"1e-4": 0.75, "2e-4": 0.80})


def _summary(tmp_path, **over):
    s = {"arch": "dinov2_s", "seed": 42, "epochs": 20, "lr": "3e-5", "probe": "inner",
         "gpu": "Tesla T4", "jobs_completed": 1,
         "folds": {"inner": {"macro_f1": 0.7, "qwk": 0.8, "balanced_accuracy": 0.7,
                             "per_class": {"Normal": {"support": 3}, "AP-ROP": {"support": 2}}}}}
    s.update(over)
    p = tmp_path / "summary.json"
    p.write_text(json.dumps(s))
    return p


def test_read_summary_checks_identity(tmp_path):
    r = read_summary(_summary(tmp_path), "dinov2_s", "3e-5")
    assert r["macro_f1"] == 0.7 and r["n_inner"] == 5
    with pytest.raises(ValueError):
        read_summary(_summary(tmp_path, lr="1e-4"), "dinov2_s", "3e-5")
    with pytest.raises(ValueError):
        read_summary(_summary(tmp_path, jobs_completed=0), "dinov2_s", "3e-5")


def test_dev_predictions_deleted_unopened(tmp_path):
    for d in ["a/x", "b"]:
        (tmp_path / d).mkdir(parents=True)
        (tmp_path / d / "dev_predictions.npz").write_bytes(b"not an npz")
    (tmp_path / "b" / "val_predictions.npz").write_bytes(b"keep")
    assert delete_dev_predictions(tmp_path) == 2
    assert not list(tmp_path.rglob("dev_predictions.npz"))
    assert (tmp_path / "b" / "val_predictions.npz").exists()


def test_slug_matches_pusher():
    assert slug("retfound_green", "3e-5") == "rop-v3-retfound-green-s42-probe-lr3e-5"
