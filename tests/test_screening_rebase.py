"""The binary ROP verdict moved off the ResNet50 head. These tests hold the move honest.

WHY IT MOVED. A user screened a healthy retina and got "ROP Detected (48.4%)". The audit
in scripts/rop_screening_external_spec.py found the served head degenerate at the held-out
hospital: at its 0.1933 threshold it flagged 663 of 663 images, every healthy eye included,
because the lowest score anywhere on that site (0.2788) sits ABOVE the cut-off. External
AUC 0.691. The structured staging model scored 0.821 on the same images — the better
detector was already in the app, labelled "research preview".

What is guarded here is not the numbers (those move when models are retrained) but the
RULE, which must keep holding whatever the numbers become:

  * the threshold satisfies the sensitivity floor at BOTH sites, not just the convenient
    one — a threshold that only works at the training hospital is the defect being fixed;
  * it is the LARGEST such threshold, so no specificity is left unclaimed;
  * the structured model applies that exact threshold and names itself (since 1 Nov 2026 the
    app serves the DINOv2 ensemble, whose own operating point is guarded elsewhere).

Run:  .venv/bin/python -m pytest tests/test_screening_rebase.py -v
"""
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

REBASE = ROOT / "results/rop/screening_rebase.json"
pytestmark = pytest.mark.skipif(not REBASE.exists(), reason="re-basing artifact absent")


@pytest.fixture(scope="module")
def art():
    return json.loads(REBASE.read_text())


@pytest.fixture(scope="module")
def scores():
    ys, ss = [], []
    for f in sorted(glob.glob(str(ROOT / "results/rop/staging_runs/structured_5fold/fold_[0-4].npz"))):
        z = np.load(f, allow_pickle=True)
        ys.append((z["y_true"] > 0).astype(int))
        ss.append(z["prob6"][:, 1:].sum(1))
    e = np.load(ROOT / "results/rop/screening_external_spec.npz", allow_pickle=True)
    return (np.concatenate(ys), np.concatenate(ss)), (e["y"], e["staging_fold0"])


def _sens(y, s, t):
    return float((s[y == 1] > t).mean())


def test_threshold_meets_the_sensitivity_floor_at_both_sites(art, scores):
    (yi, si), (ye, se) = scores
    t, floor = art["threshold"], art["sensitivity_floor"]
    assert _sens(yi, si, t) >= floor, "fails at the training hospitals"
    assert _sens(ye, se, t) >= floor, "fails at the held-out hospital — the whole point"


def test_threshold_is_the_largest_that_does_so(art, scores):
    """Otherwise specificity is being given away for nothing."""
    (yi, si), (ye, se) = scores
    t, floor = art["threshold"], art["sensitivity_floor"]
    higher = t + 0.0005
    assert min(_sens(yi, si, higher), _sens(ye, se, higher)) < floor, \
        f"{higher} also clears the floor — the selection was not maximal"


def test_the_new_head_beats_the_one_it_replaced_where_it_matters(art):
    """External discrimination is the reason for the swap; it must actually be better."""
    new, old = art["new_model"], art["retired_model"]
    assert new["auc_external_dev"] > old["auc_external_dev"] + 0.05
    assert old["external_dev"]["specificity"] == pytest.approx(0.0, abs=0.01), \
        "the retired head's degeneracy is the premise of this change"
    assert new["external_dev"]["specificity"] > 0.2, \
        "the replacement must actually clear healthy eyes, not merely score better"


def test_the_selection_caveat_is_recorded(art):
    """The external DEV half chose this threshold, so its numbers here are optimistic."""
    joined = " ".join(art["caveats"]).lower()
    assert "in-sample" in joined or "optimistic" in joined
    assert "locked" in joined, "the locked half's status must be stated"


@pytest.fixture(scope="module")
def legacy():
    """The structured model this threshold belongs to. Since 1 Nov 2026 the app serves the
    DINOv2 ensemble instead (tests/test_webapp_staging_ensemble.py holds ITS threshold); the
    structured model stays loadable for comparison, and this rule must still hold for it."""
    import webapp.inference as I
    st = I.StagingPreview(I._device())
    if not (st.loaded and st.screening):
        pytest.skip("structured weights not available")
    return st


def test_the_legacy_model_applies_exactly_this_threshold_and_says_so(legacy):
    probs = np.zeros(6, dtype=np.float32)
    probs[0] = 1.0                                   # a confident Normal
    f = legacy.screening_finding(probs)
    art = json.loads(REBASE.read_text())
    assert f["decision"]["threshold"] == pytest.approx(art["threshold"] * 100, abs=0.01)
    assert f["prediction"] == "No ROP", "a 0% any-ROP score must not be flagged"
    assert "staging" in f["decision"]["model"], "the UI must name the model that decided"


def test_a_score_just_over_the_line_is_positive_and_just_under_is_not(legacy):
    """The boundary is where a screening tool's behaviour has to be exact."""
    t = legacy.screening["threshold"]
    for delta, expected in ((+0.002, "ROP Detected"), (-0.002, "No ROP")):
        probs = np.zeros(6, dtype=np.float32)
        probs[1] = t + delta                         # all of it on stage 1
        probs[0] = 1.0 - probs[1]
        assert legacy.screening_finding(probs)["prediction"] == expected
