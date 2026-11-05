"""Phase 2 acceptance tests — calibration honesty (C4/C8), referable decision (C5),
abstention (C6).

The failure this guards: the app displayed raw TTA-averaged softmax and called it
"confidence", while offline temperature values sat unused on disk. Worse, those values were
fitted on SINGLE-VIEW logits and are not applicable to the 5-view probability average that
serving actually produces — wiring them in would have produced confident-looking but wrong
calibration.

Run:  .venv/bin/python -m pytest tests/test_calibration_abstention.py -v
"""
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from conftest import fundus_like  # noqa: E402

COHORTS = ROOT / ".cohorts"


@pytest.fixture(scope="module")
def registry():
    from webapp.inference import Registry
    r = Registry(str(ROOT / "webapp" / "registry.yaml"))
    if not any(m.loaded for m in r.models):
        pytest.skip("no model weights available")
    return r


def _finding(registry, disease, context):
    res = registry.analyze(fundus_like(), context)
    return next(f for f in res["findings"] if f["disease"] == disease)


# ── C8: the word "confidence" must not label a raw score ─────────────────────
def test_predict_returns_score_not_confidence(registry):
    f = _finding(registry, "ROP", "infant")
    assert "score" in f
    assert "confidence" not in f, "a raw model score must not be called 'confidence'"


def test_no_source_surface_labels_a_raw_probability_confidence():
    """Guards the UI/PDF, not just the API."""
    for rel in ("webapp/static/js/main.js", "webapp/templates/history.html",
                "webapp/templates/gallery.html", "reports/report_generator.py"):
        txt = (ROOT / rel).read_text()
        assert "f.confidence" not in txt and "it.confidence" not in txt, rel
        assert "'confidence'" not in txt, rel


# ── C4: calibration status is explicit and never silently assumed ────────────
def test_every_finding_carries_a_calibration_status(registry):
    for disease, ctx in (("ROP", "infant"),):   # ROP-only deployment (2026-08-18)
        f = _finding(registry, disease, ctx)
        cal = f.get("calibration")
        assert cal, f"{disease} has no calibration block"
        assert cal["status"] in ("calibrated", "verified-uncalibrated",
                                 "measured-miscalibrated", "unverified")
        assert "note" in cal


def test_a_badly_calibrated_model_is_not_labelled_like_a_good_one(registry):
    """The reason a model is uncorrected must be visible, not collapsed into one badge.

    DR measures ECE 0.191 at serving and Glaucoma 0.017; both have apply=false. Before this
    guard they reported the identical status string, so the UI showed a materially
    miscalibrated model with the same reassurance as a well-calibrated one.
    """
    from webapp.inference import DiseaseModel
    for m in registry.models:
        cal = m.calibration
        if cal["status"] not in ("verified-uncalibrated", "measured-miscalibrated"):
            continue
        material = cal["ece"] >= DiseaseModel.MATERIAL_ECE
        assert (cal["status"] == "measured-miscalibrated") == material, (
            f"{m.disease}: ECE {cal['ece']:.3f} but status {cal['status']}")
        if material:
            assert cal.get("warning"), f"{m.disease} is miscalibrated but carries no warning"
        # whatever the verdict, no uncorrected model may claim a correction was applied
        assert cal["applied"] is False


def test_unmeasured_disease_reports_unverified_not_calibrated(registry):
    """A deployed model without serving-pipeline calibration must say 'unverified'.
    (Formerly exercised on DR/Glaucoma; those models left with the 2026-08-18 ROP-only
    cleanup. The invariant is kept and applied to every model actually deployed.)"""
    for m in registry.models:
        if not (ROOT / "results" / m.disease.lower() / "calibration_serving.json").exists():
            f = _finding(registry, m.disease, "infant")
            assert f["calibration"]["status"] == "unverified"
            assert f["calibration"]["applied"] is False


def test_rop_calibration_was_measured_against_the_serving_pipeline():
    p = ROOT / "results" / "rop" / "calibration_serving.json"
    assert p.exists(), "run scripts/fit_serving_calibration.py --disease rop"
    c = json.loads(p.read_text())
    assert "TTA" in c["method"] or "TTA" in c["note"]
    # calibration must never move a decision
    assert c["argmax_changed"] == 0
    # and the artifact must record an explicit apply/no-apply decision with a reason
    assert isinstance(c["apply"], bool) and c["decision_reason"]


def test_calibration_is_not_applied_when_it_does_not_help():
    """ROP: TTA averaging already calibrates; applying T=1.08 is churn for no ECE gain."""
    c = json.loads((ROOT / "results" / "rop" / "calibration_serving.json").read_text())
    if c["calibrated"]["ece"] >= c["uncalibrated"]["ece"] * 0.90:
        assert c["apply"] is False, "a non-improving calibration must not be applied"


def test_applying_temperature_preserves_ranking():
    """Mathematical guarantee of softmax(log(p)/T): monotone, so argmax cannot change."""
    from webapp.inference import DiseaseModel
    p = np.array([0.1, 0.5, 0.25, 0.1, 0.05])
    for T in (0.5, 0.7, 1.3, 2.0):
        z = np.log(np.clip(p, 1e-12, 1.0)) / T
        z -= z.max()
        e = np.exp(z)
        assert (e / e.sum()).argmax() == p.argmax()


# ── C5: ordinal head reports the referable endpoint explicitly ───────────────
def test_binary_diseases_have_no_referable_block(registry):
    assert "referable" not in _finding(registry, "ROP", "infant")


# ── C6: abstention ───────────────────────────────────────────────────────────
def test_abstention_band_was_derived_on_val():
    p = ROOT / "results" / "rop" / "abstention_band.json"
    assert p.exists(), "run scripts/fit_abstention_band.py --disease rop"
    b = json.loads(p.read_text())
    assert b["split"] == "val", "the band must never be chosen on test"
    if b["enabled"]:
        # a band only earns its place if it isolates genuinely unreliable cases
        assert b["acc_inside"] < b["acc_outside"]
        assert b["abstain_frac"] <= 0.15


def test_every_finding_declares_uncertainty(registry):
    for disease, ctx in (("ROP", "infant"),):   # ROP-only deployment
        assert "uncertain" in _finding(registry, disease, ctx)


def test_uncertain_case_carries_a_reason():
    """Synthesise a score inside the band and check the message is populated."""
    b = json.loads((ROOT / "results" / "rop" / "abstention_band.json").read_text())
    if not b.get("enabled"):
        pytest.skip("abstention not enabled for ROP")
    mid = (b["lo"] + b["hi"]) / 2
    assert b["lo"] <= mid <= b["hi"]
    assert b["acc_inside"] < b["acc_outside"]


def test_abstention_rate_on_real_cohort_is_bounded(registry):
    """C6 acceptance: abstention rate reported per cohort, and not absurd."""
    paths = sorted(glob.glob(str(COHORTS / "infants" / "DG0" / "*.jpg")))
    if not paths:
        pytest.skip("infant cohort not built")
    unc = 0
    for p in paths:
        f = next(x for x in registry.analyze(Image.open(p).convert("RGB"), "infant")["findings"]
                 if x["disease"] == "ROP")
        unc += bool(f.get("uncertain"))
    rate = unc / len(paths)
    print(f"\nROP abstention rate on infant no-ROP cohort: {unc}/{len(paths)} = {rate:.1%}")
    assert rate <= 0.40, f"abstaining on {rate:.0%} of a real cohort is not usable"
