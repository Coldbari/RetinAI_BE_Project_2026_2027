"""Phase 0B acceptance tests — the gradability gate (G1/G3/G6/G7).

Acceptance criteria from docs/REMEDIATION_PLAN.md:
  G1  all four non-fundus probes return verdict "not_fundus"
  G3  an ungradable image produces NO disease prediction and NO downloadable PDF
  G7  false-rejection rate < 2% on legitimate cohorts

The FRR cohorts (infant TEST split, HRF) are HELD OUT from calibration — thresholds come
from the infant TRAIN split only (scripts/calibrate_gate.py). These are measurements, not fits.

Run:  .venv/bin/python -m pytest tests/test_gate.py -v
      (cohort tests skip unless .venv/bin/python tests/build_cohorts.py has been run)
"""
import glob
import io
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from models.common.data_quality import gradability, load_gate_thresholds  # noqa: E402

COHORTS = ROOT / ".cohorts"
FRR_BUDGET = 0.02


def _probes():
    rng = np.random.default_rng(42)
    return {
        "noise": Image.fromarray(rng.integers(0, 255, (800, 800, 3), dtype=np.uint8)),
        "grey": Image.fromarray(np.full((800, 800, 3), 128, np.uint8)),
        "white": Image.fromarray(np.full((800, 800, 3), 255, np.uint8)),
        "black": Image.fromarray(np.zeros((800, 800, 3), np.uint8)),
    }


# ── G1: the four original probes ─────────────────────────────────────────────
@pytest.mark.parametrize("name", ["noise", "grey", "white", "black"])
def test_synthetic_probe_is_rejected(name):
    r = gradability(_probes()[name])
    assert r["gradable"] is False
    assert r["verdict"] == "not_fundus", f"{name} -> {r['verdict']}"


@pytest.mark.parametrize("chart", ["outputs/evaluation_dashboard.png",
                                   "outputs/training_curves_unified.png"])
def test_project_chart_is_rejected(chart):
    """The original failure: the project's own matplotlib charts scored positive
    for all three diseases."""
    p = ROOT / chart
    if not p.exists():
        pytest.skip(f"{chart} not present")
    assert gradability(Image.open(p).convert("RGB"))["gradable"] is False


def test_thresholds_are_calibrated_not_default():
    thr = load_gate_thresholds()
    assert (ROOT / "results" / "gate_thresholds.json").exists(), \
        "run scripts/calibrate_gate.py first"
    assert thr["min_saturation"] > 0
    # red_hue_frac is deliberately 0.0 since the 2026-08-19 recalibration: fitted on
    # Ostrava alone it silently rejected real Shenzhen/Multi-View fundus (p1 0.03-0.09).
    # The structural checks that replaced it must be present instead.
    assert thr.get("max_corner_lum", 0) > 0, "corner-aperture check missing"
    assert thr.get("max_flat_frac", 0) > 0, "graphics flatness check missing"


# ── G7: false-rejection on held-out real cohorts ─────────────────────────────
def _cohort(pattern):
    return sorted(glob.glob(str(COHORTS / pattern)))


@pytest.mark.parametrize("name,pattern", [
    ("infant no-ROP", "infants/DG0/*.jpg"),
    ("infant ROP+", "infants/DG1/*.jpg"),
    ("HRF healthy", "hrf_test/healthy/*.jpg"),
    ("HRF DR", "hrf_test/dr/*.JPG"),
    ("HRF glaucoma", "hrf_test/glaucoma/*.jpg"),
])
def test_false_rejection_within_budget(name, pattern):
    paths = _cohort(pattern)
    if not paths:
        pytest.skip(f"{name} cohort not built — run tests/build_cohorts.py")
    rejected = [p for p in paths
                if not gradability(Image.open(p).convert("RGB"))["gradable"]]
    frr = len(rejected) / len(paths)
    assert frr <= FRR_BUDGET, (
        f"{name}: false-rejection {frr:.1%} ({len(rejected)}/{len(paths)}) exceeds "
        f"{FRR_BUDGET:.0%}. A gate that rejects real patients is worse than no gate. "
        f"Rejected: {[Path(p).name for p in rejected[:5]]}")


# ── G3: fail-closed through the serving stack ────────────────────────────────
@pytest.fixture(scope="module")
def registry():
    from webapp.inference import Registry
    r = Registry(str(ROOT / "webapp" / "registry.yaml"))
    if not any(m.loaded for m in r.models):
        pytest.skip("no model weights available")
    return r


def test_ungradable_image_yields_no_disease_prediction(registry):
    res = registry.analyze(_probes()["noise"], "infant")
    assert res["gradable"] is False
    assert res["quality"]["verdict"] == "not_fundus"
    assert all(not f.get("available") for f in res["findings"])
    assert all(f["status"] == "ungradable" for f in res["findings"])
    assert res["heatmaps"] == []


def test_ungradable_findings_carry_no_grade_or_score(registry):
    """Nothing downstream may read a grade from an ungradable result."""
    for f in registry.analyze(_probes()["grey"], "infant")["findings"]:
        assert "grade" not in f and "scores" not in f and "prediction" not in f


def test_real_fundus_still_passes_the_gate(registry):
    paths = _cohort("hrf_test/healthy/*.jpg")
    if not paths:
        pytest.skip("HRF cohort not built")
    res = registry.analyze(Image.open(paths[0]).convert("RGB"), "infant")
    assert res["gradable"] is True
    assert any(f.get("available") for f in res["findings"])


# ── G3/G6 over HTTP ──────────────────────────────────────────────────────────
@pytest.fixture(scope="module")
def client():
    import webapp.app as A
    if not any(m.loaded for m in A.REGISTRY.models):
        pytest.skip("no model weights available")
    A.app.config["TESTING"] = True
    return A.app.test_client()


def _png(img):
    buf = io.BytesIO(); img.save(buf, "PNG"); buf.seek(0)
    return buf


def test_report_refuses_ungradable_image(client):
    r = client.post("/report", data={"image": (_png(_probes()["noise"]), "n.png"),
                                     "context": "infant"},
                    content_type="multipart/form-data")
    assert r.status_code == 422
    assert r.get_json()["verdict"] == "not_fundus"


def test_predict_reports_ungradable_without_findings(client):
    r = client.post("/predict", data={"image": (_png(_probes()["grey"]), "g.png"),
                                      "context": "infant"},
                    content_type="multipart/form-data")
    assert r.status_code == 200
    body = r.get_json()
    assert body["gradable"] is False
    assert all(not f.get("available") for f in body["findings"])


def test_non_image_upload_is_rejected(client):
    r = client.post("/predict", data={"image": (io.BytesIO(b"not an image at all"), "x.png"),
                                      "context": "infant"},
                    content_type="multipart/form-data")
    assert r.status_code == 400


def test_tiny_image_is_rejected(client):
    r = client.post("/predict", data={"image": (_png(Image.new("RGB", (16, 16))), "t.png"),
                                      "context": "infant"},
                    content_type="multipart/form-data")
    assert r.status_code == 400


# ── the non-"not_fundus" verdicts were never exercised (Phase 1 upgrade) ─────
# G1 originally verified only that synthetic probes return "not_fundus". The too_blurry and
# poor_exposure paths were dead code as far as the suite was concerned: a real fundus image
# degraded until ungradable is the only thing that reaches them.
def _real_fundus():
    paths = _cohort("hrf_test/healthy/*.jpg") or _cohort("infants/DG0/*.jpg")
    if not paths:
        pytest.skip("no real fundus cohort built")
    return Image.open(paths[0]).convert("RGB")


def test_heavy_blur_is_caught_as_too_blurry():
    from PIL import ImageFilter
    img = _real_fundus().resize((512, 512)).filter(ImageFilter.GaussianBlur(radius=18))
    r = gradability(img)
    assert r["gradable"] is False
    assert r["verdict"] == "too_blurry", f"got {r['verdict']} ({r['metrics']})"


def test_severe_underexposure_is_caught():
    img = Image.fromarray((np.asarray(_real_fundus().resize((512, 512)),
                                      dtype=np.float32) * 0.06).astype(np.uint8))
    r = gradability(img)
    assert r["gradable"] is False
    assert r["verdict"] in ("poor_exposure", "not_fundus"), r["verdict"]


def test_mild_blur_is_still_accepted():
    """The gate must not reject clinically usable images — it is not a quality connoisseur."""
    from PIL import ImageFilter
    img = _real_fundus().resize((512, 512)).filter(ImageFilter.GaussianBlur(radius=1.0))
    assert gradability(img)["gradable"] is True


# ── field regression (2026-08-19): a real-world photograph must never reach a model ──────
def _warm_indoor_scene(seed=3):
    """Deterministic reproduction of the failure class a user found in one upload: a
    warm-lit furniture photo passed every colour heuristic and scored 'ROP Detected
    91.5%'. Structured like the real thing: warm wall, furniture blocks, light falloff."""
    import numpy as np
    from PIL import Image
    r = np.random.default_rng(seed)
    h, w = 800, 600
    base = np.zeros((h, w, 3), np.float32)
    wall = 180.0
    base[:, :, 0] = wall * 1.2; base[:, :, 1] = wall; base[:, :, 2] = wall / 1.15
    for _ in range(6):
        x0, y0 = int(r.integers(0, w - 60)), int(r.integers(0, h - 60))
        x1, y1 = x0 + int(r.integers(50, w // 2)), y0 + int(r.integers(50, h // 2))
        base[y0:y1, x0:x1] = (170, 130, 90)
    light = np.linspace(1.15, 0.75, h)[:, None, None]
    return Image.fromarray(
        np.clip(base * light + r.normal(0, 6, (h, w, 3)), 0, 255).astype(np.uint8))


def test_real_world_photograph_is_rejected_not_diagnosed():
    """The corner-aperture check: photographs fill their corners, fundus images cannot."""
    from models.common.data_quality import gradability
    v = gradability(_warm_indoor_scene())
    assert not v["gradable"], f"a furniture-like photo passed the gate: {v['metrics']}"
    assert v["verdict"] == "not_fundus"


def test_photograph_never_yields_a_finding_end_to_end(registry):
    res = registry.analyze(_warm_indoor_scene(), "infant")
    assert res["gradable"] is False
    assert all(f.get("status") == "ungradable" or f.get("available") is False
               for f in res["findings"])


def test_gate_relaxation_did_not_readmit_charts():
    """The colour bounds were relaxed for pale ROP-VL/Shenzhen fundus; the flat-region
    check must be what keeps graphics out now."""
    from pathlib import Path
    from PIL import Image
    from models.common.data_quality import gradability
    charts = sorted((Path(__file__).resolve().parent.parent / "graphs").rglob("*.png"))[:5]
    for c in charts:
        v = gradability(Image.open(c).convert("RGB"))
        assert not v["gradable"], f"chart passed the gate: {c.name}"


def _night_interior_scene(seed=5):
    """Second field bypass (2026-08-19): a night-time interior — dark corners defeat the
    aperture check, lamplight passes the colour checks."""
    import cv2
    import numpy as np
    from PIL import Image
    r = np.random.default_rng(seed)
    h, w = 650, 900
    img = np.zeros((h, w, 3), np.float32) + 14
    for _ in range(5):
        cv2.circle(img, (int(r.integers(0, w)), int(r.integers(0, h // 2))),
                   int(r.integers(15, 60)), (255, 170, 60), -1)
    for _ in range(5):
        x0, y0 = int(r.integers(0, w - 80)), int(r.integers(h // 4, h - 80))
        cv2.rectangle(img, (x0, y0), (x0 + 200, y0 + 120), (160, 110, 70), -1)
    img = cv2.GaussianBlur(img, (31, 31), 0)
    return Image.fromarray(np.clip(img + r.normal(0, 5, (h, w, 3)), 0, 255).astype(np.uint8))


def test_night_interior_is_rejected_not_diagnosed():
    """Dark corners are not enough: the retinal field must be ONE bright disk."""
    from models.common.data_quality import gradability
    v = gradability(_night_interior_scene())
    assert not v["gradable"], f"a night-interior photo passed the gate: {v['metrics']}"
    assert v["verdict"] == "not_fundus"


def test_embedding_gate_separates_scenes_from_fundus(registry):
    """The learned layer: real-world scenes must sit far beyond its threshold, so it can
    catch whatever image class defeats the heuristics next."""
    eg = registry.embedding_gate
    if not eg.available:
        import pytest
        pytest.skip("embedding gate artifact/weights not available")
    assert eg.distance(_warm_indoor_scene()) > eg.threshold
    assert eg.distance(_night_interior_scene()) > eg.threshold
