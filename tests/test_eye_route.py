"""Task 10: eye-level route. One eye, many photographs, one answer (the worst view).

Run:  .venv/bin/python -m pytest tests/test_eye_route.py -v
"""
import io
import json
import sys
from pathlib import Path

import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from conftest import fundus_like  # noqa: E402
from webapp.inference import StagingPreview, EyeNotGradable, eye_from_photos  # noqa: E402

CLASSES = StagingPreview.CLASSES


def _png():
    buf = io.BytesIO()
    fundus_like().save(buf, "PNG")
    buf.seek(0)
    return buf


def _photo(stage, p=0.6, gradable=True, reason="blurry"):
    """A fake per-photo analyze() result with the real staging structure."""
    if not gradable:
        return {"gradable": False, "quality": {"verdict": "not_fundus", "reason": reason},
                "findings": []}
    rest = (1 - p) / 5
    scores = {c: round((p if i == stage else rest) * 100, 1) for i, c in enumerate(CLASSES)}
    return {"gradable": True, "findings": [],
            "staging": {"prediction": CLASSES[stage], "scores": scores}}


# ── pure aggregation: the worst-view rule ────────────────────────────────────
def test_worst_view_is_the_max_stage():
    out = eye_from_photos([_photo(1), _photo(3, 0.7), _photo(0)], "OD")
    assert out["eye"]["stage_index"] == 3
    assert out["eye"]["stage"] == "Stage 3"
    assert out["eye"]["driving_photo"] == 1
    assert len(out["eye"]["prob6"]) == 6
    assert out["eye"]["prob6"][3] == pytest.approx(0.7, abs=1e-3)
    assert out["laterality"] == "OD"


def test_tie_goes_to_the_most_confident_photo():
    out = eye_from_photos([_photo(2, 0.5), _photo(2, 0.9), _photo(2, 0.7)], "OS")
    assert out["eye"]["driving_photo"] == 1


def test_ungradable_photo_is_excluded_and_named():
    out = eye_from_photos([_photo(0), _photo(4, gradable=False, reason="not a fundus"),
                           _photo(1)], "OD")
    assert out["eye"]["stage_index"] == 1          # the stage-4 photo must not count
    assert out["eye"]["driving_photo"] == 2
    assert any("photo 2 not gradable" in w and "not a fundus" in w for w in out["warnings"])


def test_limitation_warning_always_present():
    out = eye_from_photos([_photo(0)], "OD")
    assert any("acquisition view is not detected" in w and "temporal periphery" in w
               for w in out["warnings"])


def test_all_rejected_raises():
    with pytest.raises(EyeNotGradable):
        eye_from_photos([_photo(0, gradable=False), _photo(0, gradable=False)], "OD")


# ── safety: a served config must never be a cached-preprocessing config ──────
def test_cached_config_is_refused_at_serve_time():
    from models.common.config import load_config
    from webapp.inference import assert_not_cached_cfg
    cfg = load_config("configs/rop_staging_structured.yaml")
    assert_not_cached_cfg(cfg, "x.yaml")           # the served config passes
    cfg.preprocess["cached"] = True
    cfg.preprocess["cache_root"] = "/somewhere"
    with pytest.raises(RuntimeError, match="cached"):
        assert_not_cached_cfg(cfg, "x.yaml")


# ── the HTTP route ───────────────────────────────────────────────────────────
@pytest.fixture(scope="module")
def client():
    import webapp.app as A
    if not any(m.loaded for m in A.REGISTRY.models) or not A.REGISTRY.staging.loaded:
        pytest.skip("no model weights available")
    A.app.config["TESTING"] = True
    return A.app.test_client()


def _post(client, n=3, **over):
    data = {"images": [(_png(), f"p{i}.png") for i in range(n)], "eye": "OD",
            "context": "infant"}
    data.update(over)
    return client.post("/predict_eye", data=data, content_type="multipart/form-data")


def test_three_photos_one_answer(client):
    r = _post(client)
    assert r.status_code == 200, r.get_data(as_text=True)
    j = r.get_json()
    assert len(j["per_photo"]) == 3
    assert [p["index"] for p in j["per_photo"]] == [0, 1, 2]
    stages = [CLASSES.index(p["staging"]["prediction"]) for p in j["per_photo"]]
    assert j["eye"]["stage_index"] == max(stages)
    assert stages[j["eye"]["driving_photo"]] == max(stages)
    assert j["laterality"] == "OD"


def test_zero_files_is_400(client):
    r = client.post("/predict_eye", data={"eye": "OD"}, content_type="multipart/form-data")
    assert r.status_code == 400


def test_too_many_files_is_400(client):
    assert _post(client, n=17).status_code == 400


@pytest.mark.parametrize("eye", ["left", "", "od"])
def test_bad_laterality_is_400(client, eye):
    assert _post(client, n=1, eye=eye).status_code == 400


def test_bad_file_is_400_naming_the_index(client):
    data = {"images": [(_png(), "a.png"), (io.BytesIO(b"junk"), "b.png")], "eye": "OS",
            "context": "infant"}
    r = client.post("/predict_eye", data=data, content_type="multipart/form-data")
    assert r.status_code == 400
    assert r.get_json()["error"] == "Invalid image: photo 2 could not be read"


def test_all_ungradable_is_422(client):
    blank = io.BytesIO()
    Image.new("RGB", (300, 300), (255, 255, 255)).save(blank, "PNG")
    blank.seek(0)
    r = client.post("/predict_eye", data={"images": [(blank, "w.png")], "eye": "OD", "context": "infant"},
                    content_type="multipart/form-data")
    assert r.status_code == 422


def test_report_accepts_eye_payload(client):
    from test_api_and_report import _pdf_text
    summary = {"laterality": "OD", "warnings": ["acquisition view is not detected - x"],
               "eye": {"stage_index": 3, "stage": "Stage 3", "driving_photo": 1,
                       "prob6": [0.1, 0.1, 0.1, 0.5, 0.1, 0.1]}}
    r = client.post("/report", data={"image": (_png(), "a.png"), "context": "infant",
                                     "id": "eyetest", "eye_summary": json.dumps(summary)},
                    content_type="multipart/form-data")
    assert r.status_code == 200
    out = ROOT / "results/reports/report_eyetest.pdf"
    text = _pdf_text(out)
    assert "Eye summary" in text and "Stage 3" in text
    out.unlink()


def test_missing_context_is_the_same_error_as_predict(client):
    r = client.post("/predict_eye", data={"images": [(_png(), "a.png")], "eye": "OD"},
                    content_type="multipart/form-data")
    assert r.status_code == 400
    assert r.get_json()["error"] == "Patient context is required"


def test_oversized_single_photo_is_400_naming_the_photo(client):
    import webapp.app as A
    big = io.BytesIO(b"\0" * (A.MAX_UPLOAD + 10))
    r = client.post("/predict_eye", data={"images": [(_png(), "a.png"), (big, "b.png")],
                                           "eye": "OD", "context": "infant"},
                    content_type="multipart/form-data")
    assert r.status_code == 400
    assert "photo 2" in r.get_json()["error"]


def test_413_handler_returns_json(client):
    import webapp.app as A
    with A.app.test_request_context("/predict_eye", method="POST"):
        resp, code = A._too_large(None)
    assert code == 413 and "error" in resp.get_json()
    assert A.app.config["MAX_CONTENT_LENGTH"] == A.MAX_UPLOAD * 16


def test_staging_not_loaded_is_503(client, monkeypatch):
    import webapp.app as A
    monkeypatch.setattr(A.REGISTRY.staging, "loaded", False)
    r = _post(client, n=1)
    assert r.status_code == 503
    assert "not loaded" in r.get_json()["detail"]


def test_context_without_rop_is_400(client, monkeypatch):
    import webapp.app as A
    monkeypatch.setitem(A.REGISTRY.contexts, "nomodel", {"runs": ["DR"]})
    r = _post(client, n=1, context="nomodel")
    assert r.status_code == 400
    assert "does not run the ROP model" in r.get_json()["detail"]


@pytest.mark.parametrize("bad", [{"driving_photo": "x"}, {"stage_index": 9}])
def test_report_bad_eye_summary_is_400(client, bad):
    eye = {"stage_index": 3, "stage": "Stage 3", "driving_photo": 1, **bad}
    summary = {"laterality": "OD", "warnings": [], "eye": eye}
    r = client.post("/report", data={"image": (_png(), "a.png"), "context": "infant",
                                     "eye_summary": json.dumps(summary)},
                    content_type="multipart/form-data")
    assert r.status_code == 400
