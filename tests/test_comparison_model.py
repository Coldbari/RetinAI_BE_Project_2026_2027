"""The demo's comparison model (webapp/comparison.py) and the one-page app around it.

Guards: the comparison model's preprocessing is the source package's and not ours; its
agreement verdicts are right for every case including the two stages it does not know; the
eye rule is the same worst-view rule the served model uses; the page renders the comparison
payload and the seen-photo warning; the app has exactly one page; and, when the deployed
files are present, the three seeds share one backbone and the model answers beside the
served ensemble on both routes.

Model-dependent tests skip when results/comparison/*.pt are absent
(scripts/deploy_comparison_weights.py puts them there).

Run:  .venv/bin/python -m pytest tests/test_comparison_model.py -v
"""
import hashlib
import io
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from webapp import comparison as C  # noqa: E402

REG = yaml.safe_load((ROOT / "webapp" / "registry.yaml").read_text())
SPEC = REG["comparison"]
DEPLOYED = all(Path(p).exists() for p in [SPEC["backbone"], *[a["file"] for a in SPEC["adapters"]]])
ROP_IMAGES = sorted((ROOT / "test_images" / "rop").glob("*.jpg"))


def _image(i=0):
    if len(ROP_IMAGES) <= i:
        pytest.skip("test_images/rop not present")
    return Image.open(ROP_IMAGES[i]).convert("RGB")


# ── registry ─────────────────────────────────────────────────────────────────
def test_registry_declares_three_seed_adapters_and_one_backbone():
    assert [a["seed"] for a in SPEC["adapters"]] == [42, 1, 2]
    assert SPEC["backbone"].endswith(".pt") and isinstance(SPEC["tta"], bool)
    assert "comparison" in REG
    s = SPEC["scores"]
    assert s["unit"] == "photo" and s["n_test_photos"] == 315 and 0 < s["accuracy"] < 1


# ── preprocessing: the SOURCE package's pipeline, not ours ───────────────────
def test_to_input_reproduces_the_source_pipeline_shape_and_statistics():
    im = Image.fromarray(np.zeros((600, 800, 3), np.uint8))
    # a bright disc on black: the black-border crop must find it
    yy, xx = np.mgrid[0:600, 0:800]
    disc = ((yy - 300) ** 2 + (xx - 400) ** 2) < 250 ** 2
    arr = np.zeros((600, 800, 3), np.uint8)
    arr[disc] = (170, 60, 40)
    im = Image.fromarray(arr)
    pre = C.source_preprocess(im, 320)
    assert pre.shape == (320, 320, 3) and pre.dtype == np.uint8
    # the crop removed the black border: the disc now fills most of the frame
    assert (pre.mean(axis=2) > 10).mean() > 0.7
    x = C.to_input(im, 320, 280)
    assert x.shape == (1, 3, 280, 280) and x.dtype == torch.float32
    # ImageNet-normalised: a (170, 60, 40) field lands well within +-3 in every channel
    assert float(x.abs().max()) < 3.0


def test_to_input_is_deterministic():
    im = _image(0)
    a, b = C.to_input(im, 320, 280), C.to_input(im, 320, 280)
    assert torch.equal(a, b)


# ── agreement: every case, including the two stages the comparison model lacks ──
@pytest.mark.parametrize("ours,theirs,status", [
    ("Normal", "Normal", "agree"), ("Stage 2", "Stage 2", "agree"),
    ("Stage 1", "Stage 2", "disagree"), ("Normal", "Stage 3", "disagree"),
    ("Stage 4/5", "Stage 3", "not_covered"), ("AP-ROP", "Normal", "not_covered"),
    (None, "Stage 1", "no_reference"),
])
def test_compare_status(ours, theirs, status):
    a = C.compare(ours, theirs)
    assert a["status"] == status and a["theirs"] == theirs and a["message"]


def test_compare_marks_adjacent_stages():
    assert C.compare("Stage 1", "Stage 2")["adjacent"] is True
    assert C.compare("Normal", "Stage 3")["adjacent"] is False


# ── the eye rule: worst view, driving photo = highest probability for that class ──
def _photo(stage, p, gradable=True):
    if not gradable:
        return {"gradable": False}
    scores = {c: round((p if c == stage else (1 - p) / 3) * 100, 1) for c in C.CLASSES}
    return {"gradable": True, "comparison": {"prediction": stage, "scores": scores}}


def test_eye_from_comparisons_uses_the_worst_view():
    out = C.eye_from_comparisons([_photo("Normal", .9), _photo("Stage 2", .6),
                                  _photo("Stage 2", .8), _photo("Stage 1", .7)])
    assert out["stage"] == "Stage 2" and out["driving_photo"] == 2


def test_eye_from_comparisons_skips_ungradable_and_returns_none_when_empty():
    assert C.eye_from_comparisons([_photo("Stage 3", .9, gradable=False)]) is None
    out = C.eye_from_comparisons([_photo("Stage 3", .9, gradable=False), _photo("Stage 1", .5)])
    assert out["stage"] == "Stage 1" and out["driving_photo"] == 1


# ── the page: one page, renders the comparison and the seen-photo warning ────
def test_frontend_renders_the_comparison_and_the_seen_photo_guard():
    js = (ROOT / "webapp" / "static" / "js" / "main.js").read_text()
    assert "res.comparison" in js and "comparison_eye" in js and "seen_in_data" in js
    html = (ROOT / "webapp" / "templates" / "screen.html").read_text()
    assert "comparisoncard" in html and "seenbanner" in html
    assert "MIN_LOADER_MS" not in js, "no forced waiting animation on the demo"


def test_only_the_screening_page_exists():
    names = sorted(p.name for p in (ROOT / "webapp" / "templates").glob("*.html"))
    assert names == ["base.html", "screen.html"]
    assert [p.name for p in (ROOT / "webapp" / "static" / "js").glob("*.js")] == ["main.js"]


# ── the deployed model ───────────────────────────────────────────────────────
@pytest.fixture(scope="module")
def app_module():
    import webapp.app as A
    A.app.config["TESTING"] = True
    return A


def test_retired_pages_are_gone(app_module):
    c = app_module.app.test_client()
    assert c.get("/").status_code == 200
    for path in ("/about", "/history", "/gallery", "/diseases", "/screen"):
        assert c.get(path).status_code == 404, path


def test_page_carries_both_model_cards(app_module):
    html = app_module.app.test_client().get("/").get_data(as_text=True)
    assert "not a medical device" in html.lower()
    if app_module.REGISTRY.staging.loaded:
        card = app_module.REGISTRY.staging.card
        assert card["name"] in html and "0.752" in html and "230 eyes" in html
    if app_module.REGISTRY.comparison.loaded:
        assert SPEC["name"] in html and "never decides" in html


@pytest.mark.skipif(not DEPLOYED, reason="comparison weights not deployed")
def test_manifest_proves_one_shared_backbone_and_the_overlap():
    m = json.loads((ROOT / "results" / "comparison" / "MANIFEST.json").read_text())
    files = {f["file"]: f for f in m["files"]}
    assert files[SPEC["backbone"]]["n_params"] > 300_000_000
    for a in SPEC["adapters"]:
        assert 6_000_000 < files[a["file"]]["n_params"] < 7_000_000
        assert files[a["file"]]["source_sha256"]
    audit = m["overlap_audit"]
    assert audit["their_split_vs_our_split"]["train"]["multiview_locked"] > 0, \
        "the manifest must record that it trained on sealed-half photos"


@pytest.mark.skipif(not DEPLOYED, reason="comparison weights not deployed")
def test_model_answers_four_classes_from_three_members(app_module):
    cm = app_module.REGISTRY.comparison
    assert cm.loaded and len(cm.members) == 3 and [m["seed"] for m in cm.members] == [42, 1, 2]
    mp = cm.member_probs(_image(0))
    assert mp.shape == (3, 4) and np.allclose(mp.sum(1), 1.0, atol=1e-4)
    out = cm.predict(_image(0), "Stage 1")
    assert set(out["scores"]) == set(C.CLASSES) and out["agreement"]["status"] in \
        ("agree", "disagree")
    assert out["not_covered"] == ["Stage 4/5", "AP-ROP"]


@pytest.mark.skipif(not DEPLOYED, reason="comparison weights not deployed")
def test_predict_route_answers_beside_the_served_model(app_module):
    if not app_module.REGISTRY.staging.loaded:
        pytest.skip("served ensemble not deployed")
    buf = io.BytesIO()
    _image(0).save(buf, "JPEG", quality=95)
    buf.seek(0)
    r = app_module.app.test_client().post(
        "/predict", data={"image": (buf, "a.jpg"), "context": "infant"},
        content_type="multipart/form-data")
    j = r.get_json()
    assert r.status_code == 200 and j["staging"] and j["comparison"]
    assert j["comparison"]["agreement"]["ours"] == j["staging"]["prediction"]
    assert "seen_in_data" in j
    rop = next(f for f in j["findings"] if f["disease"] == "ROP")
    assert "DINOv2 ViT-S" in rop["decision"]["model"], "the verdict stays with the served model"


@pytest.mark.skipif(not DEPLOYED, reason="comparison weights not deployed")
def test_eye_route_compares_at_eye_level(app_module):
    if not app_module.REGISTRY.staging.loaded:
        pytest.skip("served ensemble not deployed")
    files = []
    for i in range(2):
        buf = io.BytesIO()
        _image(i).save(buf, "JPEG", quality=95)
        buf.seek(0)
        files.append((buf, f"p{i}.jpg"))
    r = app_module.app.test_client().post(
        "/predict_eye", data={"images": files, "eye": "OD", "context": "infant"},
        content_type="multipart/form-data")
    j = r.get_json()
    assert r.status_code == 200, j
    ce = j["comparison_eye"]
    stages = [C.CLASSES.index(p["comparison"]["prediction"]) for p in j["per_photo"]]
    assert ce["stage_index"] == max(stages) and ce["agreement"]["ours"] == j["eye"]["stage"]


# ── the seen-photo guard ─────────────────────────────────────────────────────
def test_seen_photo_guard_flags_a_sealed_photo_and_passes_a_new_one(app_module):
    if not app_module.SEEN_PHOTOS:
        pytest.skip("seen_photos.json not deployed")
    locked = ROOT / "results" / "prepared" / "rop_staging" / "test_locked.csv"
    if not locked.exists():
        pytest.skip("test_locked.csv absent")
    import pandas as pd
    path = ROOT / pd.read_csv(locked).image_path.iloc[0]
    if not path.exists():
        pytest.skip("sealed photo not on disk")
    hit = app_module.seen_in_data(path.read_bytes())
    assert hit and hit["level"] == "warn" and any("sealed" in t for t in hit["tags"]), hit
    # the demo photos are Ostrava stage-0 photos outside every split: in the data, used by nobody
    demo = sorted((ROOT / "test_images" / "rop").glob("*.jpg"))
    if demo:
        info = app_module.seen_in_data(demo[0].read_bytes())
        assert info is None or info["level"] == "info", info
    fresh = io.BytesIO()
    Image.fromarray(np.random.default_rng(0).integers(0, 255, (64, 64, 3), np.uint8)).save(fresh, "PNG")
    assert app_module.seen_in_data(fresh.getvalue()) is None
    assert hashlib.md5(b"x").hexdigest() not in app_module.SEEN_PHOTOS
