"""The served ROP model: the DINOv2 ViT-S/14 5-fold ensemble (arm B, seed 42).

Guards what the app now rests on: the config is arm B's and nothing else, the ensemble IS the
mean of its five checkpoints, the screening line is the pre-registered CV operating point (not
the retired EfficientNet threshold, which flags every normal photo at the new hospital when
applied to this model), the ViT Grad-CAM paints only the retina, the model-card numbers still
reproduce from the committed dev predictions, and both HTTP routes serve this model.

Model-dependent tests skip when results/rop_staging_v3/*.pth are absent
(scripts/deploy_rop_v3_weights.py puts them there).

Run:  .venv/bin/python -m pytest tests/test_webapp_staging_ensemble.py -v
"""
import base64
import hashlib
import io
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DEPLOY = ROOT / "configs" / "rop_staging_deploy.yaml"
ARM_B = ROOT / "configs" / "rop_staging_dinov2.yaml"
V3_SCORES = ROOT / "results" / "rop" / "v3" / "v3_scores.json"
OP_KEY = ("arms", "B", "seeds", "42", "operating_point", "cv", "threshold")
ROP_IMAGES = sorted((ROOT / "test_images" / "rop").glob("*.jpg"))


def _flat(d, prefix=""):
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out.update(_flat(v, f"{prefix}{k}."))
        else:
            out[f"{prefix}{k}"] = v
    return out


def _deploy():
    return yaml.safe_load(DEPLOY.read_text())


def _image(i=0):
    if len(ROP_IMAGES) <= i:
        pytest.skip("test_images/rop not present")
    return Image.open(ROP_IMAGES[i]).convert("RGB")


# ── the config: arm B, plus a deploy block, and nothing else ─────────────────
def test_deploy_config_is_arm_B_plus_a_deploy_block():
    dep, b = _flat(_deploy()), _flat(yaml.safe_load(ARM_B.read_text()))
    rest = {k: v for k, v in dep.items() if not k.startswith("deploy.")}
    assert rest == b, sorted(k for k in set(rest) | set(b) if rest.get(k) != b.get(k))
    assert dep["model.weights_path"] is None and dep["preprocess.cached"] is False


def test_deploy_block_names_five_fold_checkpoints_of_seed_42():
    d = _deploy()["deploy"]
    assert d["arm"] == "B" and d["seed"] == 42 and d["ensemble"] == "mean_prob"
    assert d["weights"] == [f"results/rop_staging_v3/dinov2_s42_fold{k}.pth" for k in range(5)]
    assert tuple(d["operating_point"]["key"].split(".")) == OP_KEY


def test_card_scores_reproduce_from_the_committed_dev_predictions():
    """The model card's new-hospital numbers are this exact ensemble's, not a stale copy."""
    runs = ROOT / "results/rop/v3/runs/dinov2_s_s42"
    if not (runs / "fold_4" / "dev_predictions.npz").exists():
        pytest.skip("committed dev predictions absent")
    from scripts import rop_v3_score as S
    ens = S._mean_arm([S._npz(runs / f"fold_{k}" / "dev_predictions.npz") for k in range(5)])
    ddf, keys = S._dev_frame()
    U = S.aggregate_units(keys, ddf.label.values, ens["y_pred"], ens["prob"])
    m = S.class_metrics(U["y"], U["pred"])
    s = _deploy()["deploy"]["scores"]
    assert len(U["y"]) == s["n_eyes"] and len(ddf) == s["n_images"]
    assert round(m["qwk_5class"], 3) == s["qwk_5class"]
    assert round(m["accuracy"], 3) == s["accuracy"]
    assert round(m["macro_f1_present"], 3) == s["macro_f1"]
    t = json.loads(V3_SCORES.read_text())
    for k in OP_KEY:
        t = t[k]
    op = S.op_report(U["score"], U["y"], t)
    assert round(op["any_rop"]["sensitivity"], 3) == s["any_rop_sensitivity"]
    assert round(op["severe_vs_normal"]["sensitivity"], 3) == s["severe_sensitivity"]
    assert round(S.ece(U["prob"], U["y"]), 3) == s["ece_eye"]


# ── the loaded ensemble ──────────────────────────────────────────────────────
@pytest.fixture(scope="module")
def app_module():
    import webapp.app as A
    if not A.REGISTRY.staging.loaded:
        pytest.skip("staging ensemble weights not deployed")
    A.app.config["TESTING"] = True
    return A


@pytest.fixture(scope="module")
def ens(app_module):
    return app_module.REGISTRY.staging


def test_the_registry_serves_the_ensemble_and_not_the_legacy_model(app_module):
    from webapp.inference import StagingEnsemble
    reg = app_module.REGISTRY
    assert isinstance(reg.staging, StagingEnsemble) and len(reg.staging.members) == 5
    assert reg.legacy_staging is None, "the EfficientNet model loads only behind its flag"


def test_deployed_files_match_their_manifest(ens):
    man = json.loads((ROOT / "results/rop_staging_v3/MANIFEST.json").read_text())
    assert [f["file"] for f in man["folds"]] == _deploy()["deploy"]["weights"]
    for f in man["folds"]:
        assert hashlib.sha256((ROOT / f["file"]).read_bytes()).hexdigest() == f["sha256"]


def test_probs_are_a_six_class_distribution(ens):
    p, idx = ens.probs(_image())
    assert p.shape == (6,) and (p >= 0).all()
    assert float(p.sum()) == pytest.approx(1.0, abs=1e-5)
    assert idx == int(np.argmax(p))


def test_ensemble_is_the_mean_of_five_independently_loaded_checkpoints(ens):
    import torch
    from models.common.architectures import build_from_cfg
    img = _image()
    x = ens.tfm(img).unsqueeze(0)
    singles = []
    for f in _deploy()["deploy"]["weights"]:
        m = build_from_cfg(ens.cfg, pretrained=False)
        m.load_state_dict(torch.load(ROOT / f, map_location="cpu", weights_only=True))
        with torch.no_grad():
            singles.append(m.eval()(x).float().softmax(1)[0].numpy())
    singles = np.stack(singles)
    # members run on the serving device (MPS here), the reference on CPU
    assert np.allclose(ens.member_probs(img), singles, atol=2e-3)
    assert np.allclose(ens.probs(img)[0], singles.mean(0), atol=2e-3)


def test_screening_line_is_the_preregistered_cv_operating_point(ens):
    t = json.loads(V3_SCORES.read_text())
    for k in OP_KEY:
        t = t[k]
    assert ens.screening["threshold"] == pytest.approx(t, abs=1e-12)
    retired = json.loads((ROOT / "results/rop/screening_rebase.json").read_text())["threshold"]
    assert abs(ens.screening["threshold"] - retired) > 0.5, \
        "the EfficientNet threshold must not be applied to the ensemble's scores"


def test_score_reaching_the_line_is_positive_and_just_under_is_not(ens):
    t = ens.screening["threshold"]
    for delta, expected in ((+0.001, "ROP Detected"), (0.0, "ROP Detected"),
                            (-0.001, "No ROP")):
        probs = np.zeros(6, dtype=np.float64)
        probs[0] = 1.0 - (t + delta)
        probs[3] = t + delta                     # all of it on stage 3
        f = ens.screening_finding(probs)
        assert f["prediction"] == expected, (delta, f["decision"])
    f = ens.screening_finding(np.array([1.0, 0, 0, 0, 0, 0]))
    assert f["prediction"] == "No ROP" and f["uncertain"] is False
    assert "staging" in f["decision"]["model"] and "DINOv2" in f["decision"]["model"]


def test_a_disease_stage_under_the_line_is_flagged_for_review(ens):
    probs = np.array([0.30, 0.50, 0.10, 0.05, 0.03, 0.02])   # Stage 1 wins, 1-p0 = 0.70
    f = ens.screening_finding(probs)
    assert f["prediction"] == "No ROP"
    assert f["uncertain"] is True and "Stage 1" in f["uncertain_reason"]


def test_vit_gradcam_is_on_the_patch_grid(ens):
    import torch
    x = ens.tfm(_image()).unsqueeze(0).to(ens.device)
    cam = ens._token_cam(x)
    assert cam.shape == (28, 28) and (cam >= 0).all()


def test_gradcam_map_is_frame_sized_in_unit_range_and_zero_off_the_retina(ens):
    from models.common.data_quality import retina_mask
    from webapp.inference import _model_frame
    img = _image()
    frame = _model_frame(ens.tfm, img)
    mask = retina_mask(frame)
    cam = ens.cam_map(img)
    assert cam.shape == frame.shape[:2]
    assert cam.min() >= 0.0 and cam.max() <= 1.0 and cam.max() > 0.0
    assert mask is not None and (cam[~mask] == 0.0).all()


def test_gradcam_overlay_and_evidence(ens):
    from webapp.inference import _model_frame
    img = _image()
    b64, ev = ens.gradcam(img, 1)
    overlay = np.array(Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB"))
    assert overlay.shape == _model_frame(ens.tfm, img).shape
    assert ev["measurable"] is True and 0.0 <= ev["off_retina_pct"] <= 100.0


# ── HTTP: both routes serve this model ───────────────────────────────────────
def _jpeg(i):
    buf = io.BytesIO()
    _image(i).save(buf, "JPEG", quality=95)
    buf.seek(0)
    return buf


def test_predict_route_serves_the_ensemble(app_module):
    r = app_module.app.test_client().post(
        "/predict", data={"image": (_jpeg(0), "a.jpg"), "context": "infant"},
        content_type="multipart/form-data")
    assert r.status_code == 200, r.get_data(as_text=True)
    j = r.get_json()
    assert j["gradable"] is True
    rop = next(f for f in j["findings"] if f["disease"] == "ROP")
    assert "DINOv2" in rop["decision"]["model"]
    assert j["staging"]["model"] == app_module.REGISTRY.staging.card["name"]
    assert sum(j["staging"]["scores"].values()) == pytest.approx(100.0, abs=0.5)


def test_eye_route_three_photos_one_answer(app_module):
    from webapp.inference import StagingEnsemble
    data = {"images": [(_jpeg(i), f"p{i}.jpg") for i in range(3)], "eye": "OS",
            "context": "infant"}
    r = app_module.app.test_client().post("/predict_eye", data=data,
                                          content_type="multipart/form-data")
    assert r.status_code == 200, r.get_data(as_text=True)
    j = r.get_json()
    assert len(j["per_photo"]) == 3
    stages = [StagingEnsemble.CLASSES.index(p["staging"]["prediction"]) for p in j["per_photo"]]
    assert j["eye"]["stage_index"] == max(stages)
    assert len(j["eye"]["prob6"]) == 6
    d = j["eye"]["driving_photo"]
    assert all(p["heatmaps"] == [] for i, p in enumerate(j["per_photo"]) if i != d), \
        "Grad-CAM is computed for the driving photo only"


def test_the_one_page_carries_the_model_card(app_module):
    """The about page went with the old design; its model card moved to the foot of /."""
    html = app_module.app.test_client().get("/").get_data(as_text=True)
    card = app_module.REGISTRY.staging.card
    assert card["name"] in html and "0.752" in html and "230 eyes" in html
    assert "not a medical device" in html.lower()
