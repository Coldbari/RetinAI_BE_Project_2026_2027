"""Grad-CAM honesty tests.

A user handed the app a fundus photo and got back a heatmap whose brightest blob sat in
the black corner OUTSIDE the retina. Measured on the twelve held-out test images, one of
them puts 24% of its Grad-CAM mass on the surround with the global peak outside the eye
entirely. That is the same family of defect this project has already been burned by — a
classifier on image dimensions alone scored AUC 0.911 — so the fix is not to crop the
evidence away quietly. It is to confine the PICTURE to the retina and publish the NUMBER.

These tests hold three things:
  1. the retinal mask is a real mask (finds the disc, excludes the surround),
  2. the map that reaches the screen carries no colour outside the retina,
  3. the off-retina fraction is measured on the RAW map, so it can be non-zero.

Run:  .venv/bin/python -m pytest tests/test_gradcam_evidence.py -v
"""
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from models.common.data_quality import retina_mask  # noqa: E402
from webapp.inference import _cam_evidence, _model_frame  # noqa: E402

from conftest import fundus_like  # noqa: E402

cv2 = pytest.importorskip("cv2")


def _disc(size=320, radius=0.42, value=170):
    """A synthetic fundus aperture: one bright disc on black, like every real cohort."""
    a = np.zeros((size, size, 3), np.uint8)
    cv2.circle(a, (size // 2, size // 2), int(size * radius), (value, value // 2, value // 3), -1)
    return a


# ── the mask itself ──────────────────────────────────────────────────────────
def test_retina_mask_finds_the_disc_and_excludes_the_corners():
    m = retina_mask(_disc())
    assert m is not None
    assert m[160, 160], "centre of the disc must be inside the field of view"
    for corner in [(2, 2), (2, -3), (-3, 2), (-3, -3)]:
        assert not m[corner], "black surround must be outside the field of view"
    # a disc of radius 0.42 covers pi*0.42^2 = 55% of the frame; erosion trims a little
    assert 0.40 < m.mean() < 0.58


def test_retina_mask_survives_a_surround_that_is_not_pure_black():
    """`gray > 7` is not a field-of-view mask on real uploads, which is why this exists.

    Real fundus JPEGs never have a mathematically black surround — sensor noise and
    compression ringing sit a few levels above zero. Measured consequence on the twelve
    held-out test images: preprocessing.circle_crop is a COMPLETE no-op, cropping
    1600x1200 to 1600x1200 on every one. Reproduced here with a noise floor of 8-20.
    """
    rng = np.random.default_rng(0)
    rgb = _disc()
    floor = rng.integers(8, 21, size=rgb.shape, dtype=np.uint8)
    rgb = np.maximum(rgb, floor)                     # lift the surround off pure black
    naive = (cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY) > 7).mean()
    real = retina_mask(rgb).mean()
    assert naive > 0.99, "the naive threshold should now claim the whole frame"
    assert real < 0.60, f"the real mask must still find the aperture (got {real:.2f})"


def test_retina_mask_returns_none_on_an_empty_frame():
    assert retina_mask(np.zeros((64, 64, 3), np.uint8)) is None


# ── what reaches the screen ──────────────────────────────────────────────────
@pytest.fixture(scope="module")
def rop_model():
    """Whichever model the app ACTUALLY serves for ROP.

    Since the re-basing that is the structured head, not the ResNet50 — and a Grad-CAM
    test that exercises a network the product no longer runs proves nothing.
    """
    import webapp.inference as I
    reg = I.Registry()
    if reg.staging.loaded and reg.staging.screening is not None:
        return reg.staging
    dm = next((m for m in reg.models if m.disease == "ROP" and m.loaded), None)
    if dm is None:
        pytest.skip("ROP weights not available")
    return dm


def _overlay(dm, image):
    b64, ev = dm.gradcam(image, 1)
    import base64 as b
    import io as i
    return np.array(Image.open(i.BytesIO(b.b64decode(b64))).convert("RGB")), ev


def test_no_heat_is_painted_outside_the_retina(rop_model):
    """The defect the user photographed: a red blob on the black surround."""
    img = fundus_like()
    overlay, _ = _overlay(rop_model, img)
    frame = _model_frame(rop_model.base_tfm, img)
    mask = retina_mask(frame)
    assert mask is not None
    outside = overlay[~mask].astype(np.int16)
    # jet paints high activation red (R >> B). Outside the FOV the overlay is the dimmed
    # frame, so no pixel there may be red-dominant-and-bright the way a hot blob is.
    hot = (outside[:, 0] > 150) & (outside[:, 0] > outside[:, 2] + 60)
    assert hot.sum() == 0, f"{hot.sum()} hot pixels painted outside the retina"


def test_evidence_measures_leakage_on_the_raw_map(rop_model):
    """off_retina_pct must be computable and non-trivial — not zeroed by the masking."""
    _, ev = _overlay(rop_model, fundus_like())
    assert ev["measurable"] is True
    assert 0.0 <= ev["off_retina_pct"] <= 100.0
    assert 0.0 < ev["concentration_pct"] <= 100.0
    assert ev["peak_zone"] in {"centre of the visible field", "mid-periphery",
                               "outer periphery"}
    assert 1 <= ev["peak_clock"] <= 12
    assert isinstance(ev["peak_inside_retina"], bool)


def test_peak_inside_retina_is_not_true_by_construction(rop_model):
    """Reading the peak off the MASKED map would make this flag always True.

    Guards the exact slip: the check must be able to answer False, so drive it with a map
    whose maximum is planted in the corner.
    """
    frame = _model_frame(rop_model.base_tfm, fundus_like())
    mask = retina_mask(frame)
    cam = np.zeros(frame.shape[:2], np.float32)
    cam[frame.shape[0] // 2, frame.shape[1] // 2] = 1.0    # something inside
    cam[1, 1] = 5.0                                        # the true peak, in the corner
    ev = _cam_evidence(cam, mask)
    assert ev["peak_inside_retina"] is False
    assert ev["off_retina_pct"] > 50.0


def test_overlay_is_drawn_on_the_frame_the_model_saw(rop_model):
    """Grad-CAM lives on the preprocessed grid; overlaying the raw upload misaligns it."""
    img = fundus_like()
    overlay, _ = _overlay(rop_model, img)
    assert overlay.shape == _model_frame(rop_model.base_tfm, img).shape


# ── synthetic padding is not "the model reading the frame" ───────────────────
#
# The structured model letterboxes, so 25% of its display frame is black bars added by
# preprocessing. Counting those as off-retina attention reported ~40% cohort-mean leakage
# for the served model against ~8% for the retired one, and the difference was almost
# entirely the padding: the retired head does not letterbox, so the two numbers were never
# measuring the same region. Measured on 80 images with the shipped objective: 39.7% total
# outside the retina, of which 23.5 points sat on padding and only 16.2 on the camera's own
# surround.
def test_padding_mask_is_analytic_not_a_dark_pixel_hunt(rop_model):
    """A circular aperture makes genuinely dark rows too; those are surround, not padding."""
    import webapp.inference as I
    img = fundus_like()
    pad = I._padding_mask(rop_model.base_tfm, img)
    pre = rop_model.base_tfm.transforms[0]
    if not getattr(pre, "letterbox", False):
        assert pad is None, "a model that does not letterbox must report no padding"
        return
    assert pad is not None and pad.any(), "a letterboxing model must locate its bars"
    # bars are full rows or full columns, never a ragged region
    rows_all, cols_all = pad.all(axis=1), pad.all(axis=0)
    assert rows_all.any() or cols_all.any()
    assert pad.sum() == rows_all.sum() * pad.shape[1] + cols_all.sum() * pad.shape[0] \
        - rows_all.sum() * cols_all.sum()


def test_off_retina_excludes_padding_and_still_accounts_for_all_of_it(rop_model):
    """The split must be exhaustive: surround + padding = everything outside the retina."""
    import webapp.inference as I
    img = fundus_like()
    frame = I._model_frame(rop_model.base_tfm, img)
    mask = retina_mask(frame)
    pad = I._padding_mask(rop_model.base_tfm, img)
    cam = np.zeros(frame.shape[:2], np.float32) + 1.0        # uniform map: shares == areas
    ev = I._cam_evidence(cam, mask, pad)
    outside_share = float((~mask).mean()) * 100.0
    assert ev["off_retina_pct"] + ev["on_padding_pct"] == pytest.approx(outside_share, abs=0.6)


def test_uniform_attention_scores_a_lift_of_one(rop_model):
    """The area normalisation must be right, or cross-model comparison stays broken."""
    import webapp.inference as I
    img = fundus_like()
    frame = I._model_frame(rop_model.base_tfm, img)
    mask = retina_mask(frame)
    ev = I._cam_evidence(np.ones(frame.shape[:2], np.float32), mask,
                         I._padding_mask(rop_model.base_tfm, img))
    assert ev["off_retina_lift"] == pytest.approx(1.0, abs=0.05), \
        "a uniform map must score lift 1.0 by construction"
