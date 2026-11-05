"""Inference backend for the ROP screening web app.

Loads the models declared in ``registry.yaml`` — this deployment declares ROP only — runs
inference through the SAME shared preprocessing as training (so train/serve match), and
produces a Grad-CAM taken from whichever model produced the verdict. Degrades gracefully
when a checkpoint is missing, so the app still boots and says so.

Since 1 Nov 2026 the ROP stage AND the binary ROP decision come from ``StagingEnsemble``:
DINOv2 ViT-S/14, the probability-averaged ensemble of the five CV-fold checkpoints of seed 42
(arm B of docs/paper/PROTOCOL_V2.md, configs/rop_staging_deploy.yaml). It replaced the
EfficientNetV2-S structured model (``StagingPreview``, kept loadable for comparison behind
``RETINAI_LEGACY_STAGING=1``), which had itself replaced the ResNet50 head on 19 Aug 2026
after an external audit measured that head flagging every image at a hospital it never
trained on. Serving device: CUDA, else MPS, else CPU.
"""
from __future__ import annotations

import base64
import io
import json
import os
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image

from models.common.config import load_config
from models.common.architectures import build_from_cfg, get_target_layer
from models.common.data_quality import gradability, retina_mask
from models.common.preprocessing import build_transforms, build_tta_transforms
from models.common.train_utils import predict_with_tta
from reports.report_generator import recommendation

try:
    import cv2
except Exception:                                  # pragma: no cover - optional at import
    cv2 = None

DISPLAY = 320


def _resize_f32(a: np.ndarray, size: int) -> np.ndarray:
    """Bicubic resize of a float32 map, WITHOUT the uint8 round-trip.

    Quantising a 12x12 Grad-CAM to 256 levels before upsampling flattened the low end of
    the ramp into visible bands; interpolating in float keeps the gradient smooth.
    """
    if cv2 is not None:
        return cv2.resize(a, (size, size), interpolation=cv2.INTER_CUBIC)
    idx = np.linspace(0, a.shape[0] - 1, size)
    rows = np.stack([np.interp(idx, np.arange(a.shape[1]), r) for r in a])
    return np.stack([np.interp(idx, np.arange(a.shape[0]), c) for c in rows.T], axis=1)


def _load_metrics(disease_dir):
    p = Path("results") / disease_dir / "metrics.json"
    if not p.exists():
        return None
    try:
        m = json.loads(p.read_text())
        t = m.get("test") or m.get("val") or {}
        return {"accuracy": t.get("accuracy"), "macro_f1": t.get("macro_f1"),
                "auc": t.get("auc"), "qwk": t.get("qwk")}
    except Exception:
        return None


# Below this longest-edge size, Ben Graham's fixed-pixel blur no longer corresponds to the
# scale it had during training. 1500px sits between the two conditions T13 measured (native
# 4288px vs downscaled 1024px) and above the typical size of a messenger-compressed photo.
MIN_TRUSTED_EDGE = 1500


def _resolution_notice(image):
    """Warn when an upload is small enough that preprocessing no longer matches training.

    `circle_crop -> clahe -> ben_graham(sigma=10)` runs at NATIVE resolution before the
    resize to 384, and Ben Graham's blur radius is a fixed number of PIXELS. A downscaled
    upload is therefore processed differently from the images the model was trained on.

    T13 measured the cost on 455 IDRiD images: rescaling 4288px -> 1024px moved referable
    AUC 0.978 -> 0.893 and turned 9 missed cases into 61. The model still answers; it just
    answers a different question, and silence about that is the part that is unsafe.

    This is a WARNING, not a rejection. Refusing small images would block legitimate captures
    from older cameras, and the underlying fix (resize before Ben Graham) needs all three
    models retrained — so the honest interim behaviour is to say so, not to guess.
    """
    edge = max(image.size)
    if edge >= MIN_TRUSTED_EDGE:
        return None
    return {
        "longest_edge": int(edge),
        "expected_min": MIN_TRUSTED_EDGE,
        "severity": "high" if edge < 800 else "moderate",
        "message": (
            f"This image is {image.size[0]}x{image.size[1]}. Preprocessing applies a "
            f"fixed-radius filter before resizing, so images below ~{MIN_TRUSTED_EDGE}px are "
            f"processed differently from the ones these models were trained on. On a "
            f"published test set that shift cost 0.08 AUC and missed 6x more disease. "
            f"Prefer the original camera file over a resized or messaging-app copy."),
    }


def _headline_caveat(disease_dir):
    """The qualification that must travel with the headline AUC, read from evidence.

    metrics.json holds the number the checkpoint scored on its own test split. For ROP that
    split had every ROP-positive image on one camera, so the figure is partly a
    camera-vs-camera separation — a later device-stratified experiment put device-controlled
    AUC at 0.68. A dashboard that prints 0.927 with no qualification is not wrong so much as
    *unanswerable*: the first person to ask "measured how?" gets no answer from the page.

    Built from the recorded JSON rather than hardcoded, so a rerun that moves the number
    moves the caveat with it instead of leaving a stale string behind.
    """
    d = Path("results") / disease_dir
    try:
        ci = json.loads((d / "devsplit_ci.json").read_text())
        dc = ci["device_controlled"]
        lo, hi = dc["ci"]
        best = max((v for v in ci["per_device"].values() if v.get("auc") is not None),
                   key=lambda v: v["positive_units"], default=None)
        msg = (f"Pooled figure. With capture device held constant, AUC is "
               f"{dc['auc']:.3f} [{lo:.3f}, {hi:.3f}]")
        if best:
            msg += (f"; on the one camera with enough ROP-positive infants to support an "
                    f"estimate ({best['positive_units']}), {best['auc']:.3f} "
                    f"[{best['ci'][0]:.3f}, {best['ci'][1]:.3f}]")
        return msg + "."
    except Exception:
        pass
    try:
        # DR's headline AUC is the 5-class macro average, which is NOT the number the
        # screening decision turns on. Referable (grade >= 2) is, and it is much higher —
        # showing only the macro figure understates the model on the endpoint that matters.
        c = json.loads((d / "calibration_serving.json").read_text())
        ref = c.get("referable", {})
        if ref.get("referable_auc"):
            return (f"5-class macro AUC. The screening decision turns on referable DR "
                    f"(grade ≥ 2), where AUC is {ref['referable_auc']:.3f} measured "
                    f"through this app's own 5-view TTA path.")
    except Exception:
        pass
    try:
        a = json.loads((d / "source_audit.json").read_text())
        pct = 100 * a["positives_from_single_class_sources"] / max(a["total_positives"], 1)
        return (f"Pooled across source collections of very different prevalence — "
                f"{pct:.0f}% of positives come from collections containing no negatives. "
                f"Measured WITHIN a source, AUC is "
                f"{a['source_controlled_usable']:.3f} across the "
                f"{a['n_usable']} collections large enough to support an estimate. "
                f"That within-source figure is the defensible one.")
    except Exception:
        pass
    try:
        pr = json.loads((d / "smdg_source_probe.json").read_text())
        single = [r for r in pr.get("per_source", []) if r.get("non_glaucoma") == 0
                  and r.get("glaucoma", 0) > 0]
        if single:
            tot = sum(r.get("glaucoma", 0) for r in pr["per_source"])
            frm = sum(r["glaucoma"] for r in single)
            return (f"Pooled figure across {pr.get('n_prefixes', '?')} source collections of "
                    f"differing prevalence — {frm}/{tot} ({100*frm/max(tot,1):.0f}%) of "
                    f"positives come from collections containing no negatives, so part of "
                    f"this score may be source recognition. Audit in progress.")
    except Exception:
        pass
    return None


def _device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _model_frame(tfm, image: Image.Image) -> np.ndarray:
    """The image AS THE MODEL SEES IT, at display size.

    A Grad-CAM lives on the grid of the preprocessed tensor. Overlaying it on the raw
    upload is only correct while preprocessing happens to be geometry-preserving —
    circle_crop retimes the frame and Resize((s,s)) squashes the aspect ratio, so the two
    agree by luck, not by construction. Rendering the preprocessed frame makes the
    correspondence exact and shows the operator what was actually classified.
    """
    pre = tfm.transforms[0]                          # FundusPreprocess: PIL -> PIL
    return np.array(pre(image).convert("RGB").resize((DISPLAY, DISPLAY)))


def _gradcam_core(model, target, tfm, device, image: Image.Image, objective):
    """Return ``(overlay_b64, evidence)`` — the map, and what it is worth.

    Shared by both served models so a heatmap can never be produced by a different network
    than the verdict it is supposed to explain. `objective(out)` selects the scalar to
    differentiate: one class logit for a plain classifier, log P(any ROP) for the
    structured head, whose decision is not a single logit at all.

    A heatmap with no attached measurement invites the reader to trust whatever blob
    appears. Two things are measured instead: how much attention fell OUTSIDE the retina
    (frame-reading, not disease-reading) and how CONCENTRATED it is (a diffuse map has
    localised nothing). Both are reported next to the picture.
    """
    tensor = tfm(image).unsqueeze(0).to(device)
    grads, acts = [], []
    bw = target.register_full_backward_hook(lambda m, gi, go: grads.append(go[0].detach().cpu()))
    fw = target.register_forward_hook(lambda m, i, o: acts.append(o.detach().cpu()))
    model.zero_grad()
    objective(model(tensor)).backward()
    bw.remove(); fw.remove()

    w = grads[0].mean(dim=[2, 3], keepdim=True)
    cam = torch.relu((w * acts[0]).sum(1)).squeeze().numpy().astype(np.float32)
    return _cam_render(cam, tfm, image)


def _cam_upsample(cam_small: np.ndarray) -> np.ndarray:
    # Upsample in float with a cubic kernel. Quantising the 12x12 map to uint8 and letting
    # PIL resize it banded the low end of the ramp into flat plateaus — visible as hard
    # steps in the blue background of the overlay.
    return np.clip(_resize_f32(cam_small.astype(np.float32), DISPLAY), 0.0, None)


def _retina_only(cam: np.ndarray, mask: np.ndarray | None) -> np.ndarray:
    """The map as PAINTED: zero outside the retina, scaled to [0, 1] inside it.

    Attention on the surround is real and is reported as a number alongside; painting it
    keeps the eye on a region where the model has, by construction, nothing clinical to see.
    """
    shown = cam.copy()
    if mask is not None:
        shown[~mask] = 0.0
    rng = float(shown.max())
    return shown / rng if rng > 0 else shown


def _cam_render(cam_small: np.ndarray, tfm, image: Image.Image):
    """``(overlay_b64, evidence)`` for a raw low-resolution CAM on the model's frame.

    Shared by the CNN path above and the ViT ensemble's token-grid CAM, so the overlay, the
    retina-only colouring and the evidence numbers are computed one way for every model.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cam = _cam_upsample(cam_small)
    frame = _model_frame(tfm, image)
    mask = retina_mask(frame)
    evidence = _cam_evidence(cam, mask, _padding_mask(tfm, image))
    shown = _retina_only(cam, mask)

    heat = (plt.get_cmap("jet")(shown)[:, :, :3] * 255).astype(np.uint8)
    overlay = (0.55 * heat + 0.45 * frame).astype(np.uint8)
    if mask is not None:
        outside = ~mask
        overlay[outside] = (frame[outside] * 0.45).astype(np.uint8)
        edge = cv2.dilate(mask.astype(np.uint8), np.ones((3, 3), np.uint8)) - mask
        overlay[edge.astype(bool)] = (255, 255, 255)
    buf = io.BytesIO()
    Image.fromarray(overlay).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode(), evidence


def _padding_mask(tfm, image: Image.Image) -> np.ndarray | None:
    """The letterbox bars this model's preprocessing adds — black BY CONSTRUCTION.

    Measured consequence of not separating these: the structured model letterboxes, so 24.7%
    of its display frame is synthetic padding, and 23.5 of its 39.7 percentage points of
    "off-retina" attention were landing on pixels that carry no information at all. Reported
    as one number that made the served model look like it reads the camera border far more
    than the head it replaced (8.3%) — which is not a comparison those two numbers can
    support, because the retired head does not letterbox.

    Derived analytically from the preprocessing geometry, never by looking for dark rows: a
    circular aperture inscribed in a rectangle produces genuinely dark rows too, and calling
    those "padding" would erase exactly the camera surround this is meant to isolate.
    """
    pre = getattr(tfm, "transforms", [None])[0]
    if not getattr(pre, "letterbox", False):
        return None
    from models.common.preprocessing import circle_crop
    rgb = np.array(image.convert("RGB"))
    if getattr(pre, "circle", True):
        rgb = circle_crop(rgb)
    h, w = rgb.shape[:2]
    if h == w:
        return np.zeros((DISPLAY, DISPLAY), bool)
    side = max(h, w)
    m = np.zeros((DISPLAY, DISPLAY), bool)
    if h < w:                                   # bars top and bottom
        off = (side - h) // 2
        a, b = round(off / side * DISPLAY), round((off + h) / side * DISPLAY)
        m[:a, :] = True; m[b:, :] = True
    else:                                       # bars left and right
        off = (side - w) // 2
        a, b = round(off / side * DISPLAY), round((off + w) / side * DISPLAY)
        m[:, :a] = True; m[:, b:] = True
    return m


def _cam_evidence(cam: np.ndarray, mask: np.ndarray | None,
                  pad: np.ndarray | None = None) -> dict:
    """Quantify a saliency map before anyone reads meaning into it."""
    total = float(cam.sum())
    ev: dict = {
        "note": ("Grad-CAM shows the regions whose features moved this score — it is "
                 "not a lesion outline, and a bright area is not a diagnosis."),
    }
    if total <= 0:
        ev["measurable"] = False
        return ev
    ev["measurable"] = True

    if mask is not None:
        outside = ~mask
        # Synthetic letterbox padding is not "the model reading the frame" — those pixels are
        # black by construction. Counted separately, and excluded from the headline number.
        if pad is not None:
            pad = pad & outside
            surround = outside & ~pad
            ev["on_padding_pct"] = round(float(cam[pad].sum()) / total * 100.0, 1)
            ev["padding_frac_pct"] = round(float(pad.mean()) * 100.0, 1)
        else:
            surround = outside
            ev["on_padding_pct"] = 0.0
            ev["padding_frac_pct"] = 0.0
        off = float(cam[surround].sum()) / total * 100.0
        ev["off_retina_pct"] = round(off, 1)
        ev["retina_frac_pct"] = round(float(mask.mean()) * 100.0, 1)
        # Area-normalised: 1.0 means attention is spread as if uniform over the frame; below
        # 1.0 means it prefers retina. Without this the raw % cannot be compared between two
        # models whose frames devote different areas to non-retina.
        area = float(surround.mean())
        ev["off_retina_lift"] = round(off / 100.0 / area, 2) if area > 0 else None
        # Whether the single hottest point missed the retina must be read off the RAW
        # map. Taking it from the masked copy below makes the answer "yes, inside" by
        # construction — a check that can only ever pass is not a check.
        gy, gx = np.unravel_index(int(np.argmax(cam)), cam.shape)
        ev["peak_inside_retina"] = bool(mask[gy, gx])
        inside = cam.copy(); inside[~mask] = 0.0
    else:
        inside = cam

    # Concentration: the smallest share of the frame that carries half the attention.
    # Uniform noise gives ~50%; a tight focus gives a few percent.
    flat = np.sort(inside.ravel())[::-1]
    cs = np.cumsum(flat)
    if cs[-1] > 0:
        half = int(np.searchsorted(cs, cs[-1] * 0.5)) + 1
        ev["concentration_pct"] = round(half / flat.size * 100.0, 1)

    # Peak position, described geometrically. NOT as temporal/nasal: that needs eye
    # laterality (OD/OS), which this app never asks for and must not invent.
    py, px = np.unravel_index(int(np.argmax(inside)), inside.shape)
    if mask is not None:
        ys, xs = np.nonzero(mask)
        cy, cx = ys.mean(), xs.mean()
        radius = float(np.sqrt(mask.sum() / np.pi))
        rho = float(np.hypot(py - cy, px - cx) / radius) if radius > 0 else 0.0
        ev["peak_radial_frac"] = round(rho, 2)
        ev["peak_zone"] = ("centre of the visible field" if rho < 0.34
                           else "mid-periphery" if rho < 0.67 else "outer periphery")
        ang = (np.degrees(np.arctan2(px - cx, cy - py)) + 360.0) % 360.0
        ev["peak_clock"] = int(round(ang / 30.0)) or 12
    return ev


def assert_not_cached_cfg(cfg, where):
    """Refuse to SERVE a cached-preprocessing config.

    ``preprocess.cached: true`` drops the fundus steps from the transforms on the promise
    that the files were already pre-processed. Uploads never are, so serving such a config
    would feed the model un-preprocessed images without any error.
    """
    from models.common.preprocessing import cache_root_from_cfg
    if cache_root_from_cfg(cfg) is not None:
        raise RuntimeError(f"{where}: preprocess.cached is true; a cached-preprocessing "
                           f"config cannot be served (uploads are not pre-processed)")


class StagingPreview:
    """LEGACY (not served since 1 Nov 2026): the EfficientNetV2-S structured staging model.

    Replaced as THE staging + screening model by ``StagingEnsemble`` (DINOv2 ViT-S/14). Kept
    loadable for side-by-side comparison only — ``Registry`` builds it when the environment sets
    ``RETINAI_LEGACY_STAGING=1`` and stores it as ``registry.legacy_staging``; no route reads it.

    6-class ICROP staging — RESEARCH PREVIEW, not a clinical output.

    Serves the clinically-structured staging model (ordinal CORN stages + AP-ROP branch;
    the site-adversary branch is training-time machinery and is inert here). The checkpoint
    is one cross-validation fold's model (fold 0, seed 42); 5-fold CV puts the family at
    macro-F1 0.692 ± 0.086, equivalent to a flat softmax — it is served for its
    taxonomy-faithful outputs, and every response says "research preview" because the locked
    external test has deliberately never been opened.

    Fails soft: if the checkpoint or config is absent the app runs without the preview.
    """

    CLASSES = ["Normal", "Stage 1", "Stage 2", "Stage 3", "Stage 4/5", "AP-ROP"]
    NOTE = ("Stage breakdown of the SAME model that produced the screening result above — "
            "not a second opinion. The stage label is a research output: cross-validated, "
            "single-seed, and not a clinical grade.")

    def __init__(self, device):
        self.loaded = False
        self.screening = None
        cfg_path = Path("configs/rop_staging_structured.yaml")
        weights = Path("results/rop_staging/weights.pth")
        if not (cfg_path.exists() and weights.exists()):
            return
        try:
            from models.common.structured import StructuredROPModel, decode_6class
            cfg = load_config(str(cfg_path))
            assert_not_cached_cfg(cfg, str(cfg_path))
            self.cfg = cfg
            self.tfm = build_transforms(cfg, "val")
            state = torch.load(weights, map_location=device)
            n_sites = state["site_head.3.bias"].shape[0]
            model = StructuredROPModel(cfg.model.arch, n_sites=n_sites, pretrained=False)
            model.load_state_dict(state, strict=True)
            self.model = model.to(device).eval()
            self.decode = decode_6class
            self.device = device
            self.thr = float(cfg.train.get("arop_threshold", 0.5))
            self.base_tfm = self.tfm            # for the shared Grad-CAM helpers
            self.arch = cfg.model.arch
            # The re-basing artifact. Its presence is what promotes this model from a
            # preview to the product: without it there is no vetted threshold and the
            # binary verdict stays with the retired ResNet50 head.
            p = Path("results/rop/screening_rebase.json")
            self.screening = json.loads(p.read_text()) if p.exists() else None
            self.loaded = True
            print("[webapp] structured ROP model loaded"
                  + (f" | SCREENING re-based, threshold {self.screening['threshold']}"
                     if self.screening else " | staging preview only (no rebase artifact)"))
        except Exception as e:                       # pragma: no cover - defensive boot
            print(f"[webapp] structured model unavailable: {e}")

    @torch.no_grad()
    def probs(self, image: Image.Image):
        """One forward pass, reused by both the screening verdict and the stage card."""
        t = self.tfm(image).unsqueeze(0).to(self.device)
        pred6, prob6 = self.decode(self.model(t), arop_threshold=self.thr)
        return prob6[0].cpu().numpy(), int(pred6[0])

    def predict(self, image: Image.Image, probs=None, idx=None):
        if not self.loaded:
            return None
        if probs is None:
            probs, idx = self.probs(image)
        return {
            "prediction": self.CLASSES[idx],
            "scores": {c: round(float(probs[i]) * 100, 1)
                       for i, c in enumerate(self.CLASSES)},
            "arop_probability": round(float(probs[5]) * 100, 1),
            # The argmax label alone would be misleading beside a positive verdict: a
            # "Normal" winning on 60.9% still carries 39.1% on the disease classes. This is
            # also the exact quantity the screening decision is now made on, so the two
            # cards can no longer appear to contradict each other.
            "any_rop_probability": round(float(probs[1:].sum()) * 100, 1),
            "preview": True,
            "note": self.NOTE,
        }

    def screening_finding(self, probs) -> dict | None:
        """The binary ROP verdict, decided on P(any ROP).

        Replaces the ResNet50 screening head, which an external audit found degenerate at
        the held-out hospital: at its 0.1933 threshold it flagged 663 of 663 images because
        the site's whole score distribution sits above the cut-off (external AUC 0.691 vs
        this model's 0.821). Threshold and both sites' measured behaviour come from
        scripts/rop_rebase_screening.py — never hardcoded here.
        """
        if not self.screening:
            return None
        s = self.screening
        thr = float(s["threshold"])
        any_rop = float(probs[1:].sum())
        positive = any_rop > thr
        risk, rec = recommendation("ROP", 1 if positive else 0)
        internal, external = s["new_model"]["internal_oof"], s["new_model"]["external_dev"]
        return {
            "disease": "ROP", "available": True, "status": "ok",
            "prediction": "ROP Detected" if positive else "No ROP",
            "grade": 1 if positive else 0,
            "score": round(any_rop * 100, 1),
            "scores": {"No ROP": round((1 - any_rop) * 100, 1),
                       "ROP Detected": round(any_rop * 100, 1)},
            "risk": risk, "recommendation": rec,
            # No temperature was fitted for this head, and none is claimed. The bootstrap
            # in structured_bootstrap_calibration.json covers staging metrics, not the
            # binary score's calibration, so "unverified" is the accurate badge.
            "calibration": {"status": "unverified", "applied": False,
                            "note": "No calibration has been fitted for this binary score. "
                                    "Read it as a ranking, not a probability of disease."},
            "decision": {
                "score": round(any_rop * 100, 1),
                "threshold": round(thr * 100, 2),
                "positive": positive,
                "positive_class": "ROP Detected",
                "note": "Screened positive when the score exceeds the decision line.",
                "model": "structured staging model · P(any ROP)",
                "measured": {
                    "sensitivity": round(internal["sensitivity"] * 100, 1),
                    "specificity": round(internal["specificity"] * 100, 1),
                    "false_alarm": round((1 - internal["specificity"]) * 100, 1),
                    "n": internal["n"],
                },
                "external": {
                    "n": external["n"],
                    "site": "held-out hospital (dev half)",
                    "sensitivity": round(external["sensitivity"] * 100, 1),
                    "specificity": round(external["specificity"] * 100, 1),
                    "false_alarm": round((1 - external["specificity"]) * 100, 1),
                    "auc": s["new_model"]["auc_external_dev"],
                    "selection_caveat": True,
                },
            },
            "uncertain": False,
        }

    def gradcam(self, image: Image.Image, class_idx: int):
        """Grad-CAM on the head that actually produced the verdict.

        Explaining a decision with a different model's saliency is worse than showing none.
        The target is P(any ROP) — the quantity the screening decision turns on — not one
        stage logit.
        """
        if not self.loaded:
            return None, None
        return _gradcam_core(
            self.model, get_target_layer(self.model.backbone, self.arch), self.tfm,
            self.device, image,
            lambda out: torch.log(self.decode(out, arop_threshold=self.thr)[1][0, 1:].sum()
                                  + 1e-9))


DEPLOY_CFG = "configs/rop_staging_deploy.yaml"


def _dotted(d, key: str):
    for part in key.split("."):
        d = d[part]
    return d


class StagingEnsemble:
    """THE served ROP model: the 6-class ICROP stage AND the binary screening verdict.

    Arm B of docs/paper/PROTOCOL_V2.md, DINOv2 ViT-S/14 with registers (392 px, ImageNet
    statistics, circle crop + letterbox + CLAHE, 6-way head), served as the per-image MEAN of
    the softmax vectors of the five cross-validation-fold checkpoints of seed 42
    (configs/rop_staging_deploy.yaml). The configuration is fixed by the protocol; the new
    hospital's development half selected nothing and its numbers are descriptive.

    Screening: any-ROP score = 1 - p(Normal) of the ensemble, flagged when it REACHES the
    pre-registered CV operating point of arm B seed 42 (results/rop/v3/v3_scores.json; the rule
    that produced it flags `score >= t`). The re-basing threshold in screening_rebase.json is NOT
    reused: it was chosen for the EfficientNet model's scores, and on this ensemble it flags
    every one of the new hospital's 150 normal development images.

    Fails soft like the model it replaces: a missing config, checkpoint or operating point
    leaves `loaded` (or `screening`) unset and the app still boots and says so.
    """

    CLASSES = StagingPreview.CLASSES
    NOTE = ("Stage breakdown of the SAME five-model ensemble that produced the screening result "
            "above — not a second opinion. The stage label is a research output, not a "
            "clinical grade: at a hospital the models never trained on it agreed with the "
            "reference eye stage 64% of the time (QWK 0.752, 230 eyes).")

    def __init__(self, device, cfg_path: str = DEPLOY_CFG):
        self.loaded = False
        self.screening = None
        self.card = None
        self.load_seconds = None
        cfg_path = Path(cfg_path)
        if not cfg_path.exists():
            return
        try:
            import time
            t0 = time.perf_counter()
            cfg = load_config(str(cfg_path))
            assert_not_cached_cfg(cfg, str(cfg_path))
            dep = cfg.deploy
            files = [Path(p) for p in dep.weights]
            missing = [str(p) for p in files if not p.exists()]
            if missing:
                print(f"[webapp] staging ensemble unavailable: missing {missing} "
                      f"(run scripts/deploy_rop_v3_weights.py)")
                return
            self.cfg, self.device, self.arch = cfg, device, cfg.model.arch
            self.tfm = build_transforms(cfg, "val")
            self.base_tfm = self.tfm             # for the shared Grad-CAM helpers
            self.members = []
            for f in files:
                m = build_from_cfg(cfg, pretrained=False)
                m.load_state_dict(torch.load(f, map_location="cpu", weights_only=True),
                                  strict=True)
                self.members.append(m.to(device).eval())
            self.calibration = self._calibration(dep.scores)
            self.screening = self._operating_point(dep)
            self.card = self._card(dep)
            # One forward + backward per member, so the first upload does not pay for the
            # backend's kernel compilation (about 1.5 s on MPS). Counted in load_seconds.
            self._token_cam(torch.zeros(1, 3, *self._size(), device=device))
            self.load_seconds = round(time.perf_counter() - t0, 2)
            self.loaded = True
            print(f"[webapp] staging ensemble loaded: {self.card['name']} on {device} in "
                  f"{self.load_seconds}s"
                  + (f" | screening line {self.screening['threshold']:.4f} (CV operating point)"
                     if self.screening else " | no operating point: staging only"))
        except Exception as e:                       # pragma: no cover - defensive boot
            print(f"[webapp] staging ensemble unavailable: {e}")

    def _size(self):
        s = int(self.cfg.preprocess.image_size)
        return s, s

    @staticmethod
    def _calibration(scores) -> dict:
        ece = float(scores.ece_eye)
        return {"status": "measured-miscalibrated" if ece >= DiseaseModel.MATERIAL_ECE
                else "verified-uncalibrated",
                "applied": False, "temperature": None, "ece": ece,
                "note": "No temperature is applied to the ensemble. ECE measured on the eye's "
                        "driving-photo probabilities at the new hospital (development half).",
                "warning": f"Measured ECE {ece:.3f} at a hospital the models never trained on. "
                           f"Read the percentages as relative scores, not probabilities."}

    def _operating_point(self, dep):
        op = dep.operating_point
        p = Path(op.source)
        if not p.exists():
            print(f"[webapp] {op.source} missing: no screening verdict from the ensemble")
            return None
        scores = json.loads(p.read_text())
        block = _dotted(scores, op.key.rsplit(".", 2)[0])      # ...operating_point
        return {"threshold": float(_dotted(scores, op.key)), "rule": block["rule"],
                "source": op.source, "key": op.key,
                "cv_any_rop": _dotted(scores, op.key.rsplit(".", 1)[0])["any_rop"],
                "external": dep.scores.to_dict()}

    def _card(self, dep) -> dict:
        s = dep.scores.to_dict()
        lo, hi = s["single_checkpoint_qwk_5class"]
        return {
            "name": dep.model_name, "arm": dep.arm, "seed": int(dep.seed),
            "ensemble": dep.ensemble, "n_members": len(self.members),
            "architecture": "DINOv2 ViT-S/14 with 4 register tokens, 392 px input, 6-way head",
            "preprocessing": "circle crop, letterbox to square, CLAHE, resize 392, ImageNet "
                             "normalisation",
            "selection": " ".join(str(dep.selection).split()),
            "scores": s,
            "caveat": (f"Eye level at a hospital the models never trained on "
                       f"({s['n_eyes']} eyes, development half): QWK {s['qwk_5class']:.3f}, "
                       f"accuracy {s['accuracy'] * 100:.1f}%, macro-F1 {s['macro_f1']:.3f}; "
                       f"single checkpoints {lo:.3f}–{hi:.3f}. The locked half is unopened."),
            "threshold": self.screening["threshold"] if self.screening else None,
            "status": "Research prototype, not a medical device.",
        }

    @torch.no_grad()
    def member_probs(self, image: Image.Image) -> np.ndarray:
        """(n_members, 6) softmax vectors, one row per fold checkpoint."""
        x = self.tfm(image).unsqueeze(0).to(self.device)
        return torch.cat([m(x).float().softmax(1) for m in self.members]).cpu().numpy()

    def probs(self, image: Image.Image):
        """The ensemble's 6-class vector (mean over members) and its argmax."""
        p = self.member_probs(image).mean(0)
        return p, int(p.argmax())

    def predict(self, image: Image.Image, probs=None, idx=None):
        if not self.loaded:
            return None
        if probs is None:
            probs, idx = self.probs(image)
        return {
            "prediction": self.CLASSES[idx],
            "scores": {c: round(float(probs[i]) * 100, 1)
                       for i, c in enumerate(self.CLASSES)},
            "arop_probability": round(float(probs[5]) * 100, 1),
            "any_rop_probability": round(float(1.0 - probs[0]) * 100, 1),
            "model": self.card["name"],
            "preview": True,
            "note": self.NOTE,
        }

    def screening_finding(self, probs) -> dict | None:
        """The binary ROP verdict: 1 - p(Normal) against the CV operating point."""
        if not self.screening:
            return None
        s = self.screening
        thr = float(s["threshold"])
        any_rop = float(1.0 - probs[0])
        positive = any_rop >= thr
        risk, rec = recommendation("ROP", 1 if positive else 0)
        cv, ext = s["cv_any_rop"], s["external"]
        out = {
            "disease": "ROP", "available": True, "status": "ok",
            "prediction": "ROP Detected" if positive else "No ROP",
            "grade": 1 if positive else 0,
            "score": round(any_rop * 100, 1),
            "scores": {"No ROP": round((1 - any_rop) * 100, 1),
                       "ROP Detected": round(any_rop * 100, 1)},
            "risk": risk, "recommendation": rec,
            "calibration": self.calibration,
            "decision": {
                # 3 dp: this line sits at ~98.94%, so 1 dp would print a score just under it
                # as "98.9% vs 98.9%".
                "score": round(any_rop * 100, 3),
                "threshold": round(thr * 100, 3),
                "positive": positive,
                "positive_class": "ROP Detected",
                "note": "Screened positive when the score reaches the decision line.",
                "model": f"{self.card['name']} · staging ensemble · 1 − P(Normal)",
                "rule": s["rule"],
                "measured": {
                    "unit": "infant eye per session, cross-validation out-of-fold, "
                            "single checkpoints",
                    "sensitivity": round(cv["sensitivity"] * 100, 1),
                    "specificity": round(cv["specificity"] * 100, 1),
                    "false_alarm": round((1 - cv["specificity"]) * 100, 1),
                    "n": cv["n_pos"] + cv["n_neg"],
                },
                "external": {
                    "unit": "eye (max over its photographs)",
                    "n": ext["n_eyes"],
                    "site": ext["site"],
                    "sensitivity": round(ext["any_rop_sensitivity"] * 100, 1),
                    "specificity": round(ext["any_rop_specificity"] * 100, 1),
                    "false_alarm": round((1 - ext["any_rop_specificity"]) * 100, 1),
                    "severe_sensitivity": round(ext["severe_sensitivity"] * 100, 1),
                    "single_photo_sensitivity": round(ext["any_rop_sensitivity_image"] * 100, 1),
                    "selection_caveat": False,
                },
            },
            "uncertain": False,
        }
        # The line was set for >= 90% sensitivity with almost no false alarms on CV; at the
        # new hospital it caught 66% of ROP eyes. A photo whose most likely stage is a disease
        # stage but whose score stays under the line is exactly that gap: say so.
        idx = int(np.argmax(probs))
        if not positive and idx > 0:
            out["uncertain"] = True
            out["uncertain_reason"] = (
                f"The most likely stage is {self.CLASSES[idx]} ({probs[idx] * 100:.1f}%), but "
                f"the any-ROP score {any_rop * 100:.1f}% is under the screening line "
                f"{thr * 100:.2f}%. That line missed about a third of ROP eyes at a hospital "
                f"the models never trained on. Needs human review.")
        return out

    def _token_cam(self, x: torch.Tensor) -> np.ndarray:
        """Grad-CAM of log P(any ROP) of the ENSEMBLE on the last block's patch tokens.

        Each member's `blocks[-1]` output (1, prefix + 28*28, 384) is hooked; the CLS and register
        tokens are dropped (the avg-pool head never reads them). Treating the five members' token
        features as one layer: channel weights = gradient averaged over the patch tokens,
        cam = ReLU(sum over members and channels of weight * activation), reshaped to the
        28x28 patch grid.
        """
        acts = []
        hooks = [m.blocks[-1].register_forward_hook(lambda mod, i, o: acts.append(o))
                 for m in self.members]
        try:
            with torch.enable_grad():
                p_normal = torch.stack([m(x).float().softmax(1)[0, 0]
                                        for m in self.members]).mean()
                grads = torch.autograd.grad(torch.log(1.0 - p_normal + 1e-9), acts)
        finally:
            for h in hooks:
                h.remove()
        cam = 0.0
        for m, a, g in zip(self.members, acts, grads):
            k = m.num_prefix_tokens
            a, g = a[0, k:].detach().float(), g[0, k:].float()     # (784, 384) patch tokens
            cam = cam + a @ g.mean(0)
        gh, gw = self.members[0].patch_embed.grid_size
        return torch.relu(cam).reshape(gh, gw).cpu().numpy().astype(np.float32)

    def cam_map(self, image: Image.Image) -> np.ndarray:
        """The painted map: DISPLAY x DISPLAY on the model's frame, [0, 1], 0 off the retina."""
        x = self.tfm(image).unsqueeze(0).to(self.device)
        return _retina_only(_cam_upsample(self._token_cam(x)),
                            retina_mask(_model_frame(self.tfm, image)))

    def gradcam(self, image: Image.Image, class_idx: int | None = None):
        """Grad-CAM of the ensemble on the quantity the verdict turns on.

        The target is log P(any ROP) = log(1 - mean p(Normal)), not one stage logit, so
        `class_idx` is accepted for interface parity and ignored (as the model it replaces did).
        """
        if not self.loaded:
            return None, None
        x = self.tfm(image).unsqueeze(0).to(self.device)
        return _cam_render(self._token_cam(x), self.tfm, image)


class DiseaseModel:
    def __init__(self, disease, cfg, weights, device):
        self.disease = disease
        self.cfg = cfg
        self.device = device
        self.class_names = list(cfg.data.class_names)
        self.arch = cfg.model.arch
        self.base_tfm = build_transforms(cfg, "val")
        self.tta = build_tta_transforms(cfg) if bool(cfg.tta.get("enabled", True)) else [self.base_tfm]
        self.model = build_from_cfg(cfg, pretrained=False).to(device)
        self.calibration = self._calibration_state()
        self.abstention = self._load_json("abstention_band.json")
        # The measured consequence of the served threshold. Shipped with every positive
        # because "ROP Detected" on a healthy eye is not a malfunction here — it is what
        # 0.398 specificity looks like from the operator's chair, and a verdict that does
        # not carry its own false-alarm rate will be read as a diagnosis.
        self.operating_point = self._load_json("operating_point.json")
        # How the served threshold behaves at a hospital it was NOT tuned on
        # (scripts/rop_screening_external_spec.py). Loaded because the internal
        # specificity, quoted alone, is the more flattering of two measured numbers.
        self.external = self._load_json("screening_external_spec.json")
        self.loaded = False
        if Path(weights).exists():
            self.model.load_state_dict(torch.load(weights, map_location=device))
            self.model.eval()
            self.loaded = True

    # ── calibration + abstention state (Phase 2) ─────────────────────────────
    def _load_json(self, name):
        p = Path("results") / self.disease.lower() / name
        if not p.exists():
            return None
        try:
            return json.loads(p.read_text())
        except Exception:
            return None

    # A model can be left uncorrected for two OPPOSITE reasons: it was already fine, or the
    # only correction available made something else worse. Reporting both as
    # 'verified-uncalibrated' would have shown DR (ECE 0.191) with the same badge as
    # Glaucoma (ECE 0.017) — the string reads as reassurance in one case and as a warning in
    # the other, so it cannot be the same string. 0.10 is a convention, not a derived
    # quantity; the measured ECE travels alongside so no consumer has to trust the bucket.
    MATERIAL_ECE = 0.10

    def _calibration_state(self):
        """What we actually know about this model's confidence, stated honestly.

        'unverified' is NOT the same as 'calibrated' — the UI must be able to tell the
        difference, so the status is carried through to the API rather than inferred.
        """
        c = self._load_json("calibration_serving.json")
        if c is None:
            return {"status": "unverified", "applied": False, "temperature": None,
                    "note": "Calibration has not been measured against the serving pipeline "
                            "for this disease. Displayed scores are raw model output."}
        if c.get("apply"):
            return {"status": "calibrated", "applied": True,
                    "temperature": float(c["temperature"]),
                    "ece": c["calibrated"]["ece"], "note": c.get("decision_reason", "")}
        ece = float(c["uncalibrated"]["ece"])
        status = ("measured-miscalibrated" if ece >= self.MATERIAL_ECE
                  else "verified-uncalibrated")
        state = {"status": status, "applied": False,
                 "temperature": float(c["temperature"]), "ece": ece,
                 "note": c.get("decision_reason", "")}
        if status == "measured-miscalibrated":
            state["warning"] = (
                f"Measured ECE {ece:.3f} at serving. Temperature scaling was rejected "
                f"because it degrades worst-case calibration. Ranking and all decisions are "
                f"unaffected — treat the displayed percentage as a relative score, not as a "
                f"probability.")
        # The clinically meaningful DR endpoint is 'referable' (grade >= 2), not the 5-class
        # argmax, and it calibrates differently. Carry it rather than let the 5-class verdict
        # speak for both.
        if isinstance(c.get("referable"), dict):
            r = c["referable"]
            state["referable"] = {
                "apply": bool(r.get("apply")), "temperature": float(r["temperature"]),
                "ece_uncalibrated": float(r["uncalibrated"]["ece"]),
                "ece_calibrated": float(r["calibrated"]["ece"]),
                "note": r.get("decision_reason", "")}
        return state

    def _apply_calibration(self, probs):
        if not self.calibration.get("applied"):
            return probs
        T = self.calibration["temperature"]
        z = np.log(np.clip(probs.astype(np.float64), 1e-12, 1.0)) / T
        z -= z.max()
        e = np.exp(z)
        return (e / e.sum()).astype(probs.dtype)

    def _decisive_score(self, probs):
        """The probability the screening decision actually turns on."""
        if len(self.class_names) == 2:
            return float(probs[1]), float(self.cfg.eval.get("screen_threshold", 0.5))
        ref = self.cfg.eval.get("referable_grade")
        if ref is not None:
            return float(probs[int(ref):].sum()), 0.5
        return float(probs.max()), 0.0

    def predict(self, image: Image.Image):
        if not self.loaded:
            return {"disease": self.disease, "available": False,
                    "note": "model not loaded (train on Kaggle first)"}
        probs = predict_with_tta(self.model, image, self.tta, self.device,
                                 self.cfg.model.get("head", "classification")).numpy()
        probs = self._apply_calibration(probs)

        # binary screening threshold (high sensitivity) for 2-class diseases
        if len(self.class_names) == 2:
            thr = float(self.cfg.eval.get("screen_threshold", 0.5))
            idx = 1 if probs[1] > thr else 0
        else:
            idx = int(probs.argmax())
        risk, rec = recommendation(self.disease, idx)

        score, dthr = self._decisive_score(probs)
        out = {
            "disease": self.disease, "available": True,
            "prediction": self.class_names[idx], "grade": idx,
            # NOT "confidence": this is a model score, and for most diseases it is not
            # calibrated. See `calibration.status`.
            "score": round(float(probs[idx]) * 100, 1),
            "risk": risk, "recommendation": rec,
            "scores": {self.class_names[i]: round(float(probs[i]) * 100, 1)
                       for i in range(len(self.class_names))},
            "calibration": self.calibration,
        }

        # The decision line, shipped WITH the score. Field-reported confusion: the UI drew
        # "No ROP 43.7% / ROP Detected 56.3%" as two competing bars, which every reader
        # interprets against a 50% boundary. This model's boundary is 19.33% — tuned on val
        # for >=0.90 sensitivity, measured 0.990 sens / 0.398 spec on test. A score of 56.3%
        # is therefore ~3x the decision line, not a near-tie, and the complement 43.7% is
        # arithmetic (1 - score), not a second opinion that the eye is healthy. A threshold
        # the client cannot see is a threshold the client will silently assume.
        out["decision"] = {
            "score": round(score * 100, 1),
            # 2 dp, not 1: the threshold is a fixed constant (19.33%), and rounding it to
            # 19.3% would let a score of 19.31% render as "above the 19.3% line" by 0.01.
            "threshold": round(dthr * 100, 2),
            "positive": bool(score > dthr),
            "positive_class": self.class_names[-1] if len(self.class_names) == 2 else None,
            "note": ("Screened positive when the score exceeds the decision line."
                     if len(self.class_names) == 2 else None),
        }
        op = (self.operating_point or {}).get("test") or {}
        if len(self.class_names) == 2 and op.get("specificity") is not None:
            out["decision"]["measured"] = {
                "sensitivity": round(float(op["sensitivity"]) * 100, 1),
                "specificity": round(float(op["specificity"]) * 100, 1),
                "false_alarm": round((1.0 - float(op["specificity"])) * 100, 1),
                "n": op.get("n"),
            }
            ext = self.external or {}
            sop = (ext.get("served_operating_point") or {}).get("screening") or {}
            if sop:
                out["decision"]["external"] = {
                    "n": ext["eval_set"]["n"],
                    "site": ext["eval_set"]["site"],
                    "specificity": round(float(sop["specificity"]) * 100, 1),
                    "flagged_pct": round(float(sop["flagged_frac"]) * 100, 1),
                    "auc": ext["auc"]["screening"],
                    # the staging head measured on the SAME external images, so the card
                    # can say which of the two to believe instead of just asserting one
                    "staging_auc": ext["auc"].get("staging_any_rop_fold0"),
                }

        # C5 — for an ordinal head the clinically meaningful endpoint is referable disease,
        # not the argmax grade. Report it explicitly instead of leaving it implied.
        ref = self.cfg.eval.get("referable_grade")
        if len(self.class_names) > 2 and ref is not None:
            out["referable"] = {
                "grade_threshold": int(ref),
                "probability": round(float(probs[int(ref):].sum()) * 100, 1),
                "is_referable": bool(idx >= int(ref)),
            }

        # C6 — abstention band, derived on val (scripts/fit_abstention_band.py)
        band = self.abstention
        if band and band.get("enabled") and band["lo"] <= score <= band["hi"]:
            out["uncertain"] = True
            out["uncertain_reason"] = (
                f"Score {score*100:.1f}% falls in the low-reliability band "
                f"{band['lo']*100:.1f}–{band['hi']*100:.1f}%, where validation accuracy is "
                f"{band['acc_inside']*100:.0f}% (vs {band['acc_outside']*100:.0f}% outside). "
                f"Needs human review.")
        else:
            out["uncertain"] = False
        return out

    def gradcam(self, image: Image.Image, class_idx: int):
        """Grad-CAM for one class logit of this plain classifier."""
        if not self.loaded:
            return None, None
        return _gradcam_core(self.model, get_target_layer(self.model, self.arch),
                             self.base_tfm, self.device, image,
                             lambda out: out[0, class_idx])

    # kept as a class attribute: tests drive it directly to prove the peak check can
    # answer False, and that is a property of the function, not of any one model.
    _cam_evidence = staticmethod(_cam_evidence)



class EmbeddingGate:
    """Third gradability layer: kNN distance to real fundus in IMAGENET feature space.

    The two heuristic layers (colour, structure) each fell to a real-world image class in
    field testing - a furniture photo, then a night-time interior. This layer asks the
    only robust question: does the upload even sit near the fundus distribution? Generic
    ImageNet features are used deliberately - the fine-tuned screening/staging backbones
    were measured to COLLAPSE scene information (a warm room lands nearer to fundus in
    their space than some real retinas), while ImageNet features still know what rooms,
    lamps and furniture look like. Built by scripts/build_embedding_gate.py; fails OPEN
    if the artifact or torchvision weights are unavailable (the heuristics still stand).
    """

    def __init__(self, device, tfm):
        self.available = False
        art = Path("results/rop/gate_embedding.npz")
        if not art.exists():
            print("[webapp] embedding gate: artifact missing - layer disabled")
            return
        try:
            import torchvision
            z = np.load(art, allow_pickle=True)
            self.bank = z["bank"].astype(np.float32)
            self.k = int(z["k"])
            self.threshold = float(z["threshold"])
            net = torchvision.models.resnet50(weights="IMAGENET1K_V2")
            net.fc = torch.nn.Identity()
            self.net = net.to(device).eval()
            self.device, self.tfm = device, tfm
            self.available = True
            print(f"[webapp] embedding gate loaded (bank {self.bank.shape[0]}, "
                  f"threshold {self.threshold:.3f})")
        except Exception as e:                                    # noqa: BLE001
            print(f"[webapp] embedding gate unavailable ({e}) - layer disabled")

    def distance(self, image):
        x = self.tfm(image.convert("RGB")).unsqueeze(0).to(self.device)
        with torch.no_grad():
            z = self.net(x).flatten(1).float().cpu().numpy()[0]
        z /= np.linalg.norm(z) + 1e-9
        sims = self.bank @ z
        return float(1.0 - np.sort(sims)[-self.k])

    def check(self, image):
        """None if the image resembles retinal imagery; else a rejection dict."""
        if not self.available:
            return None
        d = self.distance(image)
        if d <= self.threshold:
            return None
        return {"verdict": "not_fundus",
                "reason": "This image does not resemble retinal imagery (feature-space "
                          "check: its content is unlike any fundus photograph the gate "
                          "was calibrated on).",
                "metrics": {"embedding_distance": round(d, 4),
                            "threshold": round(self.threshold, 4)}}


class EyeNotGradable(ValueError):
    """Every photograph of the eye was rejected by the gate (or carried no stage)."""


class StagingUnavailable(RuntimeError):
    """The eye route cannot run at all: no stage is produced for this context / build.

    `status` is the HTTP code the route should answer with: 400 when the patient context
    does not run ROP (caller error), 503 when the staging model is not loaded (server).
    """
    def __init__(self, message, status):
        super().__init__(message)
        self.status = status


EYE_LIMITATION = ("acquisition view is not detected — make sure the temporal periphery "
                  "was photographed")


def eye_from_photos(per_photo, laterality):
    """The worst-view rule: ICROP grades an eye by its most severe finding.

    Eye stage = max predicted 6-class index over the photographs that passed the gate. The
    driving photo is the first of the max-stage photos by upload order after preferring the
    highest probability for that class; prob6 is its vector. Rejected photos are excluded
    and named in `warnings`.
    """
    classes = StagingPreview.CLASSES
    warnings, cands = [], []
    for i, p in enumerate(per_photo):
        st = p.get("staging") if p.get("gradable", True) else None
        if st is None:
            why = (p.get("quality") or {}).get("reason") or "no stage prediction"
            warnings.append(f"photo {i + 1} not gradable: {why}")
            continue
        idx = classes.index(st["prediction"])
        prob6 = [float(st["scores"][c]) / 100.0 for c in classes]
        cands.append((idx, prob6[idx], i, prob6))
    if not cands:
        raise EyeNotGradable("no photograph of this eye could be graded: "
                             + "; ".join(warnings))
    top = max(c[0] for c in cands)
    idx, _, drv, prob6 = max((c for c in cands if c[0] == top), key=lambda c: (c[1], -c[2]))
    warnings.append(EYE_LIMITATION)
    return {"eye": {"stage_index": idx, "stage": classes[idx], "driving_photo": drv,
                    "prob6": prob6},
            "laterality": laterality, "warnings": warnings}


class RegistryConfigError(RuntimeError):
    """Raised at boot when the routing table is missing or inconsistent.

    Deliberately fatal: an un-routed build must not be deployable by accident, because
    without routing every model runs on every patient (see docs/REMEDIATION_PLAN.md 0A).
    """


class Registry:
    def __init__(self, registry_path="webapp/registry.yaml"):
        self.device = _device()
        spec = yaml.safe_load(Path(registry_path).read_text())
        self.models = []
        for m in spec["models"]:
            cfg = load_config(m["config"])
            assert_not_cached_cfg(cfg, m["config"])
            dm = DiseaseModel(m["disease"], cfg, m["weights"], self.device)
            dm.applies_to = list(m.get("applies_to") or [])
            self.models.append(dm)
        self.contexts = self._validate_contexts(spec, registry_path)
        # THE staging + screening model (see StagingEnsemble). The EfficientNet structured model
        # it replaced stays loadable for comparison only, never served.
        self.staging = StagingEnsemble(self.device,
                                       (spec.get("staging") or {}).get("config", DEPLOY_CFG))
        self.legacy_staging = (StagingPreview(self.device)
                               if os.environ.get("RETINAI_LEGACY_STAGING") == "1" else None)
        rop = next((m for m in self.models if m.disease == "ROP"), self.models[0])
        # NOTE: the gate is an ImageNet ResNet-50 and must keep ImageNet normalisation. If a
        # ViT with `preprocess.normalize: half` is ever served, do NOT hand the gate that
        # model's transform: build it with ImageNet mean/std, as the gate bank was.
        self.embedding_gate = EmbeddingGate(self.device, rop.base_tfm)
        loaded = [m.disease for m in self.models if m.loaded]
        print(f"[webapp] device {self.device} | loaded: {loaded or 'none yet'} "
              f"| contexts: {', '.join(sorted(self.contexts))}")

    def _validate_contexts(self, spec, path):
        """Fail closed if routing is absent or names anything that does not exist."""
        contexts = spec.get("contexts")
        if not contexts or not isinstance(contexts, dict):
            raise RegistryConfigError(
                f"{path}: no 'contexts' routing table. Every model would run on every "
                f"patient. See docs/REMEDIATION_PLAN.md Phase 0A.")
        known = {m.disease for m in self.models}
        for name, spec_ in contexts.items():
            runs = (spec_ or {}).get("runs")
            if not runs:
                raise RegistryConfigError(f"{path}: context '{name}' declares no 'runs'.")
            unknown = [d for d in runs if d not in known]
            if unknown:
                raise RegistryConfigError(
                    f"{path}: context '{name}' routes to unknown disease(s) {unknown}. "
                    f"Known: {sorted(known)}")
        # every model must declare the contexts it was trained for, and they must agree
        # with the routing table — this catches a model added to a context by mistake.
        for dm in self.models:
            if not dm.applies_to:
                raise RegistryConfigError(f"{path}: model '{dm.disease}' has no 'applies_to'.")
            bad = [c for c in dm.applies_to if c not in contexts]
            if bad:
                raise RegistryConfigError(
                    f"{path}: model '{dm.disease}' applies_to unknown context(s) {bad}.")
            for cname, cspec in contexts.items():
                if dm.disease in (cspec or {}).get("runs", []) and cname not in dm.applies_to \
                        and not (cspec or {}).get("advisory"):
                    raise RegistryConfigError(
                        f"{path}: context '{cname}' runs '{dm.disease}', but that model only "
                        f"applies_to {dm.applies_to}. Mark the context advisory or fix the table.")
        return contexts

    def context_choices(self):
        """[(key, label, hint, advisory)] for the UI, in declaration order."""
        return [(k, v.get("label", k), v.get("hint", ""), bool(v.get("advisory")))
                for k, v in self.contexts.items()]

    def _source(self, dm):
        """The model that DECIDES `dm`'s disease: the staging ensemble for ROP when its
        operating point is available, else `dm` itself."""
        if dm.disease == "ROP" and self.staging.loaded and self.staging.screening is not None:
            return self.staging
        return dm

    def _heatmaps(self, image, findings):
        """One Grad-CAM per positive disease, from the model that produced its verdict."""
        by = {dm.disease: dm for dm in self.models}
        out = []
        for f in findings:
            if f.get("available") and f.get("grade", 0) > 0 and f["disease"] in by:
                cam, evidence = self._source(by[f["disease"]]).gradcam(image, f["grade"])
                if cam:
                    out.append({"disease": f["disease"], "image": cam, "evidence": evidence})
        return out

    def analyze(self, image: Image.Image, context: str, cams: bool = True):
        """Run only the models this patient context is routed to.

        Routed-out models still appear in `findings` with available=False and
        status='not_applicable', so the UI and the PDF can say "not assessed" rather than
        silently omitting them (an omission reads as "nothing found"). `cams=False` skips the
        Grad-CAMs (the eye route computes one, for its driving photo, afterwards).
        """
        if context not in self.contexts:
            raise ValueError(f"unknown patient context '{context}'; "
                             f"expected one of {sorted(self.contexts)}")
        cspec = self.contexts[context]
        allowed = set(cspec.get("runs", []))
        advisory = bool(cspec.get("advisory"))

        # ── gradability gate: fail CLOSED. No model sees an image that is not a gradable
        # fundus photograph. Before this existed, all three models returned a positive
        # finding on 100% of non-retinal input, including this project's own charts.
        quality = gradability(image)
        if quality["gradable"]:
            # heuristic layers passed - the learned layer gets the final word
            emb = self.embedding_gate.check(image)
            if emb is not None:
                quality = {"gradable": False, "verdict": emb["verdict"],
                           "reason": emb["reason"],
                           "metrics": {**quality["metrics"], **emb["metrics"]}}
        if not quality["gradable"]:
            return {
                "context": context,
                "context_label": cspec.get("label", context),
                "advisory": advisory,
                "gradable": False,
                "quality": {"verdict": quality["verdict"], "reason": quality["reason"],
                            "metrics": quality["metrics"]},
                "findings": [{"disease": dm.disease, "available": False,
                              "status": "ungradable",
                              "note": "Not graded — " + quality["reason"]}
                             for dm in self.models],
                "heatmaps": [], "heatmap": None, "heatmap_disease": None,
            }

        findings = []
        # pre-bound: a routed-out ROP model skips the loop body, and the staging block
        # below reads these
        rebased, probs6, idx6 = False, None, None
        for dm in self.models:
            if dm.disease not in allowed:
                findings.append({
                    "disease": dm.disease, "available": False, "status": "not_applicable",
                    "note": f"Not assessed — outside this model's population "
                            f"(trained for: {', '.join(dm.applies_to)}).",
                })
                continue
            # ROP is decided by the staging ENSEMBLE when its operating point is available.
            # The ResNet50 head below was audited at the held-out hospital and found
            # degenerate there — every one of 663 images flagged, healthy included, because
            # the whole site's score distribution sits above its 0.1933 cut-off (external
            # AUC 0.691 vs 0.821). One forward pass serves the verdict, the stage card and
            # the Grad-CAM, so the page cannot show three views of two different models.
            rebased = self._source(dm) is self.staging
            if rebased:
                probs6, idx6 = self.staging.probs(image)
                res = self.staging.screening_finding(probs6)
            else:
                res = dm.predict(image)
            res.setdefault("status", "ok" if res.get("available") else "not_loaded")
            if advisory:
                res["advisory"] = True
            findings.append(res)
        heatmaps = self._heatmaps(image, findings) if cams else []

        # The stage breakdown — the same forward pass as the verdict when re-based, so the
        # two cards report one model. Shown only when the product model actually ran; a
        # routed-out or unloaded model must not leave staging as the only voice on the page.
        staging = None
        if "ROP" in allowed and self.staging.loaded and any(
                f.get("available") for f in findings):
            staging = (self.staging.predict(image, probs6, idx6) if rebased
                       else self.staging.predict(image))

        return {
            "context": context,
            "context_label": cspec.get("label", context),
            "advisory": advisory,
            "gradable": True,
            "resolution_notice": _resolution_notice(image),
            "quality": {"verdict": quality["verdict"], "reason": "",
                        "metrics": quality["metrics"]},
            "findings": findings,
            "staging": staging,
            "heatmaps": heatmaps,
            # legacy single-heatmap fields kept so older callers keep working
            "heatmap": heatmaps[0]["image"] if heatmaps else None,
            "heatmap_disease": heatmaps[0]["disease"] if heatmaps else None,
        }

    def analyze_eye(self, images, eye: str, context: str = "infant"):
        """One eye, many photographs, one answer (see eye_from_photos)."""
        if eye not in ("OD", "OS"):
            raise ValueError(f"eye must be 'OD' or 'OS', got {eye!r}")
        if context not in self.contexts:
            raise ValueError(f"unknown patient context '{context}'")
        if "ROP" not in self.contexts[context].get("runs", []):
            raise StagingUnavailable(f"patient context '{context}' does not run the ROP "
                                     f"model, so no eye-level stage exists for it", 400)
        if not self.staging.loaded:
            raise StagingUnavailable("the staging model is not loaded on this server", 503)
        per_photo = [{**self.analyze(im, context, cams=False), "index": i}
                     for i, im in enumerate(images)]
        out = eye_from_photos(per_photo, eye)
        # Grad-CAM for the DRIVING photo only: it is the one the page shows, and a five-model
        # Grad-CAM per positive photo would put a 16-photo severe eye far past the 10 s budget.
        d = out["eye"]["driving_photo"]
        hm = self._heatmaps(images[d], per_photo[d]["findings"])
        per_photo[d].update(heatmaps=hm, heatmap=hm[0]["image"] if hm else None,
                            heatmap_disease=hm[0]["disease"] if hm else None)
        return {"per_photo": per_photo, **out}

    def summary(self):
        """Per-disease status + headline test metrics for the dashboard."""
        dirmap = {"DR": "dr", "ROP": "rop", "Glaucoma": "glaucoma"}
        out = []
        for dm in self.models:
            key = dirmap.get(dm.disease, dm.disease.lower())
            entry = {
                "disease": dm.disease, "loaded": dm.loaded, "arch": dm.arch,
                "classes": dm.class_names,
                "metrics": _load_metrics(key),
                # A headline number and the reason it might not mean what it looks like
                # belong in the SAME payload — a caveat that lives only on another page is
                # a caveat nobody reads.
                "caveat": _headline_caveat(key),
                "calibration": dm.calibration,
                "contexts": [c for c, spec in self.contexts.items()
                             if dm.disease in spec.get("runs", [])],
            }
            # The card shows the model that DECIDES. Since the ensemble took over, the ROP
            # entry above only anchors routing; its ResNet50 numbers are not the product's.
            if dm.disease == "ROP" and self.staging.loaded and self.staging.screening:
                c, s = self.staging.card, self.staging.card["scores"]
                entry.update({
                    "loaded": True, "arch": c["name"], "classes": list(self.staging.CLASSES),
                    "metrics": {"qwk_5class": s["qwk_5class"], "accuracy": s["accuracy"],
                                "macro_f1": s["macro_f1"], "n_eyes": s["n_eyes"]},
                    "caveat": c["caveat"], "calibration": self.staging.calibration,
                    "model_card": c,
                })
            out.append(entry)
        return out
