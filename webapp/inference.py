"""Inference backend for the ROP screening web app.

Loads the models declared in ``registry.yaml`` — this deployment declares ROP only — runs
inference through the SAME shared preprocessing as training (so train/serve match), and
produces a Grad-CAM taken from whichever model produced the verdict. Degrades gracefully
when a checkpoint is missing, so the app still boots and says so.

Since 19 Aug 2026 the binary ROP decision comes from the STRUCTURED model's P(any ROP)
rather than the ResNet50 head: an external audit measured that head flagging every image
at a hospital it never trained on. See ``StagingPreview.screening_finding``.
"""
from __future__ import annotations

import base64
import io
import json
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
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tensor = tfm(image).unsqueeze(0).to(device)
    grads, acts = [], []
    bw = target.register_full_backward_hook(lambda m, gi, go: grads.append(go[0].detach().cpu()))
    fw = target.register_forward_hook(lambda m, i, o: acts.append(o.detach().cpu()))
    model.zero_grad()
    objective(model(tensor)).backward()
    bw.remove(); fw.remove()

    w = grads[0].mean(dim=[2, 3], keepdim=True)
    cam = torch.relu((w * acts[0]).sum(1)).squeeze().numpy().astype(np.float32)
    # Upsample in float with a cubic kernel. Quantising the 12x12 map to uint8 and letting
    # PIL resize it banded the low end of the ramp into flat plateaus — visible as hard
    # steps in the blue background of the overlay.
    cam = np.clip(_resize_f32(cam, DISPLAY), 0.0, None)

    frame = _model_frame(tfm, image)
    mask = retina_mask(frame)
    evidence = _cam_evidence(cam, mask)

    # Colour ONLY the retina. Attention on the surround is real and is reported as a number
    # alongside; painting it keeps the eye on a region where the model has, by
    # construction, nothing clinical to see.
    shown = cam.copy()
    if mask is not None:
        shown[~mask] = 0.0
    rng = float(shown.max())
    shown = shown / rng if rng > 0 else shown

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


def _cam_evidence(cam: np.ndarray, mask: np.ndarray | None) -> dict:
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
        off = float(cam[~mask].sum()) / total * 100.0
        ev["off_retina_pct"] = round(off, 1)
        ev["retina_frac_pct"] = round(float(mask.mean()) * 100.0, 1)
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


class StagingPreview:
    """6-class ICROP staging — RESEARCH PREVIEW, not a clinical output.

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
            dm = DiseaseModel(m["disease"], cfg, m["weights"], self.device)
            dm.applies_to = list(m.get("applies_to") or [])
            self.models.append(dm)
        self.contexts = self._validate_contexts(spec, registry_path)
        self.staging = StagingPreview(self.device)
        rop = next((m for m in self.models if m.disease == "ROP"), self.models[0])
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

    def analyze(self, image: Image.Image, context: str):
        """Run only the models this patient context is routed to.

        Routed-out models still appear in `findings` with available=False and
        status='not_applicable', so the UI and the PDF can say "not assessed" rather than
        silently omitting them (an omission reads as "nothing found").
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

        findings, heatmaps = [], []
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
            # ROP is decided by the STRUCTURED model when a vetted threshold exists for it.
            # The ResNet50 head it replaces was audited at the held-out hospital and found
            # degenerate there — every one of 663 images flagged, healthy included, because
            # the whole site's score distribution sits above its 0.1933 cut-off (external
            # AUC 0.691 vs 0.821). One forward pass serves the verdict, the stage card and
            # the Grad-CAM, so the page cannot show three views of two different models.
            rebased = (dm.disease == "ROP" and self.staging.loaded
                       and self.staging.screening is not None)
            if rebased:
                probs6, idx6 = self.staging.probs(image)
                res = self.staging.screening_finding(probs6)
                source = self.staging
            else:
                res = dm.predict(image)
                source = dm
            res.setdefault("status", "ok" if res.get("available") else "not_loaded")
            if advisory:
                res["advisory"] = True
            findings.append(res)
            # one Grad-CAM per positive disease, not one for the whole image
            if res.get("available") and res.get("grade", 0) > 0:
                cam, evidence = source.gradcam(image, res["grade"])
                if cam:
                    heatmaps.append({"disease": dm.disease, "image": cam,
                                     "evidence": evidence})

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

    def summary(self):
        """Per-disease status + headline test metrics for the dashboard."""
        dirmap = {"DR": "dr", "ROP": "rop", "Glaucoma": "glaucoma"}
        out = []
        for dm in self.models:
            key = dirmap.get(dm.disease, dm.disease.lower())
            out.append({
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
            })
        return out
