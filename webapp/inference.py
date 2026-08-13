"""Inference backend for the multi-disease web app.

Loads the three dedicated models from ``registry.yaml``, runs TTA inference using the
SAME shared preprocessing as training (so train/inference match), and produces a
generic Grad-CAM for any architecture. Degrades gracefully when a model's weights are
missing so the app runs before all three are trained.
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
from models.common.data_quality import gradability
from models.common.preprocessing import build_transforms, build_tta_transforms
from models.common.train_utils import predict_with_tta
from reports.report_generator import recommendation

DISPLAY = 320


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

    def gradcam(self, image: Image.Image, class_idx: int) -> str | None:
        if not self.loaded:
            return None
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        tensor = self.base_tfm(image).unsqueeze(0).to(self.device)
        grads, acts = [], []
        target = get_target_layer(self.model, self.arch)
        bw = target.register_full_backward_hook(lambda m, gi, go: grads.append(go[0].detach().cpu()))
        fw = target.register_forward_hook(lambda m, i, o: acts.append(o.detach().cpu()))
        self.model.zero_grad()
        out = self.model(tensor)
        out[0, class_idx].backward()
        bw.remove(); fw.remove()

        w = grads[0].mean(dim=[2, 3], keepdim=True)
        cam = torch.relu((w * acts[0]).sum(1)).squeeze().numpy()
        cam -= cam.min()
        if cam.max() > 0:
            cam /= cam.max()
        cam_img = Image.fromarray((cam * 255).astype(np.uint8)).resize((DISPLAY, DISPLAY))
        heat = (plt.get_cmap("jet")(np.array(cam_img) / 255.0)[:, :, :3] * 255).astype(np.uint8)
        orig = np.array(image.convert("RGB").resize((DISPLAY, DISPLAY)))
        overlay = (0.55 * heat + 0.45 * orig).astype(np.uint8)
        buf = io.BytesIO()
        Image.fromarray(overlay).save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode()


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
    NOTE = ("Research preview — a cross-validated staging model whose locked external test "
            "has not been opened. Not a clinical grade; the binary screening result above "
            "is the product output.")

    def __init__(self, device):
        self.loaded = False
        cfg_path = Path("configs/rop_staging_structured.yaml")
        weights = Path("results/rop_staging/weights.pth")
        if not (cfg_path.exists() and weights.exists()):
            return
        try:
            from models.common.structured import StructuredROPModel, decode_6class
            cfg = load_config(str(cfg_path))
            self.tfm = build_transforms(cfg, "val")
            state = torch.load(weights, map_location=device)
            n_sites = state["site_head.3.bias"].shape[0]
            model = StructuredROPModel(cfg.model.arch, n_sites=n_sites, pretrained=False)
            model.load_state_dict(state, strict=True)
            self.model = model.to(device).eval()
            self.decode = decode_6class
            self.device = device
            self.thr = float(cfg.train.get("arop_threshold", 0.5))
            self.loaded = True
            print("[webapp] ICROP staging research preview loaded")
        except Exception as e:                       # pragma: no cover - defensive boot
            print(f"[webapp] staging preview unavailable: {e}")

    @torch.no_grad()
    def predict(self, image: Image.Image):
        if not self.loaded:
            return None
        t = self.tfm(image).unsqueeze(0).to(self.device)
        pred6, prob6 = self.decode(self.model(t), arop_threshold=self.thr)
        probs = prob6[0].cpu().numpy()
        idx = int(pred6[0])
        return {
            "prediction": self.CLASSES[idx],
            "scores": {c: round(float(probs[i]) * 100, 1)
                       for i, c in enumerate(self.CLASSES)},
            "arop_probability": round(float(probs[5]) * 100, 1),
            "preview": True,
            "note": self.NOTE,
        }


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
        for dm in self.models:
            if dm.disease not in allowed:
                findings.append({
                    "disease": dm.disease, "available": False, "status": "not_applicable",
                    "note": f"Not assessed — outside this model's population "
                            f"(trained for: {', '.join(dm.applies_to)}).",
                })
                continue
            res = dm.predict(image)
            res.setdefault("status", "ok" if res.get("available") else "not_loaded")
            if advisory:
                res["advisory"] = True
            findings.append(res)
            # one Grad-CAM per positive disease, not one for the whole image
            if res.get("available") and res.get("grade", 0) > 0:
                cam = dm.gradcam(image, res["grade"])
                if cam:
                    heatmaps.append({"disease": dm.disease, "image": cam})

        # ICROP staging research preview — infant context only, and only when the binary
        # model actually ran (a routed-out or unloaded product model must not leave the
        # preview as the only voice on the page).
        staging = None
        if "ROP" in allowed and self.staging.loaded and any(
                f.get("available") for f in findings):
            staging = self.staging.predict(image)

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
