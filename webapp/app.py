"""Flask web app — AI-assisted ROP screening (Retinopathy of Prematurity). ONE page.

Since 7 Nov 2026 the app is a single screening page at `/`: patient context, upload (one photo
or every photo of one eye), the verdict, the stage, the Grad-CAM, a PDF report — and, beside the
served model's stage, a COMPARISON model's stage (webapp/comparison.py: a teammate's
RETFound-DINOv2 + LoRA). The comparison never decides anything. The dashboard, diseases,
gallery, history and about pages were removed with the design that carried them; the model
cards they held now sit at the foot of the one page.

This deployment is ROP-ONLY: the DINOv2 ViT-S/14 5-fold ensemble (configs/
rop_staging_deploy.yaml) produces both the binary ROP screening verdict and the 6-class ICROP
stage. Patient context is mandatory and routed (see webapp/registry.yaml for why). Nothing is
retained between requests: there is no history and no gallery.

The seen-photo guard: results/comparison/seen_photos.json (scripts/deploy_comparison_weights.py)
maps the md5 of every photo in the project's data to the sets it belongs to, so an uploaded
photo that a model trained or tested on is flagged on the page. The demo is meant for photos
neither model has seen.

    python webapp/app.py   ->   http://127.0.0.1:5002
"""
import base64
import hashlib
import io
import json
import os
import sys
from pathlib import Path

from flask import Flask, jsonify, render_template, request, send_file
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root on path

from webapp.inference import EyeNotGradable, Registry, StagingUnavailable
from reports.report_generator import generate_pdf

MAX_UPLOAD = 10 * 1024 * 1024
REGISTRY = Registry(os.environ.get("REGISTRY", "webapp/registry.yaml"))

app = Flask(__name__)
# The cap covers the WHOLE multipart body, so the eye route needs room for 16 photographs.
# The per-file limit stays in _read_image; other routes keep the single-upload cap below.
MAX_EYE_PHOTOS = 16
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD * MAX_EYE_PHOTOS


def _load_seen_photos(spec) -> dict:
    """md5 -> tags for every photo in the project's data (empty when the table is absent)."""
    path = Path((spec or {}).get("seen_photos") or "")
    if not path.is_file():
        print("[webapp] seen-photo guard off: no seen_photos.json "
              "(run scripts/deploy_comparison_weights.py)")
        return {}
    table = json.loads(path.read_text())
    print(f"[webapp] seen-photo guard: {len(table):,} photos")
    return table


SEEN_PHOTOS = _load_seen_photos(REGISTRY.comparison.spec)


def seen_in_data(raw: bytes) -> dict | None:
    """The sets an uploaded photo belongs to, if it is a byte-identical copy of one we hold.

    `level` is "warn" when a model trained or was tested on the photo, "info" when the photo is
    in the project's data but outside every model's splits (e.g. Ostrava stage-0 photos, which
    is what test_images/rop holds): those are fair to demo on, and the page says so.
    """
    tags = SEEN_PHOTOS.get(hashlib.md5(raw).hexdigest())
    if not tags:
        return None
    used = [t for t in tags if not t.startswith("our data, outside")]
    if used:
        return {"level": "warn", "tags": tags,
                "message": "This exact photo is in the data a model trained or was tested on ("
                           + "; ".join(used) + "). A result on it is not a fair test — use a "
                           "new photo."}
    return {"level": "info", "tags": tags,
            "message": "This photo is in the project's data but outside every model's training "
                       "and test splits, so neither model has seen it."}


@app.before_request
def _single_upload_cap():
    if request.path != "/predict_eye" and (request.content_length or 0) > MAX_UPLOAD + 65536:
        return jsonify({"error": "upload too large"}), 413


@app.errorhandler(413)
def _too_large(_e):
    return jsonify({"error": "Upload too large: the request exceeds the size limit "
                             "(10 MB per photograph, 16 photographs)"}), 413


MAX_PIXELS = 60_000_000        # decompression-bomb guard (~60 MP)
Image.MAX_IMAGE_PIXELS = MAX_PIXELS


def _read_image(req, field="image", storage=None):
    """Decode an upload defensively. Raises on anything that is not a sane image.

    Returns (image, raw bytes): the bytes feed the seen-photo guard.
    """
    raw = (storage if storage is not None else req.files[field]).read()
    if not raw:
        raise ValueError("empty upload")
    if len(raw) > MAX_UPLOAD:
        raise ValueError("upload too large")
    probe = Image.open(io.BytesIO(raw))
    probe.verify()                                   # cheap header/structure check
    im = Image.open(io.BytesIO(raw))                 # re-open: verify() exhausts the file
    if im.format not in ("JPEG", "PNG", "TIFF", "BMP", "WEBP"):
        raise ValueError(f"unsupported image format {im.format}")
    w, h = im.size
    if w * h > MAX_PIXELS:
        raise ValueError("image too large")
    if w < 64 or h < 64:
        raise ValueError("image too small to grade")
    return im.convert("RGB"), raw


# ── the one page ─────────────────────────────────────────────────────────────
@app.route("/")
def index():
    diseases = [{"disease": m.disease, "loaded": m.loaded} for m in REGISTRY.models]
    return render_template("screen.html", diseases=diseases,
                           card=REGISTRY.staging.card,
                           calibration=getattr(REGISTRY.staging, "calibration", None),
                           comparison=REGISTRY.comparison.card,
                           comparison_loaded=REGISTRY.comparison.loaded,
                           seen_guard=bool(SEEN_PHOTOS))


# ── API ──────────────────────────────────────────────────────────────────────
def _require_context(req):
    """Patient context is mandatory — there is no default.

    A default would silently reinstate the pre-routing behaviour for any caller that forgot
    to send one, which is exactly the failure Phase 0A exists to remove.
    """
    ctx = (req.form.get("context") or "").strip()
    if not ctx:
        return None, (jsonify({
            "error": "Patient context is required",
            "detail": "Send 'context' as one of: " + ", ".join(sorted(REGISTRY.contexts)),
            "choices": [{"key": k, "label": lab, "hint": h, "advisory": adv}
                        for k, lab, h, adv in REGISTRY.context_choices()],
        }), 400)
    if ctx not in REGISTRY.contexts:
        return None, (jsonify({
            "error": f"Unknown patient context '{ctx}'",
            "detail": "Expected one of: " + ", ".join(sorted(REGISTRY.contexts)),
        }), 400)
    return ctx, None


@app.route("/contexts")
def contexts():
    """The routing table, so the UI never hard-codes patient types."""
    return jsonify([{"key": k, "label": lab, "hint": h, "advisory": adv}
                    for k, lab, h, adv in REGISTRY.context_choices()])


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image provided"}), 400
    ctx, err = _require_context(request)
    if err:
        return err
    try:
        image, raw = _read_image(request)
    except Exception:
        return jsonify({"error": "Invalid image"}), 400
    result = REGISTRY.analyze(image, ctx)
    result["seen_in_data"] = seen_in_data(raw)
    return jsonify(result)


@app.route("/predict_eye", methods=["POST"])
def predict_eye():
    """All photographs of ONE eye in, ONE eye-level stage out (worst-view rule)."""
    eye = (request.form.get("eye") or "").strip()
    if eye not in ("OD", "OS"):
        return jsonify({"error": "eye must be 'OD' or 'OS'"}), 400
    files = request.files.getlist("images")
    if not 1 <= len(files) <= MAX_EYE_PHOTOS:
        return jsonify({"error": f"send 1 to {MAX_EYE_PHOTOS} photographs as 'images'"}), 400
    ctx, err = _require_context(request)
    if err:
        return err
    images, seen = [], []
    for i, f in enumerate(files):
        try:
            image, raw = _read_image(request, storage=f)
        except Exception:
            return jsonify({"error": f"Invalid image: photo {i + 1} could not be read"}), 400
        images.append(image)
        seen.append(seen_in_data(raw))
    try:
        out = REGISTRY.analyze_eye(images, eye, ctx)
    except StagingUnavailable as e:
        return jsonify({"error": "Eye-level staging is unavailable", "detail": str(e)}), e.status
    except EyeNotGradable as e:
        return jsonify({"error": "No photograph of this eye is gradable — no eye-level "
                                 "result can be issued", "detail": str(e)}), 422
    for p, s in zip(out["per_photo"], seen):
        p["seen_in_data"] = s
    return jsonify(out)


@app.route("/report", methods=["POST"])
def report():
    if "image" not in request.files:
        return jsonify({"error": "No image provided"}), 400
    ctx, err = _require_context(request)
    if err:
        return err
    try:
        image, _ = _read_image(request)
    except Exception:
        return jsonify({"error": "Invalid image"}), 400
    result = REGISTRY.analyze(image, ctx)
    # Fail closed: an ungradable image must not yield a downloadable PDF. A report is the
    # artifact that outlives the session, so it must never exist for input no model graded.
    if not result.get("gradable", True):
        return jsonify({"error": "Image is not gradable — no report can be issued",
                        "verdict": result["quality"]["verdict"],
                        "detail": result["quality"]["reason"]}), 422
    # ALL findings go to the PDF, including routed-out ones. Filtering to available-only
    # made a not-assessed disease vanish from the report, which reads as "nothing found".
    # The comparison model is deliberately NOT in the PDF: it is a demo panel, not a result.
    findings = result["findings"]
    patient = {k: request.form.get(k, "") for k in ("name", "id", "age", "gender", "date")}
    patient["context_label"] = result["context_label"]
    gradcam_img = None
    if result["heatmap"]:
        gradcam_img = Image.open(io.BytesIO(base64.b64decode(result["heatmap"])))
    eye_summary = None
    if request.form.get("eye_summary"):
        try:
            eye_summary = json.loads(request.form["eye_summary"])
            e = eye_summary["eye"]
            assert isinstance(e["driving_photo"], int) and not isinstance(e["driving_photo"], bool)
            assert isinstance(e["stage_index"], int) and 0 <= e["stage_index"] <= 5
            assert isinstance(e["stage"], str) and isinstance(eye_summary["laterality"], str)
        except Exception:
            return jsonify({"error": "Invalid eye_summary"}), 400
    out = (Path("results/reports") / f"report_{patient.get('id') or 'anon'}.pdf").resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    generate_pdf(out, patient, findings, original_image=image, gradcam_image=gradcam_img,
                 staging=result.get("staging"), eye_summary=eye_summary,
                 cam_evidence=(result["heatmaps"][0].get("evidence")
                               if result.get("heatmaps") else None))
    return send_file(out, as_attachment=True, download_name=out.name)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5002))
    app.run(host="0.0.0.0", port=port, debug=os.environ.get("FLASK_ENV") != "production")
