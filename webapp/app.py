"""Flask web app — AI-assisted ROP screening (Retinopathy of Prematurity).

Pages: Dashboard (/), Screening tool (/screen), Explainability gallery (/gallery),
History (/history), About (/about). This deployment is ROP-ONLY: it serves the binary ROP
screening model, plus a clearly-labelled 6-class ICROP staging research preview. Patient
context is mandatory and routed (see webapp/registry.yaml for why), with Grad-CAM, a
downloadable clinical PDF, and an in-memory recent-screening history.

    python webapp/app.py   ->   http://127.0.0.1:5002
"""
import base64
import io
import math
import os
import sys
import uuid
from collections import deque
from datetime import datetime
from pathlib import Path

from flask import Flask, jsonify, render_template, request, send_file
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root on path

from webapp.inference import Registry
from reports.report_generator import generate_pdf

MAX_UPLOAD = 10 * 1024 * 1024
REGISTRY = Registry(os.environ.get("REGISTRY", "webapp/registry.yaml"))
HISTORY = deque(maxlen=24)          # in-memory recent screenings
GALLERY_CACHE = {}                  # disease -> [examples] (computed once)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD
app.jinja_env.globals.update(sin=math.sin, pi=math.pi)  # for the hero flip-text stagger


MAX_PIXELS = 60_000_000        # decompression-bomb guard (~60 MP)
Image.MAX_IMAGE_PIXELS = MAX_PIXELS


def _read_image(req):
    """Decode an upload defensively. Raises on anything that is not a sane image."""
    raw = req.files["image"].read()
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
    return im.convert("RGB")


def _thumb(image, size=110):
    im = image.copy(); im.thumbnail((size, size))
    buf = io.BytesIO(); im.save(buf, "JPEG", quality=70)
    return base64.b64encode(buf.getvalue()).decode()


# The actual project team (VESIT Automation & Robotics, BE Capstone 2026-27).
TEAM = [
    {"name": "Rutuja Bait", "role": "Team member", "contribution": "", "photo": ""},
    {"name": "Pravar Rangnekar", "role": "Team member", "contribution": "", "photo": ""},
    {"name": "Yash Shengale", "role": "Team member", "contribution": "", "photo": ""},
    {"name": "Praharsh Nagpure", "role": "Team member", "contribution": "", "photo": ""},
]

# No testimonials: this is a student research prototype and it has none. The earlier
# placeholder quotes attributed to invented clinicians are exactly the kind of thing this
# project's own honesty rules exist to prevent.
TESTIMONIALS = []

# Disease education content (images: Wikimedia Commons, hot-linked thumbnails).
# ROP only — this deployment screens one disease and explains one disease.
DISEASES = [
    {
        "key": "rop", "name": "Retinopathy of Prematurity", "abbr": "ROP", "accent": "#fbbf24",
        "image": "https://upload.wikimedia.org/wikipedia/commons/thumb/0/0e/Retinopathy_of_Prematurity_Zones.jpg/960px-Retinopathy_of_Prematurity_Zones.jpg",
        "hover": "A premature infant's retina mapped into the ROP screening zones.",
        "tagline": "Abnormal retinal vessels in babies born too early.",
        "what": "Retinopathy of prematurity is an eye disorder of premature babies. The retina's blood "
                "vessels normally finish growing only near full-term birth; in babies born too early "
                "this growth is interrupted, and abnormal vessels can grow instead — sometimes scarring "
                "and detaching the retina and causing blindness.",
        "causes": ["Premature birth with an incompletely vascularised retina",
                   "High or fluctuating supplemental oxygen in the NICU",
                   "Very low birth weight",
                   "Driven by abnormal VEGF (vessel-growth) signalling"],
        "age": "Premature infants — especially those born before ~31 weeks or under ~1500 g. It is "
               "screened by eye exams in the first weeks of life in the NICU.",
        "genetics": "Mostly driven by prematurity and oxygen, not inheritance. A minority of cases "
                    "overlap with genetic vascular conditions (genes shared with familial exudative "
                    "vitreoretinopathy / Norrie disease).",
        "signs": "Found on screening, not from symptoms. Graded by stage (1–5, from a demarcation "
                 "line to full retinal detachment), zone, and 'plus disease' (dilated, tortuous vessels).",
        "detect": "Our model screens a neonatal fundus image for signs of ROP (No ROP vs ROP) at high "
                  "sensitivity, acting as a second reader for NICU screening. A 6-class ICROP staging "
                  "model (Normal, Stage 1-3, Stage 4/5, AP-ROP) is included as a research preview.",
    },
]


# ── pages ────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html", summary=REGISTRY.summary(), page="dashboard",
                           team=TEAM, testimonials=TESTIMONIALS)


@app.route("/diseases")
def diseases():
    return render_template("diseases.html", diseases=DISEASES, page="diseases")


@app.route("/screen")
def screen():
    diseases = [{"disease": m.disease, "loaded": m.loaded} for m in REGISTRY.models]
    return render_template("screen.html", diseases=diseases, page="screen")


@app.route("/history")
def history():
    return render_template("history.html", history=list(HISTORY), page="history")


@app.route("/about")
def about():
    return render_template("about.html", summary=REGISTRY.summary(), page="about")


@app.route("/gallery")
def gallery():
    return render_template("gallery.html", examples=_gallery(), page="gallery")


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
        image = _read_image(request)
    except Exception:
        return jsonify({"error": "Invalid image"}), 400
    result = REGISTRY.analyze(image, ctx)
    # record in history (thumbnail + top findings)
    flagged = [f for f in result["findings"]
               if f.get("available") and f.get("grade", 0) > 0]
    HISTORY.appendleft({
        "id": uuid.uuid4().hex[:8],
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "context": ctx,
        "context_label": result["context_label"],
        "thumb": _thumb(image),
        "image": _thumb(image, 640),     # larger copy kept for View + PDF report
        # keep routed-out diseases so the history entry cannot read as "all clear"
        "findings": [{"disease": f["disease"],
                      "prediction": f.get("prediction", "Not assessed"),
                      "score": f.get("score", 0),
                      "risk": f.get("risk", ""),
                      "available": bool(f.get("available")),
                      "status": f.get("status", "ok")}
                     for f in result["findings"]],
        "flagged": len(flagged),
    })
    return jsonify(result)


@app.route("/report", methods=["POST"])
def report():
    if "image" not in request.files:
        return jsonify({"error": "No image provided"}), 400
    ctx, err = _require_context(request)
    if err:
        return err
    try:
        image = _read_image(request)
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
    findings = result["findings"]
    patient = {k: request.form.get(k, "") for k in ("name", "id", "age", "gender", "date")}
    patient["context_label"] = result["context_label"]
    gradcam_img = None
    if result["heatmap"]:
        gradcam_img = Image.open(io.BytesIO(base64.b64decode(result["heatmap"])))
    out = (Path("results/reports") / f"report_{patient.get('id') or 'anon'}.pdf").resolve()
    generate_pdf(out, patient, findings, original_image=image, gradcam_image=gradcam_img)
    return send_file(out, as_attachment=True, download_name=out.name)


# ── history item actions: view / download report / delete ─────────────────────
def _hist_get(hid):
    return next((h for h in HISTORY if h.get("id") == hid), None)


@app.route("/history/item/<hid>")
def history_item(hid):
    h = _hist_get(hid)
    if not h:
        return jsonify({"error": "not found"}), 404
    return jsonify({"id": h["id"], "time": h["time"], "image": h["image"],
                    "findings": h["findings"], "flagged": h["flagged"]})


@app.route("/history/delete/<hid>", methods=["POST"])
def history_delete(hid):
    h = _hist_get(hid)
    if h is None:
        return jsonify({"ok": False, "error": "not found"}), 404
    HISTORY.remove(h)
    return jsonify({"ok": True})


@app.route("/history/report/<hid>")
def history_report(hid):
    h = _hist_get(hid)
    if not h:
        return jsonify({"error": "not found"}), 404
    # Re-run under the SAME context the screening was performed with. Falling back to a
    # default here would let a re-download silently apply different routing than the
    # original screening — a report that no longer matches what the operator saw.
    # Checked BEFORE decoding the image, so a legacy entry returns 409 rather than crashing.
    ctx = h.get("context")
    if not ctx or ctx not in REGISTRY.contexts:
        return jsonify({"error": "This screening predates patient-context routing; "
                                 "re-screen the image to produce a report."}), 409
    image = Image.open(io.BytesIO(base64.b64decode(h["image"]))).convert("RGB")
    result = REGISTRY.analyze(image, ctx)
    findings = result["findings"]
    gradcam_img = None
    if result["heatmap"]:
        gradcam_img = Image.open(io.BytesIO(base64.b64decode(result["heatmap"])))
    patient = {"name": "", "id": hid, "age": "", "gender": "", "date": h["time"],
               "context_label": result["context_label"]}
    out = (Path("results/reports") / f"report_{hid}.pdf").resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    generate_pdf(out, patient, findings, original_image=image, gradcam_image=gradcam_img)
    fname = "screening_" + h["time"].replace(":", "-").replace(" ", "_") + ".pdf"
    return send_file(out, as_attachment=True, download_name=fname)


# ── explainability gallery (curated local examples, computed once) ────────────
def _gallery():
    if GALLERY_CACHE:
        return GALLERY_CACHE
    base = Path(__file__).resolve().parent / "static" / "gallery"
    for dm in REGISTRY.models:
        if not dm.loaded:
            continue
        folder = base / dm.disease.lower()
        if not folder.exists():
            continue
        items = []
        for img_path in sorted(folder.glob("*.jpg"))[:6]:
            try:
                image = Image.open(img_path).convert("RGB")
                res = dm.predict(image)
                heat = dm.gradcam(image, res["grade"]) if res.get("grade", 0) >= 0 else None
                items.append({"name": img_path.name, "thumb": _thumb(image, 200),
                              "heatmap": heat, "prediction": res["prediction"],
                              "score": res["score"], "risk": res.get("risk", "")})
            except Exception:
                continue
        if items:
            GALLERY_CACHE[dm.disease] = items
    return GALLERY_CACHE


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5002))
    app.run(host="0.0.0.0", port=port, debug=os.environ.get("FLASK_ENV") != "production")
