"""Phase 0A acceptance tests — API context contract (R3) and PDF honesty (R5).

R5 guards a specific measured bug: `generate_pdf` filtered to available findings only, and any
routed-out disease that did reach it fell through to `recommendation(disease, 0)` and printed
"Healthy retina — routine annual screening" for a model that never executed. A downloadable PDF
asserting health for an unrun model is the most dangerous artifact in the system because it
outlives the session.

Run:  .venv/bin/python -m pytest tests/test_api_and_report.py -v
"""
import io
import sys
from pathlib import Path

import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from reports.report_generator import generate_pdf, recommendation  # noqa: E402


from conftest import fundus_like  # noqa: E402


def _png_bytes():
    """Gate-passing image — the gradability gate (Phase 0B) runs before routing."""
    buf = io.BytesIO()
    fundus_like().save(buf, "PNG")
    buf.seek(0)
    return buf


# ── R5: the PDF must never present an unrun model as a clinical result ───────
NOT_RUN = {"disease": "ROP", "available": False, "status": "not_applicable",
           "note": "Not assessed — outside this model's population (trained for: infant)."}
RAN = {"disease": "DR", "available": True, "status": "ok", "prediction": "No DR",
       "score": 91.2, "grade": 0}


def _pdf_text(path):
    """Extract drawn text without a PDF-parser dependency.

    ReportLab encodes content streams ASCII85 then Flate. Note the delimiter regex must not
    be a plain split on b"stream" — that also matches inside b"endstream" and silently
    corrupts every chunk boundary.
    """
    import base64 as b64
    import re
    import zlib

    out = []
    for m in re.finditer(rb"stream\r?\n(.*?)endstream", Path(path).read_bytes(), re.S):
        d = m.group(1).strip()
        if d.endswith(b"~>"):
            try:
                d = b64.a85decode(d[:-2])
            except Exception:
                pass
        try:
            d = zlib.decompress(d)
        except Exception:
            pass
        out.append(d)
    assert out, "no content streams found — extractor is broken, not the PDF"
    return b"\n".join(out).decode("latin-1", "replace")


def test_unrun_model_is_not_reported_as_healthy(tmp_path):
    out = tmp_path / "r.pdf"
    generate_pdf(out, {"name": "T", "id": "1"}, [RAN, NOT_RUN],
                 original_image=Image.new("RGB", (64, 64)))
    text = _pdf_text(out)
    assert "Not assessed" in text, "routed-out disease must be labelled Not assessed"
    # the exact false-reassurance string from RISK_RULES["ROP"][0]
    assert "No ROP signs" not in text
    assert "continue scheduled screening" not in text


def test_unrun_model_still_appears_in_the_report(tmp_path):
    """It must not silently vanish — an omission reads as 'nothing found'."""
    out = tmp_path / "r.pdf"
    generate_pdf(out, {"name": "T", "id": "1"}, [RAN, NOT_RUN],
                 original_image=Image.new("RGB", (64, 64)))
    assert "ROP" in _pdf_text(out)


def test_screening_context_is_recorded_in_the_pdf(tmp_path):
    out = tmp_path / "r.pdf"
    generate_pdf(out, {"name": "T", "id": "1", "context_label": "Preterm infant"},
                 [RAN], original_image=Image.new("RGB", (64, 64)))
    assert "Preterm infant" in _pdf_text(out)


def test_recommendation_rules_unchanged_for_models_that_did_run():
    assert recommendation("ROP", 0)[0] == "Low"
    assert recommendation("DR", 2)[0] == "Moderate"


# ── R3: the HTTP contract ────────────────────────────────────────────────────
@pytest.fixture(scope="module")
def client():
    import webapp.app as A
    if not any(m.loaded for m in A.REGISTRY.models):
        pytest.skip("no model weights available")
    A.app.config["TESTING"] = True
    return A.app.test_client()


def test_predict_without_context_is_rejected(client):
    r = client.post("/predict", data={"image": (_png_bytes(), "a.png")},
                    content_type="multipart/form-data")
    assert r.status_code == 400
    assert "context" in r.get_json()["error"].lower()


def test_predict_with_unknown_context_is_rejected(client):
    r = client.post("/predict", data={"image": (_png_bytes(), "a.png"), "context": "toddler"},
                    content_type="multipart/form-data")
    assert r.status_code == 400


def test_predict_adult_context_is_rejected(client):
    # ROP-only deployment: 'adult' is no longer a known context, and the API must say so
    # rather than silently running the infant model on an adult eye.
    r = client.post("/predict", data={"image": (_png_bytes(), "a.png"), "context": "adult"},
                    content_type="multipart/form-data")
    assert r.status_code == 400


def test_predict_infant_context_returns_no_adult_results(client):
    r = client.post("/predict", data={"image": (_png_bytes(), "a.png"), "context": "infant"},
                    content_type="multipart/form-data")
    assert {f["disease"] for f in r.get_json()["findings"]} == {"ROP"}


def test_report_without_context_is_rejected(client):
    r = client.post("/report", data={"image": (_png_bytes(), "a.png")},
                    content_type="multipart/form-data")
    assert r.status_code == 400


def test_contexts_endpoint_lists_the_routing_table(client):
    body = client.get("/contexts").get_json()
    keys = {c["key"] for c in body}
    assert keys == {"infant"}
    assert all("label" in c for c in body)



# ── the decision line and the staging preview must reach the reader ──────────
#
# Two field-reported defects, same root cause: the server computed something the client
# never showed. (1) `analyze()` returned a full 6-class ICROP payload and main.js had no
# code path for it, so the "staging research preview" advertised on three pages had never
# once been rendered. (2) The finding was drawn as two competing bars — "No ROP 43.7% /
# ROP Detected 56.3%" — which reads against an implied 50% boundary. This model decides at
# 19.33%. A reader seeing 43.7% concluded the model was nearly undecided when the score was
# in fact ~3x the operating point.
def test_binary_finding_ships_its_decision_threshold(client):
    r = client.post("/predict", data={"image": (_png_bytes(), "a.png"), "context": "infant"},
                    content_type="multipart/form-data")
    rop = next(f for f in r.get_json()["findings"] if f["disease"] == "ROP")
    if not rop.get("available"):
        pytest.skip("ROP weights not loaded")
    dec = rop["decision"]
    assert dec["positive_class"] == "ROP Detected"
    # the served operating point, whatever it is, and never a 50% cut-off. Since 1 Nov 2026
    # that is the DINOv2 ensemble's pre-registered CV point (flags score >= line); before it,
    # the structured model's re-based line (flags score > line).
    import json as J
    import webapp.app as A
    st = A.REGISTRY.staging
    if getattr(st, "members", None) and st.screening:
        expected = st.screening["threshold"] * 100
        assert dec["positive"] is (dec["score"] >= dec["threshold"])
    else:
        reb = ROOT / "results/rop/screening_rebase.json"
        expected = (J.loads(reb.read_text())["threshold"] * 100 if reb.exists() else 19.33)
        assert dec["positive"] is (dec["score"] > dec["threshold"])
    assert dec["threshold"] == pytest.approx(expected, abs=0.01)
    assert dec["threshold"] != 50.0


def test_positive_verdict_ships_its_own_false_alarm_rate(client):
    """"ROP Detected" on a healthy eye is the operating point, not a malfunction.

    At the served threshold the model measures 99.0% sensitivity against 39.8%
    specificity — it flags most healthy eyes too. The card no longer prints that number
    (the decision meter was removed from the UI), so this now guards the /predict payload:
    the rate must stay in `decision.measured` for the report and any caller that reads it.
    """
    r = client.post("/predict", data={"image": (_png_bytes(), "a.png"), "context": "infant"},
                    content_type="multipart/form-data")
    rop = next(f for f in r.get_json()["findings"] if f["disease"] == "ROP")
    if not rop.get("available"):
        pytest.skip("ROP weights not loaded")
    m = rop["decision"]["measured"]
    # >= 90: the served ensemble's line is DEFINED as the largest keeping CV sensitivity >= 0.90
    assert m["sensitivity"] >= 90.0
    assert m["false_alarm"] == pytest.approx(100.0 - m["specificity"], abs=0.11)
    # The ensemble's CV line has 0.4% false alarms (99.6% specificity), the heads before it
    # 22-60%; a line claiming exactly none would be suspect.
    assert m["false_alarm"] > 0.0, "a screening head with no false alarms would be suspect"


def test_positive_verdict_discloses_its_out_of_hospital_behaviour(client):
    """The internal specificity quoted alone is the flattering half of two measurements.

    This is the defect that forced the re-basing: the retired ResNet50 head measured 39.8%
    specificity internally and 0.0% at a hospital it never trained on, and only the first
    number was published. Whatever head is served, both must stay in the response.
    """
    r = client.post("/predict", data={"image": (_png_bytes(), "a.png"), "context": "infant"},
                    content_type="multipart/form-data")
    rop = next(f for f in r.get_json()["findings"] if f["disease"] == "ROP")
    if not rop.get("available"):
        pytest.skip("ROP weights not loaded")
    ext, m = rop["decision"]["external"], rop["decision"]["measured"]
    assert ext["n"] > 0 and 0.0 <= ext["specificity"] <= 100.0
    sens = ext.get("sensitivity", 101.0)          # the ResNet50 fallback reports specificity only
    # Verified 1 Nov 2026: for the DINOv2 ensemble the worse out-of-hospital number is
    # SENSITIVITY (CV 90.0% -> 65.6% of eyes at the new hospital; specificity 99.6% -> 100%,
    # 50 normal eyes). For the heads before it, specificity. Whichever it is must ship.
    assert sens < m["sensitivity"] or ext["specificity"] < m["specificity"], \
        "the out-of-hospital behaviour must show the number that got worse"


def test_staging_reports_any_rop_not_just_the_winning_class(client):
    """"ICROP stage: Normal" under "ROP Detected" reads as a contradiction.

    It is not one — the heads answer different questions with different decision rules.
    The any-ROP total is the quantity they answer in common, and it must be present and
    consistent with the per-class scores or the two cannot be compared at all.
    """
    import webapp.app as A
    if not A.REGISTRY.staging.loaded:
        pytest.skip("staging weights not loaded")
    r = client.post("/predict", data={"image": (_png_bytes(), "a.png"), "context": "infant"},
                    content_type="multipart/form-data")
    st = r.get_json()["staging"]
    diseased = sum(v for k, v in st["scores"].items() if k != "Normal")
    assert st["any_rop_probability"] == pytest.approx(diseased, abs=0.35)
    assert st["any_rop_probability"] + st["scores"]["Normal"] == pytest.approx(100.0, abs=0.35)


def test_frontend_reconciles_a_positive_screen_with_a_normal_stage():
    js = (ROOT / "webapp" / "static" / "js" / "main.js").read_text()
    assert "stagingVsScreening" in js
    assert "any_rop_probability" in js, "the comparable quantity must reach the card"


def test_staging_preview_is_returned_to_the_client(client):
    import webapp.app as A
    if not A.REGISTRY.staging.loaded:
        pytest.skip("staging weights not loaded")
    r = client.post("/predict", data={"image": (_png_bytes(), "a.png"), "context": "infant"},
                    content_type="multipart/form-data")
    st = r.get_json()["staging"]
    assert st["preview"] is True
    assert st["prediction"] in A.REGISTRY.staging.CLASSES
    assert set(st["scores"]) == set(A.REGISTRY.staging.CLASSES)


def test_frontend_renders_the_staging_payload():
    """Guards the exact defect: a payload the server sends and the client silently drops."""
    js = (ROOT / "webapp" / "static" / "js" / "main.js").read_text()
    assert "res.staging" in js, "staging payload is computed but never rendered"
    assert (ROOT / "webapp" / "templates" / "screen.html").read_text().count("stagingcard")


def test_staging_preview_appears_in_the_pdf(tmp_path):
    out = tmp_path / "r.pdf"
    generate_pdf(out, {"name": "T", "id": "1"}, [RAN],
                 original_image=Image.new("RGB", (64, 64)),
                 staging={"prediction": "Stage 2", "preview": True,
                          "scores": {"Normal": 10.0, "Stage 1": 20.0, "Stage 2": 70.0},
                          "note": "Research preview — not a clinical grade."})
    text = _pdf_text(out)
    assert "ICROP staging" in text
    assert "Stage 2" in text
    assert "research preview" in text


def test_pdf_states_the_decision_line_not_a_50_percent_cutoff(tmp_path):
    out = tmp_path / "r.pdf"
    scored = dict(RAN, decision={"score": 56.3, "threshold": 19.3, "positive": True,
                                 "positive_class": "ROP Detected"})
    generate_pdf(out, {"name": "T", "id": "1"}, [scored],
                 original_image=Image.new("RGB", (64, 64)))
    text = _pdf_text(out)
    assert "decision line" in text
    assert "19.3" in text


CAM_EV = {"measurable": True, "off_retina_pct": 24.1, "concentration_pct": 10.2,
          "peak_zone": "mid-periphery", "peak_clock": 1, "peak_inside_retina": False,
          "note": "Grad-CAM shows what moved the score."}


def test_pdf_qualifies_the_heatmap_it_prints(tmp_path):
    """A heatmap on a clinical-looking page is the most over-read element in the report."""
    out = tmp_path / "r.pdf"
    generate_pdf(out, {"name": "T", "id": "1"}, [RAN],
                 original_image=Image.new("RGB", (64, 64)),
                 gradcam_image=Image.new("RGB", (64, 64)), cam_evidence=CAM_EV)
    text = _pdf_text(out)
    assert "outside the retina" in text
    assert "HIGH, map" in text, "a 24% leak must be called out, not just printed"


def test_staging_block_clears_the_heatmap_caption(tmp_path):
    """Both blocks existed; a constant offset put them on the same baseline."""
    import re
    out = tmp_path / "r.pdf"
    generate_pdf(out, {"name": "T", "id": "1"}, [RAN],
                 original_image=Image.new("RGB", (64, 64)),
                 gradcam_image=Image.new("RGB", (64, 64)), cam_evidence=CAM_EV,
                 staging={"prediction": "Normal", "preview": True,
                          "scores": {"Normal": 96.8}, "note": "Research preview."})
    placed = {}
    for m in re.finditer(r"1 0 0 1 ([\d.]+) ([\d.]+) Tm.*?\((.*?)\) Tj", _pdf_text(out), re.S):
        placed[m.group(3)[:30]] = (float(m.group(2)), float(m.group(1)))
    cam_last = min(y for s, (y, x) in placed.items() if "Grad-CAM shows" in s)
    staging_top = max(y for s, (y, x) in placed.items() if "ICROP staging" in s)
    assert cam_last - staging_top > 14, "staging heading overlaps the heatmap caption"

