"""Build the BE capstone project proposal as a Word document.

    .venv/bin/python scripts/make_proposal_docx.py

Produces docs/PROJECT_PROPOSAL.docx -- roughly five pages, Times New Roman 11pt,
real Word tables, laid out the way the department's proposal template expects.

Every figure quoted here is traceable to RETINAI_FINAL_SUMMARY/ and to the raw
per-photograph prediction files under FINAL_MODEL_EVIDENCE_PACKAGE/. Nothing is
carried over from the superseded multi-disease (v1) blueprint.
"""
from __future__ import annotations

from pathlib import Path

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "PROJECT_PROPOSAL.docx"

ACCENT = RGBColor(0x8C, 0x2F, 0x1E)
MUTED = RGBColor(0x55, 0x55, 0x55)
RULE = "8C2F1E"


# ---------------------------------------------------------------- primitives

def setup(doc):
    for s in doc.sections:
        s.top_margin = Inches(0.7)
        s.bottom_margin = Inches(0.7)
        s.left_margin = Inches(0.75)
        s.right_margin = Inches(0.75)
    n = doc.styles["Normal"]
    n.font.name = "Times New Roman"
    n.font.size = Pt(10)
    n._element.rPr.rFonts.set(qn("w:eastAsia"), "Times New Roman")
    n.paragraph_format.space_after = Pt(3)
    n.paragraph_format.line_spacing = 1.0


def hrule(p):
    pPr = p._p.get_or_add_pPr()
    bdr = OxmlElement("w:pBdr")
    bottom = OxmlElement("w:bottom")
    bottom.set(qn("w:val"), "single")
    bottom.set(qn("w:sz"), "6")
    bottom.set(qn("w:color"), RULE)
    bdr.append(bottom)
    pPr.append(bdr)


def h1(doc, text, before=8):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(before)
    p.paragraph_format.space_after = Pt(4)
    p.paragraph_format.keep_with_next = True
    r = p.add_run(text.upper())
    r.bold = True
    r.font.size = Pt(11)
    r.font.color.rgb = ACCENT
    r.font.name = "Times New Roman"
    hrule(p)
    return p


def h2(doc, text):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(6)
    p.paragraph_format.space_after = Pt(2)
    p.paragraph_format.keep_with_next = True
    r = p.add_run(text)
    r.bold = True
    r.font.size = Pt(10)
    r.font.name = "Times New Roman"
    return p


def _runs(p, text, size, color=None, italic=False):
    import re
    for chunk in re.split(r"(\*\*.+?\*\*)", text):
        if not chunk:
            continue
        if chunk.startswith("**") and chunk.endswith("**"):
            r = p.add_run(chunk[2:-2])
            r.bold = True
        else:
            r = p.add_run(chunk)
        r.italic = italic
        r.font.size = Pt(size)
        r.font.name = "Times New Roman"
        if color is not None:
            r.font.color.rgb = color


def para(doc, text, size=10, justify=True, italic=False, color=None, after=5, before=0):
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(after)
    p.paragraph_format.space_before = Pt(before)
    if justify:
        p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    _runs(p, text, size, color, italic)
    return p


def bullet(doc, text, size=10, indent=0.22):
    p = doc.add_paragraph()
    p.paragraph_format.left_indent = Inches(indent)
    p.paragraph_format.first_line_indent = Inches(-0.16)
    p.paragraph_format.space_after = Pt(2)
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    _runs(p, "•  " + text, size)
    return p


def numbered(doc, n, text, size=10):
    p = doc.add_paragraph()
    p.paragraph_format.left_indent = Inches(0.28)
    p.paragraph_format.first_line_indent = Inches(-0.28)
    p.paragraph_format.space_after = Pt(2)
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    _runs(p, f"{n}.  " + text, size)
    return p


def table(doc, headers, rows, widths=None, size=8.5, caption=None):
    t = doc.add_table(rows=1, cols=len(headers))
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    hdr = t.rows[0].cells
    for i, text in enumerate(headers):
        hdr[i].text = ""
        p = hdr[i].paragraphs[0]
        p.paragraph_format.space_after = Pt(1)
        p.paragraph_format.space_before = Pt(1)
        r = p.add_run(text)
        r.bold = True
        r.font.size = Pt(size)
        r.font.name = "Times New Roman"
        shd = OxmlElement("w:shd")
        shd.set(qn("w:fill"), "F2E9E7")
        hdr[i]._tc.get_or_add_tcPr().append(shd)
    for row in rows:
        cells = t.add_row().cells
        for i, text in enumerate(row):
            cells[i].text = ""
            p = cells[i].paragraphs[0]
            p.paragraph_format.space_after = Pt(1)
            p.paragraph_format.space_before = Pt(1)
            _runs(p, str(text), size)
    if widths:
        for row in t.rows:
            for i, w in enumerate(widths):
                row.cells[i].width = Inches(w)
    if caption:
        c = doc.add_paragraph()
        c.paragraph_format.space_before = Pt(2)
        c.paragraph_format.space_after = Pt(7)
        c.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = c.add_run(caption)
        r.italic = True
        r.font.size = Pt(8.5)
        r.font.color.rgb = MUTED
        r.font.name = "Times New Roman"
    else:
        doc.add_paragraph().paragraph_format.space_after = Pt(4)
    return t


# ---------------------------------------------------------------- document

def build():
    doc = Document()
    setup(doc)

    # ----- title block
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(0)
    r = p.add_run("VIVEKANAND EDUCATION SOCIETY'S INSTITUTE OF TECHNOLOGY, MUMBAI")
    r.bold = True
    r.font.size = Pt(10)
    r.font.name = "Times New Roman"

    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(9)
    r = p.add_run("Department of Automation and Robotics  ·  B.E. Capstone Project 2026–27  ·  Project Proposal")
    r.font.size = Pt(9.5)
    r.font.color.rgb = MUTED
    r.font.name = "Times New Roman"

    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(2)
    r = p.add_run("RetinAI: Explainable Screening and ICROP Staging of\nRetinopathy of Prematurity from a Single Fundus Photograph")
    r.bold = True
    r.font.size = Pt(15)
    r.font.color.rgb = ACCENT
    r.font.name = "Times New Roman"

    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(8)
    r = p.add_run("A device- and source-controlled study of how much of that screening performance is real")
    r.italic = True
    r.font.size = Pt(10)
    r.font.color.rgb = MUTED
    r.font.name = "Times New Roman"
    hrule(p)

    table(
        doc,
        ["Sr.", "Name of Student", "Roll No.", "Branch", "Email ID"],
        [
            ["1", "Rutuja Bait", "36", "Automation and Robotics", "2023.rutuja.bait@ves.ac.in"],
            ["2", "Pravar Rangnekar", "25", "Automation and Robotics", "2023.pravar.rangnekar@ves.ac.in"],
            ["3", "Yash Shengale", "63", "Automation and Robotics", "2023.yash.shengale@ves.ac.in"],
            ["4", "Praharsh Nagpure", "16", "Automation and Robotics", "2023.praharsh.nagpure@ves.ac.in"],
        ],
        widths=[0.4, 1.6, 0.6, 1.9, 2.8],
        size=9,
    )
    para(
        doc,
        "**Project Guide:** Mrs. Amudha Kumar, Department of Automation and Robotics, VESIT  ·  "
        "**Domain:** Medical Imaging / Deep Learning  ·  **Mode:** Software (no hardware fabrication)",
        size=9, justify=False, color=MUTED, after=6,
    )

    # ----- abstract
    h1(doc, "1. Abstract", before=4)
    para(
        doc,
        "Retinopathy of Prematurity (ROP) is a leading preventable cause of childhood blindness. It is a "
        "disease of the retinal blood vessels of babies born too early, it is treatable if it is caught inside "
        "a narrow window of a few weeks, and it is diagnosed by looking at a photograph of the retina. "
        "Screening programmes are limited far less by imaging capacity than by the number of ophthalmologists "
        "available to read those images. This project proposes **RetinAI**, a decision-support system that "
        "takes one colour fundus photograph of a pre-term infant and returns a screening verdict, a six-class "
        "ICROP stage prediction, a Grad-CAM heat map and a downloadable report. The system is built on an "
        "EfficientNetV2-S backbone carrying a clinically structured output head: an ordinal stage head that "
        "preserves the Normal–Stage 1–2–3–4/5 ordering, a separate binary branch for aggressive posterior "
        "ROP (which ICROP3 places outside the stage ladder), and a hospital adversary active only during "
        "training. The second half of the proposal is an **audit**: we set out to "
        "measure how much of the reported performance of such models is the disease and how much is the camera, "
        "the hospital and the labelling convention. Our own preliminary measurements on this corpus show that a "
        "classifier trained on image width and height alone — no pixels whatsoever — reaches AUC 0.911 on a "
        "conventional test split, and that 43% of what the largest public ROP dataset's binary split calls "
        "“ROP-positive” is not ROP at all. The deliverable is therefore both a working prototype and a "
        "reproducible, negative-result-inclusive evidence package, targeted at a peer-reviewed venue.",
    )

    # ----- problem statement
    h1(doc, "2. Problem Statement")
    p = doc.add_paragraph()
    p.paragraph_format.left_indent = Inches(0.25)
    p.paragraph_format.right_indent = Inches(0.25)
    p.paragraph_format.space_after = Pt(6)
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    r = p.add_run(
        "To design, build and honestly validate an explainable deep-learning assistant that screens and stages "
        "Retinopathy of Prematurity from a single colour fundus photograph, so that scarce ophthalmologist time "
        "can be directed towards the infants most likely to need it; and, through device-controlled, "
        "source-controlled and evaluation-unit-controlled experiments, to establish how much of the screening "
        "performance reported for such models is attributable to the disease rather than to acquisition "
        "artefacts and labelling convention."
    )
    r.italic = True
    r.font.size = Pt(10)
    r.font.name = "Times New Roman"
    hrule(p)

    h2(doc, "2.1  Why this disease, and why now")
    para(
        doc,
        "Every pre-term infant below a threshold birth weight and gestational age must be examined repeatedly at "
        "the bedside during the first weeks of life. India has roughly one ophthalmologist per 100,000 people, "
        "mostly in cities, while the NICUs needing the examination are far more widely distributed. Wide-field "
        "cameras have made acquisition cheap and portable, and a trained nurse can capture the photographs. "
        "**The camera is not the bottleneck. The reading is.** A model that ranks which infants need a "
        "specialist's eye first converts a scheduling problem into a triage problem.",
    )
    h2(doc, "2.2  Why the existing literature does not settle it")
    para(
        doc,
        "Published ROP classifiers routinely report 92–99% accuracy, which would appear to close the problem. "
        "Our reading of that literature, and our own measurements on the same public corpora those papers use, "
        "indicate that a substantial part of those numbers is an artefact of three specific methodological "
        "choices, none of which is usually audited:",
    )
    numbered(doc, 1, "**Splitting by photograph instead of by infant.** The largest public ROP dataset photographs each infant roughly 29 times in a single sitting; one infant contributes 470 photographs. A photograph-level split lets the model recognise the baby rather than the disease. A 2026 paper on exactly this corpus split at image level and reported 98–100% accuracy.")
    numbered(doc, 2, "**Unaudited acquisition shortcuts.** Different hospitals use different cameras, and different hospitals have different disease mixes, so image geometry alone predicts the label. On our own test split, width and height alone score AUC 0.911 — close to what a real model scores.")
    numbered(doc, 3, "**Scoring per photograph against a per-eye label.** ROP grows at the temporal (outer) edge of the retina, but a screening examination captures several views of one eye and the eye's single diagnosis is written onto all of them, including views that cannot physically show the ridge.")
    para(
        doc,
        "A fourth, prior problem sits underneath all three: the binary ROP label in the most-used public dataset "
        "is not a ROP label. Recovering the diagnosis codebook showed that its 3,024-image “positive” class "
        "contains 1,295 images (43%) of haemorrhage, toxoplasmosis, hamartoma and optic-nerve hypoplasia. A model "
        "trained on it is an abnormal-infant-retina detector, not a ROP detector.",
    )

    # ----- objectives
    h1(doc, "3. Objectives")
    table(
        doc,
        ["#", "Objective", "Measurable success criterion"],
        [
            ["O1", "Assemble a de-duplicated, provenance-tracked, multi-hospital ROP corpus with per-image ICROP stage labels.",
             "Every source traced to its DOI with licence and caveats logged; mirrors quarantined; zero duplicates across splits."],
            ["O2", "Rebuild the stage label from the original diagnosis codebook rather than the shipped binary label.",
             "100% of images resolved to an ICROP class; non-ROP pathology and treated eyes excluded with counts recorded."],
            ["O3", "Train a staging model whose output structure matches the clinical taxonomy.",
             "Ordinal stage head plus a separate AP-ROP branch; severity ordering preserved."],
            ["O4", "Suppress hospital/device shortcuts, and measure whether the suppression worked.",
             "Geometry equalised across sites; adversarial domain head; must beat a metadata-only baseline by a wide margin."],
            ["O5", "Validate on a hospital never seen in training, under a protocol committed beforehand.",
             "Held-out site split by infant into a tuning half and a sealed half; the sealed half opened exactly once."],
            ["O6", "Quantify the evaluation-unit effect and report it as a primary finding.",
             "Agreement reported per photograph, per camera direction and per eye on identical predictions."],
            ["O7", "Deliver a runnable, explainable prototype application.",
             "Local web app: upload → stage, score, heat map, PDF report; CPU-only inference."],
            ["O8", "Publish the method and the negative results.",
             "Reproducible evidence package with raw predictions committed; manuscript submitted to a peer-reviewed venue."],
        ],
        widths=[0.32, 3.1, 3.38],
        size=8.5,
    )

    # ----- scope
    h1(doc, "4. Scope of the Project")
    h2(doc, "4.1  In scope")
    bullet(doc, "**One disease: Retinopathy of Prematurity.** Six classes — Normal, Stage 1, Stage 2, Stage 3, Stage 4/5 (merged), and AP-ROP as a separate binary output.")
    bullet(doc, "**Retrospective, public, de-identified data only.** Eleven public research datasets, four of which carry per-image ICROP stage labels.")
    bullet(doc, "**A locally-run software prototype** with explainability and a report export. CPU inference, no GPU required at serving time.")
    bullet(doc, "**A methodological audit** of shortcut learning, labelling convention and evaluation unit, reported whether the results flatter the model or not.")
    h2(doc, "4.2  Explicitly out of scope")
    bullet(doc, "**Diabetic Retinopathy and Glaucoma.** Both were built earlier and withdrawn on evidence: the glaucoma evaluation was contaminated by a duplicate data mount that let the same image into train and test, so rather than publish the resulting AUC the disease was removed; DR was parked. Their measured history is retained in the project record, but neither is claimed.")
    bullet(doc, "**Plus disease and treatment-requirement prediction.** Different label, different datasets.")
    bullet(doc, "**Prospective or clinical use.** This is a research and education prototype with no regulatory clearance and no prospective validation. It is **not a medical device** and a qualified clinician must confirm every finding.")
    bullet(doc, "**Primary data collection.** Photographing infants requires ethics-board approval, a hospital partnership and parental consent, none of which is available to a final-year project. Published, licensed, de-identified data is the only legitimate route.")

    # ----- literature
    h1(doc, "5. Literature Survey and Research Gap")
    table(
        doc,
        ["Paper / system", "Data", "Split unit", "Eval unit", "Reported", "Gap we address"],
        [
            ["Vahidmoghadam et al., Sci Rep 2026", "Ostrava only (6,004 img / 188 infants)", "image", "image", "Stage acc 0.98; Plus AUC 1.000", "Our corpus. No patient grouping, device audit, CIs or external site."],
            ["VEA-Net (Wang et al.), Front Cell Dev Biol 2026", "FARFUM + Ostrava + A-Fundus (8,636 / 737)", "subject (GroupKFold-10)", "image", "Stage acc 74.8% ± 10.2", "Correct hygiene, but folds AP-ROP into Stage 4 against ICROP3; DANN used, never quantified; no held-out centre."],
            ["ROPDeepX (Kumar et al.), Discov AI 2026", "FARFUM + RIDIRP + HVDROPDB", "image (on RIDIRP)", "image", "92.6% combined", "Leakage exemplar \u2014 an architecture choice has since propagated on the strength of that number."],
            ["Peng et al., IEEE TMI 2021", "635 img / 196 exams; 1,173 exams", "not stated", "image and examination", "κ 0.979", "Ordinal and unit precedent — we claim neither as novel; we quantify how far the unit alone moves the number."],
            ["i-ROP DL (Campbell, Brown, Redd et al.), 2018–22", "multi-centre US + Nepal/Mongolia", "patient", "examination", "Exam-level plus AUC 0.968", "Severity score, not ICROP stage; private data."],
        ],
        widths=[1.5, 1.35, 0.85, 0.75, 1.05, 1.3],
        size=8,
        caption="Table 1 · The five closest works. The column that separates them is not the backbone — it is the split unit and the evaluation unit.",
    )
    para(
        doc,
        "**The gap.** The field's strongest results come from studies that split by photograph, score by "
        "photograph, and do not test for acquisition shortcuts; the studies that do split correctly report far "
        "lower numbers and are read as weaker models. No published work on these corpora measures the size of "
        "the gap between those two regimes, and none audits the device confound. Our contribution is therefore "
        "not a new architecture — we expect and will report architectural equivalence — but a quantification: "
        "**how much of a ROP model's reported performance is the disease, and how much is the instrument and "
        "the labelling unit.**",
    )

    # ----- proposed system
    h1(doc, "6. Proposed System and Methodology")
    h2(doc, "6.1  Data pipeline")
    para(
        doc,
        "Eleven public datasets are acquired (13,037 photographs), each logged with its DOI, licence, image "
        "count and known caveats; mirrors and re-uploads are quarantined. Four carry per-image ICROP stage "
        "labels and form the corpus: **Ostrava/RIDIRP** (Czechia — scale and the mild stages), **Shenzhen "
        "ROP** (China — the Stage 3 supplier), **ROP-VL** (China — the only public source of Stage 4 and "
        "Stage 5 anywhere), and **Multi-View ROP** (China), locked away as the held-out test hospital and "
        "never trained on. Every exclusion in the cascade is counted and reasoned.",
    )
    table(
        doc,
        ["Step", "Images remaining", "What is removed, and why"],
        [
            ["0 · Acquired", "13,037", "11 public datasets, mirrors already excluded"],
            ["1 · Has an ICROP stage label", "10,427", "−2,610: segmentation-only sets, plus-disease-only data, an unverifiable small sample, 20 cross-dataset duplicates"],
            ["2 · Eligible for the 6-class task", "8,024 (1,760 infants)", "−2,403: 1,295 non-ROP pathology, 1,063 treated/regressed eyes, 45 stage-0"],
            ["3 · Split by hospital", "6,700 pool / 1,324 external", "The entire Multi-View site held back; halved by infant into 663 tuning + 661 sealed test"],
            ["4 · Per-infant frame caps", "3,112 used for training", "−3,588: max 2 normal frames and 20 frames per class per infant, so heavily-photographed infants cannot dominate"],
        ],
        widths=[1.5, 1.35, 3.95],
        size=8.5,
        caption="Table 2 · Inclusion cascade. Caps apply to training only; validation scores every eligible image at natural disease rates.",
    )
    para(
        doc,
        "**Splits are grouped by infant, never by photograph**, and the grouping is verified rather than "
        "assumed: zero infants and zero photographs may appear in more than one group, and the cross-validation "
        "folds must exactly partition the pool.",
    )

    h2(doc, "6.2  Preprocessing — and why step 2 exists")
    para(
        doc,
        "Every photograph passes through the same five steps in training, testing and serving: "
        "(1) **circle crop** to the retinal disc; (2) **letterbox to square** with black padding; (3) **CLAHE** "
        "(clip 2.0, 8×8 tiles) on the luminance channel; (4) **resize to 384×384**; (5) ImageNet normalisation. "
        "Step 2 carries the argument. Resizing straight to a square stretches each photograph by a different "
        "amount, because each site's camera has a different aspect ratio — 1.16–1.21 at the three training "
        "sites against 1.333 at the test site — letting the model infer the hospital from how squashed the "
        "vessels look. Padding gives every site identical treatment. Training-only augmentation attacks the "
        "two measured site fingerprints directly: hue and grayscale jitter against the colour-balance "
        "fingerprint, resolution jitter against the sharpness fingerprint of four cameras.",
    )

    h2(doc, "6.3  Model")
    para(
        doc,
        "An ImageNet-pretrained **EfficientNetV2-S** backbone produces a 1,280-dimensional embedding, which "
        "feeds three heads (20,512,600 parameters in total, all trainable):",
    )
    bullet(doc, "**Ordinal stage head (CORN).** A plain classifier treats “Stage 2 called Normal” and “Stage 2 called Stage 3” as equally wrong. They are not. This head keeps the stages ordered so that being one stage off costs less than being four.")
    bullet(doc, "**AP-ROP branch.** ICROP3 places the aggressive form outside the 1-2-3 ladder, so it gets its own binary output and its images are masked out of the stage loss instead of being assigned a fictitious stage. Its decision threshold is fixed at 0.5 in advance and deliberately never tuned on validation data.")
    bullet(doc, "**Hospital adversary (training only).** A gradient-reversal domain classifier penalises the backbone for encoding which site a photograph came from. It is machinery, not a product feature, and is inert at inference.")
    para(
        doc,
        "Training: AdamW, peak lr 3×10⁻⁴, OneCycleLR, weight decay 0.05, batch 16, 20 epochs, dropout 0.3, "
        "mixed precision, gradient clipping 1.0, seed 42 with a replication at 1337. Two choices are "
        "deliberate and will be defended in the report: **natural class sampling rather than a balanced sampler** "
        "(a balanced sampler collapsed the ordinal cascade in an earlier run, quadratic kappa falling to 0.121, "
        "because the cascade is calibrated to the prevalence it trains on), and **keeping the last epoch rather "
        "than the best** (selecting the maximum of 20 noisy validation scores flatters the result).",
    )

    h2(doc, "6.4  Evaluation protocol")
    bullet(doc, "**Internal:** five-fold cross-validation grouped by infant across the three training sites, scoring every eligible image at natural disease rates.")
    bullet(doc, "**External:** one hospital never trained on, split by infant into a tuning half (inspectable; used for audits and threshold selection) and a **sealed half opened exactly once**, with the scoring rules committed beforehand and the script refusing a second run.")
    bullet(doc, "**Uncertainty:** 95% CIs from bootstrap resampling of **infants**, not photographs — one infant contributes up to 470 images.")
    bullet(doc, "**Controls that must be beaten:** a metadata-only classifier (the floor), and an image-geometry-only classifier (the shortcut). A result that does not clear both is not reported as a result.")
    bullet(doc, "**Unit analysis:** identical predictions scored per photograph, per camera direction, and per eye.")

    # ----- requirements
    h1(doc, "7. System Requirements")
    table(
        doc,
        ["Category", "Specification"],
        [
            ["Hardware", "**Training:** cloud GPU (Kaggle P100 / T4, 16 GB), ~3.5 min per epoch, ~70 min per fold. **Serving:** any laptop CPU — 112 ms per photograph, 83 MB checkpoint, no GPU. **Fabrication: not applicable** — software-only project, no circuit, PCB or enclosure."],
            ["Software", "Python 3.11+, PyTorch ≥2.1, torchvision, timm, albumentations, OpenCV, scikit-learn, NumPy/Pandas, Flask ≥3.0, ReportLab, Matplotlib."],
            ["Tooling", "Git/GitHub for version control and the graded log book; Kaggle kernels for GPU runs; Hugging Face for weight hosting; pytest for repository guard tests."],
        ],
        widths=[1.5, 5.3],
        size=8.5,
    )

    # ----- outcomes
    h1(doc, "8. Expected Outcomes and Preliminary Results")
    para(
        doc,
        "The pipeline above has been implemented and a first full run completed, so the outcomes below are "
        "stated as measurements rather than hopes. The sealed test half has been opened once, under the "
        "committed protocol.",
    )
    table(
        doc,
        ["Measure", "3 training hospitals (6,700 photos, unseen infants)", "New hospital (661 photos, sealed)"],
        [
            ["Exact stage accuracy", "0.786", "0.546"],
            ["Agreement with grader, QWK", "**0.897** [0.869 – 0.923]", "0.493 per photograph  ·  **0.813 per eye**"],
            ["“Any disease” screening AUC", "**0.970**", "**0.835**"],
            ["Normal recall", "0.951", "0.996"],
            ["Per-stage recall (1 / 2 / 3)", "0.499 / 0.631 / 0.713", "0.276 / 0.305 / 0.292"],
            ["Calibration error (ECE)", "0.109", "0.334"],
        ],
        widths=[1.9, 2.55, 2.35],
        size=8.5,
        caption="Table 3 · Headline results. Confidence intervals are bootstrapped over infants (2,000 resamples).",
    )
    h2(doc, "8.1  The four findings the project expects to defend")
    numbered(doc, 1, "**“Is anything wrong with this eye?” transfers to a new hospital; “which exact stage?” largely does not.** Screening AUC holds at 0.835 externally while per-stage recall falls to roughly one in four.")
    numbered(doc, 2, "**The evaluation unit moves the number more than the model does.** Scoring by camera direction with no combining at all: temporal (outer-edge) views reach QWK **0.841**, nasal 0.352, posterior pole 0.229, pooled 0.473. Outer-edge photographs scored one at a time do as well as combining an entire eye's photographs. A large share of the apparent external failure is the model correctly calling a normal-looking photograph normal and being marked wrong because the label belonged to the eye. ROP systems should be scored per eye, and papers must state which unit their labels describe.")
    numbered(doc, 3, "**The confidence scores are not probabilities, and are worst where it matters most.** Over-confidence is +0.109 internally and +0.334 externally, where certainty when right and when wrong differ by only 0.041. Worse, the model's highest average certainty (0.93) accompanies its most dangerous error — calling a diseased eye normal — so no confidence threshold can make the system safe. The output will be labelled an uncalibrated ranking score, and temperature scaling is the first remediation.")
    numbered(doc, 4, "**The clinically structured head does not beat a plain softmax, and we will say so.** Macro-F1 differs by 0.0055 and the sign reverses across seeds; QWK is identical (p = 0.977); the plain softmax is better calibrated. Three separate noise sources each exceed the effect being measured. The structured head's justification is that it represents the taxonomy correctly — not that it scores higher.")

    # ----- timeline
    h1(doc, "9. Project Timeline")
    table(
        doc,
        ["Phase", "Period", "Deliverable", "Status"],
        [
            ["I · Survey and dataset acquisition", "Jun – Jul 2026", "11 datasets acquired, provenance file, licence audit, de-identification of filenames", "Complete"],
            ["II · Label reconstruction and audit", "Jul – Aug 2026", "ICROP stage recovered from the diagnosis codebook; shortcut and device audits; scope narrowed to ROP", "Complete"],
            ["III · Model development", "Aug 2026", "Structured head, adversary, 5-fold grouped cross-validation, seed replication", "Complete"],
            ["IV · External validation", "Sep 2026", "Sealed test opened once; unit analysis; calibration and Grad-CAM audits; evidence package", "Complete"],
            ["V · Manuscript and remediation", "Sep – Dec 2026", "Paper submitted (deadline 26 Oct 2026); temperature scaling, re-derived operating point, checkpoint provenance, remaining-fold ablations", "In progress"],
            ["VI · Clinician adjudication", "Nov 2026 – Feb 2027", "Paediatric ophthalmologist re-grades ~300 stratified images to establish a human agreement ceiling", "Planned — long pole"],
            ["VII · Application and report", "Jan – Apr 2027", "Hardened local app, final capstone report, demonstration and viva", "Planned"],
        ],
        widths=[1.85, 1.1, 3.0, 0.85],
        size=8.5,
    )

    # ----- limitations
    h1(doc, "10. Limitations, Risks and Ethical Considerations")
    h2(doc, "10.1  Acknowledged limitations")
    bullet(doc, "**One external hospital is a sanity check, not a generalisation claim.** We know the model degrades at one new site; we cannot yet distinguish “that site is unusual” from “it degrades everywhere”. Only a second independent site separates those explanations.")
    bullet(doc, "**The severe classes rest on very few images.** All 89 Stage 4/5 images in the project come from one dataset, so that class may partly reflect recognising the source. The held-out site contributes 4 Stage 4/5 and 12 AP-ROP images, so both were declared not externally validated in advance.")
    bullet(doc, "**No independent grader has re-read the labels.** Without a second opinion we cannot say whether Stage 1's low precision is the model failing or the labels disagreeing, and there is no human ceiling to compare our agreement against. This needs a clinician, not compute.")
    bullet(doc, "**The heat map is audited and limited.** Grad-CAM attention inside the retina shows no lift over chance (1.04×) and 19% of peak activation lands on the letterbox padding. It is presented with its measured limits stated, never as evidence that the model understands the disease.")
    h2(doc, "10.2  Ethics, privacy and licensing")
    bullet(doc, "**No patient is recruited and no image is collected by us.** Every photograph comes from a published, de-identified research dataset used under its stated licence. One test dataset is CC BY-NC, so no commercial use or public re-mirroring of those images is permitted.")
    bullet(doc, "**Re-identification risk is actively managed.** Ostrava filenames originally encoded an infant's sex, gestational age, birth weight and postnatal age; every file was renamed and the name-map kept outside the repository. An automated guard test scans for surviving identifiers, and a public deployment was deleted outright in August 2026 when patient images were found in its git history — the right trade, and we would make it again.")
    bullet(doc, "**Sources are cited to their original registry, never to a Kaggle mirror**, following the 2025–26 wave of retractions over untraceable re-uploaded data.")
    bullet(doc, "**Safety framing.** RetinAI is not a medical device, has no regulatory clearance and was never prospectively validated. Its clinical blockers are stated in the product itself, not only in the report.")

    # ----- references
    h1(doc, "11. References")
    refs = [
        "Chiang M. F. et al. “International Classification of Retinopathy of Prematurity, Third Edition (ICROP3).” Ophthalmology, 128(10):e51–e68, 2021.",
        "Timkovič J. et al. “Retinal Image Dataset of Infants and Retinopathy of Prematurity (RIDIRP).” Scientific Data, 2024. figshare collection 6626162 (CC0).",
        "Vahidmoghadam A. et al. “Automated diagnosis of plus form and early stages of retinopathy of prematurity.” Scientific Reports, 16:7234, 2026.",
        "Wang L. et al. “VEA-Net: vascular enhancement attention network for multi-task ROP analysis.” Frontiers in Cell and Developmental Biology, 2026.",
        "Kumar S. et al. “ROPDeepX: a hybrid deep learning framework for retinopathy of prematurity.” Discover Artificial Intelligence, 2026.",
        "Peng Y. et al. “Automatic Staging for Retinopathy of Prematurity with Deep Feature Fusion and Ordinal Classification Strategy.” IEEE Transactions on Medical Imaging, 40(7):1750–1762, 2021.",
        "Brown J. M., Campbell J. P., Beers A. et al. “Automated Diagnosis of Plus Disease in Retinopathy of Prematurity Using Deep Convolutional Neural Networks.” JAMA Ophthalmology, 136(7):803–810, 2018.",
        "Shi X., Cao W. and Raschka S. “Deep neural networks for rank-consistent ordinal regression based on conditional probabilities (CORN).” Pattern Analysis and Applications, 26:941–955, 2023.",
        "DeGrave A. J., Janizek J. D. and Lee S.-I. “AI for radiographic COVID-19 detection selects shortcuts over signal.” Nature Machine Intelligence, 3:610–619, 2021.",
        "Gschließer A. et al. “Inter-expert and intra-expert agreement on the diagnosis and treatment of retinopathy of prematurity.” American Journal of Ophthalmology, 160(3):553–560, 2015.",
        "Zhao J. et al. “ROP-VL: a vision-language dataset for retinopathy of prematurity.” figshare, 2025. DOI 10.6084/m9.figshare.30143461 (CC BY 4.0).",
    ]
    for i, r in enumerate(refs, 1):
        p = doc.add_paragraph()
        p.paragraph_format.left_indent = Inches(0.3)
        p.paragraph_format.first_line_indent = Inches(-0.3)
        p.paragraph_format.space_after = Pt(1)
        p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        _runs(p, f"[{i}]  {r}", 8.5)

    # ----- declaration
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(10)
    hrule(p)
    para(
        doc,
        "**Declaration.** The work proposed here is carried out by the undersigned students under the guidance "
        "of Mrs. Amudha Kumar. All data used is publicly available, de-identified and used under its stated "
        "licence. RetinAI is a research prototype and is not a medical device.",
        size=9, color=MUTED, after=10,
    )
    para(
        doc,
        "Rutuja Bait  \u00b7  Pravar Rangnekar  \u00b7  Yash Shengale  \u00b7  Praharsh Nagpure"
        "                    Mrs. Amudha Kumar (Guide)",
        size=9, justify=False, color=MUTED, after=0,
    )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    doc.save(OUT)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    build()
