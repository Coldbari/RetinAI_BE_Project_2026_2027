"""Build the ISBI submission as a Word document (and, with --paper2, the Word copy of paper 2).

    .venv/bin/python scripts/make_paper_docx.py            # ISBI paper
    .venv/bin/python scripts/make_paper_docx.py --paper2   # docs/paper/RetinAI_Paper2_Pretraining_draft.docx (pandoc)

Produces docs/paper/ISBI2027_paper.docx — the same text as isbi2027.tex, laid out as an
editable manuscript: Times New Roman 10pt, real Word tables, figures embedded at the width
they should print at, and captions in a distinct style so they can be restyled in one pass.

This is the version to hand to a co-author who wants to leave tracked comments. The .tex
stays the submission artifact; this is the working draft.
"""
from __future__ import annotations

from pathlib import Path

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

ROOT = Path(__file__).resolve().parent.parent
FIGS = ROOT / "graphs" / "isbi"
OUT = ROOT / "docs" / "paper" / "ISBI2027_paper.docx"

ACCENT = RGBColor(0x8C, 0x2F, 0x1E)
MUTED = RGBColor(0x55, 0x55, 0x55)


def _style(doc):
    n = doc.styles["Normal"]
    n.font.name = "Times New Roman"
    n.font.size = Pt(10)
    n._element.rPr.rFonts.set(qn("w:eastAsia"), "Times New Roman")
    n.paragraph_format.space_after = Pt(6)
    n.paragraph_format.line_spacing = 1.0


def h(doc, text, size=12, before=14, after=4, color=None):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(before)
    p.paragraph_format.space_after = Pt(after)
    r = p.add_run(text)
    r.bold = True
    r.font.size = Pt(size)
    r.font.name = "Times New Roman"
    if color is not None:
        r.font.color.rgb = color
    return p


def para(doc, text, justify=True, italic=False, size=10, color=None, after=6):
    """One paragraph; **bold** and *italic* markers are honoured."""
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(after)
    if justify:
        p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    import re
    for chunk in re.split(r"(\*\*.+?\*\*|\*.+?\*)", text):
        if not chunk:
            continue
        r = p.add_run(chunk.strip("*"))
        r.font.size = Pt(size)
        r.font.name = "Times New Roman"
        r.bold = chunk.startswith("**")
        r.italic = italic or (chunk.startswith("*") and not chunk.startswith("**"))
        if color is not None:
            r.font.color.rgb = color
    return p


def figure(doc, stem, caption, width=6.3):
    doc.add_picture(str(FIGS / f"{stem}.png"), width=Inches(width))
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    p.paragraph_format.space_after = Pt(12)
    import re
    for i, chunk in enumerate(re.split(r"(\*\*.+?\*\*)", caption)):
        if not chunk:
            continue
        r = p.add_run(chunk.strip("*"))
        r.font.size = Pt(8.5)
        r.font.name = "Times New Roman"
        r.bold = chunk.startswith("**")
        r.font.color.rgb = MUTED if not chunk.startswith("**") else ACCENT


def table(doc, caption, header, rows, widths=None):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(8)
    p.paragraph_format.space_after = Pt(3)
    r = p.add_run(caption)
    r.bold = True; r.font.size = Pt(8.5); r.font.name = "Times New Roman"

    t = doc.add_table(rows=1, cols=len(header))
    t.style = "Table Grid"
    for i, txt in enumerate(header):
        c = t.rows[0].cells[i]
        c.text = ""
        run = c.paragraphs[0].add_run(txt)
        run.bold = True; run.font.size = Pt(9); run.font.name = "Times New Roman"
        shd = OxmlElement("w:shd"); shd.set(qn("w:fill"), "E8EDF2")
        c._tc.get_or_add_tcPr().append(shd)
    for row in rows:
        cells = t.add_row().cells
        for i, txt in enumerate(row):
            cells[i].text = ""
            run = cells[i].paragraphs[0].add_run(txt)
            run.font.size = Pt(9); run.font.name = "Times New Roman"
    if widths:
        for r_ in t.rows:
            for i, w in enumerate(widths):
                r_.cells[i].width = Inches(w)
    doc.add_paragraph().paragraph_format.space_after = Pt(6)
    return t


def build() -> None:
    doc = Document()
    for s in doc.sections:
        s.top_margin = s.bottom_margin = Inches(0.9)
        s.left_margin = s.right_margin = Inches(1.0)
    _style(doc)

    # ── title block ──
    p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run("What the Label Unit Hides: An Audited Multi-Hospital "
                  "Benchmark for ICROP Retinopathy of Prematurity Staging")
    r.bold = True; r.font.size = Pt(15); r.font.name = "Times New Roman"
    p.paragraph_format.space_after = Pt(8)

    p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run("P. Nagpure, R. Bait, P. Rangnekar, Y. Shengale, "); r.font.size = Pt(10.5)
    r2 = p.add_run("[CLINICAL CO-AUTHOR — required before submission]")
    r2.font.size = Pt(10.5); r2.bold = True; r2.font.color.rgb = ACCENT
    r3 = p.add_run(", A. Kumar"); r3.font.size = Pt(10.5)
    for rr in (r, r2, r3):
        rr.font.name = "Times New Roman"

    p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run("Vivekanand Education Society's Institute of Technology, Mumbai, India")
    r.italic = True; r.font.size = Pt(10); r.font.name = "Times New Roman"
    p.paragraph_format.space_after = Pt(4)

    p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run("Prepared for IEEE ISBI 2027 · 4-page limit · submission deadline "
                  "26 October 2026")
    r.font.size = Pt(8.5); r.font.color.rgb = MUTED; r.font.name = "Times New Roman"
    p.paragraph_format.space_after = Pt(14)

    # ── abstract ──
    h(doc, "ABSTRACT", 11, before=6)
    para(doc, ABSTRACT)
    para(doc, "Index Terms — Retinopathy of prematurity, ICROP staging, evaluation "
              "methodology, external validation, shortcut learning", italic=True, size=9.5)

    # ── 1 ──
    h(doc, "1.  INTRODUCTION")
    for t_ in INTRO:
        para(doc, t_)

    # ── 2 ──
    h(doc, "2.  MATERIALS AND METHODS")
    figure(doc, "fig1_cohort", FIG1_CAP, width=3.9)
    h(doc, "2.1  Corpus and inclusion", 10.5, before=8)
    para(doc, M_CORPUS)
    h(doc, "2.2  Reference standard", 10.5, before=8)
    para(doc, M_REF)
    h(doc, "2.3  Splits", 10.5, before=8)
    para(doc, M_SPLIT)
    h(doc, "2.4  Shortcut audit", 10.5, before=8)
    para(doc, M_AUDIT)
    h(doc, "2.5  Models", 10.5, before=8)
    para(doc, M_MODELS)
    h(doc, "2.6  Evaluation", 10.5, before=8)
    para(doc, M_EVAL)

    # ── 3 ──
    h(doc, "3.  RESULTS")
    h(doc, "3.1  The evaluation unit decides the result", 10.5, before=8)
    figure(doc, "fig2_unit_effect", FIG2_CAP, width=6.4)
    para(doc, R_UNIT_1)
    table(doc, "Table 1.  Locked half of the held-out hospital, scored once. "
               "Mean ± SD over five checkpoints.",
          ["Arm", "QWK per image", "QWK per eye", "Δ"],
          [["Structured, s42", "0.523 ± 0.018", "0.801 ± 0.028", "+0.278"],
           ["Structured, s1337", "0.509 ± 0.016", "0.791 ± 0.013", "+0.282"],
           ["Flat, s42", "0.561 ± 0.023", "0.795 ± 0.007", "+0.234"],
           ["Flat, s1337", "0.553 ± 0.018", "0.793 ± 0.021", "+0.240"]],
          widths=[1.9, 1.5, 1.5, 0.9])
    para(doc, R_UNIT_2)
    figure(doc, "fig3_confusion", FIG3_CAP, width=4.0)
    para(doc, R_UNIT_3)
    h(doc, "3.2  The architectures are indistinguishable", 10.5, before=8)
    para(doc, R_ARCH_1); para(doc, R_ARCH_2)
    h(doc, "3.3  A naive site probe understates residual site information", 10.5, before=8)
    para(doc, R_PROBE_1)
    table(doc, "Table 2.  Site-probe accuracy versus adversary weight "
               "(fold-0 validation, patient-clustered 95% CI).",
          ["Adversary weight", "Naive probe", "Disease-controlled probe"],
          [["w = 0.0", "0.790 [0.725–0.882]", "0.866 [0.807–0.917]"],
           ["w = 0.5", "0.809 [0.751–0.889]", "0.852 [0.788–0.909]"],
           ["w = 2.0", "0.867 [0.821–0.933]", "0.917 [0.869–0.958]"]],
          widths=[1.6, 2.1, 2.1])
    para(doc, R_PROBE_2)
    h(doc, "3.4  External validation retires a model that looked excellent at home",
      10.5, before=8)
    para(doc, R_EXT_1); para(doc, R_EXT_2)

    # ── 4-7 ──
    h(doc, "4.  DISCUSSION")
    for t_ in DISCUSSION:
        para(doc, t_)
    h(doc, "5.  CONCLUSION")
    para(doc, CONCLUSION)
    h(doc, "6.  COMPLIANCE WITH ETHICAL STANDARDS")
    para(doc, ETHICS)
    h(doc, "7.  ACKNOWLEDGMENTS")
    para(doc, ACK)

    h(doc, "8.  REFERENCES")
    for ref in REFS:
        p = doc.add_paragraph()
        p.paragraph_format.space_after = Pt(4)
        p.paragraph_format.left_indent = Inches(0.32)
        p.paragraph_format.first_line_indent = Inches(-0.32)
        r = p.add_run(ref); r.font.size = Pt(9); r.font.name = "Times New Roman"

    OUT.parent.mkdir(parents=True, exist_ok=True)
    doc.save(OUT)
    print(f"wrote {OUT.relative_to(ROOT)}  ({OUT.stat().st_size/1024:.0f} KB)")


# ── text, kept at the bottom so the layout code above stays readable ─────────
ABSTRACT = (
    "Automated ICROP staging for retinopathy of prematurity (ROP) is usually reported as an "
    "accuracy number, and that number is usually decided before the architecture is. We "
    "assemble a six-class staging corpus of 8,024 images from 1,760 infants across four "
    "cohorts, gate it with a five-part shortcut audit, and evaluate under patient-grouped "
    "five-fold cross-validation plus a held-out hospital whose second half was scored exactly "
    "once under a pre-registered protocol. Three results follow. First, the evaluation unit "
    "dominates: on the locked set the same predictions score quadratic-weighted κ of "
    "0.51–0.56 per image and 0.79–0.80 per eye, a gap that holds across two "
    "architectures and two seeds and is therefore a property of the labels rather than of the "
    "model. The misses are anatomically directed — posterior-pole frames of diseased eyes "
    "are called normal 71.7% of the time against 12.7% for temporal frames — because a "
    "frame that does not contain the ridge is, as an image, normal. Second, a taxonomy-faithful "
    "ordinal head and a flat softmax are indistinguishable, and the sign of their difference "
    "reverses between random seeds. Third, a screening head with 0.970 internal AUC falls to "
    "0.691 externally and flags 663 of 663 images. We report the negative results because they "
    "are the finding.")

INTRO = [
    "Retinopathy of prematurity is a leading preventable cause of childhood blindness. The "
    "constraint on screening it is not imaging capacity but the number of clinicians able to "
    "read the images inside the narrow window where treatment still works, which is why "
    "automated ICROP staging has attracted a steady stream of deep-learning systems reporting "
    "strong image-level metrics.",
    "We argue that a large part of those metrics is decided before any architecture is chosen, "
    "by three methodological choices that are rarely stated and never ablated: the unit the "
    "labels actually describe, the probe used to claim site-invariance, and the granularity of "
    "the data split. Each moves a reported number further than the architectural contribution "
    "in the same paper.",
    "The label unit is the most consequential. ROP annotations are frequently not image-level "
    "facts: in several corpora, including the one we use, a diagnosis is recorded per "
    "examination session and inherited by every frame captured in it, many of them peripheral "
    "or poorly centred views that do not show the finding. Scoring per image against such "
    "labels charges the model for annotations that were never about those frames. The "
    "literature contains the symptom without the diagnosis. Wang et al. [1] report "
    "referral-warranted ROP at image, eye and patient level and observe F1 falling 0.956 → "
    "0.915 → 0.898; the unit is acknowledged, the effect is small, and it points downward. "
    "We find the opposite sign at roughly three times the magnitude, and we show why: when the "
    "label's native unit is the eye, per-image scoring does not merely lose precision, it "
    "systematically understates agreement.",
    "Split granularity is the second. A recent study [2] applies EfficientNetB4 to the same "
    "Ostrava corpus [3] we use and reports 0.98 stage accuracy and 1.000 test accuracy for plus "
    "disease under an 80/16/4 image-level split of a dataset averaging roughly 32 frames per "
    "infant, with no external validation and no device-stratified analysis despite three camera "
    "systems; its own limitations section names the leakage risk. We treat it as a control "
    "condition rather than as a competitor: it is what this corpus produces when the audit is "
    "skipped.",
    "Site invariance is the third. Hospitals differ by camera, so hospital identity is written "
    "into every image; the standard remedy is domain-adversarial training [4] and the standard "
    "evidence is a linear probe whose accuracy falls. That probe is confounded by disease "
    "prevalence, which also varies by site.",
    "Our contributions are: (i) an audited multi-hospital ICROP staging benchmark with "
    "patient-grouped folds, a metadata-only performance floor, and a locked external half "
    "scored once; (ii) a quantification of the evaluation-unit effect, decomposed into "
    "population re-weighting and frame aggregation and shown to be model-independent; and (iii) "
    "two negative results — architectural equivalence, and a disease-controlled site probe "
    "that never improves — reported at the same weight as the positive ones.",
]

FIG1_CAP = ("**Fig. 1.**  Cohort cascade. Every exclusion is counted and the reasons are "
            "required to sum to the observed drop. The largest single exclusion is clinical: "
            "1,295 images labelled ROP-positive that record other pathology, i.e. 43% of this "
            "corpus's binary positive class.")
FIG2_CAP = ("**Fig. 2.**  (a) The same predictions scored at two units on the locked half "
            "(n = 661 images, 230 eyes, 116 infants), for two architectures at two seeds. Bars "
            "are means over five checkpoints; error bars are their standard deviation. Nothing "
            "about any model changes between the hatched and solid bars. (b) The anatomical "
            "cause, for stage 1–3 frames of confirmed-diseased eyes: frames aimed away "
            "from the temporal periphery, where the ROP ridge grows, are called Normal several "
            "times more often.")
FIG3_CAP = ("**Fig. 3.**  Locked-set confusion (row %, five checkpoints pooled; row n in "
            "parentheses). Errors concentrate in the Normal column (boxed) rather than around "
            "the diagonal. Stage 4/5 and AP-ROP rows are too small to read as performance and "
            "are reported as not externally validated.")

M_CORPUS = ("We pooled 13,037 images from eleven sources and reduced them to 10,427 "
            "stage-labelled images from 2,015 infants, then applied clinical exclusions: 1,295 "
            "images labelled ROP-positive that record other pathology (haemorrhage, "
            "toxoplasmosis, hamartoma, optic-nerve hypoplasia), 1,063 treated or regressed eyes "
            "whose post-treatment appearance is not a stage, and 45 stage-0 images too few to "
            "support a class. The eligible corpus is 8,024 images from 1,760 infants in six "
            "classes (Normal 3,837; S1 577; S2 959; S3 620; S4/5 89; AP-ROP 618); Fig. 1 gives "
            "the cascade. That first exclusion deserves emphasis: **43% of this corpus's binary "
            "ROP-positive class is not ROP.** A model trained on the uncorrected labels is an "
            "abnormal-retina detector reported as an ROP result.")
M_REF = ("Stages follow ICROP-3 [5], with AP-ROP treated as an aggressive form outside the "
         "stage ladder rather than as a rung on it. Labels are as released by the source "
         "institutions; we did not re-grade, and we state plainly that this is the study's "
         "principal limitation. Independent adjudication of a stratified subsample by a "
         "paediatric retina specialist is in progress.")
M_SPLIT = ("Folds are patient-grouped (five-fold, class-stratified), and a fourth cohort from a "
           "hospital contributing to no fold is held out entirely: 1,324 images from 232 "
           "infants, divided into a development half (663) used during method selection and a "
           "locked half (661 images, 116 infants) opened once, under a protocol committed to "
           "version control beforehand. It has not been re-scored since.")
M_AUDIT = ("Five gates run before any model is trained: (G1) patient leakage across folds "
           "— zero; (G2) exact and perceptual duplicate leakage, MD5 plus a dHash "
           "calibrated against known byte-duplicates — zero; (G3) site decodability from "
           "trivial metadata — balanced accuracy 0.997 against a chance rate of 0.333, "
           "which names the confound rather than removing it; (G4) a metadata-only classifier "
           "using no pixels, whose cross-validated macro-F1 of 0.187 is the floor every image "
           "model must clear; and (G5) a disease-controlled embedding probe (§3.3). On an "
           "earlier binary split, image dimensions alone reached AUC 0.911 — the shortcut "
           "is not hypothetical [6].")
M_MODELS = ("All arms share an ImageNet-initialised EfficientNetV2-S backbone [7] and identical "
            "preprocessing (circle-crop, CLAHE, letterbox, 384²). The structured head has "
            "three parts: a CORN ordinal branch [8] over stages with AP-ROP images masked from "
            "the ordinal loss, since ICROP-3 gives that form no stage; an independent AP-ROP "
            "sigmoid branch; and a DANN site adversary behind a gradient-reversal layer [4]. "
            "The flat baseline is a six-way softmax. Both are trained at natural class "
            "prevalence — an earlier version of this comparison varied the sampler as well "
            "as the head, and the sampling effect was five times the apparent architecture "
            "effect. Checkpoints are last-epoch by protocol, never selected by validation "
            "argmax.")
M_EVAL = ("The primary metric is quadratic-weighted κ (QWK). Because the human ROP "
          "literature reports unweighted κ, we report both: quadratic weighting inflates "
          "ordinal agreement when most errors are adjacent, and reporting only QWK would make "
          "our numbers incomparable to the only benchmark that matters. For context, "
          "inter-expert agreement on ROP stage among seven graders is Fleiss κ = 0.24, "
          "intra-expert κ = 0.56, with a 40% stage disagreement rate [9, 10]. Metrics are "
          "computed per image (primary) and per eye (co-primary), the latter by taking the most "
          "severe prediction across an eye's frames — the same operation by which the "
          "labels were assigned. Locked-set figures are means over five checkpoints with their "
          "standard deviation.")

R_UNIT_1 = ("Table 1 and Fig. 2(a) give the locked-set result. Nothing about any model changes "
            "between the two columns; only the unit the score is computed over changes, and it "
            "moves QWK by 0.23–0.28 in every cell — larger than any difference between "
            "the architectures, the seeds, or the two combined. **This is a property of the "
            "labels, not of the network.**")
R_UNIT_2 = ("The mechanism is anatomical rather than statistical. The ROP ridge grows in the "
            "retinal periphery, so whether a frame contains the finding depends on where the "
            "camera was pointed. Frames of confirmed-diseased eyes captured at the posterior "
            "pole are called Normal 71.7% of the time; temporal frames of the same eyes, which "
            "look at where the ridge sits, are called Normal 12.7% of the time (Fig. 2(b)). The "
            "pooled confusion matrix says the same thing from the other side (Fig. 3): 73% of "
            "stage-1 frames, 44% of stage-2 and 37% of stage-3 fall into the Normal column, "
            "while the Normal row is 99% correct. Errors are not scattered around the diagonal "
            "the way genuine ordinal confusion would scatter them; they concentrate in one "
            "column, which is the signature of frames that inherited a label their pixels do "
            "not carry.")
R_UNIT_3 = ("The same caution cuts the other way on AP-ROP, where the unit effect is real but "
            "is mostly not what it appears to be. Pooled out-of-fold, recall rises from 0.345 "
            "per image to 0.840 per session (100/119 units, CI 0.76–0.90; 0.832 at seed "
            "1337). Decomposed, +0.348 of that is population re-weighting — one infant "
            "supplies 76% of AP-ROP images — and only +0.147 is frame aggregation. Roughly "
            "70% of a jump routinely attributed to aggregation belongs to corpus composition "
            "instead.")
R_ARCH_1 = ("Under five-fold CV the structured head scores macro-F1 0.692 ± 0.086 against "
            "the flat baseline's 0.698 ± 0.099 at seed 42, and 0.708 ± 0.097 against "
            "0.686 ± 0.109 at seed 1337. The difference is −0.0055 at one seed and "
            "+0.0224 at the other — opposite signs, an order of magnitude below the "
            "per-fold spread of ±0.09. On the locked set the flat head is in fact ahead on "
            "image-level QWK and behind on eye-level, by margins inside one standard deviation.")
R_ARCH_2 = ("We report this because our own earlier draft did not. A single fold showed the "
            "structured head failing on AP-ROP, reproducibly, across five backbones; it "
            "dissolved under cross-validation, because that fold's validation split held the "
            "one heavily imaged, session-labelled infant. Five architectures failed identically "
            "because they shared an evaluation unit, not a defect.")
R_PROBE_1 = ("Domain-adversarial training is normally validated with a linear probe over all "
             "classes. Because disease prevalence varies by site, that probe can score well by "
             "reading disease. We ran both, the controlled version on normal-class images only, "
             "with patient-clustered bootstrap confidence intervals (2,000 replicates, paired "
             "across arms); Table 2 gives the dose–response.")
R_PROBE_2 = ("Two readings. The controlled probe sits above the naive one at every strength, so "
             "a paper reporting only the naive probe understates the site information its "
             "features retain. And neither probe falls as the adversary strengthens: site "
             "appearance is not removed at any setting we tested. We report the branch as a "
             "negative result.")
R_EXT_1 = ("A binary ResNet50 screening head reached sensitivity 0.990 and AUC 0.970 on its own "
           "hospitals. Scored at the held-out hospital with its deployed threshold unchanged, "
           "the lowest score it assigned any image (0.279) exceeded its own decision threshold "
           "(0.193): it flagged 663 of 663 images, including all 150 healthy eyes, for "
           "specificity 0.000 and AUC 0.691. This is the documented failure mode of cross-camera "
           "transfer — Chen et al. [11] report AUROC falling 0.99 → 0.62 in the same "
           "direction — and it is invisible to any internal metric.")
R_EXT_2 = ("We re-based screening onto the staging model's any-ROP probability using a rule "
           "fixed before the curves were drawn: the largest threshold holding sensitivity "
           "≥ 0.90 at both sites. It yields sensitivity 0.958 / specificity 0.781 "
           "internally and 0.905 / 0.387 externally, against the retired head's 1.000 / 0.000. "
           "The external half used for selection makes those two numbers in-sample and "
           "optimistic; we say so wherever they appear.")

DISCUSSION = [
    "The unit an ROP model is scored over changes its headline agreement by more than "
    "architecture, initialisation, or both together. That is this paper's central claim, and "
    "the locked-set design is what licenses it: four cells, two architectures, two seeds, one "
    "opening, same direction and similar magnitude.",
    "The practical consequence is that image-level ICROP metrics are not comparable across "
    "studies unless the label's native unit is stated. A system reporting 0.98 accuracy under "
    "an image-level split on session-labelled data [2] and a system reporting 0.52 QWK per "
    "image on the same source are not necessarily describing different models; they may be "
    "describing different denominators. Our own numbers are unremarkable next to the published "
    "state of the art, and we think that is the honest position rather than a shortfall: "
    "against a human inter-expert Fleiss κ of 0.24 on this task [9], reported model "
    "agreements above 0.95 warrant scrutiny rather than celebration. This is the "
    "hidden-stratification problem [12] in a form the label schema creates directly.",
    "Two of our three headline results are negative. The adversarial branch does not remove "
    "site information; the taxonomy-faithful head does not beat a flat softmax. We report them "
    "at full weight because the alternative — reporting the seed that agrees with us "
    "— was available, cheap, and present in an earlier version of this manuscript.",
    "**Limitations.**  First, we did not re-grade the labels; the 43% non-ROP contamination in "
    "the positive class is documented from source records rather than adjudicated, and expert "
    "re-grading is in progress. Second, stage 4/5 (n = 89) and AP-ROP are not externally "
    "validated — too few cases reach the held-out hospital — and we mark them so "
    "rather than quoting them, a decision taken before the locked set was opened. Third, the "
    "re-based screening threshold was selected using the development half of the held-out "
    "hospital, so its external figures are in-sample for that choice; confirmation requires a "
    "third site. Fourth, our corpus carries no zone or plus-disease annotation, so we cannot "
    "address the criteria on which treatment decisions are actually made. Fifth, the AP-ROP "
    "session analysis rests on genuine multi-frame structure from a single infant and is "
    "reported as a methodological caution, not a benchmark.",
]
CONCLUSION = ("On a six-class ICROP staging benchmark with patient-grouped folds and an external "
              "hospital scored once, the evaluation unit moves quadratic-weighted κ from "
              "~0.53 to ~0.80 across four independent model–seed cells, and the errors it "
              "explains are anatomically directed rather than random. Architecture choice, over "
              "the same corpus, moves nothing that survives a change of random seed. We suggest "
              "that ICROP staging results should state the unit their labels describe, report "
              "both image- and eye-level metrics, and validate externally before a threshold is "
              "deployed.")
ETHICS = ("This research used publicly available de-identified retinal image datasets and one "
          "de-identified institutional corpus released for research use. No new patient data "
          "were collected and no identifiable information was accessed. Ethical approval was "
          "not required, as confirmed by the licence terms of each constituent dataset.")
ACK = ("The authors thank University Hospital Ostrava for releasing the infant retinal image "
       "database used in this work. The authors declare no conflicts of interest. No funding "
       "was received.")

REFS = [
    "[1]  J. Wang, J. Ji, M. Zhang, et al., “Automated explainable multidimensional deep "
    "learning platform of retinal images for retinopathy of prematurity screening,” JAMA "
    "Netw. Open, vol. 4, no. 5, e218758, 2021.",
    "[2]  M. Vahidmoghadam, P. Ghorbani, M. J. Ahmadi, et al., “Automated diagnosis of plus "
    "form and early stages of ROP using deep learning models,” Sci. Rep., 2026.",
    "[3]  J. Timkovič, et al., “Retinal image dataset of infants and retinopathy of "
    "prematurity,” Sci. Data, vol. 11, 814, 2024.",
    "[4]  Y. Ganin, E. Ustinova, H. Ajakan, et al., “Domain-adversarial training of neural "
    "networks,” J. Mach. Learn. Res., vol. 17, no. 59, pp. 1–35, 2016.",
    "[5]  M. F. Chiang, G. E. Quinn, A. R. Fielder, et al., “International classification of "
    "retinopathy of prematurity, third edition,” Ophthalmology, vol. 128, no. 10, pp. "
    "e51–e68, 2021.",
    "[6]  A. J. DeGrave, J. D. Janizek, and S.-I. Lee, “AI for radiographic COVID-19 "
    "detection selects shortcuts over signal,” Nat. Mach. Intell., vol. 3, pp. "
    "610–619, 2021.",
    "[7]  M. Tan and Q. V. Le, “EfficientNetV2: Smaller models and faster training,” in "
    "Proc. Int. Conf. Machine Learning (ICML), 2021, pp. 10096–10106.",
    "[8]  X. Shi, W. Cao, and S. Raschka, “Deep neural networks for rank-consistent ordinal "
    "regression based on conditional probabilities,” Pattern Anal. Appl., vol. 26, pp. "
    "941–955, 2023.",
    "[9]  A. Gschließer, et al., “Inter-expert and intra-expert agreement on the "
    "diagnosis and treatment of retinopathy of prematurity,” Am. J. Ophthalmol., vol. 160, "
    "no. 3, pp. 553–560, 2015.",
    "[10]  J. P. Campbell, et al., “Expert diagnosis of plus disease in retinopathy of "
    "prematurity from computer-based image analysis,” Ophthalmology, vol. 123, no. 8, pp. "
    "1795–1801, 2016.",
    "[11]  J. S. Chen, A. S. Coyner, S. Ostmo, et al., “Deep learning for the diagnosis of "
    "stage in retinopathy of prematurity: Accuracy and generalizability across populations and "
    "cameras,” Ophthalmol. Retina, vol. 5, no. 10, pp. 1027–1035, 2021.",
    "[12]  L. Oakden-Rayner, J. Dunnmon, G. Carneiro, and C. Ré, “Hidden stratification "
    "causes clinically meaningful failures in machine learning for medical imaging,” in "
    "Proc. ACM Conf. Health, Inference, and Learning (CHIL), 2020, pp. 151–159.",
]

def build_paper2() -> None:
    """Word copy of paper 2 (docs/paper/paper2_pretraining.tex -> RetinAI_Paper2_Pretraining_draft.docx): pandoc on a
    simplified copy (PNG figures from docs/paper/figs2, numeric citations, a plain author block, FILL markers
    highlighted), then Times New Roman, numbered captions and Table Grid through python-docx."""
    import re
    import shutil
    import subprocess
    import tempfile
    from docx.enum.text import WD_COLOR_INDEX

    paper = ROOT / "docs" / "paper"
    out = paper / "RetinAI_Paper2_Pretraining_draft.docx"
    src = (paper / "paper2_pretraining.tex").read_text()
    with tempfile.TemporaryDirectory() as tmp:
        W = Path(tmp)
        for f in (paper / "figs2").glob("*.png"):
            shutil.copy(f, W / f.name)
        keys = re.findall(r"\\bibitem\{([^}]+)\}", src)
        num = {k: i + 1 for i, k in enumerate(keys)}
        s = re.sub(r"~?\\cite\{([^}]+)\}",
                   lambda m: " [" + ", ".join(str(num[k.strip()]) for k in m.group(1).split(",")) + "]", src)
        s = re.sub(r"\\bibitem\{([^}]+)\}", lambda m: "\\item[{[" + str(num[m.group(1)]) + "]}] ", s)
        s = s.replace("\\begin{thebibliography}{99}", "\\section*{References}\n\\begin{description}")
        s = s.replace("\\end{thebibliography}", "\\end{description}")
        s = s.replace(".pdf}", ".png}").replace("\\graphicspath{{figs2/}}", "")
        s = re.sub(r"\\usepackage\{fontspec\}\n\\setmainfont\{[^\n]*\n", "", s)
        s = s.replace("\\newcommand{\\FILL}[1]{\\textcolor{red}{\\textbf{[FILL: #1]}}}", "\\newcommand{\\FILL}[1]{\\textbf{[FILL: #1]}}")
        s = s.replace("\\documentclass[conference]{IEEEtran}", "\\documentclass{article}").replace("\\IEEEoverridecommandlockouts", "")
        a1 = src.split("\\IEEEauthorblockN{", 1)[1].split("}\n\\IEEEauthorblockA{", 1)
        a2 = a1[1].split("}}\n\n\\begin{document}", 1)[0]
        s = re.sub(r"\\author\{.*?\n\\begin\{document\}", lambda m: "\\begin{document}", s, flags=re.S)
        s = s.replace("\\begin{document}", "\\author{" + a1[0].replace("\\\\", " ") + " \\\\ " + a2 + "}\n\\begin{document}", 1)
        s = s.replace("\\begin{IEEEkeywords}", "\\textbf{Index Terms}---").replace("\\end{IEEEkeywords}", "")
        s = s.replace("figure*", "figure").replace("table*", "table").replace("[!t]", "")
        (W / "paper2_docx.tex").write_text(s)
        r = subprocess.run(["pandoc", "paper2_docx.tex", "-o", str(W / "p.docx"), "--resource-path=.", "--number-sections"],
                           cwd=W, capture_output=True, text=True)
        if r.returncode:
            raise SystemExit(r.stderr)
        doc = Document(str(W / "p.docx"))

    def paras():
        yield from doc.paragraphs
        for t in doc.tables:
            for row in t.rows:
                for c in row.cells:
                    yield from c.paragraphs
    n = 0
    for p in paras():
        for r_ in p.runs:
            if "FILL:" in r_.text:
                r_.font.highlight_color = WD_COLOR_INDEX.YELLOW
                n += 1
            r_.font.name = "Times New Roman"
    for st in doc.styles:
        try:
            if st.font is not None:
                st.font.name = "Times New Roman"
            if st.name in ("Normal", "Body Text", "First Paragraph", "Compact", "Abstract"):
                st.font.size = Pt(10)
        except Exception:  # noqa: BLE001  some latent styles have no font
            pass
    if "Table Grid" in [x.name for x in doc.styles]:   # pandoc's reference.docx may not ship it
        for t in doc.tables:
            t.style = doc.styles["Table Grid"]
    for sh in doc.inline_shapes:
        if sh.width > Inches(6.5):
            ratio = Inches(6.5) / sh.width
            sh.width, sh.height = Inches(6.5), int(sh.height * ratio)
    fig = 0
    for p in doc.paragraphs:
        if p.style.name == "Image Caption" and p.runs:
            fig += 1
            p.runs[0].text = f"Fig. {fig}. " + p.runs[0].text
        elif p.style.name == "Table Caption" and p.runs:
            p.runs[0].text = "TABLE I. " + p.runs[0].text
    doc.save(str(out))
    print(f"{out.relative_to(ROOT)}: {n} FILL runs highlighted, {len(doc.inline_shapes)} figures, {len(doc.tables)} table(s)")


if __name__ == "__main__":
    import sys
    build_paper2() if "--paper2" in sys.argv else build()
