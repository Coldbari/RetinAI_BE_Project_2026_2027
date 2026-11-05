"""Build the ROP staging preprint as an editable Word document.

    .venv/bin/python scripts/make_preprint_docx.py

Produces docs/paper/RetinAI_ROP_preprint_draft.docx — the full-length version of the
ISBI manuscript, rewritten for a preprint server (TechRxiv) and a later journal
submission. It differs from the ISBI text in three ways:

  * every number was re-verified against the committed artifacts (23 corrections,
    including the September evidence-package corrections the ISBI .tex never received);
  * the reviewers' points from the mock review are applied: the exam-level prior art is
    cited and conceded, the novelty claim is narrowed, uncertainty is reported as
    patient-clustered intervals rather than checkpoint spread, and the tone is neutral;
  * the new statistics come from results/rop/preprint_stats.json
    (scripts/rop_preprint_stats.py, CPU-only re-scoring of saved predictions).

Text the authors must supply is highlighted yellow in the document. Figures are the
committed graphs/isbi PNGs. Layout helpers are shared with scripts/make_paper_docx.py.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_COLOR_INDEX
from docx.shared import Inches, Pt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from scripts.make_paper_docx import ACCENT, MUTED, _style, figure, h, table  # noqa: E402

OUT = ROOT / "docs" / "paper" / "RetinAI_ROP_preprint_draft.docx"
STATS = json.loads((ROOT / "results" / "rop" / "preprint_stats.json").read_text())


def para(doc, text, justify=True, italic=False, size=10, after=6):
    """One paragraph. **bold**, *italic* and [[author-supplied text]] (yellow) are honoured."""
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(after)
    if justify:
        p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    for chunk in re.split(r"(\*\*.+?\*\*|\[\[.+?\]\]|\*.+?\*)", text):
        if not chunk:
            continue
        if chunk.startswith("[["):
            r = p.add_run(chunk[2:-2])
            r.font.highlight_color = WD_COLOR_INDEX.YELLOW
        else:
            r = p.add_run(chunk.strip("*"))
            r.bold = chunk.startswith("**")
            r.italic = italic or (chunk.startswith("*") and not chunk.startswith("**"))
        r.font.size = Pt(size)
        r.font.name = "Times New Roman"
    return p


def sgn(x, nd=3):
    """Signed number with a typographic minus."""
    return f"{x:+.{nd}f}".replace("-", "−")


def ci(pair, nd=3):
    return f"[{pair[0]:.{nd}f}, {pair[1]:.{nd}f}]".replace("-", "−")


def pct(x, nd=1):
    return f"{100 * x:.{nd}f}%"


# ── text ────────────────────────────────────────────────────────────────────────
TITLE = ("What the Label Unit Hides: An Audited Multi-Hospital Benchmark for ICROP "
         "Staging of Retinopathy of Prematurity")

INTRO = [
    "Retinopathy of prematurity (ROP) is a leading cause of preventable childhood blindness "
    "worldwide. The constraint on screening is rarely the camera; it is the number of "
    "clinicians able to read the images within the weeks in which treatment still works. "
    "That gap has motivated a steady stream of deep-learning systems for staging ROP "
    "according to the International Classification of ROP, third edition (ICROP-3) [1], "
    "most of which report their results per photograph.",

    "Clinically, however, stage is a property of an eye, read from the most advanced finding "
    "across all of the views taken of it. Public ROP corpora inherit that convention in their "
    "labels. In the Ostrava database [2] one diagnosis code is recorded per examination and "
    "copied to every frame captured in it; in the Multi-View ROP dataset [3] one label is "
    "recorded per eye and repeated on each of its posterior, nasal and temporal views. A "
    "posterior-pole photograph of an eye whose stage-2 ridge lies in the temporal periphery "
    "therefore carries the label *stage 2*, although nothing in its pixels can show it. "
    "Scored per image, a model that correctly calls that photograph normal is counted as wrong.",

    "The evaluation unit is not a new idea, and we do not claim it as one. DeepROP [4] "
    "predicted per eye by taking an element-wise maximum over the features of all images of "
    "that eye; J-PROP [5] reported referral-warranted ROP at image, eye and patient level "
    "(F1 0.956, 0.915 and 0.898 respectively); Peng et al. [6] evaluated staging both per "
    "image and per examination; and the i-ROP vascular severity score is applied at the "
    "examination level [7]. What has not, to our knowledge, been reported is how far the unit "
    "alone moves a multi-class staging metric when the predictions are held fixed, at a "
    "hospital the model never saw, and which photographs produce the difference.",

    "Two further choices shape published numbers on these corpora. The first is split "
    "granularity. The Ostrava corpus contains 6,004 images of 188 infants, about 32 images "
    "per infant [2], so an image-level split places photographs of the same infant on both "
    "sides of it. Vahidmoghadam et al. [8] used an image-level 80/16/4 split of this corpus, "
    "with a four-class stage task (stages 0–3; 880 images) evaluated on a 37-image test set; "
    "VEA-Net [9] grouped by subject across three centres, mapped aggressive posterior ROP "
    "(AP-ROP) into stage 4, and applied domain adaptation without measuring the site "
    "information that remained. The second is site. Hospitals differ by camera, so the "
    "hospital is written into every image; domain-adversarial training [10] is the standard "
    "remedy, and a linear probe whose accuracy falls is the standard evidence, but that probe "
    "is confounded whenever disease prevalence also differs by site.",

    "This paper makes four contributions. (i) An audited multi-source ICROP staging benchmark "
    "of four public cohorts, with patient-grouped folds, a metadata-only performance floor, "
    "and the defects we found disclosed rather than corrected silently. (ii) On a locked half "
    "of a held-out hospital, scored once under a pre-registered protocol, a measurement of the "
    "gap between image-level and eye-level agreement for identical predictions in four "
    "model–seed cells — a hypothesis formed on the development half and confirmed on the "
    "locked half — together with a view-based control that locates where the gap comes from. "
    "(iii) Two negative results reported with intervals: an ordinal, taxonomy-faithful head "
    "is indistinguishable from a flat softmax, and a domain-adversarial branch does not "
    "reduce site information under a disease-controlled probe. (iv) A cautionary external "
    "failure of a screening threshold. We do not propose a new architecture, and we make no "
    "claim of clinical validity.",
]

LIMITATIONS = [
    "**First,** the labels were used as released by the source institutions and were not "
    "re-graded; no clinician adjudicated any subset. The 43% non-ROP contamination we report "
    "is documented from the source diagnosis codes, not from re-reading the images.",
    "**Second,** patient identifiers in ROP-VL and Shenzhen behave like per-record rather than "
    "per-person keys (six byte-identical photographs carry two different Shenzhen "
    "identifiers). Infant counts for those sources are therefore upper bounds, and "
    "patient-clustered intervals computed on the cross-validation pool are narrower than true "
    "clustering would warrant. The held-out hospital, where the identifier is patient-like, is "
    "the least affected.",
    "**Third,** stage 4–5 and AP-ROP are not externally validated: the locked half contains 4 "
    "and 12 such images. All 89 stage 4–5 images in the cross-validation pool come from one "
    "source (ROP-VL), so performance on that class may partly reflect site.",
    "**Fourth,** the re-based screening threshold was selected using the development half of "
    "the held-out hospital, so its external figures are in-sample for that choice.",
    "**Fifth,** the corpus carries no zone annotation and no plus-disease label usable "
    "alongside stage, so we cannot address the criteria on which treatment decisions are made.",
    "**Sixth,** all models share one backbone family with ImageNet initialisation, and the "
    "held-out site is a single hospital. Ostrava records no eye laterality, so eye-level "
    "analyses on the cross-validation pool use the examination session instead.",
]

REFS = [
    "M. F. Chiang, G. E. Quinn, A. R. Fielder, et al., “International classification of "
    "retinopathy of prematurity, third edition,” *Ophthalmology*, vol. 128, no. 10, "
    "pp. e51–e68, 2021.",
    "J. Timkovič, et al., “Retinal image dataset of infants and retinopathy of "
    "prematurity,” *Sci. Data*, vol. 11, 814, 2024. Data: figshare collection 6626162 (CC0).",
    "Multi-View ROP dataset, Hugging Face lijuanliao/ROP-DATASET, CC BY-NC 4.0, accessed 2026.",
    "J. Wang, R. Ju, Y. Chen, et al., “Automated retinopathy of prematurity screening "
    "using deep neural networks,” *EBioMedicine*, vol. 35, pp. 361–368, 2018.",
    "J. Wang, J. Ji, M. Zhang, et al., “Automated explainable multidimensional deep "
    "learning platform of retinal images for retinopathy of prematurity screening,” "
    "*JAMA Netw. Open*, vol. 4, no. 5, e218758, 2021.",
    "Y. Peng, et al., “Automatic staging for retinopathy of prematurity with deep feature "
    "fusion and ordinal classification strategy,” *IEEE Trans. Med. Imaging*, vol. 40, "
    "no. 7, pp. 1750–1762, 2021.",
    "J. P. Campbell, S. J. Kim, J. M. Brown, et al., “Evaluation of a deep "
    "learning–derived quantitative retinopathy of prematurity severity scale,” "
    "*Ophthalmology*, vol. 128, no. 7, pp. 1070–1076, 2021.",
    "M. Vahidmoghadam, P. Ghorbani, M. J. Ahmadi, et al., “Automated diagnosis of plus "
    "form and early stages of ROP using deep learning models,” *Sci. Rep.*, vol. 16, "
    "7234, 2026.",
    "Y. Chen, J. Wan, T. Zhang, et al., “VEA-net: vascular enhancement attention with "
    "dual-backbone multi-task learning for comprehensive ROP management across multi-center "
    "datasets,” *Front. Cell Dev. Biol.*, vol. 14, 1865503, 2026.",
    "Y. Ganin, E. Ustinova, H. Ajakan, et al., “Domain-adversarial training of neural "
    "networks,” *J. Mach. Learn. Res.*, vol. 17, no. 59, pp. 1–35, 2016.",
    "Shenzhen ROP dataset, figshare, doi:10.6084/m9.figshare.25514449, CC BY 4.0, 2024.",
    "ROP-VL dataset, figshare, doi:10.6084/m9.figshare.30143461, CC BY 4.0, 2025.",
    "M. Tan and Q. V. Le, “EfficientNetV2: Smaller models and faster training,” in "
    "*Proc. Int. Conf. Machine Learning (ICML)*, 2021, pp. 10096–10106.",
    "X. Shi, W. Cao, and S. Raschka, “Deep neural networks for rank-consistent ordinal "
    "regression based on conditional probabilities,” *Pattern Anal. Appl.*, vol. 26, "
    "pp. 941–955, 2023.",
    "A. J. DeGrave, J. D. Janizek, and S.-I. Lee, “AI for radiographic COVID-19 detection "
    "selects shortcuts over signal,” *Nat. Mach. Intell.*, vol. 3, pp. 610–619, 2021.",
    "D. J. Schuirmann, “A comparison of the two one-sided tests procedure and the power "
    "approach for assessing the equivalence of average bioavailability,” *J. "
    "Pharmacokinet. Biopharm.*, vol. 15, no. 6, pp. 657–680, 1987.",
    "A. Gschließer, E. Stifter, T. Neumayer, et al., “Inter-expert and intra-expert "
    "agreement on the diagnosis and treatment of retinopathy of prematurity,” *Am. J. "
    "Ophthalmol.*, vol. 160, no. 3, pp. 553–560, 2015.",
    "J. P. Campbell, et al., “Diagnostic discrepancies in retinopathy of prematurity "
    "classification,” *Ophthalmology*, vol. 123, no. 8, pp. 1795–1801, 2016.",
    "L. Oakden-Rayner, J. Dunnmon, G. Carneiro, and C. Ré, “Hidden stratification "
    "causes clinically meaningful failures in machine learning for medical imaging,” in "
    "*Proc. ACM Conf. Health, Inference, and Learning (CHIL)*, 2020, pp. 151–159.",
    "J. S. Chen, A. S. Coyner, S. Ostmo, et al., “Deep learning for the diagnosis of "
    "stage in retinopathy of prematurity: Accuracy and generalizability across populations "
    "and cameras,” *Ophthalmol. Retina*, vol. 5, no. 10, pp. 1027–1035, 2021.",
]

REFS += [
    "C. Nadeau and Y. Bengio, “Inference for the generalization error,” *Mach. Learn.*, "
    "vol. 52, no. 3, pp. 239–281, 2003.",
]

# IEEE order = order of first citation in the text.
REFS = [REFS[i - 1] for i in (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
                              15, 13, 14, 21, 16, 20, 17, 18, 19)]

CELLS = ["structured_s42", "structured_s1337", "flat_s42", "flat_s1337"]
CELL_NAME = {"structured_s42": "Structured, seed 42", "structured_s1337": "Structured, seed 1337",
             "flat_s42": "Flat, seed 42", "flat_s1337": "Flat, seed 1337"}


def numbers() -> dict:
    """Every number the Results text quotes, read from the committed artifacts."""
    A = STATS["A_locked_clustered_bootstrap"]["cells"]
    D = STATS["D_head_equivalence_tost"]
    E = STATS["E_cohort"]
    F = STATS["F_arop_internal_control"]
    dev = STATS["B_dev_image_vs_eye_qwk"]["cells"]
    lt = json.loads((ROOT / "results/rop/locked_eval/locked_table.json").read_text())["cells"]
    probe = json.loads((ROOT / "results/rop/dann_probe_ci.json").read_text())
    reb = json.loads((ROOT / "results/rop/screening_rebase.json").read_text())
    ext = json.loads((ROOT / "results/rop/screening_external_spec.json").read_text())
    old = json.loads((ROOT / "results/rop/metrics.json").read_text())["test"]
    eq = json.loads((ROOT / "results/rop/matched_equivalence.json").read_text())["seeds"]

    def rng(vals, nd=2):
        return f"{min(vals):.{nd}f}–{max(vals):.{nd}f}"

    img = [A[c]["point_5ckpt"]["image"]["qwk"]["mean"] for c in CELLS]
    eye = [A[c]["point_5ckpt"]["eye"]["qwk"]["mean"] for c in CELLS]
    views = {v: [lt[c]["view_miss_table_ckpt_mean"][v]["called_normal_mean"] for c in CELLS]
             for v in ("posterior pole", "nasal", "temporal")}
    s42 = A["structured_s42"]
    f42 = F["seed_42"]["structured"]
    arms = {s: eq[s]["arms"] for s in ("42", "1337")}
    return dict(
        img_rng=rng(img), eye_rng=rng(eye),
        dev_img_rng=rng([dev[c]["image_qwk"]["mean"] for c in CELLS]),
        dev_eye_rng=rng([dev[c]["eye_qwk"]["mean"] for c in CELLS]),
        gap_rng=rng([A[c]["point_5ckpt"]["eye_minus_image_qwk"]["mean"] for c in CELLS]),
        post_rng=f"{100*min(views['posterior pole']):.1f}–{100*max(views['posterior pole']):.1f}%",
        nas_rng=f"{100*min(views['nasal']):.1f}–{100*max(views['nasal']):.1f}%",
        temp_rng=f"{100*min(views['temporal']):.1f}–{100*max(views['temporal']):.1f}%",
        miss_rng=f"{100*min(A[c]['point_5ckpt']['eye']['stage123_eyes_called_normal']['mean'] for c in CELLS):.1f}"
                 f"–{100*max(A[c]['point_5ckpt']['eye']['stage123_eyes_called_normal']['mean'] for c in CELLS):.1f}%",
        miss_s42=pct(s42["point_5ckpt"]["eye"]["stage123_eyes_called_normal"]["mean"]),
        miss_s42_ci=f"[{100*s42['ci95_clustered']['eye_stage123_called_normal']['ci95'][0]:.1f}, "
                    f"{100*s42['ci95_clustered']['eye_stage123_called_normal']['ci95'][1]:.1f}]%",
        img_miss_s42=pct(s42["point_5ckpt"]["image"]["stage123_images_called_normal"]["mean"]),
        kimg_rng=rng([A[c]["point_5ckpt"]["image"]["kappa_unweighted"]["mean"] for c in CELLS]),
        keye_rng=rng([A[c]["point_5ckpt"]["eye"]["kappa_unweighted"]["mean"] for c in CELLS]),
        k_cv42=STATS["C_unweighted_kappa"]["cv_pooled_from_kappa_unweighted_vs_human_json"]["values"]["seed_42"]["kappa_unweighted"],
        k_cv1337=STATS["C_unweighted_kappa"]["cv_pooled_from_kappa_unweighted_vs_human_json"]["values"]["seed_1337"]["kappa_unweighted"],
        eq_mean=D["mean_diff"], eq_naive=ci(D["naive_paired_t"]["ci95_t"]),
        eq_corr=ci(D["corrected_resampled_t"]["ci95_t"]),
        eq_corr_p=D["corrected_resampled_t"]["tost_p"], eq_naive_p=D["naive_paired_t"]["tost_p"],
        eq_max=max(p["diff_structured_minus_flat"] for p in D["pairs"]),
        cv_s42=f"{arms['42']['structured_natural']['cv_macro_f1_mean']:.3f} ± "
               f"{arms['42']['structured_natural']['cv_macro_f1_sd']:.3f}",
        cv_f42=f"{arms['42']['flat_natural']['cv_macro_f1_mean']:.3f} ± "
               f"{arms['42']['flat_natural']['cv_macro_f1_sd']:.3f}",
        cv_s1337=f"{arms['1337']['structured_natural']['cv_macro_f1_mean']:.3f} ± "
                 f"{arms['1337']['structured_natural']['cv_macro_f1_sd']:.3f}",
        cv_f1337=f"{arms['1337']['flat_natural']['cv_macro_f1_mean']:.3f} ± "
                 f"{arms['1337']['flat_natural']['cv_macro_f1_sd']:.3f}",
        arop_ost=pct(f42["ostrava_session_labelled"]["frac_below_0.1"]),
        arop_ost_ci=f"(95% CI {100*f42['ostrava_session_labelled']['ci95_clustered_by_sessions'][0]:.1f}–"
                    f"{100*f42['ostrava_session_labelled']['ci95_clustered_by_sessions'][1]:.1f}%)",
        arop_oth=pct(f42["non_ostrava_frame_labelled"]["frac_below_0.1"]),
        arop_oth_ci=f"(95% CI {100*f42['non_ostrava_frame_labelled']['ci95_clustered_by_units'][0]:.1f}–"
                    f"{100*f42['non_ostrava_frame_labelled']['ci95_clustered_by_units'][1]:.1f}%)",
        arop_oth_n=f42["non_ostrava_frame_labelled"]["infants"],
        p_naive=probe["paired_diffs"]["naive"], p_ctrl=probe["paired_diffs"]["controlled"],
        probe=probe["cells"],
        old_auc=old["auc"], old_ext_auc=reb["retired_model"]["auc_external_dev"],
        old_thr=reb["retired_model"]["threshold"], old_sens=reb["retired_model"]["internal_test"]["sensitivity"],
        old_spec=reb["retired_model"]["internal_test"]["specificity"], old_n=reb["retired_model"]["internal_test"]["n"],
        old_min=min(ext["score_distribution"]["screening_normal"]["min"],
                    ext["score_distribution"]["screening_rop"]["min"]),
        ext_n=ext["eval_set"]["n"], ext_normal=ext["eval_set"]["n_normal"],
        new=reb["new_model"], new_thr=reb["threshold"],
        totals=E["totals"], src=E["sources"],
    )


def build() -> None:
    N = numbers()
    A = STATS["A_locked_clustered_bootstrap"]["cells"]
    doc = Document()
    for s in doc.sections:
        s.top_margin = s.bottom_margin = Inches(0.9)
        s.left_margin = s.right_margin = Inches(1.0)
    _style(doc)

    # ── title block ──
    p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run(TITLE); r.bold = True; r.font.size = Pt(15); r.font.name = "Times New Roman"
    p.paragraph_format.space_after = Pt(8)
    para(doc, "[[P. Nagpure]]¹, [[R. Bait]]¹, [[P. Rangnekar]]¹, [[Y. Shengale]]¹, [[A. Kumar]]¹ "
              "[[— replace with full names in the agreed order]]", justify=False, size=10.5,
         after=2).alignment = WD_ALIGN_PARAGRAPH.CENTER
    para(doc, "¹ [[Department of ______]], Vivekanand Education Society's Institute of "
              "Technology (VESIT), Mumbai, India", justify=False, italic=True,
         after=2).alignment = WD_ALIGN_PARAGRAPH.CENTER
    para(doc, "Corresponding author: [[name]], [[college e-mail]] · ORCID [[0000-0000-0000-0000]]",
         justify=False, size=9, after=2).alignment = WD_ALIGN_PARAGRAPH.CENTER
    para(doc, "Preprint — not peer reviewed.", justify=False, size=8.5,
         after=14).alignment = WD_ALIGN_PARAGRAPH.CENTER

    # ── abstract ──
    h(doc, "Abstract", 11, before=6)
    para(doc,
         "**Background.** Deep-learning systems for staging retinopathy of prematurity (ROP) are "
         "usually evaluated per photograph, although several public corpora record the diagnosis "
         "per eye or per examination and copy it to every photograph taken. We measure how much "
         "of a reported staging metric that choice decides. "
         "**Methods.** We assembled a six-class ICROP-3 staging corpus (normal, stages 1, 2, 3, "
         "4–5, and aggressive posterior ROP kept outside the stage ladder) of 8,024 images from "
         "four public cohorts, ran a five-gate shortcut audit, and trained an ordinal head with an "
         "AP-ROP branch and a flat softmax head on one EfficientNetV2-S backbone at two seeds, "
         "under patient-grouped five-fold cross-validation. A held-out hospital (1,324 images, "
         "232 infants) was split by infant into a development half and a locked half (661 "
         "images, 230 eyes, 116 infants) that was scored once under a pre-registered protocol. "
         f"**Results.** On the locked half the same predictions scored quadratic-weighted κ "
         f"(QWK) {N['img_rng']} per image and {N['eye_rng']} per eye in all four model–seed "
         f"cells; the gap ({N['gap_rng']}) excluded zero in every cell. The gap is anatomically "
         f"directed: stage 1–3 images taken at the posterior pole were called normal "
         f"{N['post_rng']} of the time, against {N['temp_rng']} for temporal images, and scoring "
         "temporal images alone, one at a time and without aggregation, reproduced the eye-level "
         "agreement (QWK 0.799 versus 0.801). The two heads were indistinguishable, and a "
         "domain-adversarial branch did not reduce site information under a disease-controlled "
         f"probe. At eye level, {N['miss_rng']} of stage 1–3 eyes were still called normal. "
         "**Conclusions.** For ROP staging, the evaluation unit and the acquisition view moved "
         "reported agreement more than the choice of model head did. We recommend that staging "
         "studies state the unit their labels describe and report image- and eye-level metrics "
         "together.")
    para(doc, "*Keywords* — retinopathy of prematurity, ICROP staging, evaluation methodology, "
              "external validation, shortcut learning, deep learning", size=9.5)

    # ── 1 ──
    h(doc, "1.  Introduction")
    for t_ in INTRO:
        para(doc, t_)

    # ── 2 ──
    h(doc, "2.  Materials and Methods")
    h(doc, "2.1  Data sources", 10.5, before=8)
    t, src = N["totals"], N["src"]
    para(doc,
         "We used four public cohorts (Table 1). Ostrava [2] (Czech Republic; CC0) contributed "
         "images from three cameras with one diagnosis code per examination session. Shenzhen "
         "[11] and ROP-VL [12] (both Shenzhen Eye Hospital, China, different cohorts; CC BY 4.0) "
         "label each image. The Multi-View ROP dataset [3] (China; CC BY-NC 4.0) records the eye "
         "and the acquisition view of every image; it was never used for training and serves as "
         "the held-out hospital. Duplicates between ROP-VL and Shenzhen were removed with a "
         "256-bit perceptual hash calibrated against known byte-identical copies.")
    rows = []
    for key, name, unit in (("ostrava", "Ostrava", "per session"),
                            ("shenzhen", "Shenzhen", "per image"),
                            ("ropvl", "ROP-VL", "per image"),
                            ("multiview_dev", "Multi-View, dev half", "per eye (de facto)"),
                            ("multiview_locked", "Multi-View, locked half", "per eye (de facto)")):
        s = src[key]; c = s["per_class"]
        rows.append([name, f"{s['images']:,}", f"{s['infant_ids']:,}", unit,
                     f"{c['normal']:,} / {c['stage_1']:,} / {c['stage_2']:,} / {c['stage_3']} / "
                     f"{c['stage_4_5']} / {c['arop']}"])
    c = t["eligible_corpus"]["per_class"]
    rows.append(["Total", f"{t['eligible_corpus']['images']:,}",
                 f"{t['eligible_corpus']['infant_ids']:,}*", "",
                 f"{c['normal']:,} / {c['stage_1']:,} / {c['stage_2']:,} / {c['stage_3']} / "
                 f"{c['stage_4_5']} / {c['arop']}"])
    table(doc, "Table 1. Eligible images by source. Classes: normal / stage 1 / stage 2 / "
               "stage 3 / stage 4–5 / AP-ROP. *Patient-ID counts; for Shenzhen and ROP-VL the "
               "identifier behaves like a record rather than a person, so these are upper bounds "
               "on distinct infants.",
          ["Source", "Images", "Patient IDs", "Label unit", "Class counts"], rows,
          widths=[1.5, 0.65, 0.8, 1.15, 2.2])

    figure(doc, "fig1_cohort",
           "**Fig. 1.** Inclusion cascade. Every exclusion is counted, and the reasons sum to the "
           "observed drop. Infant counts are patient-ID counts (upper bounds; see Table 1).",
           width=3.6)
    h(doc, "2.2  Inclusion", 10.5, before=8)
    para(doc,
         "We acquired 13,037 images from eleven public sources and reduced them to 10,427 "
         "stage-labelled images from 2,015 patient IDs (Fig. 1). We then excluded 1,295 images "
         "whose diagnosis code records a condition other than ROP (haemorrhage, toxoplasmosis, "
         "hamartoma, optic-nerve hypoplasia); 1,063 images of treated or regressed eyes, whose "
         "appearance after treatment is not a stage; and 45 stage-0 images, too few to support a "
         f"class. The eligible corpus is {t['eligible_corpus']['images']:,} images from "
         f"{t['eligible_corpus']['infant_ids']:,} patient IDs. The first exclusion deserves "
         "emphasis: in the Ostrava release, 43% of the images in the binary *ROP-positive* "
         "class record a different diagnosis, so a binary model trained on those labels detects "
         "an abnormal infant retina rather than ROP.")
    h(doc, "2.3  Reference standard", 10.5, before=8)
    para(doc,
         "Stages follow ICROP-3 [1]. Aggressive posterior ROP (AP-ROP) is treated as a form outside "
         "the stage ladder rather than as a rung on it, and stages 4 and 5 are merged because "
         "together they number only 107 images. Labels are used as released by the source "
         "institutions; we did not re-grade them, and no clinician adjudicated any subset. The "
         "label unit differs by source: Ostrava stamps one code per examination session (both "
         "eyes) on every frame; Shenzhen and ROP-VL label single images; in Multi-View, 436 of "
         "460 eyes carry one label across all of their views, so the label is in practice a "
         "per-eye diagnosis.")
    h(doc, "2.4  Splits", 10.5, before=8)
    cv = t["cv_pool"]
    para(doc,
         f"The cross-validation pool ({cv['images']:,} images, {cv['infant_ids']:,} patient IDs; "
         "Ostrava, Shenzhen, ROP-VL) was divided into five patient-grouped, class-stratified "
         "folds. Normal images were capped at two per infant in the training folds only; "
         "validation folds keep their natural prevalence. The Multi-View cohort (1,324 images, "
         "232 infants) was split by infant into a development half (663 images), used during "
         "method selection, and a locked half (661 images, 116 infants, 230 eyes). The protocol "
         "for the locked half was committed to version control before any locked inference. "
         "One amendment was made after a dry run on the development half and before the locked "
         "half was opened: eye-level scoring was added as a co-primary analysis, because the "
         "development half had shown that its labels are per-eye diagnoses. The locked half was "
         "scored once, on 18 August 2026, and has not been re-scored; every locked-half interval "
         "below re-uses the saved predictions.")
    h(doc, "2.5  Shortcut audit", 10.5, before=8)
    para(doc,
         "Five gates were run before training. (G1) Patient leakage: no patient ID is shared "
         "across folds or between the pool and either held-out half. (G2) Duplicate leakage: no "
         "file path or byte-identical image is shared between the pool, the development half and "
         "the locked half. A later full-corpus hash found 10 exact-duplicate groups within the "
         "pool, four of which cross a fold boundary (8 images, 0.12% of the pool), three with "
         "conflicting labels; we report them rather than remove them, as they are too few to "
         "move any metric. (G3) Site decodability from trivial metadata: balanced accuracy 0.997 "
         "against chance 0.333, which names the confound rather than removing it [13]. (G4) A "
         "metadata-only classifier using no pixels reaches cross-validated macro-F1 0.187, the "
         "floor any image model must clear. (G5) A disease-controlled embedding probe (Section "
         "3.4).")
    h(doc, "2.6  Models and training", 10.5, before=8)
    para(doc,
         "All arms share an ImageNet-initialised EfficientNetV2-S backbone [14], selected from a "
         "five-backbone flat-softmax benchmark in which macro-F1 varied by only 0.028 across "
         "convolutional, transformer and hybrid backbones of 4–28 M parameters. Preprocessing is "
         "circle cropping, contrast-limited adaptive histogram equalisation (CLAHE), letterboxing "
         "and resizing to 384 × 384. The *structured* head has three parts: a CORN ordinal branch "
         "[15] over normal < stage 1 < stage 2 < stage 3 < stage 4–5, from whose loss AP-ROP "
         "images are masked because ICROP-3 gives that form no stage; an independent AP-ROP "
         "sigmoid branch with its threshold fixed at 0.5 a priori; and a domain-adversarial site "
         "branch behind a gradient-reversal layer [10]. The *flat* head is a six-way softmax. "
         "Both were trained at natural class prevalence for 20 epochs (AdamW, learning rate "
         "3 × 10⁻⁴, weight decay 0.05, one-cycle schedule, batch 16, mixed precision) with "
         "flips, rotations up to 15°, colour jitter and random resized crops, at seeds 42 and "
         "1337. The last-epoch weights are used; no checkpoint is selected on validation data.")
    h(doc, "2.7  Evaluation", 10.5, before=8)
    para(doc,
         "The primary metric is quadratic-weighted κ (QWK) over the six labels with AP-ROP at rank "
         "5; because the human ROP literature reports unweighted κ, we report that too. An eye is "
         "the pair (infant, left/right) from the Multi-View acquisition-view table. Its label is "
         "the most severe label among its views, and its prediction is the most severe predicted "
         "class among the same views — the same operation by which the labels were assigned. "
         "Locked-half figures are means over the five fold checkpoints. Intervals are 95% "
         "percentile intervals from 2,000 bootstrap replicates that resample infants, computing "
         "each metric per checkpoint and averaging within each replicate. The two heads are "
         "compared on the ten paired fold × seed macro-F1 differences, with the variance "
         "correction for overlapping training sets of Nadeau and Bengio [16], and an equivalence "
         "test (two one-sided tests [17]) at a margin of ±0.05 macro-F1.")

    # ── 3 ──
    h(doc, "3.  Results")
    h(doc, "3.1  The evaluation unit moves agreement more than the model does", 10.5, before=8)
    para(doc,
         f"Table 2 and Fig. 2(a) give the locked-half result. With the predictions unchanged, "
         f"QWK rose from {N['img_rng']} per image to {N['eye_rng']} per eye in all four "
         f"model–seed cells, and the per-cell gap ({N['gap_rng']}) excluded zero in every "
         f"bootstrap replicate. The direction and size of the effect had been seen first on the "
         f"development half (QWK {N['dev_img_rng']} per image, {N['dev_eye_rng']} per eye), which "
         "is why eye-level scoring was added to the protocol before the locked half was opened. "
         f"Unweighted κ shows the same pattern ({N['kimg_rng']} per image, {N['keye_rng']} per "
         "eye).")
    rows = []
    for c_ in CELLS:
        a = A[c_]; pt_, cc = a["point_5ckpt"], a["ci95_clustered"]
        rows.append([CELL_NAME[c_],
                     f"{pt_['image']['qwk']['mean']:.3f} {ci(cc['image_qwk']['ci95'])}",
                     f"{pt_['eye']['qwk']['mean']:.3f} {ci(cc['eye_qwk']['ci95'])}",
                     f"+{pt_['eye_minus_image_qwk']['mean']:.3f} {ci(cc['eye_minus_image_qwk']['ci95'])}",
                     f"{pt_['eye']['accuracy']['mean']:.3f}",
                     f"{100*pt_['eye']['stage123_eyes_called_normal']['mean']:.1f}%"])
    table(doc, "Table 2. Locked half of the held-out hospital (661 images, 230 eyes, 116 "
               "infants), scored once. Means over five checkpoints; 95% infant-clustered "
               "bootstrap intervals in brackets. The last column is the share of stage 1–3 eyes "
               "(n = 147) predicted normal at eye level.",
          ["Arm", "QWK per image", "QWK per eye", "Eye − image", "Eye accuracy",
           "Stage 1–3 eyes called normal"], rows, widths=[1.25, 1.2, 1.2, 1.2, 0.7, 0.9])
    figure(doc, "fig2_unit_effect",
           "**Fig. 2.** (a) The same predictions scored per image and per eye on the locked half, "
           "for two heads at two seeds. Bars are means over five checkpoints; error bars show the "
           "spread across checkpoints (standard deviation), not sampling uncertainty, which is "
           "given in Table 2. (b) Share of stage 1–3 images of the locked half called normal, by "
           "acquisition view (structured head, seed 42).")

    h(doc, "3.2  Where the gap comes from", 10.5, before=8)
    para(doc,
         "The misses depend on where the camera was pointed (Fig. 2(b)). Stage 1–3 images taken "
         f"at the posterior pole were called normal {N['post_rng']} of the time across the four "
         f"cells, nasal images {N['nas_rng']}, and temporal images {N['temp_rng']}. In ROP the "
         "temporal retina is the last to vascularise, and it is where the demarcation line or "
         "ridge is usually seen; a posterior or nasal photograph of such an eye can be normal as "
         "an image. The locked-half confusion matrix shows the same thing from the other side "
         "(Fig. 3): 73% of stage 1, 44% of stage 2 and 37% of stage 3 images fall in the normal "
         "column, while 99% of normal images are called normal. Errors concentrate in one column "
         "instead of spreading around the diagonal, which is the signature of images that carry "
         "a label their pixels do not show.")
    para(doc,
         "A view-based control locates the mechanism more precisely than the aggregation does. "
         "Scoring only the temporal images, one at a time and without any aggregation, gives QWK "
         "0.799 on the locked half (227 images) against 0.801 per eye; nasal-only and "
         "posterior-only scoring give 0.375 and 0.398. The development half behaves the same way "
         "(0.718 temporal-only, 0.721 per eye). Aggregation by majority vote instead of by the "
         "most severe view does not recover agreement (0.490). The eye-level gain therefore comes "
         "from selecting the image that can show the finding, not from averaging noise away.")
    figure(doc, "fig3_confusion",
           "**Fig. 3.** Locked-half confusion matrix per image (row percentages, five checkpoints "
           "pooled; row counts in parentheses; structured head, seed 42). Errors concentrate in "
           "the normal column (boxed). Stage 4–5 and AP-ROP rows are too small to read as "
           "performance and are reported as not externally validated.", width=3.6)
    para(doc,
         "A similar pattern appears inside the cross-validation pool for AP-ROP: "
         f"{N['arop_ost']} {N['arop_ost_ci']} of AP-ROP frames from Ostrava, where one code is "
         "stamped on every frame of a session, received an AP-ROP probability below 0.1, against "
         f"{N['arop_oth']} {N['arop_oth_ci']} in the sources that label single images "
         f"({N['arop_oth_n']} infants). We report this as consistent with the label-unit "
         "explanation, not as independent evidence: all 470 Ostrava AP-ROP frames come from one "
         "infant across 16 sessions, so the comparison cannot separate session-stamped labels "
         "from that infant.")

    h(doc, "3.3  The two heads are indistinguishable", 10.5, before=8)
    para(doc,
         f"Under five-fold cross-validation the structured head reached macro-F1 {N['cv_s42']} "
         f"against {N['cv_f42']} for the flat head at seed 42, and {N['cv_s1337']} against "
         f"{N['cv_f1337']} at seed 1337 — the sign of the difference reverses between seeds. "
         f"Over the ten paired fold × seed differences the mean was {sgn(N['eq_mean'], 4)} "
         f"macro-F1. Treating the ten pairs as independent gives a 95% interval of "
         f"{N['eq_naive']} and equivalence at ±0.05 (p = {N['eq_naive_p']:.3f}); with the "
         f"correction for overlapping training folds [16] the interval widens to {N['eq_corr']} "
         f"and equivalence at that margin can no longer be shown (p = {N['eq_corr_p']:.3f}). "
         "We therefore claim no difference in either direction, not proven equivalence. On the "
         "locked half the flat head was ahead per image (QWK 0.553–0.561 against 0.508–0.523), "
         "while per eye all four cells lie within 0.791–0.801. What the structured head offers "
         "is taxonomy-faithful output — an ordinal stage and a separate AP-ROP probability — at "
         "no detectable cost, not higher accuracy.")

    h(doc, "3.4  A naive site probe understates the site information that remains", 10.5, before=8)
    pr, pn, pc = N["probe"], N["p_naive"], N["p_ctrl"]
    para(doc,
         "Domain-adversarial training is usually validated with a linear probe over all classes. "
         "Because disease prevalence differs by site, that probe can score well by reading "
         "disease. We ran it alongside a disease-controlled probe restricted to normal images "
         "(fold 0, seed 42, adversary weight w ∈ {0, 0.5, 2.0}; Table 3). The controlled probe "
         "decodes site better than the naive one at every weight, so the naive probe understates "
         "the site information the features retain. Neither probe fell as the adversary "
         f"strengthened: from w = 0 to 0.5 the controlled probe changed by "
         f"{sgn(pc['w0.5_minus_w0.0']['point'])} {ci(pc['w0.5_minus_w0.0']['ci95'])}, and at "
         f"w = 2.0 it rose by {sgn(pc['w2.0_minus_w0.0']['point'])} "
         f"{ci(pc['w2.0_minus_w0.0']['ci95'])} relative to w = 0 (naive probe: "
         f"{sgn(pn['w2.0_minus_w0.0']['point'])} {ci(pn['w2.0_minus_w0.0']['ci95'])}). With one "
         "fold and one seed, we report this as a negative result for this configuration, not as "
         "a general property of domain-adversarial training.")
    table(doc, "Table 3. Site-probe accuracy against adversary weight (fold 0 validation, seed "
               "42; patient-clustered 95% intervals, 2,000 replicates).",
          ["Adversary weight", "Naive probe", "Disease-controlled probe"],
          [[f"w = {w}", f"{pr['w'+w]['naive']['point']:.3f} {ci(pr['w'+w]['naive']['ci95'])}",
            f"{pr['w'+w]['controlled']['point']:.3f} {ci(pr['w'+w]['controlled']['ci95'])}"]
           for w in ("0.0", "0.5", "2.0")], widths=[1.5, 2.2, 2.4])

    h(doc, "3.5  A screening threshold that did not transfer", 10.5, before=8)
    nw = N["new"]
    para(doc,
         "An earlier binary screening model (ResNet50, trained on Ostrava only) reached AUC "
         f"{N['old_auc']:.3f} on its internal test split and, at its operating threshold "
         f"{N['old_thr']:.3f}, sensitivity {N['old_sens']:.3f} and specificity "
         f"{N['old_spec']:.3f} (n = {N['old_n']:,}). On the development half of the held-out "
         f"hospital its AUC was {N['old_ext_auc']:.3f}, and its lowest score on any image "
         f"({N['old_min']:.3f}) exceeded its threshold, so it flagged all {N['ext_n']} images, "
         f"including all {N['ext_normal']} normal images. We replaced it with the structured "
         "model's any-ROP probability, using a rule fixed before the curves were drawn: the "
         "largest threshold that holds sensitivity ≥ 0.90 at both the training hospitals and the "
         f"held-out development half (threshold {N['new_thr']}). This gives sensitivity "
         f"{nw['internal_oof']['sensitivity']:.3f} and specificity "
         f"{nw['internal_oof']['specificity']:.3f} internally (n = {nw['internal_oof']['n']:,}) "
         f"and {nw['external_dev']['sensitivity']:.3f} / {nw['external_dev']['specificity']:.3f} "
         "externally. The external pair is in-sample for the threshold choice and therefore "
         "optimistic. Cross-camera loss of performance in ROP staging has been reported before "
         "[18]; the point here is that no internal metric anticipated it.")

    # ── 4 ──
    h(doc, "4.  Discussion")
    para(doc,
         "On a locked half of a hospital the models never saw, the unit over which the same "
         "predictions are scored changed staging agreement by roughly 0.23–0.28 QWK, more than "
         "the choice of head, the random seed, or both together. The view analysis shows why: a "
         "per-eye label copied onto photographs that point away from the temporal periphery "
         "makes many of those photographs look wrongly classified when they are not.")
    para(doc,
         "This sits alongside, rather than against, earlier work. DeepROP [4] and the i-ROP "
         "severity score [7] already predict at the eye or examination level, and Peng et al. [6] "
         "report both units. J-PROP [5] found that F1 fell from image to eye to patient level for "
         "referral-warranted ROP, the opposite direction to ours. The two findings are "
         "compatible: when a label describes the image, taking the maximum over an eye's images "
         "can only add positives and can lower specificity; when the label describes the eye "
         "and is copied onto its images, the per-image score is the one that is biased, and "
         "aggregation recovers agreement. Which direction a study sees therefore depends on the "
         "unit its labels describe — a fact that is seldom stated, and which we suggest should "
         "be.")
    para(doc,
         "The comparison with published accuracies should be read in the same light. Reported "
         "stage accuracies of 0.97–0.98 on the Ostrava corpus [8] come from a four-class task "
         "with an image-level split and a 37-image test set; our six-class task, infant-level "
         "split, external hospital and per-image QWK of about 0.52 are not the same measurement, "
         "and the difference between them is not evidence about either model. The human "
         "reference is also informative: seven experts reading the same photographs agreed on "
         "stage with Fleiss κ 0.24 [19], and stage was the commonest source of disagreement "
         "between centres (40% of comparisons) [20]. Our unweighted Cohen κ against the released "
         f"labels ({N['k_cv42']:.2f}–{N['k_cv1337']:.2f} in cross-validation; "
         f"{N['keye_rng']} per eye on the locked half) is a different quantity, measured against "
         "a single institution's coding, and should not be read as better-than-human "
         "agreement. It does indicate that the reference standard itself is uncertain, a form of "
         "the hidden-stratification problem [21].")
    para(doc,
         f"Clinically, the result is sobering. Even at eye level, {N['miss_rng']} of stage 1–3 "
         "eyes were called normal on the locked half, and stage 4–5 and AP-ROP could not be "
         "validated externally at all. The models described here are research instruments, not "
         "screening tools. The practical recommendations are modest: state the unit the labels "
         "describe; report image- and eye-level metrics side by side; record laterality and "
         "acquisition view when releasing ROP datasets; and validate a threshold externally "
         "before it is deployed. The view result also points to the next model: one that takes "
         "all views of an eye as its input, rather than scoring them one by one.")
    h(doc, "Limitations", 10.5, before=8)
    para(doc, "This study has several limitations.")
    for t_ in LIMITATIONS:
        para(doc, t_)

    # ── 5 ──
    h(doc, "5.  Conclusion")
    para(doc,
         "On an audited multi-source ICROP staging benchmark with patient-grouped folds and an "
         "external hospital scored once, the evaluation unit moved quadratic-weighted κ from "
         f"about {N['img_rng']} per image to {N['eye_rng']} per eye in four independent "
         "model–seed cells, and the errors it explains are anatomically directed rather than "
         "random. The choice of head moved nothing that survived a change of random seed. ROP "
         "staging results should state the unit their labels describe, report both image- and "
         "eye-level metrics, and be validated externally before a threshold is used.")

    # ── back matter ──
    h(doc, "Data and code availability")
    para(doc,
         "All four cohorts are publicly available from their original repositories: Ostrava, "
         "figshare collection 6626162 (CC0) [2]; Shenzhen, doi:10.6084/m9.figshare.25514449 "
         "(CC BY 4.0) [11]; ROP-VL, doi:10.6084/m9.figshare.30143461 (CC BY 4.0) [12]; Multi-View "
         "ROP, Hugging Face lijuanliao/ROP-DATASET (CC BY-NC 4.0) [3]. The split manifests, the "
         "locked-evaluation protocol, the saved predictions and the analysis code are available "
         "at [[repository URL]].")
    h(doc, "Compliance with ethical standards")
    para(doc,
         "This research study was conducted retrospectively using human subject data made "
         "available in open access by the sources listed above. Ethical approval was not "
         "required, as confirmed by the licence attached to each open-access dataset. No images "
         "were collected by the authors, and file names that encoded clinical attributes were "
         "replaced with anonymous identifiers before use.")
    h(doc, "Acknowledgments")
    para(doc,
         "We thank the teams who released the Ostrava, Shenzhen, ROP-VL and Multi-View ROP "
         "datasets. This work received no funding. The authors declare no conflicts of "
         "interest. [[Use of AI tools — edit to match what was actually done:]] An AI assistant "
         "(Claude, Anthropic) was used to help write analysis and figure-generation code and to "
         "draft and edit parts of the manuscript text. The authors reviewed and verified all "
         "content and take full responsibility for it.")

    h(doc, "References")
    for i, ref in enumerate(REFS, 1):
        para(doc, f"[{i}] {ref}", justify=False, size=9, after=3)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    doc.save(OUT)
    print(f"wrote {OUT.relative_to(ROOT)}  ({OUT.stat().st_size/1024:.0f} KB)")


if __name__ == "__main__":
    build()
