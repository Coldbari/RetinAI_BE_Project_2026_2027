"""Build ROP_Staging_Deck.pptx — 8 slides, ROP staging work only.

    .venv/bin/python scripts/make_rop_deck.py

Dark theme matching the existing RetinalAI deck (#060C14) and the graphs/rop_staging
figures it embeds. Every number on these slides is committed in the repo with its
evaluation regime; the deck states caveats on the same slide as the number they qualify —
a slide gets photographed and quoted without its speaker notes.
"""
from __future__ import annotations

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.util import Emu, Inches, Pt

BG = RGBColor(0x06, 0x0C, 0x14)
INK = RGBColor(0xFF, 0xFF, 0xFF)
INK2 = RGBColor(0xC3, 0xC2, 0xB7)
MUT = RGBColor(0x8A, 0x8A, 0x80)
ORANGE = RGBColor(0xD9, 0x59, 0x26)
BLUE = RGBColor(0x39, 0x87, 0xE5)
GREEN = RGBColor(0x19, 0x9E, 0x70)
GOLD = RGBColor(0xC9, 0x85, 0x00)
RED = RGBColor(0xD9, 0x4F, 0x4F)

W, H = Inches(13.333), Inches(7.5)


def slide(prs):
    s = prs.slides.add_slide(prs.slide_layouts[6])   # blank
    r = s.shapes.add_shape(1, 0, 0, W, H)            # full-bleed background
    r.fill.solid()
    r.fill.fore_color.rgb = BG
    r.line.fill.background()
    r.shadow.inherit = False
    return s


def text(s, x, y, w, h, runs, size=18, color=INK, bold=False, align=PP_ALIGN.LEFT,
         leading=1.15):
    """runs: str, or list of (text, dict) paragraphs; dict may set size/color/bold/bullet."""
    tb = s.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    if isinstance(runs, str):
        runs = [(runs, {})]
    for i, (t, kw) in enumerate(runs):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = ("•  " + t) if kw.get("bullet") else t
        p.alignment = kw.get("align", align)
        p.line_spacing = leading
        p.space_after = Pt(kw.get("space", 6))
        f = p.runs[0].font
        f.size = Pt(kw.get("size", size))
        f.color.rgb = kw.get("color", color)
        f.bold = kw.get("bold", bold)
    return tb


def pic(s, path, x, y, w=None, h=None):
    return s.shapes.add_picture(path, Inches(x), Inches(y),
                                Inches(w) if w else None, Inches(h) if h else None)


def kicker(s, t, color=ORANGE):
    text(s, 0.55, 0.32, 12, 0.4, t.upper(), size=13, color=color, bold=True)


def title(s, t, y=0.72, size=30):
    text(s, 0.55, y, 12.2, 0.9, t, size=size, bold=True)


def footer(s, t):
    text(s, 0.55, 7.05, 12.2, 0.35, t, size=10.5, color=MUT)


def main():
    prs = Presentation()
    prs.slide_width, prs.slide_height = W, H

    # ── 1 · title ────────────────────────────────────────────────────────────
    s = slide(prs)
    text(s, 0.9, 2.3, 11.5, 1.6, "Clinically-Structured ROP Staging\nAcross Four Cohorts, Three Hospitals",
         size=40, bold=True)
    text(s, 0.9, 4.15, 11.5, 0.5,
         "one shared backbone, three clinical questions — each branch built against a measured failure",
         size=17, color=INK2)
    text(s, 0.9, 5.6, 11.5, 1.2, [
        ("8,024 eligible fundus images · 1,760 infants · 4 cohorts, 3 hospitals · 6-class ICROP incl. AP-ROP", {"size": 14, "color": INK2}),
        ("RetinAI — B.E. final-year project → paper in preparation (ISBI 2027)", {"size": 13, "color": MUT}),
    ])

    # ── 2 · the field has an evaluation problem ─────────────────────────────
    s = slide(prs)
    kicker(s, "Why this work exists", RED)
    title(s, "Published ROP numbers don't survive scrutiny")
    text(s, 0.55, 1.75, 12.2, 4.4, [
        ("99.7% accuracy (Discover AI 2026) — split at IMAGE level on a dataset with ~32 images per infant: "
         "the same babies sit in train and test. Their own text applies the patient-safe split to one dataset and skips it here.",
         {"bullet": True, "size": 16}),
        ("kappa 0.979 (IEEE TMI 2021) — per-image, single hospital, no patient grouping stated.",
         {"bullet": True, "size": 16}),
        ("AP-ROP — the most urgent form of ROP — is folded into Stage 4 by the strongest recent multi-center model "
         "(VEA-Net 2026), even though the ICROP-3 standard says it is not a stage at all.",
         {"bullet": True, "size": 16}),
        ("Accuracy-only reporting hides it: VEA-Net's 90.4% on Ostrava sits next to an 89.5% majority-class prior.",
         {"bullet": True, "size": 16}),
        ("Our thesis: build the evaluation the field skips, then let the architecture follow from what it measures.",
         {"bullet": True, "size": 17, "color": ORANGE, "bold": True}),
    ])
    footer(s, "sources: ROPDeepX (Discover AI 2026) · Peng et al. (IEEE TMI 2021) · VEA-Net (Frontiers 2026) — all read in full, quotes in repo")

    # ── 3 · corpus + audit ───────────────────────────────────────────────────
    s = slide(prs)
    kicker(s, "Data + trust", BLUE)
    title(s, "Four cohorts, five gates before any number is believed")
    text(s, 0.55, 1.75, 6.1, 4.9, [
        ("CORPUS", {"size": 13, "color": MUT, "bold": True}),
        ("Ostrava 6,004 · Shenzhen 1,099 · ROP-VL 2,020 · Multi-View 1,324 (held-out hospital — never trained on)",
         {"size": 15}),
        ("Stage labels recovered from per-session diagnosis codes — 6,004/6,004 joined; "
         "43% of the old binary 'ROP' label wasn't ROP", {"size": 15, "space": 10}),
        ("De-identified (filenames carried sex, GA, birth-weight) · near-duplicates removed "
         "(calibrated 256-bit dHash) · one double-registered infant merged", {"size": 15}),
        ("Patient-grouped folds; normal capped 2/infant (train only); every path audited", {"size": 15}),
    ])
    text(s, 7.0, 1.75, 5.8, 4.9, [
        ("THE FIVE GATES", {"size": 13, "color": MUT, "bold": True}),
        ("G1  patient leakage — PASS (zero overlap, all splits)", {"size": 15}),
        ("G2  duplicate leakage — PASS (closest cross-hospital pair d=15 vs threshold 4)", {"size": 15}),
        ("G3  site in metadata — 0.997 balanced acc: the shortcut exists", {"size": 15}),
        ("G4  metadata-only model — macro-F1 0.187: the bar any image model must beat", {"size": 15}),
        ("G5  site probe on embeddings — with disease-held-constant + untrained controls (slide 7)", {"size": 15}),
    ])
    footer(s, "scripts/rop_shortcut_audit.py — run before trusting any number; test_locked opens exactly once, at the very end")

    # ── 4 · benchmark ────────────────────────────────────────────────────────
    s = slide(prs)
    kicker(s, "The measured failure", GREEN)
    title(s, "Five backbones: architecture is not the constraint", size=28)
    pic(s, "graphs/rop_staging/00_comparison_per_class_recall.png", 0.45, 1.7, w=8.1)
    text(s, 8.85, 1.9, 4.0, 4.6, [
        ("Spread across 4M–28M params, CNN/ViT/hybrid: 0.028 macro-F1 — indistinguishable.",
         {"size": 15}),
        ("But every backbone fails identically on AP-ROP: recall 0.16–0.22 at precision 0.89–0.97 — "
         "it speaks only when certain.", {"size": 15, "space": 10}),
        ("Identical failure across five architectures on this fold — later traced to the labels' "
         "evaluation unit, not the softmax (slide 6). One fold misleads; that lesson bit our own claim.",
         {"size": 15, "color": ORANGE, "bold": True}),
    ])
    footer(s, "fold 0, seed 42, patient-grouped; metadata bar 0.187 cleared ~2.8× by all five — the models read pixels, not filenames")

    # ── 5 · architecture ─────────────────────────────────────────────────────
    s = slide(prs)
    kicker(s, "The model")
    title(s, "One backbone, three clinical questions", size=28)
    pic(s, "graphs/rop_staging/architecture.png", 1.15, 1.55, w=11.0)

    # ── 6 · results ──────────────────────────────────────────────────────────
    s = slide(prs)
    kicker(s, "Results — the honest version", GREEN)
    title(s, "Equivalent headline metrics, faithful taxonomy,\nand one real discovery", size=26)
    text(s, 0.55, 2.05, 6.2, 4.6, [
        ("5-FOLD CV, MATCHED AT NATURAL SAMPLING (flat vs structured)", {"size": 13, "color": MUT, "bold": True}),
        ("macro-F1  s42: 0.698 ± 0.099 vs 0.692 ± 0.086 (flat ahead)", {"size": 16}),
        ("macro-F1  s1337: 0.686 ± 0.109 vs 0.708 ± 0.097 (structured ahead)", {"size": 16}),
        ("THE SIGN FLIPS BETWEEN SEEDS — either seed alone would have 'decided' "
         "the architecture question, in opposite directions", {"size": 14, "color": ORANGE, "bold": True}),
        ("LOCKED half of the held-out hospital, opened ONCE (18 Aug): eye-level QWK "
         "0.79–0.80 for ALL FOUR arm-seed cells — equivalence replicates externally", {"size": 14.5,
         "space": 8}),
        ("the structured head buys clinically-faithful outputs and errors that land NEXT DOOR "
         "on the severity ladder (284/506 severe misses → stage 2, only 106 → normal) — "
         "not accuracy gains", {"size": 14, "color": ORANGE, "bold": True, "space": 10}),
    ])
    text(s, 7.1, 2.05, 5.7, 4.6, [
        ("THE DISCOVERY — AP-ROP AT THE CLINICAL UNIT", {"size": 13, "color": MUT, "bold": True}),
        ("Fold 0's 'softmax failure' (recall 0.16–0.22, all five backbones) was the LABELS: "
         "per-session codes on frames that don't show the disease. Both models plateau there; "
         "flat reaches 0.83–0.97 on other folds.", {"size": 14.5}),
        ("session-level recall  0.840", {"size": 26, "color": GREEN, "bold": True}),
        ("100 of 119 sessions, 95% CI 0.76–0.90\nfalse alarms on 789 normal sessions: 0.25%", {"size": 15}),
        ("evaluate at the unit clinicians use, and the ceiling disappears", {"size": 14, "color": GREEN,
         "bold": True}),
    ])
    footer(s, "two seeds (42, 1337); a third was dropped — no finding asks for it · matched_equivalence.json · "
              "locked_eval/locked_table.json (pre-registered protocol, script refuses a second run)")

    # ── 7 · the probe finding ────────────────────────────────────────────────
    s = slide(prs)
    kicker(s, "A finding the field is missing", GOLD)
    title(s, "The standard probe overstates adversarial debiasing", size=28)
    text(s, 0.55, 1.75, 12.2, 1.2,
         "Can a linear classifier read the HOSPITAL off the model's embeddings?  (chance = 0.33; "
         "'controlled' = normal images only, so disease prevalence can't answer for the site)",
         size=15, color=INK2)
    rows = [("", "naive probe", "disease-controlled", "AP-ROP recall", "QWK"),
            ("untrained backbone", "0.890", "0.894", "—", "—"),
            ("flat, no adversary", "0.859", "0.882", "0.183", "0.594"),
            ("adversary w=0 / w=0.5", "0.790 / 0.809  ✓ looks better", "0.866 / 0.852  unchanged (ns)", "0.311", "0.632"),
            ("adversary w=2.0", "0.867", "0.917  WORSE than untrained", "0.168", "0.525")]
    tb = s.shapes.add_table(5, 5, Inches(0.55), Inches(3.0), Inches(12.2), Inches(2.6)).table
    for r, row in enumerate(rows):
        for c, val in enumerate(row):
            cell = tb.cell(r, c)
            cell.text = val
            cell.fill.solid()
            cell.fill.fore_color.rgb = RGBColor(0x10, 0x18, 0x24) if r else RGBColor(0x1B, 0x26, 0x36)
            f = cell.text_frame.paragraphs[0].runs[0].font if cell.text_frame.paragraphs[0].runs else None
            if f:
                f.size = Pt(13)
                f.color.rgb = INK if r else INK2
                f.bold = (r == 0) or (c == 0)
                if "WORSE" in val or val.startswith("0.168") or val.startswith("0.525"):
                    f.color.rgb = RED
                if "looks better" in val:
                    f.color.rgb = GOLD
    text(s, 0.55, 5.85, 12.2, 1.0,
         "The adversary never reduces controlled site decodability — and at w=2 it significantly INCREASES it "
         "(+0.066, clustered CI [+0.02, +0.12]) while destroying AP-ROP recall. Invisible without the "
         "controlled probe, which no prior ROP work runs. A negative result, instrumented. "
         "(w=0 ties w=0.5 on cross-site QWK: 0.942 vs 0.946.)",
         size=15.5, color=GOLD, bold=True)
    footer(s, "same probe, same fold, same backbone, patient-clustered bootstrap CIs · results/rop/dann_probe_ci.json")

    # ── 8 · road to the paper ────────────────────────────────────────────────
    s = slide(prs)
    kicker(s, "Road to the paper", BLUE)
    title(s, "ISBI 2027 — deadline Oct 26, 2026", size=28)
    text(s, 0.55, 1.75, 6.4, 4.8, [
        ("DONE", {"size": 13, "color": GREEN, "bold": True}),
        ("corpus · audit · 5-backbone benchmark · structured head · 5-fold CV flat AND structured · "
         "seven-arm ablation · session-level analysis of every arm · probe dose-response · "
         "patient-bootstrap CIs + calibration · seed 1337 replication · all raw predictions committed",
         {"size": 14.5}),
        ("REMAINING", {"size": 13, "color": GOLD, "bold": True}),
        ("flat seed 1337 + both arms at seed 2024 (GPU quota) · test_locked opened once · "
         "paper prose + CLAIM checklist", {"size": 15}),
        ("CLAIM, WORDED TO SURVIVE REVIEW", {"size": 13, "color": MUT, "bold": True}),
        ("“first joint ordinal + AP-ROP-branch model with probe-quantified site debiasing and "
         "held-out-hospital evaluation”", {"size": 15, "color": INK2}),
    ])
    text(s, 7.3, 1.75, 5.5, 4.8, [
        ("THE ONE BLOCKING HUMAN ACTION", {"size": 13, "color": RED, "bold": True}),
        ("A pediatric ophthalmologist co-author to adjudicate ~300 labels (reference-standard requirement, "
         "CLAIM 2024). Without it: workshop venues. With it: clinical journals open.", {"size": 15}),
        ("Lead already identified: KIDROP — Narayana Nethralaya, Bangalore. India's flagship ROP "
         "tele-screening program.", {"size": 15, "color": ORANGE, "bold": True}),
        ("Fallback venues: MIDL (Dec) · MELBA ($10, scope-perfect) · TVST", {"size": 14, "color": MUT}),
    ])
    footer(s, "full state: docs/ROP_PROJECT_STATE.md — every number reproducible from a committed script")

    out = "ROP_Staging_Deck.pptx"
    prs.save(out)
    print(f"-> {out}  ({len(prs.slides.__iter__.__self__._sldIdLst)} slides)")


if __name__ == "__main__":
    main()
