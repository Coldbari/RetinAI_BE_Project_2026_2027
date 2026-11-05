"""One-page metric sheet for the served ROP model, as a figure.

    .venv/bin/python scripts/make_metrics_table.py
    FIG_THEME=light .venv/bin/python scripts/make_metrics_table.py

Writes graphs/rop_metrics/rop_metric_sheet.png (+ light/ variant for print).

Every number here is copied from a committed artifact, and each table carries its source
path so a reader can check it without asking. Nothing is recomputed in this script — it is
a renderer, not an evaluation. If a number moves, it moves in the artifact first.

The dangerous-error rows are marked in the figure itself rather than in a caption, because
this sheet is built to be pasted into slides and captions do not survive that trip.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
from figtheme import BAD, GOOD, GRID, INK, INK2, SURF, FOOTNOTE, outdir  # noqa: E402

OUT = outdir(ROOT / "graphs" / "rop_metrics")

W, H = 17.5, 15.35
MARGIN = 0.55
GUTTER = 0.75
COLW = (W - 2 * MARGIN - GUTTER) / 2
ROW = 0.275          # inches per body row
BODY, HEAD, TITLE = 10.3, 10.3, 13.0

fig = plt.figure(figsize=(W, H), facecolor=SURF)
ax = fig.add_axes([0, 0, 1, 1]); ax.set_facecolor(SURF)
ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.axis("off")


def table(x, y, title, source, cols, rows, flag_rows=(), note=None):
    """cols = [(label, dx, align, width)] with dx measured from x. Returns the new y."""
    ax.text(x, y, title, color=INK, fontsize=TITLE, fontweight="bold", va="top")
    y += 0.30
    ax.text(x, y, source, color=FOOTNOTE, fontsize=8.4, va="top", style="italic")
    y += 0.30
    for label, dx, align, _w in cols:
        ax.text(x + dx, y, label, color=INK2, fontsize=HEAD, fontweight="bold",
                va="top", ha=align)
    y += 0.245
    ax.add_line(plt.Line2D([x, x + COLW], [y, y], color=GRID, lw=1.1))
    y += 0.075
    for i, cells in enumerate(rows):
        if i % 2 == 1:
            ax.add_patch(Rectangle((x - 0.08, y - 0.035), COLW + 0.16, ROW,
                                   facecolor=GRID, alpha=0.38, lw=0))
        danger = i in flag_rows
        if danger:
            ax.add_patch(Rectangle((x - 0.08, y - 0.035), 0.045, ROW,
                                   facecolor=BAD, lw=0))
        for cell, (_l, dx, align, _w) in zip(cells, cols):
            txt, colour, weight = cell, INK, "normal"
            if isinstance(cell, tuple):
                txt, kind = cell
                colour = {"good": GOOD, "bad": BAD, "mute": INK2}[kind]
                weight = "bold" if kind in ("good", "bad") else "normal"
            elif danger and dx == 0:
                weight = "bold"
            ax.text(x + dx, y, txt, color=colour, fontsize=BODY, va="top",
                    ha=align, fontweight=weight)
        y += ROW
    if note:
        y += 0.06
        ax.text(x, y, note, color=FOOTNOTE, fontsize=8.7, va="top", wrap=True)
        y += 0.24 * (1 + note.count("\n"))
    return y + 0.52


# ── header ───────────────────────────────────────────────────────────────────────────
ax.text(MARGIN, 0.42, "ROP staging model — every measured number", color=INK,
        fontsize=21, fontweight="bold", va="top")
ax.text(MARGIN, 0.86,
        "EfficientNetV2-S · 384 px · 6 classes (Normal / Stage 1 / 2 / 3 / 4-5 / AP-ROP) · structured "
        "ordinal head with AP-ROP kept off the stage ladder.",
        color=INK2, fontsize=10.6, va="top")
ax.text(MARGIN, 1.12,
        "Split by infant, never by photograph · trained on 3 hospitals · a 4th hospital sealed and "
        "opened exactly once, on 2026-08-18.",
        color=INK2, fontsize=10.6, va="top")
ax.add_line(plt.Line2D([MARGIN, W - MARGIN], [1.48, 1.48], color=GRID, lw=1.4))

LX, RX = MARGIN, MARGIN + COLW + GUTTER
ly = ry = 1.78

# ── left column ──────────────────────────────────────────────────────────────────────
c3 = [("Metric", 0, "left", 0), ("3 training hospitals\n(6,700 imgs, out-of-fold)", COLW * 0.62, "right", 0),
      ("New hospital, locked\n(661 imgs / 116 infants)", COLW, "right", 0)]
ly = table(LX, ly, "1 · Headline, per photograph",
           "RETINAI_FINAL_SUMMARY/05_RESULTS_AND_ACCURACY.md · results/rop/locked_eval/locked_table.json",
           c3,
           [["Accuracy", "0.786", "0.520 ± 0.015"],
            ["Balanced accuracy", "0.652", "0.520"],
            ["QWK — quadratic weighted κ", "0.897  [0.869–0.923]", "0.523  [0.433–0.594]"],
            ["Macro-F1 (6 classes)", "0.652  [0.572–0.760]", "0.412  [0.343–0.479]"],
            ["AUC — any ROP vs none", "0.970", "0.835"],
            ["ECE — calibration error", ("0.109", "bad"), ("0.334", "bad")],
            ["Mean stated certainty", "0.895", "0.880"],
            ["Over-confidence (certainty − accuracy)", ("+0.109", "bad"), ("+0.334", "bad")]],
           note="95% CIs resample INFANTS, not photographs — one infant contributes up to 470 images.\n"
                "QWK penalises by the SQUARE of the stage distance: Normal↔Stage 4 costs 16× Stage 1↔Stage 2.")

c6 = [("Class", 0, "left", 0), ("Precision", COLW * 0.40, "right", 0),
      ("Recall", COLW * 0.55, "right", 0), ("n", COLW * 0.68, "right", 0),
      ("Recall (new)", COLW * 0.89, "right", 0), ("n (new)", COLW, "right", 0)]
ly = table(LX, ly, "2 · Per class — 3 training hospitals, then the new one",
           "locked_table.json → secondary_ensemble.per_class_recall",
           c6,
           [["Normal", "0.929", "0.951", "3,837", "0.991", "232"],
            ["Stage 1", ("0.376", "bad"), "0.499", "577", ("0.257", "bad"), "105"],
            ["Stage 2", "0.559", "0.631", "959", ("0.268", "bad"), "164"],
            ["Stage 3", "0.758", "0.713", "620", ("0.278", "bad"), "144"],
            ["Stage 4/5", "0.663", "0.775", "89", "0.250", "4"],
            ["AP-ROP", "0.899", ("0.345", "bad"), "618", "0.500", "12"]],
           note="Precision holds at the new site; RECALL collapses — the model defaults to \"Normal\".\n"
                "Stage 4/5 and AP-ROP counts are too small there to support any claim either way.")

c5 = [("Scored as", 0, "left", 0), ("QWK", COLW * 0.46, "right", 0),
      ("Accuracy", COLW * 0.62, "right", 0), ("Macro-F1", COLW * 0.79, "right", 0),
      ("diseased→Normal", COLW, "right", 0)]
ly = table(LX, ly, "3 · The unit of analysis — same predictions, locked set",
           "locked_table.json → coprimary_eye_level, view_miss_table",
           c5,
           [["Per photograph, pooled  (661)", "0.523", "0.520", "0.412", "—"],
            ["   temporal photos only  (227)", ("0.841", "good"), "—", "—", ("12.7%", "good")],
            ["   nasal only  (221)", "0.352", "—", "—", "63.4%"],
            ["   posterior pole only  (213)", ("0.229", "bad"), "—", "—", ("71.7%", "bad")],
            ["Per eye, votes combined  (230)", ("0.801", "good"), "0.646", "0.562", "17.8%"]],
           flag_rows=(),
           note="ROP grows at the temporal edge; the eye's single label is written onto photographs that\n"
                "physically cannot show the ridge. The app scores ONE photograph at a time — the worst unit.")

# ── right column ─────────────────────────────────────────────────────────────────────
c6b = [("Split", 0, "left", 0), ("Sens", COLW * 0.44, "right", 0),
       ("Spec", COLW * 0.58, "right", 0), ("PPV", COLW * 0.72, "right", 0),
       ("AUC", COLW * 0.86, "right", 0), ("n", COLW, "right", 0)]
ry = table(RX, ry, "4 · The binary verdict the app serves — threshold 0.0155 on P(any ROP)",
           "results/rop/screening_rebase.json",
           c6b,
           [["Internal, out-of-fold", "0.958", "0.781", "0.766", "0.970", "6,700"],
            ["External dev half", "0.905", ("0.387", "bad"), "0.835", "0.821", "663"],
            ["retired ResNet50 @0.1933 · internal test", "0.990", "0.398", "—", "—", "1,502"],
            ["retired ResNet50 @0.1933 · external", "1.000", ("0.000", "bad"), "0.774", "0.691", "663"]],
           flag_rows=(3,),
           note="Threshold rule: the largest threshold holding sensitivity ≥ 0.90 on BOTH internal OOF and the\n"
                "external dev half — so 0.905 / 0.387 is in-sample for that choice and optimistic. The locked half\n"
                "stays closed. The retired head flagged every image at the new site; that is what forced the re-base.")

c4 = [("Error", 0, "left", 0), ("Count", COLW * 0.62, "right", 0),
      ("Share", COLW * 0.79, "right", 0), ("Mean certainty", COLW, "right", 0)]
ry = table(RX, ry, "5 · What the mistakes are — 1,433 of 6,700 (21.4%)",
           "RETINAI_FINAL_SUMMARY/05_RESULTS_AND_ACCURACY.md §3",
           c4,
           [["One stage off (adjacent)", "603", "42%", "0.73"],
            ["AP-ROP ↔ a stage", "338", "24%", "0.66"],
            ["Healthy eye called diseased", "187", "13%", "0.73"],
            ["Diseased eye called NORMAL", "186", "13%", ("0.93", "bad")],
            ["AP-ROP called NORMAL", "89", "6%", ("0.82", "bad")],
            ["Several stages off", "27", "2%", "0.66"],
            ["Stage 4/5 called NORMAL", "3", "0.2%", ("0.91", "bad")]],
           flag_rows=(3, 4, 6),
           note="Certainty is HIGHEST exactly where the model is most dangerous: 141 of the 186 missed-disease\n"
                "cases were made above 90% stated certainty. \"Act only on confident predictions\" keeps almost\n"
                "all of them — the failure mode cannot be thresholded away.")

c4b = [("Measure", 0, "left", 0), ("Ordinal (served)", COLW * 0.66, "right", 0),
       ("Plain softmax", COLW * 0.86, "right", 0), ("Verdict", COLW, "right", 0)]
ry = table(RX, ry, "6 · Does the complicated head earn itself? No.",
           "results/rop/matched_equivalence.json · identical backbone, folds, preprocessing",
           c4b,
           [["Macro-F1, seed 42", "0.6923", "0.6977", ("softmax", "mute")],
            ["Macro-F1, seed 1337", "0.7079", "0.6855", ("ordinal", "mute")],
            ["QWK", "0.8964", "0.8962", ("tie, p = 0.977", "mute")],
            ["Calibration error (ECE)", ("0.109", "bad"), ("0.035", "good"), ("softmax", "mute")],
            ["New hospital, per photo", "0.473", "0.492", ("softmax", "mute")],
            ["AP-ROP kept off the stage ladder", ("yes", "good"), ("no", "bad"), ("ordinal only", "mute")]],
           note="The sign of the F1 gap flips with the seed. Three noise sources each EXCEED the difference being\n"
                "measured: seed 0.018–0.043, last-five-epoch wobble 0.007–0.056, fold spread 0.086–0.099.\n"
                "Equivalent in accuracy; the simple head is better calibrated. Reported as a finding, not buried.")

# ── footer ───────────────────────────────────────────────────────────────────────────
fy = max(ly, ry) + 0.05
ax.add_line(plt.Line2D([MARGIN, W - MARGIN], [fy, fy], color=GRID, lw=1.4))
fy += 0.22
ax.text(MARGIN, fy, "What limits every number above", color=INK, fontsize=12.4,
        fontweight="bold", va="top")
fy += 0.34
limits = [
    "Labels — 43% of the original binary \"ROP-positive\" class is not ROP; image dimensions ALONE score AUC 0.911 on the old test split (device confound).",
    "Scale — 188 infants, and all 89 Stage 4/5 images come from a single dataset. Stage 4/5 and AP-ROP at the new site are 4 and 12 images.",
    "Grad-CAM is not evidence — attention-on-retina 56.0% against a retina area of 53.8%, a lift of 1.04 (chance); 30.7% of peaks fall off the retina, 19.3% on padding.",
    "The score is a ranking, not a probability — uncalibrated, and at the new hospital certainty is 0.898 when right against 0.858 when wrong.",
]
for t in limits:
    ax.text(MARGIN + 0.02, fy, "•", color=BAD, fontsize=11, va="top")
    ax.text(MARGIN + 0.22, fy, t, color=INK2, fontsize=9.9, va="top")
    fy += 0.265
fy += 0.20
ax.text(MARGIN, fy,
        "One line: \"is anything wrong with this eye\" transfers (AUC 0.970 → 0.835). "
        "\"Which stage is it\" does not (QWK 0.897 → 0.523 per photograph, 0.801 per eye).",
        color=INK, fontsize=10.6, va="top", fontweight="bold")
ax.text(W - MARGIN, fy + 0.03, "scripts/make_metrics_table.py · 2026-09-23",
        color=FOOTNOTE, fontsize=8.4, va="top", ha="right")

p = OUT / "rop_metric_sheet.png"
fig.savefig(p, dpi=200, facecolor=SURF)
print("wrote", p)
