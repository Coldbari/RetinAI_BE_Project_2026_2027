"""Architecture diagram for the clinically-structured ROP staging model.

    .venv/bin/python scripts/make_architecture_diagram.py

Writes graphs/rop_staging/architecture.png. One figure, three visual bands: the
preprocessing chain, the shared backbone, and the three heads — each head annotated with
the MEASURED result that justifies it, because in this paper the architecture is the
consequence of the measurements, and the diagram should say so.

Same dark surface and palette as every other figure in graphs/rop_staging/ (validated:
all six dataviz checks pass on this 5-slot palette against #1a1a19).
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
# Palette from the shared theme module; FIG_THEME=light renders the print variant into
# graphs/rop_staging/light/ (see scripts/figtheme.py for why the light roles are not a
# straight recolour of the dark ones).
from figtheme import (SURF, INK, INK2, GRID, ORANGE, BLUE, GREEN, GOLD, RED,  # noqa: E402
                      PANEL, PANEL_STAGE, PANEL_AROP, PANEL_SITE, PANEL_GRL, PANEL_BB,
                      DIAGRAM_TEXT, FOOTNOTE, outdir)

FIG_W, FIG_H = 15.5, 10.4


def box(ax, x, y, w, h, fc, label, sub=None, lw=1.6, ec=None, fs=11, sub_fs=8.6,
        text_color=INK):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012",
                                facecolor=fc, edgecolor=ec or fc, linewidth=lw))
    cy = y + h * (0.84 if sub else 0.5)
    ax.text(x + w / 2, cy, label, ha="center", va="center", color=text_color,
            fontsize=fs, fontweight="bold")
    if sub:
        ax.text(x + w / 2, y + h * 0.40, sub, ha="center", va="center",
                color=text_color, fontsize=sub_fs, alpha=0.92)


def arrow(ax, x0, y0, x1, y1, color=INK2, lw=2.2, style="-", mut=14):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>",
                                 mutation_scale=mut, color=color, lw=lw,
                                 linestyle=style, shrinkA=2, shrinkB=2))


def main():
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    fig.patch.set_facecolor(SURF)
    ax.set_facecolor(SURF)
    ax.set_xlim(0, 100)
    ax.set_ylim(-18.5, 100)
    ax.axis("off")

    ax.text(2, 96.5, "Clinically-structured ROP network — the served model",
            color=INK, fontsize=17, fontweight="bold")
    ax.text(2, 92.8, "one shared backbone, three clinical questions — each branch built "
            "against a measured failure. Since 19 Aug 2026 this one network produces the "
            "screening verdict, the stage AND the heatmap.", color=INK2, fontsize=10.5)

    # ── band 1: preprocessing ────────────────────────────────────────────────
    ax.text(2, 86.5, "PREPROCESSING  (identical for this model and every baseline)",
            color=INK2, fontsize=9, fontweight="bold")
    steps = [("fundus image", "480×640 … 1200×1600\nvaries by camera"),
             ("circle crop", "to the retinal\nfield of view"),
             ("CLAHE", "local contrast,\nL channel (LAB)"),
             ("letterbox", "pad square — aspect\nkept: anti-shortcut"),
             ("resize 384²", "+ train-time site\nrandomisation")]
    x = 2.0
    for i, (t, s) in enumerate(steps):
        box(ax, x, 74.5, 15.5, 9.6, PANEL, t, s, ec=GRID, fs=10, sub_fs=7.8)
        if i < len(steps) - 1:
            arrow(ax, x + 15.5, 79.3, x + 18.0, 79.3, lw=1.8)
        x += 18.0

    arrow(ax, 47, 74.0, 47, 68.6)

    # ── band 2: backbone ─────────────────────────────────────────────────────
    box(ax, 14, 58.5, 66, 9.6, PANEL_BB, "EfficientNetV2-S backbone  (ImageNet-pretrained, 20.2M)",
        "stem → Fused-MBConv ×10 → MBConv+SE ×30 → 1×1 conv → global average pool",
        ec=ORANGE, fs=11.5, sub_fs=8.6)
    arrow(ax, 47, 58.2, 47, 52.8)
    box(ax, 34, 46.0, 26, 6.6, PANEL, "embedding  z ∈ ℝ¹²⁸⁰", "Dropout 0.3",
        ec=INK2, fs=11, sub_fs=8)

    # ── band 3: three heads ──────────────────────────────────────────────────
    heads_y, head_h = 25.5, 15.5
    # stage head
    arrow(ax, 40, 45.7, 17.5, heads_y + head_h + 0.4, color=BLUE)
    box(ax, 3.5, heads_y, 28, head_h, PANEL_STAGE, "STAGE head — ordinal (CORN)",
        "Linear 1280 → 4 conditional logits\nNormal < S1 < S2 < S3 < S4/5\nAP-ROP images MASKED (ICROP-3:\nthe aggressive form has no stage)",
        ec=BLUE, fs=10.5, sub_fs=8.0)
    # arop head
    arrow(ax, 47, 45.7, 47, heads_y + head_h + 0.4, color=GREEN)
    box(ax, 36, heads_y, 22, head_h, PANEL_AROP, "AP-ROP head — binary",
        "Linear 1280 → 1, sigmoid\n“is the aggressive form present”\nthreshold 0.5, fixed a priori",
        ec=GREEN, fs=10.5, sub_fs=8.0)
    # site head with GRL above it
    arrow(ax, 54, 45.7, 76, heads_y + head_h + 0.4, color=GOLD)
    box(ax, 66.5, heads_y + 11.2, 19, 4.3, PANEL_GRL, "gradient reversal  −λ·∇",
        None, ec=RED, fs=9.5)
    arrow(ax, 76, heads_y + 11.0, 76, heads_y + 10.2, color=GOLD)
    box(ax, 64, heads_y, 24, 10.0, PANEL_SITE, "SITE head — adversary",
        "1280 → 256 → ReLU → 3 hospitals\ntraining only; λ ramped 0 → w_site",
        ec=GOLD, fs=10.5, sub_fs=8.0)

    # measured justifications, clear BELOW each head box
    cap_y = heads_y - 3.4
    ax.text(17.5, cap_y, "Stage 2 recall 0.45 → 0.61\nQWK 0.594 → 0.632 (fold 0)",
            ha="center", va="center", color=BLUE, fontsize=8.4)
    ax.text(47, cap_y, "AP-ROP recall 0.18 → 0.35 image-level\n0.84 session-level "
            "(CI 0.76–0.90), FP 0.25%", ha="center", va="center", color=GREEN, fontsize=8.4)
    ax.text(76, cap_y, "probe-quantified: naive probe improves,\ndisease-controlled "
            "probe does not — a finding", ha="center", va="center", color=GOLD, fontsize=8.4)

    # ── band 4: decode ───────────────────────────────────────────────────────
    arrow(ax, 17.5, cap_y - 2.6, 38, 11.6, color=BLUE)
    arrow(ax, 47, cap_y - 2.6, 47, 11.6, color=GREEN)
    box(ax, 25, 4.0, 40, 7.4, PANEL,
        "decode:  AP-ROP if pₐ > 0.5, else CORN rank",
        "p(6-class) = [ (1−pₐ)·P(rank=k) …, pₐ ]  — sums to 1; scored by the\nsame metrics "
        "code as every baseline", ec=INK2, fs=10, sub_fs=7.8)

    # loss, to the right of the decode box
    ax.text(97.5, 7.0, "L  =  L_CORN(stages, masked)  +  1.0·L_AP-ROP  +  λ(t)·L_site",
            ha="right", color=INK2, fontsize=9.5, style="italic")

    # ── band 5: SERVING — what the app actually shows, from this same forward pass ──
    ax.text(2, -0.9, "SERVING", color=INK2, fontsize=9, fontweight="bold")
    arrow(ax, 38, 3.8, 17.5, -2.9, color=GREEN)
    arrow(ax, 45, 3.8, 45, -2.9, color=BLUE)
    arrow(ax, 52, 3.8, 76, -2.9, color=ORANGE)

    box(ax, 3.5, -15.2, 28, 12.0, PANEL_AROP,
        "SCREENING VERDICT  (product)",
        "P(any ROP) = Σ p(Stage 1 … AP-ROP)\nflagged if > 0.0155 — the largest\nthreshold "
        "holding sens ≥ 0.90 at BOTH\nthe training and held-out hospitals",
        ec=GREEN, fs=10.2, sub_fs=7.5)
    box(ax, 34, -15.2, 24, 12.0, PANEL_STAGE,
        "ICROP STAGE  (research preview)",
        "argmax of the same 6-class vector.\nNot a clinical grade: locked-test\nQWK 0.52 "
        "per image vs 0.80 per eye",
        ec=BLUE, fs=10.2, sub_fs=7.5)
    box(ax, 60.5, -15.2, 28, 12.0, PANEL_BB,
        "GRAD-CAM  (explanation)",
        "∂ log P(any ROP) / ∂ activations —\ntargeted at the quantity the verdict\nturns on, "
        "masked to the retina, with\noff-retina attention reported as a number",
        ec=ORANGE, fs=10.2, sub_fs=7.5)
    ax.text(2, -17.6, "one forward pass — the verdict, the stage and the heatmap cannot "
            "disagree about which model produced them", color=FOOTNOTE, fontsize=8.2,
            style="italic")

    ax.text(90.5, -9.2, "the ResNet50 screening\nhead this replaced is\nretired: it flagged\n"
            "663 of 663 images at\nthe held-out hospital",
            ha="left", va="center", color=RED, fontsize=7.8, style="italic", linespacing=1.6)

    fig.text(0.995, 0.008, "fold-0 justifications shown; 5-fold CV: macro-F1 0.692 ± 0.086, "
             "QWK 0.826 ± 0.145 (seed 42)", ha="right", color=FOOTNOTE,
             fontsize=7.6, style="italic")

    out = str(outdir("graphs/rop_staging") / "architecture.png")
    fig.savefig(out, dpi=150, facecolor=SURF, bbox_inches="tight")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
