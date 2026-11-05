"""Publication-ready figures for the IEEE ISBI 2027 submission.

    .venv/bin/python scripts/make_isbi_figures.py

Different job from make_paper_figures.py. That script explains the project to a person on a
screen; this one produces figures that drop straight into a two-column IEEE manuscript:

  * IEEE column geometry — 3.5 in single column, 7.16 in double column, no guessing
  * Times New Roman throughout, at the sizes IEEE figures actually use (7-8 pt)
  * white ground, no theme switching
  * vector PDF for LaTeX (\\includegraphics) AND 600-dpi PNG for Word
  * legible in grayscale: every paired series differs in fill AND hatch, not colour alone,
    because print reviewers still print in black and white

Three figures, chosen against the 4-page limit (see docs/paper/isbi2027_paper.md, note A):
  fig1  cohort cascade            single column
  fig2  the unit effect + its anatomical cause    double column, two panels
  fig3  confusion matrix, locked set              single column

Writes graphs/isbi/*.pdf and *.png. Every number is read from a committed artifact.
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, Rectangle

ROOT = Path(__file__).resolve().parent.parent
R = ROOT / "results" / "rop"
OUT = ROOT / "graphs" / "isbi"
OUT.mkdir(parents=True, exist_ok=True)

COL1, COL2 = 3.5, 7.16                      # IEEE single / double column width, inches

# print palette: dark fill vs light fill, so the pair survives a grayscale printer
DARK, LIGHT = "#1F3B57", "#BFD3E0"
ACCENT, WARN = "#8C2F1E", "#C8721F"
INK, MUTED, RULE = "#111111", "#555555", "#BBBBBB"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "STIX Two Text", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "pdf.fonttype": 42,                     # embed real fonts, not Type-3 outlines
    "ps.fonttype": 42,
})


def finish(fig, stem: str) -> None:
    """Write both formats a manuscript might need, tightly cropped."""
    for ext, dpi in (("pdf", None), ("png", 600)):
        fig.savefig(OUT / f"{stem}.{ext}", dpi=dpi, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"  graphs/isbi/{stem}.pdf + .png")


def tidy(ax, grid: str | None = "y") -> None:
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(RULE)
    ax.tick_params(colors=MUTED, length=2.5)
    if grid:
        ax.grid(axis=grid, color=RULE, lw=0.4, alpha=0.55)
        ax.set_axisbelow(True)


# ── Fig. 1 · cohort cascade ──────────────────────────────────────────────────
def fig1_cohort() -> None:
    d = json.loads((R / "inclusion_flow.json").read_text())
    acquired = d["layer_0_acquired"]["images"]
    labelled = d["layer_1_stage_labelled_sources"]["kept"]["images"]
    infants_l = d["layer_1_stage_labelled_sources"]["kept"]["infants"]

    fig, ax = plt.subplots(figsize=(COL1, 3.05))
    ax.axis("off")
    ax.set_xlim(0, 10); ax.set_ylim(0, 10)

    BX, BW, CX = 0.15, 5.75, 0.15 + 5.75 / 2          # box left, width, centre
    TX = BX + BW + 0.30                                # exclusion text column

    for y, txt in ((9.05, f"{acquired:,} images acquired\n11 sources"),
                   (6.35, f"{labelled:,} stage-labelled\n{infants_l:,} infants"),
                   (3.35, "8,024 eligible, 6-class\n1,760 infants")):
        ax.add_patch(Rectangle((BX, y - 0.62), BW, 1.24, facecolor=DARK, edgecolor="none"))
        ax.text(CX, y, txt, ha="center", va="center", color="white", fontsize=7.2,
                linespacing=1.35)

    for ytop, ybot, txt in (
        (8.43, 6.97, "\u2212 2,610\nno stage label,\nduplicates"),
        (5.73, 3.97, "\u2212 1,295 other pathology\n(43% of the positive class)\n"
                     "\u2212 1,063 treated / regressed\n\u2212 45 stage 0"),
    ):
        ax.annotate("", xy=(CX, ybot), xytext=(CX, ytop),
                    arrowprops=dict(arrowstyle="-|>", color=INK, lw=0.8, shrinkA=0, shrinkB=0))
        ax.text(TX, (ytop + ybot) / 2, txt, ha="left", va="center", color=ACCENT,
                fontsize=5.8, linespacing=1.3)

    ax.annotate("", xy=(CX, 2.05), xytext=(CX, 2.73),
                arrowprops=dict(arrowstyle="-|>", color=INK, lw=0.8, shrinkA=0, shrinkB=0))
    for x, w, txt in ((BX, 2.75, "6,700 CV pool\n1,528 infants"),
                      (BX + 3.0, 2.90, "1,324 held-out\nhospital")):
        ax.add_patch(Rectangle((x, 0.95), w, 1.10, facecolor=LIGHT, edgecolor=INK, lw=0.5))
        ax.text(x + w / 2, 1.50, txt, ha="center", va="center", color=INK, fontsize=6.6,
                linespacing=1.3)
    ax.text(BX + 3.0 + 1.45, 0.50, "dev 663  \u00b7  LOCKED 661",
            ha="center", va="center", color=ACCENT, fontsize=6.2, style="italic")
    finish(fig, "fig1_cohort")


# ── Fig. 2 · the unit effect, and its anatomical cause ───────────────────────
def fig2_unit_effect() -> None:
    d = json.loads((R / "locked_eval" / "locked_table.json").read_text())
    cells = list(d["cells"])
    img = [d["cells"][c]["primary_per_checkpoint"]["qwk"]["mean"] for c in cells]
    imgsd = [d["cells"][c]["primary_per_checkpoint"]["qwk"]["sd"] for c in cells]
    eye = [d["cells"][c]["coprimary_eye_level"]["qwk"]["mean"] for c in cells]
    eyesd = [d["cells"][c]["coprimary_eye_level"]["qwk"]["sd"] for c in cells]
    labels = [c.replace("structured_s", "struct.\ns").replace("flat_s", "flat\ns")
              for c in cells]

    fig, axes = plt.subplots(1, 2, figsize=(COL2, 2.45),
                             gridspec_kw={"width_ratios": [1.15, 0.85], "wspace": 0.52})

    # (a) four cells, two units
    ax = axes[0]
    xp = np.arange(len(cells))
    ax.bar(xp - 0.19, img, 0.36, yerr=imgsd, capsize=2, color=LIGHT, edgecolor=INK,
           lw=0.6, hatch="////", label="per image", error_kw=dict(ecolor=INK, lw=0.7))
    ax.bar(xp + 0.19, eye, 0.36, yerr=eyesd, capsize=2, color=DARK, edgecolor=INK,
           lw=0.6, label="per eye", error_kw=dict(ecolor=INK, lw=0.7))
    for x, v in zip(xp - 0.19, img):
        ax.text(x, v + 0.055, f"{v:.2f}", ha="center", fontsize=6.6, color=INK)
    for x, v in zip(xp + 0.19, eye):
        ax.text(x, v + 0.055, f"{v:.2f}", ha="center", fontsize=6.6, color=INK,
                fontweight="bold")
    ax.set_xticks(xp); ax.set_xticklabels(labels, fontsize=6.6, linespacing=1.25)
    ax.set_ylabel("quadratic-weighted $\\kappa$")
    ax.set_ylim(0, 1.0)
    tidy(ax)
    ax.legend(frameon=False, ncol=2, loc="upper center", handlelength=1.5,
              columnspacing=1.2, borderpad=0.1)
    ax.set_title("(a) identical predictions, two units", loc="left", fontsize=7.6, pad=4)

    # (b) the anatomical cause
    ax = axes[1]
    v = d["cells"]["structured_s42"]["view_miss_table_ckpt_mean"]
    names = list(v)
    vals = [v[k]["called_normal_mean"] for k in names]
    ns = [v[k]["n"] for k in names]
    order = np.argsort(vals)
    names = [names[i].replace("_", " ") for i in order]
    vals = [vals[i] for i in order]; ns = [ns[i] for i in order]
    cols = [ACCENT if x > 0.4 else DARK for x in vals]
    bars = ax.barh(np.arange(len(names)), vals, 0.62, color=cols, edgecolor=INK, lw=0.5)
    for b, x, n in zip(bars, vals, ns):
        ax.text(x + 0.018, b.get_y() + b.get_height() / 2, f"{x*100:.1f}%",
                va="center", fontsize=6.8, color=INK, fontweight="bold")
    ax.set_yticks(np.arange(len(names)))
    ax.set_yticklabels([f"{n_.replace('posterior pole', 'posterior')}\n(n={c})"
                        for n_, c in zip(names, ns)], fontsize=6.4, linespacing=1.2)
    ax.set_xlim(0, 1.02)
    ax.set_xlabel("stage 1–3 frames called Normal")
    tidy(ax, grid="x")
    ax.set_title("(b) where the misses are", loc="left", fontsize=7.6, pad=4)
    finish(fig, "fig2_unit_effect")


# ── Fig. 3 · confusion matrix, locked set ────────────────────────────────────
def fig3_confusion() -> None:
    fs = sorted(glob.glob(str(R / "locked_eval/structured_s42_ckpt*.npz")))
    yt = np.load(fs[0], allow_pickle=True)["y_true"]
    preds = [np.load(f, allow_pickle=True)["y_pred"] for f in fs]
    cm = np.zeros((6, 6))
    for p in preds:
        for t, q in zip(yt, p):
            cm[t, q] += 1
    row = cm.sum(1, keepdims=True)
    cmn = cm / np.maximum(row, 1)
    names = ["Norm", "S1", "S2", "S3", "S4/5", "AP"]

    fig, ax = plt.subplots(figsize=(COL1, 2.85))
    im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    for i in range(6):
        for j in range(6):
            if row[i, 0] == 0:
                continue
            ax.text(j, i, f"{cmn[i, j]*100:.0f}", ha="center", va="center", fontsize=6.4,
                    color="white" if cmn[i, j] > 0.55 else INK)
    # mark the column the errors flow into
    ax.add_patch(Rectangle((-0.5, -0.5), 1, 6, fill=False, edgecolor=ACCENT, lw=1.3))
    ax.set_xticks(range(6)); ax.set_xticklabels(names, fontsize=6.8)
    ax.set_yticks(range(6)); ax.set_yticklabels(
        [f"{n} ({int(r)})" for n, r in zip(names, row.ravel() / len(preds))], fontsize=6.6)
    ax.set_xlabel("predicted"); ax.set_ylabel("reference standard")
    ax.tick_params(length=0, colors=MUTED)
    for s in ax.spines.values():
        s.set_color(RULE)
    cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
    cb.ax.tick_params(labelsize=6, length=2, colors=MUTED)
    cb.outline.set_edgecolor(RULE)
    cb.set_label("row %", fontsize=6.8, color=MUTED)
    finish(fig, "fig3_confusion")


if __name__ == "__main__":
    print(f"writing to {OUT.relative_to(ROOT)}/")
    fig1_cohort()
    fig2_unit_effect()
    fig3_confusion()
    print("\nIEEE geometry: fig1/fig3 single column (3.5 in), fig2 double column (7.16 in).")
