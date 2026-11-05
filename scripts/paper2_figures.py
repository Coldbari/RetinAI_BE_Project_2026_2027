"""Figures for paper 2 (IEEEtran, two columns), from results/rop/v3/report_data.json and v3_scores.json only.

    .venv/bin/python scripts/paper2_figures.py          # -> docs/paper/figs2/*.pdf (+ .png for the Word copy)

Double-column figures reuse scripts/rop_v3_figures.py (7.2 in, printed at \\textwidth = 7.16 in); single-column
figures are drawn here at the IEEE column width (3.5 in) so their text prints at 7-8 pt. No model is run and no
number is typed: every value is read from the two committed result files.
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from scripts import rop_v3_figures as V3F  # noqa: E402  data, palette, names, rcParams

import matplotlib.pyplot as plt  # noqa: E402

V3F.OUT = ROOT / "docs/paper/figs2"
plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})   # embed TrueType, not Type 3 (IEEE PDF checks)
R, SCO, CL, CLN, COL, NAMES = V3F.R, V3F.SCO, V3F.CL, V3F.CLN, V3F.COL, V3F.NAMES
COLW = 3.5          # IEEE column width, inches
SHORT = {c: CLN[c].replace("Stage ", "S").replace("AP-ROP", "AP") for c in CL}
SMALL = {"font.size": 7, "axes.titlesize": 7.5, "axes.labelsize": 7, "legend.fontsize": 6.5,
         "xtick.labelsize": 6.5, "ytick.labelsize": 6.5}


def fig_dataset_col():
    pool = R["dataset"]["pool"]["by_source"]
    srcs = ["ostrava", "ropvl", "shenzhen"]
    cmap = ["#9ca3af", "#fde68a", "#fbbf24", "#f59e0b", "#b45309", "#7f1d1d"]
    with plt.rc_context(SMALL):
        fig, axes = plt.subplots(2, 1, figsize=(COLW, 2.9), gridspec_kw={"height_ratios": [1.0, 1.0], "hspace": 0.95})
        ax = axes[0]
        left = np.zeros(len(srcs))
        lab = [f"{V3F.SRC[s]} ({pool[s]['infants']:,})" for s in srcs]
        for c, col in zip(CL, cmap):
            v = np.array([pool[s]["by_class"][c] for s in srcs])
            ax.barh(lab, v, left=left, color=col, label=CLN[c], edgecolor="white", linewidth=0.4)
            left += v
        for i, s in enumerate(srcs):
            ax.text(left[i] + 40, i, f"{pool[s]['images']:,}", va="center", fontsize=6.5)
        ax.set_xlim(0, left.max() * 1.13); ax.invert_yaxis()
        ax.set_xlabel("Photographs (infants in brackets)")
        ax.set_title(f"(a) Cross-validation pool, {R['dataset']['pool']['images']:,} photographs", loc="left")
        ax.legend(ncol=6, frameon=False, loc="upper center", bbox_to_anchor=(0.42, -0.42), fontsize=6, handlelength=1.0,
                  columnspacing=0.8)
        ax = axes[1]
        dev = R["dataset"]["dev"]
        x = np.arange(6)
        ax.bar(x - 0.2, [dev["images_by_class"][c] for c in CL], 0.4, color="#9ca3af", label="photographs")
        ax.bar(x + 0.2, [dev["eyes_by_class"][c] for c in CL], 0.4, color="#1b6ca8", label="eyes")
        for i, c in enumerate(CL):
            ax.text(i + 0.2, dev["eyes_by_class"][c] + 3, str(dev["eyes_by_class"][c]), ha="center", fontsize=6)
        ax.set_xticks(x, [SHORT[c] for c in CL])
        ax.set_title(f"(b) Unseen hospital, development half: {dev['images']} photographs, {dev['eyes']} eyes", loc="left")
        ax.legend(frameon=False)
        V3F.save(fig, "fig_dataset")


def fig_loso_col(arms=V3F.LOSO_ARMS):
    srcs = SCO["loso"]["B"]["sources"]
    with plt.rc_context(SMALL):
        fig, ax = plt.subplots(figsize=(COLW, 2.15))
        for j, s in enumerate(srcs):
            for k, a in enumerate(arms):
                L = SCO["loso"][a][s]
                y = j + (k - (len(arms) - 1) / 2) * 0.25
                ax.plot(L["ci95"]["qwk_5class"], [y, y], color=COL[a], lw=1.6, solid_capstyle="round")
                ax.plot(L["qwk_5class"], y, "o", color=COL[a], ms=4, mec="white", mew=0.6,
                        label=f"{a} ({SCO['loso'][a]['pool']} pool{', v4 recipe' if a == 'I' else ''}): mean "
                              f"{SCO['loso'][a]['mean_qwk_5class']:.3f}" if j == 0 else None)
                ax.text(1.005, y, f"{L['qwk_5class']:.3f}", va="center", fontsize=6, color=COL[a],
                        transform=ax.get_yaxis_transform())
        L0 = SCO["loso"]["B"]
        ax.set_yticks(range(len(srcs)), [f"{V3F.SRC[s]} held out\n{L0[s]['n_images']:,} ph., {L0[s]['n_infants']} inf." for s in srcs])
        ax.set_ylim(len(srcs) - 0.5, -0.5)
        ax.set_xlabel("QWK (5-class), photograph level, 95% infant-clustered CI")
        ax.grid(axis="x", lw=0.4, alpha=0.5)
        ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.4, -0.27), ncol=1, fontsize=6.3)
        V3F.save(fig, "fig_loso")


def fig_calibration_col(arms=("B", "I")):
    with plt.rc_context(SMALL):
        fig, ax = plt.subplots(figsize=(COLW, 2.3))
        ax.plot([0, 1], [0, 1], "--", color="#9ca3af", lw=0.7)
        for a in arms:
            rel = R["arms"][a]["reliability_dev_eye"]
            for key, lab, ls, mk in (("uncalibrated", "as trained", "-", "o"), ("cv_temperature", "CV temperature", "--", "s")):
                b = rel[key]["bins"]
                ax.plot([x["confidence"] for x in b], [x["accuracy"] for x in b], ls, color=COL[a], marker=mk, ms=3,
                        lw=1.1, alpha=1 if key == "uncalibrated" else 0.75, label=f"{a}, {lab} (ECE {rel[key]['ece']:.3f})")
        ax.set_xlim(0.2, 1); ax.set_ylim(0, 1)
        ax.set_xlabel("Confidence of the eye's predicted stage"); ax.set_ylabel("Share correct")
        ax.legend(frameon=False, loc="upper left", fontsize=6.2)
        V3F.save(fig, "fig_calibration")


def fig_confusion_BI(a="B", b="I"):
    lab = [SHORT[c] for c in CL]
    cm = {k: np.array(R["arms"][k]["dev_eye"]["confusion"]["counts"]) for k in (a, b)}
    share = {k: np.array(R["arms"][k]["dev_eye"]["confusion"]["row_normalised"]) for k in (a, b)}
    diff = cm[b] - cm[a]
    lim = np.abs(diff).max()
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.4), gridspec_kw={"wspace": 0.12})
    for ax, k in zip(axes[:2], (a, b)):
        ax.imshow(share[k], cmap="Oranges", vmin=0, vmax=1)
        for i in range(6):
            for j in range(6):
                v = cm[k][i, j]
                txt = f"{v:.0f}" if abs(v - round(v)) < 1e-6 else f"{v:.1f}"
                ax.text(j, i, f"{txt}\n{share[k][i, j]:.0%}", ha="center", va="center", fontsize=5.6,
                        color="white" if share[k][i, j] > 0.55 else "black")
        ax.set_title(f"({'ab'[k == b]}) {NAMES[k]}", fontsize=8)
    axes[2].imshow(diff, cmap="RdBu", vmin=-lim, vmax=lim)
    for i in range(6):
        for j in range(6):
            if abs(diff[i, j]) >= 0.05:
                axes[2].text(j, i, f"{diff[i, j]:+.1f}", ha="center", va="center", fontsize=5.8)
    axes[2].set_title(f"(c) {b} − {a}, eyes (blue: more in {b})", fontsize=8)
    for k, ax in enumerate(axes):
        ax.set_xticks(range(6), lab, fontsize=6.5)
        ax.set_yticks(range(6), lab if k == 0 else [], fontsize=6.5)
        ax.set_xlabel("Predicted stage", fontsize=7.5)
        ax.spines[:].set_visible(False)
    axes[0].set_ylabel("Reference stage", fontsize=7.5)
    V3F.save(fig, "fig_confusion_BI")


def main():
    print("->", V3F.OUT.relative_to(ROOT))
    fig_dataset_col()
    V3F.fig_compare(size=(7.2, 2.75))
    V3F.fig_contrasts(size=(7.2, 2.85))
    V3F.fig_stage_recall(size=(7.2, 2.55))
    fig_confusion_BI()
    fig_loso_col()
    fig_calibration_col()


if __name__ == "__main__":
    sys.exit(main())
