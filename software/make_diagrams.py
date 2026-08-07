"""Generate the two diagrams the README references.

    python software/make_diagrams.py

Writes images/system_architecture.png and images/flowchart.png. Both are drawn
from this file alone -- no external assets, no hand editing -- so the diagram and
the description in the README can be kept in step by editing one place.

Palette is the colourblind-validated one used by the evidence figures:
DR blue #3987e5, ROP orange #d95926, Glaucoma aqua #199e70.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon

INK = "#14202e"
MUTED = "#5b6b7d"
LINE = "#9aa9b8"
PAPER = "#ffffff"
BAND_TRAIN = "#eef3f9"
BAND_SERVE = "#f2f7f4"
DR = "#3987e5"
ROP = "#d95926"
GLA = "#199e70"
GATE = "#8e5bb5"
OUT = MUTED

IMAGES = Path(__file__).resolve().parents[1] / "images"


def box(ax, x, y, w, h, title, sub=None, colour=INK, fill=PAPER, lw=1.6, fs=10.5):
    ax.add_patch(
        FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.006,rounding_size=0.016",
            linewidth=lw, edgecolor=colour, facecolor=fill, zorder=3,
        )
    )
    # Offsets scale with the box so a short box never pushes its subtitle out.
    ty = (y + h - h * 0.30) if sub else (y + h / 2)
    ax.text(x + w / 2, ty, title, ha="center", va="center",
            fontsize=fs, color=INK, fontweight="600", zorder=4)
    if sub:
        ax.text(x + w / 2, y + h * 0.34, sub, ha="center", va="center",
                fontsize=8.1, color=MUTED, linespacing=1.45, zorder=4)


def arrow(ax, xy_from, xy_to, colour=LINE, lw=1.5, style="-|>", rad=0.0):
    ax.add_patch(
        FancyArrowPatch(
            xy_from, xy_to, arrowstyle=style, mutation_scale=13,
            linewidth=lw, color=colour, zorder=2,
            connectionstyle=f"arc3,rad={rad}",
            shrinkA=2, shrinkB=2,
        )
    )


def band(ax, x, y, w, h, fill, label):
    ax.add_patch(
        FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.008,rounding_size=0.014",
            linewidth=0, facecolor=fill, zorder=1,
        )
    )
    ax.text(x + 0.012, y + h - 0.028, label, ha="left", va="top",
            fontsize=9, color=MUTED, fontweight="700", zorder=2)


def new_ax(w, h):
    fig, ax = plt.subplots(figsize=(w, h), dpi=190)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.patch.set_facecolor(PAPER)
    return fig, ax


def system_architecture():
    fig, ax = new_ax(13.0, 7.4)

    ax.text(0.5, 0.972, "RetinAI — System Architecture", ha="center", va="top",
            fontsize=15.5, color=INK, fontweight="700")
    ax.text(0.5, 0.930,
            "Training runs on Kaggle GPU; serving runs CPU-only. "
            "One shared preprocessing module is used by both.",
            ha="center", va="top", fontsize=9.4, color=MUTED)

    # ---------------------------------------------------------------- training
    band(ax, 0.035, 0.560, 0.930, 0.320, BAND_TRAIN, "TRAINING  ·  Kaggle GPU (P100 / T4)")

    ty, th = 0.610, 0.185
    box(ax, 0.055, ty, 0.150, th, "Datasets",
        "EyePACS · APTOS\nSMDG-19 · G1020\nROP database")
    box(ax, 0.235, ty, 0.160, th, "Kaggle kernel",
        "pin torch 2.5.1\nglob-discover mounts\nextract archives")
    box(ax, 0.425, ty, 0.160, th, "prepare_data.py",
        "pooled manifest\npatient-grouped\nsplit · dataset QA")
    box(ax, 0.615, ty, 0.160, th, "train.py",
        "config-driven loop\nAMP · OneCycleLR\nfocal + CB weights")
    box(ax, 0.805, ty, 0.150, th, "results/<disease>/",
        "weights.pth\nmetrics.json\nplots · errors")

    for a, b in [(0.205, 0.235), (0.395, 0.425), (0.585, 0.615), (0.775, 0.805)]:
        arrow(ax, (a, ty + th / 2), (b, ty + th / 2))

    # weights handoff
    arrow(ax, (0.880, ty), (0.880, 0.505), colour=MUTED, lw=1.7)
    ax.text(0.893, 0.545, "weights pulled back\n(Git LFS)", ha="left", va="center",
            fontsize=8.2, color=MUTED, linespacing=1.4)

    # ----------------------------------------------------------------- serving
    band(ax, 0.035, 0.075, 0.930, 0.400, BAND_SERVE, "SERVING  ·  CPU-only (HuggingFace Spaces)")

    sy, sh = 0.240, 0.150
    box(ax, 0.055, sy, 0.140, sh, "Capture",
        "web upload\nor Android\nCameraX", colour=MUTED)
    box(ax, 0.222, sy, 0.140, sh, "Gradability gate",
        "redness · blur\nluminance · FOV\nreject, don't score", colour=GATE)
    box(ax, 0.389, sy, 0.140, sh, "Context router",
        "infant → ROP\nadult → DR, Gla.\nsuppress impossible", colour=GATE)
    box(ax, 0.556, sy, 0.140, sh, "Preprocess",
        "circle-crop\nCLAHE\nBen-Graham (DR)")

    for a, b in [(0.195, 0.222), (0.362, 0.389), (0.529, 0.556)]:
        arrow(ax, (a, sy + sh / 2), (b, sy + sh / 2))

    # three models — the stack is centred on the row and may exceed its height
    mx, mw, mh, mgap = 0.742, 0.140, 0.055, 0.012
    stack_h = 3 * mh + 2 * mgap
    row_mid = sy + sh / 2
    stack_top = row_mid + stack_h / 2
    stack_bottom = row_mid - stack_h / 2
    for i, (name, arch, colour) in enumerate([
        ("DR", "EfficientNetV2-S", DR),
        ("ROP", "ResNet50", ROP),
        ("Glaucoma", "EfficientNetV2-S", GLA),
    ]):
        my = stack_top - mh - i * (mh + mgap)
        box(ax, mx, my, mw, mh, name, arch, colour=colour, fs=9.6)
        arrow(ax, (0.696, row_mid), (mx, my + mh / 2), colour=LINE)

    ax.text(mx + mw / 2, stack_bottom - 0.014, "each with 5-view TTA",
            ha="center", va="top", fontsize=8.2, color=MUTED)

    # outputs
    oy, ow, oh = 0.098, 0.135, 0.052
    centres = []
    for i, (name, sub) in enumerate([
        ("Grad-CAM", "heatmap overlay"),
        ("Risk + grade", "calibrated confidence"),
        ("PDF report", "ReportLab, A4"),
    ]):
        ox = 0.222 + i * 0.172
        box(ax, ox, oy, ow, oh, name, sub, colour=OUT, fs=9.6)
        centres.append(ox + ow / 2)

    # route the output bus around the right of the stack, clear of the TTA note
    bus = oy + oh + 0.024
    spine = mx + mw + 0.045
    ax.plot([mx + mw, spine], [row_mid, row_mid], color=LINE, lw=1.6, zorder=2)
    ax.plot([spine, spine], [row_mid, bus], color=LINE, lw=1.6, zorder=2)
    ax.plot([centres[0], spine], [bus, bus], color=LINE, lw=1.6, zorder=2)
    for cxo in centres:
        arrow(ax, (cxo, bus), (cxo, oy + oh), colour=LINE, lw=1.6)

    # shared-preprocessing tie
    ax.annotate(
        "", xy=(0.626, 0.610), xytext=(0.626, sy + sh),
        arrowprops=dict(arrowstyle="<|-|>", color=GATE, lw=1.4,
                        linestyle=(0, (4, 3)), mutation_scale=11), zorder=2,
    )
    ax.text(0.638, (0.610 + sy + sh) / 2,
            "same preprocessing module\nin training and serving",
            ha="left", va="center", fontsize=8.3, color=GATE, linespacing=1.4)

    fig.savefig(IMAGES / "system_architecture.png", bbox_inches="tight",
                facecolor=PAPER, pad_inches=0.22)
    plt.close(fig)


def flowchart():
    fig, ax = new_ax(8.6, 12.4)

    ax.text(0.5, 0.985, "RetinAI — Screening Flow", ha="center", va="top",
            fontsize=15, color=INK, fontweight="700")

    cx = 0.355
    w = 0.42
    rows = [
        ("term", "Start", None, INK),
        ("proc", "Acquire fundus image", "web upload or Android CameraX capture", MUTED),
        ("proc", "Collect patient context", "age band: infant | adult", MUTED),
        ("dec", "Gradable?", None, GATE),
        ("proc", "Route by context", "infant → ROP     adult → DR, Glaucoma\nno context → run all, mark unrouted", GATE),
        ("proc", "Preprocess", "circle-crop · CLAHE · Ben-Graham (DR)\nresize 384 · ImageNet normalise", INK),
        ("proc", "Infer, 5-view TTA", "original · H-flip · V-flip · ±10°\naverage the softmax outputs", INK),
        ("proc", "Threshold and grade", "ROP thr 0.1933 · DR referable ≥ 2\n→ risk level → recommendation", INK),
        ("proc", "Explain and calibrate", "Grad-CAM on last conv layer\napply fitted temperature", INK),
        ("proc", "Generate PDF report", "ReportLab A4, with disclaimer", MUTED),
        ("term", "Stop", None, INK),
    ]

    top, gap = 0.930, 0.0855
    ys = []
    for i, (kind, *_rest) in enumerate(rows):
        h = 0.050 if kind == "dec" else (0.038 if kind == "term" else 0.060)
        y = top - i * gap - h
        ys.append((y, h))

    for i, ((kind, title, sub, colour), (y, h)) in enumerate(zip(rows, ys)):
        if kind == "dec":
            pts = [(cx + w / 2, y + h / 2), (cx, y + h), (cx - w / 2, y + h / 2), (cx, y)]
            ax.add_patch(Polygon(pts, closed=True, linewidth=1.7,
                                 edgecolor=colour, facecolor=PAPER, zorder=3))
            ax.text(cx, y + h / 2, title, ha="center", va="center",
                    fontsize=11, color=INK, fontweight="600", zorder=4)
            # criteria sit outside the diamond -- they will not fit inside one
            ax.text(cx - w / 2 - 0.020, y + h / 2,
                    "redness · saturation\nred-hue · blur\nluminance · FOV ratio",
                    ha="right", va="center", fontsize=7.9, color=MUTED,
                    linespacing=1.5, zorder=4)
        elif kind == "term":
            box(ax, cx - 0.10, y, 0.20, h, title, None, colour=colour, fs=10.5)
        else:
            box(ax, cx - w / 2, y, w, h, title, sub, colour=colour, fs=10.5)

        if i + 1 < len(rows):
            ny = ys[i + 1][0] + ys[i + 1][1]
            arrow(ax, (cx, y), (cx, ny))

    # reject branch off the decision diamond
    dy, dh = ys[3]
    dmid = dy + dh / 2
    rx, rw = cx + w / 2 + 0.062, 0.290
    rh = 0.058
    box(ax, rx, dmid - rh / 2, rw, rh, "Reject: ungradable",
        "return the specific reason;\nno model is run", colour=GATE, fs=9.6)
    arrow(ax, (cx + w / 2, dmid), (rx, dmid), colour=GATE)
    ax.text((cx + w / 2 + rx) / 2, dmid + 0.006, "no", ha="center", va="bottom",
            fontsize=8.6, color=GATE, fontweight="700")
    ax.text(cx + 0.012, dy - 0.016, "yes", ha="left", va="center",
            fontsize=8.6, color=GATE, fontweight="700")

    # the reject path skips every model and rejoins at the end
    ly, lh = ys[-1]
    spine = rx + rw / 2
    ax.plot([spine, spine], [dmid - rh / 2, ly + lh / 2], color=GATE, lw=1.4, zorder=2)
    arrow(ax, (spine, ly + lh / 2), (cx + 0.10, ly + lh / 2), colour=GATE, lw=1.4)

    fig.savefig(IMAGES / "flowchart.png", bbox_inches="tight",
                facecolor=PAPER, pad_inches=0.22)
    plt.close(fig)


if __name__ == "__main__":
    IMAGES.mkdir(exist_ok=True)
    system_architecture()
    flowchart()
    print(f"wrote {IMAGES/'system_architecture.png'}")
    print(f"wrote {IMAGES/'flowchart.png'}")
