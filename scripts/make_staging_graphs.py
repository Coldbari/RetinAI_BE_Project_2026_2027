"""Per-model and comparison figures for the ROP 6-class staging benchmark.

    .venv/bin/python scripts/make_staging_graphs.py

Reads  results/rop/staging_runs/staging_<arch>_s<seed>.json   (final metrics)
       results/rop/staging_runs/rop-staging-<arch>-s<seed>.log (per-epoch curves)
Writes graphs/rop_staging/<arch>/…                             one folder per model
       graphs/rop_staging/00_comparison_*.png                  across all models

EVERY FIGURE IS STAMPED PROVISIONAL. These are single-fold, single-seed results. The spread
across five architectures is 0.028 macro-F1 while a single fold gives no spread estimate at
all, so any apparent ranking here is unsupported. The stamp is in the figure itself, not just
in a caption, because figures get pasted into slides without their captions.

Palette is the project's existing one (scripts/make_dataset_graphs.py), re-validated for this
use: 5 categorical slots, worst adjacent pair dE 8.4 protan / 19.8 normal, all six checks PASS
against the dark surface. Series are ALSO direct-labelled, which is required rather than
optional when a CVD pair sits in the 6-8 band.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "results" / "rop" / "staging_runs"
AUDIT = ROOT / "results" / "rop" / "staging_shortcut_audit.json"
sys.path.insert(0, str(Path(__file__).resolve().parent))
SEED = 42

# Palette + surface come from the shared theme module so the dark (screen/deck) and light
# (print) variants cannot drift apart. FIG_THEME=light writes to graphs/rop_staging/light/.
# Fixed hue order, assigned by MODEL IDENTITY and never by rank — a model keeps its colour
# whichever position it lands in, so a re-sorted chart is still readable against an earlier one.
from figtheme import COLOR, GOOD, BAD, GRID, INK, INK2, SURF, outdir  # noqa: E402

OUT = outdir(ROOT / "graphs" / "rop_staging")

LABEL = {
    "effnetv2s": "EfficientNetV2-S", "convnext_in12k": "ConvNeXt-T in12k",
    "caformer": "CAFormer-S18", "deit3": "DeiT3-S", "effnetb0": "EfficientNet-B0",
}
PARAMS = {"effnetv2s": "20.2M", "convnext_in12k": "27.8M", "caformer": "24.3M",
          "deit3": "21.8M", "effnetb0": "4.0M"}
CLASSES = ["Normal", "Stage 1", "Stage 2", "Stage 3", "Stage 4/5", "AP-ROP"]


def style(ax, title=None, xlabel=None, ylabel=None):
    ax.set_facecolor(SURF)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9)
    if title:
        ax.set_title(title, color=INK, fontsize=12, fontweight="bold", pad=12, loc="left")
    if xlabel:
        ax.set_xlabel(xlabel, color=INK2, fontsize=9.5)
    if ylabel:
        ax.set_ylabel(ylabel, color=INK2, fontsize=9.5)
    ax.grid(axis="y", color=GRID, lw=0.8, alpha=0.6)
    ax.set_axisbelow(True)


def fig_new(w, h):
    f, ax = plt.subplots(figsize=(w, h))
    f.patch.set_facecolor(SURF)
    return f, ax


def stamp(fig, extra=""):
    """PROVISIONAL, in the figure itself — figures get pasted without their captions."""
    fig.text(0.995, 0.012, "PROVISIONAL — single fold, single seed (42); no spread estimate. "
             + extra, ha="right", color="#8a8a80", fontsize=7.5, style="italic")


def save(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, facecolor=SURF, bbox_inches="tight")
    plt.close(fig)
    print(f"  {path.relative_to(ROOT)}")


# ── data ─────────────────────────────────────────────────────────────────────
EPOCH_RE = re.compile(r"Epoch\s+(\d+)/(\d+)\s+loss ([\d.]+)\s+acc ([\d.]+)%\s+F1 ([\d.]+)%")


def load_curves(arch):
    """Per-epoch (loss, acc, macro-F1) split by which fold the run was on."""
    p = RUNS / f"rop-staging-{arch.replace('_', '-')}-s{SEED}.log"
    if not p.exists():
        return {}
    lines = [e.get("data", "").rstrip() for e in json.loads(p.read_text())]
    curves, cur = {}, None
    for line in lines:
        if line.startswith("=== "):
            # "=== effnetv2s | fold0 | train ..." -> fold0
            cur = line.split("|")[1].strip()
            curves[cur] = {"epoch": [], "loss": [], "acc": [], "f1": []}
        m = EPOCH_RE.match(line)
        if m and cur:
            curves[cur]["epoch"].append(int(m.group(1)))
            curves[cur]["loss"].append(float(m.group(3)))
            curves[cur]["acc"].append(float(m.group(4)) / 100)
            curves[cur]["f1"].append(float(m.group(5)) / 100)
    return curves


def load_runs():
    out = {}
    for p in sorted(RUNS.glob(f"staging_*_s{SEED}.json")):
        d = json.loads(p.read_text())
        out[d["arch"]] = d
    return out


# ── per-model figures ────────────────────────────────────────────────────────
def fig_curves(arch, curves):
    """Train loss against val macro-F1. Two measures, two scales -> two stacked axes,
    never a dual-axis chart on one frame."""
    if not curves:
        return
    folds = [f for f in ("fold0", "foldsite") if f in curves]
    fig, axes = plt.subplots(2, len(folds), figsize=(5.6 * len(folds), 6.4), sharex=True)
    fig.patch.set_facecolor(SURF)
    axes = np.atleast_2d(axes.reshape(2, len(folds)))
    c = COLOR[arch]
    for j, f in enumerate(folds):
        d = curves[f]
        name = "fold 0 (held-out patients)" if f == "fold0" else "cross-site (Shenzhen held out)"
        ax = axes[0, j]
        ax.plot(d["epoch"], d["loss"], color=c, lw=2)
        style(ax, title=name, ylabel="train loss" if j == 0 else None)
        ax = axes[1, j]
        ax.plot(d["epoch"], d["f1"], color=c, lw=2, label="val macro-F1")
        # The LAST epoch is the reported score — no early stopping, no best-epoch pick.
        ax.scatter([d["epoch"][-1]], [d["f1"][-1]], s=46, color=c, zorder=5,
                   edgecolor=SURF, lw=2)
        ax.annotate(f"reported {d['f1'][-1]:.3f}", (d["epoch"][-1], d["f1"][-1]),
                    textcoords="offset points", xytext=(-8, 10), ha="right",
                    color=INK, fontsize=9, fontweight="bold")
        peak = max(d["f1"])
        if peak > d["f1"][-1] + 0.005:
            ax.axhline(peak, color=INK2, ls=":", lw=1.2)
            ax.annotate(f"peak {peak:.3f} — NOT reported (would be selection on noise)",
                        (d["epoch"][0], peak), textcoords="offset points", xytext=(2, 5),
                        color=INK2, fontsize=8, style="italic")
        style(ax, ylabel="val macro-F1" if j == 0 else None, xlabel="epoch")
        ax.set_ylim(0, 1)
    fig.suptitle(f"{LABEL[arch]}  ({PARAMS[arch]})  —  training behaviour",
                 color=INK, fontsize=13.5, fontweight="bold", x=0.02, ha="left", y=0.985)
    stamp(fig)
    fig.tight_layout(rect=[0, 0.02, 1, 0.96])
    save(fig, OUT / arch / "01_training_curves.png")


def fig_per_class(arch, run):
    """Per-class precision and recall. The clinically load-bearing view: macro-F1 alone
    hides which classes the model is failing on."""
    pc = run["folds"].get("fold_0", {}).get("per_class")
    if not pc:
        return
    names = [c for c in CLASSES if c in pc]
    rec = [pc[c]["recall"] for c in names]
    prec = [pc[c]["precision"] for c in names]
    sup = [pc[c]["support"] for c in names]
    # Reversed so Normal reads at the TOP and the ICROP stages descend in order — barh puts
    # index 0 at the bottom, which silently inverted an ordinal scale.
    y = np.arange(len(names))[::-1]
    fig, ax = fig_new(9.4, 5.0)
    ax.barh(y + 0.19, rec, height=0.34, color=COLOR[arch], edgecolor=SURF, lw=1.6,
            label="recall (of true cases, how many found)")
    ax.barh(y - 0.19, prec, height=0.34, color="#6d6d64", edgecolor=SURF, lw=1.6,
            label="precision (of predictions, how many right)")
    for i, (r, p) in enumerate(zip(rec, prec)):
        ax.text(r + 0.012, y[i] + 0.19, f"{r:.2f}", va="center", color=INK, fontsize=9,
                fontweight="bold")
        ax.text(p + 0.012, y[i] - 0.19, f"{p:.2f}", va="center", color=INK2, fontsize=9)
    # Support goes INTO the tick label. As a separate text at x<0 it printed on top of the
    # class name, which is only visible once the figure is actually rendered.
    ax.set_yticks(y, [f"{n}\nn={s}" for n, s in zip(names, sup)], color=INK, fontsize=9.5)
    ax.set_xlim(0, 1.12)
    style(ax, title=f"{LABEL[arch]} — per-class performance, fold 0",
          xlabel="score (1.0 = perfect)")
    ax.grid(axis="x", color=GRID, lw=0.8, alpha=0.6)
    ax.grid(axis="y", visible=False)
    leg = ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2, frameon=False,
                    fontsize=9)
    for t in leg.get_texts():
        t.set_color(INK2)
    stamp(fig)
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    save(fig, OUT / arch / "02_per_class.png")


def fig_headline(arch, run):
    """The four headline numbers, in-fold vs cross-site, against the metadata bar."""
    f0, fs = run["folds"].get("fold_0", {}), run["folds"].get("fold_site", {})
    metrics = [("macro_f1", "macro-F1"), ("balanced_accuracy", "balanced acc"),
               ("qwk", "QWK (ordinal)"), ("accuracy", "accuracy")]
    have = [(k, lbl) for k, lbl in metrics
            if isinstance(f0.get(k), (int, float)) or isinstance(fs.get(k), (int, float))]
    x = np.arange(len(have))
    fig, ax = fig_new(8.4, 4.4)
    a = [f0.get(k) if isinstance(f0.get(k), (int, float)) else 0 for k, _ in have]
    b = [fs.get(k) if isinstance(fs.get(k), (int, float)) else 0 for k, _ in have]
    ax.bar(x - 0.2, a, width=0.36, color=COLOR[arch], edgecolor=SURF, lw=1.6, label="fold 0")
    ax.bar(x + 0.2, b, width=0.36, color="#6d6d64", edgecolor=SURF, lw=1.6,
           label="cross-site (Shenzhen)")
    for i, (va, vb) in enumerate(zip(a, b)):
        for off, v in ((-0.2, va), (0.2, vb)):
            if v:
                ax.text(i + off, v + 0.02, f"{v:.3f}", ha="center", color=INK, fontsize=9,
                        fontweight="bold")
    ax.set_xticks(x, [lbl for _, lbl in have], color=INK)
    ax.set_ylim(0, 1.1)
    style(ax, title=f"{LABEL[arch]} — headline metrics", ylabel="score")
    leg = ax.legend(frameon=False, fontsize=9, loc="upper right")
    for t in leg.get_texts():
        t.set_color(INK2)
    stamp(fig, "cross-site scores 4 classes (Shenzhen has no Stage 4/5 or AP-ROP).")
    fig.tight_layout()
    save(fig, OUT / arch / "03_headline_metrics.png")


# ── comparison figures ───────────────────────────────────────────────────────
def fig_compare(runs, bar):
    order = sorted(runs, key=lambda a: -runs[a]["folds"]["fold_0"]["macro_f1"])
    cv = [runs[a]["folds"]["fold_0"]["macro_f1"] for a in order]
    cs = [runs[a]["folds"].get("fold_site", {}).get("macro_f1") or 0 for a in order]
    y = np.arange(len(order))[::-1]
    fig, ax = fig_new(9.6, 5.0)
    ax.barh(y + 0.19, cv, height=0.34, color=[COLOR[a] for a in order], edgecolor=SURF, lw=1.6)
    ax.barh(y - 0.19, cs, height=0.34, color=[COLOR[a] for a in order], edgecolor=SURF,
            lw=1.6, alpha=0.45)
    for i, a in enumerate(order):
        yy = y[i]
        ax.text(cv[i] + 0.008, yy + 0.19, f"{cv[i]:.3f}  fold 0", va="center", color=INK,
                fontsize=9, fontweight="bold")
        ax.text(cs[i] + 0.008, yy - 0.19, f"{cs[i]:.3f}  cross-site", va="center",
                color=INK2, fontsize=9)
    if bar:
        ax.axvline(bar, color=BAD, ls="--", lw=1.8)
        # Anchored to the TOP of the plot area. Placed below the lowest bar it collided with
        # the x tick labels, which is the kind of thing only rendering the figure reveals.
        ax.text(bar + 0.008, y[0] + 1.02, f"metadata-only bar {bar:.3f} (no pixels decoded)",
                color=BAD, fontsize=8.5, va="center")
        # Headroom for that label. Without it the text sits behind the top bar — below the
        # bars it hit the tick labels instead, so the band has to be made, not borrowed.
        ax.set_ylim(y[-1] - 0.6, y[0] + 1.4)
    ax.set_yticks(y, [f"{LABEL[a]}\n{PARAMS[a]}" for a in order], color=INK, fontsize=9.5)
    ax.set_xlim(0, max(max(cv), max(cs)) + 0.16)
    style(ax, title="ROP 6-class staging — five backbones, fold 0", xlabel="macro-F1")
    ax.grid(axis="x", color=GRID, lw=0.8, alpha=0.6)
    ax.grid(axis="y", visible=False)
    spread = max(cv) - min(cv)
    fig.text(0.012, 0.060, f"Spread across all five architectures: {spread:.3f} macro-F1 "
             f"(4.0M to 27.8M params, CNN / ViT / hybrid). No model is separable from another "
             f"on one fold.", color=INK2, fontsize=9)
    stamp(fig)
    fig.tight_layout(rect=[0, 0.10, 1, 1])
    save(fig, OUT / "00_comparison_macro_f1.png")


def fig_compare_per_class(runs):
    """Per-class RECALL across models — where the failures actually are."""
    order = sorted(runs, key=lambda a: -runs[a]["folds"]["fold_0"]["macro_f1"])
    order = [a for a in order if runs[a]["folds"]["fold_0"].get("per_class")]
    if not order:
        return
    names = CLASSES
    x = np.arange(len(names))
    w = 0.8 / len(order)
    fig, ax = fig_new(11.0, 4.8)
    for i, a in enumerate(order):
        pc = runs[a]["folds"]["fold_0"]["per_class"]
        vals = [pc.get(c, {}).get("recall", 0) for c in names]
        ax.bar(x + (i - (len(order) - 1) / 2) * w, vals, width=w * 0.88, color=COLOR[a],
               edgecolor=SURF, lw=1.2, label=LABEL[a])
    ax.set_xticks(x, names, color=INK)
    ax.set_ylim(0, 1.05)
    style(ax, title="Per-class recall across backbones, fold 0 — where the model fails",
          ylabel="recall")
    leg = ax.legend(frameon=False, fontsize=9, ncol=len(order), loc="upper center",
                    bbox_to_anchor=(0.5, -0.12))
    for t in leg.get_texts():
        t.set_color(INK2)
    missing = [LABEL[a] for a in runs if a not in order]
    note = ("Recall is the clinically load-bearing number: it is the fraction of true cases "
            "the model finds. Macro-F1 averages this away.")
    if missing:
        # Say who is absent. A four-bar chart in a five-model comparison otherwise reads as
        # "these are the models" rather than "one arm has no data".
        note += f"  {' and '.join(missing)} absent: run predates the per-class metric fix."
    fig.text(0.012, 0.055, note, color=INK2, fontsize=9)
    stamp(fig)
    fig.tight_layout(rect=[0, 0.13, 1, 1])
    save(fig, OUT / "00_comparison_per_class_recall.png")


def main():
    runs = load_runs()
    if not runs:
        print(f"no runs in {RUNS}", file=sys.stderr)
        return 1
    bar = None
    if AUDIT.exists():
        bar = json.loads(AUDIT.read_text()).get("G4_metadata_baseline", {}).get("bar_cv_macro_f1")

    print(f"{len(runs)} model(s): {', '.join(sorted(runs))}\n")
    for arch in sorted(runs):
        print(f"{LABEL.get(arch, arch)}:")
        fig_curves(arch, load_curves(arch))
        fig_per_class(arch, runs[arch])
        fig_headline(arch, runs[arch])
    print("\ncomparison:")
    fig_compare(runs, bar)
    fig_compare_per_class(runs)

    (OUT / "README.md").write_text(
        "# ROP 6-class staging — model figures\n\n"
        "**PROVISIONAL.** Single fold (fold 0), single seed (42). The spread across all five\n"
        "architectures is smaller than any spread estimate we have, so nothing here supports a\n"
        "ranking. Regenerate once the full 5-fold CV completes.\n\n"
        "- `00_comparison_macro_f1.png` — all five, in-fold and cross-site, vs the metadata bar\n"
        "- `00_comparison_per_class_recall.png` — per-class recall across models\n"
        "- `<model>/01_training_curves.png` — loss and val macro-F1 per epoch\n"
        "- `<model>/02_per_class.png` — precision and recall per ICROP class\n"
        "- `<model>/03_headline_metrics.png` — fold 0 vs cross-site\n\n"
        "Generated by `scripts/make_staging_graphs.py`.\n")
    print(f"\n-> {OUT.relative_to(ROOT)}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
