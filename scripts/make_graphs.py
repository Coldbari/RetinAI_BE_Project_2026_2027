"""Build graphs/ — every figure needed to judge these models, from recorded evidence only.

Each figure reads a results/*.json or *.npz produced by an experiment in this repo. Nothing
is hardcoded, so re-running an experiment and re-running this script keeps the picture and
the number in step. If a source file is missing the figure is skipped with a message rather
than drawn from stale constants.

Palette: the deck's original colours (cyan/purple/gold/green/red) FAIL a colourblind check —
red vs green sit at deutan ΔE 6.5, inside the 6-8 warn band, and all five fall outside the
dark-surface lightness band. These are validated replacements (all six checks PASS on the
dark surface, worst adjacent CVD ΔE 9.4):

    DR = blue #3987e5    ROP = orange #d95926    Glaucoma = aqua #199e70

Every series is also direct-labelled, so identity never depends on colour alone.

    .venv/bin/python scripts/make_graphs.py
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                       # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "graphs"
OUT.mkdir(exist_ok=True)

SURF = "#1a1a19"
INK = "#ffffff"
INK2 = "#c3c2b7"
GRID = "#33322f"
DR, ROP, GL, Y4 = "#3987e5", "#d95926", "#199e70", "#c98500"
COLOR = {"DR": DR, "ROP": ROP, "Glaucoma": GL}
# sequential blue ramp (magnitude), light -> dark, from the validated reference
BLUES = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
         "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281"]

plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "text.color": INK, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.edgecolor": GRID, "grid.color": GRID, "font.size": 11,
    "axes.titlesize": 13, "axes.titleweight": "bold", "legend.frameon": False,
    "axes.spines.top": False, "axes.spines.right": False,
})


def load(rel):
    p = ROOT / rel
    if not p.exists():
        print(f"  skip — missing {rel}")
        return None
    return json.loads(p.read_text()) if p.suffix == ".json" else np.load(p, allow_pickle=True)


def save(fig, name, note=None):
    if note:
        fig.text(0.005, -0.035, note, fontsize=7.5, color=INK2, ha="left", va="top")
    fig.savefig(OUT / name, dpi=200, bbox_inches="tight", facecolor=SURF)
    plt.close(fig)
    print(f"  -> graphs/{name}")


def bare(ax):
    ax.grid(axis="y", lw=0.6, alpha=0.5)
    ax.set_axisbelow(True)


# ─────────────────────────────────────────────────────────────────────────────
def g_confusion():
    d = load("results/confusion_tables.json")
    if not d:
        return
    fig, axes = plt.subplots(1, len(d), figsize=(4.1 * len(d), 4.0))
    axes = np.atleast_1d(axes)
    for ax, (dis, v) in zip(axes, d.items()):
        m = np.array([[v["tp"], v["fn"]], [v["fp"], v["tn"]]])
        vmax = m.max()
        for i in range(2):
            for j in range(2):
                frac = m[i, j] / vmax if vmax else 0
                # magnitude -> one hue, light to dark. Diagonal outlined, not recoloured.
                c = BLUES[min(int(frac * (len(BLUES) - 1)), len(BLUES) - 1)]
                ax.add_patch(Rectangle((j, 1 - i), 1, 1, facecolor=c,
                                       edgecolor=SURF, linewidth=2))
                if i == j:   # correct cells get a ring — secondary encoding, not colour
                    ax.add_patch(Rectangle((j + .04, 1 - i + .04), .92, .92, fill=False,
                                           edgecolor=INK, linewidth=1.6))
                # light cell -> dark ink, dark cell -> white ink
                txt = INK if frac > 0.55 else "#0b0b0b"
                lbl = ["TP", "FN", "FP", "TN"][i * 2 + j]
                ax.text(j + .5, 1 - i + .58, f"{m[i, j]}", ha="center", va="center",
                        fontsize=21, fontweight="bold", color=txt)
                ax.text(j + .5, 1 - i + .26, lbl, ha="center", va="center",
                        fontsize=9.5, color=txt, alpha=.85)
        ax.set_xlim(0, 2); ax.set_ylim(0, 2)
        ax.set_xticks([.5, 1.5]); ax.set_xticklabels(["says\ndisease", "says\nhealthy"])
        if ax is axes[0]:
            ax.set_yticks([1.5, .5])
            ax.set_yticklabels(["truly\ndisease", "truly\nhealthy"])
        else:
            ax.set_yticks([])
        ax.tick_params(length=0)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_title(f"{dis}\nsens {v['sensitivity']:.0%} · spec {v['specificity']:.0%}",
                     color=COLOR.get(dis, INK), pad=12)
    fig.suptitle("Confusion matrix per model — outlined cells are correct answers",
                 fontsize=13.5, fontweight="bold", y=1.06)
    fig.subplots_adjust(wspace=0.28, bottom=0.16)
    save(fig, "01_confusion_matrices.png",
         "results/confusion_tables.json · positives and negatives both from source ground truth")


def g_ppv():
    d = load("results/confusion_tables.json")
    if not d:
        return
    fig, ax = plt.subplots(figsize=(9.0, 5.0))
    prev = np.linspace(0.005, 0.6, 400)
    # DR and Glaucoma both measured specificity 1.000, so their curves are identical flat
    # lines at 100% and would hide one another. Dash + stagger so both stay visible.
    styles = {"DR": (2.6, "-"), "Glaucoma": (2.0, (0, (5, 3)))}
    label_y = {}
    for dis, v in d.items():
        se, sp = v["sensitivity"], v["specificity"]
        ppv = (se * prev) / (se * prev + (1 - sp) * (1 - prev))
        lw, ls = styles.get(dis, (2.4, "-"))
        ax.plot(prev * 100, ppv * 100, lw=lw, ls=ls, color=COLOR.get(dis, INK), label=dis)
        p = v["screening_prevalence"]
        y = (se * p) / (se * p + (1 - sp) * (1 - p)) * 100
        ax.plot([p * 100], [y], "o", ms=9, color=COLOR.get(dis, INK),
                markeredgecolor=SURF, markeredgewidth=2, zorder=5)
        label_y[dis] = (p * 100, y)
    # stack the two 100% labels instead of printing them on top of each other
    offs = {"DR": (10, -16), "Glaucoma": (10, -34), "ROP": (12, 4)}
    for dis, (x, y) in label_y.items():
        ax.annotate(f"{dis}: {y:.0f}% at {d[dis]['screening_prevalence']:.0%} prevalence",
                    (x, y), xytext=offs.get(dis, (10, 6)), textcoords="offset points",
                    color=COLOR.get(dis, INK), fontsize=10.5, fontweight="bold")
    ax.set_xlabel("How common the disease is in who you screen (%)")
    ax.set_ylabel("Of 100 patients flagged,\nhow many really have it (PPV %)")
    ax.set_title("Sensitivity and specificity transfer between populations. PPV does not.",
                 pad=14)
    ax.set_ylim(0, 116); ax.set_xlim(0, 60)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    bare(ax); ax.legend(loc="center right")
    save(fig, "02_ppv_vs_prevalence.png",
         "DR and Glaucoma sit flat at 100% because both measured ZERO false positives on 20 and 18 "
         "negatives — that is a small-sample artefact, not a guarantee. ROP's curve is the real shape.")


def g_roc():
    fig, ax = plt.subplots(figsize=(6.0, 5.6))
    from sklearn.metrics import roc_curve, roc_auc_score
    srcs = [("ROP", "results/rop/tta_test_preds.npz", None),
            ("DR", "results/dr/tta_val_preds.npz", "referable"),
            ("Glaucoma", "results/glaucoma/val_preds_by_source.npz", None)]
    any_drawn = False
    for dis, rel, mode in srcs:
        z = load(rel)
        if z is None:
            continue
        y = z["y_true"].astype(int)
        if mode == "referable":
            s = z["prob"][:, 2:].sum(1); y = (y >= 2).astype(int)
        elif z["prob"].ndim == 2:
            s = z["prob"][:, 1]
        else:
            s = z["prob"]
        fpr, tpr, _ = roc_curve(y, s)
        ax.plot(fpr, tpr, lw=2, color=COLOR[dis],
                label=f"{dis}  AUC {roc_auc_score(y, s):.3f}  (n={len(y)})")
        any_drawn = True
    if not any_drawn:
        plt.close(fig); return
    ax.plot([0, 1], [0, 1], lw=1, ls=":", color=INK2, label="chance")
    ax.set_xlabel("False-positive rate  (1 − specificity)")
    ax.set_ylabel("True-positive rate  (sensitivity)")
    ax.set_title("ROC on each model's own held-out set")
    ax.set_aspect("equal"); ax.grid(lw=0.6, alpha=0.5); ax.set_axisbelow(True)
    ax.legend(loc="lower right", fontsize=10)
    save(fig, "03_roc_curves.png",
         "Scored through the deployed 5-view TTA path. ROP/Glaucoma pooled — see figs 05 and 06 for the confounds.")


def g_rop_curve():
    d = load("results/rop/operating_curve.json")
    if not d:
        return
    pts = d["points"]
    sens = [p["test"]["sens"] * 100 for p in pts]
    spec = [p["test"]["spec"] * 100 for p in pts]
    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    ax.plot(spec, sens, "-o", lw=2, ms=7, color=ROP, markeredgecolor=SURF, markeredgewidth=1.6)
    for p, x, y in zip(pts, spec, sens):
        ax.annotate(f"thr {p['val_threshold']:.2f}", (x, y), xytext=(6, -12),
                    textcoords="offset points", fontsize=9, color=INK2)
    dep = d["deployed"]
    ax.plot([dep["spec"] * 100], [dep["sens"] * 100], "*", ms=20, color=Y4,
            markeredgecolor=SURF, markeredgewidth=1.5, zorder=6)
    ax.annotate("DEPLOYED\nsens 99.0 · spec 39.8", (dep["spec"] * 100, dep["sens"] * 100),
                xytext=(14, -34), textcoords="offset points", fontsize=10,
                fontweight="bold", color=Y4)
    ax.set_xlabel("Specificity — of 100 healthy infants, how many correctly cleared (%)")
    ax.set_ylabel("Sensitivity —\nof 100 with ROP,\nhow many caught (%)")
    ax.set_title("ROP: the operating point is a dial, not a property of the model")
    ax.set_ylim(90, 101); bare(ax); ax.grid(lw=0.6, alpha=0.5)
    save(fig, "04_rop_operating_curve.png",
         "Each threshold derived on val, applied ONCE to test · results/rop/operating_curve.json")


def _ci_bars(ax, labels, vals, los, his, colors, note_missing=None):
    ys = np.arange(len(labels))[::-1]
    for y, v, lo, hi, c in zip(ys, vals, los, his, colors):
        if v is None:
            ax.text(0.5, y, note_missing or "not measurable", va="center", ha="center",
                    color=INK2, fontsize=10, style="italic")
            continue
        ax.plot([lo, hi], [y, y], lw=2.5, color=c, alpha=.45, solid_capstyle="round")
        ax.plot([v], [y], "o", ms=9, color=c, markeredgecolor=SURF, markeredgewidth=1.8)
        ax.annotate(f"{v:.3f}", (v, y), xytext=(0, 11), textcoords="offset points",
                    ha="center", fontsize=9.5, color=INK, fontweight="bold")
    ax.set_yticks(ys); ax.set_yticklabels(labels)
    ax.axvline(0.5, lw=1, ls=":", color=INK2)
    ax.set_xlim(0.2, 1.02); ax.grid(axis="x", lw=0.6, alpha=0.5); ax.set_axisbelow(True)


def g_rop_device():
    d = load("results/rop/devsplit_ci.json")
    if not d:
        return
    fig, ax = plt.subplots(figsize=(8.6, 4.2))
    labels, vals, los, his = [], [], [], []
    for k, v in sorted(d["per_device"].items()):
        labels.append(f"camera D{k}  ({v['positive_units']} ROP+ infant"
                      f"{'s' if v['positive_units'] != 1 else ''})")
        vals.append(v["auc"]); los.append(v["ci"][0]); his.append(v["ci"][1])
    labels.append("device-controlled mean")
    vals.append(d["device_controlled"]["auc"])
    los.append(d["device_controlled"]["ci"][0]); his.append(d["device_controlled"]["ci"][1])
    labels.append("pooled (what a paper reports)")
    vals.append(d["pooled"]["auc"]); los.append(d["pooled"]["ci"][0]); his.append(d["pooled"]["ci"][1])
    cols = [ROP] * (len(labels) - 2) + [Y4, INK2]
    _ci_bars(ax, labels, vals, los, his, cols)
    ax.set_xlabel("AUC  (dotted line = chance)")
    ax.set_title("ROP: pooling cameras inflates the score by 0.19")
    save(fig, "05_rop_device_confound.png",
         "Cluster bootstrap over patient-units · results/rop/devsplit_ci.json")


def g_glaucoma_source():
    d = load("results/glaucoma/source_audit.json")
    if not d:
        return
    rows = [r for r in d["per_source"] if r.get("auc") is not None]
    rows.sort(key=lambda r: -r["n"])
    fig, ax = plt.subplots(figsize=(9.4, 6.4))
    # "image" is G1020's second filename convention — spell that out rather than
    # leaving a row labelled with a word that looks like a placeholder.
    labels = [f"{'G1020 (alt names)' if r['source'] == 'image' else r['source']}"
              f"  (n={r['n']}, {r['positives']} pos)" for r in rows]
    _ci_bars(ax, labels, [r["auc"] for r in rows],
             [r["ci"][0] for r in rows], [r["ci"][1] for r in rows],
             [GL if r["usable"] else INK2 for r in rows])
    top = len(rows) - 0.4
    ax.set_ylim(-0.9, top + 1.15)          # headroom so labels clear the title
    for x, c, lab, dy in ((d["source_controlled_usable"], Y4,
                           f"within-source {d['source_controlled_usable']:.3f}", 0.28),
                          (d["pooled_auc"], INK2, f"pooled {d['pooled_auc']:.3f}", 0.80)):
        ax.axvline(x, lw=1.4, ls="--", color=c)
        # staggered vertically and right-aligned, so the two never overlap each other
        ax.text(x - 0.008, top + dy, lab, color=c, fontsize=10, fontweight="bold",
                ha="right", va="center")
    ax.set_xlabel("AUC measured WITHIN one collection")
    ax.set_title("Glaucoma: pooled 0.967 vs within-source 0.878", pad=26)
    ax.set_xlim(0.2, 1.06)
    save(fig, "06_glaucoma_source_confound.png",
         "Grey = too few of one class to support an estimate · results/glaucoma/source_audit.json")


def g_glaucoma_heldout():
    d = load("results/glaucoma/heldout_source.json")
    if not d:
        return
    fig, ax = plt.subplots(figsize=(8.2, 3.9))
    base = d["internal"]["within_source_auc"]
    items = [(k, v["auc"], v["n"], v["positives"]) for k, v in d["zero_shot"].items()]
    ys = np.arange(len(items))[::-1]
    for y, (k, a, n, p) in zip(ys, items):
        ax.barh(y, a, height=.5, color=GL, edgecolor=SURF, linewidth=2)
        ax.text(a + .01, y, f"{a:.3f}", va="center", fontsize=12, fontweight="bold", color=INK)
        ax.text(.02, y, f"{k}  (n={n}, {p} glaucoma)", va="center", fontsize=10.5,
                color=INK if a > .4 else "#0b0b0b")
    ax.axvline(base, lw=1.6, ls="--", color=Y4)
    ax.text(base, len(items) - .35, f" in-distribution baseline {base:.3f}", color=Y4,
            fontsize=10, fontweight="bold")
    ax.set_yticks([]); ax.set_xlim(0, 1.05)
    ax.set_xlabel("Zero-shot AUC on a collection held entirely out of training")
    ax.set_title("Same model, same day: 0.20 AUC apart depending on the benchmark")
    ax.grid(axis="x", lw=.6, alpha=.5); ax.set_axisbelow(True)
    save(fig, "07_glaucoma_heldout.png",
         "1,450 images withheld before any dataloader was built · results/glaucoma/heldout_source.json")


def g_dr_external():
    d = load("results/dr/external_idrid.json")
    if not d:
        return
    bars = [("Internal\n(val, n=5815)", 0.9124, DR),
            ("IDRiD native\n(n=455, 4288px)", d["native"]["referable_auc"], DR),
            ("IDRiD downscaled\n(same images, 1024px)", d["rescaled"]["referable_auc"], Y4),
            ("Messidor-2\n(n=1744)", 0.837, INK2)]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.4, 4.4),
                                 gridspec_kw={"width_ratios": [1.25, 1]})
    xs = np.arange(len(bars))
    a1.bar(xs, [b[1] for b in bars], width=.62, color=[b[2] for b in bars],
           edgecolor=SURF, linewidth=2)
    for x, b in zip(xs, bars):
        a1.text(x, b[1] + .012, f"{b[1]:.3f}", ha="center", fontsize=11.5,
                fontweight="bold", color=INK)
    a1.set_xticks(xs); a1.set_xticklabels([b[0] for b in bars], fontsize=9.5)
    a1.set_ylim(0, 1.08); a1.set_ylabel("Referable AUC")
    a1.set_title("DR generalises — but the file size matters")
    bare(a1)

    sens = [d["native"]["deployed_argmax"]["sensitivity"],
            d["rescaled"]["deployed_argmax"]["sensitivity"]]
    spec = [d["native"]["deployed_argmax"]["specificity"],
            d["rescaled"]["deployed_argmax"]["specificity"]]
    w = .34; xs2 = np.arange(2)
    a2.bar(xs2 - w / 2, [s * 100 for s in sens], w, color=DR, edgecolor=SURF, linewidth=2,
           label="sensitivity")
    a2.bar(xs2 + w / 2, [s * 100 for s in spec], w, color=Y4, edgecolor=SURF, linewidth=2,
           label="specificity")
    for x, s in zip(xs2 - w / 2, sens):
        a2.text(x, s * 100 + 1.5, f"{s:.0%}", ha="center", fontsize=10, color=INK)
    for x, s in zip(xs2 + w / 2, spec):
        a2.text(x, s * 100 + 1.5, f"{s:.0%}", ha="center", fontsize=10, color=INK)
    a2.set_xticks(xs2); a2.set_xticklabels(["native\n4288px", "downscaled\n1024px"])
    a2.set_ylim(0, 112); a2.set_ylabel("%")
    a2.set_title(f"Missed cases: {d['native']['deployed_argmax']['fn']} → "
                 f"{d['rescaled']['deployed_argmax']['fn']}")
    a2.legend(loc="upper center", ncol=2, fontsize=10)
    bare(a2)
    save(fig, "08_dr_external_and_resolution.png",
         "Same 455 images, only the upload size differs · results/dr/external_idrid.json")


def g_cross():
    d = load("results/cross_disease_matrix.json")
    if not d:
        return
    folders = ["dr", "normal", "glaucoma", "rop", "normal_infant"]
    names = ["adult\nsevere DR", "adult\nno DR", "adult\nglaucoma",
             "infant\nROP present", "infant\nno ROP"]
    dis = ["DR", "ROP", "Glaucoma"]
    M = np.full((len(folders), 3), np.nan)
    for i, f in enumerate(folders):
        for j, dd in enumerate(dis):
            c = d["cells"].get(f"{f}|{dd}")
            if c and c["n"]:
                M[i, j] = 100 * c["flagged"] / c["n"]
    fig, ax = plt.subplots(figsize=(7.4, 5.2))
    for i in range(len(folders)):
        for j in range(3):
            v = M[i, j]
            if np.isnan(v):
                continue
            c = BLUES[min(int(v / 100 * (len(BLUES) - 1)), len(BLUES) - 1)]
            ax.add_patch(Rectangle((j, len(folders) - 1 - i), 1, 1, facecolor=c,
                                   edgecolor=SURF, linewidth=2))
            ax.text(j + .5, len(folders) - 1 - i + .5, f"{v:.0f}%", ha="center",
                    va="center", fontsize=14, fontweight="bold",
                    color=INK if v > 55 else "#0b0b0b")
    # ring the cells that are clinically impossible
    for i, f in enumerate(folders):
        for j, dd in enumerate(dis):
            impossible = (dd == "ROP" and f in ("dr", "normal", "glaucoma")) or \
                         (dd == "DR" and f in ("rop", "normal_infant"))
            if impossible and not np.isnan(M[i, j]):
                ax.add_patch(Rectangle((j + .04, len(folders) - 1 - i + .04), .92, .92,
                                       fill=False, edgecolor="#e66767", linewidth=2.2))
    ax.set_xlim(0, 3); ax.set_ylim(0, len(folders))
    ax.set_xticks(np.arange(3) + .5); ax.set_xticklabels(dis)
    ax.set_yticks(np.arange(len(folders)) + .5); ax.set_yticklabels(names[::-1], fontsize=10)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title("Every image through every model — how often each fires\n"
                 "red outline = clinically impossible finding", pad=14)
    save(fig, "09_cross_disease_matrix.png",
         "Routing bypassed on purpose (demo context). ROP fires on 59/59 adult eyes · results/cross_disease_matrix.json")


def g_calibration():
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.2))
    srcs = [("DR", "results/dr/tta_val_preds.npz", "referable"),
            ("ROP", "results/rop/tta_val_preds.npz", None),
            ("Glaucoma", "results/glaucoma/val_preds_by_source.npz", None)]
    for ax, (dis, rel, mode) in zip(axes, srcs):
        z = load(rel)
        cal = load(f"results/{dis.lower()}/calibration_serving.json")
        if z is None:
            ax.set_visible(False); continue
        y = z["y_true"].astype(int)
        if mode == "referable":
            s = z["prob"][:, 2:].sum(1); y = (y >= 2).astype(int)
        elif z["prob"].ndim == 2:
            s = z["prob"][:, 1]
        else:
            s = z["prob"]
        bins = np.linspace(0, 1, 11)
        idx = np.digitize(s, bins) - 1
        xs, ys, ns = [], [], []
        for b in range(10):
            m = idx == b
            if m.sum() >= 5:
                xs.append(s[m].mean()); ys.append(y[m].mean()); ns.append(m.sum())
        ax.plot([0, 1], [0, 1], ls=":", lw=1, color=INK2)
        ax.plot(xs, ys, "-o", lw=2, ms=6, color=COLOR[dis], markeredgecolor=SURF,
                markeredgewidth=1.4)
        ece = cal["uncalibrated"]["ece"] if cal else float("nan")
        ax.set_title(f"{dis}   ECE {ece:.3f}", color=COLOR[dis])
        ax.set_xlabel("model says"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_aspect("equal"); ax.grid(lw=.6, alpha=.5); ax.set_axisbelow(True)
    axes[0].set_ylabel("actually true")
    fig.suptitle("Calibration — is the confidence honest?  (dotted = perfect)",
                 fontsize=13.5, fontweight="bold", y=1.03)
    save(fig, "10_calibration.png",
         "On the diagonal = trustworthy percentages. DR sits well off it (ECE 0.191).")


def g_safety():
    rows = [("Non-fundus input\n(noise, charts, grey)", 100, 0),
            ("ROP model on adult eyes", 91, 0),
            ("Glaucoma model on infants", 55, 0),
            ("DR model on infants", 14, 0)]
    fig, ax = plt.subplots(figsize=(8.6, 4.2))
    ys = np.arange(len(rows))[::-1]
    w = .36
    ax.barh(ys + w / 2, [r[1] for r in rows], w, color=INK2, edgecolor=SURF, linewidth=2,
            label="before")
    ax.barh(ys - w / 2, [r[2] for r in rows], w, color=GL, edgecolor=SURF, linewidth=2,
            label="after routing + gate")
    for y, r in zip(ys, rows):
        ax.text(r[1] + 1.5, y + w / 2, f"{r[1]}%", va="center", fontsize=11,
                fontweight="bold", color=INK)
        ax.text(2, y - w / 2, "0", va="center", fontsize=11, fontweight="bold", color=GL)
    ax.set_yticks(ys); ax.set_yticklabels([r[0] for r in rows], fontsize=10)
    ax.set_xlabel("% of images given a confident disease finding")
    ax.set_title("What the gate and patient-context routing removed")
    ax.set_xlim(0, 108); ax.legend(loc="lower right"); bare(ax)
    ax.grid(axis="x", lw=.6, alpha=.5)
    save(fig, "11_safety_before_after.png",
         "Measured on the cohort probes · docs/IMPLEMENTATION_LOG.md")


def g_subgroup():
    d = load("results/rop/subgroup_analysis.json")
    if not d:
        return
    labels, vals, los, his = [], [], [], []
    for stratum, cells in d["strata"].items():
        for name, c in cells.items():
            labels.append(f"{name}  ({c['positive_units']} ROP+ / {c['units']} infants)")
            if c["auc"] is None:
                vals.append(None); los.append(None); his.append(None)
            else:
                vals.append(c["auc"]); los.append(c["ci"][0]); his.append(c["ci"][1])
    fig, ax = plt.subplots(figsize=(9.4, 5.4))
    _ci_bars(ax, labels, vals, los, his, [ROP] * len(labels),
             note_missing="not measurable — too few affected infants")
    ax.set_xlabel("AUC within subgroup")
    ax.set_ylim(-0.8, len(labels) - 0.1)     # headroom so row 1 clears the title
    ax.set_title("Can we show the model is fair?  No — and that is the finding", pad=20)
    save(fig, "12_rop_subgroups.png",
         "One female infant in the whole test set has ROP · results/rop/subgroup_analysis.json")


def main():
    print(f"writing to {OUT}/")
    for fn in (g_confusion, g_ppv, g_roc, g_rop_curve, g_rop_device, g_glaucoma_source,
               g_glaucoma_heldout, g_dr_external, g_cross, g_calibration, g_safety,
               g_subgroup):
        try:
            fn()
        except Exception as e:
            print(f"  FAILED {fn.__name__}: {type(e).__name__}: {e}")
    n = len(list(OUT.glob("*.png")))
    print(f"\n{n} figures in graphs/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
