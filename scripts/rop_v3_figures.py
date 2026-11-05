"""Figures for the BE project report, from results/rop/v3/report_data.json and v3_scores.json only.

    .venv/bin/python scripts/rop_v3_figures.py           # -> docs/report/figures/*.pdf (+ .png)

No model runs; every number is read from the committed report data (scripts/rop_v3_report_data.py) and the
scores file (LOSO folds and the Amendment 5c selection). Arms A-G are the v3 round, H and I the v4 round.
Paper 2's figures (IEEE column widths) are drawn by scripts/paper2_figures.py, which reuses this module.
"""
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs/report/figures"
R = json.loads((ROOT / "results/rop/v3/report_data.json").read_text())
SCO = json.loads((ROOT / "results/rop/v3/v3_scores.json").read_text())
CL = R["classes"]
CLN = {"normal": "Normal", "stage_1": "Stage 1", "stage_2": "Stage 2", "stage_3": "Stage 3", "stage_4_5": "Stage 4/5", "arop": "AP-ROP"}
ARMS = "ABCDEFGHI"
NAMES = {"A": "A RETFound-Green", "B": "B DINOv2", "C": "C EfficientNet", "D": "D DINOv2+exam loss",
         "E": "E Ensemble", "F": "F DINOv2+adapt.", "G": "G DINOv2 518px", "H": "H DINOv2, v4 pool",
         "I": "I DINOv2, v4 pool+recipe"}
COL = {"A": "#1b6ca8", "B": "#c2410c", "C": "#4b5563", "D": "#0f766e", "E": "#7c3aed", "F": "#b45309", "G": "#be185d",
       "H": "#0891b2", "I": "#4338ca"}  # H, I as in scripts/rop_v4_model_report.py
SRC = {"ostrava": "Ostrava", "ropvl": "ROP-VL", "shenzhen": "Shenzhen"}
LOSO_ARMS = "BHI"
plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
                     "legend.fontsize": 8, "figure.dpi": 150, "savefig.bbox": "tight", "axes.spines.top": False,
                     "axes.spines.right": False})


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.pdf")
    fig.savefig(OUT / f"{name}.png", dpi=200)
    plt.close(fig)
    print("  ", name)


def fig_dataset():
    pool = R["dataset"]["pool"]["by_source"]
    srcs = ["ostrava", "ropvl", "shenzhen"]
    labels = {"ostrava": "Ostrava\n(148 infants)", "ropvl": "ROP-VL\n(983 infants)", "shenzhen": "Shenzhen\n(397 infants)"}
    cls = ["normal", "stage_1", "stage_2", "stage_3", "stage_4_5", "arop"]
    cmap = ["#9ca3af", "#fde68a", "#fbbf24", "#f59e0b", "#b45309", "#7f1d1d"]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8), gridspec_kw={"width_ratios": [1.6, 1]})
    ax = axes[0]
    left = np.zeros(len(srcs))
    for c, col in zip(cls, cmap):
        v = np.array([pool[s]["by_class"][c] for s in srcs])
        ax.barh([labels[s] for s in srcs], v, left=left, color=col, label=CLN[c], edgecolor="white", linewidth=0.5)
        left += v
    for i, s in enumerate(srcs):
        ax.text(left[i] + 40, i, f"{pool[s]['images']:,}", va="center", fontsize=8)
    ax.set_xlabel("Photographs in the cross-validation pool")
    ax.set_xlim(0, left.max() * 1.15)
    ax.invert_yaxis()
    ax.legend(ncol=3, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.28), fontsize=7.5)
    ax.set_title("(a) Training pool (6,700 photographs)")
    ax = axes[1]
    dev = R["dataset"]["dev"]
    x = np.arange(6)
    ax.bar(x - 0.2, [dev["images_by_class"][c] for c in cls], 0.4, color="#9ca3af", label="photographs")
    ax.bar(x + 0.2, [dev["eyes_by_class"][c] for c in cls], 0.4, color="#1b6ca8", label="eyes")
    ax.set_xticks(x, [CLN[c].replace("Stage ", "S").replace("AP-ROP", "AP") for c in cls], fontsize=7.5)
    ax.set_title("(b) Held-out hospital, dev half")
    ax.legend(frameon=False)
    save(fig, "fig_dataset")


def fig_pipeline():
    fig, ax = plt.subplots(figsize=(7.2, 2.4))
    ax.set_xlim(0, 10.4); ax.set_ylim(0, 3); ax.axis("off")
    boxes = [("All photographs\nof one eye", "#e5e7eb"), ("Fundus pre-\nprocessing: circle-\ncrop, letterbox,\nCLAHE, 392 px", "#dbeafe"),
             ("DINOv2 ViT-S/14\n(model B): mean of\nfive fold models,\n6-way, per photo", "#fed7aa"), ("Worst-view rule:\nmax stage over\nthe eye's\nphotographs", "#d1fae5"),
             ("Eye-level stage,\ndriving photo-\ngraph, PDF\nreport", "#e5e7eb")]
    w, gap = 1.8, 0.35
    for k, (text, col) in enumerate(boxes):
        x = 0.1 + k * (w + gap)
        ax.add_patch(FancyBboxPatch((x, 0.85), w, 1.4, boxstyle="round,pad=0.04", fc=col, ec="#374151", lw=0.8))
        ax.text(x + w / 2, 1.55, text, ha="center", va="center", fontsize=7)
        if k:
            ax.annotate("", xy=(x - 0.03, 1.55), xytext=(x - gap + 0.03, 1.55), arrowprops=dict(arrowstyle="->", lw=0.9, color="#374151"))
    ax.text(5.2, 0.35, "Training: patient-grouped 5-fold CV on three hospitals (6,700 photographs), two seeds, last-epoch weights, "
                       "pre-registered protocol", ha="center", fontsize=7, color="#374151")
    ax.text(5.2, 2.7, "Inference path of the RetinAI eye-level screening pipeline", ha="center", fontsize=9, weight="bold")
    save(fig, "fig_pipeline")


def fig_compare(size=(7.2, 3.5)):
    arms = [a for a in ARMS if a in R["arms"]]
    fig, axes = plt.subplots(1, 2, figsize=size, sharey=True)
    for ax, lv, title in zip(axes, ("cv_image", "dev_eye"), ("(a) Cross-validation, per photograph (selection)", "(b) Held-out hospital, per eye")):
        for i, a in enumerate(arms):
            m = R["arms"][a][lv]
            lo, hi = m["ci95"]["qwk_5class"]
            ax.plot([lo, hi], [i, i], color=COL[a], lw=1.8)
            ax.plot(m["qwk_5class"], i, "o", color=COL[a], ms=6, mec="white")
            ax.text(hi + 0.006, i, f"{m['qwk_5class']:.3f}", va="center", fontsize=7.5)
        base = R["arms"]["C"][lv]["ci95"]["qwk_5class"]
        ax.axvspan(base[0], base[1], color="#9ca3af", alpha=0.15, lw=0)
        if lv == "cv_image":   # Amendment 5c: H or I adopted only at >= B + margin
            bar = SCO["selection_rule"]["table"]["B"]["qwk_5class"] + SCO["v4_selection"]["margin_cv"]
            ax.axvline(bar, color="#111827", lw=0.9, ls="--")
            ax.text(bar + 0.004, len(arms) - 0.05, f"adoption bar B + {SCO['v4_selection']['margin_cv']:.3f}", fontsize=6.5, va="center")
        ax.set_yticks(range(len(arms)), [NAMES[a] for a in arms])
        if ax is axes[0]:
            ax.set_ylim(len(arms) + 0.25, -0.6)   # shared y, inverted once (arm A on top); room for the bar's label
        ax.set_xlabel("QWK (5-class) with 95% infant-clustered CI")
        ax.set_title(title)
        ax.grid(axis="x", lw=0.4, alpha=0.5)
    save(fig, "fig_compare")


def fig_stage_recall(size=(7.2, 3.0)):
    arms = [a for a in ARMS if a in R["arms"]]
    fig, axes = plt.subplots(1, 2, figsize=size, sharey=True)
    for ax, lv, title in zip(axes, ("cv_image", "dev_eye"), ("(a) Cross-validation, per photograph", "(b) Held-out hospital, per eye")):
        x = np.arange(6)
        w = 0.8 / len(arms)
        for k, a in enumerate(arms):
            cls = R["arms"][a][lv]["classes"]
            v = [cls[c]["recall"] if cls[c]["recall"] is not None else 0 for c in CL]
            ax.bar(x + (k - len(arms) / 2 + 0.5) * w, v, w, color=COL[a], label=NAMES[a] if lv == "cv_image" else None)
        n = [R["arms"]["B"][lv]["classes"][c]["support"] for c in CL]
        ax.set_xticks(x, [f"{CLN[c].replace('Stage ', 'S')}\nn={v}" for c, v in zip(CL, n)], fontsize=7.5)
        ax.set_ylim(0, 1); ax.set_ylabel("Recall"); ax.set_title(title)
    axes[0].legend(ncol=5, frameon=False, fontsize=7, loc="upper center", bbox_to_anchor=(1.05, -0.22))
    save(fig, "fig_stage_recall")


def fig_confusion(arm="B"):
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.3), gridspec_kw={"wspace": 0.35})
    for ax, lv, title in zip(axes, ("cv_image", "dev_eye"), ("(a) Cross-validation, per photograph", "(b) Held-out hospital, per eye")):
        cm = np.array(R["arms"][arm][lv]["confusion"]["counts"])
        share = np.array(R["arms"][arm][lv]["confusion"]["row_normalised"])
        ax.imshow(share, cmap="Oranges", vmin=0, vmax=1)
        for i in range(6):
            for j in range(6):
                if cm[i].sum() == 0:
                    continue
                v = cm[i, j]
                txt = f"{v:.0f}" if abs(v - round(v)) < 1e-6 else f"{v:.1f}"
                ax.text(j, i, f"{txt}\n{share[i, j]:.0%}", ha="center", va="center", fontsize=6.5,
                        color="white" if share[i, j] > 0.55 else "black")
        lab = [CLN[c].replace("Stage ", "S") for c in CL]
        ax.set_xticks(range(6), lab, fontsize=7.5); ax.set_yticks(range(6), lab, fontsize=7.5)
        ax.set_xlabel("Predicted stage"); ax.set_title(title)
        if lv == "cv_image":
            ax.set_ylabel("True stage")
        ax.spines[:].set_visible(False)
    fig.suptitle(f"Confusion matrices of model {NAMES[arm]} (counts, mean over checkpoints; row share)", fontsize=9)
    save(fig, f"fig_confusion_{arm}")


def fig_curves():
    arms = [a for a in ARMS if a in R["arms"]]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.1))
    for a in arms:
        c = R["arms"][a]["curves"]["dev_eye_any_rop"]
        lw, al = (2.2, 1.0) if a == "B" else (1.1, 0.55)
        axes[0].plot(c["roc"]["fpr"], c["roc"]["tpr"], color=COL[a], lw=lw, alpha=al, label=f"{NAMES[a]} ({R['arms'][a]['dev_eye']['any_rop_auc']:.3f})")
        axes[1].plot(c["pr"]["recall"], c["pr"]["precision"], color=COL[a], lw=lw, alpha=al, label=f"{NAMES[a]} ({R['arms'][a]['dev_eye']['any_rop_ap']:.3f})")
    axes[0].plot([0, 1], [0, 1], "--", color="#9ca3af", lw=0.8)
    axes[0].set_xlabel("False-positive rate (1 − specificity)"); axes[0].set_ylabel("Sensitivity"); axes[0].set_title("(a) ROC, any ROP vs normal, per eye")
    axes[1].set_xlabel("Recall"); axes[1].set_ylabel("Precision"); axes[1].set_title("(b) Precision–recall, per eye"); axes[1].set_ylim(0.6, 1.02)
    axes[0].legend(frameon=False, fontsize=6.5, title="model (AUC)", title_fontsize=7); axes[1].legend(frameon=False, fontsize=6.5, title="model (AP)", title_fontsize=7, loc="lower left")
    save(fig, "fig_curves")


def fig_calibration(arms=("B", "C", "I")):
    fig, axes = plt.subplots(1, len(arms), figsize=(7.2, 2.7), sharey=True)
    for ax, a in zip(axes, arms):
        rel = R["arms"][a]["reliability_dev_eye"]
        ax.plot([0, 1], [0, 1], "--", color="#9ca3af", lw=0.8)
        for key, lab, col, ls in (("uncalibrated", "as trained", COL[a], "-"), ("cv_temperature", "after CV temperature", "#6b7280", "--")):
            b = rel[key]["bins"]
            ax.plot([x["confidence"] for x in b], [x["accuracy"] for x in b], ls, color=col, marker="o", ms=4, lw=1.4,
                    label=f"{lab} (ECE {rel[key]['ece']:.3f})")
        ax.set_xlabel("Model confidence"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_title(f"{NAMES[a]}", fontsize=9); ax.legend(frameon=False, fontsize=6.5, loc="upper left")
    axes[0].set_ylabel("Share correct (held-out hospital, per eye)")
    save(fig, "fig_calibration")


def fig_training():
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8))
    for a in ("A", "B", "D", "F", "G", "H", "I"):
        if "training" not in R["arms"][a]:
            continue
        cv = R["arms"][a]["training"]["curves"]
        ep = np.arange(1, len(cv["train_loss"]["mean"]) + 1)
        axes[0].plot(ep, cv["train_loss"]["mean"], color=COL[a], lw=1.4, label=NAMES[a])
        axes[0].fill_between(ep, cv["train_loss"]["min"], cv["train_loss"]["max"], color=COL[a], alpha=0.12, lw=0)
        axes[1].plot(ep, cv["val_macro_f1"]["mean"], color=COL[a], lw=1.4)
        axes[1].fill_between(ep, cv["val_macro_f1"]["min"], cv["val_macro_f1"]["max"], color=COL[a], alpha=0.12, lw=0)
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Training loss"); axes[0].set_title("(a) Training loss (mean, range over 10 runs)")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Held-out fold macro-F1"); axes[1].set_title("(b) Held-out macro-F1 during training")
    axes[0].legend(frameon=False, fontsize=7, ncol=4, loc="upper center", bbox_to_anchor=(1.1, -0.2))
    save(fig, "fig_training")


def fig_folds():
    fig, ax = plt.subplots(figsize=(7.2, 2.6))
    arms = [a for a in ARMS if a in R["arms"] and R["arms"][a]["per_fold"]]
    w = 0.8 / len(arms)
    for k, a in enumerate(arms):
        pf = R["arms"][a]["per_fold"]
        for f in range(5):
            v = [p["qwk_5class"] for p in pf if p["fold"] == f]
            x = f + (k - len(arms) / 2 + 0.5) * w
            ax.bar(x, np.mean(v), w, color=COL[a], label=NAMES[a] if f == 0 else None)
            ax.plot([x] * len(v), v, "k.", ms=3)
    ax.set_xticks(range(5), [f"Fold {f}" for f in range(5)]); ax.set_ylabel("QWK (5-class), per photograph"); ax.set_ylim(0.4, 1)
    ax.set_title("Cross-validation score on each fold (bar: mean of two seeds; dots: seeds)")
    ax.legend(ncol=5, frameon=False, fontsize=6.5, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    save(fig, "fig_folds")


def fig_units(size=(7.2, 2.9)):
    arms = [a for a in ARMS if a in R["arms"]]
    srcs = [("ostrava", "Ostrava · per examination session"), ("ropvl", "ROP-VL · per eye examination"), ("shenzhen", "Shenzhen · per photograph")]
    fig, axes = plt.subplots(1, 3, figsize=size, sharey=False)
    for ax, (s, title) in zip(axes, srcs):
        for i, a in enumerate(arms):
            u = R["arms"][a]["cv_units_by_source"][s]
            ax.plot(u["ci95"]["qwk_5class"], [i, i], color=COL[a], lw=1.6)
            ax.plot(u["qwk_5class"], i, "o", color=COL[a], ms=5, mec="white")
        ax.set_yticks(range(len(arms)), [a for a in arms]); ax.invert_yaxis(); ax.set_title(title, fontsize=8.5)
        ax.set_xlabel("QWK (5-class)"); ax.grid(axis="x", lw=0.4, alpha=0.5)
    save(fig, "fig_units")


def fig_contrasts(size=(7.2, 3.4)):
    fig, axes = plt.subplots(1, 2, figsize=size, sharey=True)
    for ax, blk, title in zip(axes, ("cv_image", "dev_eye"), ("(a) Cross-validation, per photograph", "(b) Held-out hospital, per eye")):
        pairs = list(R["contrasts"][blk].items())
        for i, (p, m) in enumerate(pairs):
            q = m["qwk_5class"]
            sig = q["diff_ci"][0] > 0 or q["diff_ci"][1] < 0
            col = "#111827" if sig else "#9ca3af"
            ax.plot(q["diff_ci"], [i, i], color=col, lw=1.8)
            ax.plot(q["diff"], i, "o", color=col, ms=5)
            ax.text(max(q["diff_ci"][1], 0) + 0.004, i, f"{q['diff']:+.3f}", va="center", fontsize=7)
        ax.axvline(0, color="#374151", lw=0.8, ls="--")
        ax.set_yticks(range(len(pairs)), [p.replace("-", " − ") + (" †" if m.get("status") else "") for p, m in pairs])
        if ax is axes[0]:
            ax.invert_yaxis()
        ax.set_xlabel("Paired difference in QWK with 95% CI"); ax.set_title(title); ax.grid(axis="x", lw=0.4, alpha=0.5)
    fig.text(0.5, -0.06, "† model F saw the held-out hospital's photographs (without labels) during training; "
                         "H − B, I − B and I − H were added by Amendment 5", ha="center", fontsize=7)
    save(fig, "fig_contrasts")


def fig_loso(arms=LOSO_ARMS):
    """Leave-one-source-out folds of the v4 round (v3_scores.json -> loso), QWK5 and macro-F1 with 95% CI."""
    srcs = SCO["loso"]["B"]["sources"]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8), sharey=True)
    for ax, key, xl in zip(axes, ("qwk_5class", "macro_f1_present"), ("QWK (5-class) with 95% CI", "Macro-F1 over present classes with 95% CI")):
        for j, s in enumerate(srcs):
            for k, a in enumerate(arms):
                L = SCO["loso"][a][s]
                y = j + (k - (len(arms) - 1) / 2) * 0.24
                ax.plot(L["ci95"][key], [y, y], color=COL[a], lw=1.8, solid_capstyle="round")
                ax.plot(L[key], y, "o", color=COL[a], ms=5, mec="white", mew=0.7,
                        label=f"{a}: {SCO['loso'][a]['pool']} pool{' + v4 recipe' if a == 'I' else ''}, mean QWK {SCO['loso'][a]['mean_qwk_5class']:.3f}"
                        if j == 0 and key == "qwk_5class" else None)
        L0 = SCO["loso"]["B"]
        ax.set_yticks(range(len(srcs)), [f"{SRC[s]} held out\n{L0[s]['n_images']:,} photos, {L0[s]['n_infants']} infants" for s in srcs], fontsize=7.5)
        ax.set_ylim(len(srcs) - 0.5, -0.5)
        ax.set_xlabel(xl); ax.grid(axis="x", lw=0.4, alpha=0.5)
    axes[0].legend(frameon=False, fontsize=7, loc="upper center", bbox_to_anchor=(1.05, -0.2), ncol=3)
    save(fig, "fig_loso")


def main():
    print("->", OUT.relative_to(ROOT))
    fig_dataset(); fig_pipeline(); fig_compare(); fig_stage_recall(); fig_confusion("B"); fig_confusion("C"); fig_confusion("I")
    fig_curves(); fig_calibration(); fig_training(); fig_folds(); fig_units(); fig_contrasts(); fig_loso()


if __name__ == "__main__":
    sys.exit(main())
