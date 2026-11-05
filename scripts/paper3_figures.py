"""Paper 3 (results compendium): every figure, every table body and every number in the prose, read from files.

    .venv/bin/python scripts/paper3_figures.py           # docs/paper/figs3/{fig_*.pdf,.png, tab_*.tex, numbers.tex}
    .venv/bin/python scripts/paper3_figures.py --docx    # also docs/paper/RetinAI_Paper3_Results_Compendium.docx

Inputs (nothing is typed by hand): results/rop/v3/{report_data,v3_scores}.json (arms A-I), the v4 pool summary,
the arm configs, docs/paper/PROTOCOL_V2.md, and for the earlier generations the files named in
docs/paper/MODEL_HISTORY.md (FINAL_MODEL_EVIDENCE_PACKAGE CSV/MD, results/sweep_rop.md, results/rop/metrics.json,
junk/unified_model/evaluation_unified.txt). No model is run; nothing named test_locked or locked_eval is read.
The LaTeX source docs/paper/paper3_results.tex \\input{}s the tab_*.tex files and numbers.tex written here.
"""
import argparse
import csv
import json
import re
import shutil
import subprocess
import sys
import tempfile
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from scripts import rop_v3_figures as V3F  # noqa: E402  palette + matplotlib rcParams of papers 1-2 and the v4 report

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

PAPER = ROOT / "docs/paper"
FIG = PAPER / "figs3"
TEX = PAPER / "paper3_results.tex"
PDF_OUT = PAPER / "RetinAI_Paper3_Results_Compendium.pdf"
DOCX_OUT = PAPER / "RetinAI_Paper3_Results_Compendium.docx"
PKG = "FINAL_MODEL_EVIDENCE_PACKAGE"
SRC_FILES = {
    "report_data": "results/rop/v3/report_data.json",
    "scores": "results/rop/v3/v3_scores.json",
    "v4_summary": "results/prepared/rop_staging/v4/v4_summary.json",
    "protocol": "docs/paper/PROTOCOL_V2.md",
    "history": "docs/paper/MODEL_HISTORY.md",
    "v1": "junk/unified_model/evaluation_unified.txt",
    "sweep": "results/sweep_rop.md",
    "resnet_metrics": "results/rop/metrics.json",
    "v2_qwk": f"{PKG}/20_AUDIT/verification_pass_2/06_VALIDATION/qwk_5class_vs_6class_recalculation.csv",
    "v2_cv": f"{PKG}/07_TEST_RESULTS/metrics_summary.csv",
    "v2_eye_ci": f"{PKG}/08_EYE_LEVEL_ANALYSIS/eye_confidence_intervals.csv",
    "v2_test_md": f"{PKG}/07_TEST_RESULTS/test_summary.md",
    "v2_card": f"{PKG}/00_START_HERE/MODEL_CARD.md",
    "five_backbone": f"{PKG}/19_COMPARISONS/retired_models/five_backbone_benchmark.md",
    "sampler": f"{PKG}/19_COMPARISONS/retired_models/weighted_sampler_v1_arm.md",
    "screener": f"{PKG}/19_COMPARISONS/retired_models/resnet50_binary_screener.md",
    "legacy_qwk": f"{PKG}/19_COMPARISONS/retired_models/legacy_qwk_values.csv",
}
CFG_FILES = {"A": "configs/rop_staging_rfg.yaml", "B": "configs/rop_staging_dinov2.yaml",
             "C": "configs/rop_staging.yaml", "D": "configs/rop_staging_dinov2_bag.yaml",
             "F": "configs/rop_staging_dinov2_uda.yaml", "G": "configs/rop_staging_dinov2_518.yaml",
             "H": "configs/rop_staging_dinov2_v4data.yaml", "I": "configs/rop_staging_dinov2_v4recipe.yaml"}
C_KERNEL = "kaggle_kernels/rop_staging_benchmark.py"
SCRIPTS = {"scorer": "scripts/rop_v3_score.py", "report_data": "scripts/rop_v3_report_data.py",
           "v4_report": "scripts/rop_v4_model_report.py", "this": "scripts/paper3_figures.py"}


def _guard(p):
    if "test_locked" in str(p) or "locked_eval" in str(p):
        raise RuntimeError(f"refusing to read {p}")
    return ROOT / p


def jload(rel):
    return json.loads(_guard(rel).read_text())


def text(rel):
    return _guard(rel).read_text()


def csv_rows(rel):
    with open(_guard(rel), newline="") as fh:
        return list(csv.DictReader(fh))


def md_tables(rel):
    """Every pipe table in a Markdown file as (header, rows); ** and backticks stripped."""
    out, cur = [], []
    for ln in text(rel).splitlines() + [""]:
        if ln.strip().startswith("|"):
            cur.append([c.strip().replace("**", "").replace("`", "") for c in ln.strip().strip("|").split("|")])
        elif cur:
            body = [r for r in cur[1:] if not all(set(c) <= set(":- ") for c in r)]
            out.append((cur[0], body))
            cur = []
    return out


def md_table_with(rel, first_header):
    return next(t for t in md_tables(rel) if t[0][0].lower().startswith(first_header.lower()))


R = jload(SRC_FILES["report_data"])
SCO = jload(SRC_FILES["scores"])
V4 = jload(SRC_FILES["v4_summary"])
ARMS = "ABCDEFGHI"
TRAINED = [a for a in ARMS if "training" in R["arms"][a]]
CL = R["classes"]
CLN = V3F.CLN
SHORT = {c: CLN[c].replace("Stage ", "S").replace("AP-ROP", "AP") for c in CL}
COL = {**V3F.COL, "H": "#0891b2", "I": "#4338ca"}  # as scripts/rop_v4_model_report.py
MARK = {"A": "o", "B": "s", "C": "D", "D": "^", "E": "v", "F": "P", "G": "X", "H": "<", "I": ">"}  # secondary encoding
NAMES = {**V3F.NAMES, "H": "H DINOv2, v4 pool", "I": "I DINOv2, v4 pool + recipe"}
SRC = {"ostrava": "Ostrava", "ropvl": "ROP-VL", "shenzhen": "Shenzhen"}
SEEDS = [str(s) for s in R["arms"]["B"]["seeds"]]
TRANS = [a for a in ARMS if R["arms"][a]["dev_status"].startswith("transductive")]
GREY, INK = "#9ca3af", "#111827"
MAC = {}
FIGLOG = []


# ------------------------------------------------------------------ formatting (LaTeX)
def rnd(x, d=3):
    """Round half-up on the decimal text of the stored value (the files hold 4-dp numbers; binary floats would
    round 0.0185 down)."""
    return float(Decimal(repr(float(x))).quantize(Decimal(1).scaleb(-d), rounding=ROUND_HALF_UP))


def f3(x, d=3):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "--"
    s = f"{rnd(x, d):.{d}f}"
    return s.replace("-", "$-$")


def s3(x, d=3):
    if x is None:
        return "--"
    v = rnd(x, d)
    return f"+{v:.{d}f}" if v >= 0 else f"$-${abs(v):.{d}f}"


def p3(x, d=3, sign=False):
    """Plain-text (matplotlib) number with half-up rounding and a true minus sign."""
    v = rnd(x, d)
    return (f"{v:+.{d}f}" if sign else f"{v:.{d}f}").replace("-", "−")


def ci(c, d=3, signed=False):
    if not c:
        return "--"
    f = s3 if signed else f3
    return f"[{f(c[0], d)}, {f(c[1], d)}]"


def n0(x):
    return f"{int(round(x)):,}".replace(",", "{,}")


def esc(s):
    """Plain text from a source file -> LaTeX: specials escaped, Unicode symbols mapped, "quoted" -> ``quoted''."""
    s = str(s)
    for a, b in (("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"), ("_", r"\_"), ("#", r"\#"),
                 ("±", r"$\pm$"), ("×", r"$\times$"), ("→", r"$\to$"), ("≥", r"$\ge$"), ("−", "$-$"),
                 ("–", "--"), ("—", "---"), ("≈", r"$\approx$")):
        s = s.replace(a, b)
    return re.sub(r'"([^"]*)"', r"``\1''", s)


def mac(name, value):
    assert re.fullmatch(r"[A-Za-z]+", name), name
    MAC[name] = value
    return value


def tabular(name, spec, header, rows):
    """Write docs/paper/figs3/tab_<name>.tex: a booktabs tabular. rows: list of cell lists, or raw LaTeX lines."""
    out = [rf"\begin{{tabular}}{{{spec}}}", r"\toprule", *header, r"\midrule"]
    for r in rows:
        out.append(r if isinstance(r, str) else " & ".join(r) + r" \\")
    out += [r"\bottomrule", r"\end{tabular}", ""]
    FIG.mkdir(parents=True, exist_ok=True)
    (FIG / f"tab_{name}.tex").write_text("\n".join(out))
    print("   tab_" + name)


def save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"{name}.pdf", metadata={"CreationDate": None, "Creator": None, "Producer": None})
    fig.savefig(FIG / f"{name}.png", dpi=220, metadata={"Software": None})
    plt.close(fig)
    FIGLOG.append(name)
    print("  ", name)


def git_last(rel, *more):
    out = subprocess.run(["git", "log", "-1", "--format=%h\t%ad", "--date=short", "--", rel, *more], cwd=ROOT,
                         capture_output=True, text=True).stdout.strip()
    if out:
        return out.split("\t")
    ign = subprocess.run(["git", "check-ignore", "-q", rel], cwd=ROOT).returncode == 0
    return ["git-ignored, on disk" if ign else "not yet committed", ""]


def git_first(rel):
    out = subprocess.run(["git", "log", "--diff-filter=A", "--format=%h\t%ad", "--date=short", "--", rel], cwd=ROOT,
                         capture_output=True, text=True).stdout.strip().splitlines()
    return out[-1].split("\t") if out else ["untracked", ""]


def arm(a):
    return R["arms"][a]


def lab(a):
    """Arm letter for tables: bold for the selected arm, dagger for transductive."""
    s = rf"\textbf{{{a}}}" if a == SCO["v4_selection"]["chosen"] else a
    return s + (r"$^\dagger$" if a in TRANS else "")


# ------------------------------------------------------------------ numbers used in prose and captions
def numbers():
    B, ds = arm("B"), R["dataset"]
    sel = SCO["v4_selection"]["chosen"]
    mac("selArm", sel)
    mac("BcvQ", f3(SCO["selection_rule"]["table"][sel]["qwk_5class"]))
    mac("BcvQci", ci(SCO["selection_rule"]["table"][sel]["ci95"]))
    mac("BdevQ", f3(arm(sel)["dev_eye"]["qwk_5class"]))
    mac("BdevQci", ci(arm(sel)["dev_eye"]["ci95"]["qwk_5class"]))
    mac("BlosoQ", f3(SCO["loso"][sel]["mean_qwk_5class"]))
    mac("nArms", str(len(ARMS)))
    mac("nTrained", str(len(TRAINED)))
    mac("poolImages", n0(ds["pool"]["images"]))
    mac("poolInfants", n0(ds["pool"]["infants"]))
    mac("poolSources", str(len(ds["pool"]["by_source"])))
    mac("dupExcl", str(ds["pool"]["duplicates_excluded"]))
    mac("cvScored", n0(SCO["contrasts"]["cv_image"]["n_images"]))
    mac("devImages", n0(ds["dev"]["images"]))
    mac("devEyes", str(ds["dev"]["eyes"]))
    mac("devInfants", str(ds["dev"]["infants"]))
    mac("lockedImages", n0(ds["locked"]["images"]))
    mac("lockedEyes", str(ds["locked"]["eyes"]))
    mac("nReps", n0(SCO["bootstrap"]["replicates"]))
    mac("bootSeed", str(SCO["bootstrap"]["seed"]))
    mac("cvLists", str(B["cv_image"]["n_lists"]))
    mac("devLists", str(B["dev_eye"]["n_lists"]))
    mac("seedA", SEEDS[0])
    mac("seedB", SEEDS[1])
    mac("marginCV", f"{SCO['v4_selection']['margin_cv']:.3f}")
    mac("marginLOSO", f"{SCO['v4_selection']['margin_loso']:.3f}")
    proto = text(SRC_FILES["protocol"])
    amend = re.findall(r"^## Amendment (\d+) — (\d{4}-\d{2}-\d{2})", proto, flags=re.M)
    mac("nAmend", str(len(amend)))
    mac("lastAmendDate", amend[-1][1])
    h, d = git_first(SRC_FILES["protocol"])
    mac("protoCommit", h)
    mac("protoDate", d)
    hist = text(SRC_FILES["history"])
    m = re.search(r"Totals: (\d+) distinct architectures, ~(\d+) trained arms", hist)
    mac("nArch", m.group(1))
    mac("nArmsAll", m.group(2))
    mac("transArms", ", ".join(TRANS))
    mac("nPairs", str(len(R["contrasts"]["cv_image"])))
    mac("nRuns", str(arm("B")["training"]["curves"]["train_loss"]["n_runs"]))
    mac("shortcutAUC", re.search(r"predicted ROP with AUC ([\d.]+)", hist).group(1))
    mac("lockedOpened", re.search(r"Sealed half \(opened once, (\d{4}-\d{2}-\d{2})\)", hist).group(1))
    mac("nSevereFrom", "3")  # severe = label >= 3 (stage 3, 4/5, AP-ROP): scripts/rop_v3_score.py op_report
    assert "sev = label >= 3" in text(SCRIPTS["scorer"])
    mac("gpu", sorted({r["gpu"] for a in TRAINED for rows in arm(a)["training"]["timing"].values() for r in rows})[0])


# ------------------------------------------------------------------ figures
def forest(ax, arms, getter, ref, dagger=()):
    vals = [getter(a) for a in arms]
    for i, (a, (q, lo, hi)) in enumerate(zip(arms, vals)):
        ax.plot([lo, hi], [i, i], color=COL[a], lw=2.4 if a == ref else 1.5, solid_capstyle="round")
        ax.plot(q, i, MARK[a], color=COL[a], ms=6 if a == ref else 4.8, mec="white", mew=0.6)
    xmax = max(v[2] for v in vals)
    for i, (a, (q, lo, hi)) in enumerate(zip(arms, vals)):
        ax.text(xmax + 0.01, i, p3(q), va="center", fontsize=6.8, weight="bold" if a == ref else "normal")
    ax.set_xlim(min(v[1] for v in vals) - 0.01, xmax + 0.05)
    ax.set_yticks(range(len(arms)), [NAMES[a] + (" †" if a in dagger else "") for a in arms], fontsize=7)
    for t, a in zip(ax.get_yticklabels(), arms):
        t.set_weight("bold" if a == ref else "normal")
    ax.set_ylim(len(arms) - 0.4, -1.2)
    ax.grid(axis="x", lw=0.4, alpha=0.5)
    ax.tick_params(axis="x", labelsize=7)


def fig_selection():
    tab, sel = SCO["selection_rule"]["table"], SCO["v4_selection"]
    ref, m = sel["chosen"], sel["margin_cv"]
    fig, axes = plt.subplots(2, 1, figsize=(3.5, 3.9), gridspec_kw={"height_ratios": [3.2, 1.25], "hspace": 0.62})
    ax = axes[0]
    forest(ax, list(ARMS), lambda a: (tab[a]["qwk_5class"], *tab[a]["ci95"]), ref)
    b = tab[ref]["qwk_5class"]
    ax.axvline(b, color=COL[ref], lw=0.9, ls=":")
    ax.axvline(b + m, color=INK, lw=0.9, ls="--")
    ax.text(b + m + 0.004, -0.85, f"adoption bar {p3(b + m)}", fontsize=6.3, va="center")
    ax.set_xlabel("CV QWK5 per photograph, pooled OOF (95% CI)", fontsize=7.5)
    ax.set_title("(a) Selection metric, all arms", fontsize=8)
    ax = axes[1]
    lo = sel["loso"]
    arms = list(lo)
    for i, a in enumerate(arms):
        ax.plot(lo[a], i, MARK[a], color=COL[a], ms=6, mec="white", mew=0.6)
        ax.text(lo[a] + 0.0012, i, p3(lo[a], 4), va="center", fontsize=6.8)
    floor = lo[ref] - sel["margin_loso"]
    ax.axvline(floor, color=INK, lw=0.9, ls="--")
    ax.axvline(lo[ref], color=COL[ref], lw=0.9, ls=":")
    ax.text(floor + 0.0008, -0.75, f"floor {p3(floor, 4)}", fontsize=6.3, va="center")
    ax.set_yticks(range(len(arms)), [NAMES[a] for a in arms], fontsize=7)
    ax.set_ylim(len(arms) - 0.5, -1.1)
    ax.set_xlim(floor - 0.004, max(lo.values()) + 0.012)
    ax.set_xlabel("Mean LOSO QWK5 over the three held-out sources", fontsize=7.5)
    ax.set_title("(b) Second condition (v4 arms vs B)", fontsize=8)
    ax.grid(axis="x", lw=0.4, alpha=0.5)
    ax.tick_params(axis="x", labelsize=7)
    save(fig, "fig_selection")


def fig_dev():
    fig, ax = plt.subplots(figsize=(3.5, 2.75))
    ref = SCO["v4_selection"]["chosen"]
    forest(ax, list(ARMS), lambda a: (arm(a)["dev_eye"]["qwk_5class"], *arm(a)["dev_eye"]["ci95"]["qwk_5class"]),
           ref, dagger=TRANS)
    ax.axvline(arm(ref)["dev_eye"]["qwk_5class"], color=COL[ref], lw=0.9, ls=":")
    ax.set_xlabel("Unseen hospital (dev half), eye-level QWK5 (95% CI)", fontsize=7.5)
    save(fig, "fig_dev_eye")


PAIRS = list(R["contrasts"]["cv_image"])


def fig_contrasts():
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 3.0), sharey=True, gridspec_kw={"wspace": 0.08})
    allv = [v for blk in ("cv_image", "dev_eye") for p in PAIRS for v in R["contrasts"][blk][p]["qwk_5class"]["diff_ci"]]
    lim = max(abs(v) for v in allv) + 0.03
    for ax, blk, title in zip(axes, ("cv_image", "dev_eye"),
                              ("(a) CV, per photograph", "(b) Unseen hospital (dev half), per eye")):
        for i, p in enumerate(PAIRS):
            q = R["contrasts"][blk][p]["qwk_5class"]
            ax.plot(q["diff_ci"], [i, i], color=COL[p[0]], lw=1.8, solid_capstyle="round")
            ax.plot(q["diff"], i, MARK[p[0]], color=COL[p[0]], ms=5, mec="white", mew=0.6)
            ax.text(lim - 0.003, i, p3(q["diff"], sign=True), ha="right", va="center", fontsize=6.5)
        ax.axvline(0, color="#374151", lw=0.8)
        ax.set_xlim(-lim, lim)
        ax.set_yticks(range(len(PAIRS)), [p.replace("-", " − ") for p in PAIRS], fontsize=7.5)
        ax.set_ylim(len(PAIRS) - 0.5, -0.6)
        ax.set_xlabel("Paired difference in QWK5 (95% CI)", fontsize=8)
        ax.set_title(title, fontsize=8.5)
        ax.grid(axis="x", lw=0.4, alpha=0.5)
    save(fig, "fig_contrasts")


def fig_stage_recall():
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.75), sharey=True, gridspec_kw={"wspace": 0.06})
    off = np.linspace(-0.34, 0.34, len(ARMS))
    for ax, lv, title, unit in zip(axes, ("cv_image", "dev_eye"),
                                   ("(a) CV, per photograph", "(b) Unseen hospital (dev half), per eye"),
                                   ("photos", "eyes")):
        for k, a in enumerate(ARMS):
            cls = arm(a)[lv]["classes"]
            v = [cls[c]["recall"] for c in CL]
            ax.plot(np.arange(len(CL)) + off[k], v, MARK[a], color=COL[a], ms=4.2, mec="white", mew=0.4,
                    label=NAMES[a], ls="none")
        for i in range(1, len(CL)):
            ax.axvline(i - 0.5, color="#e5e7eb", lw=0.6)
        n = [arm("B")[lv]["classes"][c]["support"] for c in CL]
        ax.set_xticks(range(len(CL)), [f"{SHORT[c]}\nn={v:,}" for c, v in zip(CL, n)], fontsize=7)
        ax.set_xlim(-0.5, len(CL) - 0.5)
        ax.set_ylim(0, 1.03)
        ax.set_title(title, fontsize=8.5)
        ax.grid(axis="y", lw=0.4, alpha=0.5)
    axes[0].set_ylabel("Recall")
    axes[0].legend(ncol=5, frameon=False, fontsize=6.6, loc="upper center", bbox_to_anchor=(1.02, -0.24),
                   handletextpad=0.2, columnspacing=1.0)
    save(fig, "fig_stage_recall")


def fig_confusion():
    fig, axes = plt.subplots(3, 3, figsize=(7.16, 6.0), gridspec_kw={"wspace": 0.04, "hspace": 0.3})
    labs = [SHORT[c] for c in CL]
    xl = ["N" if c == "normal" else SHORT[c] for c in CL]
    for ax, a in zip(axes.flat, ARMS):
        share = np.array(arm(a)["dev_eye"]["confusion"]["row_normalised"])
        sup = [arm(a)["dev_eye"]["classes"][c]["support"] for c in CL]
        ax.imshow(share, cmap="Blues", vmin=0, vmax=1)
        for i in range(6):
            for j in range(6):
                v = share[i, j]
                if v >= 0.005:
                    ax.text(j, i, f"{100 * v:.0f}", ha="center", va="center", fontsize=6.3,
                            color="white" if v > 0.55 else INK)
        col = axes.flat[list(ARMS).index(a)]
        r, c = divmod(list(ARMS).index(a), 3)
        ax.set_xticks(range(6), xl, fontsize=6.3)
        ax.set_yticks(range(6), [f"{lb} ({n})" for lb, n in zip(labs, sup)] if c == 0 else [], fontsize=6.3)
        if r == 2:
            ax.set_xlabel("Predicted", fontsize=7)
        if c == 0:
            ax.set_ylabel("True (eyes)", fontsize=7)
        ax.set_title(NAMES[a] + (" †" if a in TRANS else ""), fontsize=7.8, color=INK)
        ax.spines[:].set_visible(False)
        ax.tick_params(length=0)
        del col
    save(fig, "fig_confusion_dev_eye")


def fig_roc_pr():
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 3.0), gridspec_kw={"wspace": 0.25})
    ref = SCO["v4_selection"]["chosen"]
    for a in ARMS:
        c = arm(a)["curves"]["dev_eye_any_rop"]
        lw = 1.8 if a == ref else 1.0
        axes[0].plot(c["roc"]["fpr"], c["roc"]["tpr"], color=COL[a], lw=lw, marker=MARK[a], markevery=0.25, ms=3.5,
                     label=f"{NAMES[a]}: AUC {p3(c['roc']['auc'])}, AP {p3(c['pr']['ap'])}")
        axes[1].plot(c["pr"]["recall"], c["pr"]["precision"], color=COL[a], lw=lw, marker=MARK[a], markevery=0.25,
                     ms=3.5)
    prev = arm(ref)["curves"]["dev_eye_any_rop"]["pr"]["prevalence"]
    axes[0].plot([0, 1], [0, 1], "--", color=GREY, lw=0.8)
    axes[1].axhline(prev, ls=":", color=GREY, lw=0.9)
    axes[1].text(0.02, prev + 0.006, f"prevalence {p3(prev)}", fontsize=6.5, color=INK, va="bottom")
    axes[0].set_xlabel("1 − specificity")
    axes[0].set_ylabel("Sensitivity")
    axes[0].set_title("(a) ROC, any ROP vs normal, per eye", fontsize=8.5)
    axes[1].set_xlabel("Recall (sensitivity)")
    axes[1].set_ylabel("Precision")
    axes[1].set_title("(b) Precision-recall, per eye", fontsize=8.5)
    axes[1].set_ylim(prev - 0.06, 1.005)
    axes[0].legend(frameon=False, fontsize=6.2, loc="upper center", bbox_to_anchor=(1.12, -0.2), ncol=3,
                   columnspacing=1.0, handlelength=1.8)
    save(fig, "fig_roc_pr_dev_eye")


def fig_training():
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.7), gridspec_kw={"wspace": 0.22})
    for a in TRAINED:
        cur = arm(a)["training"]["curves"]
        for ax, key in zip(axes, ("train_loss", "val_macro_f1")):
            y = cur[key]["mean"]
            ax.plot(np.arange(1, len(y) + 1), y, color=COL[a], lw=1.8 if a == "B" else 1.1, marker=MARK[a],
                    markevery=4, ms=3.6, label=f"{NAMES[a]} (n={cur[key]['n_runs']})")
    axes[0].set_ylabel("Training loss (mean over runs)")
    axes[1].set_ylabel("Validation-fold macro-F1 (mean)")
    for ax, t in zip(axes, ("(a) Training loss", "(b) Held-out fold macro-F1 per epoch")):
        ax.set_xlabel("Epoch")
        ax.set_title(t, fontsize=8.5)
        ax.grid(lw=0.3, alpha=0.5)
    axes[0].legend(frameon=False, fontsize=6.3, loc="upper center", bbox_to_anchor=(1.1, -0.2), ncol=4)
    save(fig, "fig_training_curves")


def fig_loso():
    lo = SCO["loso"]
    srcs = ["shenzhen", "ostrava", "ropvl"]
    fig, ax = plt.subplots(figsize=(3.5, 2.55))
    x0, w, ticks = 0.0, 0.17, []
    for s in srcs:
        arms = [a for a in ARMS if a in lo and s in lo[a]]
        xs = x0 + (np.arange(len(arms)) - (len(arms) - 1) / 2) * w
        for xx, a in zip(xs, arms):
            L = lo[a][s]
            ax.bar(xx, L["qwk_5class"], w * 0.86, color=COL[a], label=NAMES[a] if s == "shenzhen" else None)
            ax.plot([xx, xx], L["ci95"]["qwk_5class"], color=INK, lw=0.8)
            ax.text(xx, 0.03, a, ha="center", va="bottom", fontsize=6.3, color="white", weight="bold")
        n = lo["B"][s]
        ticks.append((x0, f"{SRC[s]} held out\n{n['n_images']:,} photos,\n{n['n_infants']} infants"))
        x0 += 1.0
    ax.set_xticks([t[0] for t in ticks], [t[1] for t in ticks], fontsize=6.6)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("QWK5 per photograph (95% CI)", fontsize=7.5)
    ax.tick_params(axis="y", labelsize=7)
    ax.grid(axis="y", lw=0.4, alpha=0.5)
    ax.legend(frameon=False, fontsize=6.0, loc="upper center", bbox_to_anchor=(0.5, 1.22), ncol=3, columnspacing=0.8)
    save(fig, "fig_loso")


def fig_reliability():
    fig, axes = plt.subplots(2, 5, figsize=(7.16, 3.35), gridspec_kw={"wspace": 0.12, "hspace": 0.42})
    for ax, a in zip(axes.flat, ARMS):
        rel = arm(a)["reliability_dev_eye"]
        ax.plot([0, 1], [0, 1], "--", color=GREY, lw=0.7)
        for key, col, ls, mk in (("uncalibrated", COL[a], "-", MARK[a]), ("cv_temperature", "#6b7280", "--", ".")):
            b = [x for x in rel[key]["bins"] if x["n"]]
            ax.plot([x["confidence"] for x in b], [x["accuracy"] for x in b], ls, color=col, marker=mk, ms=3, lw=1.1)
        ax.text(0.04, 0.95, f"ECE {p3(rel['uncalibrated']['ece'])}\nwith T {p3(rel['cv_temperature']['ece'])}",
                fontsize=5.8, va="top", color=INK)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        k = list(ARMS).index(a)
        ax.set_xticks([0, 0.5, 1], ["0", ".5", "1"] if k >= 4 else [], fontsize=6)
        ax.set_yticks([0, 0.5, 1], ["0", ".5", "1"] if k % 5 == 0 else [], fontsize=6)
        ax.set_title(NAMES[a].replace("pool + recipe", "pool+recipe") + (" †" if a in TRANS else ""), fontsize=6.2)
    leg = axes.flat[-1]
    leg.axis("off")
    leg.legend(handles=[Line2D([], [], color=INK, lw=1.1, marker="o", ms=3, label="as trained"),
                        Line2D([], [], color="#6b7280", lw=1.1, ls="--", marker=".", label="after CV temperature T"),
                        Line2D([], [], color=GREY, lw=0.7, ls="--", label="perfect calibration")],
               loc="center", frameon=False, fontsize=6.3)
    fig.text(0.5, 0.0, "Confidence (top-class probability)", ha="center", fontsize=7.5)
    fig.text(0.075, 0.5, "Share correct", va="center", rotation=90, fontsize=7.5)
    save(fig, "fig_reliability_dev_eye")


# ------------------------------------------------------------------ tables: current generation
def tab_headline():
    rows = []
    for a in ARMS:
        cv, dv = arm(a)["cv_image"], arm(a)["dev_eye"]
        rows.append([lab(a), esc(R["arm_names"][a]),
                     f3(cv["qwk_5class"]) + " " + ci(cv["ci95"]["qwk_5class"]), f3(cv["accuracy"]),
                     f3(cv["balanced_accuracy"]), f3(cv["macro_f1"]), f3(cv["any_rop_auc"]),
                     f3(dv["qwk_5class"]) + " " + ci(dv["ci95"]["qwk_5class"]), f3(dv["accuracy"]),
                     f3(dv["macro_f1"]), f3(dv["any_rop_auc"]), f3(dv["any_rop_ap"])])
    tabular("headline", "@{}ll@{\\hspace{4pt}}lcccc@{\\hspace{6pt}}lcccc@{}",
            [r" & & \multicolumn{5}{c}{CV, per photograph (" + MAC["cvScored"] + r" photos)} & "
             r"\multicolumn{5}{c}{Unseen hospital, dev half, per eye (" + MAC["devEyes"] + r" eyes)}\\",
             r"\cmidrule(lr){3-7}\cmidrule(l){8-12}",
             r"Arm & Model & QWK5 [95\% CI] & Acc. & Bal.\,acc. & mF1 & AUC & QWK5 [95\% CI] & Acc. & mF1 & AUC & AP\\"],
            rows)


def tab_stage():
    rows = []
    for lv, title, unit in (("cv_image", "CV, per photograph", "photos"),
                            ("dev_eye", "Unseen hospital (dev half), per eye", "eyes")):
        sup = [arm("B")[lv]["classes"][c]["support"] for c in CL]
        rows.append(r"\multicolumn{7}{@{}l}{\emph{" + title + "; n " + unit + ": " +
                    ", ".join(f"{SHORT[c]} {n0(n)}" for c, n in zip(CL, sup)) + r"}}\\")
        for a in ARMS:
            cls = arm(a)[lv]["classes"]
            rows.append([lab(a)] + [f"{f3(cls[c]['recall'])} / {f3(cls[c]['precision'])}" for c in CL])
        if lv == "cv_image":
            rows.append(r"\midrule")
    tabular("stage", "@{}lcccccc@{}",
            ["Arm & " + " & ".join(CLN[c] for c in CL) + r"\\"], rows)


def tab_folds():
    rows = []
    for a in ARMS:
        for s in SEEDS:
            q = [p["qwk_5class"] for p in sorted(arm(a)["per_fold"], key=lambda p: p["fold"]) if str(p["seed"]) == s]
            rows.append([lab(a) if s == SEEDS[0] else "", s] + [f3(v) for v in q] +
                        [f3(float(np.mean(q))), f3(float(np.std(q, ddof=1)))])
    tabular("folds", "@{}ll@{\\hspace{4pt}}ccccc@{\\hspace{4pt}}cc@{}",
            [r"Arm & Seed & F0 & F1 & F2 & F3 & F4 & Mean & SD\\"], rows)


def tab_loso():
    lo = SCO["loso"]
    rows = []
    for s in ["shenzhen", "ostrava", "ropvl"]:
        arms = [a for a in ARMS if a in lo and s in lo[a]]
        n = lo[arms[0]][s]
        rows.append(r"\multicolumn{10}{@{}l}{\emph{" + SRC[s] + " held out (" + n["fold_file"].replace("_", r"\_") +
                    f"): {n0(n['n_images'])} photos, {n['n_infants']} infants, {n['n_excluded_duplicates']} duplicates excluded" +
                    r"}}\\")
        for a in arms:
            L = lo[a][s]
            rec = L["per_class_recall_mean"]
            rows.append([lab(a), lo[a]["pool"], f3(L["qwk_5class"]) + " " + ci(L["ci95"]["qwk_5class"]),
                         f3(L["macro_f1_present"]) + " " + ci(L["ci95"]["macro_f1_present"]),
                         f3(L["accuracy"]) + " " + ci(L["ci95"]["accuracy"])] +
                        [f3(rec[c], 2) if c in rec else "--" for c in CL])
        rows.append(r"\midrule")
    rows.append(r"\multicolumn{10}{@{}l}{\emph{Mean QWK5 over the three held-out sources (Amendment 5c): " +
                ", ".join(f"{a} {f3(lo[a]['mean_qwk_5class'])}" for a in ARMS if a in lo and lo[a]["all_three_sources"]) +
                r"}}\\")
    tabular("loso", "@{}ll@{\\hspace{4pt}}lll@{\\hspace{4pt}}cccccc@{}",
            [r" & & & & & \multicolumn{6}{c}{Mean recall per class}\\", r"\cmidrule(l){6-11}",
             r"Arm & Pool & QWK5 [95\% CI] & mF1 [95\% CI] & Acc. [95\% CI] & " + " & ".join(SHORT[c] for c in CL) + r"\\"],
            rows)


DESC = {"A-C": "retinal vs CNN (primary)", "A-B": "retinal vs natural", "B-C": "natural vs CNN",
        "E-C": "ensemble vs CNN", "E-A": "ensemble vs retinal", "D-A": "exam loss vs retinal",
        "D-B": "exam loss vs per photo", "F-B": "adaptation", "G-B": "518 vs 392 px", "H-B": "v4 pool",
        "I-B": "v4 pool + recipe", "I-H": "v4 recipe"}


def tab_contrasts():
    rows = []
    for p in PAIRS:
        cv, dv = R["contrasts"]["cv_image"][p], R["contrasts"]["dev_eye"][p]
        rows.append([p.replace("-", r"$-$"), DESC.get(p, "")] +
                    [s3(b[m]["diff"]) + " " + ci(b[m]["diff_ci"], signed=True)
                     for b in (cv, dv) for m in ("qwk_5class", "macro_f1_present")])
    tabular("contrasts", "@{}llllll@{}",
            [r" & & \multicolumn{2}{c}{CV, per photograph} & \multicolumn{2}{c}{Dev half, per eye}\\",
             r"\cmidrule(lr){3-4}\cmidrule(l){5-6}",
             r"Pair & Contrast & $\Delta$QWK5 [95\% CI] & $\Delta$mF1 [95\% CI] & $\Delta$QWK5 [95\% CI] & $\Delta$mF1 [95\% CI]\\"],
            rows)


def tab_operating():
    rows = []
    for a in ARMS:
        rel = arm(a)["reliability_dev_eye"]
        for s in SEEDS:
            op, T = arm(a)["operating_point"][s], arm(a)["temperature"][s]
            cv, dv = op["cv"], op["dev_eye_mean"]
            first = s == SEEDS[0]
            rows.append([lab(a) if first else "", s, f"{op['threshold']:.4f}",
                         f3(cv["any_rop"]["sensitivity"]), f3(cv["any_rop"]["specificity"]),
                         f3(dv["any_rop"]), f3(dv["any_rop_spec"]),
                         f3(cv["severe_vs_all_nonsevere"]["sensitivity"]), f3(cv["severe_vs_all_nonsevere"]["specificity"]),
                         f3(dv["severe_vs_all_nonsevere"]), f3(dv["severe_vs_all_nonsevere_spec"]),
                         f3(T), f3(rel["uncalibrated"]["ece"]) if first else "",
                         f3(rel["cv_temperature"]["ece"]) if first else ""])
    tabular("operating", "@{}llc@{\\hspace{5pt}}cccc@{\\hspace{5pt}}cccc@{\\hspace{5pt}}ccc@{}",
            [r" & & & \multicolumn{4}{c}{Any ROP vs normal} & \multicolumn{4}{c}{Severe (S3+) vs rest} & & "
             r"\multicolumn{2}{c}{Dev-eye ECE}\\",
             r"\cmidrule(lr){4-7}\cmidrule(lr){8-11}\cmidrule(l){13-14}",
             r"Arm & Seed & Threshold & CV sens & CV spec & Dev sens & Dev spec & CV sens & CV spec & Dev sens & Dev spec & "
             r"$T$ & as trained & with $T$\\"], rows)


def c_recipe():
    """Arm C's recipe: configs/rop_staging.yaml as run by the kernel (EPOCHS, arch override), per PROTOCOL_V2 parity."""
    cfg = yaml.safe_load(text(CFG_FILES["C"]))
    k = text(C_KERNEL)
    ep = int(re.search(r"^EPOCHS = (\d+)", k, flags=re.M).group(1))
    arch = json.loads(text("results/rop/staging_runs/flat_natural_s42_a.json"))["arch"]
    return cfg, ep, arch


def timing(a):
    rows = [r for rs in arm(a)["training"]["timing"].values() for r in rs if r["fold"] in {f"fold{k}" for k in range(5)}]
    return float(np.mean([r["minutes"] for r in rows])), float(np.mean([r["img_per_s"] for r in rows])), len(rows)


def recipe_note(a, base):
    t = arm(a)["training"]["config"]
    tr, pre = t["train"], t["preprocess"]
    notes = []
    if a == "A":
        notes.append("RETFound-Green weights")
    if tr.get("bag_loss", {}).get("enabled"):
        bl = tr["bag_loss"]
        notes.append(f"LSE bag loss on {SRC[bl['source']]} sessions, $K{{=}}{bl['k']}$, $\\tau{{=}}{bl['tau']:g}$")
    if tr.get("uda", {}).get("enabled"):
        u = tr["uda"]
        notes.append(f"self-ensembling on unlabelled dev photos, EMA {u['ema_decay']:g}, conf.\\ $\\ge${u['conf_threshold']:g}")
    if pre["image_size"] != base["preprocess"]["image_size"]:
        notes.append(f"{pre['image_size']} px")
    if tr.get("ema_decay"):
        notes.append(f"EMA weights {tr['ema_decay']:g}")
    if tr.get("head_warmup_epochs"):
        notes.append(f"{tr['head_warmup_epochs']} head-only epoch")
    if tr.get("label_smoothing") != base["train"]["label_smoothing"]:
        notes.append(f"label smoothing {tr['label_smoothing']:g}")
    if tr.get("aug_v4", {}).get("enabled"):
        g = tr["aug_v4"]
        notes.append(f"rot.\\ $\\pm${g['rotation']}$^\\circ$, crop $\\ge${g['min_crop_scale']:g}, gamma, noise, "
                     f"Fourier swap $p\\,{g['fourier_p']:g}$")
    return notes


def tab_recipe():
    base = arm("B")["training"]["config"]
    pool = {a: SCO["loso"][a]["pool"] if a in SCO["loso"] else "v3" for a in ARMS}
    rows = []
    for a in ARMS:
        if a in TRAINED:
            t = arm(a)["training"]["config"]
            m, pre, tr = t["model"], t["preprocess"], t["train"]
            mins, ips, n = timing(a)
            notes = recipe_note(a, base)
            if a == "H":
                notes.append("v4 pool (per-device quality filter)")
            rows.append([lab(a), r"\texttt{" + esc(m["arch"]) + "}", str(pre["image_size"]), esc(pre["normalize"]), pool[a],
                         f"{tr['lr']:.0e}".replace("e-0", "e-"), f"{tr['layer_decay']:g}", f"{tr['label_smoothing']:g}",
                         str(tr["epochs"]), "; ".join(notes) if notes else "reference recipe",
                         f"{mins:.1f}", f"{ips:.1f}"])
        elif a == "C":
            cfg, ep, archc = c_recipe()
            rows.append([lab(a), r"\texttt{" + esc(archc) + "}", str(cfg["preprocess"]["image_size"]), "imagenet", pool[a],
                         f"{cfg['train']['lr']:.0e}".replace("e-0", "e-"), "--", f"{cfg['train']['label_smoothing']:g}",
                         str(ep), "v2 flat softmax, natural sampler; saved predictions re-scored", "--", "--"])
        else:
            rows.append([lab(a), "--", "--", "--", pool[a], "--", "--", "--", "--",
                         "mean of A, B, C probabilities, matched by seed; no training", "--", "--"])
    tabular("recipe", "@{}llcllcccc>{\\raggedright\\arraybackslash}p{6.2cm}cc@{}",
            [r"Arm & Backbone & px & Norm. & Pool & Peak lr & LLRD & LS & Ep. & Differences from B's recipe & "
             r"min/fold & img/s\\"], rows)


def tab_secondary():
    rows = []
    for a in ARMS:
        p = SCO["arms"][a]["pooled_over_seeds"]
        u, dec = p["cv_per_source_unit"], p["decoded"]
        rows.append([lab(a), f3(p["dev_image"]["qwk_5class"]) + " " + ci(p["dev_image"]["ci95"]["qwk_5class"])] +
                    [f3(u[s]["qwk_5class"]) for s in ("ostrava", "ropvl", "shenzhen")] +
                    [f3(dec["cv_image"]["qwk_5class"]), f3(dec["dev_eye"]["qwk_5class"])])
    tabular("secondary", "@{}ll@{\\hspace{6pt}}ccc@{\\hspace{6pt}}cc@{}",
            [r" & Dev half & \multicolumn{3}{c}{CV, each cohort at its label unit} & \multicolumn{2}{c}{Ordinal decoder}\\",
             r"\cmidrule(lr){3-5}\cmidrule(l){6-7}",
             r"Arm & per photo QWK5 [95\% CI] & Ostrava exam & ROP-VL eye-exam & Shenzhen photo & CV & Dev eye\\"],
            rows)


# ------------------------------------------------------------------ tables: earlier generations
def tab_v1():
    t = text(SRC_FILES["v1"])
    rows = []
    for split in ("VAL", "TEST"):
        blk = t.split(f"===== ROP {split} RESULTS =====")[1].split("=====")[0]
        acc = float(re.search(r"Accuracy: ([\d.]+)%", blk).group(1)) / 100
        cls = {m.group(1).strip(): [float(m.group(i)) for i in (2, 3, 4)] + [int(m.group(5))]
               for m in re.finditer(r"^\s*(No ROP|ROP Detected|macro avg)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+(\d+)", blk, re.M)}
        rows.append([f"ROP {split.lower()}", n0(cls["macro avg"][3]), f3(acc), f3(cls["macro avg"][2]),
                     f"{f3(cls['No ROP'][0])} / {f3(cls['No ROP'][1])}", f"{f3(cls['ROP Detected'][0])} / {f3(cls['ROP Detected'][1])}",
                     f3(cls["ROP Detected"][2])])
    dr = t.split("===== DR RESULTS")[1]
    dacc = float(re.search(r"Val Accuracy: ([\d.]+)%", dr).group(1)) / 100
    dmf = float(re.search(r"Macro F1:\s+([\d.]+)%", dr).group(1)) / 100
    ep = re.search(r"Epoch (\d+)", t.split("===== DR RESULTS")[1].splitlines()[0]).group(1)
    rows.append([f"DR val, ep.\\ {ep}", "--", f3(dacc), f3(dmf), "--", "--", "--"])
    tabular("v1", "@{}>{\\raggedright\\arraybackslash}p{1.45cm}rccccc@{}",
            [r"Split & n & Acc. & mF1 & No ROP P / R & ROP P / R & ROP F1\\"], rows)


def tab_sweep():
    hdr, body = md_tables(SRC_FILES["sweep"])[0]
    ix = {h: i for i, h in enumerate(hdr)}
    rows = [[esc(r[ix["arch"]]), f3(float(r[ix["accuracy"]])), f3(float(r[ix["macro_f1"]])), f3(float(r[ix["qwk"]])),
             f3(float(r[ix["auc"]])), r[ix["params_M"]], r[ix["cpu_ms"]]] for r in body]
    t = jload(SRC_FILES["resnet_metrics"])
    for split in ("val", "test"):
        m = t[split]
        rows.append([r"resnet50$^\ast$ " + split, f3(m["accuracy"]), f3(m["macro_f1"]), f3(m["qwk"]),
                     f3(m["auc"]), "--", "--"])
    rows.insert(len(body), r"\midrule")
    tabular("sweep", "@{}lcccccc@{}", [r"Architecture & Acc. & mF1 & QWK$_2$ & AUC & Params (M) & CPU ms\\"], rows)


def v2_rows():
    q = [r for r in csv_rows(SRC_FILES["v2_qwk"])]
    cvm = {(r["arm"].split()[0].lower().replace("current", "structured"), r["arm"].split("s")[-1]): r
           for r in csv_rows(SRC_FILES["v2_cv"])}
    return q, cvm


def tab_v2():
    q, _ = v2_rows()
    ms = {}
    for r in csv_rows(SRC_FILES["v2_cv"]):
        head = "flat" if r["arm"].startswith("flat") else "structured"
        seed = re.search(r"s(\d+)", r["arm"]).group(1)
        ms[(head, seed)] = r
    get = {(r["dataset"], r["arm"], r["seed"], r["unit"]): r for r in q}
    rows = []
    for head in ("structured", "flat"):
        for seed in SEEDS:
            cv = get[("cv_pooled_oof", head, seed, "image")]
            dv = get[("multiview_dev", head, seed, "eye")]
            li = get[("multiview_locked", head, seed, "image")]
            le = get[("multiview_locked", head, seed, "eye")]
            m = ms[(head, seed)]
            rows.append([head.capitalize() if seed == SEEDS[0] else "", seed,
                         f3(float(cv["qwk_6class"])), f3(float(cv["qwk_5class_drop"])), f3(float(cv["qwk_5class_map"])),
                         f3(float(cv["accuracy"])), f3(float(cv["macro_f1_present"])), f3(float(m["pooled_any_rop_auc"])),
                         f3(float(dv["qwk_5class_map"])),
                         f3(float(li["qwk_5class_drop"])), f3(float(li["qwk_5class_map"])),
                         f3(float(le["qwk_5class_drop"])), f3(float(le["qwk_5class_map"])), f3(float(le["accuracy"]))])
    eci = {r["metric"]: r for r in csv_rows(SRC_FILES["v2_eye_ci"])}
    hdr, body = next(t for t in md_tables(SRC_FILES["v2_test_md"]) if any("served fold-0" in h for h in t[0]))
    served = {r[0].lower(): r[1] for r in body if len(r) > 1}
    qimg = next(v for k, v in served.items() if k.startswith("qwk (corrected"))
    acci = served["accuracy"]
    auci = next(v for k, v in served.items() if "any rop" in k)
    e = eci["qwk_ordinal_5class"]
    mac("vtServedEyeQ", f3(float(e["point"])))
    mac("vtServedEyeQci", ci([float(e["ci95_lo"]), float(e["ci95_hi"])]))
    mac("vtServedEyeAcc", f3(float(eci["accuracy"]["point"])))
    mac("vtServedEyeMF", f3(float(eci["macro_f1_present"]["point"])))
    mac("vtServedImgQ", f3(float(qimg)))
    mac("vtServedImgAcc", f3(float(acci)))
    mac("vtServedImgAUC", f3(float(auci)))
    mac("vtStructParams", re.search(r"Parameters \| ([\d,]*\d)", text(SRC_FILES["v2_card"])).group(1).replace(",", "{,}"))
    leg = [r for r in csv_rows(SRC_FILES["legacy_qwk"]) if r["scope"] == "CV mean over 5 folds"]
    mac("vtDropPairs", n0(int(leg[0]["n_pairs_used_after_excluding_AP_ROP"])))
    tabular("v2", "@{}ll@{\\hspace{4pt}}cccccc@{\\hspace{5pt}}c@{\\hspace{5pt}}cc@{\\hspace{5pt}}ccc@{}",
            [r" & & \multicolumn{6}{c}{CV pooled OOF, " + n0(int(get[('cv_pooled_oof', 'flat', SEEDS[0], 'image')]['n']))
             + r" photos} & Dev & \multicolumn{2}{c}{Locked, photo} & \multicolumn{3}{c}{Locked, eye (max)}\\",
             r"\cmidrule(lr){3-8}\cmidrule(lr){9-9}\cmidrule(lr){10-11}\cmidrule(l){12-14}",
             r"Head & Seed & QWK$_6$ & QWK5$_d$ & QWK5 & Acc. & mF1 & AUC & QWK5 & QWK5$_d$ & QWK5 & QWK5$_d$ & QWK5 & Acc.\\"],
            rows)


def tab_backbones():
    hdr, body = md_table_with(SRC_FILES["five_backbone"], "backbone")
    rows = [[esc(c) for c in r] for r in body]
    assert [h.lower() for h in hdr] == ["backbone", "params", "macro-f1 (fold 0)", "cross-site macro-f1", "ap-rop recall"], hdr
    tabular("backbones", "@{}lcccc@{}", [r"Backbone & Params & mF1, fold 0 & mF1, site fold & AP-ROP recall\\"], rows)


def tab_sampler():
    hdr, body = md_table_with(SRC_FILES["sampler"], "metric")
    rows = [[esc(c) for c in r] for r in body]
    tabular("sampler", "@{}>{\\raggedright\\arraybackslash}p{2.7cm}ccc@{}",
            [r"Fold 0 & Balanced sampler & Natural (v2) & Flat\\"], rows)


def tab_screener():
    hdr, body = md_tables(SRC_FILES["screener"])[0]
    rows = [[esc(r[0]), esc(r[1])] for r in body]
    cmp_ = next(t for t in md_tables(SRC_FILES["screener"]) if any("external dev AUC" in r[0] for r in t[1]))
    for r in cmp_[1]:
        if "AUC" in r[0]:
            rows.append([esc(r[0]) + " (retired / v2 structured)", f"{esc(r[1])} / {esc(r[2])}"])
    tabular("screener", "@{}>{\\raggedright\\arraybackslash}p{4.4cm}>{\\raggedright\\arraybackslash}p{3.9cm}@{}",
            [r"External audit & Value\\"], rows)


# ------------------------------------------------------------------ tables: data and reproducibility
def tab_pool():
    ds, tb = R["dataset"], V4["training_by_source"]
    rows = []
    srcs = ["ostrava", "ropvl", "shenzhen"]

    def line(label, unit, counts, total, infants):
        return [label, unit] + [n0(counts[c]) for c in CL] + [n0(total), infants]

    rows.append(r"\multicolumn{10}{@{}l}{\emph{Cross-validation pool (every photo is in exactly one validation fold)}}\\")
    for s in srcs:
        p = ds["pool"]["by_source"][s]
        rows.append(line(SRC[s], "photos", p["by_class"], p["images"], n0(p["infants"])))
    tot = {c: sum(ds["pool"]["by_source"][s]["by_class"][c] for s in srcs) for c in CL}
    rows.append(line("Total", "photos", tot, ds["pool"]["images"], n0(ds["pool"]["infants"])))
    for key, title in (("v3_kept", "Training-eligible after per-infant caps, v3 pool (arms A--G)"),
                       ("v4", "Training-eligible, v4 pool (arms H, I)")):
        rows.append(r"\midrule")
        rows.append(r"\multicolumn{10}{@{}l}{\emph{" + title + r"}}\\")
        for s in srcs:
            rows.append(line(SRC[s], "photos", tb[s][key], sum(tb[s][key].values()), ""))
        tc = V4["training_by_class"][key]
        rows.append(line("Total", "photos", tc, V4["training_rows"][key], ""))
    rows.append(r"\midrule")
    rows.append(r"\multicolumn{10}{@{}l}{\emph{Unseen hospital (Multi-View), never in any fold}}\\")
    dv = ds["dev"]
    rows.append(line("Dev half", "photos", dv["images_by_class"], dv["images"], str(dv["infants"])))
    rows.append(line("Dev half", "eyes", dv["eyes_by_class"], dv["eyes"], str(dv["infants"])))
    lk = ds["locked"]
    rows.append([r"Locked half$^\ast$", "photos"] + ["--"] * 6 + [n0(lk["images"]), str(lk["infants"])])
    rows.append([r"Locked half$^\ast$", "eyes"] + ["--"] * 6 + [n0(lk["eyes"]), str(lk["infants"])])
    tabular("pool", "@{}llrrrrrrrr@{}",
            ["Set & Unit & " + " & ".join(CLN[c] for c in CL) + r" & Total & Infants\\"], rows)


def tab_loso_folds():
    lo = SCO["loso"]
    rows = []
    for s in ["shenzhen", "ostrava", "ropvl"]:
        for pool, a in (("v3", "B"), ("v4", "H")):
            L = lo[a][s]
            ts = L["train_support"]
            rows.append([SRC[s] if pool == "v3" else "", L["fold_file"].replace("_", r"\_") if pool == "v3" else "",
                         n0(L["n_images"]) if pool == "v3" else "", str(L["n_infants"]) if pool == "v3" else "", pool] +
                        [n0(ts[c]) for c in CL] + [n0(sum(ts.values()))])
    tabular("loso_folds", "@{}llrrlrrrrrrr@{}",
            [r" & & & & & \multicolumn{7}{c}{Training photographs}\\", r"\cmidrule(l){6-12}",
             r"Held out & Fold file & Test photos & Infants & Pool & " + " & ".join(SHORT[c] for c in CL) + r" & Total\\"],
            rows)


def tab_repro():
    items = [("Protocol and amendments (first commit)", SRC_FILES["protocol"], git_first(SRC_FILES["protocol"])),
             ("Protocol, latest amendment", SRC_FILES["protocol"], git_last(SRC_FILES["protocol"])),
             ("Scores, arms A--I (selection, LOSO, contrasts)", SRC_FILES["scores"], git_last(SRC_FILES["scores"])),
             ("Per-arm report data (tables, curves)", SRC_FILES["report_data"], git_last(SRC_FILES["report_data"])),
             ("v4 pool summary", SRC_FILES["v4_summary"], git_last(SRC_FILES["v4_summary"])),
             ("Scorer", SCRIPTS["scorer"], git_last(SCRIPTS["scorer"])),
             ("Report-data builder", SCRIPTS["report_data"], git_last(SCRIPTS["report_data"])),
             ("v4 model report", SCRIPTS["v4_report"], git_last(SCRIPTS["v4_report"])),
             ("This paper's figures and tables", SCRIPTS["this"], git_last(SCRIPTS["this"])),
             ("Model chronology", SRC_FILES["history"], git_last(SRC_FILES["history"])),
             ("v2 QWK recalculation (3 definitions)", SRC_FILES["v2_qwk"], git_last(SRC_FILES["v2_qwk"])),
             ("v2 CV summary", SRC_FILES["v2_cv"], git_last(SRC_FILES["v2_cv"])),
             ("v2 locked, served checkpoint (eye CIs)", SRC_FILES["v2_eye_ci"], git_last(SRC_FILES["v2_eye_ci"])),
             ("Five-backbone sweep", SRC_FILES["five_backbone"], git_last(SRC_FILES["five_backbone"])),
             ("Weighted-sampler arm", SRC_FILES["sampler"], git_last(SRC_FILES["sampler"])),
             ("ResNet50 screener audit", SRC_FILES["screener"], git_last(SRC_FILES["screener"])),
             ("June 2026 binary sweep", SRC_FILES["sweep"], git_last(SRC_FILES["sweep"])),
             ("ResNet50 val/test metrics", SRC_FILES["resnet_metrics"], git_last(SRC_FILES["resnet_metrics"])),
             ("v1 EfficientNet-B3 evaluation", SRC_FILES["v1"], git_last(SRC_FILES["v1"])),
             ("Arm configs (A, B, D, F--I; C)", "configs/rop_staging_*.yaml", git_last(*CFG_FILES.values()))]
    rows = []
    for name, path, (h, d) in items:
        p = path.replace(PKG + "/", "PKG/")
        rows.append([esc(name), r"\texttt{" + esc(p).replace("/", r"/\allowbreak{}") + "}", (r"\texttt{" + h + "}" if re.fullmatch(r"[0-9a-f]{7,}", h) else h) +
                     (f" {d}" if d else "")])
    tabular("repro", "@{}>{\\raggedright\\arraybackslash}p{4.6cm}>{\\raggedright\\arraybackslash}p{9.6cm}l@{}",
            [r"Item & Path & Commit\\"], rows)


def write_numbers():
    lines = ["% generated by scripts/paper3_figures.py from the files it names; do not edit"]
    lines += [rf"\newcommand{{\{k}}}{{{v}}}" for k, v in MAC.items()]
    (FIG / "numbers.tex").write_text("\n".join(lines) + "\n")
    print("   numbers.tex", len(MAC), "macros")


# ------------------------------------------------------------------ DOCX (pandoc on a simplified copy)
def docx():
    src = TEX.read_text()

    def inline(m):
        return (PAPER / (m.group(1) + ("" if m.group(1).endswith(".tex") else ".tex"))).read_text()

    s = re.sub(r"\\input\{([^}]+)\}", inline, src)
    s = s.replace(r"\documentclass[conference]{IEEEtran}", r"\documentclass{article}")
    s = s.replace(r"\IEEEoverridecommandlockouts", "")
    s = re.sub(r"\\IEEEauthorblockN\{", "{", s)
    s = re.sub(r"\\IEEEauthorblockA\{", "{", s)
    s = re.sub(r"\\begin\{IEEEkeywords\}(.*?)\\end\{IEEEkeywords\}", r"\\noindent Keywords: \1", s, flags=re.S)
    s = s.replace("table*", "table").replace("figure*", "figure")
    s = re.sub(r"\\includegraphics\[[^]]*\]\{([^}]+)\.pdf\}", r"\\includegraphics[width=6in]{\1.png}", s)
    s = re.sub(r"\\(scriptsize|footnotesize|small)\b", "", s)
    s = re.sub(r"\\setlength\{\\tabcolsep\}\{[^}]*\}", "", s)
    s = re.sub(r"\\resizebox\{[^}]*\}\{!\}\{", "{", s)
    s = re.sub(r"@\{(?:[^{}]|\{[^{}]*\})*\}", "", s)                      # @{...} column glue: pandoc drops such tables
    s = s.replace(r">{\raggedright\arraybackslash}", "")
    keys = re.findall(r"\\bibitem\{([^}]+)\}", s)                         # numeric citations, as in the PDF
    num = {k: i + 1 for i, k in enumerate(keys)}
    s = re.sub(r"\\cite\{([^}]+)\}", lambda m: "[" + ", ".join(str(num[k.strip()]) for k in m.group(1).split(",")) + "]", s)
    s = re.sub(r"\\bibitem\{([^}]+)\}", lambda m: f"\\item[[{num[m.group(1)]}]]", s)
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td) / "paper3_simplified.tex"
        tmp.write_text(s)
        subprocess.run(["pandoc", str(tmp), "--from", "latex", "--to", "docx", "-o", str(DOCX_OUT),
                        f"--resource-path={FIG}:{PAPER}"], check=True)
    print("  ", DOCX_OUT.relative_to(ROOT))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--docx", action="store_true")
    args = ap.parse_args()
    numbers()
    for f in (fig_selection, fig_dev, fig_contrasts, fig_stage_recall, fig_confusion, fig_roc_pr, fig_training,
              fig_loso, fig_reliability):
        f()
    for t in (tab_headline, tab_stage, tab_folds, tab_loso, tab_contrasts, tab_operating, tab_recipe, tab_secondary,
              tab_v1, tab_sweep, tab_v2, tab_backbones, tab_sampler, tab_screener, tab_pool, tab_loso_folds, tab_repro):
        t()
    write_numbers()
    if args.docx:
        docx()


if __name__ == "__main__":
    main()
