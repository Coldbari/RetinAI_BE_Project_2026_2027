"""v4 model-development report: arms H and I against the selected arm B, every number read from a file.

    .venv/bin/python scripts/rop_v4_model_report.py            # docs/v4/figures/f*.{pdf,png} + docs/v4/V4_MODEL_REPORT.md
    .venv/bin/python scripts/rop_v4_model_report.py --render   # also V4_MODEL_REPORT.pdf (pandoc + tectonic) and .docx

Inputs, all committed: results/rop/v3/report_data.json and v3_scores.json (scores), the per-fold histories and
summaries under results/rop/v3/runs/ (training curves, kernel slugs), results/prepared/rop_staging/v4/
(v4_summary.json, the v4 fold files), results/prepared/rop_staging/{splits_summary.json, fold_*.csv},
results/rop/v4/audit/{audit_summary,thresholds}.json, the three arm configs (and the train transforms they build),
docs/paper/PROTOCOL_V2.md and docs/v4/README.md (quoted verbatim). No model is run and nothing named test_locked
is read. Output is deterministic: no timestamps in the figures or the Markdown; commit hashes are those of the
input files, not of HEAD.
"""
import argparse
import json
import math
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from scripts import rop_v3_figures as V3F  # noqa: E402  thesis palette + matplotlib style (sets rcParams)
from scripts import rop_v3_report_data as RD  # noqa: E402  run-dir names
from scripts import rop_v3_score as SC  # noqa: E402  run root, fold names, classes

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

DOC = ROOT / "docs/v4"
FIG = DOC / "figures"
MD = DOC / "V4_MODEL_REPORT.md"
FILES = {"report_data": "results/rop/v3/report_data.json", "scores": "results/rop/v3/v3_scores.json",
         "v4_summary": "results/prepared/rop_staging/v4/v4_summary.json",
         "splits_summary": "results/prepared/rop_staging/splits_summary.json",
         "audit": "results/rop/v4/audit/audit_summary.json", "thresholds": "results/rop/v4/audit/thresholds.json"}
CFG_FILES = {"B": "configs/rop_staging_dinov2.yaml", "H": "configs/rop_staging_dinov2_v4data.yaml",
             "I": "configs/rop_staging_dinov2_v4recipe.yaml"}
PROTOCOL = "docs/paper/PROTOCOL_V2.md"
README = "docs/v4/README.md"
RUN_LOG = "docs/v4/V4_RUN_LOG.md"
AUDIT_MD = "results/rop/v4/audit/AUDIT.md"


def _guard(p):
    if "test_locked" in str(p) or "locked_eval" in str(p):
        raise RuntimeError(f"refusing to read {p}")
    return Path(p)


def jload(rel):
    return json.loads(_guard(ROOT / rel).read_text())


R = jload(FILES["report_data"])
SCO = jload(FILES["scores"])
V4 = jload(FILES["v4_summary"])
SPL = jload(FILES["splits_summary"])
AUD = jload(FILES["audit"])
THR = jload(FILES["thresholds"])
CFG = {a: yaml.safe_load(_guard(ROOT / p).read_text()) for a, p in CFG_FILES.items()}

ARMS = "ABCDEFGHI"
NEW = "BHI"
CL = R["classes"]
CLN = V3F.CLN
SHORT = {c: CLN[c].replace("Stage ", "S").replace("AP-ROP", "AP") for c in CL}
COL = {**V3F.COL, "H": "#0891b2", "I": "#4338ca"}  # B/H/I pass the dataviz CVD validator as a triple
NAMES = {**V3F.NAMES, "H": "H DINOv2, v4 pool", "I": "I DINOv2, v4 pool + recipe"}
SRC = {"ostrava": "Ostrava", "ropvl": "ROP-VL", "shenzhen": "Shenzhen"}
DEVICE = {"clarity_retcam3": "Clarity RetCam3", "phoenix_icon": "Phoenix ICON", "retcam_envision": "RetCam Envision"}
SEEDS = R["arms"]["B"]["seeds"]
FOLDS = [f"fold_{k}" for k in range(SC.NFOLD)] + [SC.LOSO_FOLDS[s] for s in ("shenzhen", "ostrava", "ropvl")]
FOLD_LABEL = {**{f"fold_{k}": f"CV {k}" for k in range(SC.NFOLD)}, "fold_site": "LOSO Shenzhen (site)",
              "fold_loso_ostrava": "LOSO Ostrava", "fold_loso_ropvl": "LOSO ROP-VL"}
GREY, INK = "#9ca3af", "#111827"
MINUS = "−"
FIGS = {}


# ---------------------------------------------------------------- formatting
def f3(x):
    return "n/a" if x is None else f"{x:.3f}".replace("-", MINUS)


def s3(x):
    return "n/a" if x is None else f"{x:+.3f}".replace("-", MINUS)


def ci3(c):
    return "n/a" if not c else f"[{f3(c[0])}, {f3(c[1])}]"


def pct(x):
    return f"{100 * x:.1f}%"


def n0(x):
    return f"{int(round(x)):,}"


def g(v):
    """A config value as text: floats in plain notation, lists in brackets."""
    if isinstance(v, bool):
        return str(v).lower()
    if isinstance(v, float):
        return f"{v:g}"
    if isinstance(v, list):
        return "[" + ", ".join(g(x) for x in v) + "]"
    return str(v)


def table(header, rows, align=None):
    """A pandoc/GitHub pipe table; separator dashes proportional to column width (pandoc's relative widths)."""
    cells = [[str(c).replace("|", "\\|") for c in header]] + [[str(c).replace("|", "\\|") for c in r] for r in rows]
    w = [max(len(r[i]) for r in cells) for i in range(len(header))]
    dash = [max(3, round(64 * wi / sum(w))) for wi in w]
    align = align or ("l" + "r" * (len(header) - 1))
    sep = [(":" + "-" * d) if a == "l" else ("-" * d + ":") if a == "r" else (":" + "-" * d + ":")
           for a, d in zip(align, dash)]
    out = ["| " + " | ".join(cells[0]) + " |", "|" + "|".join(sep) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in cells[1:]]
    return "\n".join(out) + "\n"


def get(d, dotted, default=None):
    for k in dotted.split("."):
        if not isinstance(d, dict) or k not in d:
            return default
        d = d[k]
    return d


def git_last(rel):
    out = subprocess.run(["git", "log", "-1", "--format=%h\t%ad\t%s", "--date=short", "--", rel], cwd=ROOT,
                         capture_output=True, text=True).stdout.strip()
    return out.split("\t", 2) if out else ["uncommitted", "", ""]


def save(fig, name, caption):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"{name}.pdf", metadata={"CreationDate": None, "Creator": None, "Producer": None})
    fig.savefig(FIG / f"{name}.png", dpi=300, metadata={"Software": None})
    plt.close(fig)
    FIGS[name] = caption
    print("  ", name)


def figure(name):
    """Image without alt text (pandoc then adds no auto-numbered float) and the numbered caption as its own paragraph,
    so the F1–F12 numbers hold in Markdown, PDF and DOCX alike."""
    return f"![](figures/{name}.png){{width=100%}}\n\n*{FIGS[name]}*\n"


# ---------------------------------------------------------------- derived data
def history(arm, seed, fold):
    return json.loads(_guard(SC.RUNS / f"{RD.RUN_DIR[arm]}_s{seed}" / fold / "metrics.json").read_text())


def summary(arm, seed):
    return json.loads(_guard(SC.RUNS / f"{RD.RUN_DIR[arm]}_s{seed}" / "summary.json").read_text())


def tkey(fold):
    return fold.replace("fold_", "fold", 1)


def timing(arm):
    """{seed: {fold dir: timing row}} from report_data training.timing."""
    t = R["arms"][arm]["training"]["timing"]
    return {int(s): {f: next((r for r in rows if r["fold"] == tkey(f)), None) for f in FOLDS} for s, rows in t.items()}


def aug_steps(arm):
    """The train transform the arm's YAML builds (models/common/preprocessing.build_transforms), as text."""
    from models.common.config import load_config
    from models.common.preprocessing import build_transforms
    cfg = load_config(ROOT / CFG_FILES[arm])
    pre = CFG[arm]["preprocess"]
    out = []
    for x in build_transforms(cfg, "train").transforms:
        n = type(x).__name__
        if n == "FundusPreprocess":
            nice = {"circle_crop": "circle crop", "letterbox": "letterbox", "ben_graham": "Ben Graham", "clahe": "CLAHE"}
            out.append("fundus stage: " + ", ".join(v for k, v in nice.items() if pre.get(k)))
        elif n == "RandomResizedCrop":
            out.append(f"random resized crop to {x.size[0]} px, scale {x.scale[0]:g}–{x.scale[1]:g}")
        elif n == "RandomHorizontalFlip":
            out.append(f"horizontal flip, p {x.p:g}")
        elif n == "RandomVerticalFlip":
            out.append(f"vertical flip, p {x.p:g}")
        elif n == "RandomRotation":
            out.append(f"rotation ±{x.degrees[1]:g}°")
        elif n == "ColorJitter":
            parts = [f"{k} {getattr(x, k)[0]:g}–{getattr(x, k)[1]:g}" for k in ("brightness", "contrast", "saturation")
                     if getattr(x, k)]
            parts += [f"hue ±{x.hue[1]:g}"] if x.hue else []
            out.append("colour jitter: " + ", ".join(parts))
        elif n == "GammaJitter":
            out.append(f"gamma U({x.lo:g}, {x.hi:g}), p {x.p:g}")
        elif n == "RandomGrayscale":
            out.append(f"greyscale, p {x.p:g}")
        elif n == "ResolutionJitter":
            out.append(f"resolution jitter down to {x.min_frac:g}, p {x.p:g}")
        elif n == "RandomAffine":
            out.append(f"affine: translate {x.translate[0]:g}, scale {x.scale[0]:g}–{x.scale[1]:g}, shear ±{x.shear[1]:g}°")
        elif n == "RandomPerspective":
            out.append(f"perspective {x.distortion_scale:g}, p {x.p:g}")
        elif n == "GaussianBlur":
            out.append(f"Gaussian blur {x.kernel_size[0]}×{x.kernel_size[1]}, σ {x.sigma[0]:g}–{x.sigma[1]:g}")
        elif n == "GaussianNoise":
            out.append(f"pixel noise σ {x.sigma:g}, p {x.p:g}")
        elif n == "Normalize":
            out.append(f"normalise ({pre['normalize']} statistics)")
        elif n == "RandomErasing":
            out.append(f"random erasing {x.scale[0]:g}–{x.scale[1]:g} of the area, p {x.p:g}")
        elif n != "ToTensor":
            out.append(n)
    a4 = get(CFG[arm], "train.aug_v4")
    if a4 and a4.get("enabled"):
        out.append(f"batch: Fourier amplitude swap, centred {a4['fourier_block']}×{a4['fourier_block']} block, "
                   f"random batch partner, p {a4['fourier_p']:g}")
    return out


def code_constant(rel, pattern):
    m = re.search(pattern, _guard(ROOT / rel).read_text())
    return m.group(1) if m else "n/a"


def protocol_block(start, stop):
    """Lines of PROTOCOL_V2.md from the line starting with `start` up to (not incl.) the next line starting with `stop`."""
    lines = _guard(ROOT / PROTOCOL).read_text().splitlines()
    i = next(k for k, ln in enumerate(lines) if ln.strip().startswith(start))
    j = next((k for k in range(i + 1, len(lines)) if lines[k].strip().startswith(stop)), len(lines))
    return " ".join(ln.strip() for ln in lines[i:j] if ln.strip())


def readme_paragraphs(prefixes):
    paras = [p.strip() for p in _guard(ROOT / README).read_text().split("\n\n")]
    return [" ".join(p.split("\n")) for p in paras if any(p.startswith(x) for x in prefixes)]


def ls_floor(eps, k):
    """Cross-entropy of a prediction equal to the smoothed target (unweighted): the loss floor label smoothing sets."""
    hi, lo = 1 - eps + eps / k, eps / k
    return -(hi * math.log(hi) + (k - 1) * lo * math.log(lo))


def fold_arop_composition():
    """Per CV fold: AP-ROP validation photos and infants by source, from the v4 fold files (val sides equal v3's)."""
    out, same = {}, True
    for k in range(SC.NFOLD):
        f4 = pd.read_csv(_guard(SC.PREP / "v4" / f"fold_{k}.csv"))
        f3_ = pd.read_csv(_guard(SC.PREP / f"fold_{k}.csv"))
        same &= set(f4[f4.split == "val"].image_path) == set(f3_[f3_.split == "val"].image_path)
        v = f4[(f4.split == "val") & (f4.cls6 == "arop")]
        t = f4[(f4.split == "train") & (f4.cls6 == "arop")]
        out[k] = {s: (int((v.source == s).sum()), int(v[v.source == s].patient_id.nunique())) for s in sorted(v.source.unique())}
        out[("train", k)] = {s: int((t.source == s).sum()) for s in SRC}
    return out, same


def near_duplicate_overlap():
    """(near-duplicate exclusions, exact duplicates, overlap) from the two committed exclusion lists."""
    exact = set(pd.read_csv(_guard(SC.PREP / "exclude_duplicates.csv")).image_path)
    near = set(pd.read_csv(_guard(SC.PREP / "v4" / "exclude_near_duplicates.csv")).image_path)
    return len(near), len(exact), len(near & exact)


def arop_recall_by_fold(arm):
    """Kernel-side AP-ROP recall and support per (seed, CV fold) from metrics.json (6-class, duplicates not excluded)."""
    return {(s, k): (history(arm, s, f"fold_{k}")["val"]["per_class"]["AP-ROP"]["recall"],
                     history(arm, s, f"fold_{k}")["val"]["per_class"]["AP-ROP"]["support"])
            for s in SEEDS for k in range(SC.NFOLD)}


def pooled_recall(rec, folds):
    hits = sum(r * n for (s, k), (r, n) in rec.items() if k in folds)
    return hits / sum(n for (s, k), (r, n) in rec.items() if k in folds)


# ---------------------------------------------------------------- figures
def fig_data():
    tb, q, aq = V4["training_by_class"], V4["ostrava_quality"], AUD["ostrava_quality"]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.9), gridspec_kw={"width_ratios": [2.0, 1, 1], "wspace": 0.8})
    ax = axes[0]
    x, w = np.arange(len(CL)), 0.4
    v3, v4 = [tb["v3_kept"][c] for c in CL], [tb["v4"][c] for c in CL]
    ax.bar(x - w / 2, v3, w * 0.95, color=COL["B"], label=f"v3 pool (B): {n0(V4['training_rows']['v3_kept'])}")
    ax.bar(x + w / 2, v4, w * 0.95, color=COL["H"], label=f"v4 pool (H, I): {n0(V4['training_rows']['v4'])}")
    for i, (a, b) in enumerate(zip(v3, v4)):
        ax.text(i, max(a, b) + 25, f"{b - a:+d}".replace("-", MINUS) if b != a else "±0", ha="center", fontsize=6.5)
    ax.set_xticks(x, [SHORT[c] for c in CL], fontsize=6.8)
    ax.set_ylabel("Photos eligible for training")
    ax.set_ylim(0, max(v3 + v4) * 1.18)
    ax.legend(frameon=False, fontsize=6.5, loc="upper right")
    ax.set_title("(a) Training pool after caps", fontsize=8.5)
    for ax, key, items, title in ((axes[1], "removed_by_class", [c for c in CL if aq["by_class"][c]["frames"]], "(b) Ostrava quality, by class"),
                                  (axes[2], "removed_by_device", list(q["removed_by_device"]), "(c) Ostrava quality, by camera")):
        tot = [aq["by_class"][c]["frames"] if key == "removed_by_class" else aq["by_device"][c]["frames"] for c in items]
        rem = [q[key][c] for c in items]
        y = np.arange(len(items))
        ax.barh(y, rem, 0.6, color="#6b7280")
        for i, (r, t) in enumerate(zip(rem, tot)):
            ax.text(r + max(rem) * 0.04, i, f"{r} of {n0(t)}\n({pct(r / t)})", va="center", fontsize=6)
        ax.set_yticks(y, [SHORT[c] if key == "removed_by_class" else DEVICE[c].replace(" ", "\n", 1) for c in items], fontsize=7)
        ax.invert_yaxis()
        ax.set_xlim(0, max(rem) * 1.9)
        ax.set_xlabel("Frames removed")
        ax.set_title(title, fontsize=8.5)
    d = {c: tb["v4"][c] - tb["v3_kept"][c] for c in CL}
    big = min(d, key=d.get)
    chg = f"{d[big]:+d}".replace("-", MINUS)
    save(fig, "f01_data",
         f"Figure 1. Data. (a) Photos eligible for training after the per-infant caps: v3 pool "
         f"{n0(V4['training_rows']['v3_kept'])} (arm B) vs v4 pool {n0(V4['training_rows']['v4'])} (arms H and I); "
         f"labels give v4 {MINUS} v3 (largest change: {CLN[big]} {chg}). (b, c) Ostrava frames that failed "
         f"the per-camera quality rule and left the training candidates: {q['removed_by_quality']} of {n0(q['frames'])} "
         f"({pct(q['removed_by_quality'] / q['frames'])}), by class and by camera ({q['failing_kept_by_floor']} failing "
         f"frame kept by the {V4['floor']}-frame floor). Validation rows are never filtered.")


def fig_recipe():
    import textwrap
    cfg = CFG["I"]
    tr, md, pre = cfg["train"], cfg["model"], cfg["preprocess"]
    steps_i, steps_b = aug_steps("I"), set(aug_steps("B"))
    timm = code_constant("models/common/architectures.py", r'"dinov2_s":\s*"([^"]+)"')
    patch = int(re.search(r"patch(\d+)", timm).group(1))
    pct_start = code_constant("models/common/optim.py", r"pct_start=([\d.]+)")
    clip = code_constant("models/common/train_utils.py", r'"grad_clip",\s*([\d.]+)')
    b_ls, wu = CFG["B"]["train"]["label_smoothing"], tr["head_warmup_epochs"]
    pre_steps = [x for x in steps_i if x.startswith(("fundus", "random resized"))]
    photo_steps = [x for x in steps_i if x not in pre_steps and not x.startswith("batch")]
    batch_steps = [x.replace("batch: ", "") for x in steps_i if x.startswith("batch")]
    left = [("1  Input", [("one fundus photo of the training fold", False)] + [(x, x not in steps_b) for x in pre_steps]),
            ("2  Per-photo augmentation, in order", [(x, x not in steps_b) for x in photo_steps]),
            (f"3  Per batch of {tr['batch_size']}", [(x, True) for x in batch_steps]),
            ("4  Backbone", [(f"DINOv2 ViT-S/{patch} with registers, ImageNet-pretrained (timm {timm})", False),
                             (f"{pre['image_size'] // patch}×{pre['image_size'] // patch} patches of {patch} px; "
                              f"stochastic depth (drop path) {g(md['drop_path'])}", False)]),
            ("5  Classification head", [(f"dropout {g(md['dropout'])}, linear layer to {cfg['data']['num_classes']} classes", False)])]
    right = [("Loss", [(f"cross-entropy, class-balanced weights (β {g(tr['class_balanced_beta'])})", False),
                       (f"label smoothing {g(tr['label_smoothing'])} (B: {g(b_ls)})", tr["label_smoothing"] != b_ls)]),
             ("Optimiser", [(f"AdamW, peak lr {g(tr['lr'])}, layer decay {g(tr['layer_decay'])}", False),
                            (f"weight decay {g(tr['weight_decay'])}, gradient clip {clip}", False),
                            (f"OneCycle: warm-up over {float(pct_start):.0%} of steps, then cosine", False),
                            (f"{tr['epochs']} epochs, batch {tr['batch_size']}, mixed precision", False)]),
             ("Head warm-up", [(("epoch 1" if wu == 1 else f"epochs 1–{wu}") + ": head only, everything else frozen", True),
                               ("OneCycle schedule unchanged", False)]),
             ("Weight EMA", [(f"decay {g(tr['ema_decay'])} per optimiser step", True),
                             ("the EMA weights are validated and saved", True),
                             ("last epoch kept, no early stopping", False)])]
    LH, TH, PAD, GAP, RGAP = 0.28, 0.4, 0.3, 0.42, 0.22

    def lay(col, width):
        out = []
        for title, items in col:
            lines = [(ln, ch, k == 0) for x, ch in items for k, ln in enumerate(textwrap.wrap(x, width))]
            out.append((title, lines, PAD + TH + LH * len(lines)))
        return out

    lcol, rcol = lay(left, 62), lay(right, 37)
    Y = max(sum(b[2] for b in lcol) + GAP * (len(lcol) - 1), sum(b[2] for b in rcol) + RGAP * len(rcol) + 1.2) + 0.7
    fig, ax = plt.subplots(figsize=(7.2, 0.6 * Y))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, Y)
    ax.axis("off")

    def draw(x0, top, w, title, lines, h, fc):
        changed = any(ch for _, ch, _ in lines)
        ax.add_patch(FancyBboxPatch((x0, top - h), w, h, boxstyle="round,pad=0.04", fc=fc,
                                    ec=COL["I"] if changed else "#374151", lw=1.5 if changed else 0.8))
        ax.text(x0 + 0.15, top - 0.18, title, fontsize=7.6, weight="bold", va="top", color=INK)
        for k, (ln, ch, first) in enumerate(lines):
            ax.text(x0 + 0.15 + (0 if first and ch else 0.17), top - 0.18 - TH - k * LH,
                    ("▸ " if first and ch else "") + ln, fontsize=6.4, va="top", weight="bold" if ch else "normal", color=INK)

    arrow = dict(arrowstyle="->", lw=0.9, color="#374151")
    x0, wl, xr, wr = 0.08, 5.62, 5.98, 3.97
    top, mid = Y - 0.06, {}
    for i, (title, lines, h) in enumerate(lcol):
        draw(x0, top, wl, title, lines, h, "#f3f4f6")
        mid[title[0]] = top - h / 2
        if i < len(lcol) - 1:
            ax.annotate("", xy=(x0 + wl / 2, top - h - GAP + 0.07), xytext=(x0 + wl / 2, top - h - 0.06), arrowprops=arrow)
        top -= h + GAP
    top = Y - 0.06
    for title, lines, h in rcol:
        draw(xr, top, wr, title, lines, h, "#ffffff")
        top -= h + RGAP
    ax.text(xr + wr / 2, top + 0.05, "training loop for 4 and 5", ha="center", va="top", fontsize=7, style="italic", color=INK)
    ax.annotate("", xy=(x0 + wl + 0.07, mid["4"]), xytext=(xr + wr / 2, top - 0.25),
                arrowprops={**arrow, "ls": "--", "connectionstyle": "arc3,rad=-0.25"})
    ax.text(xr, 0.12, "▸ bold, coloured outline: differs from arm B\n   (arm H trains with B's recipe)", fontsize=6.8, color=INK)
    save(fig, "f02_recipe",
         f"Figure 2. Arm I's training recipe as built from `{CFG_FILES['I']}` by "
         f"`models/common/preprocessing.build_transforms` and the training loop (`models/common/train_utils.py`, "
         f"`models/common/optim.py`). Marked items differ from arm B; everything else, including the "
         f"{pre['image_size']}-px input and the {tr['epochs']}-epoch OneCycle schedule, equals B. Arm H is B's recipe on "
         f"the v4 pool.")


def forest(ax, arms, level, ref="B", dagger=()):
    vals = []
    for i, a in enumerate(arms):
        m = SCO["selection_rule"]["table"][a] if level == "selection" else R["arms"][a][level]
        q = m["qwk_5class"]
        lo, hi = m["ci95"] if level == "selection" else m["ci95"]["qwk_5class"]
        key = a in NEW
        ax.plot([lo, hi], [i, i], color=COL[a], lw=2.6 if a == ref else 1.5, alpha=1 if key else 0.5, solid_capstyle="round")
        ax.plot(q, i, "o", color=COL[a], ms=7.5 if a == ref else 5.5, mec="white", mew=0.8, alpha=1 if key else 0.65)
        vals.append((q, lo, hi))
    xmax = max(v[2] for v in vals)
    for i, (a, (q, lo, hi)) in enumerate(zip(arms, vals)):
        ax.text(xmax + 0.012, i, f3(q), va="center", fontsize=7.5, weight="bold" if a == ref else "normal")
    ax.set_xlim(min(v[1] for v in vals) - 0.01, xmax + 0.055)
    ax.set_yticks(range(len(arms)), [NAMES[a] + (" †" if a in dagger else "") for a in arms])
    for t, a in zip(ax.get_yticklabels(), arms):
        t.set_weight("bold" if a == ref else "normal")
    ax.set_ylim(len(arms) - 0.4, -1.5)
    ax.grid(axis="x", lw=0.4, alpha=0.5)
    return vals


def fig_selection():
    tab = SCO["selection_rule"]["table"]
    m = SCO["v4_selection"]["margin_cv"]
    arms = [a for a in ARMS if a in tab]
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    forest(ax, arms, "selection")
    b = tab["B"]["qwk_5class"]
    ax.axvline(b, color=COL["B"], lw=1.0, ls=":")
    ax.axvline(b + m, color=INK, lw=1.0, ls="--")
    ax.text(b + m + 0.003, -1.05, f"adoption bar: B + {m:.3f} = {f3(b + m)}", fontsize=7, va="center")
    ax.text(b - 0.003, -1.05, f"B = {f3(b)}", fontsize=7, va="center", ha="right", color=INK)
    ax.set_xlabel("CV QWK5, pooled out-of-fold, per photograph (95% infant-clustered CI)")
    best = max("HI", key=lambda a: tab[a]["qwk_5class"])
    save(fig, "f03_selection",
         f"Figure 3. Selection metric of Amendment 5c: CV pooled out-of-fold photo-level QWK5 with 95% "
         f"infant-clustered bootstrap CI ({SCO['bootstrap']['replicates']:,} replicates), all nine arms, both seeds pooled. "
         f"Dotted: arm B ({f3(b)}); dashed: the adoption bar B + {m:.3f} = {f3(b + m)}. No arm reaches it; "
         f"the closer v4 arm is {best} at {f3(tab[best]['qwk_5class'])}.")


def fig_dev():
    arms = [a for a in ARMS if a in R["arms"]]
    trans = [a for a in arms if R["arms"][a]["dev_status"].startswith("transductive")]
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    forest(ax, arms, "dev_eye", dagger=trans)
    b = R["arms"]["B"]["dev_eye"]["qwk_5class"]
    ax.axvline(b, color=COL["B"], lw=1.0, ls=":")
    ax.text(b - 0.003, -1.05, f"B = {f3(b)}", fontsize=7, va="center", ha="right")
    ax.set_xlabel("Multi-View dev half, eye-level QWK5 (95% infant-clustered CI)")
    q = {a: R["arms"][a]["dev_eye"]["qwk_5class"] for a in arms}
    dev = R["dataset"]["dev"]
    save(fig, "f04_dev_eye",
         f"Figure 4. The new hospital (Multi-View dev half: {dev['eyes']} eyes, {dev['infants']} infants, "
         f"{dev['images']} photos), eye-level QWK5 with 95% CI, mean over each arm's "
         f"{R['arms']['B']['dev_eye']['n_lists']} fold-seed models. Descriptive (development-external), not a selection "
         f"input. Range {f3(min(q.values()))} ({min(q, key=q.get)}) to {f3(max(q.values()))} ({max(q, key=q.get)}). "
         f"† {', '.join(trans)}: dev photos were seen unlabelled during training.")


def fig_contrasts():
    pairs = ["H-B", "I-B", "I-H"]
    m = SCO["v4_selection"]["margin_cv"]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.6), sharey=True)
    allci = [SCO["contrasts"][b]["pairs"][p]["qwk_5class"]["diff_ci"] for b in ("cv_image", "dev_eye") for p in pairs]
    lim = max(abs(v) for c in allci for v in c) + 0.012
    for ax, blk, title in zip(axes, ("cv_image", "dev_eye"),
                              ("(a) CV, per photograph (the selection data)", "(b) Multi-View dev, per eye (descriptive)")):
        for i, p in enumerate(pairs):
            q = SCO["contrasts"][blk]["pairs"][p]["qwk_5class"]
            ax.plot(q["diff_ci"], [i, i], color=COL[p[0]], lw=2.0, solid_capstyle="round")
            ax.plot(q["diff"], i, "o", color=COL[p[0]], ms=6, mec="white", mew=0.8)
            ax.text(q["diff"], i - 0.3, f"{s3(q['diff'])}  {ci3(q['diff_ci'])}", ha="center", fontsize=6.8,
                    bbox=dict(fc="white", ec="none", pad=0.6, alpha=0.85))
        ax.axvspan(-m, m, color=GREY, alpha=0.2, lw=0)
        ax.axvline(0, color="#374151", lw=0.8)
        ax.set_xlim(-lim, lim)
        ax.set_yticks(range(len(pairs)), [p.replace("-", f" {MINUS} ") for p in pairs])
        ax.set_ylim(len(pairs) - 0.5, -0.8)
        ax.set_xlabel("Paired difference in QWK5 (95% CI)")
        ax.set_title(title, fontsize=8.5)
        ax.grid(axis="x", lw=0.4, alpha=0.5)
    c = SCO["contrasts"]
    save(fig, "f05_contrasts",
         f"Figure 5. Paired contrasts of the v4 arms (infant-clustered bootstrap, {SCO['bootstrap']['replicates']:,} "
         f"replicates, the same resample for both arms). Grey band: ±{m:.3f}, the Amendment 5c margin; line: zero. "
         f"CV: {c['cv_image']['n_images']:,} photos (duplicates excluded); dev: {c['dev_eye']['n_eyes']} eyes.")


def fig_stage_recall():
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9), sharey=True)
    for ax, lv, title in zip(axes, ("cv_image", "dev_eye"), ("(a) CV, per photograph", "(b) Multi-View dev, per eye")):
        x, w = np.arange(len(CL)), 0.27
        for k, a in enumerate(NEW):
            cls = R["arms"][a][lv]["classes"]
            v = [cls[c]["recall"] or 0 for c in CL]
            ax.bar(x + (k - 1) * w, v, w * 0.92, color=COL[a], label=NAMES[a])
            for c in ("stage_1", "arop"):
                i = CL.index(c)
                ax.text(i + (k - 1) * w, v[i] + 0.02, f"{v[i]:.2f}", ha="center", fontsize=5.6, rotation=90)
        n = [R["arms"]["B"][lv]["classes"][c]["support"] for c in CL]
        ax.set_xticks(x, [f"{SHORT[c]}\nn={v}" for c, v in zip(CL, n)], fontsize=7)
        ax.set_ylim(0, 1.1)
        ax.set_title(title)
    axes[0].set_ylabel("Recall")
    axes[0].legend(ncol=3, frameon=False, fontsize=7, loc="upper center", bbox_to_anchor=(1.05, -0.25))
    save(fig, "f06_stage_recall",
         "Figure 6. Recall per stage for B, H and I (mean over the arm's lists: 2 pooled CV lists, "
         f"{R['arms']['B']['dev_eye']['n_lists']} dev lists). n = photos (CV) or eyes (dev). Stage 1 and AP-ROP are labelled.")


def fig_confusion():
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.75), gridspec_kw={"wspace": 0.12})
    lab = [SHORT[c] for c in CL]
    for ax, a in zip(axes, NEW):
        cm = np.array(R["arms"][a]["dev_eye"]["confusion"]["counts"])
        share = np.array(R["arms"][a]["dev_eye"]["confusion"]["row_normalised"])
        ax.imshow(share, cmap="Oranges", vmin=0, vmax=1)
        for i in range(6):
            for j in range(6):
                if cm[i].sum() == 0:
                    continue
                v = cm[i, j]
                txt = f"{v:.0f}" if abs(v - round(v)) < 1e-6 else f"{v:.1f}"
                ax.text(j, i, f"{txt}\n{share[i, j]:.0%}", ha="center", va="center", fontsize=5.2,
                        color="white" if share[i, j] > 0.55 else "black")
        ax.set_xticks(range(6), lab, fontsize=6.5)
        ax.set_yticks(range(6), lab if a == "B" else [], fontsize=6.5)
        ax.set_xlabel("Predicted stage", fontsize=7.5)
        ax.set_title(NAMES[a], fontsize=8)
        ax.spines[:].set_visible(False)
    axes[0].set_ylabel("True stage", fontsize=7.5)
    save(fig, "f07_confusion_dev_eye",
         f"Figure 7. Eye-level confusion matrices on the Multi-View dev half ({R['dataset']['dev']['eyes']} eyes): counts "
         f"are means over each arm's {R['arms']['B']['dev_eye']['n_lists']} fold-seed models (hence decimals); the "
         "percentage is the row share (recall on the diagonal).")


def fig_loso():
    srcs = SCO["loso"]["B"]["sources"]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), sharey=True)
    for ax, key, xl in zip(axes, ("qwk_5class", "macro_f1_present"), ("QWK5 (95% CI)", "macro-F1 over present classes (95% CI)")):
        for j, s in enumerate(srcs):
            for k, a in enumerate(NEW):
                L = SCO["loso"][a][s]
                y = j + (k - 1) * 0.24
                ax.plot(L["ci95"][key], [y, y], color=COL[a], lw=1.8, solid_capstyle="round")
                ax.plot(L[key], y, "o", color=COL[a], ms=5.5, mec="white", mew=0.7,
                        label=(f"{a}: {SCO['loso'][a]['pool']} pool, {'v4' if get(CFG[a], 'train.ema_decay') else 'B' + chr(39) + 's'} "
                               f"recipe; mean LOSO QWK5 {f3(SCO['loso'][a]['mean_qwk_5class'])}"
                               if j == 0 and key == "qwk_5class" else None))
        L0 = SCO["loso"]["B"]
        ax.set_yticks(range(len(srcs)), [f"{SRC[s]} held out\n{n0(L0[s]['n_images'])} photos, {L0[s]['n_infants']} infants"
                                         for s in srcs], fontsize=7)
        ax.set_ylim(len(srcs) - 0.25, -0.55)
        ax.set_xlabel(xl)
        ax.grid(axis="x", lw=0.4, alpha=0.5)
    for j, s in enumerate(srcs):
        zero = sorted({c for a in NEW for c, v in SCO["loso"][a][s]["train_support"].items() if v == 0})
        if zero:
            axes[0].annotate("trained with 0 " + ", ".join(CLN[c] for c in zero) + " photos (B, H, I)", xy=(0.02, j + 0.45),
                             xycoords=("axes fraction", "data"), fontsize=6.3, color=INK, style="italic", va="center")
    axes[0].legend(frameon=False, fontsize=6.3, loc="upper center", bbox_to_anchor=(1.0, -0.2), ncol=1)
    save(fig, "f08_loso",
         "Figure 8. Leave-one-source-out: each model trains on two sources and is scored on every photo of the third "
         f"(both seeds pooled; 95% infant-clustered CI). B trains on the v3 pool, H and I on the v4 pool. ROP-VL is the "
         "only source with stage 4/5, so its fold scores that stage with zero training examples.")


def fig_training():
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.0), sharex=True)
    bc = R["arms"]["B"]["training"]["curves"]
    for r, a in enumerate("HI"):
        wu = get(CFG[a], "train.head_warmup_epochs", 0) or 0
        for s in SEEDS:
            for k in range(SC.NFOLD):
                h = history(a, s, f"fold_{k}")["history"]
                ep = np.arange(1, len(h["train_loss"]) + 1)
                for c, key in ((0, "train_loss"), (1, "val_macro_f1")):
                    axes[r, c].plot(ep, h[key], color=COL[a], lw=0.7, alpha=0.8, ls="-" if s == SEEDS[0] else "--")
        ep = np.arange(1, len(bc["train_loss"]["mean"]) + 1)
        for c, key, yl, short in ((0, "train_loss", "Training loss", "training loss"),
                                  (1, "val_macro_f1", "Held-out fold macro-F1", "held-out macro-F1")):
            ax = axes[r, c]
            ax.plot(ep, bc[key]["mean"], color=INK, lw=1.7)
            if wu:
                ax.axvspan(0.5, wu + 0.5, color=GREY, alpha=0.25, lw=0)
                ax.text(wu + 0.6, ax.get_ylim()[1] * (0.97 if c == 0 else 0.95), "head-only epoch", fontsize=6.3, va="top")
            ax.set_ylabel(yl)
            ax.set_title(f"({'abcd'[2 * r + c]}) {NAMES[a]}: {short}", fontsize=8.5)
            ax.grid(lw=0.3, alpha=0.5)
    for ax in axes[1]:
        ax.set_xlabel("Epoch")
    handles = [Line2D([], [], color=GREY, lw=0.9, ls="-", label=f"seed {SEEDS[0]}, one line per CV fold"),
               Line2D([], [], color=GREY, lw=0.9, ls="--", label=f"seed {SEEDS[1]}, one line per CV fold"),
               Line2D([], [], color=INK, lw=1.7, label=f"B, mean of {bc['train_loss']['n_runs']} runs (reference)")]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=7, bbox_to_anchor=(0.5, -0.03))
    fig.subplots_adjust(hspace=0.35, wspace=0.28, bottom=0.12)
    save(fig, "f09_training_curves",
         f"Figure 9. Training curves of H and I on the five CV folds, both seeds (thin lines), with arm B's mean curve. "
         f"I trains with label smoothing {g(CFG['I']['train']['label_smoothing'])} against B's and H's "
         f"{g(CFG['B']['train']['label_smoothing'])}, so its loss sits higher by construction; its held-out macro-F1 is "
         f"measured on the EMA weights, whose early epochs still carry most of the initial weights.")


def fig_any_rop():
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
    for a in NEW:
        c = R["arms"][a]["curves"]["dev_eye_any_rop"]
        axes[0].plot(c["roc"]["fpr"], c["roc"]["tpr"], color=COL[a], lw=1.6, label=f"{NAMES[a]} (AUC {f3(c['roc']['auc'])})")
        axes[1].plot(c["pr"]["recall"], c["pr"]["precision"], color=COL[a], lw=1.6, label=f"{NAMES[a]} (AP {f3(c['pr']['ap'])})")
    prev = R["arms"]["B"]["curves"]["dev_eye_any_rop"]["pr"]["prevalence"]
    axes[0].plot([0, 1], [0, 1], "--", color=GREY, lw=0.8)
    axes[1].axhline(prev, ls=":", color=GREY, lw=0.9)
    axes[1].text(0.02, prev + 0.008, f"prevalence {f3(prev)}", fontsize=6.5, color=INK)
    axes[0].set_xlabel(f"False-positive rate (1 {MINUS} specificity)")
    axes[0].set_ylabel("Sensitivity")
    axes[0].set_title("(a) ROC, any ROP vs normal, per eye")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title("(b) Precision–recall, per eye")
    axes[1].set_ylim(prev - 0.05, 1.01)
    axes[0].legend(frameon=False, fontsize=6.5, loc="lower right")
    axes[1].legend(frameon=False, fontsize=6.5, loc="center left")
    save(fig, "f10_any_rop_dev_eye",
         f"Figure 10. Any-ROP vs normal on the Multi-View dev eyes for B, H and I. Curves and legend values use the eye "
         f"score averaged over each arm's {R['arms']['B']['dev_eye']['n_lists']} fold-seed models; the tables report the "
         "mean of the per-model AUC and AP, which is slightly lower.")


def fig_reliability():
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
    for ax, a in zip(axes, ("B", "I")):
        rel = R["arms"][a]["reliability_dev_eye"]
        ax.plot([0, 1], [0, 1], "--", color=GREY, lw=0.8)
        for key, lab, col, ls in (("uncalibrated", "as trained", COL[a], "-"), ("cv_temperature", "after the CV temperature", "#6b7280", "--")):
            b = rel[key]["bins"]
            ax.plot([x["confidence"] for x in b], [x["accuracy"] for x in b], ls, color=col, marker="o", ms=4, lw=1.4,
                    label=f"{lab} (ECE {f3(rel[key]['ece'])})")
        T = R["arms"][a]["temperature"]
        ax.set_xlabel("Confidence (top-class probability)")
        ax.set_ylabel("Share correct")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title(f"{NAMES[a]}: T = " + " / ".join(f"{T[str(s)]:.3f}" for s in SEEDS), fontsize=8.5)
        ax.legend(frameon=False, fontsize=7, loc="upper left")
    save(fig, "f11_reliability_dev_eye",
         "Figure 11. Reliability of the eye-level stage probabilities on the Multi-View dev half, 10 equal-width bins, "
         f"both seeds pooled. Grey dashed: after each seed's temperature T (seed {SEEDS[0]} / {SEEDS[1]}), fitted on CV "
         "out-of-fold predictions. T < 1 sharpens the probabilities.")


def fig_timing():
    T = {a: timing(a) for a in NEW}
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9), gridspec_kw={"width_ratios": [2.4, 1], "wspace": 0.3})
    ax = axes[0]
    x, w = np.arange(len(FOLDS)), 0.27
    for k, a in enumerate(NEW):
        first = True
        for j, f in enumerate(FOLDS):
            vals = [T[a][s][f]["minutes"] for s in SEEDS if T[a][s][f]]
            xx = x[j] + (k - 1) * w
            if vals:
                ax.bar(xx, np.mean(vals), w * 0.92, color=COL[a], label=NAMES[a] if first else None)
                ax.plot([xx] * len(vals), vals, ".", color=INK, ms=2.5)
                first = False
            else:
                ax.text(xx, 1.5, "not in timing", rotation=90, fontsize=5.3, ha="center", va="bottom", color=INK)
    tick = {"fold_site": "Shenzhen\n(site)", "fold_loso_ostrava": "Ostrava\n(LOSO)", "fold_loso_ropvl": "ROP-VL\n(LOSO)"}
    ax.set_xticks(x, [tick.get(f, FOLD_LABEL[f]) for f in FOLDS], fontsize=6.3)
    ax.set_ylabel("Minutes per fold")
    ax.legend(frameon=False, fontsize=6.5, loc="upper left")
    ax.set_title("(a) Wall-clock per fold, Kaggle " + "/".join(sorted({r['gpu'] for a in NEW for s in SEEDS for r in T[a][s].values() if r})))
    ax = axes[1]
    for k, a in enumerate(NEW):
        v = [r["img_per_s"] for s in SEEDS for r in T[a][s].values() if r]
        ax.plot(k + np.linspace(-0.18, 0.18, len(v)), sorted(v), "o", color=COL[a], ms=3.5, mec="white", mew=0.4)
        ax.plot([k - 0.28, k + 0.28], [np.mean(v)] * 2, color=INK, lw=1.4)
        ax.text(k + 0.3, np.mean(v), f"{np.mean(v):.1f}", va="center", fontsize=6.5)
    ax.set_xticks(range(len(NEW)), list(NEW))
    ax.set_xlim(-0.5, len(NEW) - 0.2)
    ax.set_ylabel("Training throughput (img/s)")
    ax.set_title("(b) Throughput per fold-seed run")
    save(fig, "f12_timing",
         "Figure 12. Cost from the Kaggle kernel logs (report_data training.timing): minutes from one fold's start to the "
         "next (bar: mean of the two seeds; dots: seeds) and the throughput logged after 50 steps. B's two LOSO folds ran in a separate kernel whose timing is not "
         "in training.timing (see the run log).")


# ---------------------------------------------------------------- report
def report():
    OUT = []
    add = OUT.append
    tab = SCO["selection_rule"]["table"]
    sel = SCO["v4_selection"]
    c_cv, c_dev = SCO["contrasts"]["cv_image"]["pairs"], SCO["contrasts"]["dev_eye"]["pairs"]
    m = sel["margin_cv"]
    data_commit = git_last(FILES["report_data"])
    arms9 = [a for a in ARMS if a in R["arms"]]
    q = {a: tab[a]["qwk_5class"] for a in arms9}
    qd = {a: R["arms"][a]["dev_eye"]["qwk_5class"] for a in arms9}
    widths = {a: tab[a]["ci95"][1] - tab[a]["ci95"][0] for a in arms9}
    recB = {a: arop_recall_by_fold(a) for a in NEW}
    comp, val_same = fold_arop_composition()
    fold0_src = comp[0]
    ost0 = fold0_src.get("ostrava", (0, 0))
    arop_val0 = sum(v[0] for v in fold0_src.values())
    rec0 = {a: pooled_recall(recB[a], {0}) for a in NEW}
    rec14 = {a: pooled_recall(recB[a], set(range(1, SC.NFOLD))) for a in NEW}
    arop_bin = {a: tab[a]["arop_binary"]["recall"] for a in NEW}

    add("---")
    add('title: "RetinAI ROP v4: model-development report"')
    add('subtitle: "Arms H and I against the selected arm B"')
    add("toc: true")
    add("toc-depth: 2")
    add("fontsize: 10pt")
    add("papersize: a4")
    add("colorlinks: true")
    add("---\n")
    add(f"*Generated by `scripts/rop_v4_model_report.py` from the committed score files (data commit "
        f"`{data_commit[0]}`, {data_commit[1]}). Every number is read from a file; regenerate, never edit. "
        f"Rounding: 3 decimals, percentages 1 decimal. CI = 95% percentile interval of the infant-clustered bootstrap "
        f"({SCO['bootstrap']['replicates']:,} replicates, seed {SCO['bootstrap']['seed']}). QWK5 = quadratic weighted "
        f"kappa after folding AP-ROP onto stage 4/5. Arms: B = {R['arms']['B']['name']} (v3 pool, selected in v3); "
        f"H = {R['arms']['H']['name']} (B's recipe); I = {R['arms']['I']['name']}.*\n")

    # 1 ------------------------------------------------------------------
    add("# 1 Summary and verdict\n")
    add(f"**Rule (Amendment 5c, as scored).** {sel['rule']}\n")
    rows = []
    for a in ("I", "H"):
        dcv = round(q[a] - q["B"], 4)
        dl = round(sel["loso"][a] - sel["loso"]["B"], 4)
        rows.append([a, f3(q[a]), s3(dcv), f"≥ +{m:.3f}", "yes" if dcv >= m else "no", f3(sel["loso"][a]), s3(dl),
                     f"≥ {MINUS}{sel['margin_loso']:.3f}", "yes" if dl >= -sel["margin_loso"] else "no",
                     "**adopted**" if a == sel["chosen"] else "not adopted"])
    rows.append(["B", f3(q["B"]), "", "", "", f3(sel["loso"]["B"]), "", "", "", "**stays**" if sel["chosen"] == "B" else ""])
    add(table(["arm", "CV QWK5", "Δ vs B", "needs", "met", "mean LOSO QWK5", "Δ vs B", "needs", "met", "outcome"], rows,
              "lrrrlrrrll"))
    add(f"Δ is the difference of the 4-dp selection-table numbers, as the rule prescribes. The paired bootstrap "
        f"contrasts (section 4.5) use unrounded scores, hence H {MINUS} B = {s3(c_cv['H-B']['qwk_5class']['diff'])} there.\n")
    add("Reasons, verbatim from `v3_scores.json` (`v4_selection.reasons`):\n")
    for r in sel["reasons"]:
        add(f"- {r}")
    add(f"\n**Chosen arm: {sel['chosen']}.**\n")
    pi, ph = c_cv["I-B"]["qwk_5class"], c_cv["H-B"]["qwk_5class"]
    di, dih, dhb = (c_dev[k]["qwk_5class"] for k in ("I-B", "I-H", "H-B"))
    mf = c_cv["I-B"]["macro_f1_present"]
    add(f"Neither v4 arm moved the selection metric. On CV, I scores {f3(q['I'])} against B's {f3(q['B'])} "
        f"(paired difference {s3(pi['diff'])}, CI {ci3(pi['diff_ci'])}) and H {f3(q['H'])} ({s3(ph['diff'])}, "
        f"{ci3(ph['diff_ci'])}); both intervals contain zero and lie wholly below the +{m:.3f} bar. Both arms passed the "
        f"LOSO guard (mean LOSO QWK5 I {f3(sel['loso']['I'])}, H {f3(sel['loso']['H'])}, B {f3(sel['loso']['B'])}). "
        f"**B stays the selected model.** On the new hospital (Multi-View dev, eye level) I is "
        f"{s3(di['diff'])} QWK5 above B (CI {ci3(di['diff_ci'])}) and {s3(dih['diff'])} above H "
        f"(CI {ci3(dih['diff_ci'])}), both intervals excluding zero, while H {MINUS} B is {s3(dhb['diff'])} "
        f"(CI {ci3(dhb['diff_ci'])}); "
        f"dev is development-external and descriptive and does not enter the rule. The cost of the v4 changes shows in "
        f"the rare classes: CV macro-F1 of I minus B is {s3(mf['diff'])} (CI {ci3(mf['diff_ci'])}), and AP-ROP recall "
        f"falls from {f3(arop_bin['B'])} (B) to {f3(arop_bin['H'])} (H) and {f3(arop_bin['I'])} (I). That fall sits in "
        f"CV fold 0, where {ost0[0]} of the {arop_val0} AP-ROP validation photos come from {ost0[1]} Ostrava infant "
        f"whom the fold-0 models never train on (section 4.3).\n")
    add(figure("f03_selection"))

    # 2 ------------------------------------------------------------------
    add("# 2 What changed vs v3\n")
    add("## 2.1 Data audit and the v4 pool\n")
    add(f"Sources:\n\n- protocol: Amendment 5a items 1–5 with the 5a revision, `{PROTOCOL}`\n- audit: `{AUDIT_MD}`\n"
        f"- pool: `{FILES['v4_summary']}`\n")
    add(f"Four changes, all to **training** rows only; validation rows, the infant-to-fold "
        f"assignment and the 14 exact-duplicate exclusions are unchanged, so v4 CV is scored on the same "
        f"{SCO['contrasts']['cv_image']['n_images']:,} photos as v3 (validation sides of every v3 and v4 CV fold file "
        f"identical: {'yes' if val_same else 'NO'}).\n")
    add(f"1. **Ostrava frame quality (per camera).** {V4['rule']}")
    add(f"2. **Cap re-ranking.** Caps unchanged (normal {V4['caps']['normal']}, other classes {V4['caps']['_default']} "
        f"photos per infant and class); within a cap the kept frames are ranked by `{V4['ranking']}`.")
    add(f"3. **Floor.** An (infant, class) cell with fewer than {V4['floor']} passing frames keeps its {V4['floor']} "
        f"best-ranked frames; {len(V4['floor_cells'])} cell triggered it (" +
        "; ".join(f"`{c['patient_id']}` {c['cls6']}: {c['frames']} frames, {c['passing']} passing, {c['kept_failing']} failing frame kept"
                  for c in V4["floor_cells"]) + ").")
    nd = AUD["near_duplicates"]
    nd_n, nd_x, nd_o = near_duplicate_overlap()
    add(f"4. **Near-duplicates.** Both members of a cross-fold pair at dHash Hamming distance ≤ {nd['threshold']} "
        f"**and** pixel MAD ≤ {nd['max_pixel_mad']:g} grey levels ({nd['pixel_mad_size']}×{nd['pixel_mad_size']} greyscale) "
        f"leave the training rows: {V4['near_duplicate_exclusions']['images']} images "
        f"({', '.join(f'{SRC[s]} {v}' for s, v in V4['near_duplicate_exclusions']['by_source'].items())}). The "
        f"superseded hash-only rule would have removed {nd['superseded_exclusion']['images']}. Of {nd['pairs_total']} pairs at "
        f"distance ≤ {nd['threshold']}, {nd['true_duplicate']['total']} are true duplicates; {nd['different_label']} carry "
        f"different labels and go to the clinician list (nothing relabelled). " +
        (f"All {nd_n} are among the {nd_x} exact duplicates (`exclude_duplicates.csv`) that every kernel already drops "
         "at run time, so in practice this rule changes no trained-on photo; the v4 pool differs from v3 through items 1–3."
         if nd_o == nd_n else f"{nd_o} of the {nd_n} are among the {nd_x} exact duplicates every kernel already drops.") + "\n")
    add("**Per-camera thresholds** (`thresholds.json`; percentiles over each camera's non-empty Ostrava frames: "
        + ", ".join(f"`{k}` {v:g}th" for k, v in AUD["thresholds"]["percentiles"].items()) + "; no field-of-view rule):\n")
    rows = []
    for d, t in THR["by_device"].items():
        a = AUD["ostrava_quality"]["by_device"][d]
        rows.append([DEVICE[d], n0(t["n"]), f3(t["min_blur"]), f3(t["lum_lo"]), f3(t["lum_hi"]), a["failing"],
                     V4["ostrava_quality"]["removed_by_device"][d], pct(V4["ostrava_quality"]["removed_by_device"][d] / t["n"]),
                     a["by_reason"]["blur"], a["by_reason"]["dark"], a["by_reason"]["bright"]])
    add(table(["camera", "frames", "min blur_var", "lum_lo", "lum_hi", "failing", "removed", "share", "blur", "dark", "bright"], rows))
    ref = THR["reference_single_photo_hospitals"]
    add(f"Blur/dark/bright count failures per criterion (a frame can fail more than one). The superseded single-photo-hospital "
        f"reference ({', '.join(SRC[s] for s in ref['sources'])}: min blur_var {f3(ref['min_blur'])}, luminance "
        f"{f3(ref['lum_lo'])}–{f3(ref['lum_hi'])}, min FOV {f3(ref['min_fov'])}) is kept for the record only: "
        f"{ref['status']}. It would have failed {AUD['superseded_hospital_rule']['failing']:,} of "
        f"{AUD['superseded_hospital_rule']['frames']:,} Ostrava frames, including all "
        f"{AUD['superseded_hospital_rule']['by_class']['stage_3']['failing']} stage-3 frames.\n")
    add(figure("f01_data"))

    add("## 2.2 Recipe\n")
    add(f"Values from the three YAMLs (`{CFG_FILES['B']}` = B, `{CFG_FILES['H']}` = H, `{CFG_FILES['I']}` = I). "
        "Where a key is not set, the value the code builds is given in brackets "
        "(`models/common/preprocessing.build_transforms`).\n")
    steps = {a: aug_steps(a) for a in NEW}
    built = {}
    for a in NEW:
        rot = next(s for s in steps[a] if s.startswith("rotation"))
        crop = next(s for s in steps[a] if s.startswith("random resized crop"))
        built[a] = {"rotation": rot.replace("rotation ", ""), "min_crop_scale": crop.split("scale ")[1].split("–")[0]}
    keys = [("image size (px)", "preprocess.image_size"), ("normalize", "preprocess.normalize"), ("lr (peak, top layer)", "train.lr"),
            ("layer decay", "train.layer_decay"), ("weight decay", "train.weight_decay"), ("epochs", "train.epochs"),
            ("batch", "train.batch_size"), ("label smoothing", "train.label_smoothing"),
            ("class-balanced β", "train.class_balanced_beta"), ("drop path", "model.drop_path"), ("dropout", "model.dropout"),
            ("strong_aug", "train.strong_aug"), ("site_randomization", "train.site_randomization"),
            ("rotation (°)", "train.aug_v4.rotation"), ("min crop scale", "train.aug_v4.min_crop_scale"),
            ("gamma range", "train.aug_v4.gamma"), ("gamma p", "train.aug_v4.gamma_p"), ("noise σ", "train.aug_v4.noise_sigma"),
            ("noise p", "train.aug_v4.noise_p"), ("Fourier block", "train.aug_v4.fourier_block"), ("Fourier p", "train.aug_v4.fourier_p"),
            ("EMA decay", "train.ema_decay"), ("head warm-up epochs", "train.head_warmup_epochs"), ("fold files", "data.manifest")]
    rows = []
    for lab, k in keys:
        vals = []
        for a in NEW:
            v = get(CFG[a], k)
            if v is None:
                extra = {"train.aug_v4.rotation": f" (built: ±{built[a]['rotation'].lstrip('±')})",
                         "train.aug_v4.min_crop_scale": f" (built: {built[a]['min_crop_scale']})"}.get(k, "")
                vals.append("not set" + extra)
            else:
                vals.append(("v4" if "/v4/" in v else "v3") if k == "data.manifest" else g(v))
        rows.append([lab, *vals, "" if vals[0] == vals[1] == vals[2] else "differs"])
    add(table(["parameter", "B", "H", "I", ""], rows, "llllc"))
    dirs = {("v4" if "/v4/" in get(CFG[a], "data.manifest") else "v3"): str(Path(get(CFG[a], "data.manifest")).parent) for a in NEW}
    add("Fold files: " + ", ".join(f"{k} pool `{v}/`" for k, v in sorted(dirs.items())) +
        " (the YAML names fold 0; the kernel swaps in each fold).\n")
    diffs = {}
    for a in NEW:
        asrun = R["arms"][a]["training"]["config"]
        for sect in ("model", "preprocess", "train"):
            for k, v in asrun[sect].items():
                yv = CFG[a].get(sect, {}).get(k)
                if yv != v:
                    diffs.setdefault((f"{sect}.{k}", g(yv), g(v)), []).append(a)
    add("**As-run check.** The as-run configs (`training.config` in `report_data.json`, read from each run's "
        "`config.yaml`) equal the YAMLs on every model, preprocess and train key " +
        ("except:\n" if diffs else "without exception.\n"))
    if diffs:
        add(table(["key", "YAML", "as run", "arms"], [[f"`{k}`", y, f"`{v}`" if "/" in v else v, ", ".join(a)]
                                                     for (k, y, v), a in diffs.items()], "llll"))
        add("The kernels run the fundus stage once and cache it, identically for all three arms, so the cache does not "
            "bear on the comparison.\n")
    add("**Augmentation as built**, in order (B and H build the same list):\n")
    n = max(len(steps["B"]), len(steps["I"]))
    rows = [[k + 1, steps["B"][k] if k < len(steps["B"]) else "", steps["I"][k] if k < len(steps["I"]) else ""] for k in range(n)]
    add(table(["#", "B, H", "I"], rows, "rll"))
    add(figure("f02_recipe"))

    # 3 ------------------------------------------------------------------
    add("# 3 Data\n")
    pool = R["dataset"]["pool"]
    add(f"## 3.1 Pool and training rows\n")
    add(f"The cross-validation pool holds {n0(pool['images'])} photos of {n0(pool['infants'])} infants from three sources; "
        f"validation always scores all of them (minus {pool['duplicates_excluded']} exact duplicates). Training uses the "
        f"capped rows: {n0(V4['training_rows']['v3_kept'])} in v3, {n0(V4['training_rows']['v4'])} in v4 "
        f"({V4['training_infants']:,} infants). Cells: pool photos · v3 training rows → v4 training rows.\n")
    rows = []
    for s, ps in pool["by_source"].items():
        tb = V4["training_by_source"][s]
        rows.append([f"{SRC[s]} ({ps['infants']} infants)"] +
                    [f"{n0(ps['by_class'][c])} · {n0(tb['v3_kept'][c])} → {n0(tb['v4'][c])}" for c in CL] +
                    [f"{n0(ps['images'])} · {n0(sum(tb['v3_kept'].values()))} → {n0(sum(tb['v4'].values()))}"])
    tb = V4["training_by_class"]
    rows.append(["**all**"] + [f"{n0(sum(p['by_class'][c] for p in pool['by_source'].values()))} · {n0(tb['v3_kept'][c])} → {n0(tb['v4'][c])}" for c in CL]
                + [f"{n0(pool['images'])} · {n0(V4['training_rows']['v3_kept'])} → {n0(V4['training_rows']['v4'])}"])
    add(table(["source", *[CLN[c] for c in CL], "total"], rows))
    chg = V4["selection_changed_vs_v3"]
    add("Rows that changed identity between the pools (same caps, different frames): " +
        "; ".join(f"{SRC[s]}: {v['v3_kept_only']} v3-only, {v['v4_only']} v4-only" for s, v in chg.items()) +
        f". {V4['ostrava_quality']['v3_kept_removed']} of v3's Ostrava training rows failed the quality rule.\n")

    add("## 3.2 Ostrava frames removed by quality\n")
    oq, aq = V4["ostrava_quality"], AUD["ostrava_quality"]
    rows = [[CLN[c], n0(aq["by_class"][c]["frames"]), aq["by_class"][c]["infants"], oq["failing_by_class"][c],
             oq["removed_by_class"][c], pct(oq["removed_by_class"][c] / aq["by_class"][c]["frames"]) if aq["by_class"][c]["frames"] else "n/a",
             aq["by_class"][c]["by_reason"]["blur"], aq["by_class"][c]["by_reason"]["dark"], aq["by_class"][c]["by_reason"]["bright"]]
            for c in CL]
    rows.append(["**all**", n0(oq["frames"]), aq["infants"], oq["failing"], oq["removed_by_quality"],
                 pct(oq["removed_by_quality"] / oq["frames"]), aq["by_reason"]["blur"], aq["by_reason"]["dark"], aq["by_reason"]["bright"]])
    add(table(["class", "Ostrava frames", "infants", "failing", "removed", "share", "blur", "dark", "bright"], rows))
    rows = []
    for d, v in aq["by_device"].items():
        rows.append([DEVICE[d]] + [f"{v['by_class'][c]['failing']} / {n0(v['by_class'][c]['frames'])}" if c in v["by_class"] else "–" for c in CL]
                    + [f"{v['failing']} / {n0(v['frames'])}"])
    add("Failing / total frames per camera and class (the AP-ROP frames all come from one camera and "
        f"{aq['by_class']['arop']['infants']} infant):\n")
    add(table(["camera", *[CLN[c] for c in CL], "all"], rows))

    add("## 3.3 CV folds\n")
    rows = []
    for k in range(SC.NFOLD):
        f4 = V4["folds"][f"fold_{k}"]
        f3v = R["dataset"]["folds"][k]
        kt = [timing(a)[SEEDS[0]][f"fold_{k}"] for a in NEW]
        rows.append([f"CV {k}", n0(f3v["train_infants"]), n0(f3v["train_images"]), n0(sum(f4["train"].values()))] +
                    [n0(t["train_images"]) if t else "n/a" for t in kt] + [n0(sum(f4["val"].values()))] +
                    [f"{f4['train'][c]} / {f4['val'][c]}" for c in CL])
    add(table(["fold", "train infants", "v3 train", "v4 train", "B kernel", "H kernel", "I kernel", "val",
               *[f"{SHORT[c]} tr/val" for c in CL]], rows))
    add("v3/v4 train = rows in the fold files; kernel = rows the kernel reports training on (exact duplicates dropped at "
        "run time). Per-class cells: v4 train / validation photos.\n")

    add("## 3.4 Leave-one-source-out folds\n")
    rows = []
    for s in SCO["loso"]["B"]["sources"]:
        seen = []
        for a in NEW:
            L = SCO["loso"][a][s]
            sig = json.dumps(L["train_support"], sort_keys=True)
            hit = next((r for r in seen if r[0] == sig), None)
            if hit:
                hit[1].append(a)
                continue
            seen.append([sig, [a], L])
        for sig, arms_, L in seen:
            rows.append([SRC[s], ", ".join(arms_), n0(L["n_images"]), L["n_infants"]] + [n0(L["train_support"][c]) for c in CL] + [n0(sum(L["train_support"].values()))])
    add(table(["held out", "arms", "val photos", "val infants", *[f"{SHORT[c]} train" for c in CL], "train"], rows))
    add("Fold files: " + ", ".join(f"{SRC[s]} `{SCO['loso']['B'][s]['fold_file']}`" for s in SCO["loso"]["B"]["sources"]) + ". "
        "Validation photos exclude exact duplicates (" +
        ", ".join(f"{SRC[s]} {SCO['loso']['B'][s]['n_excluded_duplicates']}" for s in SCO["loso"]["B"]["sources"]) + "). "
        "Training columns: rows each class contributes to the LOSO model's training set (`train_support` in "
        "`v3_scores.json`). A zero means the held-out source is the only one with that class.\n")

    add("## 3.5 The new hospital (Multi-View)\n")
    dev = R["dataset"]["dev"]
    rows = [["photos", *[dev["images_by_class"][c] for c in CL], dev["images"]], ["eyes", *[dev["eyes_by_class"][c] for c in CL], dev["eyes"]]]
    add(f"Dev half: {dev['source']}; {dev['infants']} infants. Eye label = the most severe photo label; eye prediction = "
        "the most severe predicted class over the eye's photos.\n")
    add(table(["unit", *[CLN[c] for c in CL], "total"], rows))
    lk = R["dataset"]["locked"]
    add(f"Locked half: {lk['images']} photos, {lk['eyes']} eyes, {lk['infants']} infants ({lk['note']}). Not opened for v4.\n")

    add("## 3.6 Why 5 folds, 2 seeds, 20 epochs\n")
    add(f"*Verbatim from `{README}`.*\n")
    for p in readme_paragraphs(("**Why 5 folds", "**Why the folds", "**Why 2 seeds", "**Why 20 epochs", "**Why batch",
                                "**Why per-infant", "**Why validation", "**Leave-one-source-out")):
        add(p + "\n")

    # 4 ------------------------------------------------------------------
    add("# 4 Results\n")
    add("## 4.1 Headline, all arms\n")
    trans = [a for a in arms9 if R["arms"][a]["dev_status"].startswith("transductive")]
    rows = [[NAMES[a] + (" †" if a in trans else ""), f3(q[a]), ci3(tab[a]["ci95"]), f3(R["arms"][a]["cv_image"]["accuracy"]),
             f3(R["arms"][a]["cv_image"]["balanced_accuracy"]), f3(R["arms"][a]["cv_image"]["macro_f1"]),
             f3(R["arms"][a]["cv_image"]["any_rop_auc"]), f3(tab[a]["arop_binary"]["recall"])] for a in arms9]
    add(f"**CV, pooled out-of-fold, per photograph** ({SCO['contrasts']['cv_image']['n_images']:,} photos, both seeds):\n")
    add(table(["arm", "QWK5", "QWK5 95% CI", "accuracy", "balanced acc.", "macro-F1", "any-ROP AUC", "AP-ROP recall"], rows))
    rows = []
    for a in arms9:
        d = R["arms"][a]["dev_eye"]
        rows.append([NAMES[a] + (" †" if a in trans else ""), f3(d["qwk_5class"]), ci3(d["ci95"]["qwk_5class"]), f3(d["accuracy"]),
                     f3(d["balanced_accuracy"]), f3(d["macro_f1"]), f3(d["any_rop_auc"]), f3(d["any_rop_ap"])])
    add(f"**Multi-View dev, per eye** ({dev['eyes']} eyes; mean over {R['arms']['B']['dev_eye']['n_lists']} fold-seed models; descriptive):\n")
    add(table(["arm", "QWK5", "QWK5 95% CI", "accuracy", "balanced acc.", "macro-F1", "any-ROP AUC", "any-ROP AP"], rows))
    add(f"† {', '.join(trans)}: transductive (dev photos seen unlabelled during training). AP-ROP recall = the binary "
        "AP-ROP branch metric of the selection table (mean over seeds).\n")
    add(figure("f04_dev_eye"))

    add("## 4.2 Per stage: B, H, I\n")
    for lv, title in (("cv_image", "CV, per photograph"), ("dev_eye", "Multi-View dev, per eye")):
        rows = []
        for c in CL:
            for a in NEW:
                k = R["arms"][a][lv]["classes"][c]
                rows.append([CLN[c] if a == "B" else "", a, f3(k["precision"]), f3(k["recall"]), f3(k["specificity"]),
                             f3(k["f1"]), f3(k["auc_ovr"]), n0(k["support"])])
        add(f"**{title}:**\n")
        add(table(["stage", "arm", "precision", "recall", "specificity", "F1", "AUC (one-vs-rest)", "support"], rows, "llrrrrrr"))
    add(figure("f06_stage_recall"))
    add(figure("f07_confusion_dev_eye"))

    add("## 4.3 Per fold\n")
    pf = {a: {(p["seed"], p["fold"]): p for p in R["arms"][a]["per_fold"]} for a in NEW}
    rows = []
    for s in SEEDS:
        for k in range(SC.NFOLD):
            rows.append([s, k] + [f3(pf[a][(s, k)]["qwk_5class"]) for a in NEW] + [f3(pf[a][(s, k)]["macro_f1"]) for a in NEW]
                        + [f3(pf[a][(s, k)]["accuracy"]) for a in NEW])
    for stat, fn in (("mean", np.mean), ("SD", lambda v: np.std(v, ddof=1))):
        rows.append([stat, ""] + [f3(fn([pf[a][kk][key] for kk in pf[a]])) for key in ("qwk_5class", "macro_f1", "accuracy") for a in NEW])
    add("Per seed × CV fold, duplicates excluded (`report_data.json` → `per_fold`):\n")
    add(table(["seed", "fold", "QWK5 B", "QWK5 H", "QWK5 I", "mF1 B", "mF1 H", "mF1 I", "acc B", "acc H", "acc I"], rows, "llrrrrrrrrr"))
    add(f"**AP-ROP by fold.** Kernel-side AP-ROP recall (`metrics.json`, 6-class, before duplicate exclusion), seed "
        f"{SEEDS[0]} / {SEEDS[1]}, with the source of the AP-ROP validation photos (v4 fold files):\n")
    rows = []
    for k in range(SC.NFOLD):
        src = ", ".join(f"{SRC[s]} {n} ({i} inf.)" for s, (n, i) in comp[k].items())
        rows.append([f"CV {k}", src] + [" / ".join(f"{recB[a][(s, k)][0]:.3f}" for s in SEEDS) for a in NEW])
    rows.append(["CV 0 pooled", ""] + [f3(rec0[a]) for a in NEW])
    rows.append([f"CV 1–{SC.NFOLD - 1} pooled", ""] + [f3(rec14[a]) for a in NEW])
    add(table(["fold", "AP-ROP val photos by source", "B", "H", "I"], rows, "llrrr"))
    tr0 = comp[("train", 0)]
    sd0 = {a: abs(recB[a][(SEEDS[0], 0)][0] - recB[a][(SEEDS[1], 0)][0]) for a in NEW}
    add(f"Fold 0 holds {ost0[0]} AP-ROP photos of {ost0[1]} Ostrava infant, scored by models trained on "
        f"{tr0['ostrava']} Ostrava and {tr0['ropvl']} ROP-VL AP-ROP photos; there B reaches {f3(rec0['B'])}, H "
        f"{f3(rec0['H'])} and I {f3(rec0['I'])}, and the two seeds of one arm differ by up to {f3(max(sd0.values()))} "
        f"({max(sd0, key=sd0.get)}). On folds 1–{SC.NFOLD - 1} (ROP-VL AP-ROP only) the arms sit at {f3(rec14['B'])}, "
        f"{f3(rec14['H'])} and {f3(rec14['I'])}.\n")

    add("## 4.4 Leave-one-source-out\n")
    rows = []
    for a in NEW:
        for s in SCO["loso"][a]["sources"]:
            L = SCO["loso"][a][s]
            rows.append([a, SCO["loso"][a]["pool"], SRC[s], n0(L["n_images"]), L["n_infants"], f3(L["qwk_5class"]),
                         ci3(L["ci95"]["qwk_5class"]), f3(L["macro_f1_present"]), ci3(L["ci95"]["macro_f1_present"]),
                         f3(L["accuracy"]), L["train_support"]["stage_4_5"]])
        rows.append([a, "", "**mean**", "", "", f"**{f3(SCO['loso'][a]['mean_qwk_5class'])}**", "", "", "", "", ""])
    add(table(["arm", "pool", "held out", "photos", "infants", "QWK5", "95% CI", "macro-F1", "95% CI", "accuracy", "S4/5 train"], rows,
              "lllrrrrrrrr"))
    rows = []
    for s in SCO["loso"]["B"]["sources"]:
        for a in NEW:
            pr = SCO["loso"][a][s]["per_class_recall_mean"]
            rows.append([SRC[s] if a == "B" else "", a] + [f3(pr[c]) if c in pr else "–" for c in CL])
    add("Recall per class on the held-out source (mean over seeds; – = class absent there):\n")
    add(table(["held out", "arm", *[CLN[c] for c in CL]], rows, "llrrrrrr"))
    add(figure("f08_loso"))

    add("## 4.5 Paired contrasts\n")
    rows = []
    for p in ("H-B", "I-B", "I-H", "A-B", "A-C"):
        for blk, lab in (("cv_image", "CV photo"), ("dev_eye", "dev eye")):
            x = SCO["contrasts"][blk]["pairs"][p]
            qq, mm = x["qwk_5class"], x["macro_f1_present"]
            rows.append([p.replace("-", f" {MINUS} ") if blk == "cv_image" else "", lab, f3(qq["a"]), f3(qq["b"]), s3(qq["diff"]),
                         ci3(qq["diff_ci"]), s3(mm["diff"]), ci3(mm["diff_ci"]), n0(qq["n_infants"])])
    add(table(["pair", "level", "QWK5 a", "QWK5 b", "Δ QWK5", "95% CI", "Δ macro-F1", "95% CI", "infants"], rows, "llrrrrrrr"))
    add(f"CV: {SCO['contrasts']['cv_image']['status']}. Dev: {SCO['contrasts']['dev_eye']['status']}. A − B and A − C are "
        "the v3 secondary/primary pairs, shown for scale.\n")
    add(figure("f05_contrasts"))

    add("## 4.6 Operating point and calibration\n")
    rule = SCO["arms"]["B"]["seeds"][str(SEEDS[0])]["operating_point"]["rule"]
    dop = SCO["arms"]["B"]["seeds"][str(SEEDS[0])]["dev"]["operating_point_at_cv_threshold"][0]
    rows = []
    for a in NEW:
        for s in SEEDS:
            o = R["arms"][a]["operating_point"][str(s)]
            rows.append([a, s, f3(o["threshold"]), f3(o["cv"]["any_rop"]["sensitivity"]), f3(o["cv"]["any_rop"]["specificity"]),
                         f3(o["dev_eye_mean"]["any_rop"]), f3(o["dev_eye_mean"]["any_rop_spec"]),
                         f3(o["dev_eye_mean"]["severe_vs_all_nonsevere"]), f3(o["dev_eye_mean"]["severe_vs_all_nonsevere_spec"])])
    add(f"Threshold rule: {rule}. CV units: {R['arms']['B']['operating_point'][str(SEEDS[0])]['cv']['any_rop']['n_pos']:,} "
        f"any-ROP, {R['arms']['B']['operating_point'][str(SEEDS[0])]['cv']['any_rop']['n_neg']:,} normal. Dev at that threshold, "
        f"mean over the seed's fold models: {dop['any_rop']['n_pos']} any-ROP and {dop['any_rop']['n_neg']} normal eyes; "
        f"severe = stage 3 or worse, {dop['severe_vs_all_nonsevere']['n_pos']} eyes vs {dop['severe_vs_all_nonsevere']['n_neg']} others.\n")
    add(table(["arm", "seed", "threshold", "CV sens.", "CV spec.", "dev any-ROP sens.", "dev any-ROP spec.", "dev severe sens.", "dev severe spec."],
              rows, "llrrrrrrr"))
    rows = []
    for a in NEW:
        rel = R["arms"][a]["reliability_dev_eye"]
        for s in SEEDS:
            t = SCO["arms"][a]["seeds"][str(s)]["temperature"]
            cu = t["cv_unit_driving_photo"]
            rows.append([a, s, f3(t["T"]), f3(cu["before"]["ece"]), f3(cu["after"]["ece"]), f3(cu["before"]["nll"]), f3(cu["after"]["nll"]),
                         f3(rel["uncalibrated"]["ece"]) if s == SEEDS[0] else "", f3(rel["cv_temperature"]["ece"]) if s == SEEDS[0] else ""])
    add(f"Temperature: {SCO['arms']['B']['seeds'][str(SEEDS[0])]['temperature']['fitted_on']}. Dev-eye ECE pools both seeds "
        "(10 bins).\n")
    add(table(["arm", "seed", "T", "CV ECE before", "CV ECE after", "CV NLL before", "CV NLL after", "dev-eye ECE as trained", "dev-eye ECE after T"],
              rows, "llrrrrrrr"))
    add(figure("f10_any_rop_dev_eye"))
    add(figure("f11_reliability_dev_eye"))

    # 5 ------------------------------------------------------------------
    add("# 5 Training behaviour\n")
    add(f"Per-epoch histories from `results/rop/v3/runs/<run dir>/<fold>/metrics.json`; full per-epoch tables in "
        f"[the run log]({Path(RUN_LOG).name}). All arms keep the last-epoch weights (`eval.early_stopping: "
        f"{g(CFG['B']['eval']['early_stopping'])}`), so the best-epoch figures below describe the curves, not the models.\n")
    H = {(a, s, f): history(a, s, f)["history"] for a in NEW for s in SEEDS for f in FOLDS}
    rows = []
    for f in FOLDS:
        rows.append([FOLD_LABEL[f]] + [" / ".join(f3(H[(a, s, f)]["train_loss"][-1]) for s in SEEDS) for a in NEW]
                    + [" / ".join(str(int(np.argmax(H[(a, s, f)]["val_macro_f1"])) + 1) for s in SEEDS) for a in NEW])
    add(f"Final-epoch training loss and the epoch of the best held-out macro-F1 (seed {SEEDS[0]} / {SEEDS[1]}):\n")
    add(table(["fold", "loss B", "loss H", "loss I", "best epoch B", "best epoch H", "best epoch I"], rows))
    ep = len(H[("B", SEEDS[0], FOLDS[0])]["train_loss"])
    rows = []
    for a in NEW:
        cv = [(s, f) for s in SEEDS for f in FOLDS[:SC.NFOLD]]
        fin = [H[(a, s, f)]["val_macro_f1"][-1] for s, f in cv]
        best = [max(H[(a, s, f)]["val_macro_f1"]) for s, f in cv]
        late = sum(int(np.argmax(H[(a, s, f)]["val_macro_f1"])) + 1 > ep - 3 for s, f in cv)
        rows.append([a, f3(np.mean([H[(a, s, f)]["train_loss"][-1] for s, f in cv])), f3(ls_floor(CFG[a]["train"]["label_smoothing"], len(CL))),
                     f3(np.mean([H[(a, s, f)]["train_loss"][0] for s, f in cv])), f3(np.mean(fin)), f3(np.mean(best)),
                     f3(np.mean(np.array(best) - np.array(fin))), f"{late} of {len(cv)}"])
    add(f"CV folds, both seeds ({2 * SC.NFOLD} runs per arm):\n")
    add(table(["arm", "final loss", "smoothing floor", "epoch-1 loss", "final val mF1", "best val mF1", "best − final", f"best in last 3 of {ep}"], rows))
    late = {a: sum(int(np.argmax(H[(a, s_, f)]["val_macro_f1"])) + 1 > ep - 3 for s_ in SEEDS for f in FOLDS[:SC.NFOLD]) for a in NEW}
    e1 = {a: np.mean([H[(a, s_, f)]["val_macro_f1"][0] for s_ in SEEDS for f in FOLDS[:SC.NFOLD]]) for a in NEW}
    add(f"I's held-out macro-F1 peaks in the last three epochs in {late['I']} of {2 * SC.NFOLD} CV runs (B {late['B']}, "
        f"H {late['H']}), and its epoch-1 value is {f3(e1['I'])} against B's {f3(e1['B'])}: validation runs on the EMA weights, "
        "which lag the student and still carry most of the initial weights early on (table below), so I's curve rises "
        "later and is still climbing when training stops. B and H usually peak earlier; all three arms keep the last "
        "epoch, whatever the peak.\n")
    add("The smoothing floor is the cross-entropy of a prediction equal to the smoothed target with unweighted classes "
        f"(ε = label smoothing, K = {len(CL)}); I's floor is higher because ε is higher, so I's loss is not comparable "
        "with B's or H's. Validation macro-F1 is comparable.\n")
    tI = timing("I")
    bs, epo, decay = CFG["I"]["train"]["batch_size"], CFG["I"]["train"]["epochs"], CFG["I"]["train"]["ema_decay"]
    rows = []
    for f in FOLDS:
        r = tI[SEEDS[0]][f]
        if not r:
            continue
        spe = math.ceil(r["train_images"] / bs)
        rows.append([FOLD_LABEL[f], n0(r["train_images"]), spe, n0(spe * epo), pct(decay ** spe), pct(decay ** (spe * epo))])
    add(f"**EMA residual (arm I).** Protocol, Amendment 5 pre-run note: *{protocol_block('4. *EMA start.*', '###')}* "
        f"Computed below from the kernel-reported training rows, batch {bs}, {epo} epochs, decay {g(decay)} (one EMA "
        "update per optimiser step; the loader keeps the last partial batch):\n")
    add(table(["fold", "train photos", "steps/epoch", "steps", "initial weights left after epoch 1", "after the last epoch"], rows))
    add(figure("f09_training_curves"))

    # 6 ------------------------------------------------------------------
    add("# 6 Cost\n")
    rows = []
    gpu_h = {}
    for a in NEW:
        T = timing(a)
        allr = [T[s][f] for s in SEEDS for f in FOLDS if T[s][f]]
        common = [T[s][f] for s in SEEDS for f in FOLDS[:SC.NFOLD + 1] if T[s][f]]
        gpu_h[a] = sum(r["minutes"] for r in allr) / 60
        rows.append([a, len(allr), f"{sum(r['minutes'] for r in allr) / 60:.1f}", f"{sum(r['minutes'] for r in common) / 60:.1f}",
                     f"{np.mean([T[s][f]['minutes'] for s in SEEDS for f in FOLDS[:SC.NFOLD]]):.1f}",
                     f"{np.mean([r['img_per_s'] for r in allr]):.1f}", ", ".join(sorted({r['gpu'] for r in allr}))])
    add(table(["arm", "fold-seed runs timed", "GPU-hours (timed)", f"GPU-hours, CV + site ({SC.NFOLD + 1} folds × {len(SEEDS)})",
               "minutes per CV fold", "img/s", "GPU"], rows))
    rows = []
    for f in FOLDS:
        rows.append([FOLD_LABEL[f]] + [" / ".join(f"{timing(a)[s][f]['minutes']:.1f}" if timing(a)[s][f] else "n/a" for s in SEEDS) for a in NEW])
    add(f"Minutes per fold (seed {SEEDS[0]} / {SEEDS[1]}); n/a = not in `training.timing`:\n")
    add(table(["fold", "B", "H", "I"], rows))
    add(f"Budget set in advance (Amendment 5d, verbatim): *{protocol_block('d. **Budget.**', '### Amendment 5a')}* "
        f"Timed H + I: {gpu_h['H'] + gpu_h['I']:.1f} GPU-hours. I's per-fold time is "
        f"{np.mean([timing('I')[s][f]['minutes'] for s in SEEDS for f in FOLDS[:SC.NFOLD]]) / np.mean([timing('H')[s][f]['minutes'] for s in SEEDS for f in FOLDS[:SC.NFOLD]]):.3f}× H's on the CV folds.\n")
    add(figure("f12_timing"))

    # 7 ------------------------------------------------------------------
    add("# 7 Interpretation and limitations\n")
    amin, amax = min(q, key=q.get), max(q, key=q.get)
    dmin, dmax = min(qd, key=qd.get), max(qd, key=qd.get)
    s1 = {lv: {a: R["arms"][a][lv]["classes"]["stage_1"]["recall"] for a in NEW} for lv in ("cv_image", "dev_eye")}
    worse_T = all(R["arms"][a]["reliability_dev_eye"]["cv_temperature"]["ece"] > R["arms"][a]["reliability_dev_eye"]["uncalibrated"]["ece"]
                  for a in NEW)
    add(f"- **Nine arms, one plateau.** CV QWK5 runs from {f3(q[amin])} ({amin}) to {f3(q[amax])} ({amax}), a span of "
        f"{f3(q[amax] - q[amin])}, while a single arm's 95% CI is {f3(min(widths.values()))}–{f3(max(widths.values()))} wide. "
        f"Dev-eye QWK5 runs from {f3(qd[dmin])} ({dmin}) to {f3(qd[dmax])} ({dmax}), a span of {f3(qd[dmax] - qd[dmin])}. "
        "Backbone, resolution, adaptation, exam-level loss, data cleaning and the v4 recipe all land inside a few "
        "hundredths of each other.")
    add(f"- **Paired intervals are what can separate arms.** Because both arms are resampled on the same infants, the "
        f"I − B CV interval is {f3(pi['diff_ci'][1] - pi['diff_ci'][0])} wide against marginal CIs of about "
        f"{f3(widths['B'])}. Even so, no v4 interval reaches the +{m:.3f} bar.")
    add(f"- **I's dev gain excludes zero but is descriptive.** I − B on dev eyes is {s3(di['diff'])} {ci3(di['diff_ci'])} and "
        f"I − H {s3(c_dev['I-H']['qwk_5class']['diff'])} {ci3(c_dev['I-H']['qwk_5class']['diff_ci'])}. The dev half "
        f"({dev['eyes']} eyes, {dev['infants']} infants) has been read for all nine arms, so it is one of many looks, and "
        "the protocol fixed CV as the selection data before any v4 run. The protocol's minimum detectable difference "
        f"for the locked half, verbatim: *{protocol_block('Minimum detectable difference at 80', '## Deduplication')}*")
    add(f"- **AP-ROP is one infant deep in CV.** AP-ROP recall drops from {f3(arop_bin['B'])} (B) to {f3(arop_bin['H'])} "
        f"(H) and {f3(arop_bin['I'])} (I), and the drop is in fold 0 ({f3(rec0['B'])} → {f3(rec0['H'])} → {f3(rec0['I'])}), "
        f"whose {ost0[0]} Ostrava AP-ROP photos belong to {ost0[1]} infant. The fold-0 models train on "
        f"{comp[('train', 0)]['ostrava']} Ostrava AP-ROP photos in every arm, and the pools hold the same number of AP-ROP "
        f"training rows ({tb['v3_kept']['arop']} → {tb['v4']['arop']}), so the fall is not lost AP-ROP training data: it is "
        "how the rest of the v4 pool (sharper Ostrava frames of other infants and classes) shifts the models' reading of "
        "one infant's near-identical frames, and the seed-to-seed spread on that fold is of the same size. On the ROP-VL "
        f"AP-ROP folds the arms are within {f3(max(rec14.values()) - min(rec14.values()))} of each other. Dev has only "
        f"{dev['eyes_by_class']['arop']} AP-ROP eyes.")
    add(f"- **Stage 1.** Dev-eye stage-1 recall is unchanged across B, H and I ({', '.join(f3(s1['dev_eye'][a]) for a in NEW)}). "
        f"On CV, H matches B ({f3(s1['cv_image']['H'])} vs {f3(s1['cv_image']['B'])}) and I is lower "
        f"({f3(s1['cv_image']['I'])}); the v4 pool has {tb['v3_kept']['stage_1'] - tb['v4']['stage_1']} fewer stage-1 training "
        "rows (" + ", ".join(f"{SRC[s_]} {V4['training_by_source'][s_]['v3_kept']['stage_1'] - V4['training_by_source'][s_]['v4']['stage_1']}"
                             for s_ in SRC if V4['training_by_source'][s_]['v3_kept']['stage_1'] != V4['training_by_source'][s_]['v4']['stage_1'])
        + "); H shares that pool and matches B, so I's lower CV stage-1 recall goes with the recipe, not the pool.")
    add(f"- **Calibration.** I is better calibrated as trained (dev-eye ECE {f3(R['arms']['I']['reliability_dev_eye']['uncalibrated']['ece'])} "
        f"vs B {f3(R['arms']['B']['reliability_dev_eye']['uncalibrated']['ece'])}); its CV temperature is further below 1 "
        f"than B's ({' / '.join(f3(R['arms']['I']['temperature'][str(s)]) for s in SEEDS)} vs "
        f"{' / '.join(f3(R['arms']['B']['temperature'][str(s)]) for s in SEEDS)}), i.e. I is more under-confident on CV, "
        "consistent with EMA plus heavier label smoothing. " +
        (f"The CV-fitted temperature makes each of B, H and I worse on dev (ECE after T: " +
         ", ".join(f"{a} {f3(R['arms'][a]['reliability_dev_eye']['cv_temperature']['ece'])}" for a in NEW) +
         "), so it does not transfer to the new hospital."
         if worse_T else "The CV-fitted temperature does not improve every arm on dev."))
    mdd_txt = protocol_block('Minimum detectable difference at 80', '## Deduplication')
    mdd = float(re.search(r"≈ ([\d.]+) eye QWK", mdd_txt).group(1))
    add(f"- **What would move the numbers.** Not another recipe: nine arms agree within the CI. The bottlenecks are in the "
        f"data: AP-ROP rests on {aq['by_class']['arop']['infants']} Ostrava infant plus ROP-VL; stage 4/5 exists "
        f"only in ROP-VL ({tb['v4']['stage_4_5']} training photos, {V4['training_infants_by_class']['stage_4_5']} infants), so the "
        f"ROP-VL LOSO fold scores it with zero training rows; Ostrava LOSO CIs span {ci3(SCO['loso']['B']['ostrava']['ci95']['qwk_5class'])} "
        f"for B because {SCO['loso']['B']['ostrava']['n_infants']} infants supply {n0(SCO['loso']['B']['ostrava']['n_images'])} photos. "
        f"More infants with stage 4/5 and AP-ROP (data, not recipe) and clinician review of the {nd['different_label']} "
        "different-label near-duplicate pairs are the levers. " +
        (f"I's dev gain ({s3(di['diff'])}) lies below the protocol's minimum detectable difference ({mdd:g} eye QWK), so "
         "the locked half (opened once, for arms A–I together, after supervisor approval) cannot confirm a gain of that "
         "size either." if di["diff"] < mdd else "The locked half is opened once, for arms A–I together, after supervisor approval."))
    add(f"- **Scope.** Dev and LOSO are descriptive; F used dev photos without labels during training; B's LOSO folds "
        "ran in a separate kernel after the v3 round, on the v3 pool.\n")

    # 8 ------------------------------------------------------------------
    add("# 8 Reproducibility\n")
    add("Regenerate this report (figures and Markdown are deterministic), then the PDF and DOCX:\n")
    add("```\n.venv/bin/python scripts/rop_v4_model_report.py\n.venv/bin/python scripts/rop_v4_model_report.py --render\n```\n")
    add("`--render` runs pandoc from `docs/v4/`, so the figure paths resolve; font size, paper size and link colours "
        "come from the front matter:\n")
    add("```\ncd docs/v4\npandoc V4_MODEL_REPORT.md -o V4_MODEL_REPORT.pdf --pdf-engine=tectonic \\\n"
        "    -V geometry:margin=2cm -V mainfont=\"Times New Roman\"\npandoc V4_MODEL_REPORT.md -o V4_MODEL_REPORT.docx\n```\n")
    rows = []
    for rel in [FILES["report_data"], FILES["scores"], FILES["v4_summary"], FILES["splits_summary"], FILES["audit"],
                FILES["thresholds"], *CFG_FILES.values(), PROTOCOL, README, RUN_LOG,
                "scripts/rop_v3_score.py", "scripts/rop_v3_report_data.py", "scripts/build_rop_v4_pool.py",
                "scripts/rop_v4_audit.py", "scripts/collect_v3_run.py", "scripts/push_v3_kernels.py", "scripts/rop_v4_runlog.py",
                "scripts/rop_v3_figures.py"]:
        h, d, _ = git_last(rel)
        rows.append([f"`{rel}`", f"`{h}`", d])
    add("Last commit touching each input (`git log -1 -- <file>`):\n")
    add(table(["file", "commit", "date"], rows, "lll"))
    rows, krows = [], []
    for a in NEW:
        for s in SEEDS:
            sm = summary(a, s)
            ks = {}
            for fk, fv in sm["folds"].items():
                ks.setdefault(fv.get("kernel", "n/a"), []).append(fk.replace("fold", ""))
            rd = f"results/rop/v3/runs/{RD.RUN_DIR[a]}_s{s}"
            h, d, _ = git_last(rd)
            rows.append([a, s, f"`{rd}`", f"`{h}`", d])
            krows += [[a, s, f"`{k}`", ", ".join(v)] for k, v in ks.items()]
    add("Run directories (`git log -1 -- <dir>`):\n")
    add(table(["arm", "seed", "run dir", "commit", "date"], rows, "lllll"))
    add("Kaggle kernels (from each run's `summary.json`):\n")
    add(table(["arm", "seed", "kernel", "folds"], krows, "llll"))
    heads = [ln.lstrip("#").strip() for ln in _guard(ROOT / PROTOCOL).read_text().splitlines()
             if ln.startswith("#") and "Amendment 5" in ln]
    add(f"Protocol sections that govern v4 (`{PROTOCOL}`): " + "; ".join(heads) + ". Also: Selection rule, Units, "
        "Descriptive endpoints, Minimum detectable difference.\n")

    add("# Appendix\n")
    add(f"- Run log, every arm, seed, fold and epoch: [`{RUN_LOG}`]({Path(RUN_LOG).name}) (sections 3 and 4).")
    add(f"- Data audit: [`{AUDIT_MD}`](../../{AUDIT_MD}).")
    add(f"- Why the numbers: [`{README}`]({Path(README).name}).")
    add(f"- Figures: `docs/v4/figures/` ({len(FIGS)} figures, PDF and 300-dpi PNG).")
    return "\n".join(OUT) + "\n"


def render():
    for ext, extra in (("pdf", ["--pdf-engine=tectonic", "-V", "geometry:margin=2cm", "-V", "mainfont=Times New Roman"]),
                       ("docx", [])):
        cmd = ["pandoc", MD.name, "-o", f"{MD.stem}.{ext}", *extra]
        print(" ".join(cmd))
        subprocess.run(cmd, cwd=DOC, check=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--render", action="store_true", help="also convert the Markdown to PDF and DOCX with pandoc")
    args = ap.parse_args()
    print("->", FIG.relative_to(ROOT))
    fig_data(); fig_recipe(); fig_selection(); fig_dev(); fig_contrasts(); fig_stage_recall()
    fig_confusion(); fig_loso(); fig_training(); fig_any_rop(); fig_reliability(); fig_timing()
    MD.write_text(report())
    print("->", MD.relative_to(ROOT))
    if args.render:
        render()


if __name__ == "__main__":
    sys.exit(main())
