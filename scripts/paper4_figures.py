"""Paper 4 (the development journey, classical ML -> CNN -> ViT): figures, tables and prose numbers, from files.

    .venv/bin/python scripts/paper4_figures.py           # docs/paper/figs4/{fig_*.pdf,.png, tab_*.tex, numbers4.tex}
    .venv/bin/python scripts/paper4_figures.py --docx    # also docs/paper/RetinAI_Paper4_Journey.docx

Reuses scripts/paper3_figures.py (its parsers, macros and the figs3/ tables the paper inputs) and adds what the
journey paper needs: the acquisition cascade and data-source table (results/rop/inclusion_flow.json, the prepared
manifest and splits), the shortcut audit (results/rop/staging_shortcut_audit.json), the human-agreement ceiling
(results/rop/kappa_unweighted_vs_human.json), the generation timeline (docs/paper/MODEL_HISTORY.md), a preprocessing
strip and Grad-CAM examples from the served ensemble on development-half photographs only, and the webapp screenshot.
No file named test_locked or locked_eval is read; no locked-half photograph is opened. numbers4.tex holds every macro
of paper 3 plus the ones added here, so paper 4 inputs one numbers file.
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from scripts import paper3_figures as P3  # noqa: E402  parsers, macros, shared style

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

PAPER = ROOT / "docs/paper"
FIG = PAPER / "figs4"
TEX = PAPER / "paper4_journey.tex"
DOCX_OUT = PAPER / "RetinAI_Paper4_Journey.docx"
SRC = {"flow": "results/rop/inclusion_flow.json", "audit": "results/rop/staging_shortcut_audit.json",
       "human": "results/rop/kappa_unweighted_vs_human.json", "manifest": "results/prepared/rop_staging/manifest.csv",
       "splits": "results/prepared/rop_staging/splits.csv", "dev": "results/prepared/rop_staging/test_dev.csv",
       "provenance": "data/rop/PROVENANCE.json", "webapp_shot": "docs/paper/figs4/fig_webapp.png"}
DEVICE = {"clarity_retcam3": "Clarity RetCam 3", "phoenix_icon": "Phoenix ICON", "retcam_envision": "RetCam Envision"}
COHORT = {"ostrava": ("Ostrava", "University Hospital Ostrava, CZ", "examination session", "CC0"),
          "ropvl": ("ROP-VL", "Shenzhen Eye Hospital, CN", "photograph (eye-exam id)", "CC BY 4.0"),
          "shenzhen": ("Shenzhen", "Shenzhen Eye Hospital, CN", "photograph", "CC BY 4.0"),
          "multiview": ("Multi-View", "undisclosed (dataset card)", "eye (3 views)", "CC BY-NC 4.0")}
mac, f3, n0, esc, ci = P3.mac, P3.f3, P3.n0, P3.esc, P3.ci
GREY, INK = P3.GREY, P3.INK


def jload(rel):
    return P3.jload(rel)


def tabular(name, spec, header, rows):
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
    print("  ", name)


# ------------------------------------------------------------------ numbers
def numbers4():
    flow, audit, human = jload(SRC["flow"]), jload(SRC["audit"]), jload(SRC["human"])
    mac("acqImages", n0(flow["layer_0_acquired"]["images"]))
    mac("acqSources", str(flow["layer_0_acquired"]["sources"]))
    l1, l2 = flow["layer_1_stage_labelled_sources"], flow["layer_2_six_class_eligible"]
    mac("acqExcluded", n0(l1["excluded"]["total"]))
    mac("stageLabelled", n0(l1["kept"]["images"]))
    mac("stageInfants", n0(l1["kept"]["infants"]))
    mac("sixElig", n0(l2["kept"]["images"]))
    mac("sixInfants", n0(l2["kept"]["infants"]))
    ex = l2["excluded"]["by_reason"]
    mac("exclOther", n0(ex["other_pathology"]))
    mac("exclTreated", n0(ex["treated"]))
    mac("exclStageZero", n0(ex["stage_0"]))
    mac("exclSix", n0(l2["excluded"]["total"]))
    mac("trainEligible", n0(P3.V4["training_rows"]["v3_kept"]))
    mac("vfourTrain", n0(P3.V4["training_rows"]["v4"]))
    mac("poolValOnly", n0(P3.R["dataset"]["pool"]["images"] - P3.V4["training_rows"]["v3_kept"]))
    caps = P3.V4["caps"]
    mac("capNormal", str(caps["normal"]))
    mac("capOther", str(caps["_default"]))
    m = pd.read_csv(ROOT / SRC["manifest"])
    for s, g in m.groupby("source"):
        key = s.capitalize()
        mac("raw" + key, n0(len(g)))
        mac("raw" + key + "Inf", n0(g.patient_id.nunique()))
    pool = P3.R["dataset"]["pool"]["by_source"]
    for s in pool:
        mac("pool" + s.capitalize(), n0(pool[s]["images"]))
        mac("pool" + s.capitalize() + "Inf", n0(pool[s]["infants"]))
        mac("perInfant" + s.capitalize(), f"{pool[s]['images'] / pool[s]['infants']:.1f}")
    mac("siteDecodable", f3(audit["G3_site_decodability"]["balanced_accuracy"]))
    mac("metaFloorCV", f3(audit["G4_metadata_baseline"]["bar_cv_macro_f1"]))
    mac("metaFloorDev", f3(audit["G4_metadata_baseline"]["bar_dev_macro_f1"]))
    mac("dupExact", str(audit["G2_duplicate_leakage"]["exact_md5"]))
    mac("dupNear", str(audit["G2_duplicate_leakage"]["near_dupes"]))
    hb = human["human_benchmark"]
    mac("humanInterKappa", f"{hb['inter_expert_stage_fleiss_kappa']:.2f}")
    mac("humanIntraKappa", f"{hb['intra_expert_stage_cohen_kappa']:.2f}")
    mac("humanDisagree", f"{hb['stage_disagreement_rate'] * 100:.0f}\\%")
    mac("vTwoKappaU", f3(human["ours"]["seed_42"]["kappa_unweighted"]))
    B = P3.SCO["arms"]["B"]["seeds"]["42"]
    mac("BkappaU", f3(B["image"]["kappa_unweighted"]))
    mac("BdevKappaU", f3(B["dev"]["eye"]["kappa_unweighted"]))
    # earlier generations, as paper 3's at-a-glance table reads them
    v1 = P3.text(P3.SRC_FILES["v1"]).split("===== ROP TEST RESULTS =====")[1].split("=====")[0]
    mac("vOneAcc", f3(float(re.search(r"Accuracy: ([\d.]+)%", v1).group(1)) / 100))
    mac("vOneFone", f3(float(re.search(r"^\s*ROP Detected\s+[\d.]+\s+[\d.]+\s+([\d.]+)", v1, re.M).group(1))))
    rm = jload(P3.SRC_FILES["resnet_metrics"])
    mac("sweepAUC", f3(rm["test"]["auc"]))
    mac("sweepValAUC", f3(rm["val"]["auc"]))
    q = {(r["dataset"], r["arm"], r["seed"], r["unit"]): r for r in P3.csv_rows(P3.SRC_FILES["v2_qwk"])}
    s0 = P3.SEEDS[0]
    mac("vTwoCvQsix", f3(float(q[("cv_pooled_oof", "structured", s0, "image")]["qwk_6class"])))
    mac("vTwoCvAcc", f3(float(q[("cv_pooled_oof", "structured", s0, "image")]["accuracy"])))
    mac("vTwoLockedImgQ", f3(float(q[("multiview_locked", "structured", s0, "image")]["qwk_5class_drop"])))
    mac("vTwoLockedEyeQ", f3(float(q[("multiview_locked", "structured", s0, "eye")]["qwk_5class_drop"])))
    mac("vTwoLockedEyeAcc", f3(float(q[("multiview_locked", "structured", s0, "eye")]["accuracy"])))
    gaps = [float(q[("multiview_locked", h, s, "eye")]["qwk_5class_drop"]) - float(q[("multiview_locked", h, s, "image")]["qwk_5class_drop"])
            for h in ("structured", "flat") for s in P3.SEEDS]
    mac("vTwoGapMin", f"{min(gaps):.2f}")
    mac("vTwoGapMax", f"{max(gaps):.2f}")
    hdr, body = P3.md_tables(P3.SRC_FILES["sweep"])[0]
    ix = {h: i for i, h in enumerate(hdr)}
    mac("sweepSelAUC", f3(max(float(r[ix["auc"]]) for r in body)))
    mac("sweepArchs", str(len(body)))
    mac("mateMLReportAcc", f3(P3.MATE["ml"]["best_report"]["accuracy"]))
    mac("BkappaUpooled", f3(P3.R["arms"]["B"]["cv_image"]["kappa_unweighted"]))
    mac("BdevKappaUpooled", f3(P3.R["arms"]["B"]["dev_eye"]["kappa_unweighted"]))
    # the served arm's screening behaviour and per-class recall (seed 42 = the served checkpoints)
    op = P3.R["arms"]["B"]["operating_point"]["42"]
    mac("BthrPct", f"{op['cv']['threshold'] * 100:.2f}\\%")
    mac("BcvSens", f3(op["cv"]["any_rop"]["sensitivity"]))
    mac("BcvSpec", f3(op["cv"]["any_rop"]["specificity"]))
    mac("BdevSens", f3(op["dev_eye_mean"]["any_rop"]))
    mac("BdevSpec", f3(op["dev_eye_mean"]["any_rop_spec"]))
    mac("BdevSevereSens", f3(op["dev_eye_mean"]["severe_vs_all_nonsevere"]))
    rel = P3.R["arms"]["B"]["reliability_dev_eye"]
    mac("BdevECE", f3(rel["uncalibrated"]["ece"]))
    dv, cv = P3.R["arms"]["B"]["dev_eye"], P3.R["arms"]["B"]["cv_image"]
    KEY = {"normal": "Normal", "stage_1": "StageOne", "stage_2": "StageTwo", "stage_3": "StageThree",
           "stage_4_5": "StageFourFive", "arop": "Arop"}
    for c in P3.CL:
        mac("BdevRec" + KEY[c], f3(dv["classes"][c]["recall"]))
        mac("BcvRec" + KEY[c], f3(cv["classes"][c]["recall"]))
    mac("BdevAcc", f3(dv["accuracy"]))
    mac("BdevMF", f3(dv["macro_f1"]))
    mac("BdevAUC", f3(dv["any_rop_auc"]))
    mac("BcvAcc", f3(cv["accuracy"]))
    mac("BcvAUC", f3(cv["any_rop_auc"]))
    tim = P3.R["arms"]["B"]["training"].get("timing", {})
    mins = [r.get("minutes") for rows in tim.values() for r in rows if isinstance(r, dict) and r.get("minutes")]
    if mins:
        mac("BfoldMinutes", f"{np.mean(mins):.0f}")
    mac("nGenerations", "4")
    # paired contrasts (first arm minus second, same infant resample), as prose macros
    for lv, tag in (("cv_image", "cv"), ("dev_eye", "dev")):
        for pair, v in P3.SCO["contrasts"][lv]["pairs"].items():
            q = v["qwk_5class"]
            key = tag + "Diff" + pair.replace("-", "")
            mac(key, P3.p3(q["diff"], sign=True))
            mac(key + "ci", ci(q["diff_ci"], signed=True))
    # training at the label unit: Ostrava examinations, pooled over seeds when the scorer pooled them
    for a in ("B", "D", "C"):
        arm = P3.SCO["arms"][a]
        src = (arm.get("pooled_over_seeds") or {}).get("cv_per_source_unit")
        if src is None:                                   # fall back to report_data (both seeds pooled there too)
            src = {k: {"qwk_5class": v["qwk_5class"]} for k, v in P3.R["arms"][a]["cv_units_by_source"].items()}
        mac(a + "ostravaQ", f3(src["ostrava"]["qwk_5class"]))
        mac(a + "ropvlQ", f3(src["ropvl"]["qwk_5class"]))
        mac(a + "shenzhenQ", f3(src["shenzhen"]["qwk_5class"]))


# ------------------------------------------------------------------ figures
def fig_cascade():
    flow = jload(SRC["flow"])
    ds = P3.R["dataset"]
    l0, l1, l2 = flow["layer_0_acquired"], flow["layer_1_stage_labelled_sources"], flow["layer_2_six_class_eligible"]
    k = lambda x: f"{int(x):,}"  # noqa: E731  plain thousands separator for matplotlib text
    fig, ax = plt.subplots(figsize=(3.5, 4.5))
    ax.set_xlim(0, 100)
    ax.set_ylim(11, 100)
    ax.axis("off")
    NAVY, LIGHT, RED = "#1f3a5f", "#e8eef6", "#8c2f1e"
    X0, W, XS = 2, 60, 65              # box left edge and width; side-note column

    def box(y, h, title, sub, fill=NAVY, fc="white"):
        ax.add_patch(FancyBboxPatch((X0, y), W, h, boxstyle="round,pad=0,rounding_size=1.5", fc=fill, ec=fill, lw=0.8))
        ax.text(X0 + W / 2, y + h * 0.64, title, ha="center", va="center", fontsize=6.6, color=fc, weight="bold")
        ax.text(X0 + W / 2, y + h * 0.28, sub, ha="center", va="center", fontsize=5.7, color=fc)

    def side(y, lines, color=RED):
        ax.text(XS, y, "\n".join(lines), fontsize=5.5, color=color, va="center", ha="left", linespacing=1.3)

    def arrow(y0, y1):
        ax.annotate("", xy=(X0 + W / 2, y1), xytext=(X0 + W / 2, y0), arrowprops=dict(arrowstyle="-|>", color=GREY, lw=0.9))

    box(89, 10, f"{k(l0['images'])} photographs acquired", f"{l0['sources']} public sources")
    side(84.5, [f"$-$ {k(l1['excluded']['total'])}: no stage", "label, segmentation-", "only sets, duplicates"])
    arrow(89, 80)
    box(70, 10, f"{k(l1['kept']['images'])} stage-labelled", f"{k(l1['kept']['infants'])} infants, 4 cohorts")
    ex = l2["excluded"]["by_reason"]
    side(65, [f"$-$ {k(ex['other_pathology'])} other pathology", f"$-$ {k(ex['treated'])} treated / regressed",
              f"$-$ {ex['stage_0']} stage 0"])
    arrow(70, 61)
    box(51, 10, f"{k(l2['kept']['images'])} eligible, six classes", f"{k(l2['kept']['infants'])} infants")
    arrow(51, 42)
    box(32, 10, f"{k(ds['pool']['images'])} cross-validation pool", f"{k(ds['pool']['infants'])} infants, 3 cohorts",
        fill=LIGHT, fc=INK)
    side(37, [f"trained on after per-infant", f"caps: {k(P3.V4['training_rows']['v3_kept'])}; the rest",
              "stay in validation"], color=INK)
    ax.text(X0 + W / 2, 27.5, "the 4th cohort contributes to no fold", ha="center", va="center", fontsize=5.5, color=GREY,
            style="italic")
    box(13, 10, f"{k(ds['dev']['images'] + ds['locked']['images'])} held-out cohort",
        f"{ds['dev']['infants'] + ds['locked']['infants']} infants, never trained on", fill=LIGHT, fc=INK)
    side(18, [f"dev half {ds['dev']['images']} photos, {ds['dev']['eyes']} eyes;", f"locked half {ds['locked']['images']}:",
              "scored once (Aug 2026),", "unopened since"])
    save(fig, "fig_cascade")


def fig_timeline():
    """Generations in order, each with its split type and own-protocol headline (MODEL_HISTORY + MATE)."""
    m = P3.MAC
    events = [
        ("8 Jun 2026", "v1 EfficientNet-B3", "ROP yes/no\nimage-level split", f"test acc {m['vOneAcc']}", "#4b5563"),
        ("28 Jun 2026", "CNN sweep: ResNet50", "ROP yes/no\ndevice shortcut", f"test AUC {m['sweepAUC']}", "#4b5563"),
        ("18 Aug 2026", "v2 EfficientNetV2-S", "6 classes, patient-grouped CV\nlocked half opened once",
         f"locked eye QWK5$_d$ {m['vTwoLockedEyeQ']}", "#0f766e"),
        (pd.Timestamp(P3.MAC["protoDate"]).strftime("%-d %b %Y"), "Protocol v2 committed", "arms A--C, endpoints, rule\nfixed before training", "", "#8c2f1e"),
        ("25 Oct 2026", "v3: ViT arms A--G", "RETFound-Green, DINOv2,\nvariants; 2 seeds", f"B: CV QWK5 {m['BcvQ']}", "#c2410c"),
        ("1 Nov 2026", "v4: arms H, I; B served", "cleaned pool, recipe;\nB stays by rule", f"B: eye QWK5 {m['BdevQ']}", "#c2410c"),
        ("5 Nov 2026", "Independent packages", "classical ML, CNNs, LoRA ViT-L\nphoto-level split", f"best acc {m['mateBestAcc']}", "#6b7280"),
    ]
    fig, ax = plt.subplots(figsize=(7.2, 2.5))
    n = len(events)
    ax.hlines(0, -0.5, n - 0.5, color=GREY, lw=1.2)
    for i, (d, title, sub, head, col) in enumerate(events):
        up = i % 2 == 0
        y = 1 if up else -1
        va = "bottom" if up else "top"
        ax.plot([i, i], [0, y * 0.5], color=col, lw=0.8)
        ax.plot(i, 0, "o", color=col, ms=4.5, mec="white", mew=0.6)
        ax.text(i, y * 0.14, d, ha="center", va=va, fontsize=5.4, color=GREY)
        ax.text(i, y * 0.58, title, ha="center", va=va, fontsize=6.6, weight="bold", color=col)
        ax.text(i, y * 1.0, sub, ha="center", va=va, fontsize=5.6, color=INK, linespacing=1.25)
        if head:
            ax.text(i, y * 1.85, head, ha="center", va=va, fontsize=5.9, color=col, style="italic")
    ax.set_xlim(-0.6, n - 0.4)
    ax.set_ylim(-2.6, 2.6)
    ax.axis("off")
    fig.tight_layout()
    save(fig, "fig_timeline")


def _dev_examples():
    """Development-half photographs for the Grad-CAM strip: served ensemble (seed 42) mean over its five checkpoints."""
    runs = ROOT / "results/rop/v3/runs/dinov2_s_s42"
    zs = [np.load(runs / f"fold_{k}" / "dev_predictions.npz", allow_pickle=True) for k in range(5)]
    p = np.mean([z["prob"] for z in zs], 0)
    y, paths = zs[0]["y_true"], zs[0]["image_path"]
    pred = p.argmax(1)
    pick = {}
    for want, cond in (("Normal, correct", (y == 0) & (pred == 0)), ("Stage 3, correct", (y == 3) & (pred == 3)),
                       ("Stage 2, correct", (y == 2) & (pred == 2)), ("Stage 2, called Normal", (y == 2) & (pred == 0))):
        idx = np.where(cond)[0]
        if len(idx):
            j = idx[np.argmax(p[idx, pred[idx]])]        # the most confident example of that kind
            pick[want] = (str(paths[j]), int(y[j]), int(pred[j]), float(p[j, pred[j]]))
    return pick


def fig_gradcam():
    from PIL import Image
    from webapp.inference import StagingEnsemble, _device
    ens = StagingEnsemble(_device())
    if not ens.loaded:
        print("   fig_gradcam skipped: served ensemble not deployed")
        return
    picks = _dev_examples()
    fig, axes = plt.subplots(2, len(picks), figsize=(7.2, 3.9))
    names = ens.CLASSES
    for j, (title, (path, yt, yp, conf)) in enumerate(picks.items()):
        im = Image.open(ROOT / path).convert("RGB")
        b64, ev = ens.gradcam(im, yp)
        cam = Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
        axes[0, j].imshow(im)
        axes[1, j].imshow(cam)
        axes[0, j].set_title(f"{title}\npredicted {names[yp]} ({conf * 100:.0f}%)", fontsize=6.6)
        off = ev.get("off_retina_pct") if ev else None
        conc = ev.get("concentration_pct") if ev else None
        axes[1, j].set_xlabel((f"off-retina {off:.0f}%; half of attention\nin {conc:.0f}% of the frame"
                               if off is not None and conc is not None else ""), fontsize=6)
        for a in axes[:, j]:
            a.set_xticks([])
            a.set_yticks([])
            for sp in a.spines.values():
                sp.set_visible(False)
    axes[0, 0].set_ylabel("photograph", fontsize=7)
    axes[1, 0].set_ylabel("Grad-CAM, predicted class", fontsize=7)
    fig.tight_layout()
    save(fig, "fig_gradcam")


def fig_preprocess():
    from PIL import Image
    from models.common.preprocessing import FundusPreprocess
    dev = pd.read_csv(ROOT / SRC["dev"])
    path = ROOT / dev[dev.label == 3].image_path.iloc[0]        # a stage-3 development-half photograph
    im = Image.open(path).convert("RGB")
    steps = [("original", im),
             ("circle crop", FundusPreprocess(circle=True, do_clahe=False, letterbox=False)(im)),
             ("+ letterbox", FundusPreprocess(circle=True, do_clahe=False, letterbox=True)(im)),
             ("+ CLAHE", FundusPreprocess(circle=True, do_clahe=True, letterbox=True)(im))]
    steps.append(("resize 392 px (model input)", steps[-1][1].resize((392, 392))))
    fig, axes = plt.subplots(1, len(steps), figsize=(7.2, 1.75))
    for a, (t, s) in zip(axes, steps):
        a.imshow(s)
        a.set_title(t, fontsize=7)
        a.set_xticks([])
        a.set_yticks([])
        for sp in a.spines.values():
            sp.set_visible(False)
    fig.tight_layout()
    save(fig, "fig_preprocess")


WEBAPP_SHOT_FULL = "docs/paper/figs4/fig_webapp_full.png"     # the whole page, captured with headless Chrome


def fig_webapp():
    """The webapp's results panel (served verdict and stage, comparison stage, seen-photo notice), cropped from the
    full-page screenshot in docs/paper/figs4/fig_webapp_full.png; the crop window is a fixed share of the page."""
    from PIL import Image
    full = ROOT / WEBAPP_SHOT_FULL
    if not full.exists():
        print("   fig_webapp: full-page screenshot absent; keeping the existing fig_webapp.png")
        return
    im = Image.open(full)
    w, h = im.size
    im.crop((int(0.08 * w), int(0.355 * h), int(0.92 * w), int(0.82 * h))).save(ROOT / SRC["webapp_shot"])
    print("   fig_webapp (cropped results panel)")


# ------------------------------------------------------------------ tables
def tab_sources():
    from PIL import Image
    m = pd.read_csv(ROOT / SRC["manifest"])
    pool, ds = P3.R["dataset"]["pool"]["by_source"], P3.R["dataset"]
    rows = []
    for s in ("ostrava", "ropvl", "shenzhen", "multiview"):
        g = m[m.source == s]
        name, site, unit, lic = COHORT[s]
        devs = sorted(set(g.device) - {"unknown"})
        cams = ", ".join(DEVICE.get(d, d) for d in devs) if devs else "not stated"
        sizes = []
        for d in (devs or ["unknown"]):
            p = g[g.device == d].image_path.iloc[0]
            w, h = Image.open(p if Path(p).is_absolute() else ROOT / p).size
            sizes.append(f"{w}$\\times${h}")
        if s == "multiview":
            use = f"held out: dev {ds['dev']['images']}, locked {ds['locked']['images']}"
        else:
            use = f"CV pool {n0(pool[s]['images'])} ({pool[s]['infants']} infants)"
        rows.append([name, site, cams, ", ".join(sizes), unit, lic, n0(len(g)), n0(g.patient_id.nunique()), use])
    tabular("sources", "@{}l>{\\raggedright\\arraybackslash}p{2.3cm}>{\\raggedright\\arraybackslash}p{2.7cm}"
                       ">{\\raggedright\\arraybackslash}p{2.0cm}>{\\raggedright\\arraybackslash}p{1.9cm}lrr>{\\raggedright\\arraybackslash}p{2.6cm}@{}",
            [r"Cohort & Hospital & Cameras & Native size & Label unit & Licence & Photos & Infants & Use\\"], rows)


def tab_funnel():
    flow = jload(SRC["flow"])
    ds = P3.R["dataset"]
    l0, l1, l2 = flow["layer_0_acquired"], flow["layer_1_stage_labelled_sources"], flow["layer_2_six_class_eligible"]
    ex1, ex2 = l1["excluded"]["by_reason"], l2["excluded"]["by_reason"]
    rows = [["Acquired", n0(l0["images"]), "--", "--", f"{l0['sources']} publicly released sources"],
            ["Stage-labelled cohorts", n0(l1["kept"]["images"]), n0(l1["kept"]["infants"]), n0(l1["excluded"]["total"]),
             "no ICROP stage label (plus-only, segmentation sets), too small to fold, re-encoded duplicates"],
            ["Six-class eligible", n0(l2["kept"]["images"]), n0(l2["kept"]["infants"]), n0(l2["excluded"]["total"]),
             f"other pathology {n0(ex2['other_pathology'])}, treated {n0(ex2['treated'])}, stage 0 {ex2['stage_0']}"],
            ["Cross-validation pool", n0(ds["pool"]["images"]), n0(ds["pool"]["infants"]), "--", "three cohorts; every photo in one validation fold"],
            ["Training-eligible", n0(P3.V4["training_rows"]["v3_kept"]), n0(ds["pool"]["infants"]), n0(ds["pool"]["images"] - P3.V4["training_rows"]["v3_kept"]),
             f"per-infant caps: {P3.V4['caps']['normal']} normal, {P3.V4['caps']['_default']} per disease class; capped photos stay in validation"],
            ["Held-out cohort, dev half", n0(ds["dev"]["images"]), str(ds["dev"]["infants"]), "--", f"{ds['dev']['eyes']} eyes; checks and every reported external number"],
            ["Held-out cohort, locked half", n0(ds["locked"]["images"]), str(ds["locked"]["infants"]), "--", f"{ds['locked']['eyes']} eyes; opened once, Aug 2026, for the CNN generation"]]
    tabular("funnel", "@{}lrrr>{\\raggedright\\arraybackslash}p{7.6cm}@{}",
            [r"Step & Photos & Infants & Removed & Reason / note\\"], rows)


def tab_audit():
    a = jload(SRC["audit"])
    g4 = a["G4_metadata_baseline"]
    rows = [["G1", "Patient leakage across folds", "none" if a["G1_patient_leakage"]["pass"] else "FAIL"],
            ["G2", "Exact (MD5) and perceptual (256-bit dHash) duplicate leakage", f"{a['G2_duplicate_leakage']['exact_md5']} / {a['G2_duplicate_leakage']['near_dupes']} pairs; closest distance {a['G2_duplicate_leakage']['closest_distance']}"],
            ["G3", "Site decodable from trivial metadata (balanced accuracy; chance)", f"{f3(a['G3_site_decodability']['balanced_accuracy'])} (chance {f3(a['G3_site_decodability']['chance'])}); stage 4/5 exists in one source only"],
            ["G4", "Metadata-only classifier, no pixels (macro-F1 floor)", f"CV {f3(g4['bar_cv_macro_f1'])}, unseen hospital {f3(g4['bar_dev_macro_f1'])}"],
            ["G5", "Disease-controlled site probe on embeddings", "reported per model (Section~\\ref{sec:cnn})"],
            ["--", "Image dimensions alone, earlier binary split (AUC)", P3.MAC["shortcutAUC"]]]
    rows[4] = ["G5", "Disease-controlled site probe on embeddings", "per model; not run for A--I (reported for the CNN generation in~\\cite{nagpure2026})"]
    tabular("audit", "@{}l>{\\raggedright\\arraybackslash}p{4.5cm}>{\\raggedright\\arraybackslash}p{3.1cm}@{}",
            [r"Gate & What it measures & Result\\"], rows)


# ------------------------------------------------------------------ added after review: seeds, equivalence, units, eyes
def numbers_review():
    """Macros the reviews asked for: per-seed values, equivalence against the protocol's minimum detectable difference,
    six-label QWK, per-cohort unit QWK with CIs, and the dev-half eye analyses (photographs per eye, fixed-view QWK,
    sensitivity / specificity with infant-clustered CIs) of the served seed-42 ensemble."""
    S, R = P3.SCO, P3.R
    for a in ("A", "B", "C"):
        for tag, s in (("sA", P3.SEEDS[0]), ("sB", P3.SEEDS[1])):
            sd = S["arms"][a]["seeds"][s]
            mac(f"cvQ{a}{tag}", f3(sd["image"]["qwk_5class"]))
            mac(f"devQ{a}{tag}", f3(sd["dev"]["eye"]["qwk_5class"]))
    proto = P3.text(P3.SRC_FILES["protocol"])
    mdd = float(re.search(r"Minimum detectable difference at 80 ?% power: ≈ ?([\d.]+)", proto).group(1))
    mac("mdd", f"{mdd:.2f}")
    for lv, tag in (("cv_image", "Cv"), ("dev_eye", "Dev")):
        pairs = S["contrasts"][lv]["pairs"]
        within = sum(1 for v in pairs.values() if -mdd <= v["qwk_5class"]["diff_ci"][0] and v["qwk_5class"]["diff_ci"][1] <= mdd)
        zero = sum(1 for v in pairs.values() if v["qwk_5class"]["diff_ci"][0] <= 0 <= v["qwk_5class"]["diff_ci"][1])
        mac(f"eq{tag}Within", str(within))
        mac(f"eq{tag}Zero", str(zero))
        mac(f"eq{tag}Total", str(len(pairs)))
    pooled = S["arms"]["B"].get("pooled_over_seeds") or {}
    q6 = (pooled.get("image") or {}).get("qwk_6label") or np.mean([S["arms"]["B"]["seeds"][s]["image"]["qwk_6label"] for s in P3.SEEDS])
    mac("BcvQsix", f3(q6))
    for a in ("B", "C", "D"):
        u = R["arms"][a]["cv_units_by_source"]
        for src in ("ostrava", "ropvl", "shenzhen"):
            mac(f"{a}{src}Qci", ci(u[src]["ci95"]["qwk_5class"]))
            mac(f"{a}{src}Qr", f3(u[src]["qwk_5class"]))     # report_data value (both seeds pooled)
    # --- dev-half eyes, served seed-42 ensemble
    from sklearn.metrics import cohen_kappa_score
    runs = ROOT / "results/rop/v3/runs/dinov2_s_s42"
    zs = [np.load(runs / f"fold_{k}" / "dev_predictions.npz", allow_pickle=True) for k in range(5)]
    prob = np.mean([z["prob"] for z in zs], 0)
    y, paths = zs[0]["y_true"], zs[0]["image_path"]
    av = pd.read_excel(ROOT / "data/rop/multiview/ROP-DATASET/Dataset Acquisition View Table.xlsx")
    vc = ["Left temporal", "Left posterior pole", "Left nasal", "Right nasal", "Right posterior pole", "Right temporal"]
    av["eye"] = av[vc].notna().idxmax(axis=1).str.split().str[0]
    df = pd.DataFrame({"path": paths, "label": y, "pred": prob.argmax(1), "anyrop": 1 - prob[:, 0]})
    df["Image_name"] = df.path.str.rsplit("/", n=1).str[1].str.replace(".jpg", "", regex=False)
    df["infant"] = df.path.str.split("/").str[-2]
    df = df.merge(av[["Image_name", "eye"]], on="Image_name", how="left")
    assert df.eye.notna().all(), "view-table join incomplete"
    g = df.groupby(["infant", "eye"]).agg(y=("label", "max"), p=("pred", "max"), s=("anyrop", "max"), n=("label", "size")).reset_index()
    thr = S["arms"]["B"]["seeds"]["42"]["operating_point"]["cv"]["threshold"]
    m5 = lambda v: np.minimum(v, 4)  # noqa: E731  AP-ROP folded onto stage 4/5
    mac("devPhotosPerEyeMedian", str(int(g.n.median())))
    mac("devPhotosPerEyeMin", str(int(g.n.min())))
    mac("devPhotosPerEyeMax", str(int(g.n.max())))
    mac("devEyesThreeViews", str(int((g.n == 3).sum())))
    mac("devPhotosNormalMean", f"{g[g.y == 0].n.mean():.2f}")
    mac("devPhotosDiseasedMean", f"{g[g.y > 0].n.mean():.2f}")
    mac("devPhotosSevereMean", f"{g[g.y >= 3].n.mean():.2f}")

    def stats(sub):
        q = cohen_kappa_score(m5(sub.y.values), m5(sub.p.values), weights="quadratic")
        pos, neg = sub[sub.y > 0], sub[sub.y == 0]
        sens = float((pos.s >= thr).mean()) if len(pos) else np.nan
        spec = float((neg.s < thr).mean()) if len(neg) else np.nan
        return q, sens, spec

    q_all, sens, spec = stats(g)
    q3 = stats(g[g.n == 3])[0]
    rng = np.random.default_rng(42)
    inf = g.infant.unique()
    boots = []
    for _ in range(2000):
        pick = rng.choice(inf, len(inf), replace=True)
        sub = pd.concat([g[g.infant == i] for i in pick])
        boots.append(stats(sub) + (stats(sub[sub.n == 3])[0],))
    b = np.array(boots, dtype=float)
    lo, hi = np.nanpercentile(b, 2.5, axis=0), np.nanpercentile(b, 97.5, axis=0)
    mac("BdevQens", f3(q_all))
    mac("BdevQensci", ci([lo[0], hi[0]]))
    mac("BdevSensEns", f3(sens))
    mac("BdevSensEnsci", ci([lo[1], hi[1]]))
    mac("BdevSpecEns", f3(spec))
    mac("BdevSpecEnsci", ci([lo[2], hi[2]]))
    mac("BdevQthree", f3(q3))
    mac("BdevQthreeci", ci([lo[3], hi[3]]))
    mac("devEyesPos", str(int((g.y > 0).sum())))
    mac("devEyesNeg", str(int((g.y == 0).sum())))


def fig_confusion_abc():
    """Eye-level confusion matrices at the held-out cohort for the selected arm and its two reference arms."""
    arms = ("B", "A", "C")
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.35), gridspec_kw={"wspace": 0.08})
    labs = [P3.SHORT[c] for c in P3.CL]
    xl = ["N" if c == "normal" else P3.SHORT[c] for c in P3.CL]
    for k, (ax, a) in enumerate(zip(axes, arms)):
        share = np.array(P3.arm(a)["dev_eye"]["confusion"]["row_normalised"])
        sup = [P3.arm(a)["dev_eye"]["classes"][c]["support"] for c in P3.CL]
        ax.imshow(share, cmap="Blues", vmin=0, vmax=1)
        for i in range(6):
            for j in range(6):
                v = share[i, j]
                if v >= 0.005:
                    ax.text(j, i, f"{100 * v:.0f}", ha="center", va="center", fontsize=6.3, color="white" if v > 0.55 else INK)
        ax.set_xticks(range(6), xl, fontsize=6.3)
        ax.set_yticks(range(6), [f"{lb} ({n})" for lb, n in zip(labs, sup)] if k == 0 else [], fontsize=6.3)
        ax.set_xlabel("Predicted", fontsize=7)
        if k == 0:
            ax.set_ylabel("True (eyes)", fontsize=7)
        ax.set_title(P3.NAMES[a], fontsize=7.8, color=INK)
        ax.spines[:].set_visible(False)
        ax.tick_params(length=0)
    save(fig, "fig_confusion_abc")


def write_numbers():
    lines = ["% generated by scripts/paper4_figures.py (includes every macro of scripts/paper3_figures.py); do not edit"]
    lines += [rf"\newcommand{{\{k}}}{{{v}}}" for k, v in P3.MAC.items()]
    FIG.mkdir(parents=True, exist_ok=True)
    (FIG / "numbers4.tex").write_text("\n".join(lines) + "\n")
    print("   numbers4.tex", len(P3.MAC), "macros")


def docx():
    src = TEX.read_text()

    def inline(m):
        return (PAPER / (m.group(1) + ("" if m.group(1).endswith(".tex") else ".tex"))).read_text()

    s = re.sub(r"\\input\{([^}]+)\}", inline, src)
    s = re.sub(r"\\documentclass\[[^\]]*\]\{IEEEtran\}", r"\\documentclass{article}", s)
    s = s.replace(r"\IEEEoverridecommandlockouts", "").replace(r"\IEEEpeerreviewmaketitle", "")
    s = re.sub(r"\\IEEEauthorblockN\{", "{", s)
    s = re.sub(r"\\IEEEauthorblockA\{", "{", s)
    s = re.sub(r"\\begin\{IEEEkeywords\}(.*?)\\end\{IEEEkeywords\}", r"\\noindent Keywords: \1", s, flags=re.S)
    s = re.sub(r"\\IEEEPARstart\{(\w)\}\{([^}]*)\}", r"\1\2", s)
    s = s.replace("table*", "table").replace("figure*", "figure")
    s = re.sub(r"\\includegraphics\[[^]]*\]\{([^}]+)\.pdf\}", r"\\includegraphics[width=6in]{\1.png}", s)
    s = re.sub(r"\\(scriptsize|footnotesize|small)\b", "", s)
    s = re.sub(r"\\setlength\{\\tabcolsep\}\{[^}]*\}", "", s)
    s = re.sub(r"@\{(?:[^{}]|\{[^{}]*\})*\}", "", s)
    s = s.replace(r">{\raggedright\arraybackslash}", "")
    keys = re.findall(r"\\bibitem\{([^}]+)\}", s)
    num = {k: i + 1 for i, k in enumerate(keys)}
    s = re.sub(r"\\cite\{([^}]+)\}", lambda m: "[" + ", ".join(str(num[k.strip()]) for k in m.group(1).split(",")) + "]", s)
    s = re.sub(r"\\bibitem\{([^}]+)\}", lambda m: f"\\item[[{num[m.group(1)]}]]", s)
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td) / "paper4_simplified.tex"
        tmp.write_text(s)
        subprocess.run(["pandoc", str(tmp), "--from", "latex", "--to", "docx", "-o", str(DOCX_OUT),
                        f"--resource-path={FIG}:{P3.FIG}:{PAPER / 'figs2'}:{PAPER}"], check=True)
    print("  ", DOCX_OUT.relative_to(ROOT))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--docx", action="store_true")
    ap.add_argument("--no-gradcam", action="store_true", help="skip the Grad-CAM strip (needs the served weights)")
    args = ap.parse_args()
    P3.numbers()
    P3.mate_numbers()
    numbers4()
    numbers_review()
    # paper-3 tables and figures this paper inputs, regenerated so a clean checkout builds
    for f in (P3.fig_selection, P3.fig_dev, P3.fig_stage_recall, P3.fig_confusion, P3.fig_roc_pr, P3.fig_reliability,
              P3.fig_loso, P3.fig_training, P3.fig_teammate):
        f()
    for t in (P3.tab_headline, P3.tab_stage, P3.tab_operating, P3.tab_recipe, P3.tab_sweep, P3.tab_v1, P3.tab_v2,
              P3.tab_mate_data, P3.tab_mate_ml, P3.tab_mate_cnn, P3.tab_mate_fm, P3.tab_best):
        t()
    for f in (fig_cascade, fig_timeline, fig_preprocess, fig_webapp, fig_confusion_abc):
        f()
    if not args.no_gradcam:
        fig_gradcam()
    for t in (tab_sources, tab_funnel, tab_audit):
        t()
    write_numbers()
    if args.docx:
        docx()


if __name__ == "__main__":
    main()
