"""LaTeX tables for the BE report, generated from results/rop/v3/report_data.json and v3_scores.json so no number
is typed by hand.

    .venv/bin/python scripts/rop_v3_report_tables.py   # -> docs/report/BE_Project_Report/tables/*.tex

deployed_macros.tex scores the configuration the application serves (arm B, seed 42, mean of the five fold
checkpoints' probabilities) on the Multi-View development half with the scorer's own functions
(scripts/rop_v3_score.py); test_locked is never read (the scorer's path guard refuses it).
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs/report/BE_Project_Report/tables"
R = json.loads((ROOT / "results/rop/v3/report_data.json").read_text())
SCO = json.loads((ROOT / "results/rop/v3/v3_scores.json").read_text())
CL = R["classes"]
CLN = {"normal": "Normal", "stage_1": "Stage 1", "stage_2": "Stage 2", "stage_3": "Stage 3", "stage_4_5": "Stage 4/5", "arop": "AP-ROP"}
NAMES = {"A": "A RETFound-Green", "B": "B DINOv2", "C": "C EfficientNet (baseline)", "D": "D DINOv2 + exam loss",
         "E": "E Ensemble A+B+C", "F": "F DINOv2 + adaptation$^\\dagger$", "G": "G DINOv2 at 518 px",
         "H": "H DINOv2, v4 pool", "I": "I DINOv2, v4 pool + recipe"}
ARMS = [a for a in "ABCDEFGHI" if a in R["arms"]]
SRC = {"shenzhen": "Shenzhen", "ostrava": "Ostrava", "ropvl": "ROP-VL"}
f3 = lambda x: f"{x:.3f}"  # noqa: E731
f2 = lambda x: "--" if x is None else f"{x:.2f}"  # noqa: E731


def bold_max(vals, fmt):
    m = max(v for v in vals if v is not None)
    return [("\\textbf{" + fmt(v) + "}") if v == m else fmt(v) for v in vals]


def headline():
    rows = []
    cols = {k: [] for k in ("cv", "cva", "cvf", "ey", "eya", "eyf")}
    for a in ARMS:
        A = R["arms"][a]; cv, ey = A["cv_image"], A["dev_eye"]
        cols["cv"].append(cv["qwk_5class"]); cols["cva"].append(cv["accuracy"]); cols["cvf"].append(cv["macro_f1"])
        cols["ey"].append(ey["qwk_5class"]); cols["eya"].append(ey["accuracy"]); cols["eyf"].append(ey["macro_f1"])
    b = {k: bold_max(v, f3) for k, v in cols.items()}
    for i, a in enumerate(ARMS):
        A = R["arms"][a]; cv, ey = A["cv_image"], A["dev_eye"]
        rows.append(f"{NAMES[a]} & {b['cv'][i]} ({f3(cv['ci95']['qwk_5class'][0])}--{f3(cv['ci95']['qwk_5class'][1])}) & {b['cva'][i]} & {b['cvf'][i]} & "
                    f"{b['ey'][i]} ({f3(ey['ci95']['qwk_5class'][0])}--{f3(ey['ci95']['qwk_5class'][1])}) & {b['eya'][i]} & {b['eyf'][i]}\\\\")
    return "\n".join(rows) + "\n"


def per_class(arm="B"):
    rows = []
    for c in CL:
        cv = R["arms"][arm]["cv_image"]["classes"][c]; ey = R["arms"][arm]["dev_eye"]["classes"][c]
        rows.append(f"{CLN[c]} & {cv['support']:,} & {f2(cv['precision'])} & {f2(cv['recall'])} & {f2(cv['f1'])} & {f2(cv['auc_ovr'])} & "
                    f"{ey['support']} & {f2(ey['precision'])} & {f2(ey['recall'])} & {f2(ey['f1'])} & {f2(ey['auc_ovr'])}\\\\")
    return "\n".join(rows) + "\n"


def per_fold():
    rows = []
    for a in ARMS:
        pf = R["arms"][a]["per_fold"]
        if not pf:
            continue
        v = {(p["seed"], p["fold"]): p["qwk_5class"] for p in pf}
        s42 = " & ".join(f2(v.get((42, k))) for k in range(5)); s13 = " & ".join(f2(v.get((1337, k))) for k in range(5))
        rows.append(f"{NAMES[a].replace('$^\\dagger$', '')} & {s42} & {s13}\\\\")
    return "\n".join(rows) + "\n"


def timing():
    rows = []
    for a in ("A", "B", "D", "F", "G", "H", "I"):
        t = R["arms"][a].get("training", {}).get("timing", {})
        runs = [f for s in t.values() for f in s]
        if not runs:
            continue
        cv = [f for f in runs if f["fold"] in {f"fold{k}" for k in range(5)}]
        thr = np.mean([f["img_per_s"] for f in cv if f["img_per_s"]])
        per_fold_min = np.mean([f["minutes"] for f in cv])
        # the CV folds plus the site fold, so the v4 arms (which also ran two LOSO folds) compare with A-G
        sess = [sum(f["minutes"] for f in s if f["fold"] in {*(f"fold{k}" for k in range(5)), "foldsite"}) for s in t.values()]
        kernels = ("one kernel per seed" if a in "ABD" else "two kernels per seed" if a in "FG"
                   else "two per seed (CV; site + 2 LOSO)")
        rows.append(f"{NAMES[a].replace('$^\\dagger$', '')} & {thr:.0f} & {per_fold_min:.0f} & {min(sess):.0f}--{max(sess):.0f} & {kernels}\\\\")
    return "\n".join(rows) + "\n"


def loso():
    """Leave-one-source-out folds (Amendment 5a.5): QWK5 with CI per held-out source, and the guard's mean."""
    rows = []
    for a in "BHI":
        L = SCO["loso"][a]
        cells = [f"{f3(L[s]['qwk_5class'])} ({f3(L[s]['ci95']['qwk_5class'][0])}--{f3(L[s]['ci95']['qwk_5class'][1])})"
                 for s in ("shenzhen", "ostrava", "ropvl")]
        rows.append(f"{NAMES[a]} & {L['pool']} & " + " & ".join(cells) + f" & {f3(L['mean_qwk_5class'])}\\\\")
    return "\n".join(rows) + "\n"


def selection():
    """Amendment 5c, as scored (v3_scores.json -> v4_selection)."""
    v = SCO["v4_selection"]
    rows = []
    for a in ("I", "H"):
        dcv, dl = v["cv"][a] - v["cv"]["B"], v["loso"][a] - v["loso"]["B"]
        ok = dcv >= v["margin_cv"] and dl >= -v["margin_loso"]
        rows.append(f"{NAMES[a]} & {f3(v['cv'][a])} & ${dcv:+.4f}$ & \\mbox{{$\\ge +{v['margin_cv']:.3f}$}} & {f3(v['loso'][a])} & ${dl:+.4f}$ & "
                    f"\\mbox{{$\\ge -{v['margin_loso']:.3f}$}} & {'adopted' if ok else 'not adopted'}\\\\")
    rows.append(f"{NAMES['B']} & {f3(v['cv']['B'])} & & & {f3(v['loso']['B'])} & & & "
                f"{'\\textbf{stays}' if v['chosen'] == 'B' else 'replaced'}\\\\")
    return "\n".join(rows) + "\n"


def deployed():
    """The served configuration, scored with the scorer's functions on the dev half (never test_locked)."""
    sys.path.insert(0, str(ROOT))
    from scripts import rop_v3_score as SC
    zs = SC.load_arm_dir(SC.RUNS / "dinov2_s_s42")["dev"]
    prob = np.mean([z["prob"] for z in zs], 0)
    y, inf, preds = SC._dev_eyes([{"y_true": zs[0]["y_true"], "y_pred": prob.argmax(1), "prob": prob}])
    m = SC.class_metrics(y, preds[0])
    ci = SC.boot_ci(y, preds, inf, {"qwk_5class": SC.qwk5, "macro_f1_present": SC.macro_f1_present},
                    SCO["bootstrap"]["replicates"], SCO["bootstrap"]["seed"])
    r1 = m["per_class_recall"]["stage_1"]["recall"]
    return "\n".join([
        "% generated by scripts/rop_v3_report_tables.py: arm B, seed 42, mean of the 5 fold checkpoints, Multi-View dev, per eye",
        f"\\newcommand{{\\DeployedQWK}}{{{f3(m['qwk_5class'])}}}",
        f"\\newcommand{{\\DeployedQWKCI}}{{{f3(ci['qwk_5class'][0])}--{f3(ci['qwk_5class'][1])}}}",
        f"\\newcommand{{\\DeployedAcc}}{{{100 * m['accuracy']:.1f}\\%}}",
        f"\\newcommand{{\\DeployedMF}}{{{f3(m['macro_f1_present'])}}}",
        f"\\newcommand{{\\DeployedMFCI}}{{{f3(ci['macro_f1_present'][0])}--{f3(ci['macro_f1_present'][1])}}}",
        f"\\newcommand{{\\DeployedSOne}}{{{r1:.2f}}}",
        f"\\newcommand{{\\DeployedEyes}}{{{len(y)}}}"]) + "\n"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for name, fn in (("headline_rows", headline), ("perclass_rows", per_class), ("perfold_rows", per_fold), ("timing_rows", timing),
                     ("loso_rows", loso), ("selection_rows", selection)):
        (OUT / f"{name}.tex").write_text(fn() + "\\bottomrule\n")   # the rule lives here: \input + \bottomrule inside tabularx misparses
        print("  ", name)
    (OUT / "deployed_macros.tex").write_text(deployed())
    print("   deployed_macros")
    return 0


if __name__ == "__main__":
    sys.exit(main())
