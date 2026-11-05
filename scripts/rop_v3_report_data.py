"""Everything the v3 model report shows, computed from committed predictions only.

    .venv/bin/python scripts/rop_v3_report_data.py      # -> results/rop/v3/report_data.json

Reads the same inputs as scripts/rop_v3_score.py (CV out-of-fold and dev predictions of arms
A-I, each arm only once its run dirs exist; C from its saved npz) and reuses its loaders, metrics and unit rules, so every number agrees
with results/rop/v3/v3_scores.json. Adds what a model report needs beyond the protocol scores:
per-class precision / recall / specificity / F1 / one-vs-rest AUC, confusion matrices (counts and
row-normalised), any-ROP ROC and precision-recall curves, reliability bins, per-fold scores,
training curves, timings, hyperparameters and dataset counts. test_locked is never read: its
counts are quoted from PROTOCOL_V2.
"""
import glob
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import (average_precision_score, precision_recall_curve, roc_auc_score,
                             roc_curve)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from scripts import rop_v3_score as S  # noqa: E402

OUT = ROOT / "results/rop/v3/report_data.json"
NAMES = {"A": "RETFound-Green ViT-S", "B": "DINOv2 ViT-S", "C": "EfficientNetV2-S (baseline)",
         "D": "DINOv2 + exam-level loss", "E": "Ensemble A+B+C", "F": "DINOv2 + adaptation",
         "G": "DINOv2 at 518 px", "H": "DINOv2 on the v4 pool", "I": "DINOv2, v4 pool + v4 recipe"}
RUN_DIR = {"A": "retfound_green", "B": "dinov2_s", "D": "dinov2_s_bag", "F": "dinov2_s_uda",
           "G": "dinov2_s_r518", "H": "dinov2_s_v4data", "I": "dinov2_s_v4recipe"}
KERNEL = {"A": "rop-v3-retfound-green-s{s}-lr2e-4", "B": "rop-v3-dinov2-s-s{s}-lr1e-4",
          "D": "rop-v3-dinov2-s-bag-s{s}-lr1e-4", "F": "rop-v3-dinov2-s-uda-s{s}-lr1e-4-[ab]",
          "G": "rop-v3-dinov2-s-r518-s{s}-lr1e-4-[ab]",
          "H": "rop-v4-dinov2-s-v4data-s{s}-lr1e-4-[ab]", "I": "rop-v4-dinov2-s-v4recipe-s{s}-lr1e-4-[ab]"}
CL = S.CLASSES
r4 = lambda x: None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), 4)  # noqa: E731


def class_table(y, pred, prob):
    """Per-class precision, recall, specificity, F1, one-vs-rest AUC and support for one list."""
    out = {}
    for c in range(6):
        t, p = y == c, pred == c
        tp, fp, fn, tn = (t & p).sum(), (~t & p).sum(), (t & ~p).sum(), (~t & ~p).sum()
        prec = tp / (tp + fp) if tp + fp else np.nan
        rec = tp / (tp + fn) if tp + fn else np.nan
        f1 = 2 * prec * rec / (prec + rec) if tp else (0.0 if t.any() else np.nan)
        auc = roc_auc_score(t, prob[:, c]) if 0 < t.sum() < len(t) else np.nan
        out[CL[c]] = {"precision": prec, "recall": rec, "specificity": tn / (tn + fp) if tn + fp else np.nan,
                      "f1": f1, "auc_ovr": auc, "support": int(t.sum())}
    return out


def summary(y, pred, prob, score):
    """Headline numbers for one prediction list. score = any-ROP score (1 - p_normal, or unit max)."""
    ct = class_table(y, pred, prob)
    present = [c for c in CL if ct[c]["support"]]
    from sklearn.metrics import cohen_kappa_score, f1_score
    return {"accuracy": float((y == pred).mean()),
            "balanced_accuracy": float(np.mean([ct[c]["recall"] for c in present])),
            "macro_precision": float(np.nanmean([ct[c]["precision"] for c in present])),
            "macro_recall": float(np.mean([ct[c]["recall"] for c in present])),
            "macro_f1": S.macro_f1_present(y, pred),
            "weighted_f1": float(f1_score(y, pred, average="weighted", zero_division=0)),
            "qwk_5class": S.qwk5(y, pred), "qwk_6label": S.qwk6(y, pred),
            "kappa_unweighted": float(cohen_kappa_score(y, pred)),
            "macro_auc_ovr": float(np.nanmean([ct[c]["auc_ovr"] for c in present])),
            "any_rop_auc": float(roc_auc_score(y > 0, score)),
            "any_rop_ap": float(average_precision_score(y > 0, score)),
            "classes": ct}


def mean_lists(rows):
    """Average a list of summary() dicts (nan-aware), keeping the nested class table."""
    keys = [k for k in rows[0] if k != "classes"]
    out = {k: r4(np.nanmean([r[k] for r in rows])) for k in keys}
    out["classes"] = {c: {m: (rows[0]["classes"][c][m] if m == "support"
                              else r4(np.nanmean([r["classes"][c][m] for r in rows])))
                          for m in rows[0]["classes"][c]} for c in CL}
    out["n_lists"] = len(rows)
    return out


def confusion(y, preds):
    m = np.mean([np.bincount(y * 6 + p, minlength=36).reshape(6, 6) for p in preds], 0)
    rows = m.sum(1, keepdims=True)
    return {"counts": np.round(m, 1).tolist(),
            "row_normalised": np.round(np.divide(m, rows, out=np.zeros_like(m), where=rows > 0), 4).tolist()}


def curve(y, score, n=120):
    fpr, tpr, _ = roc_curve(y, score)
    pr, rc, _ = precision_recall_curve(y, score)
    pick = lambda a: np.unique(np.linspace(0, len(a) - 1, min(n, len(a))).astype(int))  # noqa: E731
    i, j = pick(fpr), pick(rc)
    return {"roc": {"fpr": np.round(fpr[i], 4).tolist(), "tpr": np.round(tpr[i], 4).tolist(),
                    "auc": r4(roc_auc_score(y, score))},
            "pr": {"recall": np.round(rc[j], 4).tolist(), "precision": np.round(pr[j], 4).tolist(),
                   "ap": r4(average_precision_score(y, score)), "prevalence": r4(np.mean(y))}}


def reliability(conf, correct, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(conf, edges[1:-1]), 0, bins - 1)
    out = []
    for b in range(bins):
        m = idx == b
        if m.any():
            out.append({"lo": r4(edges[b]), "hi": r4(edges[b + 1]), "n": int(m.sum()),
                        "confidence": r4(conf[m].mean()), "accuracy": r4(correct[m].mean())})
    ece = sum(o["n"] * abs(o["confidence"] - o["accuracy"]) for o in out) / len(conf)
    return {"bins": out, "ece": r4(ece)}


def kernel_timing(arm, seed, cache=ROOT / "kaggle_cache/v3", slug=None):
    """Minutes per fold, train rows and throughput from the downloaded kernel logs (if present).
    ``slug`` is a kernel-slug glob used in place of KERNEL[arm] (arm B's LOSO-only kernel)."""
    folds = []
    for path in sorted(glob.glob(str(Path(cache) / (slug or KERNEL[arm]).format(s=seed) / "*.log"))):
        if "probe" in path:
            continue
        log = json.load(open(path))
        starts = [(e["time"], e["data"]) for e in log if e["data"].startswith("=== ")]
        thr = [float(re.search(r"([\d.]+) img/s", e["data"]).group(1)) for e in log
               if e["data"].startswith("[throughput]")]
        gpu = next((e["data"].split(":", 1)[1].strip().split(" (")[0] for e in log if e["data"].startswith("GPU:")), None)
        end = log[-1]["time"]
        for k, (t, x) in enumerate(starts):
            m = re.search(r"\| (fold\w+) \|.*\| train (\d+) val (\d+)", x)
            nxt = starts[k + 1][0] if k + 1 < len(starts) else end
            folds.append({"fold": m.group(1), "train_images": int(m.group(2)), "val_images": int(m.group(3)),
                          "minutes": round((nxt - t) / 60, 1),
                          "img_per_s": thr[k] if k < len(thr) else None, "gpu": gpu})
    return folds


def main() -> int:
    loaded = S.load_v3(S.RUNS)
    pool = S.pool_frame(S.PREP)
    keep = ~pool.image_path.isin(S.read_exclusions(S.PREP / "exclude_duplicates.csv")).values
    pk = pool[keep].reset_index(drop=True)
    infants = pk.patient_id.values
    ddf, dkeys = S._dev_frame()
    dy = ddf.label.values
    scores = json.loads((ROOT / "results/rop/v3/v3_scores.json").read_text())

    rep = {"generated_from": "results/rop/v3/runs/*, results/rop/staging_runs/flat_natural_5fold*, "
                             "results/rop/dev_eval/flat_s*_ckpt*.npz, results/rop/v3/v3_scores.json",
           "classes": CL, "arm_names": NAMES, "arms": {}}

    # ---------------------------------------------------------------- dataset
    pool_tab = pool.groupby(["source", "cls6"]).size().unstack(fill_value=0)
    eyes = S.aggregate_units(dkeys, dy, dy, np.eye(6)[dy])
    folds = []
    for k in list(range(5)) + ["site"]:
        f = pd.read_csv(S.PREP / f"fold_{k}.csv")
        t = f[(f.split == "train") & ~f.image_path.isin(S.read_exclusions(S.PREP / "exclude_duplicates.csv"))]
        # (= v3_launch.drop_duplicate_train_rows; not imported because it pulls in torch)
        folds.append({"fold": str(k), "train_images": int(len(t)), "val_images": int((f.split == "val").sum()),
                      "train_infants": int(t.patient_id.nunique()),
                      "train_by_class": {c: int((t.label == i).sum()) for i, c in enumerate(CL)},
                      "val_sources": sorted(f[f.split == "val"].source.unique().tolist())})
    rep["dataset"] = {
        "pool": {"images": int(len(pool)), "infants": int(pool.patient_id.nunique()),
                 "by_source": {s: {"images": int(pool_tab.loc[s].sum()),
                                   "infants": int(pool[pool.source == s].patient_id.nunique()),
                                   "by_class": {c: int(pool_tab.loc[s].get(c, 0)) for c in
                                                ["normal", "stage_1", "stage_2", "stage_3", "stage_4_5", "arop"]}}
                               for s in pool_tab.index},
                 "duplicates_excluded": int((~keep).sum())},
        "dev": {"images": int(len(ddf)), "eyes": int(len(eyes["y"])), "infants": int(ddf.patient_id.nunique()),
                "images_by_class": {c: int((dy == i).sum()) for i, c in enumerate(CL)},
                "eyes_by_class": {c: int((eyes["y"] == i).sum()) for i, c in enumerate(CL)},
                "source": "Multi-View (held-out hospital), development half"},
        "locked": {"images": 661, "eyes": 230, "infants": 116,
                   "note": "untouched for v3; counts quoted from PROTOCOL_V2, labels not read"},
        "folds": folds}

    # ---------------------------------------------------------------- per arm
    for lab in "ABCDEFGHI":
        seeds = loaded.get(lab) or {}
        if not seeds:                   # no collected run dir (H and I until their kernels run)
            continue
        arms = [a for _, a in sorted(seeds.items())]
        cvs = [S.assemble_cv(a, pool) for a in arms]
        y = cvs[0]["y"][keep]
        cv_rows, cv_pred = [], []
        for c in cvs:
            pr, pb = c["pred"][keep], c["prob"][keep]
            cv_rows.append(summary(y, pr, pb, 1 - pb[:, 0])); cv_pred.append(pr)
        cv_prob = np.mean([c["prob"][keep] for c in cvs], 0)

        zs = [z for a in arms for z in a["dev"]]
        dU = [S.aggregate_units(dkeys, dy, z["y_pred"], z["prob"]) for z in zs]
        uy = dU[0]["y"]
        eye_rows = [summary(uy, u["pred"], u["prob"], u["score"]) for u in dU]
        img_rows = [summary(dy, z["y_pred"], z["prob"], 1 - z["prob"][:, 0]) for z in zs]

        # per-fold CV (dedup applied), per seed
        per_fold = []
        for s, a in sorted(seeds.items()):
            for k, z in enumerate(a["val"]):
                m = (pool.fold.values == k) & keep
                yk = pool.label.values[m]
                pk_ = z["y_pred"][keep[pool.fold.values == k]]
                per_fold.append({"seed": s, "fold": k, "qwk_5class": r4(S.qwk5(yk, pk_)),
                                 "macro_f1": r4(S.macro_f1_present(yk, pk_)), "accuracy": r4((yk == pk_).mean())})

        # reliability at the eye level, before / after each seed's CV temperature
        conf_b, corr_b, conf_a, corr_a = [], [], [], []
        for s, a in sorted(seeds.items()):
            T = scores["arms"][lab]["seeds"][str(s)]["temperature"]["T"]
            for z in a["dev"]:
                u = S.aggregate_units(dkeys, dy, z["y_pred"], z["prob"])
                cal = S.calibrate(u["prob"], T)
                conf_b.append(u["prob"].max(1)); corr_b.append(u["prob"].argmax(1) == u["y"])
                conf_a.append(cal.max(1)); corr_a.append(cal.argmax(1) == u["y"])

        po = scores["arms"][lab]["pooled_over_seeds"]
        rep["arms"][lab] = {
            "name": NAMES[lab], "seeds": sorted(seeds),
            "dev_status": po["dev_eye"].get("status", "descriptive, development-external"),
            "cv_image": mean_lists(cv_rows) | {"ci95": po["cv_image"]["ci95"],
                                               "confusion": confusion(y, cv_pred)},
            "cv_units_by_source": {s_: {k: po["cv_per_source_unit"][s_][k]
                                        for k in ("qwk_5class", "qwk_6label", "macro_f1_present", "ci95")}
                                   for s_ in po["cv_per_source_unit"]},
            "dev_eye": mean_lists(eye_rows) | {"ci95": po["dev_eye"]["ci95"],
                                               "confusion": confusion(uy, [u["pred"] for u in dU])},
            "dev_image": mean_lists(img_rows) | {"ci95": po["dev_image"]["ci95"],
                                                 "confusion": confusion(dy, [z["y_pred"] for z in zs])},
            "curves": {"cv_image_any_rop": curve(y > 0, 1 - cv_prob[:, 0]),
                       "dev_eye_any_rop": curve(uy > 0, np.mean([u["score"] for u in dU], 0))},
            "reliability_dev_eye": {"uncalibrated": reliability(np.concatenate(conf_b), np.concatenate(corr_b)),
                                    "cv_temperature": reliability(np.concatenate(conf_a), np.concatenate(corr_a))},
            "per_fold": per_fold,
            "operating_point": {str(s): {"threshold": r4(scores["arms"][lab]["seeds"][str(s)]["operating_point"]["cv"]["threshold"]),
                                         "cv": scores["arms"][lab]["seeds"][str(s)]["operating_point"]["cv"],
                                         "dev_eye_mean": {k: r4(np.mean([o[k]["sensitivity"] for o in scores["arms"][lab]["seeds"][str(s)]["dev"]["operating_point_at_cv_threshold"]]))
                                                          for k in ("any_rop", "severe_vs_all_nonsevere")} |
                                                         {k + "_spec": r4(np.mean([o[k]["specificity"] for o in scores["arms"][lab]["seeds"][str(s)]["dev"]["operating_point_at_cv_threshold"]]))
                                                          for k in ("any_rop", "severe_vs_all_nonsevere")}}
                                for s in sorted(seeds)},
            "temperature": {str(s): r4(scores["arms"][lab]["seeds"][str(s)]["temperature"]["T"]) for s in sorted(seeds)},
        }

        # training: config, curves, timing (trained arms only)
        if lab in RUN_DIR:
            hist = {"train_loss": [], "val_macro_f1": [], "val_accuracy": []}
            cfg = None
            for s in sorted(seeds):
                for k in range(5):
                    d = S.RUNS / f"{RUN_DIR[lab]}_s{s}" / f"fold_{k}"
                    h = json.loads((d / "metrics.json").read_text())["history"]
                    for key in hist:
                        hist[key].append(h[key])
                    cfg = cfg or yaml.safe_load((d / "config.yaml").read_text())
            rep["arms"][lab]["training"] = {
                "curves": {key: {"mean": np.round(np.mean(v, 0), 4).tolist(), "min": np.round(np.min(v, 0), 4).tolist(),
                                 "max": np.round(np.max(v, 0), 4).tolist(), "n_runs": len(v)} for key, v in hist.items()},
                "config": {"model": cfg["model"], "preprocess": cfg["preprocess"],
                           "train": {k: v for k, v in cfg["train"].items() if k != "num_workers"}},
                "timing": {str(s): kernel_timing(lab, s) for s in sorted(seeds)}}

    rep["contrasts"] = {blk: {p: {k: v for k, v in m.items()} for p, m in scores["contrasts"][blk]["pairs"].items()}
                        for blk in ("cv_image", "dev_eye")}
    rep["selection_rule"] = scores["selection_rule"]
    if "torch" in sys.modules:
        raise RuntimeError("torch was imported - this script must run no model")
    OUT.write_text(json.dumps(rep, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)) + "\n")
    print("->", OUT.relative_to(ROOT), f"{OUT.stat().st_size / 1e3:.0f} kB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
