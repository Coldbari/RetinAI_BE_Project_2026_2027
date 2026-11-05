"""Step 5 — bag-level metrics, per-source breakdown, bootstraps.  Prints markdown tables and
writes results.json.   .venv/bin/python experiments/eye_spike/s5_report.py
Metrics: QWK over 6 labels (arop = index 5, labels=0..5 fixed), macro-F1 over classes present in
y, accuracy, per-class recall. Seeds (2) averaged metric-wise; reference dev = mean of 5 ckpts.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from common import CACHE, CLASSES, HERE, bag_metrics, mf1, paired_cluster_boot, qwk

ENCS = ["rfg", "dino", "effnet", "rfg_clahe"]
METHODS = ["img_lr_max", "img_mlp_max", "meanpool", "maxpool", "attn"]
NICE = {"rfg": "RETFound-Green (no CLAHE)", "rfg_clahe": "RETFound-Green (+CLAHE)",
        "dino": "DINOv2 ViT-S/14-reg", "effnet": "EffNetV2-S ImageNet (+CLAHE)"}


def mean_metrics(y, preds):
    ms = [bag_metrics(y, p) for p in preds]
    out = {k: float(np.mean([m[k] for m in ms])) for k in ("qwk", "macro_f1", "acc")}
    out["qwk_seed_sd"] = float(np.std([m["qwk"] for m in ms])) if len(ms) > 1 else 0.0
    out["recall"] = {c: float(np.mean([m["recall"][c] for m in ms])) for c in ms[0]["recall"]}
    out["n"] = ms[0]["n"]
    return out


def main():
    bags = pd.read_csv(HERE / "bags.csv")
    pb, db = bags[bags.fold >= 0].reset_index(drop=True), bags[bags.fold < 0].reset_index(drop=True)
    y, yd = pb.label.values, db.label.values
    mem = np.load(CACHE / "membership.npz")
    gp, gd = pb.patient_hash.values.astype("U16"), db.patient_hash.values.astype("U16")
    subsets = {"ALL": np.ones(len(pb), bool), "ostrava": (pb.source == "ostrava").values,
               "ropvl": (pb.source == "ropvl").values, "shenzhen": (pb.source == "shenzhen").values,
               "multi_image": (pb.n_images > 1).values, "single_image": (pb.n_images == 1).values}

    # ---- collect predictions: arms[name] = (list of oof preds, list of dev preds)
    arms = {}
    ref = np.load(CACHE / "ref_preds.npz")
    ib, dib = mem["pool_img_bag"], mem["dev_img_bag"]
    arms["REF finetuned structured s42"] = (
        [pd.Series(ref["pool_pred"]).groupby(ib).max().sort_index().values],
        [pd.Series(p).groupby(dib).max().sort_index().values for p in ref["dev_pred"]])
    z = np.load(CACHE / "preds_size.npz")
    arms["SIZE-ONLY control"] = ([z["oof|size_only|0"]], [z["dev|size_only|0"]])
    chosen_C = {}
    for e in ENCS:
        f = CACHE / f"preds_{e}.npz"
        if not f.exists():
            continue
        z = np.load(f)
        chosen_C[e] = dict(z["chosen_C"].tolist()) if z["chosen_C"].size else {}
        for m in METHODS:
            ks = sorted(k for k in z.files if k.startswith(f"oof|{m}|"))
            arms[f"{e}|{m}"] = ([z[k] for k in ks], [z[k.replace("oof|", "dev|")] for k in ks])

    res = {"arms": {}, "chosen_C": chosen_C}
    for name, (oo, dd) in arms.items():
        r = {"cv": {s: mean_metrics(y[m], [p[m] for p in oo]) for s, m in subsets.items()},
             "dev": mean_metrics(yd, dd), "n_seeds_or_ckpts": len(oo) if "REF" not in name else len(dd)}
        r["cv"]["per_fold_qwk"] = [float(np.mean([qwk(y[pb.fold == k], p[pb.fold == k]) for p in oo]))
                                   for k in range(5)]
        r["cv"]["per_fold_macro_f1"] = [float(np.mean([mf1(y[pb.fold == k], p[pb.fold == k]) for p in oo]))
                                        for k in range(5)]
        res["arms"][name] = r

    # ---- size shortcut AUCs (raw bag size as the score; normal vs any disease)
    dis = (y > 0).astype(int)
    ost = subsets["ostrava"]
    res["size_auc"] = {"all_bags": float(roc_auc_score(dis, pb.n_images)),
                       "ostrava_sessions": float(roc_auc_score(dis[ost], pb.n_images[ost])),
                       "non_ostrava": float(roc_auc_score(dis[~ost], pb.n_images[~ost])),
                       "dev_eyes": float(roc_auc_score((yd > 0).astype(int), db.n_images))}

    # ---- pick best encoder: mean CV (QWK + macro-F1) over the five heads
    encs = [e for e in ENCS if f"{e}|attn" in arms]
    score = {e: float(np.mean([res["arms"][f"{e}|{m}"]["cv"]["ALL"]["qwk"] + res["arms"][f"{e}|{m}"]["cv"]["ALL"]["macro_f1"]
                               for m in METHODS])) / 2 for e in encs}
    best_e = max(score, key=score.get)
    best_m = max(METHODS, key=lambda m: res["arms"][f"{best_e}|{m}"]["cv"]["ALL"]["macro_f1"])
    res["selection"] = {"encoder_score_mean_qwk_mf1": score, "best_encoder": best_e,
                        "best_head_by_cv_macro_f1": best_m}

    # ---- bootstraps (1000 reps, patient-clustered, paired)
    REF = "REF finetuned structured s42"
    boots = {}
    comps = [("cv", f"{best_e}|attn", f"{best_e}|img_mlp_max"),
             ("cv_multi_image", f"{best_e}|attn", f"{best_e}|img_mlp_max"),
             ("dev", f"{best_e}|attn", f"{best_e}|img_mlp_max"),
             ("cv", f"{best_e}|{best_m}", REF),
             ("dev", f"{best_e}|{best_m}", REF)]
    if best_m != "attn":
        comps += [("cv", f"{best_e}|attn", REF), ("dev", f"{best_e}|attn", REF)]
    for where, a, b in comps:
        for mname, fn in (("qwk", qwk), ("macro_f1", mf1)):
            if where == "dev":
                yy, g, PA, PB = yd, gd, arms[a][1], arms[b][1]
            else:
                msk = subsets["multi_image"] if where == "cv_multi_image" else subsets["ALL"]
                yy, g = y[msk], gp[msk]
                PA, PB = [p[msk] for p in arms[a][0]], [p[msk] for p in arms[b][0]]
            boots[f"{where}: {a} - {b} [{mname}]"] = paired_cluster_boot(yy, PA, PB, g, fn, n=1000)
    res["bootstrap"] = boots
    (HERE / "results.json").write_text(json.dumps(res, indent=2) + "\n")

    # ---- markdown
    def row(name, label):
        r = res["arms"][name]
        c, d = r["cv"]["ALL"], r["dev"]
        sd = f" ±{c['qwk_seed_sd']:.3f}" if c["qwk_seed_sd"] else ""
        return (f"| {label} | {c['qwk']:.3f}{sd} | {c['macro_f1']:.3f} | {c['acc']:.3f} | "
                f"{r['cv']['ostrava']['qwk']:.3f} / {r['cv']['ostrava']['macro_f1']:.3f} | "
                f"{r['cv']['multi_image']['qwk']:.3f} / {r['cv']['multi_image']['macro_f1']:.3f} | "
                f"{d['qwk']:.3f} | {d['macro_f1']:.3f} |")
    print("| encoder / method | CV bag QWK | CV bag mF1 | CV acc | Ostrava sess QWK/mF1 | multi-img bags QWK/mF1 | dev eye QWK | dev eye mF1 |")
    print("|---|---|---|---|---|---|---|---|")
    for e in encs:
        for m in METHODS:
            print(row(f"{e}|{m}", f"{NICE[e]} · {m}"))
    print(row(REF, "**fine-tuned EffNetV2-S structured (s42), max of photos**"))
    print(row("SIZE-ONLY control", "size-only control (bag size bins → LR)"))
    print("\nsize AUC:", json.dumps(res["size_auc"]))
    print("selection:", json.dumps(res["selection"]))
    print("\nper-class recall (CV pooled | dev):")
    for name in [f"{best_e}|{m}" for m in METHODS] + [REF]:
        cv, dv = res["arms"][name]["cv"]["ALL"]["recall"], res["arms"][name]["dev"]["recall"]
        print(f"| {name} | " + " | ".join(f"{cv[c]:.2f}/{dv.get(c, float('nan')):.2f}" for c in CLASSES) + " |")
    print("\nper-source (QWK/mF1), best encoder:")
    for name in [f"{best_e}|{m}" for m in METHODS] + [REF]:
        cv = res["arms"][name]["cv"]
        print(f"| {name} | " + " | ".join(f"{cv[s]['qwk']:.3f}/{cv[s]['macro_f1']:.3f}"
                                          for s in ("ostrava", "ropvl", "shenzhen", "single_image", "multi_image")) + " |")
    print("\nbootstraps:")
    for k, v in boots.items():
        print(f"  {k}: a {v['a']} {v['a_ci']}  b {v['b']} {v['b_ci']}  diff {v['diff']} {v['diff_ci']}")
    print("\nper-fold QWK:")
    for name in res["arms"]:
        print(f"  {name}: {[round(x, 3) for x in res['arms'][name]['cv']['per_fold_qwk']]}  "
              f"mF1 {[round(x, 3) for x in res['arms'][name]['cv']['per_fold_macro_f1']]}")


if __name__ == "__main__":
    main()
