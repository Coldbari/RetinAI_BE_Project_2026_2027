"""Phase 1 — honest ROP statistics (S4, S5, S6, S7).

S4  patient-level metrics with cluster-bootstrap CIs (resampling infants, not images)
S5  re-select the operating threshold on VAL, then measure it once on TEST
S6  McNemar + DeLong on the architecture choice (ResNet50 vs EfficientNetV2-S)
S7  bootstrap CIs on every headline number computable from the on-disk predictions

SCORING PATH — read before quoting anything from here. This reads `preds_*.npz`, which are
SINGLE-VIEW predictions. Serving averages 5 TTA views, so the sensitivity/specificity printed
here are NOT the deployed pair. At the same threshold 0.1933 the two paths give:

    single view (here)                 sens 0.982 / spec 0.456
    5-view TTA (results/rop/operating_point.json, what the app serves)
                                       sens 0.990 / spec 0.398

Both are correct for their own path. Mixing one from each — an AUC from here beside a
sensitivity from there — produces a pair that describes no real system. For anything
user-facing or in the model card, use operating_point.json.

    .venv/bin/python scripts/rop_statistical_analysis.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import yaml
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from models.evaluation.statistical_tests import (  # noqa: E402
    bootstrap_ci, clopper_pearson, cluster_bootstrap_ci, delong_test, mcnemar_test)

P = ROOT / "results" / "rop"


def _deployed_threshold():
    """Read the operating point from the config that serving reads.

    This was hardcoded to 0.371 with a comment claiming it came from configs/rop.yaml. It did
    not: task S5/U4 moved the deployed value to 0.1933 (0.371 was read off the TEST split, so
    quoting performance at it is circular), and the constant here was never updated. Every
    'at the deployed threshold' sensitivity this script printed was therefore computed at a
    threshold the app does not use. Reading the file makes that class of drift impossible.
    """
    cfg = yaml.safe_load((ROOT / "configs" / "rop.yaml").read_text())
    thr = cfg.get("eval", {}).get("screen_threshold")
    if thr is None:
        raise SystemExit("configs/rop.yaml has no eval.screen_threshold")
    return float(thr)


DEPLOYED_THR = _deployed_threshold()
TARGET_SENS = 0.90


def load(tag, split):
    d = np.load(P / f"preds_{tag}_{split}.npz", allow_pickle=True)
    paths = [str(x) for x in d["image_path"]]
    return {"y": d["y_true"].astype(int), "prob": d["y_prob"][:, 1],
            "pred": d["y_pred"].astype(int), "paths": paths,
            "patient": np.array([Path(p).name.split("_")[0] for p in paths])}


def sens_spec(y, prob, thr):
    pred = (prob >= thr).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum()); fn = int(((pred == 0) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
    return tp, fn, tn, fp


def threshold_for_sensitivity(y, prob, target):
    """Lowest threshold achieving >= target sensitivity (highest specificity among those)."""
    best = None
    for thr in np.unique(np.round(prob, 4)):
        tp, fn, tn, fp = sens_spec(y, prob, thr)
        sens = tp / max(tp + fn, 1)
        spec = tn / max(tn + fp, 1)
        if sens >= target and (best is None or spec > best[2]):
            best = (float(thr), sens, spec)
    return best or (0.5, 0.0, 0.0)


def main():
    out, L = {}, []

    def say(s=""):
        print(s); L.append(s)

    test = load("weights", "test")
    val = load("weights", "val")

    # ── S4: what the test set actually contains ─────────────────────────────
    n_img = len(test["y"])
    pats = np.unique(test["patient"])
    pos_pats = np.unique(test["patient"][test["y"] == 1])
    say("=" * 78)
    say("S4 — PATIENT-LEVEL ROP METRICS (ResNet50, deployed checkpoint)")
    say("=" * 78)
    say(f"test split: {n_img} images from {len(pats)} infants "
        f"({len(pos_pats)} with ROP, {len(pats)-len(pos_pats)} without)")
    say(f"images per infant: median {int(np.median([np.sum(test['patient']==p) for p in pats]))}, "
        f"max {max(np.sum(test['patient']==p) for p in pats)}")
    say("")

    auc_i, lo_i, hi_i = bootstrap_ci(test["y"], test["prob"], roc_auc_score, n_boot=2000)
    auc_c, lo_c, hi_c, ncl = cluster_bootstrap_ci(
        test["y"], test["prob"], test["patient"], roc_auc_score, n_boot=2000)
    say(f"AUC (image-level bootstrap)   {auc_i:.4f}  95% CI [{lo_i:.3f}, {hi_i:.3f}]  "
        f"width {hi_i-lo_i:.3f}")
    say(f"AUC (patient-level bootstrap) {auc_c:.4f}  95% CI [{lo_c:.3f}, {hi_c:.3f}]  "
        f"width {hi_c-lo_c:.3f}   <- honest ({ncl} clusters)")
    say(f"the image-level interval is {(hi_c-lo_c)/(hi_i-lo_i):.1f}x too narrow")
    say("")

    tp, fn, tn, fp = sens_spec(test["y"], test["prob"], DEPLOYED_THR)
    s, slo, shi = clopper_pearson(tp, tp + fn)
    p, plo, phi = clopper_pearson(tn, tn + fp)
    say(f"at the deployed threshold {DEPLOYED_THR}:")
    say(f"  sensitivity {s:.3f}  95% CI [{slo:.3f}, {shi:.3f}]   ({tp}/{tp+fn} images)")
    say(f"  specificity {p:.3f}  95% CI [{plo:.3f}, {phi:.3f}]   ({tn}/{tn+fp} images)")
    say("")
    say("PER-INFANT (an infant counts as ROP-detected if ANY of their images is flagged):")
    det = [int((test["prob"][test["patient"] == q] >= DEPLOYED_THR).any()) for q in pats]
    truth = [int((test["y"][test["patient"] == q] == 1).any()) for q in pats]
    det, truth = np.array(det), np.array(truth)
    ptp = int(((det == 1) & (truth == 1)).sum()); pfn = int(((det == 0) & (truth == 1)).sum())
    ptn = int(((det == 0) & (truth == 0)).sum()); pfp = int(((det == 1) & (truth == 0)).sum())
    ps, pslo, pshi = clopper_pearson(ptp, ptp + pfn)
    pp, pplo, pphi = clopper_pearson(ptn, ptn + pfp)
    say(f"  sensitivity {ps:.3f}  95% CI [{pslo:.3f}, {pshi:.3f}]   ({ptp}/{ptp+pfn} infants)")
    say(f"  specificity {pp:.3f}  95% CI [{pplo:.3f}, {pphi:.3f}]   ({ptn}/{ptn+pfp} infants)")
    say(f"  -> with only {ptp+pfn} ROP-positive infants, the sensitivity interval spans "
        f"{pshi-pslo:.2f}. This is the binding constraint on the ROP claim.")
    out["s4"] = {"n_images": n_img, "n_patients": int(len(pats)),
                 "n_positive_patients": int(len(pos_pats)),
                 "auc_image_ci": [auc_i, lo_i, hi_i], "auc_patient_ci": [auc_c, lo_c, hi_c],
                 "image_sens": [s, slo, shi], "image_spec": [p, plo, phi],
                 "patient_sens": [ps, pslo, pshi], "patient_spec": [pp, pplo, pphi]}

    # ── S4b: patient-weighted AUC (UPGRADE) ─────────────────────────────────
    # The image-level AUC is dominated by whoever was photographed most: one infant
    # contributes 470/1502 images (31%). Weight each image by 1/n_images_for_that_infant
    # so every infant counts equally, and report the leave-one-infant-out swing.
    counts = {q: int((test["patient"] == q).sum()) for q in pats}
    w = np.array([1.0 / counts[q] for q in test["patient"]])
    auc_w = roc_auc_score(test["y"], test["prob"], sample_weight=w)

    def _wauc(y, s, idx=None):
        return roc_auc_score(y, s)

    loo = []
    for q in pats:
        m = test["patient"] != q
        if len(np.unique(test["y"][m])) > 1:
            loo.append((roc_auc_score(test["y"][m], test["prob"][m]), q))
    loo.sort()
    top = sorted(counts.items(), key=lambda kv: -kv[1])[:3]
    say("")
    say("S4b — HOW CONCENTRATED IS THE TEST SET? (upgrade)")
    say(f"  largest infant: {top[0][1]}/{n_img} images = {top[0][1]/n_img:.1%} of the test set")
    say(f"  top 3 infants : {sum(c for _, c in top)/n_img:.1%} of the test set")
    say(f"  image-level AUC    {auc_i:.4f}   (each IMAGE counts once - biased to the "
        f"most-photographed infants)")
    say(f"  patient-weighted   {auc_w:.4f}   (each INFANT counts once)")
    say(f"  leave-one-infant-out AUC ranges {loo[0][0]:.4f} - {loo[-1][0]:.4f} "
        f"(swing {loo[-1][0]-loo[0][0]:.4f} from removing ONE infant)")
    out["s4b"] = {"largest_patient_share": top[0][1] / n_img,
                  "auc_image_weighted": float(auc_i), "auc_patient_weighted": float(auc_w),
                  "loo_min": float(loo[0][0]), "loo_max": float(loo[-1][0])}

    # ── S5: threshold provenance ────────────────────────────────────────────
    say("")
    say("=" * 78)
    say("S5 — OPERATING THRESHOLD PROVENANCE")
    say("=" * 78)
    v_thr, v_sens, v_spec = threshold_for_sensitivity(val["y"], val["prob"], TARGET_SENS)
    t_thr, t_sens, t_spec = threshold_for_sensitivity(test["y"], test["prob"], TARGET_SENS)
    say(f"threshold chosen on VAL  for sens>={TARGET_SENS}: {v_thr:.3f} "
        f"(val sens {v_sens:.3f} / spec {v_spec:.3f})")
    say(f"threshold chosen on TEST for sens>={TARGET_SENS}: {t_thr:.3f} "
        f"(test sens {t_sens:.3f} / spec {t_spec:.3f})")
    say(f"deployed threshold in configs/rop.yaml: {DEPLOYED_THR}")
    say("")
    tp2, fn2, tn2, fp2 = sens_spec(test["y"], test["prob"], v_thr)
    hs = tp2 / max(tp2 + fn2, 1); hp = tn2 / max(tn2 + fp2, 1)
    say("HONEST estimate — apply the VAL-selected threshold once to TEST:")
    say(f"  thr={v_thr:.3f} -> test sensitivity {hs:.3f}, specificity {hp:.3f}")
    say(f"OPTIMISTIC (currently reported) — threshold picked on TEST itself:")
    say(f"  thr={t_thr:.3f} -> test sensitivity {t_sens:.3f}, specificity {t_spec:.3f}")
    say(f"  specificity is overstated by {t_spec-hp:+.3f} by selecting on the test set.")
    out["s5"] = {"val_threshold": v_thr, "test_threshold": t_thr, "deployed": DEPLOYED_THR,
                 "honest_test_sens": hs, "honest_test_spec": hp,
                 "optimistic_test_sens": t_sens, "optimistic_test_spec": t_spec}

    # ── S6: architecture comparison ─────────────────────────────────────────
    say("")
    say("=" * 78)
    say("S6 — ARCHITECTURE COMPARISON (paired, same test images)")
    say("=" * 78)
    out["s6"] = {}
    for rival in ("weights_v2s_reg", "weights_overfit_v2s"):
        r = load(rival, "test")
        assert (r["y"] == test["y"]).all(), "paired comparison requires identical labels"
        d = delong_test(test["y"], test["prob"], r["prob"])
        m = mcnemar_test(test["y"], (test["prob"] >= DEPLOYED_THR).astype(int),
                         (r["prob"] >= DEPLOYED_THR).astype(int))
        say(f"ResNet50 vs {rival} (efficientnet_v2_s):")
        say(f"  DeLong : AUC {d['auc_a']:.4f} vs {d['auc_b']:.4f}  "
            f"z={d['z']:.3f}  p={d['p_value']:.4f}  "
            f"{'SIGNIFICANT' if d['p_value'] < 0.05 else 'NOT significant'}")
        say(f"  McNemar: b={m['b']} c={m['c']}  p={m['p_value']:.4f} ({m['test']})  "
            f"{'SIGNIFICANT' if m['p_value'] < 0.05 else 'NOT significant'}")
        out["s6"][rival] = {"delong": d, "mcnemar": m}
    # UPGRADE: do the patient-level test rather than only caveating DeLong.
    say("")
    say("PATIENT-LEVEL AUC DIFFERENCE (cluster bootstrap over infants — upgrade):")
    say("DeLong assumes independent samples; 1,502 images from 29 infants violates that, so")
    say("its p-values are anti-conservative. This resamples INFANTS instead.")
    rng = np.random.default_rng(42)
    idx_by = {q: np.flatnonzero(test["patient"] == q) for q in pats}
    for rival in ("weights_v2s_reg", "weights_overfit_v2s"):
        r = load(rival, "test")
        diffs = []
        for _ in range(2000):
            picked = rng.choice(pats, size=len(pats), replace=True)
            idx = np.concatenate([idx_by[q] for q in picked])
            if len(np.unique(test["y"][idx])) < 2:
                continue
            diffs.append(roc_auc_score(test["y"][idx], test["prob"][idx])
                         - roc_auc_score(r["y"][idx], r["prob"][idx]))
        lo, hi = np.percentile(diffs, [2.5, 97.5])
        crosses = lo <= 0 <= hi
        say(f"  ResNet50 - {rival}: dAUC {np.mean(diffs):+.4f} "
            f"95% CI [{lo:+.4f}, {hi:+.4f}]  "
            f"{'NOT significant (CI spans 0)' if crosses else 'significant'}")
        out["s6"][rival]["cluster_delta_auc"] = [float(np.mean(diffs)), float(lo), float(hi)]

    # ── S7: CIs on the headline numbers ─────────────────────────────────────
    say("")
    say("=" * 78)
    say("S7 — HEADLINE METRICS WITH INTERVALS")
    say("=" * 78)
    say(f"{'metric':<26}{'point':>8}{'95% CI (patient-level)':>28}")
    say("-" * 78)
    for name, fn in (("AUC", roc_auc_score),
                     ("accuracy", lambda y, s: accuracy_score(y, (s >= DEPLOYED_THR).astype(int))),
                     ("macro F1", lambda y, s: f1_score(y, (s >= DEPLOYED_THR).astype(int),
                                                        average="macro"))):
        pt, lo, hi, _ = cluster_bootstrap_ci(test["y"], test["prob"], test["patient"], fn)
        say(f"{name:<26}{pt:>8.4f}{f'[{lo:.3f}, {hi:.3f}]':>28}")
        out.setdefault("s7", {})[name] = [pt, lo, hi]

    (ROOT / "results" / "rop_statistics.json").write_text(json.dumps(out, indent=2, default=float))
    (ROOT / "results" / "rop_statistics.md").write_text(
        "# ROP — patient-level statistical analysis\n\n```\n" + "\n".join(L) + "\n```\n")
    print("\n-> results/rop_statistics.json, results/rop_statistics.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
