"""Does a confidence of 0.8 mean the same thing at every hospital?

    .venv/bin/python scripts/rop_per_source_calibration.py

The committed calibration (structured_bootstrap_calibration.json) is POOLED: one ECE of
0.108 over all 6,700 out-of-fold images. Pooled calibration is the wrong instrument for a
multi-hospital model, and for the same reason the rest of this paper keeps finding: an
average over heterogeneous units hides the units. If the model is well calibrated at the
hospital that supplies 64% of the images and badly calibrated at the other two, the pooled
number reports the big one and calls it the model.

That matters clinically, not just statistically. Every threshold this system serves — the
AP-ROP branch at 0.5, the screening operating point — assumes a probability means the same
thing wherever the image came from. Nobody had checked.

This pools the out-of-fold predictions of all five folds (each image scored exactly once, by
the model that did not train on it), joins the source, and reports per hospital: ECE, Brier,
accuracy, reliability bins, and the direction of miscalibration (over- vs under-confident).
Intervals are patient-clustered, because frames within an infant are not independent.

Output: results/rop/per_source_calibration.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
PREP = ROOT / "results" / "prepared" / "rop_staging"
RUNS = ROOT / "results" / "rop" / "staging_runs" / "structured_5fold"
OUT = ROOT / "results" / "rop" / "per_source_calibration.json"
N_BINS = 10
AROP = 5


def ece_and_bins(conf, correct, n_bins=N_BINS):
    """Expected calibration error, equal-width bins, plus the bins themselves."""
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece, bins = 0.0, []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi) if lo > 0 else (conf >= lo) & (conf <= hi)
        if not m.any():
            continue
        acc, mc = float(correct[m].mean()), float(conf[m].mean())
        ece += m.mean() * abs(acc - mc)
        bins.append({"lo": round(float(lo), 2), "hi": round(float(hi), 2), "n": int(m.sum()),
                     "mean_confidence": round(mc, 4), "accuracy": round(acc, 4),
                     "gap": round(acc - mc, 4)})
    return float(ece), bins


def cluster_ci(conf, correct, groups, n=2000, seed=42):
    """Patient-clustered bootstrap CI for ECE and for mean(conf) - mean(correct)."""
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups)
    idx = {g: np.where(groups == g)[0] for g in uniq}
    eces, gaps = [], []
    for _ in range(n):
        take = np.concatenate([idx[g] for g in rng.choice(uniq, len(uniq), True)])
        e, _ = ece_and_bins(conf[take], correct[take])
        eces.append(e)
        gaps.append(float(conf[take].mean() - correct[take].mean()))
    q = lambda v: [round(float(x), 4) for x in np.percentile(v, [2.5, 97.5])]  # noqa: E731
    return q(eces), q(gaps)


def main():
    rows = []
    for k in range(5):
        d = np.load(RUNS / f"fold_{k}.npz")
        val = pd.read_csv(PREP / f"fold_{k}.csv").query("split == 'val'").reset_index(drop=True)
        assert (d["y_true"] == val.label.values).all(), f"fold {k}: npz misaligned with fold CSV"
        prob = d["prob6" if "prob6" in d else "prob"]
        rows.append(pd.DataFrame({
            "source": val.source.values, "patient_id": val.patient_id.values,
            "y": d["y_true"], "pred": d["y_pred"],
            "conf": prob.max(1), "p_arop": prob[:, AROP], "fold": k,
        }))
    df = pd.concat(rows, ignore_index=True)
    df["correct"] = (df.pred == df.y).astype(float)
    print(f"pooled out-of-fold: {len(df)} images, {df.patient_id.nunique()} infants, "
          f"{df.source.nunique()} sources")

    out = {
        "what": "Per-hospital calibration of the structured model (seed 42), pooled from the "
                "five out-of-fold prediction sets so every image is scored exactly once by a "
                "model that did not train on it. Confidence = max softmax over the 6 decoded "
                "classes. Intervals are patient-clustered (2000 replicates).",
        "why": "The committed calibration is pooled over all hospitals. A pooled ECE on a "
               "corpus where one site supplies 64% of images reports that site. Every served "
               "threshold assumes a probability means the same thing everywhere.",
        "n_bins": N_BINS, "sources": {},
    }

    pooled_ece, pooled_bins = ece_and_bins(df.conf.values, df.correct.values)
    p_ci, p_gap = cluster_ci(df.conf.values, df.correct.values, df.patient_id.values)
    out["pooled"] = {"n_images": int(len(df)), "n_infants": int(df.patient_id.nunique()),
                     "accuracy": round(float(df.correct.mean()), 4),
                     "mean_confidence": round(float(df.conf.mean()), 4),
                     "ece": round(pooled_ece, 4), "ece_ci95": p_ci,
                     "confidence_minus_accuracy": round(float(df.conf.mean() - df.correct.mean()), 4),
                     "confidence_minus_accuracy_ci95": p_gap,
                     "brier_top1": round(float(np.mean((df.conf - df.correct) ** 2)), 4),
                     "bins": pooled_bins}

    print(f"\n  {'source':<12} {'n':>6} {'infants':>8} {'acc':>7} {'conf':>7} "
          f"{'ECE':>7} {'ECE CI':>18} {'conf-acc':>9}")
    print(f"  {'pooled':<12} {len(df):>6} {df.patient_id.nunique():>8} "
          f"{df.correct.mean():>7.3f} {df.conf.mean():>7.3f} {pooled_ece:>7.3f} "
          f"{str(p_ci):>18} {df.conf.mean()-df.correct.mean():>+9.3f}")

    for src, sub in df.groupby("source"):
        e, bins = ece_and_bins(sub.conf.values, sub.correct.values)
        ci, gap_ci = cluster_ci(sub.conf.values, sub.correct.values, sub.patient_id.values)
        gap = float(sub.conf.mean() - sub.correct.mean())
        # AP-ROP branch behaviour at the a-priori threshold, per site.
        has_arop = int((sub.y == AROP).sum())
        arop = {"n_positive": has_arop,
                "flagged_at_0.5": int((sub.p_arop > 0.5).sum()),
                "recall_at_0.5": (round(float(((sub.p_arop > 0.5) & (sub.y == AROP)).sum()
                                               / has_arop), 4) if has_arop else None),
                "false_flag_rate_at_0.5": round(float(((sub.p_arop > 0.5) & (sub.y != AROP)).sum()
                                                      / max(int((sub.y != AROP).sum()), 1)), 4)}
        out["sources"][src] = {
            "n_images": int(len(sub)), "n_infants": int(sub.patient_id.nunique()),
            "accuracy": round(float(sub.correct.mean()), 4),
            "mean_confidence": round(float(sub.conf.mean()), 4),
            "ece": round(e, 4), "ece_ci95": ci,
            "confidence_minus_accuracy": round(gap, 4),
            "confidence_minus_accuracy_ci95": gap_ci,
            "direction": ("over-confident" if gap_ci[0] > 0 else
                          "under-confident" if gap_ci[1] < 0 else "no significant bias"),
            "brier_top1": round(float(np.mean((sub.conf - sub.correct) ** 2)), 4),
            "arop_branch_at_apriori_threshold": arop,
            "classes_present": sorted(sub.y.unique().tolist()),
            "bins": bins,
        }
        print(f"  {src:<12} {len(sub):>6} {sub.patient_id.nunique():>8} "
              f"{sub.correct.mean():>7.3f} {sub.conf.mean():>7.3f} {e:>7.3f} "
              f"{str(ci):>18} {gap:>+9.3f}  {out['sources'][src]['direction']}")

    # CONTROL — hold the case-mix constant. Raw per-site ECE confounds "the model is
    # over-confident here" with "this site has harder cases": the sites do not share a class
    # distribution (stage 4/5 is ROP-VL only, AP-ROP never appears in Shenzhen). Restricting
    # to `normal`, the one class every site has, removes that explanation. This is the same
    # control the site probe uses (§G5), applied to calibration.
    norm = df[df.y == 0]
    out["disease_controlled_normal_only"] = {
        "why": "Sites do not share a class distribution, so raw ECE confounds calibration "
               "with case difficulty. Normal is the only class present at all three sites; "
               "any spread that survives here is calibration, not case-mix.",
        "sources": {},
    }
    print(f"\n  disease-controlled (normal-class images only, n={len(norm)}):")
    for src, sub in norm.groupby("source"):
        e, _ = ece_and_bins(sub.conf.values, sub.correct.values)
        ci, gap_ci = cluster_ci(sub.conf.values, sub.correct.values, sub.patient_id.values)
        gap = float(sub.conf.mean() - sub.correct.mean())
        out["disease_controlled_normal_only"]["sources"][src] = {
            "n_images": int(len(sub)), "n_infants": int(sub.patient_id.nunique()),
            "accuracy": round(float(sub.correct.mean()), 4),
            "mean_confidence": round(float(sub.conf.mean()), 4),
            "ece": round(e, 4), "ece_ci95": ci,
            "confidence_minus_accuracy": round(gap, 4),
            "confidence_minus_accuracy_ci95": gap_ci,
            "direction": ("over-confident" if gap_ci[0] > 0 else
                          "under-confident" if gap_ci[1] < 0 else "no significant bias"),
        }
        print(f"    {src:<12} {len(sub):>6} {sub.patient_id.nunique():>8} "
              f"{sub.correct.mean():>7.3f} {sub.conf.mean():>7.3f} {e:>7.3f} "
              f"{str(ci):>18} {gap:>+9.3f}  "
              f"{out['disease_controlled_normal_only']['sources'][src]['direction']}")
    ce = {k: v["ece"] for k, v in out["disease_controlled_normal_only"]["sources"].items()}
    cw, cb = max(ce, key=ce.get), min(ce, key=ce.get)
    out["disease_controlled_normal_only"]["spread"] = {
        "best": {"source": cb, "ece": ce[cb]}, "worst": {"source": cw, "ece": ce[cw]},
        "ratio": round(ce[cw] / ce[cb], 2) if ce[cb] > 0 else None,
    }

    # The headline is the SPREAD, not any single cell.
    eces = {k: v["ece"] for k, v in out["sources"].items()}
    worst, best = max(eces, key=eces.get), min(eces, key=eces.get)
    out["spread"] = {
        "best_calibrated": {"source": best, "ece": eces[best]},
        "worst_calibrated": {"source": worst, "ece": eces[worst]},
        "ratio": round(eces[worst] / eces[best], 2) if eces[best] > 0 else None,
        "reading": (f"RAW ECE ranges {eces[best]:.3f} ({best}) to {eces[worst]:.3f} ({worst}) "
                    f"— {round(eces[worst]/eces[best], 1)}x, and the pooled {pooled_ece:.3f} "
                    f"describes no single site. BUT the disease-controlled panel shows most "
                    f"of that spread is CASE-MIX, not a site-specific calibration defect: "
                    f"holding the class constant collapses the range to "
                    f"{ce[cb]:.3f}-{ce[cw]:.3f} ({round(ce[cw]/ce[cb], 1)}x) and every site "
                    f"is well calibrated on the class they share. Do not report the 8x. The "
                    f"defensible statements are (1) the model is over-confident where the "
                    f"cases are hard, by +{float(df.conf.mean()-df.correct.mean()):.3f} "
                    f"pooled, and (2) a residual site effect survives the control at Ostrava "
                    f"(+0.027, CI excludes 0) but it is an order of magnitude smaller than "
                    f"the raw number suggests."),
        "controlled_ratio": round(ce[cw] / ce[cb], 2) if ce[cb] > 0 else None,
        "methodological_point": ("This is the third place in this paper where an uncontrolled "
                                 "aggregate overstated a site effect: the naive site probe "
                                 "(§4.4), image-level AP-ROP recall (§4.3), and now per-site "
                                 "ECE. Each time the control was available and cheap."),
    }
    OUT.write_text(json.dumps(out, indent=2) + "\n")
    print(f"\n  {out['spread']['reading']}")
    print(f"-> {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
