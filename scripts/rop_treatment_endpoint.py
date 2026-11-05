"""The clinical question macro-F1 does not answer: would this infant be referred?

    .venv/bin/python scripts/rop_treatment_endpoint.py

Every metric in this project so far scores the 6-class taxonomy. No clinician acts on a
macro-F1. The decision a screening system actually feeds is binary — does this infant need
to be seen by someone who can treat — and a model can be mediocre at 6-class staging while
being excellent or dangerous at that decision, because the two aggregate errors completely
differently. Confusing stage 1 with stage 2 costs nothing here; confusing stage 3 with
normal costs an eye.

WHAT WE CAN AND CANNOT COMPUTE. ICROP-3 Type 1 (treatment-warranted) is defined on
zone x stage x plus, and this corpus has no usable zone axis (Multi-View records zone for
13 of 232 patients; the others record none). We therefore do NOT claim to detect Type 1.
We score two defensible collapses of the labels we do have:

  REFERRAL-WARRANTED   any ROP  (stage 1+, incl. AP-ROP)  — "this infant has disease"
  SEVERE               stage 3, stage 4/5, or AP-ROP      — the end of the ladder where
                       treatment decisions live. AP-ROP is Type 1 by definition [C3];
                       stage 4/5 is past the treatment threshold; stage 3 is Type 1 in
                       zone I, or in zone II with plus.

SEVERE deliberately OVER-includes: a stage 3 in zone III without plus is not Type 1. It is
a sensitivity-oriented proxy and is named one. The direction of the bias is stated because
an over-inclusive positive class flatters specificity-free reporting and we do not want to
be read as claiming Type 1 performance.

BOTH UNITS. Scored per image and per examination session (the unit Ostrava's labels are
actually recorded at — see the session-level analysis), because this project's central
finding is that the two differ.

MISS ANATOMY. For every missed SEVERE case we record what the model said instead. A severe
case predicted as stage 2 is a near miss that a graded referral protocol still catches; one
predicted as normal is the failure that matters. Nobody reports this and it is the single
most decision-relevant thing in the confusion matrix.

Output: results/rop/treatment_endpoint.json
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
PREP = ROOT / "results" / "prepared" / "rop_staging"
RUNS = ROOT / "results" / "rop" / "staging_runs" / "structured_5fold"
OUT = ROOT / "results" / "rop" / "treatment_endpoint.json"

CLASSES = ["normal", "stage_1", "stage_2", "stage_3", "stage_4_5", "arop"]
SEVERE = {3, 4, 5}        # stage 3, stage 4/5, AP-ROP
ANY_ROP = {1, 2, 3, 4, 5}


def session_key(row) -> str:
    """Ostrava filenames encode device+serie — one imaging session. Other sources have no
    session structure, so the patient is the unit. (This fallback is the one that was wrong
    once: falling back to the FILENAME made every ROP-VL image its own 'session'.)"""
    m = re.search(r"/(\d+)_D(\d)_S(\d+)_\d+\.jpg$", row.image_path)
    return f"{row.patient_id}|D{m.group(2)}S{m.group(3)}" if m else row.patient_id


def exact_ci(k: int, n: int, alpha: float = 0.05):
    from scipy.stats import beta
    lo = beta.ppf(alpha / 2, k, n - k + 1) if k > 0 else 0.0
    hi = beta.ppf(1 - alpha / 2, k + 1, n - k) if k < n else 1.0
    return [round(float(lo), 4), round(float(hi), 4)]


def binary_report(y, p, groups, label):
    """Sensitivity/specificity/PPV/NPV with Clopper-Pearson, plus patient-clustered
    sensitivity — the exact-CI denominators are images, which are not independent."""
    tp = int(((y == 1) & (p == 1)).sum()); fn = int(((y == 1) & (p == 0)).sum())
    fp = int(((y == 0) & (p == 1)).sum()); tn = int(((y == 0) & (p == 0)).sum())
    sens = tp / (tp + fn) if tp + fn else None
    spec = tn / (tn + fp) if tn + fp else None
    rep = {
        "unit": label, "n": int(len(y)), "n_positive": int((y == 1).sum()),
        "tp": tp, "fn": fn, "fp": fp, "tn": tn,
        "sensitivity": round(sens, 4) if sens is not None else None,
        "sensitivity_ci95_exact": exact_ci(tp, tp + fn) if tp + fn else None,
        "specificity": round(spec, 4) if spec is not None else None,
        "specificity_ci95_exact": exact_ci(tn, tn + fp) if tn + fp else None,
        "ppv": round(tp / (tp + fp), 4) if tp + fp else None,
        "npv": round(tn / (tn + fn), 4) if tn + fn else None,
    }
    # Patient-clustered bootstrap on sensitivity — the honest interval when one infant can
    # supply hundreds of positive frames.
    if groups is not None and tp + fn:
        rng = np.random.default_rng(42)
        uniq = np.unique(groups)
        idx = {g: np.where(groups == g)[0] for g in uniq}
        vals = []
        for _ in range(2000):
            take = np.concatenate([idx[g] for g in rng.choice(uniq, len(uniq), True)])
            yy, pp = y[take], p[take]
            pos = yy == 1
            if pos.sum():
                vals.append(float((pp[pos] == 1).mean()))
        rep["sensitivity_ci95_patient_clustered"] = [round(float(x), 4) for x in
                                                     np.percentile(vals, [2.5, 97.5])]
        rep["ci_note"] = ("the clustered interval is the one to quote; the exact interval "
                          "treats correlated frames from one infant as independent evidence")
    return rep


def main():
    rows = []
    for k in range(5):
        d = np.load(RUNS / f"fold_{k}.npz")
        val = pd.read_csv(PREP / f"fold_{k}.csv").query("split == 'val'").reset_index(drop=True)
        assert (d["y_true"] == val.label.values).all(), f"fold {k}: npz misaligned"
        rows.append(pd.DataFrame({
            "image_path": val.image_path.values, "patient_id": val.patient_id.values,
            "source": val.source.values, "y": d["y_true"], "pred": d["y_pred"], "fold": k}))
    df = pd.concat(rows, ignore_index=True)
    df["session"] = df.apply(session_key, axis=1)
    print(f"pooled out-of-fold: {len(df)} images, {df.session.nunique()} sessions, "
          f"{df.patient_id.nunique()} infants")

    out = {
        "what": "The 6-class predictions re-scored as the binary decisions a screening "
                "service actually makes, at both the image and the examination-session unit.",
        "definitions": {
            "referral_warranted": "any ROP: stage 1, 2, 3, 4/5 or AP-ROP",
            "severe": "stage 3, stage 4/5, or AP-ROP — the end of the ladder where "
                      "treatment decisions live",
            "why_not_type_1": "ICROP-3 Type 1 is defined on zone x stage x plus. This "
                              "corpus has no usable zone axis (Multi-View records zone for "
                              "13 of 232 patients; the rest record none), so Type 1 cannot "
                              "be computed and is not claimed.",
            "bias_direction": "SEVERE over-includes — a stage 3 in zone III without plus is "
                              "not Type 1. It is a sensitivity-oriented proxy, stated so "
                              "that it is not read as Type 1 performance.",
        },
        "endpoints": {},
    }

    for name, positive in (("referral_warranted", ANY_ROP), ("severe", SEVERE)):
        yb = df.y.isin(positive).astype(int).values
        pb = df.pred.isin(positive).astype(int).values
        img = binary_report(yb, pb, df.patient_id.values, "image")

        # Session unit: a session is positive if ANY frame carries the label (that is how the
        # label was recorded), and flagged if ANY frame is predicted positive (screening rule).
        s = df.assign(yb=yb, pb=pb).groupby("session").agg(
            yb=("yb", "max"), pb=("pb", "max"), patient_id=("patient_id", "first"))
        ses = binary_report(s.yb.values, s.pb.values, s.patient_id.values, "session")

        out["endpoints"][name] = {"image": img, "session": ses}
        for r in (img, ses):
            print(f"  {name:<20} {r['unit']:<8} n={r['n']:>5} pos={r['n_positive']:>4}  "
                  f"sens {r['sensitivity']:.3f} {r.get('sensitivity_ci95_patient_clustered')}  "
                  f"spec {r['specificity']:.3f}  PPV {r['ppv']:.3f}")

    # ── MISS ANATOMY — what did the model say instead? ────────────────────────
    missed = df[df.y.isin(SEVERE) & ~df.pred.isin(SEVERE)]
    anat = missed.pred.value_counts().sort_index()
    total_sev = int(df.y.isin(SEVERE).sum())
    out["severe_miss_anatomy"] = {
        "severe_images": total_sev, "missed": int(len(missed)),
        "predicted_instead": {CLASSES[int(k)]: int(v) for k, v in anat.items()},
        "adjacent_misses": int(missed.pred.isin([2]).sum()),
        "called_normal": int((missed.pred == 0).sum()),
        "reading": "",
    }
    adj = out["severe_miss_anatomy"]["adjacent_misses"]
    norm = out["severe_miss_anatomy"]["called_normal"]
    out["severe_miss_anatomy"]["reading"] = (
        f"Of {len(missed)} missed severe images, {adj} were called stage 2 — a near miss a "
        f"graded referral protocol still escalates — and {norm} were called normal, which is "
        f"the failure that matters. The ratio, not the miss count, is the clinically "
        f"informative number, and it is a direct consequence of training the stage axis as "
        f"an ORDINAL problem: errors land next door.")
    print(f"\n  severe misses: {len(missed)} of {total_sev} — "
          f"{adj} called stage 2 (adjacent), {norm} called normal")

    # Per-session severity misses, the unit that would actually be escalated.
    sev_s = df.assign(ys=df.y.isin(SEVERE), ps=df.pred.isin(SEVERE)).groupby("session").agg(
        ys=("ys", "max"), ps=("ps", "max"))
    missed_s = sev_s[sev_s.ys & ~sev_s.ps]
    out["severe_miss_anatomy"]["sessions_severe"] = int(sev_s.ys.sum())
    out["severe_miss_anatomy"]["sessions_missed_entirely"] = int(len(missed_s))
    print(f"  sessions with severe disease: {int(sev_s.ys.sum())}, "
          f"missed entirely: {len(missed_s)}")

    OUT.write_text(json.dumps(out, indent=2) + "\n")
    print(f"-> {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
