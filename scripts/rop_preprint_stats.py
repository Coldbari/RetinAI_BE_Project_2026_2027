"""Preprint statistics for the ROP paper — re-scores SAVED predictions; runs NO inference.

    .venv/bin/python -I scripts/rop_preprint_stats.py

NO MODEL IS RUN. No network is built, no checkpoint is opened, and torch is never imported
(asserted before the output is written). test_locked in particular is NOT re-opened: its
predictions were written exactly once, on 2026-08-18, by scripts/rop_locked_eval.py into
results/rop/locked_eval/*.npz, and this script only re-scores those saved arrays.

Units and metrics are the repo's own, imported rather than re-implemented, so every number
that overlaps locked_table.json / dev_table.json reproduces it exactly:
  eye unit       rop_locked_eval.eye_frame — (patient_id, Left/Right) from the Multi-View
                 Acquisition View Table; eye label and eye prediction = max hard class index
  per-eye        rop_locked_eval.eye_metrics      per-image   rop_locked_eval.metrics
  QWK            models.common.metrics.quadratic_weighted_kappa (the sklearn call every
                 script uses)
  CV per-fold    rop_matched_equivalence.fold_metrics
  sessions       rop_session_level_analysis.session_key

What results/rop/preprint_stats.json holds:
  reproduction_check  every locked_table.json / dev_table.json point estimate, plus the
           committed clustered CIs, re-derived BEFORE anything new is computed. The script
           refuses to write if any of them disagrees.
  A  locked half, 4 cells: patient-clustered bootstrap CIs (2,000 replicates, infants
     resampled, seed 42; metric per checkpoint, then averaged over the 5 checkpoints inside
     each replicate) for image/eye QWK, accuracy, eye macro-F1 (present classes), the share
     of stage 1-3 eyes called Normal, and the paired difference eye QWK - image QWK.
  B  dev half: the same image-vs-eye QWK point estimates, from results/rop/dev_eval/*.npz.
  C  unweighted Cohen's kappa: kappa_unweighted_vs_human.json copied with its definitions
     (and re-derived), plus image/eye unweighted kappa on the locked half.
  D  head equivalence: the 10 fold x seed macro-F1 differences from matched_equivalence.json
     (re-derived from staging_runs npz), t-interval and TOST at +/-0.05.
  E  cohort table for the 8,024-image eligible corpus, built from the COMMITTED split CSVs.
  F  AP-ROP internal control per seed: AP-ROP frames scored p<0.1, Ostrava (session-
     stamped) vs frame-labelled sources.

BOOTSTRAP DRAWS. The infant draws replay rop_locked_eval.cluster_ci exactly (same seed, same
rng.choice call sequence, re-seeded per cell), which is why the image-level QWK and
macro-F1 CIs reproduce the committed ones to the last digit. At eye level, a resampled infant
contributes copies of its own eyes. Aggregating eyes after resampling would merge an
infant drawn twice back into one eye, so eyes are formed first and then resampled with
their infant.
"""
from __future__ import annotations

import datetime
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import cohen_kappa_score, f1_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from models.common.metrics import quadratic_weighted_kappa as qwk          # noqa: E402
from scripts.build_rop_staging_manifest import DEVICE                       # noqa: E402
from scripts.rop_locked_eval import (CLASSES, eye_frame, eye_metrics,       # noqa: E402
                                     metrics)
from scripts.rop_matched_equivalence import ARMS, ARMS_S1337, fold_metrics  # noqa: E402
from scripts.rop_session_level_analysis import session_key                  # noqa: E402

PREP = ROOT / "results" / "prepared" / "rop_staging"
RES = ROOT / "results" / "rop"
OUT = RES / "preprint_stats.json"
CELLS = [("structured", 42), ("structured", 1337), ("flat", 42), ("flat", 1337)]
N_BOOT, BOOT_SEED, N_CKPT = 2000, 42, 5
AROP = CLASSES.index("arop")
EQ_MARGIN = 0.05
INPUTS: set[Path] = set()   # every file read, for the provenance block


def read(path: Path) -> Path:
    INPUTS.add(path)
    return path


def r4(x):
    return None if x is None else round(float(x), 4)


def ci95(v):
    return [round(float(x), 4) for x in np.percentile(v, [2.5, 97.5])]


def mean_sd(v):
    return {"mean": r4(np.mean(v)), "sd": r4(np.std(v, ddof=1))}


# ── saved predictions ───────────────────────────────────────────────────────────────────

def load_set(which: str):
    """test_{which}.csv, its eye frame, and the saved per-checkpoint hard predictions."""
    df = pd.read_csv(read(PREP / f"test_{which}.csv")).reset_index(drop=True)
    edf = eye_frame(df)
    INPUTS.add(ROOT / "data/rop/multiview/ROP-DATASET/Dataset Acquisition View Table.xlsx")
    assert len(edf) == len(df) and (edf.image_path.values == df.image_path.values).all(), \
        "eye join changed row order or count - refusing to align saved predictions"
    d = RES / ("locked_eval" if which == "locked" else "dev_eval")
    preds = {}
    for arm, seed in CELLS:
        ps = []
        for k in range(N_CKPT):
            z = np.load(read(d / f"{arm}_s{seed}_ckpt{k}.npz"))
            assert (z["y_true"] == df.label.values).all(), f"{arm}_s{seed}_ckpt{k}: misaligned"
            ps.append(z["y_pred"])
        preds[f"{arm}_s{seed}"] = ps
    return df, edf, preds


def eye_table(edf: pd.DataFrame, pred: np.ndarray) -> pd.DataFrame:
    """The groupby eye_metrics uses, kept as a table so eyes can be resampled."""
    return edf.assign(pred=pred).groupby(["patient_id", "eye"]).agg(
        y=("label", "max"), p=("pred", "max"))


def point_estimates(df, edf, ps):
    """5-checkpoint mean/sd via the repo's own metrics() and eye_metrics()."""
    y = df.label.values
    im = [metrics(y, p) for p in ps]
    em = [eye_metrics(edf, p) for p in ps]
    s123 = np.isin(y, [1, 2, 3])
    eyes = [eye_table(edf, p) for p in ps]
    return {
        "image": {k: mean_sd([m[k] for m in im]) for k in
                  ("qwk", "accuracy", "macro_f1_present", "macro_f1_6label",
                   "macro_f1_classes_ge30")}
                 | {"kappa_unweighted": mean_sd([cohen_kappa_score(y, p) for p in ps]),
                    "stage123_images_called_normal": mean_sd([(p[s123] == 0).mean()
                                                             for p in ps])},
        "eye": {k: mean_sd([m[k] for m in em]) for k in
                ("qwk", "accuracy", "macro_f1_present", "stage123_eyes_called_normal")}
               | {"kappa_unweighted": mean_sd([cohen_kappa_score(e.y, e.p) for e in eyes])},
        "eye_minus_image_qwk": mean_sd([e["qwk"] - i["qwk"] for e, i in zip(em, im)]),
        "n_images": int(len(df)), "n_infants": int(df.patient_id.nunique()),
        "n_eyes": em[0]["n_eyes"], "n_stage123_eyes": em[0]["n_stage123_eyes"],
        "n_stage123_images": int(s123.sum()),
    }


# ── reproduction check (runs before anything new) ────────────────────────────────────────

def check_against(table_path: Path, pe: dict, boot: dict | None) -> list[dict]:
    """Compare every overlapping field with the committed table. Returns the rows checked."""
    t = json.loads(read(table_path).read_text())
    rows = []
    for cell, v in t["cells"].items():
        for key, val in v["primary_per_checkpoint"].items():
            rows.append((cell, f"image.{key}.mean", val["mean"], pe[cell]["image"][key]["mean"]))
        for key, val in v["coprimary_eye_level"].items():
            rows.append((cell, f"eye.{key}.mean", val["mean"], pe[cell]["eye"][key]["mean"]))
        rows.append((cell, "n_eyes", v["n_eyes"], pe[cell]["n_eyes"]))
        rows.append((cell, "n_stage123_eyes", v["n_stage123_eyes"], pe[cell]["n_stage123_eyes"]))
        if boot is not None:
            rows.append((cell, "image.qwk.ci95_clustered", v["primary_qwk_ci95_clustered"],
                         boot[cell]["image_qwk"]["ci95"]))
            rows.append((cell, "image.macro_f1_present.ci95_clustered",
                         v["primary_macro_f1_present_ci95_clustered"],
                         boot[cell]["image_macro_f1_present"]["ci95"]))
    return [{"cell": c, "field": f, "committed": a, "reproduced": b, "match": a == b}
            for c, f, a, b in rows]


# ── A: patient-clustered bootstrap on the locked half ────────────────────────────────────

def bootstrap_cell(df, edf, ps):
    """Mean-over-checkpoints of each metric, per replicate, on one shared set of draws."""
    y, groups = df.label.values, df.patient_id.values
    uniq = np.unique(groups)
    idx = {g: np.where(groups == g)[0] for g in uniq}
    eyes = [eye_table(edf, p) for p in ps]
    for e, p in zip(eyes, ps):      # the resampling table IS eye_metrics' unit
        assert r4(qwk(e.y.values, e.p.values)) == eye_metrics(edf, p)["qwk"]
    ey = eyes[0].y.values
    ep = [e.p.values for e in eyes]
    eye_pid = eyes[0].index.get_level_values("patient_id").values
    eidx = {g: np.where(eye_pid == g)[0] for g in uniq}

    rng = np.random.default_rng(BOOT_SEED)   # == rop_locked_eval.cluster_ci's draws
    keys = ("image_qwk", "image_accuracy", "image_macro_f1_present", "image_kappa_unweighted",
            "image_stage123_called_normal", "eye_qwk", "eye_accuracy", "eye_macro_f1_present",
            "eye_kappa_unweighted", "eye_stage123_called_normal")
    out = {k: [] for k in keys}
    for _ in range(N_BOOT):
        draw = rng.choice(uniq, len(uniq), True)
        t = np.concatenate([idx[g] for g in draw])
        te = np.concatenate([eidx[g] for g in draw])
        yy, ye = y[t], ey[te]
        lab_i, lab_e = sorted(set(yy.tolist())), sorted(set(ye.tolist()))
        si, se = np.isin(yy, [1, 2, 3]), np.isin(ye, [1, 2, 3])
        per = {k: [] for k in keys}
        for p, pe_ in zip(ps, ep):
            pp, pe = p[t], pe_[te]
            per["image_qwk"].append(qwk(yy, pp))
            per["image_accuracy"].append((yy == pp).mean())
            per["image_macro_f1_present"].append(
                f1_score(yy, pp, average="macro", labels=lab_i, zero_division=0))
            per["image_kappa_unweighted"].append(cohen_kappa_score(yy, pp))
            per["image_stage123_called_normal"].append((pp[si] == 0).mean())
            per["eye_qwk"].append(qwk(ye, pe))
            per["eye_accuracy"].append((ye == pe).mean())
            per["eye_macro_f1_present"].append(
                f1_score(ye, pe, average="macro", labels=lab_e, zero_division=0))
            per["eye_kappa_unweighted"].append(cohen_kappa_score(ye, pe))
            per["eye_stage123_called_normal"].append((pe[se] == 0).mean())
        for k in keys:
            out[k].append(float(np.mean(per[k])))
    res = {k: {"ci95": ci95(v)} for k, v in out.items()}
    diff = np.array(out["eye_qwk"]) - np.array(out["image_qwk"])
    res["eye_minus_image_qwk"] = {"ci95": ci95(diff),
                                  "replicates_with_diff_le_0": int((diff <= 0).sum())}
    return res


# ── D: head equivalence ─────────────────────────────────────────────────────────────────

def equivalence():
    me = json.loads(read(RES / "matched_equivalence.json").read_text())
    rows, rederived = [], []
    for seed, arms in (("42", ARMS), ("1337", ARMS_S1337)):
        for row in me["seeds"][seed]["per_fold"]:
            k = row["fold"]
            val = pd.read_csv(read(PREP / f"fold_{k}.csv")).query("split == 'val'")
            y_ref = val.label.values
            s = fold_metrics(read(arms["structured_natural"] / f"fold_{k}.npz"), y_ref)
            f = fold_metrics(read(arms["flat_natural"] / f"fold_{k}.npz"), y_ref)
            d = round(s["macro_f1"] - f["macro_f1"], 4)
            rederived.append(d == row["diff_structured_minus_flat"])
            rows.append({"seed": int(seed), "fold": k, "structured": row["structured"],
                         "flat_natural": row["flat_natural"],
                         "diff_structured_minus_flat": row["diff_structured_minus_flat"]})
    d = np.array([r["diff_structured_minus_flat"] for r in rows])
    n, m, sd = len(d), float(d.mean()), float(d.std(ddof=1))

    def tost(se, label):
        df_ = n - 1
        p_lower = float(stats.t.sf((m + EQ_MARGIN) / se, df_))   # H0: diff <= -margin
        p_upper = float(stats.t.cdf((m - EQ_MARGIN) / se, df_))  # H0: diff >= +margin
        q95, q90 = stats.t.ppf(0.975, df_), stats.t.ppf(0.95, df_)
        return {"variance": label, "se": r4(se), "df": df_,
                "ci95_t": [r4(m - q95 * se), r4(m + q95 * se)],
                "ci90_t_tost_equivalent": [r4(m - q90 * se), r4(m + q90 * se)],
                "p_lower_H0_diff_le_minus_margin": round(p_lower, 5),
                "p_upper_H0_diff_ge_plus_margin": round(p_upper, 5),
                "tost_p": round(max(p_lower, p_upper), 5),
                "equivalent_at_alpha_0.05": bool(max(p_lower, p_upper) < 0.05)}

    ratio = 1 / 4   # 5-fold CV: each val fold is 1/4 the size of its training pool
    return {
        "metric": "macro-F1 over all 6 labels (matched_equivalence.fold_metrics, "
                  "present_only=False); every CV val fold contains all 6 classes, so this "
                  "equals present-class macro-F1 here. Natural sampling, both heads.",
        "source": "results/rop/matched_equivalence.json seeds.*.per_fold",
        "rederived_from_npz_all_match": bool(all(rederived)),
        "pairs": rows, "n_pairs": n, "mean_diff": r4(m), "sd_diff": r4(sd),
        "margin": EQ_MARGIN,
        "naive_paired_t": tost(sd / np.sqrt(n), "naive: s^2/n, treats the 10 pairs as "
                                                "independent"),
        "corrected_resampled_t": tost(sd * np.sqrt(1 / n + ratio),
                                      "Nadeau-Bengio / Bouckaert-Frank corrected repeated-CV "
                                      "variance (1/n + n_test/n_train) s^2 with "
                                      "n_test/n_train = 1/4"),
        "caveat": "The 10 differences are not independent: within a seed the 5 training sets "
                  "overlap by 3/4, and both seeds reuse the same folds. The naive t understates "
                  "the variance; the corrected test is the conservative reading.",
    }


# ── E: cohort table from the committed split CSVs ────────────────────────────────────────

def cohort():
    cv = pd.concat([pd.read_csv(read(PREP / f"fold_{k}.csv")).query("split == 'val'")
                    for k in range(5)], ignore_index=True)
    assert cv.image_path.is_unique, "an image sits in two val folds"
    dev = pd.read_csv(read(PREP / "test_dev.csv"))
    lck = pd.read_csv(read(PREP / "test_locked.csv"))

    def counts(d):
        c = d.cls6.value_counts()
        return {k: int(c.get(k, 0)) for k in CLASSES}

    def unit_consistency(d, keys):
        g = d.groupby(keys).cls6.nunique()
        return {"units": int(len(g)), "single_label": int((g == 1).sum()),
                "frac_single_label": r4((g == 1).mean())}

    blocks = {
        "ostrava": (cv[cv.source == "ostrava"], "Czechia (Ostrava)", "per session"),
        "shenzhen": (cv[cv.source == "shenzhen"], "China (Shenzhen Eye Hospital)", "per image"),
        "ropvl": (cv[cv.source == "ropvl"], "China (Shenzhen Eye Hospital; different cohort "
                                            "from 'shenzhen')", "per image"),
        "multiview_dev": (dev, "China (hospital not named in the release)",
                          "per image in the annotation table; de facto per eye"),
        "multiview_locked": (lck, "China (hospital not named in the release)",
                             "per image in the annotation table; de facto per eye"),
    }
    table = {}
    for name, (d, country, unit) in blocks.items():
        row = {"images": int(len(d)), "infant_ids": int(d.patient_id.nunique()),
               "per_class": counts(d), "country": country, "label_unit": unit}
        if name == "ostrava":
            dev_code = d.image_path.str.extract(r"_D(\d)_S\d+_")[0]
            row["cameras"] = {DEVICE[c]: int(n) for c, n in dev_code.value_counts().items()}
            sess = d.assign(session=d.apply(session_key, axis=1))
            row["label_unit_evidence"] = {
                "how": "DG code per (ID, device, serie) from docs/infant_retinal_database_info"
                       ".csv, stamped on every frame of the session",
                "sessions": unit_consistency(sess, ["session"])}
        elif name in ("shenzhen", "ropvl"):
            row["cameras"] = "not stated in the release files (manifest device = 'unknown')"
            row["ids_note"] = ("IDs are the release's patient_id records. They are an upper "
                               "bound on distinct infants; nothing in the release links "
                               "records across IDs.")
            row.update(_source_unit_evidence(name, d))
        else:
            row["cameras"] = "not stated in the release files (manifest device = 'unknown')"
            e = eye_frame(d)
            row["label_unit_evidence"] = {
                "how": "eye = (patient_id, Left/Right) from the Acquisition View Table",
                "eyes": unit_consistency(e, ["patient_id", "eye"])}
        table[name] = row

    full = pd.concat([cv, dev, lck], ignore_index=True)
    audit = {"normal": 4219, "stage_1": 872, "stage_2": 1258, "stage_3": 932,
             "stage_4_5": 107, "arop": 636}
    totals = {
        "eligible_corpus": {"images": int(len(full)), "infant_ids": int(full.patient_id.nunique()),
                            "per_class": counts(full)},
        "cv_pool": {"images": int(len(cv)), "infant_ids": int(cv.patient_id.nunique()),
                    "per_class": counts(cv)},
        "held_out_multiview": {"images": int(len(dev) + len(lck)),
                               "infant_ids": int(pd.concat([dev, lck]).patient_id.nunique()),
                               "per_class": counts(pd.concat([dev, lck]))},
    }
    checks = {
        "total_is_8024": len(full) == 8024,
        "audit_per_class": audit,
        "audit_per_class_confirmed": counts(full) == audit,
        "cv_pool_is_6700": len(cv) == 6700,
        "stage_4_5_note": "89 of the 107 stage_4_5 images are ROP-VL (the whole CV-pool "
                          "count); the other 18 are Multi-View stage 4, held out "
                          "(dev 14, locked 4).",
    }
    flow = json.loads(read(RES / "inclusion_flow.json").read_text())
    checks["matches_inclusion_flow"] = (
        flow["layer_2_six_class_eligible"]["kept"]["images"] == len(full)
        and flow["layer_2_six_class_eligible"]["kept"]["infants"] == full.patient_id.nunique()
        and flow["layer_3_analysis_sets"]["sets"]["train_cv"]["images"] == len(cv)
        and flow["layer_3_analysis_sets"]["sets"]["train_cv"]["infants"]
        == cv.patient_id.nunique())
    sp = PREP / "splits.csv"   # local only (gitignored: absolute paths) — cross-check if present
    if sp.exists():
        s = pd.read_csv(read(sp))
        name = {"train_cv": "cv_pool", "multiview_dev": "multiview_dev",
                "multiview_locked": "multiview_locked"}
        checks["matches_local_splits_csv"] = bool(
            len(s) == len(full)
            and all(counts(s[s.split == k]) == (totals["cv_pool"]["per_class"] if v == "cv_pool"
                                                 else table[v]["per_class"])
                    for k, v in name.items()))
    return {"built_from": "committed fold_{0..4}.csv val rows (= CV pool) + test_dev.csv + "
                          "test_locked.csv; no manifest or splits.csv needed",
            "sources": table, "totals": totals, "checks": checks}


def _source_unit_evidence(name, d):
    """Label-unit and ID evidence from the release spreadsheets (counts only, no IDs)."""
    fn = d.image_path.str.rsplit("/", n=1).str[1]
    out = {}
    if name == "shenzhen":
        x = pd.ExcelFile(read(ROOT / "data/rop/shenzhen/zip information.xlsx"))
        a, b = x.parse("Sheet1"), x.parse("Sheet2")
        e = d.assign(eye=fn.map(dict(zip(a.img_name, a.eye))))
        assert e.eye.notna().all()
        g = e.groupby(["patient_id", "eye"]).cls6.nunique()
        out["label_unit_evidence"] = {
            "how": "per-image stage folders; eye (OD/OS) from zip information.xlsx",
            "infant_eye": {"units": int(len(g)), "frac_single_label": r4((g == 1).mean())}}
        b = b[("shenzhen:" + b.ID.astype(str)).isin(set(d.patient_id))]
        prof = b.drop(columns=["ID"]).astype(str).agg("|".join, axis=1)
    else:
        m = fn.str.extract(r"^\d+_(\d{4}-\d\d-\d\d)_([LR])_\d+\.")
        assert m.notna().all().all()
        e = d.assign(exam=m[0].values, eye=m[1].values)
        g = e.groupby(["patient_id", "exam", "eye"]).cls6.nunique()
        out["label_unit_evidence"] = {
            "how": "per-image class folders + img_info.xlsx; exam date and L/R eye parsed "
                   "from the filename (counts only - the dates are never written out)",
            "infant_exam_eye": {"units": int(len(g)), "frac_single_label": r4((g == 1).mean())}}
        info = pd.read_excel(read(ROOT / "data/rop/rop_vl/img_info.xlsx"))
        info = info[("ropvl:" + info.patient_id.astype(str)).isin(set(d.patient_id))]
        info = info.drop_duplicates("patient_id")
        prof = info[["gender", "gestational_age", "birth_weight", "plurality",
                     "delivery_mode"]].astype(str).agg("|".join, axis=1)
    out["ids_sharing_identical_demographics"] = {
        "ids": int(len(prof)), "distinct_profiles": int(prof.nunique()),
        "ids_sharing_a_profile": int((prof.map(prof.value_counts()) > 1).sum()),
        "reading": "twins and coincident GA/BW also share a profile, so this is NOT a "
                   "duplicate count; it only shows the ID count cannot be read as people "
                   "without further linkage."}
    return out


# ── F: AP-ROP internal control ───────────────────────────────────────────────────────────

def arop_control(run_dir: Path, rng_seed=42):
    """AP-ROP-labelled frames scored p_arop < 0.1, pooled out of fold over 5 folds."""
    parts = []
    for k in range(5):
        val = pd.read_csv(read(PREP / f"fold_{k}.csv")).query("split == 'val'").reset_index(
            drop=True)
        z = np.load(read(run_dir / f"fold_{k}.npz"))
        assert (z["y_true"] == val.label.values).all()
        parts.append(val.assign(p_arop=z["prob6"][:, AROP]))
    ar = pd.concat(parts, ignore_index=True).query("cls6 == 'arop'")
    ar = ar.assign(unit=ar.apply(session_key, axis=1), low=(ar.p_arop < 0.1).astype(float))

    def block(d, unit_name):
        u = d.groupby("unit").low
        rng = np.random.default_rng(rng_seed)
        keys = np.array(sorted(d.unit.unique()))
        lows = {k: v.values for k, v in u}
        bs = [np.concatenate([lows[k] for k in rng.choice(keys, len(keys), True)]).mean()
              for _ in range(N_BOOT)]
        per_unit = u.mean()
        return {"frames": int(len(d)), unit_name: int(len(keys)),
                "infants": int(d.patient_id.nunique()),
                "frames_below_0.1": int(d.low.sum()), "frac_below_0.1": r4(d.low.mean()),
                f"ci95_clustered_by_{unit_name}": ci95(bs),
                f"per_{unit_name[:-1]}_frac_quartiles": [r4(q) for q in
                                                         per_unit.quantile([.25, .5, .75])],
                f"median_frames_per_{unit_name[:-1]}": float(u.size().median())}

    return {"ostrava_session_labelled": block(ar[ar.source == "ostrava"], "sessions"),
            "non_ostrava_frame_labelled": block(ar[ar.source != "ostrava"], "units")}


# ── provenance ──────────────────────────────────────────────────────────────────────────

def provenance():
    tracked = set(subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True,
                                 text=True).stdout.splitlines())
    files = []
    for p in sorted(INPUTS):
        rel = str(p.relative_to(ROOT))
        files.append({"path": rel, "git_tracked": rel in tracked,
                      "sha256_16": hashlib.sha256(p.read_bytes()).hexdigest()[:16]})
    head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                          capture_output=True, text=True).stdout.strip()
    import scipy
    import sklearn
    return {"script": "scripts/rop_preprint_stats.py", "git_head": head,
            "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
            "inference_run": False,
            "torch_imported": "torch" in sys.modules,
            "code_reused": ["scripts/rop_locked_eval.py: eye_frame, eye_metrics, metrics",
                            "models/common/metrics.py: quadratic_weighted_kappa",
                            "scripts/rop_matched_equivalence.py: fold_metrics, ARMS, ARMS_S1337",
                            "scripts/rop_session_level_analysis.py: session_key",
                            "scripts/build_rop_staging_manifest.py: DEVICE"],
            "bootstrap": {"replicates": N_BOOT, "seed": BOOT_SEED, "unit": "infant",
                          "aggregation": "metric per checkpoint, mean over 5 checkpoints "
                                         "within each replicate; percentile 95% CI"},
            "versions": {"numpy": np.__version__, "pandas": pd.__version__,
                         "scipy": scipy.__version__, "sklearn": sklearn.__version__},
            "inputs": files}


def main():
    out = {"what": "CPU-only statistics for the ROP preprint, re-scored from saved "
                   "predictions. No inference; test_locked's saved arrays are re-scored, "
                   "never re-opened."}

    # 1. Reproduce before adding anything.
    ldf, ledf, lpreds = load_set("locked")
    lpe = {c: point_estimates(ldf, ledf, ps) for c, ps in lpreds.items()}
    print("locked half: bootstrapping 4 cells x 2,000 replicates ...")
    boot = {}
    for c, ps in lpreds.items():
        boot[c] = bootstrap_cell(ldf, ledf, ps)
        print(f"  {c}: image QWK {lpe[c]['image']['qwk']['mean']} {boot[c]['image_qwk']['ci95']}"
              f" | eye QWK {lpe[c]['eye']['qwk']['mean']} {boot[c]['eye_qwk']['ci95']}")
    rows = check_against(RES / "locked_eval" / "locked_table.json", lpe, boot)
    ddf, dedf, dpreds = load_set("dev")
    dpe = {c: point_estimates(ddf, dedf, ps) for c, ps in dpreds.items()}
    drows = check_against(RES / "dev_eval" / "dev_table.json", dpe, None)
    s42 = lpe["structured_s42"]
    headline = {
        "structured_s42_image_qwk": {"expected": 0.523, "reproduced": s42["image"]["qwk"]["mean"],
                                     "match_3dp": round(s42["image"]["qwk"]["mean"], 3) == 0.523},
        "structured_s42_eye_qwk": {"expected": 0.801, "reproduced": s42["eye"]["qwk"]["mean"],
                                   "match_3dp": round(s42["eye"]["qwk"]["mean"], 3) == 0.801},
    }
    bad = [r for r in rows + drows if not r["match"]]
    out["reproduction_check"] = {
        "headline": headline,
        "locked_table_fields_checked": len(rows),
        "dev_table_fields_checked": len(drows),
        "mismatches": bad,
        "passed": not bad and all(v["match_3dp"] for v in headline.values()),
        "note": "includes the committed image-level clustered QWK and macro-F1 CIs: same draws "
                "as rop_locked_eval.cluster_ci, so they must agree exactly.",
    }
    if not out["reproduction_check"]["passed"]:
        raise SystemExit(f"REFUSING to write: reproduction check failed: {bad or headline}")
    print(f"  reproduction check passed ({len(rows)} locked + {len(drows)} dev fields)")

    # A. Locked half.
    out["A_locked_clustered_bootstrap"] = {
        "set": f"test_locked ({s42['n_images']} images, {s42['n_infants']} infants, "
               f"{s42['n_eyes']} eyes) - saved predictions only",
        "cells": {c: {"point_5ckpt": lpe[c], "ci95_clustered": boot[c]} for c in lpreds},
        "definitions": {
            "eye": "(patient_id, Left/Right) from the Multi-View Acquisition View Table; eye "
                   "label = max view label, eye prediction = max view prediction (hard class "
                   "index), exactly rop_locked_eval.eye_metrics",
            "eye_macro_f1_present": "macro-F1 over the classes present in the (resampled) eye "
                                    "labels",
            "stage123_eyes_called_normal": "share of eyes whose eye label is stage 1-3 and "
                                           "whose eye prediction is Normal",
            "point_estimates": "mean (sd) over the 5 fold checkpoints of the per-checkpoint "
                               "metric; image/eye metric means are of 4-dp per-checkpoint "
                               "values, as in locked_table.json",
        },
    }

    # B. Dev half.
    out["B_dev_image_vs_eye_qwk"] = {
        "available": True,
        "source": "results/rop/dev_eval/*.npz (saved by rop_locked_eval.py --set dev)",
        "cells": {c: {"image_qwk": dpe[c]["image"]["qwk"], "eye_qwk": dpe[c]["eye"]["qwk"],
                      "eye_minus_image_qwk": dpe[c]["eye_minus_image_qwk"],
                      "stage123_eyes_called_normal": dpe[c]["eye"]["stage123_eyes_called_normal"],
                      "stage123_images_called_normal":
                          dpe[c]["image"]["stage123_images_called_normal"],
                      "n_images": dpe[c]["n_images"], "n_eyes": dpe[c]["n_eyes"]}
                  for c in dpreds},
    }

    # C. Unweighted kappa.
    kj = json.loads(read(RES / "kappa_unweighted_vs_human.json").read_text())
    rk = {}
    for label, d in (("seed_42", ARMS["structured_natural"]),
                     ("seed_1337", ARMS_S1337["structured_natural"])):
        z = [np.load(read(d / f"fold_{k}.npz")) for k in range(5)]
        y = np.concatenate([q["y_true"] for q in z])
        p = np.concatenate([q["y_pred"] for q in z])
        rk[label] = r4(cohen_kappa_score(y, p))
        if label == "seed_42":
            m = (y != AROP) & (p != AROP)
            rk["seed_42_stage_only"] = r4(cohen_kappa_score(y[m], p[m]))
    out["C_unweighted_kappa"] = {
        "cv_pooled_from_kappa_unweighted_vs_human_json": {
            "values": kj["ours"],
            "definition": "Cohen's kappa between the structured head (natural sampling) and the "
                          "released labels, pooled out of fold: the 5 CV val folds concatenated "
                          "(n=6,700 images, per IMAGE). stage_only = images where neither the "
                          "label nor the prediction is AP-ROP (n=6,058). Definition re-derived "
                          "here from staging_runs/structured_5fold*/fold_*.npz.",
            "rederived": rk,
            "rederived_match": all(rk[k] == kj["ours"][k]["kappa_unweighted"] for k in rk),
            "human_benchmark": kj["human_benchmark"],
            "not_a_superiority_claim": kj["not_a_superiority_claim"],
        },
        "locked_half_per_image_and_per_eye": {
            c: {"image": {"point_5ckpt": lpe[c]["image"]["kappa_unweighted"],
                          "ci95_clustered": boot[c]["image_kappa_unweighted"]["ci95"]},
                "eye": {"point_5ckpt": lpe[c]["eye"]["kappa_unweighted"],
                        "ci95_clustered": boot[c]["eye_kappa_unweighted"]["ci95"]}}
            for c in lpreds},
        "locked_note": "computed here from the saved locked arrays (sklearn cohen_kappa_score, "
                       "unweighted); kappa_unweighted_vs_human.json holds no held-out or "
                       "per-eye values.",
    }

    # D. Head equivalence.
    out["D_head_equivalence_tost"] = equivalence()

    # E. Cohort.
    out["E_cohort"] = cohort()

    # F. AP-ROP internal control.
    sl = json.loads(read(RES / "session_level_by_seed.json").read_text())
    f42 = arop_control(ARMS["structured_natural"])
    committed = sl["frames_without_the_appearance"]
    f42_match = (f42["ostrava_session_labelled"]["frames"]
                 == committed["ostrava_session_labelled"]["frames"]
                 and f42["ostrava_session_labelled"]["frac_below_0.1"]
                 == committed["ostrava_session_labelled"]["frac_below_0.1"]
                 and f42["ostrava_session_labelled"]["sessions"]
                 == committed["ostrava_session_labelled"]["sessions"]
                 and f42["ostrava_session_labelled"]["per_session_frac_quartiles"][1]
                 == committed["ostrava_session_labelled"]["median_per_session"]
                 and f42["non_ostrava_frame_labelled"]["frames"]
                 == committed["non_ostrava_frame_labelled"]["frames"]
                 and f42["non_ostrava_frame_labelled"]["frac_below_0.1"]
                 == committed["non_ostrava_frame_labelled"]["frac_below_0.1"])
    out["F_arop_internal_control"] = {
        "definition": committed["what"] + " Ostrava units = imaging sessions (device+serie); "
                      "other sources have no session structure, so a unit is the infant. CIs "
                      "resample those units.",
        "seed_42": {"structured": f42, "committed_values": committed,
                    "rederived_match": bool(f42_match),
                    "note": "committed median_frames_per_session=24 is a truncation: 16 "
                            "sessions, middle pair 21 and 28, median 24.5",
                    "session_recall": sl["seed_42"]},
        "seed_1337": {"structured": arop_control(ARMS_S1337["structured_natural"]),
                      "note": "not in session_level_by_seed.json; derived here from "
                              "staging_runs/structured_5fold_s1337/fold_*.npz with the same "
                              "definition",
                      "session_recall": sl["seed_1337"]},
    }

    assert "torch" not in sys.modules, "torch was imported - this script must run no model"
    out["provenance"] = provenance()
    OUT.write_text(json.dumps(out, indent=2) + "\n")
    print(f"-> {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
