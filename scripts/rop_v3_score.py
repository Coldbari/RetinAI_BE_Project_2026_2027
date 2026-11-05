"""Protocol-v2 scoring from SAVED predictions (docs/paper/PROTOCOL_V2.md is the authority).

    .venv/bin/python scripts/rop_v3_score.py            # scores baseline C -> results/rop/v3/baseline_C_scores.json
    .venv/bin/python scripts/rop_v3_score.py --v3       # arms A-I (results/rop/v3/runs/) and C -> results/rop/v3/v3_scores.json

Amendment 3: arm E (mean of A, B, C probabilities, matched by seed) is derived in load_v3, and every
arm is also scored with the CV-cross-fitted ordinal decoder ("decoded" keys, contrasts_decoded)
beside argmax, which stays primary.

Amendment 5: arms H (v4 pool) and I (v4 pool + v4 recipe) are scored like every other arm (their CV
val rows equal v3's), every arm with held-out-source folds gets a "loso" block (score_loso), and
"v4_selection" applies the 5c comparison rule (select_v4) to the selection table and those blocks.

No model is built or run, torch is never imported, and nothing named test_locked is read.
Outputs carry aggregate numbers only (no filenames, dates, or patient ids).

Per-arm input layout (the Kaggle kernels' outputs), one directory per fold:
    <run_dir>/fold_k/val_predictions.npz   y_true, y_pred, prob   (rows = fold_k.csv val rows)
    <run_dir>/fold_k/dev_predictions.npz   y_true, prob[, y_pred] (rows = test_dev.csv; k = checkpoint)
    <run_dir>/fold_site/, fold_loso_ostrava/, fold_loso_ropvl/val_predictions.npz   (optional; rows =
                                           the val rows of the same-named fold file = one held-out source)
C's existing files are mapped into the same in-memory structure by `load_baseline_C`.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
PREP = ROOT / "results" / "prepared" / "rop_staging"
CLASSES = ["normal", "stage_1", "stage_2", "stage_3", "stage_4_5", "arop"]
SOURCES = ["multiview", "ropvl", "ostrava", "shenzhen"]
NFOLD = 5
RUNS = ROOT / "results" / "rop" / "v3" / "runs"
V3_ARMS = {"A": "retfound_green", "B": "dinov2_s", "D": "dinov2_s_bag"}
V3_ARMS["E"] = "ensemble(A,B,C)"        # Amendment 3(a): derived in load_v3, not a run directory
V3_ARMS["G"] = "dinov2_s_r518"          # Amendment 4(a): B at 518 px; five CV folds, no site fold
V3_ARMS["F"] = "dinov2_s_uda"           # Amendment 4(b): B + self-ensembling on unlabelled dev photos
# Amendment 4(b): F saw the dev IMAGES (never their labels) in training, so its dev numbers are
# transductive; every dev block of F carries this status. Its locked numbers will be inductive.
TRANSDUCTIVE_DEV = {"F": "transductive: dev images seen unlabelled during training"}
V3_ARMS["H"] = "dinov2_s_v4data"        # Amendment 5b: B's recipe on the v4 pool
V3_ARMS["I"] = "dinov2_s_v4recipe"      # Amendment 5b: the v4 pool and the v4 recipe
V4_ARMS = ("H", "I")                    # train on results/prepared/rop_staging/v4/ (val rows = v3's)
# held-out source -> fold directory of a run = stem of its fold file (Amendment 5a.5)
LOSO_FOLDS = {"shenzhen": "fold_site", "ostrava": "fold_loso_ostrava", "ropvl": "fold_loso_ropvl"}
V4_MARGIN_CV = 0.010                    # Amendment 5c: CV QWK5 gain over B needed to adopt
V4_MARGIN_LOSO = 0.010                  # Amendment 5c: the most mean LOSO QWK5 may fall below B
ENSEMBLE = ("A", "B", "C")
V3_SEEDS = (42, 1337)
PAIRS = (("A", "C"), ("A", "B"), ("B", "C"), ("E", "C"), ("E", "A"), ("D", "A"), ("D", "B"), ("F", "B"), ("G", "B"),
         ("H", "B"), ("I", "B"), ("I", "H"))      # the last three: Amendment 5c
DECODER_RULE = ("Amendment 3(b): s = sum_k k*p_k on the 5-class map (AP-ROP folded onto 4), cut by four "
                "monotone thresholds into stages 0-4, AP-ROP (5) wherever it is the photo's argmax; "
                "thresholds maximise image QWK5 on CV OOF rows only (14 duplicates excluded): CV is "
                "cross-fitted (fold k decoded with cuts fitted on the other four folds), dev uses cuts "
                "fitted on all five folds. Eye/unit = most severe decoded photo. Argmax stays primary.")
EQUIV = ("A", "B")                      # secondary contrast, also a 90% CI against +-MARGIN
MARGIN = 0.05


# ------------------------------------------------------------------ guards / loading
def guard_path(p) -> Path:
    if "test_locked" in str(p) or "locked_eval" in str(p):
        raise ValueError(f"refusing to read {p}")
    return Path(p)


def _npz(p) -> dict:
    z = np.load(guard_path(p))
    d = {k: z[k] for k in z.files}
    if "prob" not in d and "prob6" in d:
        d["prob"] = d.pop("prob6")
    d["prob"] = np.asarray(d["prob"], dtype=np.float64)
    d["y_true"] = np.asarray(d["y_true"]).astype(np.int64)
    if "y_pred" not in d:
        d["y_pred"] = d["prob"].argmax(1)
    d["y_pred"] = np.asarray(d["y_pred"]).astype(np.int64)
    return {k: d[k] for k in ("y_true", "y_pred", "prob")}


def load_arm_dir(run_dir) -> dict:
    run_dir = Path(run_dir)
    val = [_npz(run_dir / f"fold_{k}" / "val_predictions.npz") for k in range(NFOLD)]
    dev = [_npz(run_dir / f"fold_{k}" / "dev_predictions.npz") for k in range(NFOLD)
           if (run_dir / f"fold_{k}" / "dev_predictions.npz").exists()]
    loso = {s: _npz(run_dir / d / "val_predictions.npz") for s, d in LOSO_FOLDS.items()
            if (run_dir / d / "val_predictions.npz").exists()}
    return {"val": val, "dev": dev or None, "loso": loso}


def load_baseline_C(seed: int) -> dict:
    d = ROOT / "results" / "rop" / ("staging_runs/flat_natural_5fold" if seed == 42
                                    else "staging_runs/flat_natural_5fold_s1337")
    dev = ROOT / "results" / "rop" / "dev_eval"
    return {"val": [_npz(d / f"fold_{k}.npz") for k in range(NFOLD)],
            "dev": [_npz(dev / f"flat_s{seed}_ckpt{k}.npz") for k in range(NFOLD)]}


def pool_frame(folds_dir) -> pd.DataFrame:
    parts = []
    for k in range(NFOLD):
        d = pd.read_csv(Path(folds_dir) / f"fold_{k}.csv")
        d = d[d.split == "val"].reset_index(drop=True)
        d["fold"] = k
        parts.append(d)
    return pd.concat(parts, ignore_index=True)


def assemble_cv(arm: dict, pool: pd.DataFrame) -> dict:
    ys, ps, prs = [], [], []
    for k, z in enumerate(arm["val"]):
        n = int((pool.fold == k).sum())
        if len(z["y_true"]) != n:
            raise ValueError(f"fold {k}: {len(z['y_true'])} predictions vs {n} val rows")
        if not (z["y_true"] == pool.label.values[pool.fold.values == k]).all():
            raise ValueError(f"fold {k} misaligned")
        ys.append(z["y_true"]); ps.append(z["y_pred"]); prs.append(z["prob"])
    return {"y": np.concatenate(ys), "pred": np.concatenate(ps), "prob": np.concatenate(prs)}


# ------------------------------------------------------------------ metrics
def to5(x):
    """5-class map: AP-ROP folded onto stage 4/5 (QWK_DEFINITION_CORRECTION.md)."""
    return np.where(np.asarray(x) == 5, 4, np.asarray(x))


def qwk(y, p, k=6) -> float:
    """Quadratic weighted kappa over a FIXED label set; equals sklearn cohen_kappa_score
    (weights='quadratic', labels=range(k)), written with bincount so bootstraps are fast."""
    y, p = np.asarray(y), np.asarray(p)
    O = np.bincount(y * k + p, minlength=k * k).reshape(k, k).astype(float)
    E = np.outer(O.sum(1), O.sum(0)) / O.sum()
    i = np.arange(k)
    W = (i[:, None] - i[None, :]) ** 2
    den = (W * E).sum()
    return float(1.0 - (W * O).sum() / den) if den > 0 else 0.0


def qwk6(y, p): return qwk(y, p, 6)
def qwk5(y, p): return qwk(to5(y), to5(p), 5)


def accuracy(y, p) -> float:
    return float((np.asarray(y) == np.asarray(p)).mean())


def macro_f1_present(y, p) -> float:
    from sklearn.metrics import f1_score
    return float(f1_score(y, p, average="macro", labels=sorted(set(np.asarray(y).tolist())),
                          zero_division=0))


def class_metrics(y, p) -> dict:
    from sklearn.metrics import cohen_kappa_score
    y, p = np.asarray(y), np.asarray(p)
    present = sorted(set(y.tolist()))
    out = {"n": int(len(y)), "qwk_5class": qwk5(y, p), "qwk_6label": qwk6(y, p),
           "accuracy": float((y == p).mean()), "macro_f1_present": macro_f1_present(y, p),
           "kappa_unweighted": float(cohen_kappa_score(y, p)),
           "per_class_recall": {CLASSES[c]: {"recall": float((p[y == c] == c).mean()),
                                             "n": int((y == c).sum())} for c in present}}
    if (y == 5).any():
        tp = int(((y == 5) & (p == 5)).sum())
        out["arop_binary"] = {"recall": tp / int((y == 5).sum()),
                              "f1": 2 * tp / max(int((y == 5).sum()) + int((p == 5).sum()), 1)}
    return out


# ------------------------------------------------------------------ bootstrap
def _draws(groups, n, seed):
    """Same resampling as scripts/rop_locked_eval.cluster_ci: infants drawn with replacement."""
    rng = np.random.default_rng(seed)
    groups = np.asarray(groups)
    uniq = np.unique(groups)
    idx = {g: np.where(groups == g)[0] for g in uniq}
    for _ in range(n):
        yield np.concatenate([idx[g] for g in rng.choice(uniq, len(uniq), True)])


def _pct(v, level=0.95):
    lo = 100 * (1 - level) / 2
    return [round(float(x), 4) for x in np.percentile(v, [lo, 100 - lo])]


def boot_ci(y, preds, groups, metrics: dict, n=2000, seed=42) -> dict:
    """Infant-clustered percentile 95% CI of mean-over-`preds` of each metric."""
    y = np.asarray(y)
    vals = {m: [] for m in metrics}
    for t in _draws(groups, n, seed):
        for m, fn in metrics.items():
            vals[m].append(np.mean([fn(y[t], p[t]) for p in preds]))
    return {m: _pct(v) for m, v in vals.items()}


def paired_bootstrap(y, a, b, groups, metric, n=2000, seed=42, level=0.95) -> dict:
    """a, b: lists of prediction vectors (checkpoints/seeds, averaged inside each replicate).
    Infants are resampled ONCE per replicate and both arms scored on that draw."""
    y = np.asarray(y)
    A, B, D = [], [], []
    for t in _draws(groups, n, seed):
        va = np.mean([metric(y[t], p[t]) for p in a])
        vb = np.mean([metric(y[t], p[t]) for p in b])
        A.append(va); B.append(vb); D.append(va - vb)
    a0 = float(np.mean([metric(y, p) for p in a])); b0 = float(np.mean([metric(y, p) for p in b]))
    return {"a": round(a0, 4), "b": round(b0, 4), "diff": round(a0 - b0, 4),
            "a_ci": _pct(A, level), "b_ci": _pct(B, level), "diff_ci": _pct(D, level), "level": level,
            "reps": n, "n_infants": int(len(np.unique(groups))),
            "diff_replicates": np.asarray(D)}      # for Holm / two-sided p; strip before JSON


def contrasts(y, lists: dict, groups, metrics: dict, n_boot=2000, seed=42, pairs=PAIRS) -> dict:
    """Paired arm differences, Amendment 1(c): lists = {arm: every prediction list of that arm
    (all its present seeds; on dev every checkpoint)}, averaged inside each replicate; ONE infant
    resample per replicate for both arms (no seed-to-seed pairing). Pairs with an absent arm are
    skipped. EQUIV's QWK also gets the 90% CI against +-MARGIN. Replicates stripped (JSON-safe)."""
    y = np.asarray(y)
    out = {}
    for a, b in pairs:
        if not lists.get(a) or not lists.get(b):
            continue
        if any(len(p) != len(y) for p in lists[a] + lists[b]):
            raise ValueError(f"{a}-{b}: prediction lists not aligned to the {len(y)} scored rows")
        r = {}
        for m, fn in metrics.items():
            pb = paired_bootstrap(y, lists[a], lists[b], groups, fn, n_boot, seed)
            D = pb.pop("diff_replicates")
            if (a, b) == EQUIV and m == "qwk_5class":
                ci = _pct(D, 0.90)
                pb["level_90"] = {"diff_ci": ci, "level": 0.90, "margin": MARGIN,
                                  "inside_pm_0.05": bool(-MARGIN < ci[0] and ci[1] < MARGIN)}
            r[m] = pb
        r["n_lists"] = {a: len(lists[a]), b: len(lists[b])}
        out[f"{a}-{b}"] = r
    return out


# ------------------------------------------------------------------ units
def aggregate_units(keys, y, pred, prob) -> dict:
    """Unit label = most severe label; unit prediction = most severe predicted class; unit any-ROP
    score = max over photos of (1 - p_normal); unit probability vector = the DRIVING photo's
    (highest predicted class, ties broken by that class's probability)."""
    keys = np.asarray(keys)
    _, inv = np.unique(keys, return_inverse=True)
    m = inv.max() + 1
    uy = np.zeros(m, int); up = np.zeros(m, int); us = np.zeros(m)
    np.maximum.at(uy, inv, y); np.maximum.at(up, inv, pred); np.maximum.at(us, inv, 1.0 - prob[:, 0])
    drive = pred * 2.0 + prob[np.arange(len(pred)), pred]
    order = np.lexsort((drive, inv))                       # last row of each unit = the driver
    last = np.r_[np.where(np.diff(inv[order]) != 0)[0], len(order) - 1]
    uprob = prob[order[last]]
    first = np.zeros(m, int); first[inv[::-1]] = np.arange(len(inv))[::-1]
    return {"y": uy, "pred": up, "score": us, "prob": uprob, "first": first, "inv": inv}


def operating_point(score, label, target_sens=0.90) -> float:
    """Largest t with unit-level any-ROP sensitivity (score >= t) >= target on diseased units."""
    pos = np.sort(np.asarray(score)[np.asarray(label) > 0])[::-1]
    k = int(np.ceil(target_sens * len(pos) - 1e-9))
    return float(pos[k - 1])


def _sens_spec(score, flag_t, label, positive):
    flagged = score >= flag_t
    pos = positive
    neg = ~positive
    return {"sensitivity": float(flagged[pos].mean()) if pos.any() else None,
            "specificity": float((~flagged[neg]).mean()) if neg.any() else None,
            "n_pos": int(pos.sum()), "n_neg": int(neg.sum())}


def op_report(score, label, t) -> dict:
    """Any-ROP (label>0) and severe (label>=3: stage 3, 4/5, AP-ROP) at threshold t. Severe
    specificity is reported against normal units and against all non-severe units."""
    label = np.asarray(label); score = np.asarray(score)
    sev = label >= 3
    vs_normal = _sens_spec(score[(label == 0) | sev], t, None, sev[(label == 0) | sev])
    return {"threshold": t, "any_rop": _sens_spec(score, t, None, label > 0),
            "severe_vs_normal": vs_normal, "severe_vs_all_nonsevere": _sens_spec(score, t, None, sev)}


# ------------------------------------------------------------------ calibration
def calibrate(prob, T):
    z = np.log(np.clip(prob, 1e-12, None)) / T
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def nll(prob, y) -> float:
    return float(-np.log(np.clip(prob[np.arange(len(y)), y], 1e-12, None)).mean())


def brier(prob, y) -> float:
    oh = np.eye(prob.shape[1])[y]
    return float(((prob - oh) ** 2).sum(1).mean())


def ece(prob, y, bins=15) -> float:
    conf = prob.max(1); acc = (prob.argmax(1) == y).astype(float)
    edges = np.linspace(0, 1, bins + 1)
    b = np.clip(np.digitize(conf, edges[1:-1]), 0, bins - 1)
    return float(sum((b == i).mean() * abs(acc[b == i].mean() - conf[b == i].mean())
                     for i in range(bins) if (b == i).any()))


def fit_temperature(prob, y) -> float:
    """Single scalar T minimising NLL of softmax(log(prob)/T) on the given (CV OOF) rows."""
    from scipy.optimize import minimize_scalar
    r = minimize_scalar(lambda lt: nll(calibrate(prob, float(np.exp(lt))), y),
                        bounds=(-3, 3), method="bounded", options={"xatol": 1e-8})
    return float(np.exp(r.x))


def cal_block(prob, y, T) -> dict:
    c = calibrate(prob, T)
    f = lambda q: {"ece": round(ece(q, y), 4), "brier": round(brier(q, y), 4), "nll": round(nll(q, y), 4)}  # noqa: E731
    return {"before": f(prob), "after": f(c), "n": int(len(y))}


# ------------------------------------------------------------------ ordinal decoder (Amendment 3b)
def expected_stage(prob) -> np.ndarray:
    """s = sum_k k * p5_k, p5 = prob with the AP-ROP column added into stage 4/5."""
    prob = np.asarray(prob, dtype=np.float64)
    return prob[:, :5] @ np.arange(5.0) + 4.0 * prob[:, 5]


def decode(prob, thresholds) -> np.ndarray:
    """Stage = number of cuts that s exceeds (0-4); AP-ROP (5) wherever it is the argmax."""
    prob = np.asarray(prob, dtype=np.float64)
    st = (expected_stage(prob)[:, None] > np.asarray(thresholds, dtype=np.float64)[None, :]).sum(1)
    return np.where(prob.argmax(1) == 5, 5, st).astype(np.int64)


def fit_decoder(prob, y, max_cand=200, max_sweeps=50) -> list:
    """Four monotone cuts on s maximising QWK5 of the decoded stage on (prob, y). Coordinate search
    from [0.5, 1.5, 2.5, 3.5]: each cut in turn takes the best of the candidates lying between its
    neighbours (midpoints of the sorted unique s values, evenly subsampled to <= max_cand); only a
    strict gain moves a cut, and a full sweep with no move ends the search. Deterministic."""
    prob = np.asarray(prob, dtype=np.float64)
    y5 = to5(np.asarray(y))
    s = expected_stage(prob)
    ap = prob.argmax(1) == 5                    # decoded 5 -> 4 on the 5-class map whatever the cuts
    u = np.unique(s)
    mid = (u[:-1] + u[1:]) / 2

    def score(t):                               # t is sorted, so searchsorted = decode's count of cuts < s
        return qwk(y5, np.where(ap, 4, np.searchsorted(t, s, side="left")), 5)

    t = np.array([0.5, 1.5, 2.5, 3.5])
    best = score(t)
    for _ in range(max_sweeps):
        moved = False
        for i in range(4):
            lo = t[i - 1] if i > 0 else -np.inf
            hi = t[i + 1] if i < 3 else np.inf
            c = mid[(mid >= lo) & (mid <= hi)]
            if len(c) > max_cand:
                c = c[np.unique(np.linspace(0, len(c) - 1, max_cand).round().astype(int))]
            for v in c:
                tt = t.copy()
                tt[i] = v
                q = score(tt)
                if q > best + 1e-12:
                    best, t, moved = q, tt, True
        if not moved:
            break
    return [float(x) for x in t]


def crossfit_decode(prob, y, fold) -> tuple:
    """CV decoded OOF prediction: the rows of fold k decoded with cuts fitted on every other fold's
    rows. Returns (decoded predictions, [cuts of fold k for k in sorted folds])."""
    prob, y, fold = np.asarray(prob), np.asarray(y), np.asarray(fold)
    pred = np.zeros(len(y), dtype=np.int64)
    ths = []
    for k in np.unique(fold):
        out = fold == k
        t = fit_decoder(prob[~out], y[~out])
        pred[out] = decode(prob[out], t)
        ths.append(t)
    return pred, ths


def decoded_lists(arm: dict, cv: dict, keep, fold) -> dict:
    """Amendment 3(b) for one arm-seed. cv = assemble_cv(arm, pool); keep = the dedup mask over the
    pooled rows; fold = fold of each kept row. Dev cuts are fitted on all kept CV OOF rows."""
    prob, y = cv["prob"][keep], cv["y"][keep]
    pred, ths = crossfit_decode(prob, y, fold)
    t_all = fit_decoder(prob, y)
    dev = [decode(z["prob"], t_all) for z in arm["dev"]] if arm.get("dev") else None
    return {"cv": pred, "thresholds_per_fold": ths, "thresholds_all_folds": t_all, "dev": dev}


# ------------------------------------------------------------------ scoring
def read_exclusions(path) -> set:
    return set(pd.read_csv(path).image_path)


def _dev_frame():
    from models.common.units import unit_keys
    df = pd.read_csv(PREP / "test_dev.csv")
    return df, unit_keys(df)


def _unit_metrics(u, groups_unit, n_boot, seed) -> dict:
    r = class_metrics(u["y"], u["pred"])
    r["ci95"] = boot_ci(u["y"], [u["pred"]], groups_unit,
                        {"qwk_5class": qwk5, "qwk_6label": qwk6, "macro_f1_present": macro_f1_present},
                        n_boot, seed)
    return r


def _per_source_units(U, src_u, inf_u, sources, n_boot, seed) -> dict:
    out = {}
    for s in sources:
        m = src_u == s
        out[s] = _unit_metrics({k: U[k][m] for k in ("y", "pred")}, inf_u[m], n_boot, seed)
        out[s]["n_infants"] = int(len(np.unique(inf_u[m])))
    return out


def _confusion_mean(y, preds) -> list:
    y = np.asarray(y)
    return np.mean([np.bincount(y * 6 + p, minlength=36).reshape(6, 6) for p in preds], 0).round(4).tolist()


def _list_metrics(y, preds, groups, n_boot, seed, confusion=False) -> dict:
    """Mean over prediction lists (dev checkpoints) of class_metrics; one clustered CI of the mean."""
    per = [class_metrics(y, p) for p in preds]
    keysm = ("qwk_5class", "qwk_6label", "accuracy", "macro_f1_present", "kappa_unweighted")
    ms = {"qwk_5class": qwk5, "qwk_6label": qwk6, "macro_f1_present": macro_f1_present}
    r = {k: round(float(np.mean([x[k] for x in per])), 4) for k in keysm} | {
        "ci95": boot_ci(y, preds, groups, ms, n_boot, seed),
        "per_class_recall_mean": {c: round(float(np.mean([x["per_class_recall"][c]["recall"] for x in per
                                                          if c in x["per_class_recall"]])), 4)
                                  for c in per[0]["per_class_recall"]}}
    if confusion:
        r["confusion_mean"] = {"classes": CLASSES, "rows": "label", "cols": "prediction",
                               "n_lists": len(preds), "mean": _confusion_mean(y, preds)}
    return r


def score_arm(run, folds_dir=PREP, exclude=PREP / "exclude_duplicates.csv", n_boot=2000, seed=42,
              dev_csv=None) -> dict:
    """run: a directory in the kernel layout, or an already-loaded arm dict (see load_arm_dir)."""
    from models.common.units import unit_keys
    arm = load_arm_dir(run) if isinstance(run, (str, Path)) else run
    pool = pool_frame(folds_dir)
    cv = assemble_cv(arm, pool)
    keep = ~pool.image_path.isin(read_exclusions(exclude)).values if exclude is not None else np.ones(len(pool), bool)
    pool = pool[keep].reset_index(drop=True)
    y, pred, prob = cv["y"][keep], cv["pred"][keep], cv["prob"][keep]
    infants = pool.patient_id.values
    out = {"n_images_scored": int(len(y)), "n_excluded_duplicates": int((~keep).sum())}

    img = class_metrics(y, pred)
    img["ci95"] = boot_ci(y, [pred], infants, {"qwk_5class": qwk5, "qwk_6label": qwk6,
                                               "macro_f1_present": macro_f1_present}, n_boot, seed)
    out["image"] = img

    keys = unit_keys(pool)
    U = aggregate_units(keys, y, pred, prob)
    src_u = pool.source.values[U["first"]]
    inf_u = infants[U["first"]]
    present_src = [s for s in SOURCES if (src_u == s).any()]     # Multi-View is dev-only (no CV rows)
    out["per_source_unit"] = _per_source_units(U, src_u, inf_u, present_src, n_boot, seed)

    # operating point and temperature: fitted on CV OOF only
    t = operating_point(U["score"], U["y"], 0.90)
    T = fit_temperature(U["prob"], U["y"])      # Amendment 1: fitted on CV-OOF UNIT driving-photo vectors
    out["operating_point"] = {"rule": "largest threshold on max-over-photos(1-p_normal) with CV-OOF "
                                      "unit-level any-ROP sensitivity >= 0.90 (units pooled over sources)",
                              "cv": op_report(U["score"], U["y"], t),
                              "cv_per_source": {s: op_report(U["score"][src_u == s], U["y"][src_u == s], t)
                                                for s in present_src}}
    out["temperature"] = {"T": T, "fitted_on": "CV-OOF unit-level driving-photo probability vectors",
                          "cv_unit_driving_photo": cal_block(U["prob"], U["y"], T),
                          "cv_image_same_T_descriptive": cal_block(prob, y, T)}

    if arm.get("dev"):
        ddf, dkeys = _dev_frame()
        dinf = ddf.patient_id.values
        for z in arm["dev"]:
            if len(z["y_true"]) != len(ddf) or not (z["y_true"] == ddf.label.values).all():
                raise ValueError("dev misaligned")
        dy = ddf.label.values
        dU = [aggregate_units(dkeys, dy, z["y_pred"], z["prob"]) for z in arm["dev"]]
        ufirst = dU[0]["first"]; dinf_u = dinf[ufirst]; uy = dU[0]["y"]
        out["dev"] = {
            "n_images": int(len(dy)), "n_eyes": int(len(uy)), "n_checkpoints": len(arm["dev"]),
            "image": _list_metrics(dy, [z["y_pred"] for z in arm["dev"]], dinf, n_boot, seed),
            "eye": _list_metrics(uy, [u["pred"] for u in dU], dinf_u, n_boot, seed),
            "operating_point_at_cv_threshold": [op_report(u["score"], u["y"], t) for u in dU],
            "calibration_at_cv_T": {
                "eye_driving_photo": [cal_block(u["prob"], u["y"], T) for u in dU],
                "image_descriptive": [cal_block(z["prob"], dy, T) for z in arm["dev"]]},
        }

    # Amendment 3(b): the ordinal decoder beside argmax (argmax above stays primary)
    dec = decoded_lists(arm, cv, keep, pool.fold.values)
    DU = aggregate_units(keys, y, dec["cv"], prob)
    out["decoded"] = {"rule": DECODER_RULE, "thresholds_per_fold": dec["thresholds_per_fold"],
                      "image": _unit_metrics({"y": y, "pred": dec["cv"]}, infants, n_boot, seed),
                      "per_source_unit": _per_source_units(DU, src_u, inf_u, present_src, n_boot, seed)}
    if arm.get("dev"):
        dUd = [aggregate_units(dkeys, dy, p, z["prob"]) for p, z in zip(dec["dev"], arm["dev"])]
        out["dev"]["decoded"] = {
            "thresholds": dec["thresholds_all_folds"], "fitted_on": "all five folds' CV OOF rows of this arm-seed",
            "image": _list_metrics(dy, dec["dev"], dinf, n_boot, seed, confusion=True),
            "eye": _list_metrics(uy, [u["pred"] for u in dUd], dinf_u, n_boot, seed, confusion=True)}
    return out


# ------------------------------------------------------------------ reproduction check
def reproduction_check(tol=6e-4) -> dict:
    """Run FIRST, WITHOUT the duplicate exclusion (that is how the committed numbers were made)."""
    me = json.loads((ROOT / "results/rop/matched_equivalence.json").read_text())
    dt = json.loads((ROOT / "results/rop/dev_eval/dev_table.json").read_text())
    pool = pool_frame(PREP)
    ddf, dkeys = _dev_frame()
    rec, ok = {"seeds": {}}, True
    for seed in (42, 1337):
        arm = load_baseline_C(seed)
        assemble_cv(arm, pool)                                  # alignment asserts
        got = [round(macro_f1_present(z["y_true"], z["y_pred"]), 4) for z in arm["val"]]
        ref = me["seeds"][str(seed)]["arms"]["flat_natural"]["per_fold_macro_f1"]
        eyes = []
        for z in arm["dev"]:
            u = aggregate_units(dkeys, ddf.label.values, z["y_pred"], z["prob"])
            eyes.append(qwk6(u["y"], u["pred"]))
        eye = round(float(np.mean(eyes)), 4)
        eref = dt["cells"][f"flat_s{seed}"]["coprimary_eye_level"]["qwk"]["mean"]
        good = all(abs(a - b) < tol for a, b in zip(got, ref)) and abs(eye - eref) < tol
        ok &= good
        rec["seeds"][str(seed)] = {"per_fold_macro_f1": {"got": got, "committed": ref},
                                   "dev_eye_qwk_6label": {"got": eye, "committed": eref}, "match": bool(good)}
    rec["passed"] = bool(ok)
    rec["tolerance"] = "abs diff < 0.0006 (3 dp on 4-dp committed values)"
    rec["exclusion_applied"] = False
    return rec


def pooled_scores(arms, folds_dir=PREP, exclude=PREP / "exclude_duplicates.csv", n_boot=2000, seed=42,
                  decoded=False) -> dict:
    """Amendment 1(c): headline numbers pooled over every prediction list of every seed
    (CV: one OOF list per seed; dev: seeds x 5 fold checkpoints), per-replicate mean over the
    lists, one infant-clustered CI per number. decoded=True: the same from each arm-seed's
    Amendment 3(b) decoded predictions (CV cross-fitted; dev with its all-fold cuts)."""
    from models.common.units import unit_keys
    pool = pool_frame(folds_dir)
    cvs = [assemble_cv(a, pool) for a in arms]
    keep = ~pool.image_path.isin(read_exclusions(exclude)).values
    pool = pool[keep].reset_index(drop=True)
    y = cvs[0]["y"][keep]
    decs = [decoded_lists(a, c, keep, pool.fold.values) for a, c in zip(arms, cvs)] if decoded else None
    preds = [d["cv"] for d in decs] if decoded else [c["pred"][keep] for c in cvs]
    ms = {"qwk_5class": qwk5, "qwk_6label": qwk6, "macro_f1_present": macro_f1_present}
    point = lambda yy, P: {m: round(float(np.mean([f(yy, p) for p in P])), 4) for m, f in ms.items()}  # noqa: E731
    out = {"n_lists_cv": len(preds), "cv_image": point(y, preds) | {"ci95": boot_ci(y, preds, pool.patient_id.values, ms, n_boot, seed)}}
    Us = [aggregate_units(unit_keys(pool), y, p, c["prob"][keep]) for p, c in zip(preds, cvs)]
    first = Us[0]["first"]; src_u = pool.source.values[first]; inf_u = pool.patient_id.values[first]
    out["cv_per_source_unit"] = {}
    for s_ in SOURCES:
        m = src_u == s_
        if m.any():
            P = [u["pred"][m] for u in Us]
            out["cv_per_source_unit"][s_] = point(Us[0]["y"][m], P) | {
                "ci95": boot_ci(Us[0]["y"][m], P, inf_u[m], ms, n_boot, seed)}
    if all(a.get("dev") for a in arms):
        ddf, dkeys = _dev_frame()
        dy = ddf.label.values
        zs = [z for a in arms for z in a["dev"]]
        P = [p for d in decs for p in d["dev"]] if decoded else [z["y_pred"] for z in zs]
        dU = [aggregate_units(dkeys, dy, p, z["prob"]) for p, z in zip(P, zs)]
        dinf_u = ddf.patient_id.values[dU[0]["first"]]
        PU = [u["pred"] for u in dU]
        out["n_lists_dev"] = len(zs)
        out["dev_image"] = point(dy, P) | {"ci95": boot_ci(dy, P, ddf.patient_id.values, ms, n_boot, seed)}
        out["dev_eye"] = point(dU[0]["y"], PU) | {"ci95": boot_ci(dU[0]["y"], PU, dinf_u, ms, n_boot, seed)}
    return out


# ------------------------------------------------------------------ v3 arms
def _mean_arm(zs) -> dict:
    if len({len(z["y_true"]) for z in zs}) != 1 or len({z["prob"].shape for z in zs}) != 1:
        raise ValueError("ensemble: prediction lists of different length")
    if any(not (z["y_true"] == zs[0]["y_true"]).all() for z in zs[1:]):
        raise ValueError("ensemble: y_true differs between arms")
    prob = np.mean([z["prob"] for z in zs], 0)
    return {"y_true": zs[0]["y_true"].copy(), "y_pred": prob.argmax(1).astype(np.int64), "prob": prob}


def ensemble(arms: list) -> dict:
    """Amendment 3(a), arm E: per-image mean of the arms' probability vectors (no weights fitted),
    val fold k over fold k of each arm, dev checkpoint k over checkpoint k of each arm."""
    if any(len(a["val"]) != NFOLD for a in arms):
        raise ValueError("ensemble: every arm needs all five folds")
    out = {"val": [_mean_arm([a["val"][k] for a in arms]) for k in range(NFOLD)], "dev": None}
    if all(a.get("dev") for a in arms):
        if len({len(a["dev"]) for a in arms}) != 1:
            raise ValueError("ensemble: arms have different numbers of dev checkpoints")
        out["dev"] = [_mean_arm([a["dev"][k] for a in arms]) for k in range(len(arms[0]["dev"]))]
    return out


def load_v3(runs=RUNS) -> dict:
    """{arm: {seed: arm dict}}: the run arms from the collected run dirs that exist, C from saved npz,
    E = ensemble(A, B, C) for every seed present for all three."""
    out = {}
    for lab, arch in V3_ARMS.items():
        if lab == "E":
            continue
        d = {s: guard_path(Path(runs) / f"{arch}_s{s}") for s in V3_SEEDS}
        out[lab] = {s: load_arm_dir(p) for s, p in d.items() if p.is_dir()}
    out["C"] = {s: load_baseline_C(s) for s in (42, 1337)}
    out["E"] = {s: ensemble([out[k][s] for k in ENSEMBLE]) for s in V3_SEEDS
                if all(s in out[k] for k in ENSEMBLE)}
    return out


def _dev_eyes(zs, preds=None) -> tuple:
    """Multi-View eye units of dev prediction lists: (eye labels, eye infants, [eye predictions]).
    preds: per-image predictions to aggregate instead of each list's y_pred (the decoded variant)."""
    ddf, dkeys = _dev_frame()
    for z in zs:
        if len(z["y_true"]) != len(ddf) or not (z["y_true"] == ddf.label.values).all():
            raise ValueError("dev misaligned")
    preds = [z["y_pred"] for z in zs] if preds is None else preds
    dU = [aggregate_units(dkeys, ddf.label.values, p, z["prob"]) for p, z in zip(preds, zs)]
    return dU[0]["y"], ddf.patient_id.values[dU[0]["first"]], [u["pred"] for u in dU]


def arm_contrasts(arms: dict, folds_dir=PREP, exclude=PREP / "exclude_duplicates.csv", n_boot=2000, seed=42,
                  decoded=False) -> dict:
    """arms = {arm: [arm dict per present seed]}. CV: one OOF list per seed, image level, dedup applied.
    Dev: every checkpoint of every seed, Multi-View eye level. decoded=True: the same pairs on the
    Amendment 3(b) decoded predictions (CV cross-fitted per arm-seed; dev with its all-fold cuts)."""
    ms = {"qwk_5class": qwk5, "macro_f1_present": macro_f1_present}
    arms = {k: v for k, v in arms.items() if v}
    skipped = lambda P: [f"{a}-{b}" for a, b in PAIRS if f"{a}-{b}" not in P]  # noqa: E731
    pool = pool_frame(folds_dir)
    keep = ~pool.image_path.isin(read_exclusions(exclude)).values
    cvs = {k: [assemble_cv(a, pool) for a in v] for k, v in arms.items()}
    y = next(iter(cvs.values()))[0]["y"][keep]
    if decoded:
        decs = {k: [decoded_lists(a, c, keep, pool.fold.values[keep]) for a, c in zip(arms[k], v)]
                for k, v in cvs.items()}
        cvp = {k: [d["cv"] for d in v] for k, v in decs.items()}
    else:
        cvp = {k: [c["pred"][keep] for c in v] for k, v in cvs.items()}
    P = contrasts(y, cvp, pool.patient_id.values[keep], ms, n_boot, seed)
    out = {"cv_image": {"status": "CV pooled out-of-fold, image level, 14 duplicates excluded; selection-rule "
                                  "context, not a pre-registered endpoint (no 'significant' language)",
                        "n_images": int(len(y)), "pairs": P, "not_computed": skipped(P)}}
    dev = {k: [z for a in v for z in a["dev"]] for k, v in arms.items() if all(a.get("dev") for a in v)}
    if dev:
        eyes = {k: _dev_eyes(zs, [p for d in decs[k] for p in d["dev"]] if decoded else None)
                for k, zs in dev.items()}
        uy, ginf, _ = next(iter(eyes.values()))
        P = contrasts(uy, {k: e[2] for k, e in eyes.items()}, ginf, ms, n_boot, seed)
        out["dev_eye"] = {"status": "descriptive, not an endpoint (dev is development-external)",
                          "unit": "Multi-View eye; groups = infants", "n_eyes": int(len(uy)),
                          "pairs": P, "not_computed": skipped(P)}
        out["dev_eye_confusion"] = {
            "classes": CLASSES, "rows": "label", "cols": "prediction",
            "arms": {k: {"n_lists": len(e[2]), "mean": _confusion_mean(e[0], e[2])} for k, e in eyes.items()}}
        for name, pr in P.items():                    # a pair with a transductive arm is transductive
            note = transductive_note(name)
            if note:
                pr["status"] = note
        for k, c in out["dev_eye_confusion"]["arms"].items():
            if k in TRANSDUCTIVE_DEV:
                c["status"] = TRANSDUCTIVE_DEV[k]
    if decoded:
        for k in ("cv_image", "dev_eye"):
            if k in out:
                out[k]["decode"] = DECODER_RULE
    return out


def transductive_note(pair: str):
    """The transductive status of a dev contrast "X-Y" if either arm saw the dev images, else None."""
    return next((TRANSDUCTIVE_DEV[a] for a in pair.split("-") if a in TRANSDUCTIVE_DEV), None)


def mark_dev(r: dict, lab: str) -> None:
    """Amendment 4(b): label every dev block of a transductive arm (TRANSDUCTIVE_DEV) in place."""
    if lab not in TRANSDUCTIVE_DEV:
        return
    blocks = [s["dev"] for s in r.get("seeds", {}).values() if "dev" in s]
    for po in (r.get("pooled_over_seeds", {}), r.get("pooled_over_seeds", {}).get("decoded", {})):
        blocks += [po[k] for k in ("dev_image", "dev_eye") if k in po]
    for b in blocks:
        b["status"] = TRANSDUCTIVE_DEV[lab]


# ------------------------------------------------------------------ Amendment 5: held-out sources, v4 selection
V4_RULE = ("Amendment 5c: adopt I if CV QWK5(I) - QWK5(B) >= +0.010 and LOSO(I) >= LOSO(B) - 0.010; else "
           "adopt H under the same two conditions; else B stays. CV QWK5 = the selection table (pooled "
           "out-of-fold, both seeds); LOSO = mean QWK5 over the three held-out sources (Shenzhen site fold, "
           "Ostrava, ROP-VL), pooled over both seeds; differences of the reported 4-dp numbers. An arm "
           "without both seeds or without all three held-out sources is not adoptable.")


def score_loso(arm_seeds: list, folds_dir=PREP, exclude=PREP / "exclude_duplicates.csv", n_boot=2000,
               seed=42) -> dict:
    """Amendment 5a.5/5c for one arm: every held-out source that EVERY given seed carries (arm["loso"],
    see load_arm_dir), scored on the val rows of its fold file (fold_site.csv = Shenzhen,
    fold_loso_<src>.csv), image level, the 14 exact duplicates excluded as in CV. Each number is the
    mean over the seeds' lists, with one infant-clustered CI of that mean (as pooled_scores).
    train_support = class counts of the fold file's training rows less the duplicates (the rows the
    kernel trained on), so folds_dir is the arm's pool; the val sides of the v3 and v4 fold files are
    identical. mean_qwk_5class = mean over the scored sources; the 5c guard needs all three."""
    ms = {"qwk_5class": qwk5, "macro_f1_present": macro_f1_present, "accuracy": accuracy}
    dups = read_exclusions(exclude) if exclude is not None else set()
    srcs = [s for s in LOSO_FOLDS if arm_seeds and all(s in a.get("loso", {}) for a in arm_seeds)]
    per = {}
    for s in srcs:
        path = guard_path(Path(folds_dir) / f"{LOSO_FOLDS[s]}.csv")
        f = pd.read_csv(path)
        val = f[f.split == "val"].reset_index(drop=True)
        zs = [a["loso"][s] for a in arm_seeds]
        if any(len(z["y_true"]) != len(val) or not (z["y_true"] == val.label.values).all() for z in zs):
            raise ValueError(f"{s}: predictions not aligned with the {len(val)} val rows of {path.name}")
        keep = ~val.image_path.isin(dups).values
        y, g = val.label.values[keep], val.patient_id.values[keep]
        P = [z["y_pred"][keep] for z in zs]
        tr = f[(f.split == "train") & ~f.image_path.isin(dups)]
        per[s] = {"fold_file": path.name, "n_images": int(len(y)), "n_excluded_duplicates": int((~keep).sum()),
                  "n_infants": int(len(np.unique(g))),
                  **{m: float(np.mean([fn(y, p) for p in P])) for m, fn in ms.items()},
                  "ci95": boot_ci(y, P, g, ms, n_boot, seed),
                  "per_class_recall_mean": {CLASSES[c]: round(float(np.mean([(p[y == c] == c).mean() for p in P])), 4)
                                            for c in sorted(set(y.tolist()))},
                  "train_support": {c: int((tr.label == i).sum()) for i, c in enumerate(CLASSES)}}
    mean = round(float(np.mean([per[s]["qwk_5class"] for s in srcs])), 4) if srcs else None
    for r in per.values():
        r.update({m: round(r[m], 4) for m in ms})
    return {"mean_qwk_5class": mean, "sources": srcs, "all_three_sources": len(srcs) == len(LOSO_FOLDS),
            "n_lists": len(arm_seeds)} | per


def select_v4(scores: dict) -> dict:
    """Amendment 5c on what score_v3 writes: cv = selection_rule.table[arm].qwk_5class, loso =
    loso[arm].mean_qwk_5class. I first, then H, else B. An arm that was not run, or whose CV or
    LOSO numbers rest on fewer than both seeds or fewer than the three held-out sources, is not
    adoptable (named in reasons); if B is incomplete, nothing is compared and B stays. Never raises
    on absent arms."""
    tab = (scores.get("selection_rule") or {}).get("table") or {}
    lo = scores.get("loso") or {}
    cv = {a: tab[a]["qwk_5class"] if a in tab else None for a in ("B", "H", "I")}
    loso = {a: lo[a].get("mean_qwk_5class") if a in lo else None for a in ("B", "H", "I")}
    n = len(V3_SEEDS)

    def incomplete(a):
        if a not in tab:
            return f"{a}: not run (no CV scores)"
        if tab[a].get("n_lists") != n:
            return f"{a}: CV pooled over {tab[a].get('n_lists')} of {n} seeds"
        if a not in lo:
            return f"{a}: no held-out-source folds scored (no LOSO guard)"
        if set(lo[a].get("sources", ())) != set(LOSO_FOLDS) or lo[a].get("n_lists") != n:
            return (f"{a}: LOSO scored on {lo[a].get('sources')} over {lo[a].get('n_lists')} of {n} seeds; the "
                    f"guard is the mean over all three held-out sources {list(LOSO_FOLDS)}, both seeds")
        return None

    reasons, chosen = [], "B"
    base = incomplete("B")
    if base:
        reasons.append(f"{base} -> no arm can be compared with B")
    for a in ("I", "H"):
        why = incomplete(a)
        if why or base:
            reasons.append(f"{why or f'{a}: inputs complete, B is not'} -> {a} not adoptable")
            continue
        dc, dl = round(cv[a] - cv["B"], 4), round(loso[a] - loso["B"], 4)
        ok = dc >= V4_MARGIN_CV and dl >= -V4_MARGIN_LOSO
        reasons.append(f"{a}: CV QWK5 {a}-B = {dc:+.4f} (needs >= +{V4_MARGIN_CV:.3f}), LOSO {a}-B = {dl:+.4f} "
                       f"(needs >= -{V4_MARGIN_LOSO:.3f}) -> {'adopt ' + a if ok else 'not adopted'}")
        if ok:
            chosen = a
            break
    if chosen == "B":
        reasons.append("B stays")
    return {"rule": V4_RULE, "chosen": chosen, "margin_cv": V4_MARGIN_CV, "margin_loso": V4_MARGIN_LOSO,
            "cv": cv, "loso": loso, "reasons": reasons}


def score_v3(runs=RUNS, n_boot=2000, seed=42) -> dict:
    """Every arm per seed (score_arm) and pooled over its present seeds (pooled_scores), the
    selection-rule table, the paired contrasts with C, and dev eye confusion matrices; each also
    for the Amendment 3(b) decoded variant (under "decoded" keys and contrasts_decoded).
    Amendment 5: "loso" for every arm whose seeds all carry held-out-source folds (score_loso, on
    the arm's own pool) and "v4_selection" (select_v4)."""
    loaded = load_v3(runs)
    arms, sel = {}, {}
    for lab, seeds in loaded.items():
        r = {"arch": V3_ARMS.get(lab, "flat_natural EfficientNetV2-S (saved npz)"),
             "seeds_present": sorted(seeds), "status": "present" if seeds else "absent"}
        if seeds:
            r["seeds"] = {str(s): score_arm(a, n_boot=n_boot, seed=seed) for s, a in seeds.items()}
            r["pooled_over_seeds"] = pooled_scores(list(seeds.values()), n_boot=n_boot, seed=seed)
            r["pooled_over_seeds"]["decoded"] = pooled_scores(list(seeds.values()), n_boot=n_boot, seed=seed,
                                                              decoded=True)
            cv = r["pooled_over_seeds"]["cv_image"]
            ar = {s: x["image"]["arop_binary"] for s, x in r["seeds"].items()}
            sel[lab] = {"qwk_5class": cv["qwk_5class"], "ci95": cv["ci95"]["qwk_5class"],
                        "n_lists": r["pooled_over_seeds"]["n_lists_cv"],
                        "arop_binary": {k: round(float(np.mean([v[k] for v in ar.values()])), 4) for k in ("recall", "f1")},
                        "arop_binary_per_seed": ar}
        mark_dev(r, lab)
        arms[lab] = r
    con = arm_contrasts({k: list(v.values()) for k, v in loaded.items()}, n_boot=n_boot, seed=seed)
    cond = arm_contrasts({k: list(v.values()) for k, v in loaded.items()}, n_boot=n_boot, seed=seed, decoded=True)
    loso = {lab: {"pool": "v4" if lab in V4_ARMS else "v3"}
            | score_loso(list(seeds.values()), folds_dir=PREP / "v4" if lab in V4_ARMS else PREP,
                         n_boot=n_boot, seed=seed)
            for lab, seeds in loaded.items() if seeds and all(a.get("loso") for a in seeds.values())}
    # every v3 arm (E included) with both seeds; until then pooled numbers rest on fewer lists than C's
    complete = all(sorted(loaded.get(k, {})) == [42, 1337] for k in V3_ARMS if k not in V4_ARMS)
    complete_v4 = all(sorted(loaded.get(k, {})) == [42, 1337] for k in "ABCDEFGHI")    # Amendment 5: A-I
    res = {"complete": complete, "complete_v4": complete_v4, "arms": arms,
            "selection_rule": {"rule": "CV pooled out-of-fold image-level QWK, 5-class map (AP-ROP folded onto "
                                       "stage 4/5), computed once per arm, no dev involvement; AP-ROP scored "
                                       "separately as the binary branch metric (mean over the arm's seeds)",
                               "table": sel},
            "contrasts": {k: v for k, v in con.items() if k != "dev_eye_confusion"},
            "dev_eye_confusion": con.get("dev_eye_confusion"),
            "decoder": {"rule": DECODER_RULE, "status": "Amendment 3(b), pre-registered addition made after the "
                                                        "dev results; argmax remains the primary decode"},
            "contrasts_decoded": {k: v for k, v in cond.items() if k != "dev_eye_confusion"},
            "dev_eye_confusion_decoded": cond.get("dev_eye_confusion"),
            "loso": loso}
    res["v4_selection"] = select_v4(res)
    return res


# ------------------------------------------------------------------ entry points
def _reproduced() -> dict:
    rec = reproduction_check()
    print(json.dumps(rec, indent=1))
    if not rec["passed"]:
        raise SystemExit("reproduction check FAILED - writing nothing")
    return rec


def _write(out: dict, path) -> None:
    if "torch" in sys.modules:
        raise RuntimeError("torch was imported - this script must run no model")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(out, indent=1) + "\n")
    print("->", path)


def run_c(out_path, n_boot=2000) -> dict:
    rec = _reproduced()
    res = {s: score_arm(load_baseline_C(s), n_boot=n_boot) for s in (42, 1337)}
    out = {"protocol": "docs/paper/PROTOCOL_V2.md", "arm": "C (flat_natural, EfficientNetV2-S)",
           "bootstrap": {"replicates": n_boot, "seed": 42, "cluster": "infant", "ci": "percentile 95"},
           "dedup": "14 images of exclude_duplicates.csv dropped from every CV metric",
           "reproduction_check": rec,
           "seeds": {str(s): r for s, r in res.items()},
           "pooled_over_seeds": pooled_scores([load_baseline_C(42), load_baseline_C(1337)], n_boot=n_boot),
           "note": "T and the operating point are per arm-seed (seeds block); never averaged"}
    _write(out, out_path)
    return out


def run_v3(runs=RUNS, out_path=ROOT / "results/rop/v3/v3_scores.json", n_boot=2000) -> dict:
    rec = _reproduced()
    out = {"protocol": "docs/paper/PROTOCOL_V2.md",
           "bootstrap": {"replicates": n_boot, "seed": 42, "cluster": "infant", "ci": "percentile 95",
                         "pooling": "Amendment 1(c): mean over every list of every present seed, per replicate"},
           "dedup": "14 images of exclude_duplicates.csv dropped from every CV metric",
           "reproduction_check": rec} | score_v3(runs, n_boot) | {
           "note": "T and the operating point are per arm-seed (seeds blocks); never averaged. Dev is "
                   "development-external: every dev number is descriptive, not an endpoint."}
    _write(out, out_path)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "results/rop/v3/baseline_C_scores.json"))
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--v3", action="store_true", help="score arms A-I from --runs (C from saved npz); write --out-v3")
    ap.add_argument("--runs", default=str(RUNS))
    ap.add_argument("--out-v3", default=str(ROOT / "results/rop/v3/v3_scores.json"))
    a = ap.parse_args()
    if a.v3:
        run_v3(a.runs, a.out_v3, a.boot)
    else:
        run_c(a.out, a.boot)


if __name__ == "__main__":
    main()
