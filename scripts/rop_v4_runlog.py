"""The ROP v4 run log (spec docs/superpowers/specs/2026-10-31-rop-v4-design.md, "The run log"): one
markdown document recording, for every arm, seed, fold and epoch, what went in and what came out.

    .venv/bin/python scripts/rop_v4_runlog.py        # -> docs/v4/V4_RUN_LOG.md

Every number is read or computed from a data file; nothing is typed by hand:
  runs    results/rop/v3/runs/<arm dir>_s<seed>/summary.json and, per fold, metrics.json (the per-epoch
          history), config.yaml (the as-run config) and val_predictions.npz (final-epoch QWK5)
  rows    the fold manifests (results/prepared/rop_staging/ for pool v3, .../v4/ for pool v4) less the
          exclude_duplicates.csv rows in training, which the kernel drops at run time
  timing  the downloaded Kaggle logs kaggle_cache/v3/<slug>/*.log (rop_v3_report_data.kernel_timing;
          absent logs give "n/a")
  scores  results/rop/v3/v3_scores.json (scripts/rop_v3_score.py --v3)
  data    results/rop/v4/audit/audit_summary.json, results/prepared/rop_staging/v4/v4_summary.json,
          results/prepared/rop_staging/splits_summary.json and test_dev.csv
  text    docs/v4/README.md (hand-written, included verbatim) and the "Ruling" lines of the
          gitignored SDD ledger, verbatim
No model is run and torch is never imported. Any path containing "test_locked" is refused.
"""
from __future__ import annotations

import argparse
import datetime
import glob
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
from scripts import rop_v3_report_data as R  # noqa: E402
from scripts import rop_v3_score as S  # noqa: E402

RUNS = S.RUNS
PREP = S.PREP
SCORES = ROOT / "results/rop/v3/v3_scores.json"
OUT = ROOT / "docs/v4/V4_RUN_LOG.md"
AUDIT = ROOT / "results/rop/v4/audit/audit_summary.json"
POOL_SUMMARY = PREP / "v4" / "v4_summary.json"
SPLITS_SUMMARY = PREP / "splits_summary.json"
README = ROOT / "docs/v4/README.md"
LEDGER = ROOT / ".superpowers/sdd/2026-10-31-rop-v4-implementation/progress.md"
KAGGLE = ROOT / "kaggle_cache/v3"

ARMS = "ABCDEFGHI"
CLASSES = S.CLASSES
RUN_ARMS = {lab: d for lab, d in S.V3_ARMS.items() if lab != "E"}   # E is derived by the scorer
# Arm B's LOSO-only kernel (push_v3_kernels.py --loso --slug-suffix loso, v3 pool): its folds join B's
# run dir, its log lives under its own slug. Every other kernel slug is rop_v3_report_data.KERNEL.
EXTRA_KERNELS = {"B": ("rop-v3-dinov2-s-s{s}-lr1e-4-loso",)}
FOLD_ORDER = [f"fold_{k}" for k in range(S.NFOLD)] + [S.LOSO_FOLDS[s] for s in ("shenzhen", "ostrava", "ropvl")]
FOLD_LABEL = {**{f"fold_{k}": f"CV fold {k}" for k in range(S.NFOLD)},
              "fold_site": "held-out source Shenzhen (the v3 site fold)",
              "fold_loso_ostrava": "held-out source Ostrava (leave-one-source-out)",
              "fold_loso_ropvl": "held-out source ROP-VL (leave-one-source-out)"}
RECIPE_KEYS = ["model.arch", "preprocess.image_size", "preprocess.normalize", "preprocess.clahe",
               "train.lr", "train.layer_decay", "train.weight_decay", "train.epochs", "train.batch_size",
               "train.grad_accum", "train.label_smoothing", "train.loss", "train.class_balanced",
               "train.class_balanced_beta", "train.weighted_sampler", "train.strong_aug",
               "train.site_randomization", "train.amp", "train.ema_decay", "train.head_warmup_epochs",
               "model.pretrained", "model.dropout", "model.drop_path", "eval.early_stopping",
               "eval.primary_metric"]
BLOCKS = ("train.aug_v4", "train.bag_loss", "train.uda")
PER_RUN_KEYS = {"data.manifest", "seed"}      # differ by design between folds / seeds


# ------------------------------------------------------------------ guarded readers
_guard = S.guard_path            # raises on any path containing "test_locked" (or "locked_eval")


def _json(p):
    p = _guard(p)
    return json.loads(p.read_text()) if p.is_file() else None


def _yaml(p):
    p = _guard(p)
    return yaml.safe_load(p.read_text()) if p.is_file() else None


def _csv(p):
    p = _guard(p)
    return pd.read_csv(p) if p.is_file() else None


def _rel(p) -> str:
    p = Path(p)
    try:
        return str(p.resolve().relative_to(ROOT))
    except ValueError:
        return str(p)


# ------------------------------------------------------------------ formatting
def f4(x) -> str:
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{float(x):.4f}"


def ci(c) -> str:
    return f"[{f4(c[0])}, {f4(c[1])}]" if c else ""


def table(header: list, rows: list) -> list[str]:
    out = ["| " + " | ".join(map(str, header)) + " |", "|" + "---|" * len(header)]
    return out + ["| " + " | ".join(map(str, r)) + " |" for r in rows]


def flat(d: dict, pre: str = "") -> dict:
    out = {}
    for k, v in (d or {}).items():
        key = f"{pre}{k}"
        if isinstance(v, dict):
            out |= flat(v, key + ".")
        else:
            out[key] = v
    return out


def kv(d: dict) -> str:
    return ", ".join(f"{k} {show(v)}" for k, v in (d or {}).items())


def show(v) -> str:
    if isinstance(v, float):
        return f"{v:g}"
    if isinstance(v, list):
        return "[" + ", ".join(show(x) for x in v) + "]"
    return str(v)


# ------------------------------------------------------------------ inputs
def git_head() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True,
                              text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def arm_seeds(runs_dir: Path, lab: str) -> dict:
    """{seed: run dir} for the arm's run dirs <name>_s<seed>/ that carry a summary.json."""
    if lab not in RUN_ARMS or not Path(runs_dir).is_dir():
        return {}
    rx = re.compile(rf"^{re.escape(RUN_ARMS[lab])}_s(\d+)$")
    found = {int(m.group(1)): p for p in Path(runs_dir).iterdir()
             if (m := rx.match(p.name)) and (p / "summary.json").is_file()}
    order = [s for s in S.V3_SEEDS if s in found] + sorted(set(found) - set(S.V3_SEEDS))
    return {s: found[s] for s in order}


def fold_dirs(run: Path) -> list[str]:
    have = [d for d in FOLD_ORDER if (run / d).is_dir()]
    return have + sorted(p.name for p in run.iterdir() if p.is_dir() and p.name.startswith("fold_")
                         and p.name not in have)


def fold_rows(prep: Path, pool: str, fold: str, dups: set) -> dict | None:
    """Training rows per class (exclude_duplicates.csv rows dropped, as the kernel does), validation
    rows per class, and the sources / infants of each side, from the pool's fold file."""
    path = (Path(prep) if pool == "v3" else Path(prep) / pool) / f"{fold}.csv"
    df = _csv(path)
    if df is None:
        return None
    tr_all = df[df.split == "train"]
    tr = tr_all[~tr_all.image_path.isin(dups)]
    va = df[df.split == "val"]
    count = lambda x: {c: int((x.cls6 == c).sum()) for c in CLASSES}  # noqa: E731
    by_src = lambda x: ", ".join(f"{s} {n}" for s, n in sorted(x.source.value_counts().items()))  # noqa: E731
    return {"path": path, "train": count(tr), "val": count(va), "n_train_file": len(tr_all),
            "n_dups": len(tr_all) - len(tr), "val_sources": by_src(va), "val_infants": int(va.patient_id.nunique()),
            "train_sources": by_src(tr), "train_infants": int(tr.patient_id.nunique())}


def _logs(arm: str, seed: int, cache: Path) -> list[str]:
    slugs = ((R.KERNEL[arm],) + EXTRA_KERNELS.get(arm, ())) if arm in R.KERNEL else ()
    return sorted(p for s in slugs for p in glob.glob(str(Path(cache) / s.format(s=seed) / "*.log"))
                  if "probe" not in p)


def timing(arm: str, seed: int, cache: Path) -> dict:
    """{summary fold key: kernel_timing entry} over the arm's kernels (n/a when no log is present)."""
    if arm not in R.KERNEL:
        return {}
    out = {}
    for slug in (R.KERNEL[arm],) + EXTRA_KERNELS.get(arm, ()):
        out |= {t["fold"]: t for t in R.kernel_timing(arm, seed, cache=cache, slug=slug)}
    return out


def bag_epochs(arm: str, seed: int, cache: Path) -> dict:
    """Arm D's epoch composition per fold from the kernel log: the "[bag] plan" and "[bag] epoch 1" lines."""
    out, fold = {}, None
    for path in _logs(arm, seed, cache):
        for e in json.loads(Path(path).read_text()):
            x = e["data"]
            if x.startswith("=== ") and (m := re.search(r"\| (fold\w+) \|", x)):
                fold = m.group(1)
            elif x.startswith("[bag] plan:") and fold and (m := re.search(r"(\d+) image rows", x)):
                out.setdefault(fold, {})["image_rows"] = int(m.group(1))
                if k := re.search(r"k=(\d+)", x):
                    out[fold]["k"] = int(k.group(1))
            elif x.startswith("[bag] epoch 1:") and fold and (m := re.search(
                    r"(\d+) image batches .*?, (\d+) bag batches = (\d+) sessions / (\d+) frames", x)):
                out.setdefault(fold, {}).update(dict(zip(("image_batches", "bag_batches", "sessions", "frames"),
                                                         map(int, m.groups()))))
    return out


def final_qwk5(fold_dir: Path):
    p = fold_dir / "val_predictions.npz"
    if not _guard(p).is_file():
        return None
    z = np.load(p)
    return S.qwk5(z["y_true"], z["y_pred"])


# ------------------------------------------------------------------ section 2: the data
def section_data(prep: Path, scores: dict, audit: dict | None, pool: dict | None, splits: dict | None,
                 n_dups: int) -> list[str]:
    L = ["## 2. The data", ""]
    vals = {f"fold_{k}": v for k in range(S.NFOLD) if (v := _csv(Path(prep) / f"fold_{k}.csv")) is not None}
    if vals:
        v = pd.concat([x[x.split == "val"] for x in vals.values()], ignore_index=True)
        L += [f"### 2.1 The pool: {len(v)} photos, {v.patient_id.nunique()} infants", "",
              f"The validation rows of {', '.join(vals)} (each pool photo is validated in exactly one CV fold).", ""]
        srcs = sorted(v.source.unique())
        rows = [[s] + [int(((v.source == s) & (v.cls6 == c)).sum()) for c in CLASSES]
                + [int((v.source == s).sum()), int(v[v.source == s].patient_id.nunique())] for s in srcs]
        rows.append(["**all**"] + [int((v.cls6 == c).sum()) for c in CLASSES] + [len(v), int(v.patient_id.nunique())])
        L += table(["source"] + CLASSES + ["photos", "infants"], rows) + [""]
    else:
        L += ["Pool: fold files not present.", ""]

    L += ["### 2.2 Trained-on rows per class, v3 pool vs v4 pool (`v4_summary.json`)", ""]
    if pool:
        tb = pool["training_by_class"]
        L += table(["pool"] + CLASSES + ["total"],
                   [[k] + [tb[k][c] for c in CLASSES] + [sum(tb[k].values())] for k in tb]) + [""]
        L += table(["source", "pool"] + CLASSES,
                   [[s, k] + [d[k][c] for c in CLASSES] for s, d in pool["training_by_source"].items() for k in d])
        caps = ", ".join(f"{'other classes' if k == '_default' else k} {v}" for k, v in pool["caps"].items())
        L += ["", f"Training photos per infant and class are capped ({caps}; v4 floor {pool['floor']}); within a cap "
                  f"the v4 pool keeps the best-ranked frames ({pool['ranking']}). These counts are before the {n_dups} "
                  "exact duplicates of `exclude_duplicates.csv`, which every kernel drops from training at run time.", ""]
        oq, nde = pool.get("ostrava_quality") or {}, pool.get("near_duplicate_exclusions") or {}
        if oq:
            L += [f"The v4 pool leaves out {oq['removed_by_quality']} Ostrava frames that fail the quality check "
                  f"(by class: {kv(oq['removed_by_class'])}; {oq['v3_kept_removed']} of them were v3 training rows; "
                  f"{oq['failing_kept_by_floor']} failing frame kept by the floor) and {nde.get('images', 'n/a')} "
                  f"near-duplicate images (by source: {kv(nde.get('by_source'))}).", ""]
    else:
        L += ["v4 pool summary not present.", ""]

    L += ["### 2.3 The audit (`audit_summary.json`)", ""]
    if audit:
        nd = audit["near_duplicates"]
        L += [f"Near-duplicates: dHash distance <= {nd['threshold']} over {nd['photos']} photos "
              f"({nd['pairs_compared']} pairs compared).", ""]
        L += table(["", "pairs", "cross-fold", "different label", "same Ostrava session", "different infant"],
                   [["dHash <= " + str(nd["threshold"]), nd["pairs_total"], nd["cross_fold"], nd["different_label"],
                     nd["same_session"], nd["different_infant"]],
                    [f"true duplicates (pixel MAD <= {nd['max_pixel_mad']:g})", nd["true_duplicate"]["total"],
                     nd["true_duplicate"]["cross_fold"], nd["true_duplicate"]["different_label"],
                     nd["true_duplicate"]["same_session"], ""]])
        ex = nd["excluded_from_training"]
        L += ["", f"Removed from the v4 training rows: {ex['images']} images (by source: {kv(ex['by_source'])}).", ""]
        q = audit["ostrava_quality"]
        L += [f"Ostrava frame quality: {q['failing']} of {q['frames']} frames fail (by reason: {kv(q['by_reason'])}); "
              f"{q['v3_kept_failing']} of the {q['v3_kept_frames']} frames v3 trained on fail; "
              f"{q['infants_with_a_failing_frame']} of {q['infants']} infants have a failing frame. Rule: {q['rule']}", ""]
        L += table(["class", "frames", "failing", "blur", "dark", "bright", "infants"],
                   [[c, d["frames"], d["failing"], d["by_reason"]["blur"], d["by_reason"]["dark"],
                     d["by_reason"]["bright"], d["infants"]] for c, d in q["by_class"].items()]) + [""]
        L += table(["device", "frames", "failing", "blur", "dark", "bright", "failing by class"],
                   [[dv, d["frames"], d["failing"], d["by_reason"]["blur"], d["by_reason"]["dark"], d["by_reason"]["bright"],
                     ", ".join(f"{c} {x['failing']}/{x['frames']}" for c, x in d["by_class"].items())]
                    for dv, d in q["by_device"].items()]) + [""]
        th = audit["thresholds"]
        L += [f"Thresholds per device (percentiles: {kv(th['percentiles'])}; work size {th['work_size']}):", ""]
        L += table(["device", "min_blur", "lum_lo", "lum_hi", "frames"],
                   [[dv, f"{t['min_blur']:.2f}", f"{t['lum_lo']:.2f}", f"{t['lum_hi']:.2f}", t["n"]]
                    for dv, t in th["by_device"].items()]) + [""]
    else:
        L += ["Audit summary not present.", ""]

    L += ["### 2.4 Multi-View (the held-out hospital)", ""]
    dev = _csv(Path(prep) / "test_dev.csv")
    n_eyes = ((scores.get("contrasts") or {}).get("dev_eye") or {}).get("n_eyes")
    if dev is not None:
        L += [f"Dev half (`test_dev.csv`): {len(dev)} photos, {n_eyes if n_eyes is not None else 'n/a'} eyes "
              f"(v3_scores.json), {dev.patient_id.nunique()} infants; per class "
              + ", ".join(f"{c} {int((dev.cls6 == c).sum())}" for c in CLASSES) + "."]
    else:
        L += ["Dev half: `test_dev.csv` not present."]
    lk = ((splits or {}).get("test") or {}).get("multiview_locked")
    L += [f"Locked half: {lk['images']} photos, {lk['infants']} infants (from `splits_summary.json`; "
          "the locked file itself is never read)." if lk else "Locked half: `splits_summary.json` not present.", ""]
    return L


# ------------------------------------------------------------------ section 3: one arm
def recipe_lines(cfg: dict, summary: dict) -> list[str]:
    fc = flat(cfg)
    rows = [[k, show(fc[k]) if k in fc else ("1 (default)" if k == "train.grad_accum" else "not set")]
            for k in RECIPE_KEYS]
    for b in BLOCKS:
        rows += [[k, show(v)] for k, v in fc.items() if k.startswith(b + ".")] or [[b, "not set"]]
    rows += [["pool (summary.json)", summary.get("pool", "v3")], ["config source", fc.get("_source", "n/a")]]
    return (table(["parameter", "value"], rows)
            + ["", "<details><summary>Every key of that as-run config.yaml</summary>", ""]
            + table(["key", "value"], [[k, show(v)] for k, v in fc.items()]) + ["", "</details>"])


def fold_block(lab: str, seed: int, fold: str, run: Path, summary: dict, recipe: dict, prep: Path,
               dups: set, tim: dict, bags: dict) -> tuple[list[str], list]:
    key = fold.replace("fold_", "fold", 1)
    fs = summary.get("folds", {}).get(key, {})
    d = run / fold
    cfg = _yaml(d / "config.yaml") or {}
    m = _json(d / "metrics.json") or {}
    h = m.get("history") or {}
    rows = fold_rows(prep, summary.get("pool", "v3"), fold, dups)
    tr = cfg.get("train") or {}
    bs, ga = int(tr.get("batch_size", 0) or 0), int(tr.get("grad_accum", 1) or 1)
    L = [f"#### Arm {lab} · seed {seed} · {fold} ({FOLD_LABEL.get(fold, fold)})", ""]
    L += [f"Kernel `{fs.get('kernel', 'n/a')}` · run dir `{_rel(d)}` · as-run manifest "
          f"`{(cfg.get('data') or {}).get('manifest', 'n/a')}`", ""]
    n_img = n_steps = None
    if rows is None:
        L += ["Fold file not present: row counts n/a.", ""]
    else:
        ntr, nva = sum(rows["train"].values()), sum(rows["val"].values())
        L += [f"Fold file `{_rel(rows['path'])}`: validation = {nva} photos of {rows['val_infants']} infants "
              f"({rows['val_sources']}); training = {ntr} photos of {rows['train_infants']} infants "
              f"({rows['train_sources']}).", ""]
        L += table(["rows"] + CLASSES + ["total"],
                   [["training rows"] + [rows["train"][c] for c in CLASSES] + [ntr],
                    ["validation rows"] + [rows["val"][c] for c in CLASSES] + [nva]]) + [""]
        zero = [c for c in CLASSES if rows["train"][c] == 0 and rows["val"][c] > 0]
        if zero:
            L += [f"- validated with zero training rows: {', '.join(zero)}"]
        b = bags.get(key)
        bag_cfg = tr.get("bag_loss") or {}
        if bag_cfg.get("enabled"):
            if b and {"image_rows", "frames", "image_batches", "bag_batches"} <= set(b):
                n_img, n_steps = b["image_rows"] + b["frames"], b["image_batches"] + b["bag_batches"]
                L += [f"- images per epoch = {n_img} under the bag loss ({b['image_rows']} single-image rows + "
                      f"{b['frames']} frames drawn from {b['sessions']} multi-frame Ostrava sessions, at most "
                      f"k = {b.get('k', bag_cfg.get('k'))} per session; kernel log); training rows = {ntr}",
                      f"- steps per epoch = {n_steps} ({b['image_batches']} image batches + {b['bag_batches']} bag "
                      "batches; kernel log)"]
            else:
                L += [f"- images per epoch = n/a under the bag loss (the {ntr} training rows are visited as single "
                      f"images and as k = {bag_cfg.get('k')} frame bags; the kernel log that records the plan is "
                      "not present)", "- steps per epoch = n/a (kernel log not present)"]
        else:
            n_img = ntr
            n_steps = math.ceil(ntr / (bs * ga)) if bs else None
            L += [f"- images per epoch = {n_img} (the training rows: the fold file's {rows['n_train_file']} less "
                  f"{rows['n_dups']} exact duplicate(s) dropped at run time)",
                  f"- steps per epoch = {n_steps if n_steps is not None else 'n/a'} (= ceil({ntr} / ({bs} x {ga})), "
                  "batch_size x grad_accum)"]
        uda = tr.get("uda") or {}
        if uda.get("enabled") and n_steps:
            L += [f"- plus {uda.get('target_batch')} unlabelled Multi-View dev photos per step (self-ensembling) = "
                  f"{n_steps * int(uda.get('target_batch', 0))} unlabelled images per epoch"]
    if tr.get("head_warmup_epochs"):
        L += [f"- epochs 1..{tr['head_warmup_epochs']} train the classification head only (train.head_warmup_epochs)"]
    t = tim.get(key)
    gpu = (t or {}).get("gpu") or fs.get("gpu", "n/a")
    if t:
        L += [f"- minutes {t['minutes']} (fold start to next fold start, kernel log) · {t['img_per_s']} img/s · "
              f"GPU {gpu} · kernel log reports train {t['train_images']} val {t['val_images']}"]
        if rows and (t["train_images"] != sum(rows["train"].values()) or t["val_images"] != sum(rows["val"].values())):
            L += ["- **the kernel log's row counts differ from the fold file's**"]
    else:
        L += [f"- minutes n/a · img/s n/a (kernel log not present) · GPU {gpu} (summary.json)"]
    diff = {k: (show(recipe.get(k, "not set")), show(v)) for k, v in flat(cfg).items()
            if k not in PER_RUN_KEYS and recipe.get(k) != v}
    diff |= {k: (show(v), "not set") for k, v in recipe.items() if k not in PER_RUN_KEYS and k not in flat(cfg)}
    if cfg and diff:
        L += ["- config differs from the recipe table: "
              + "; ".join(f"`{k}` {a} -> {b}" for k, (a, b) in sorted(diff.items()))]
    L += [""]

    n_ep = len(h.get("train_loss", []))
    if n_ep:
        L += ["Per epoch (`metrics.json` history; lr = the first optimizer group, i.e. the lowest layer under "
              "layer-wise decay, at the end of the epoch):", ""]
        L += table(["Epoch", "train loss", "lr", "val acc", "val macro-F1", "val macro-P", "val macro-R"],
                   [[f"Epoch {e + 1}", f4(h["train_loss"][e]),
                     f"{h['lr'][e]:.3e}" if e < len(h.get("lr", [])) else "n/a"]
                    + [f4(h[k][e]) if e < len(h.get(k, [])) else "n/a"
                       for k in ("val_accuracy", "val_macro_f1", "val_macro_precision", "val_macro_recall")]
                    for e in range(n_ep)]) + [""]
        want = tr.get("epochs")
        if want and n_ep != int(want):
            L += [f"**History has {n_ep} epochs; the config asks for {want}.**", ""]
    else:
        L += ["Per-epoch history not present.", ""]

    q5 = final_qwk5(d)
    pc = fs.get("per_class") or {}
    nva = sum(rows["val"].values()) if rows else "n/a"
    L += [f"Final epoch (Epoch {n_ep or 'n/a'}), all {nva} validation rows: QWK (6 labels, summary.json) "
          f"{f4(fs.get('qwk'))} · QWK5 (AP-ROP folded onto stage 4/5, val_predictions.npz) {f4(q5)} · accuracy "
          f"{f4(fs.get('accuracy'))} · macro-F1 {f4(fs.get('macro_f1'))} · balanced accuracy "
          f"{f4(fs.get('balanced_accuracy'))} · AUC {f4(fs.get('auc'))}", ""]
    if pc:
        L += table(["final epoch"] + list(pc), [["recall"] + [f4(v.get("recall")) for v in pc.values()],
                                                ["support"] + [v.get("support", "n/a") for v in pc.values()]]) + [""]
    tm = (t or {})
    overview = [fold, sum(rows["train"].values()) if rows else "n/a", nva, n_img if n_img is not None else "n/a",
                n_steps if n_steps is not None else "n/a", tm.get("minutes", "n/a"), tm.get("img_per_s", "n/a"),
                f4(fs.get("accuracy")), f4(fs.get("macro_f1")), f4(fs.get("qwk")), f4(q5), f4(fs.get("auc"))]
    return L, overview


def section_arm(lab: str, runs_dir: Path, prep: Path, scores: dict, dups: set, cache: Path) -> list[str]:
    name = R.NAMES.get(lab, lab)
    L = [f"### Arm {lab}: {name}", ""]
    if lab not in RUN_ARMS:
        a = (scores.get("arms") or {}).get(lab)
        if a:
            L += [f"No run directory: `{a.get('arch')}` (seeds {a.get('seeds_present')}), scored by "
                  "`scripts/rop_v3_score.py` from saved predictions; its numbers are in section 4.", ""]
        else:
            L += ["No run directory, and not in v3_scores.json: not run yet.", ""]
        return L
    seeds = arm_seeds(runs_dir, lab)
    if not seeds:
        return L + [f"No run directory `{RUN_ARMS[lab]}_s<seed>/` under `{_rel(runs_dir)}`: not run yet.", ""]
    s0 = 42 if 42 in seeds else min(seeds)
    summ = {s: _json(p / "summary.json") for s, p in seeds.items()}
    fd = fold_dirs(seeds[s0])
    rcfg = _yaml(seeds[s0] / fd[0] / "config.yaml") if fd else None
    L += [f"Run dirs: {', '.join(f'`{_rel(p)}`' for p in seeds.values())}. Missing seeds of {list(S.V3_SEEDS)}: "
          f"{[s for s in S.V3_SEEDS if s not in seeds] or 'none'}.", ""]
    if rcfg:
        L += [f"Recipe (the as-run `config.yaml` of {fd[0]}, seed {s0}; any fold whose config differs says so):", ""]
        L += recipe_lines(rcfg, summ[s0]) + [""]
    recipe = flat(rcfg or {})
    for s in list(S.V3_SEEDS) + sorted(set(seeds) - set(S.V3_SEEDS)):
        if s not in seeds:
            L += [f"#### Arm {lab} · seed {s}: not run yet", ""]
            continue
        run = seeds[s]
        tim, bags = timing(lab, s, cache), bag_epochs(lab, s, cache)
        blocks, over = [], []
        for fold in fold_dirs(run):
            b, o = fold_block(lab, s, fold, run, summ[s], recipe, prep, dups, tim, bags)
            blocks += b
            over.append(o)
        L += [f"#### Arm {lab} · seed {s}: overview ({len(over)} folds: {', '.join(o[0] for o in over)})", ""]
        L += table(["fold", "train rows", "val rows", "images/epoch", "steps/epoch", "minutes", "img/s",
                    "final acc", "final macro-F1", "final QWK (6)", "final QWK5", "AUC"], over) + [""]
        L += blocks
    return L


# ------------------------------------------------------------------ section 4: results
def section_results(scores: dict) -> list[str]:
    arms = scores.get("arms") or {}
    L = ["## 4. Results (`v3_scores.json`)", ""]
    sel = (scores.get("selection_rule") or {})
    L += ["### 4.1 Selection table", "", f"Rule: {sel.get('rule', 'n/a')}", ""]
    tab = sel.get("table") or {}
    L += table(["arm", "CV QWK5", "95% CI", "seeds pooled", "AP-ROP binary recall", "AP-ROP binary F1"],
               [[a, f4(t["qwk_5class"]), ci(t.get("ci95")), t.get("n_lists"),
                 f4((t.get("arop_binary") or {}).get("recall")), f4((t.get("arop_binary") or {}).get("f1"))]
                if (t := tab.get(a)) else [a, "not scored yet", "", "", "", ""] for a in ARMS]) + [""]

    L += ["### 4.2 Headline per arm (pooled over seeds)", "",
          "CV = pooled out-of-fold, image level (14 exact duplicates excluded); dev = Multi-View dev half, eye level "
          "(descriptive, development-external).", ""]
    rows = []
    for a in ARMS:
        po = (arms.get(a) or {}).get("pooled_over_seeds")
        if not po:
            rows.append([a, "not scored yet"] + [""] * 7)
            continue
        cv, dv = po.get("cv_image") or {}, po.get("dev_eye") or {}
        cell = lambda b, k: f"{f4(b.get(k))} {ci((b.get('ci95') or {}).get(k))}".strip() if b else "n/a"  # noqa: E731
        rows.append([a, po.get("n_lists_cv", "n/a"), cell(cv, "qwk_5class"), cell(cv, "qwk_6label"),
                     cell(cv, "macro_f1_present"), po.get("n_lists_dev", "n/a"), cell(dv, "qwk_5class"),
                     cell(dv, "macro_f1_present"), dv.get("status", "")])
    L += table(["arm", "CV lists", "CV QWK5 [CI]", "CV QWK (6) [CI]", "CV macro-F1 [CI]", "dev lists",
                "dev eye QWK5 [CI]", "dev eye macro-F1 [CI]", "note"], rows) + [""]

    L += ["### 4.3 Held-out sources (leave-one-source-out)", ""]
    lo = scores.get("loso")
    if not lo:
        L += ["Not scored yet: v3_scores.json has no `loso` block (it appears once `scripts/rop_v3_score.py --v3` "
              "runs on collected held-out-source folds).", ""]
    else:
        rows = []
        for a in ARMS:
            if a not in lo:
                continue
            b = lo[a]
            for s in b.get("sources", []):
                r = b[s]
                rows.append([a, b.get("pool"), s, r["n_images"], r["n_infants"], f4(r["qwk_5class"]),
                             ci((r.get("ci95") or {}).get("qwk_5class")), f4(r["macro_f1_present"]), f4(r["accuracy"]),
                             r["train_support"].get("stage_4_5"), b.get("n_lists")])
        L += table(["arm", "pool", "held-out source", "photos", "infants", "QWK5", "QWK5 95% CI", "macro-F1",
                    "accuracy", "stage_4_5 training rows", "seeds"], rows) + [""]
        L += table(["arm", "mean QWK5 over sources", "sources", "all three"],
                   [[a, f4(lo[a].get("mean_qwk_5class")), ", ".join(lo[a].get("sources", [])), lo[a].get("all_three_sources")]
                    for a in ARMS if a in lo]) + [""]

    L += ["### 4.4 Paired contrasts (infant-clustered bootstrap)", ""]
    con = scores.get("contrasts") or {}
    if not con:
        L += ["No contrasts in v3_scores.json.", ""]
    for lvl, c in con.items():
        L += [f"**{lvl}** ({c.get('status', '')}; {c.get('n_images', c.get('n_eyes', 'n/a'))} "
              f"{'images' if 'n_images' in c else 'eyes'})", ""]
        L += table(["pair", "QWK5 a", "QWK5 b", "QWK5 diff", "diff 95% CI", "macro-F1 diff", "diff 95% CI", "lists", "note"],
                   [[p, f4(v["qwk_5class"]["a"]), f4(v["qwk_5class"]["b"]), f4(v["qwk_5class"]["diff"]),
                     ci(v["qwk_5class"]["diff_ci"]), f4(v["macro_f1_present"]["diff"]),
                     ci(v["macro_f1_present"]["diff_ci"]), kv(v.get("n_lists")), v.get("status", "")]
                    for p, v in (c.get("pairs") or {}).items()]) + [""]
        if c.get("not_computed"):
            L += [f"Not computed: {c['not_computed']}", ""]

    L += ["### 4.5 The v4 comparison rule (Amendment 5c)", ""]
    v4 = scores.get("v4_selection")
    if not v4:
        L += ["Not computed yet: v3_scores.json has no `v4_selection` block (it appears once "
              "`scripts/rop_v3_score.py --v3` runs after the v4 round).", ""]
    else:
        L += [f"Rule: {v4['rule']}", "", f"**Chosen arm: {v4['chosen']}**", ""]
        L += table(["arm", "CV QWK5", "mean LOSO QWK5"], [[a, f4(v4["cv"].get(a)), f4(v4["loso"].get(a))] for a in v4["cv"]])
        L += [""] + [f"- {r}" for r in v4["reasons"]] + [""]
    return L


def section_rulings(ledger: Path) -> list[str]:
    L = ["## 5. Decisions and rulings", ""]
    p = _guard(ledger)
    if not p.is_file():
        return L + [f"SDD ledger not present (`{_rel(p)}` is gitignored and lives only in the working tree that "
                    "ran the implementation).", ""]
    lines = [x for x in p.read_text().splitlines() if x.startswith("Ruling")]
    L += [f"Every line of the SDD ledger `{_rel(p)}` that starts with \"Ruling\", verbatim ({len(lines)} lines):", ""]
    return L + [f"- {x}" for x in lines] + [""]


# ------------------------------------------------------------------ the document
def build_runlog(runs_dir, prep_dir, scores_path, out_md, audit_path=AUDIT, pool_summary=POOL_SUMMARY,
                 readme_path=README, splits_summary=SPLITS_SUMMARY, ledger_path=LEDGER, kaggle_dir=KAGGLE) -> str:
    """Write the run log to out_md and return its text."""
    for p in (runs_dir, prep_dir, scores_path, out_md, audit_path, pool_summary, readme_path, splits_summary,
              ledger_path, kaggle_dir):
        _guard(p)
    runs_dir, prep_dir = Path(runs_dir), Path(prep_dir)
    scores = _json(scores_path) or {}
    dups_df = _csv(prep_dir / "exclude_duplicates.csv")
    dups = set(dups_df.image_path) if dups_df is not None else set()
    L = ["# ROP v4 run log", "",
         f"Generated {datetime.date.today().isoformat()} by `scripts/rop_v4_runlog.py` at commit `{git_head()}` "
         "(HEAD when generated). Every number below is computed from the data files named beside it "
         "(run summaries, per-epoch histories, as-run configs, fold manifests, Kaggle logs, the audit and "
         "pool summaries, `v3_scores.json`); none is typed by hand. Regenerate, never edit.", "",
         "Contents: 1. Why these numbers (`docs/v4/README.md`, verbatim) · 2. The data · 3. Every arm, seed, "
         "fold and epoch · 4. Results · 5. Decisions and rulings", ""]
    readme = _guard(readme_path)
    L += [readme.read_text().rstrip(), ""] if readme.is_file() else ["## 1. Why these numbers", "",
                                                                     f"`{_rel(readme)}` not present.", ""]
    L += section_data(prep_dir, scores, _json(audit_path), _json(pool_summary), _json(splits_summary), len(dups))
    L += ["## 3. Every arm, seed, fold and epoch", "",
          f"Arms {', '.join(ARMS)} in that order; seeds {list(S.V3_SEEDS)}. Per fold: rows per class, images and "
          "optimizer steps per epoch, time, one table row per epoch, and the final epoch's per-class recall.", ""]
    for lab in ARMS:
        L += section_arm(lab, runs_dir, prep_dir, scores, dups, Path(kaggle_dir))
    L += section_results(scores)
    L += section_rulings(Path(ledger_path))
    text = "\n".join(L).rstrip() + "\n"
    out = Path(out_md)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text)
    return text


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path, default=OUT)
    a = ap.parse_args(argv)
    text = build_runlog(RUNS, PREP, SCORES, a.out)
    print(f"-> {_rel(a.out)} ({len(text.splitlines())} lines)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
