"""Build the training-ready splits for the 6-class ROP staging benchmark.

    .venv/bin/python scripts/build_rop_splits.py

Reads  results/prepared/rop_staging/manifest.csv  (built by build_rop_staging_manifest.py)
Writes results/prepared/rop_staging/splits.csv    — one row per image, with fold assignment
       results/prepared/rop_staging/splits_summary.json

Every design decision below came out of an adversarial audit of an earlier plan. The comments
say WHY, because each one is a trap someone will otherwise walk back into.

SIX CLASSES: normal, stage_1, stage_2, stage_3, stage_4_5 (merged), arop.
  stage_4 and stage_5 are merged because stage_5 is 30 images from 27 infants at ONE hospital.
  stage_0 (45 img / 4 infants) and treated (1,063) are OUT of scope, not silently folded in.

PER-INFANT CAPS: normal 2/infant, every other class 20/infant.
  One Ostrava infant supplied 470 of 618 AP-ROP images (76% of the class). Uncapped, the model
  learns that one baby. The cap is applied AFTER fold assignment so it can never move a patient
  across a fold boundary.

THE TEST SITE IS SPLIT IN TWO. Multi-View is the held-out hospital, but if the shortcut audits
and debugging run against all of it, it is a development set that we then call held-out — the
first thing an examiner attacks. So it is split BY PATIENT into:
    multiview_dev    — audits, debugging, threshold selection. Touch freely.
    multiview_locked — opened once, for the final number. Every touch belongs in the log.

NO EARLY STOPPING, so there is no single val split. Instead:
  - 5 patient-grouped CV folds over the three training sites, stratified on class so that the
    89 stage_4_5 images (all ROP-VL) are spread across folds instead of landing in one.
  - one extra leave-one-site-out fold (Shenzhen held out) to estimate cross-site transfer
    WITHOUT touching Multi-View.
  Model selection reads pooled out-of-fold metrics; the final model retrains on everything for
  a fixed epoch budget.
"""
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
MAN = ROOT / "results" / "prepared" / "rop_staging" / "manifest.csv"
OUT = ROOT / "results" / "prepared" / "rop_staging"

SEED = 42
N_FOLDS = 5
TEST_SITE = "multiview"
SITE_VAL = "shenzhen"          # leave-one-site-out fold, chosen inside the training sites
# normal is capped hard. The floor is set by INFANTS, not images: there are 646 normal
# infants, so no cap can take normal below 646 images without discarding patient diversity —
# and patient diversity is the only thing that makes "normal" generalise to a new hospital.
# Cap 2 keeps all 646 infants while cutting normal 1,659 -> 1,109 (18.6:1 -> 12.5:1).
# The residual imbalance is corrected at the LOSS (class weights / logit adjustment), not by
# deleting more data: undersampling normal does not improve stage_4_5 recall, it only raises
# the false-positive rate on healthy infants — the project's existing 13% clinic PPV problem.
CAPS = {"normal": 2}           # every other in-scope class:
DEFAULT_CAP = 20

CLASS_MAP = {"normal": "normal", "stage_1": "stage_1", "stage_2": "stage_2",
             "stage_3": "stage_3", "stage_4": "stage_4_5", "stage_5": "stage_4_5",
             "arop": "arop"}
CLASSES = ["normal", "stage_1", "stage_2", "stage_3", "stage_4_5", "arop"]


def grouped_stratified_folds(df, n_folds, seed):
    """Assign each PATIENT to a fold, balancing class counts across folds.

    sklearn's StratifiedGroupKFold would do this, but it is not a hard dependency of this repo
    and its behaviour with a class present in only ~64 groups is not obviously better than the
    explicit greedy assignment below, which we can reason about: process patients from rarest
    dominant-class to commonest, and always drop the next patient into whichever fold currently
    has the fewest images of that patient's dominant class.
    """
    rng = np.random.default_rng(seed)
    pat = (df.groupby("patient_id")
             .agg(n=("cls6", "size"), dom=("cls6", lambda s: s.value_counts().index[0]))
             .reset_index())
    rarity = df.cls6.value_counts()
    pat["rank"] = pat.dom.map(rarity)
    pat = pat.sample(frac=1.0, random_state=seed).sort_values(["rank", "n"], ascending=[True, False])

    load = {f: Counter() for f in range(n_folds)}
    size = {f: 0 for f in range(n_folds)}
    assign = {}
    for r in pat.itertuples():
        best = min(range(n_folds), key=lambda f: (load[f][r.dom], size[f], rng.random()))
        assign[r.patient_id] = best
        load[best][r.dom] += r.n
        size[best] += r.n
    return assign


def main():
    if not MAN.exists():
        print(f"missing {MAN} — run scripts/build_rop_staging_manifest.py first", file=sys.stderr)
        return 1
    df = pd.read_csv(MAN)
    df = df[df.icrop.isin(CLASS_MAP)].copy()
    df["cls6"] = df.icrop.map(CLASS_MAP)
    print(f"in-scope images: {len(df):,}  ({df.patient_id.nunique():,} infants)")

    # ---- 1. carve off the held-out hospital, and split it by patient -------------------
    test = df[df.source == TEST_SITE].copy()
    train = df[df.source != TEST_SITE].copy()

    rng = np.random.default_rng(SEED)
    tp = np.array(sorted(test.patient_id.unique()))
    rng.shuffle(tp)
    locked = set(tp[: len(tp) // 2])
    test["split"] = np.where(test.patient_id.isin(locked), "multiview_locked", "multiview_dev")

    # ---- 2. patient-grouped, class-balanced folds over the training sites --------------
    assign = grouped_stratified_folds(train, N_FOLDS, SEED)
    train["fold"] = train.patient_id.map(assign)
    train["split"] = "train_cv"
    # the leave-one-site-out fold is a SEPARATE column: a Shenzhen patient still belongs to a
    # CV fold, it is just additionally flagged so a site-transfer run can be scored without
    # touching Multi-View at all.
    train["site_val"] = train.source == SITE_VAL

    test["fold"] = -1
    test["site_val"] = False
    out = pd.concat([train, test], ignore_index=True)

    # ---- 3. per-infant caps, applied AFTER folds so no patient can move ----------------
    out["cap"] = out.cls6.map(lambda c: CAPS.get(c, DEFAULT_CAP))
    out = out.sort_values(["patient_id", "cls6", "image_path"])
    out["rank_in_patient_class"] = out.groupby(["patient_id", "cls6"]).cumcount()
    out["kept"] = out.rank_in_patient_class < out.cap
    # caps apply to TRAINING only — val/test must stay at natural prevalence or the metrics lie
    out.loc[out.split != "train_cv", "kept"] = True

    # ---- 4. assertions that must hold, or the split is broken --------------------------
    k = out[out.kept]
    leak = set(k[k.split == "train_cv"].patient_id) & set(k[k.split != "train_cv"].patient_id)
    assert not leak, f"patient leak between train and test: {list(leak)[:5]}"
    lk = set(k[k.split == "multiview_locked"].patient_id) & set(k[k.split == "multiview_dev"].patient_id)
    assert not lk, "patient in both locked and dev halves of the test site"
    for f in range(N_FOLDS):
        pats = set(k[(k.split == "train_cv") & (k.fold == f)].patient_id)
        others = set(k[(k.split == "train_cv") & (k.fold != f)].patient_id)
        assert not (pats & others), f"patient in fold {f} and another fold"

    OUT.mkdir(parents=True, exist_ok=True)
    cols = ["image_path", "source", "patient_id", "cls6", "icrop", "device",
            "split", "fold", "site_val", "kept"]
    out[cols].to_csv(OUT / "splits.csv", index=False)

    # ---- 5. report ---------------------------------------------------------------------
    tr = k[k.split == "train_cv"]
    print("\n" + "=" * 78)
    print("TRAINING POOL (after per-infant caps)")
    print("=" * 78)
    t = pd.DataFrame({"images": tr.groupby("cls6").size(),
                      "infants": tr.groupby("cls6").patient_id.nunique()}).reindex(CLASSES)
    t["%"] = (100 * t.images / t.images.sum()).round(1)
    print(t.to_string())
    print(f"  total {len(tr):,} images / {tr.patient_id.nunique():,} infants  "
          f"| imbalance {t.images.max() / t.images.min():.1f}:1")
    print(f"  (uncapped would be {len(out[out.split == 'train_cv']):,})")

    print("\nPER-FOLD class counts — stage_4_5 must appear in EVERY fold:")
    print(pd.crosstab(tr.fold, tr.cls6).reindex(columns=CLASSES, fill_value=0).to_string())

    print("\nLEAVE-ONE-SITE-OUT fold (Shenzhen held out from the training sites):")
    print(f"  site-val images {int(tr.site_val.sum()):,} / infants {tr[tr.site_val].patient_id.nunique():,}")

    print("\n" + "=" * 78)
    print("HELD-OUT HOSPITAL (Multi-View) — split by patient")
    print("=" * 78)
    for s in ("multiview_dev", "multiview_locked"):
        sub = k[k.split == s]
        cc = sub.groupby("cls6").size().reindex(CLASSES, fill_value=0)
        print(f"  {s:<18} {len(sub):>5} images / {sub.patient_id.nunique():>3} infants   "
              + " ".join(f"{c.replace('stage_', 's')}:{int(n)}" for c, n in cc.items()))
    print("\n  multiview_dev    -> shortcut audits, debugging, threshold selection")
    print("  multiview_locked -> opened ONCE. Log every touch.")

    summary = {
        "seed": SEED, "n_folds": N_FOLDS, "test_site": TEST_SITE, "site_val": SITE_VAL,
        "caps": {**CAPS, "_default": DEFAULT_CAP},
        "train_pool": {c: int(t.images.get(c, 0)) for c in CLASSES},
        "train_infants": int(tr.patient_id.nunique()),
        "imbalance_ratio": round(float(t.images.max() / t.images.min()), 2),
        "test": {s: {"images": int((k.split == s).sum()),
                     "infants": int(k[k.split == s].patient_id.nunique())}
                 for s in ("multiview_dev", "multiview_locked")},
    }
    (OUT / "splits_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"\n-> {OUT / 'splits.csv'}\n-> {OUT / 'splits_summary.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
