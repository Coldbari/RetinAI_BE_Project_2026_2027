"""T14 — does the ROP model work equally well across infants?

Every ROP filename carries the demographics this question needs and nobody has looked:

    <id>_F_GA##_BW####_PA##_DG#_PF#_D1_S16_2.jpg
        |  |     |     |
        |  |     |     +-- postnatal age at capture (days)
        |  |     +-------- birth weight (g)
        |  +-------------- gestational age (weeks)
        +----------------- sex

Subgroup performance is expected of clinical ML, and this dataset carries richer metadata
than most published ROP work. Run on the DEVICE-STRATIFIED split and checkpoint, so that a
subgroup difference cannot be a device difference wearing a disguise.

Two things this script refuses to do, both learned from T9b:

1. **It reports patient-units, not images, as the sample size** and resamples units in the
   bootstrap. 941 test images come from 27 infants; a subgroup of "400 images" may be 9
   babies.
2. **It will not print an AUC for a cell that cannot support one.** A subgroup with fewer
   than MIN_POS_UNITS positive infants is reported as UNMEASURABLE, not as a number. D3's
   0.6138 in T9b looked precise because one positive cluster makes the bootstrap unable to
   vary the positive side at all — exactly the artefact that must not be repeated per-cell.

Prevalence is reported beside every cell. ROP risk rises steeply as gestational age falls,
so subgroups differ in case-mix by construction; an AUC gap is NOT automatically a fairness
finding, and the prevalence column is what stops it being read as one.

    .venv/bin/python scripts/rop_subgroup_analysis.py
"""
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from models.common.config import load_config  # noqa: E402
from webapp.inference import DiseaseModel, _device  # noqa: E402

N_BOOT = 2000
SEED = 42
MIN_POS_UNITS = 3        # below this the positive side of a bootstrap cannot vary
CACHE = ROOT / "results" / "rop" / "devsplit_test_preds.npz"

PAT = re.compile(r"(?P<pid>\d+)_(?P<sex>[MF])_GA(?P<ga>\d+)_BW(?P<bw>\d+)_"
                 r"PA(?P<pa>\d+)_DG(?P<dg>\d)_PF(?P<pf>\d)_D(?P<dev>\d+)_S(?P<sess>\d+)")


def load_test():
    man = pd.read_csv(ROOT / "results" / "prepared" / "rop" / "manifest_devsplit.csv")
    t = man[man.split == "test"].reset_index(drop=True)
    meta = []
    for p in t.image_path:
        m = PAT.match(Path(p).name)
        if not m:
            raise ValueError(f"unparseable: {p}")
        g = m.groupdict()
        meta.append({"pid": g["pid"], "sex": g["sex"], "ga": int(g["ga"]),
                     "bw": int(g["bw"]), "pa": int(g["pa"]), "dev": g["dev"]})
    meta = pd.DataFrame(meta)
    # same infant-merging rule as the split builder, or the clusters are wrong
    sig2pids = defaultdict(set)
    for r in meta.itertuples():
        sig2pids[(r.sex, r.ga, r.bw)].add(r.pid)
    pid2unit = {}
    for pids in sig2pids.values():
        root = sorted(pids)[0]
        for p in pids:
            pid2unit[p] = root
    for c in meta.columns:
        t[c] = meta[c].values
    t["unit"] = meta["pid"].map(pid2unit).values
    return t


def predictions(t, weights):
    if CACHE.exists():
        z = np.load(CACHE, allow_pickle=True)
        if len(z["prob"]) == len(t) and list(z["image_path"]) == list(t.image_path):
            print(f"using cached predictions ({CACHE.name})")
            return z["prob"]
        print("cache does not match the current manifest — recomputing")
    cfg = load_config("configs/rop.yaml")
    dm = DiseaseModel("ROP", cfg, str(weights), _device())
    if not dm.loaded:
        raise SystemExit(f"could not load {weights}")
    from models.common.train_utils import predict_with_tta
    print(f"serving-path TTA inference on {len(t)} images…")
    out = []
    for i, r in enumerate(t.itertuples()):
        p = ROOT / r.image_path if not Path(r.image_path).is_absolute() else Path(r.image_path)
        out.append(predict_with_tta(dm.model, Image.open(p).convert("RGB"), dm.tta,
                                    dm.device, "classification").numpy()[1])
        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{len(t)}", flush=True)
    out = np.asarray(out)
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez(CACHE, prob=out, image_path=np.array(list(t.image_path), dtype=object))
    return out


def cell(sub, rng):
    """AUC + cluster-bootstrap CI for one subgroup, or an honest refusal."""
    units = sub.unit.nunique()
    pos_units = sub[sub.label == 1].unit.nunique()
    neg_units = sub[sub.label == 0].unit.nunique()
    base = {"images": int(len(sub)), "units": int(units), "positive_units": int(pos_units),
            "negative_units": int(neg_units), "prevalence": float(sub.label.mean())}
    if pos_units < MIN_POS_UNITS or neg_units < MIN_POS_UNITS:
        base.update({"auc": None, "ci": None,
                     "note": f"UNMEASURABLE — {pos_units} positive / {neg_units} negative "
                             f"infants; a bootstrap cannot vary a side with <"
                             f"{MIN_POS_UNITS} clusters"})
        return base
    y, p, u = sub.label.to_numpy(), sub.prob.to_numpy(), sub.unit.to_numpy()
    uq = np.array(sorted(set(u)))
    idx = {k: np.flatnonzero(u == k) for k in uq}
    vals, skipped = [], 0
    for _ in range(N_BOOT):
        pick = rng.choice(uq, size=len(uq), replace=True)
        ii = np.concatenate([idx[k] for k in pick])
        if len(np.unique(y[ii])) < 2:
            skipped += 1
            continue
        vals.append(roc_auc_score(y[ii], p[ii]))
    lo, hi = (np.percentile(vals, [2.5, 97.5]) if vals else (None, None))
    base.update({"auc": float(roc_auc_score(y, p)),
                 "ci": [float(lo), float(hi)] if vals else None,
                 "degenerate_draws": skipped})
    return base


def main():
    w = ROOT / "results" / "rop" / "weights_devsplit.pth"
    if not w.exists():
        print(f"ERROR: {w} missing — this analysis must use the device-stratified checkpoint,")
        print("or a subgroup gap could just be the device confound in another costume.")
        return 1

    t = load_test()
    t["prob"] = predictions(t, w)
    rng = np.random.default_rng(SEED)

    # Clinically meaningful cut-points, not data-driven ones: <28 wk is extremely preterm and
    # <1000 g extremely low birth weight — the strata ROP screening guidelines are built on.
    strata = {
        "sex": [("F", t.sex == "F"), ("M", t.sex == "M")],
        "gestational age": [("GA < 28 wk (extremely preterm)", t.ga < 28),
                            ("GA >= 28 wk", t.ga >= 28)],
        "birth weight": [("BW < 1000 g (ELBW)", t.bw < 1000),
                         ("BW >= 1000 g", t.bw >= 1000)],
        "postnatal age at capture": [("PA < 40 d", t.pa < 40), ("PA >= 40 d", t.pa >= 40)],
    }

    print("\n" + "=" * 92)
    print("T14 — ROP SUBGROUP PERFORMANCE (device-stratified split + checkpoint)")
    print("=" * 92)
    print(f"overall: {len(t)} images, {t.unit.nunique()} infants "
          f"({t[t.label==1].unit.nunique()} with ROP), prevalence {t.label.mean():.3f}")
    print(f"\n{'subgroup':<34}{'img':>6}{'inf':>5}{'+inf':>6}{'prev':>7}"
          f"{'AUC':>8}   95% CI")
    out = {"min_positive_units": MIN_POS_UNITS, "n_boot": N_BOOT, "strata": {}}
    for name, cuts in strata.items():
        print(f"\n{name.upper()}")
        out["strata"][name] = {}
        for lab, mask in cuts:
            c = cell(t[mask], rng)
            out["strata"][name][lab] = c
            if c["auc"] is None:
                print(f"  {lab:<32}{c['images']:>6}{c['units']:>5}{c['positive_units']:>6}"
                      f"{c['prevalence']:>7.3f}{'—':>8}   {c['note']}")
            else:
                print(f"  {lab:<32}{c['images']:>6}{c['units']:>5}{c['positive_units']:>6}"
                      f"{c['prevalence']:>7.3f}{c['auc']:>8.4f}   "
                      f"[{c['ci'][0]:.4f}, {c['ci'][1]:.4f}]")

    # A gap only means something if the two intervals do not overlap. Say so explicitly
    # rather than leaving a reader to eyeball two point estimates.
    print("\n" + "-" * 92)
    print("INTERPRETATION")
    any_sep = False
    for name, cells in out["strata"].items():
        ms = [(k, v) for k, v in cells.items() if v["auc"] is not None]
        if len(ms) < 2:
            print(f"  {name}: not assessable — {len(ms)}/2 cells have enough infants.")
            continue
        (ka, a), (kb, b) = ms[0], ms[1]
        sep = a["ci"][1] < b["ci"][0] or b["ci"][1] < a["ci"][0]
        any_sep |= sep
        print(f"  {name}: {a['auc']:.3f} vs {b['auc']:.3f} — intervals "
              f"{'DO NOT overlap (real difference)' if sep else 'overlap (no evidence of a difference)'}"
              f"; prevalence {a['prevalence']:.2f} vs {b['prevalence']:.2f}")
    if not any_sep:
        print("\n  No subgroup difference is separable at this sample size. With 27 infants")
        print("  that is the expected result, and it is a statement about the DATASET, not")
        print("  evidence that the model is equitable. Do not report it as a fairness pass.")

    dst = ROOT / "results" / "rop" / "subgroup_analysis.json"
    dst.write_text(json.dumps(out, indent=2))
    print(f"\n-> {dst}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
