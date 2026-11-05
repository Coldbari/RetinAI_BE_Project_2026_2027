"""T9b — cluster-bootstrap confidence intervals for the device-stratified ROP result.

The T9 kernel reported point estimates only:

    D1 0.8171   D2 0.6451   D3 0.6312   device-controlled mean 0.6978

Those three numbers are NOT equally trustworthy, and the mean hides why. From the split
build, the ROP-POSITIVE side of each device's test set is:

    D1  6 patient-units      D2  2 patient-units      D3  1 patient-unit

So D3's AUC is one infant against five; D2's is two against seven. An AUC computed over
121 images that come from 6 babies has the uncertainty of n=6, not n=121 — images from the
same eye at the same session are not independent draws. Quoting 0.6978 without an interval
would repeat exactly the error this whole effort exists to remove (see S4: the original
image-level ROP interval was 3.8x too narrow).

Resampling is therefore over PATIENT-UNITS, not images, using the same unit definition as
scripts/build_device_stratified_split.py (patient IDs sharing sex+GA+BW are one infant).

    .venv/bin/python scripts/rop_devsplit_ci.py --weights results/rop/weights_devsplit.pth
"""
import argparse
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


def parse(name):
    return re.match(r"(?P<pid>\d+)_(?P<sex>[MF])_GA(?P<ga>\d+)_BW(?P<bw>\d+)_"
                    r"PA(?P<pa>\d+)_DG(?P<dg>\d)_PF(?P<pf>\d)_D(?P<dev>\d+)_S(?P<sess>\d+)",
                    name)


def attach_units(df):
    """Same infant-merging rule as the split builder — must match, or the clusters lie."""
    meta = []
    for p in df.image_path:
        m = parse(Path(p).name)
        if not m:
            raise ValueError(f"unparseable filename: {p}")
        meta.append(m.groupdict())
    meta = pd.DataFrame(meta)
    sig2pids = defaultdict(set)
    for r in meta.itertuples():
        sig2pids[(r.sex, r.ga, r.bw)].add(r.pid)
    pid2unit = {}
    for pids in sig2pids.values():
        root = sorted(pids)[0]
        for p in pids:
            pid2unit[p] = root
    df = df.copy()
    df["dev"] = meta["dev"].values
    df["unit"] = meta["pid"].map(pid2unit).values
    return df


def cluster_boot_auc(y, p, unit, rng, n_boot=N_BOOT):
    """Resample patient-units with replacement; degenerate draws are counted, not silently
    dropped — a high skip rate is itself the finding."""
    units = np.array(sorted(set(unit)))
    idx_by_unit = {u: np.flatnonzero(unit == u) for u in units}
    vals, skipped = [], 0
    for _ in range(n_boot):
        pick = rng.choice(units, size=len(units), replace=True)
        idx = np.concatenate([idx_by_unit[u] for u in pick])
        if len(np.unique(y[idx])) < 2:
            skipped += 1
            continue
        vals.append(roc_auc_score(y[idx], p[idx]))
    if not vals:
        return None, None, None, skipped
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(np.mean(vals)), float(lo), float(hi), skipped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="results/rop/weights_devsplit.pth")
    ap.add_argument("--manifest", default="results/prepared/rop/manifest_devsplit.csv")
    ap.add_argument("--config", default="configs/rop.yaml")
    args = ap.parse_args()

    w = ROOT / args.weights
    if not w.exists():
        print(f"ERROR: {w} not found. Copy the T9 kernel's rop_devsplit_weights.pth there.")
        print("Scoring with the DEPLOYED weights would silently answer a different question.")
        return 1

    man = pd.read_csv(ROOT / args.manifest)
    test = attach_units(man[man.split == "test"].reset_index(drop=True))

    print("test composition (patient-units, not images — this is the real sample size)")
    print(f"{'device':<8}{'images':>8}{'ROP+ img':>10}{'units':>7}{'+units':>8}{'-units':>8}")
    for dv, g in test.groupby("dev"):
        pu = g[g.label == 1].unit.nunique()
        nu = g[g.label == 0].unit.nunique()
        print(f"D{dv:<7}{len(g):>8}{int(g.label.sum()):>10}{g.unit.nunique():>7}{pu:>8}{nu:>8}")

    cfg = load_config(args.config)
    dm = DiseaseModel("ROP", cfg, str(w), _device())
    if not dm.loaded:
        print(f"ERROR: could not load {w}")
        return 1

    from models.common.train_utils import predict_with_tta
    print(f"\nserving-path TTA inference on {len(test)} test images…")
    probs = []
    for i, r in enumerate(test.itertuples()):
        img = Image.open(ROOT / r.image_path if not Path(r.image_path).is_absolute()
                         else r.image_path).convert("RGB")
        probs.append(predict_with_tta(dm.model, img, dm.tta, dm.device,
                                      cfg.model.get("head", "classification")).numpy()[1])
        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{len(test)}", flush=True)
    test["prob"] = probs

    rng = np.random.default_rng(SEED)
    y_all = test.label.to_numpy()
    p_all = test.prob.to_numpy()
    u_all = test.unit.to_numpy()

    print("\n" + "=" * 74)
    print("T9b — DEVICE-STRATIFIED ROP, CLUSTER-BOOTSTRAP INTERVALS (by patient-unit)")
    print("=" * 74)

    out = {"n_boot": N_BOOT, "seed": SEED, "per_device": {}}
    per_dev_point = {}
    for dv, g in test.groupby("dev"):
        pt = roc_auc_score(g.label, g.prob)
        _, lo, hi, sk = cluster_boot_auc(g.label.to_numpy(), g.prob.to_numpy(),
                                         g.unit.to_numpy(), rng)
        per_dev_point[dv] = pt
        pu = int(g[g.label == 1].unit.nunique())
        out["per_device"][dv] = {"auc": float(pt), "ci": [lo, hi], "positive_units": pu,
                                 "degenerate_draws": sk}
        width = hi - lo if lo is not None else float("nan")
        print(f"D{dv}  AUC {pt:.4f}   95% CI [{lo:.4f}, {hi:.4f}]   width {width:.3f}   "
              f"({pu} positive infant{'s' if pu != 1 else ''})")

    pooled = roc_auc_score(y_all, p_all)
    _, plo, phi, psk = cluster_boot_auc(y_all, p_all, u_all, rng)
    out["pooled"] = {"auc": float(pooled), "ci": [plo, phi], "degenerate_draws": psk}
    print(f"\npooled AUC          {pooled:.4f}   95% CI [{plo:.4f}, {phi:.4f}]")

    # Device-controlled mean, resampling units INSIDE each device independently.
    devs = sorted(test.dev.unique())
    by_dev = {dv: test[test.dev == dv] for dv in devs}
    means, skipped = [], 0
    for _ in range(N_BOOT):
        vals, bad = [], False
        for dv in devs:
            g = by_dev[dv]
            units = np.array(sorted(set(g.unit)))
            pick = rng.choice(units, size=len(units), replace=True)
            sub = pd.concat([g[g.unit == u] for u in pick])
            if sub.label.nunique() < 2:
                bad = True
                break
            vals.append(roc_auc_score(sub.label, sub.prob))
        if bad:
            skipped += 1
            continue
        means.append(float(np.mean(vals)))
    ctrl_pt = float(np.mean(list(per_dev_point.values())))
    clo, chi = (np.percentile(means, [2.5, 97.5]) if means else (None, None))
    out["device_controlled"] = {"auc": ctrl_pt, "ci": [float(clo), float(chi)],
                                "degenerate_draws": skipped}
    print(f"device-controlled   {ctrl_pt:.4f}   95% CI [{clo:.4f}, {chi:.4f}]"
          f"   ({skipped}/{N_BOOT} draws degenerate)")
    print(f"\ngap pooled - device-controlled: {pooled - ctrl_pt:+.4f}  "
          f"— the part of the pooled number that is device separation, not disease.")

    dst = ROOT / "results" / "rop" / "devsplit_ci.json"
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(out, indent=2))
    print(f"\n-> {dst}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
