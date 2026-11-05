"""Paired significance tests between the fold-0 arms — run at last, not just implemented.

    .venv/bin/python scripts/rop_significance_tests.py

The audit's finding: McNemar and DeLong live in models/evaluation/statistical_tests.py,
are cited in two documents as available, and had never been executed against any pair of
staging arms. This runs them on every committed fold-0 prediction file, all of which score
the SAME 1,691 validation images — the paired setting both tests require.

  McNemar   : paired test on 6-class argmax correctness (b = A right/B wrong, c = the
              reverse). Answers "do these two arms disagree more than chance?"
  DeLong    : paired test on AP-ROP one-vs-rest ROC-AUC. Answers "does either arm rank
              AP-ROP better?" — the right instrument now that we know image-level recall
              at 0.5 is a threshold artifact (image AUC 0.917).

Caveats stated with the numbers, not hidden: fold 0 only (the one fold where every arm has
committed predictions); images are not independent (one infant supplies 470 of the 476
AP-ROP frames), so DeLong's independence assumption is violated for AP-ROP — we report it
alongside a patient-clustered bootstrap of the AUC difference, which respects clustering
and is the number to trust when they disagree.

Output: results/rop/significance_fold0.json
"""
from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from models.evaluation.statistical_tests import delong_test, mcnemar_test  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "results" / "rop" / "staging_runs"
OUT = ROOT / "results" / "rop" / "significance_fold0.json"
AROP = 5

ARMS = {
    "full": RUNS / "structured_5fold" / "fold_0.npz",
    "full_s1337": RUNS / "structured_5fold_s1337" / "fold_0.npz",
    "flat_natural": RUNS / "ablations" / "natural_fold0.npz",
    "wsite0": RUNS / "ablations" / "wsite0_fold0.npz",
    "nobranch": RUNS / "ablations" / "nobranch_fold0.npz",
    "noordinal": RUNS / "ablations" / "noordinal_fold0.npz",
    "sampler": RUNS / "ablations" / "sampler_fold0.npz",
}
# The pairs the paper actually argues about; everything else is noise-vs-noise.
PAIRS = [
    ("full", "flat_natural"),     # the equivalence claim, sampling-matched
    ("full", "nobranch"),         # does the branch matter?
    ("full", "noordinal"),        # does CORN matter?
    ("full", "wsite0"),           # does the adversary matter?
    ("full", "sampler"),          # the sampler catastrophe
    ("full", "full_s1337"),       # seed-to-seed: the noise yardstick
]


def cluster_boot_auc_diff(y, sa, sb, groups, n=2000, seed=42):
    """Patient-clustered bootstrap CI for AUC_a - AUC_b (paired, resampling infants)."""
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups)
    idx_by_g = {g: np.where(groups == g)[0] for g in uniq}
    diffs = []
    for _ in range(n):
        take = np.concatenate([idx_by_g[g] for g in rng.choice(uniq, len(uniq), True)])
        yt = y[take]
        if yt.min() == yt.max():        # a replicate without both classes has no AUC
            continue
        diffs.append(roc_auc_score(yt, sa[take]) - roc_auc_score(yt, sb[take]))
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return round(float(lo), 4), round(float(hi), 4), len(diffs)


def main():
    val = pd.read_csv(ROOT / "results/prepared/rop_staging/fold_0.csv").query(
        "split == 'val'").reset_index(drop=True)
    y = val.label.values
    groups = val.patient_id.values
    data = {}
    for name, p in ARMS.items():
        d = np.load(p)
        key = "prob6" if "prob6" in d else "prob"
        assert (d["y_true"] == y).all(), f"{name}: misaligned with fold_0 val"
        data[name] = {"pred": d["y_pred"], "p_arop": d[key][:, AROP]}

    out = {"context": "fold 0, 1691 images, paired on identical validation rows. "
                      "DeLong assumes independent images and one infant supplies 470/476 "
                      "AP-ROP frames — where DeLong and the patient-clustered bootstrap "
                      "disagree, trust the bootstrap.",
           "pairs": {}}
    is_arop = (y == AROP).astype(int)
    from sklearn.metrics import roc_auc_score
    for a, b in PAIRS:
        A, B = data[a], data[b]
        mc = mcnemar_test(y, A["pred"], B["pred"])
        dl = delong_test(is_arop, A["p_arop"], B["p_arop"])
        lo, hi, nrep = cluster_boot_auc_diff(is_arop, A["p_arop"], B["p_arop"], groups)
        row = {
            "mcnemar": {k: (round(float(v), 6) if isinstance(v, (int, float)) else v)
                        for k, v in mc.items()},
            "arop_auc": {a: round(float(roc_auc_score(is_arop, A["p_arop"])), 4),
                         b: round(float(roc_auc_score(is_arop, B["p_arop"])), 4)},
            "delong_p": round(float(dl["p_value"]), 6),
            "auc_diff_ci95_clustered": [lo, hi],
            "clustered_replicates": nrep,
        }
        out["pairs"][f"{a}_vs_{b}"] = row
        sig = "SIG" if not (lo <= 0 <= hi) else "ns "
        print(f"  {a:>12} vs {b:<12} McNemar p={row['mcnemar'].get('p_value', float('nan')):.4f}  "
              f"AUC {row['arop_auc'][a]:.3f} vs {row['arop_auc'][b]:.3f}  "
              f"clustered dAUC CI [{lo:+.3f},{hi:+.3f}] {sig}")

    OUT.write_text(json.dumps(out, indent=2) + "\n")
    print(f"-> {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
