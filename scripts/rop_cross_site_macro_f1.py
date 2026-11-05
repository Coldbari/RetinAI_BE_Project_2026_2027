"""Cross-site macro-F1: the 0.545 is an arithmetic artifact, recomputed honestly.

    .venv/bin/python scripts/rop_cross_site_macro_f1.py

The held-out-hospital fold (fold_site) contains only 4 of the 6 classes — stage 4/5 and
AP-ROP simply do not occur at that site (740 images: 234 normal / 89 s1 / 163 s2 / 254 s3).
A macro average over all 6 labels therefore adds two guaranteed zeros to the numerator and
two to the denominator: a model that classified every present image PERFECTLY would score
0.667. The committed 0.528/0.545 numbers say more about the denominator than the model.

This recomputes macro-F1 over the classes that exist at the held-out site, from the same
committed artifacts the originals came from (fold_site.npz for both structured seeds; the
per-class table in the flat 5-fold JSON, which predates val-prediction saving). Both
numbers are kept: the 6-label figure is what leaderboards would show, the present-class
figure is what the model actually did.

Output: results/rop/cross_site_macro_f1.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "results" / "rop" / "staging_runs"
OUT = ROOT / "results" / "rop" / "cross_site_macro_f1.json"


def from_npz(path: Path):
    d = np.load(path)
    y, p = d["y_true"], d["y_pred"]
    present = sorted(set(y.tolist()))
    return {
        "n_images": int(len(y)),
        "classes_present": present,
        "macro_f1_6label": round(float(f1_score(y, p, average="macro",
                                                labels=list(range(6)), zero_division=0)), 4),
        "macro_f1_present": round(float(f1_score(y, p, average="macro",
                                                 labels=present, zero_division=0)), 4),
        "per_class_f1": [round(float(v), 4) for v in
                         f1_score(y, p, average=None, labels=list(range(6)), zero_division=0)],
        "source": str(path.relative_to(ROOT)),
    }


def main():
    arms = {
        "structured_s42": from_npz(RUNS / "structured_5fold" / "fold_site.npz"),
        "structured_s1337": from_npz(RUNS / "structured_5fold_s1337" / "fold_site.npz"),
    }
    # The flat run predates val-prediction saving; its per-class table is the artifact.
    flat = json.loads((RUNS / "staging_effnetv2s_s42_5fold.json").read_text())
    pc = flat["folds"]["fold_site"]["per_class"]
    present = [v["f1"] for v in pc.values() if v["support"] > 0]
    arms["flat_s42"] = {
        "n_images": int(sum(v["support"] for v in pc.values())),
        "classes_present": [k for k, v in pc.items() if v["support"] > 0],
        "macro_f1_6label": round(float(flat["folds"]["fold_site"]["macro_f1"]), 4),
        "macro_f1_present": round(float(np.mean(present)), 4),
        "per_class_f1": [round(float(v["f1"]), 4) for v in pc.values()],
        "source": "results/rop/staging_runs/staging_effnetv2s_s42_5fold.json (per_class table)",
    }

    out = {
        "what": "Held-out-hospital macro-F1 with the denominator stated. The site has no "
                "stage 4/5 or AP-ROP, so a 6-label macro adds two structural zeros — a "
                "perfect classifier would score 0.667. macro_f1_present averages only the "
                "classes that occur at the site and is the number that describes the model.",
        "perfect_classifier_would_score_6label": round(4 / 6, 4),
        "arms": arms,
        "reading": "Over the four classes the held-out hospital actually has, every arm "
                   "clears 0.79 (flat 0.792, structured 0.818/0.820 across seeds). The "
                   "previously quoted 0.528/0.545 measured the class roster of the site, "
                   "not the classifier. QWK (0.930/0.946), computed only on present "
                   "classes by construction, never had this problem — which is why the "
                   "two metrics appeared to disagree.",
    }
    OUT.write_text(json.dumps(out, indent=2) + "\n")
    for k, v in arms.items():
        print(f"  {k:>16}: 6-label {v['macro_f1_6label']:.3f} -> present-class "
              f"{v['macro_f1_present']:.3f}  ({len(v['classes_present'])} classes, "
              f"{v['n_images']} images)")
    print(f"-> {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
