"""Phase 2 (C6) — derive the abstention band on VAL, from serving-path predictions.

The band is the region around the operating threshold where the model is not reliable enough
to answer. It is chosen on VAL (never test) as the widest band whose retained accuracy meets a
target, subject to a cap on how much may be abstained.

Reported for every candidate band: how much is abstained, accuracy INSIDE the band (what we
would have got wrong by answering), and accuracy OUTSIDE it (what the operator actually sees).
A band only earns its place if accuracy inside it is materially worse than outside — otherwise
it is refusing to answer cases it would have got right.

    .venv/bin/python scripts/fit_abstention_band.py --disease rop
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from models.common.config import load_config  # noqa: E402

MAX_ABSTAIN = 0.15          # never refuse more than 15% of cases


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--disease", required=True)
    ap.add_argument("--split", default="val")
    args = ap.parse_args()

    src = ROOT / "results" / args.disease / f"tta_{args.split}_preds.npz"
    if not src.exists():
        print(f"ERROR: {src} missing — run scripts/fit_serving_calibration.py first.")
        return 1
    d = np.load(src, allow_pickle=True)
    P, Y = d["prob"], d["y_true"].astype(int)
    cfg = load_config(f"configs/{args.disease}.yaml")
    n_cls = P.shape[1]

    if n_cls == 2:
        thr = float(cfg.eval.get("screen_threshold", 0.5))
        score = P[:, 1]
        decision = (score >= thr).astype(int)
    else:
        ref = int(cfg.eval.get("referable_grade", 2))
        thr = 0.5
        score = P[:, ref:].sum(axis=1)
        decision = (score >= thr).astype(int)
        Y = (Y >= ref).astype(int)

    correct = (decision == Y)
    base_acc = correct.mean()
    print(f"{args.disease.upper()} {args.split}: n={len(Y)}, decision threshold {thr}")
    print(f"baseline accuracy (answer everything): {base_acc:.4f}\n")
    print(f"{'half-width':>11}{'band':>22}{'abstained':>11}{'acc inside':>12}"
          f"{'acc outside':>13}{'gain':>8}")
    print("-" * 78)

    best, rows = None, []
    for hw in np.arange(0.02, 0.31, 0.02):
        lo, hi = thr - hw, thr + hw
        inside = (score >= lo) & (score <= hi)
        frac = inside.mean()
        if frac > MAX_ABSTAIN or inside.sum() < 10 or (~inside).sum() < 10:
            continue
        acc_in = correct[inside].mean()
        acc_out = correct[~inside].mean()
        rows.append({"half_width": float(hw), "lo": float(lo), "hi": float(hi),
                     "abstain_frac": float(frac), "acc_inside": float(acc_in),
                     "acc_outside": float(acc_out), "gain": float(acc_out - base_acc)})
        print(f"{hw:>11.2f}{f'[{lo:.3f}, {hi:.3f}]':>22}{frac:>10.1%}"
              f"{acc_in:>12.4f}{acc_out:>13.4f}{acc_out-base_acc:>+8.4f}")
        if best is None or (acc_out - base_acc) > best["gain"]:
            best = rows[-1]

    print()
    if not best:
        print("No band satisfies the constraints — abstention NOT enabled.")
        out = {"disease": args.disease, "enabled": False,
               "reason": f"no band abstains <={MAX_ABSTAIN:.0%} with enough cases either side"}
    elif best["gain"] < 0.01:
        print(f"Best band improves retained accuracy by only {best['gain']:+.4f} — "
              f"NOT worth refusing {best['abstain_frac']:.1%} of cases.")
        print("Abstention NOT enabled. A band that refuses cases it would have got right is "
              "pure cost.")
        out = {"disease": args.disease, "enabled": False, "candidate": best,
               "reason": f"best retained-accuracy gain {best['gain']:+.4f} below 0.01 threshold"}
    else:
        print(f"SELECTED band [{best['lo']:.3f}, {best['hi']:.3f}] — abstains "
              f"{best['abstain_frac']:.1%}, retained accuracy {best['acc_outside']:.4f} "
              f"({best['gain']:+.4f} vs answering everything).")
        print(f"Accuracy inside the band is {best['acc_inside']:.4f} — these are the cases "
              f"worth sending to a human.")
        out = {"disease": args.disease, "enabled": True, **best,
               "threshold": thr, "split": args.split, "n": int(len(Y))}

    out["all_candidates"] = rows
    dst = ROOT / "results" / args.disease / "abstention_band.json"
    dst.write_text(json.dumps(out, indent=2))
    print(f"\n-> {dst}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
