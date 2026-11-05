"""Phase 3 (T8) — ROP device-confound audit.

The question: does the model read the DISEASE, or the DEVICE?

If the capture device correlates with diagnosis in the private database, a model can score
well by recognising the camera rather than the retina — and no amount of patient-level
splitting would catch it, because device and patient are confounded together.

ROP filenames encode  patient_sex_GA_BW_PA_DG_PF_D#_S##_index, so `D#` is recoverable
without any extra metadata. This measures:
  1. prevalence of ROP per device (is the confound present at all?)
  2. how device is distributed across train/val/test
  3. model AUC computed WITHIN each device (a real signal survives stratification;
     a device shortcut collapses)

    .venv/bin/python scripts/rop_device_audit.py
"""
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from models.evaluation.statistical_tests import cluster_bootstrap_ci  # noqa: E402


def parse(name):
    m = re.match(r"(?P<pid>\d+)_(?P<sex>[MF])_GA(?P<ga>\d+)_BW(?P<bw>\d+)_"
                 r"PA(?P<pa>\d+)_DG(?P<dg>\d)_PF(?P<pf>\d)_D(?P<dev>\d+)_S(?P<sess>\d+)",
                 Path(name).name)
    return m.groupdict() if m else None


def main():
    L = []
    out_extra = {}

    def say(s=""):
        print(s); L.append(s)

    man = pd.read_csv(ROOT / "results" / "prepared" / "rop" / "manifest.csv")
    rec = []
    for _, r in man.iterrows():
        g = parse(r["image_path"])
        if g:
            rec.append({**g, "split": r["split"], "label": int(r["label"]),
                        "path": r["image_path"]})
    df = pd.DataFrame(rec)
    say("=" * 78)
    say("T8 — ROP DEVICE-CONFOUND AUDIT")
    say("=" * 78)
    say(f"parsed {len(df)}/{len(man)} filenames")
    say("")

    # 1. is the confound present?
    say("1) ROP PREVALENCE BY DEVICE (images, and patients)")
    say(f"{'device':<9}{'images':>8}{'ROP+ img':>10}{'prev':>8}"
        f"{'patients':>10}{'ROP+ pat':>10}{'prev':>8}")
    say("-" * 78)
    dev_rows = {}
    for dev, g in df.groupby("dev"):
        pats = g.groupby("pid")["label"].max()
        row = {"images": len(g), "img_pos": int(g.label.sum()),
               "img_prev": float(g.label.mean()), "patients": int(len(pats)),
               "pat_pos": int(pats.sum()), "pat_prev": float(pats.mean())}
        dev_rows[dev] = row
        say(f"D{dev:<8}{row['images']:>8}{row['img_pos']:>10}{row['img_prev']:>8.1%}"
            f"{row['patients']:>10}{row['pat_pos']:>10}{row['pat_prev']:>8.1%}")
    prevs = [r["pat_prev"] for r in dev_rows.values()]
    spread = max(prevs) - min(prevs)
    say("")
    say(f"patient-level prevalence spread across devices: {spread:.1%}")
    if spread > 0.30:
        say("  -> STRONG confound: device carries substantial information about the label.")
    elif spread > 0.15:
        say("  -> MODERATE confound: device is informative about the label.")
    else:
        say("  -> weak confound.")

    # 2. device distribution across splits
    say("")
    say("2) DEVICE DISTRIBUTION ACROSS SPLITS (images)")
    ct = pd.crosstab(df["split"], df["dev"], normalize="index")
    say(ct.round(3).to_string())
    say("")
    say("If a device is concentrated in one split, test performance partly measures")
    say("that device rather than generalisation.")

    # 3. does the signal survive within-device stratification?
    say("")
    say("3) WITHIN-DEVICE AUC ON THE TEST SPLIT (deployed ResNet50)")
    p = ROOT / "results" / "rop" / "preds_weights_test.npz"
    if not p.exists():
        say("  (predictions missing — run evaluate.py --dump-preds first)")
    else:
        d = np.load(p, allow_pickle=True)
        pred = pd.DataFrame({"path": [str(x) for x in d["image_path"]],
                             "y": d["y_true"].astype(int), "prob": d["y_prob"][:, 1]})
        pred["dev"] = [parse(x)["dev"] if parse(x) else None for x in pred["path"]]
        pred["pid"] = [parse(x)["pid"] if parse(x) else None for x in pred["path"]]
        overall = roc_auc_score(pred.y, pred.prob)
        say(f"  overall test AUC {overall:.4f}")
        say(f"{'device':<9}{'images':>8}{'ROP+':>7}{'AUC':>9}   note")
        say("-" * 78)
        within = {}
        for dev, g in pred.groupby("dev"):
            if g.y.nunique() < 2:
                say(f"D{dev:<8}{len(g):>8}{int(g.y.sum()):>7}{'n/a':>9}   single-class "
                    f"({'all ROP+' if g.y.iloc[0] == 1 else 'all ROP-'})")
                continue
            a = roc_auc_score(g.y, g.prob)
            within[dev] = a
            say(f"D{dev:<8}{len(g):>8}{int(g.y.sum()):>7}{a:>9.4f}   "
                f"{a-overall:+.4f} vs overall")
        # How many devices even ADMIT a discrimination measurement? If the positive class
        # lives on one device, the overall AUC is partly a device separation and the
        # magnitude of the drop understates the problem.
        measurable = len(within)
        single = [d for d, g in pred.groupby("dev") if g.y.nunique() < 2]
        pos_dev = sorted(pred.loc[pred.y == 1, "dev"].unique())
        say("")
        say(f"  devices in test: {pred.dev.nunique()}   measurable (both classes present): "
            f"{measurable}   ROP+ present on: {['D'+d for d in pos_dev]}")
        if measurable <= 1:
            say("")
            say("  -> CRITICAL STRUCTURAL FINDING: every ROP-positive test image comes from a")
            say("     SINGLE device. Discrimination can only be measured inside that device;")
            say("     for the others there is nothing to discriminate. The overall AUC is")
            say("     therefore inflated by an easy device-vs-device separation that has")
            say("     nothing to do with disease.")
        if within:
            wmean = float(np.mean(list(within.values())))
            say("")
            say(f"  overall AUC (all devices pooled) {overall:.4f}")
            say(f"  within-device AUC               {wmean:.4f}   ({wmean-overall:+.4f})")
            say("")
            if measurable <= 1:
                say(f"  -> The defensible device-controlled ROP AUC is {wmean:.4f}, not "
                    f"{overall:.4f}.")
            elif wmean < overall - 0.05:
                say("  -> WARNING: performance drops materially once device is held constant.")
            else:
                say("  -> the signal largely survives stratification by device.")
            out_extra["within_device_auc"] = wmean
            out_extra["overall_auc"] = float(overall)
            out_extra["measurable_devices"] = measurable
            out_extra["positive_devices"] = pos_dev

    out = {"device_rows": dev_rows, "prevalence_spread": spread, **out_extra,
           "split_by_device": ct.round(4).to_dict()}
    (ROOT / "results" / "rop_device_audit.json").write_text(json.dumps(out, indent=2, default=float))
    (ROOT / "results" / "rop_device_audit.md").write_text(
        "# ROP device-confound audit\n\n```\n" + "\n".join(L) + "\n```\n")
    print("\n-> results/rop_device_audit.json, results/rop_device_audit.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
