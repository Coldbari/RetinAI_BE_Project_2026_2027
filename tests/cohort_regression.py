"""Phase 0A acceptance — re-run every cohort through the ROUTED stack (R6).

Compares against the pre-routing baseline measured in docs/IMPLEMENTATION_LOG.md. The exit
criterion for Phase 0A is ZERO cross-domain firings:
  * the ROP model must never fire on an adult cohort
  * the DR / Glaucoma models must never fire on an infant cohort

This is a script rather than a pytest case because it needs the large local cohorts (private
infant images, downloaded HRF) which are not committed. Run it manually and paste the table
into the log.

    .venv/bin/python tests/cohort_regression.py [--cohort-root DIR]
"""
import argparse
import glob
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from PIL import Image  # noqa: E402

from webapp.inference import Registry  # noqa: E402

# Which context an operator would legitimately declare for each cohort.
COHORTS = [
    ("INFANT no-ROP",  "infants/DG0/*.jpg",       "infant"),
    ("INFANT ROP+",    "infants/DG1/*.jpg",       "infant"),
    ("ADULT healthy",  "hrf_test/healthy/*.jpg",  "adult"),
    ("ADULT DR",       "hrf_test/dr/*.JPG",       "adult"),
    ("ADULT glaucoma", "hrf_test/glaucoma/*.jpg", "adult"),
    ("NON-FUNDUS",     "nonfundus/*.png",         "adult"),
]

# Pre-routing baseline (docs/IMPLEMENTATION_LOG.md), as % flagged.
BASELINE = {
    "INFANT no-ROP":  {"DR": 14, "ROP": 32, "Glaucoma": 59},
    "INFANT ROP+":    {"DR": 14, "ROP": 100, "Glaucoma": 43},
    "ADULT healthy":  {"DR": 0,  "ROP": 93, "Glaucoma": 0},
    "ADULT DR":       {"DR": 53, "ROP": 93, "Glaucoma": 13},
    "ADULT glaucoma": {"DR": 0,  "ROP": 87, "Glaucoma": 87},
    "NON-FUNDUS":     {"DR": 100, "ROP": 100, "Glaucoma": 100},
}


def flagged(finding):
    """A 'flag' = the model ran AND returned an abnormal grade (DR: referable, >=2)."""
    if not finding.get("available"):
        return False
    g = finding.get("grade", 0)
    return g >= 2 if finding["disease"] == "DR" else g >= 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort-root", default=os.environ.get("COHORT_ROOT", ""),
                    help="directory holding infants/, hrf_test/, nonfundus/")
    args = ap.parse_args()
    root = Path(args.cohort_root or ".")

    reg = Registry("webapp/registry.yaml")
    if not all(m.loaded for m in reg.models):
        print("WARNING: not all models loaded —", [m.disease for m in reg.models if not m.loaded])

    print(f"\n{'cohort':<16}{'ctx':<8}{'n':>4} | "
          f"{'DR':>13} | {'ROP':>13} | {'Glaucoma':>13}   (was → now)")
    print("-" * 92)

    violations, rows, missing = [], [], []
    for name, pattern, ctx in COHORTS:
        paths = sorted(glob.glob(str(root / pattern)))
        if not paths:
            missing.append(f"{name} ({root / pattern})")
            print(f"{name:<16}{ctx:<8}{'--':>4} | COHORT MISSING ({root / pattern})")
            continue
        counts = {m.disease: 0 for m in reg.models}
        ran = {m.disease: 0 for m in reg.models}
        for p in paths:
            res = reg.analyze(Image.open(p).convert("RGB"), ctx)
            for f in res["findings"]:
                if f.get("available"):
                    ran[f["disease"]] += 1
                if flagged(f):
                    counts[f["disease"]] += 1
        n = len(paths)
        cells = []
        for d in ("DR", "ROP", "Glaucoma"):
            pct = round(100 * counts[d] / n)
            cells.append(f"{BASELINE[name][d]:>3}%→{pct:>3}%")
            # cross-domain check: a model that should not have run at all
            if ran[d] and ((ctx == "adult" and d == "ROP") or (ctx == "infant" and d in ("DR", "Glaucoma"))):
                violations.append(f"{name}: {d} RAN under context '{ctx}' ({ran[d]}/{n})")
        rows.append((name, n, counts, ran))
        print(f"{name:<16}{ctx:<8}{n:>4} | {cells[0]:>13} | {cells[1]:>13} | {cells[2]:>13}")

    print("\n" + "=" * 92)
    # A run that evaluated nothing must NEVER report PASS. An empty pass is indistinguishable
    # from a real one in a log, and silently certifies a system that was never exercised.
    if missing:
        print(f"ERROR — {len(missing)}/{len(COHORTS)} cohorts missing; this run proves nothing.")
        for m in missing:
            print("  -", m)
        print("\nBuild them first:  .venv/bin/python tests/build_cohorts.py")
        return 2
    if violations:
        print("FAIL — cross-domain firings remain:")
        for v in violations:
            print("  -", v)
        return 1
    total = sum(r[1] for r in rows)
    print(f"PASS — zero cross-domain firings across {total} images in {len(rows)} cohorts.")
    print("No model ran outside its declared population.")
    nf = next((r for r in rows if r[0] == "NON-FUNDUS"), None)
    if nf and any(nf[2].values()):
        print("\nWARNING: NON-FUNDUS input still produces findings — the Phase 0B gradability")
        print("gate is not active or has regressed.")
        return 1
    print("\nNON-FUNDUS is now rejected by the Phase 0B gradability gate before any model runs.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
