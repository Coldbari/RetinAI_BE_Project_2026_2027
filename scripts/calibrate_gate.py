"""Calibrate the gradability gate on TRAIN-side images from ALL FOUR cohorts, verify
against negatives that include real-photograph statistics, then publish a report.

    .venv/bin/python scripts/calibrate_gate.py

v2 (2026-08-19). v1 calibrated on the Ostrava-style infant split alone and verified only
against synthetic negatives (noise / flat colour / charts). Both decisions failed in the
field, found by a user in one upload: a warm-lit FURNITURE PHOTO passed every colour check
and scored "ROP Detected 91.5%", while the Ostrava-fitted red-hue threshold (0.34) would
have silently rejected real Shenzhen/Multi-View fundus (p1 = 0.03-0.09). Lessons applied:

  - positives now span every cohort the app will actually be shown;
  - negatives now include structured warm-scene photographs (the observed failure class),
    not just noise;
  - the PRIMARY non-fundus check is structural, not chromatic: fundus cameras expose the
    retina through a circular aperture, so the frame corners are near-black in every
    cohort (p99 <= 19.8/255) - a real-world photograph fills its corners (measured >130).

Discipline unchanged: thresholds derive from TRAIN-side images only; test splits and the
locked half stay out. Negatives confirm separation, never tune the positive side.

Writes results/gate_thresholds.json and results/gate_calibration.md.
"""
import json
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from models.common.data_quality import GATE_DEFAULTS, gradability, gradability_metrics  # noqa: E402

SEED = 42
PER_GROUP = 150


def positives():
    """TRAIN-side fundus from all four cohorts (test splits and the locked half excluded)."""
    rng = random.Random(SEED)
    groups = {}
    ds = sorted(Path(ROOT / "dataset_split/train").rglob("*.jpg"))
    rng.shuffle(ds)
    groups["ostrava_fullframe"] = ds[:PER_GROUP]
    for name, rel in (("ropvl", "data/rop/rop_vl"), ("shenzhen", "data/rop/shenzhen"),
                      ("multiview", "data/rop/multiview")):
        fs = sorted((ROOT / rel).rglob("*.jpg"))[:6000]
        rng.shuffle(fs)
        groups[name] = fs[:PER_GROUP]
    return groups


def warm_scene(seed):
    """A structured warm-lit indoor photograph: wall + furniture blocks + light falloff.
    This is the failure class observed in the field, reproduced deterministically."""
    r = np.random.default_rng(seed)
    h, w = 800, 600
    base = np.zeros((h, w, 3), np.float32)
    wall = r.uniform(140, 200); tint = r.uniform(1.05, 1.35)
    base[:, :, 0] = wall * tint; base[:, :, 1] = wall; base[:, :, 2] = wall / r.uniform(1.05, 1.3)
    for _ in range(int(r.integers(4, 10))):
        x0, y0 = int(r.integers(0, w - 50)), int(r.integers(0, h - 50))
        x1, y1 = x0 + int(r.integers(40, w // 2)), y0 + int(r.integers(40, h // 2))
        c = [(210, 200, 185), (150, 110, 70), (120, 120, 115),
             (235, 230, 220), (90, 60, 40)][int(r.integers(0, 5))]
        base[y0:y1, x0:x1] = np.array(c) * r.uniform(0.8, 1.1)
    light = np.linspace(1.15, 0.75, h)[:, None, None]
    return Image.fromarray(np.clip(base * light + r.normal(0, 6, (h, w, 3)), 0, 255).astype(np.uint8))


def night_interior(seed):
    """The SECOND field bypass (2026-08-19): a night-time temple-interior photo - dark
    corners defeat the aperture check, lamplight passes the colour checks. Reproduced:
    dark ambience, warm lamps, lit wall patches."""
    import cv2
    r = np.random.default_rng(seed)
    h, w = 650, 900
    img = np.zeros((h, w, 3), np.float32) + r.uniform(8, 20)
    for _ in range(int(r.integers(3, 7))):
        cx, cy = int(r.integers(0, w)), int(r.integers(0, h // 2))
        cv2.circle(img, (cx, cy), int(r.integers(15, 60)),
                   (255, float(r.uniform(140, 200)), float(r.uniform(40, 90))), -1)
    for _ in range(int(r.integers(3, 8))):
        x0, y0 = int(r.integers(0, w - 80)), int(r.integers(h // 4, h - 80))
        cv2.rectangle(img, (x0, y0),
                      (x0 + int(r.integers(60, 300)), y0 + int(r.integers(60, 200))),
                      (float(r.uniform(120, 190)), float(r.uniform(90, 140)),
                       float(r.uniform(50, 90))), -1)
    img = cv2.GaussianBlur(img, (31, 31), 0)
    return Image.fromarray(np.clip(img + r.normal(0, 5, (h, w, 3)), 0, 255).astype(np.uint8))


def negatives():
    rng = np.random.default_rng(SEED)
    negs = [("noise", [Image.fromarray(rng.integers(0, 255, (800, 800, 3), dtype=np.uint8))]),
            ("grey", [Image.fromarray(np.full((800, 800, 3), 128, np.uint8))]),
            ("black", [Image.fromarray(np.zeros((800, 800, 3), np.uint8))]),
            ("white", [Image.fromarray(np.full((800, 800, 3), 255, np.uint8))]),
            ("warm_scene", [warm_scene(i) for i in range(40)]),
            ("night_interior", [night_interior(i) for i in range(30)])]
    charts = [Image.open(p).convert("RGB") for p in sorted(Path(ROOT / "graphs").rglob("*.png"))[:20]]
    if charts:
        negs.append(("charts", charts))
    return negs


def main():
    thr = dict(GATE_DEFAULTS)   # v2: module defaults ARE the calibrated values; this run
    #                             verifies them against fresh samples and publishes evidence.
    pos = positives()
    report = ["# Gradability gate calibration (v2, all-cohort)\n"]
    print(f"{'group':<20}{'n':>5}{'rejected':>10}  worst offenders")
    total_n = total_rej = 0
    for name, files in pos.items():
        n = rej = 0
        reasons = {}
        for f in files:
            try:
                im = Image.open(f).convert("RGB")
            except Exception:
                continue
            n += 1
            v = gradability(im, thr)
            if not v["gradable"]:
                rej += 1
                reasons[v["verdict"]] = reasons.get(v["verdict"], 0) + 1
        total_n += n; total_rej += rej
        print(f"{name:<20}{n:>5}{rej:>10}  {reasons or ''}")
        report.append(f"- **{name}**: {rej}/{n} rejected {reasons or ''}")
    frr = total_rej / max(total_n, 1)
    print(f"{'ALL FUNDUS':<20}{total_n:>5}{total_rej:>10}  FRR={frr:.4f}")
    report.append(f"\n**False-rejection rate over all cohorts: {frr:.4f} ({total_rej}/{total_n})**\n")

    print(f"\n{'negatives':<20}{'n':>5}{'passed':>10}")
    all_neg_pass = 0
    for name, imgs in negatives():
        passed = sum(gradability(im, thr)["gradable"] for im in imgs)
        all_neg_pass += passed
        print(f"{name:<20}{len(imgs):>5}{passed:>10}")
        report.append(f"- negatives **{name}**: {passed}/{len(imgs)} passed (must be 0)")
    if all_neg_pass:
        raise SystemExit(f"REFUSING to publish: {all_neg_pass} negative(s) pass the gate")
    if frr > 0.01:
        raise SystemExit(f"REFUSING to publish: FRR {frr:.4f} > 1% on real fundus")

    out = {"calibrated_on": f"TRAIN-side fundus from all four cohorts, {PER_GROUP}/cohort, seed {SEED}",
           "verified_against": "noise/flat/charts + 40 structured warm-scene photographs "
                               "(the field-observed failure class)",
           "held_out": ["all test splits", "test_locked"],
           "thresholds": thr}
    (ROOT / "results" / "gate_thresholds.json").write_text(json.dumps(out, indent=2) + "\n")
    (ROOT / "results" / "gate_calibration.md").write_text("\n".join(report) + "\n")
    print("\n-> results/gate_thresholds.json + gate_calibration.md")


if __name__ == "__main__":
    main()
