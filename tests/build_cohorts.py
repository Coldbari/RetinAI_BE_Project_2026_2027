"""Deterministically materialise the evaluation cohorts into .cohorts/ (gitignored).

  infants/DG0, infants/DG1  - PRIVATE patient data, extracted from the local rop_dataset.zip.
                              One image per patient (repeat photos of the same baby are not
                              independent samples). NEVER commit these.
  hrf_test/{healthy,dr,glaucoma} - HRF, Dept. of Ophthalmology, FAU Erlangen-Nuremberg.
                              CC-BY 4.0. Budai et al., Int. J. Biomedical Imaging, 2013.
                              NOTE: HRF is dataset #9 inside SMDG-19, so the Glaucoma model
                              may have trained on these images — see docs/REMEDIATION_PLAN.md
                              Phase 3. Valid as an unseen set for DR and ROP only.
  nonfundus/                - synthetic out-of-distribution probes.

    .venv/bin/python tests/build_cohorts.py
"""
import io
import os
import random
import re
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / ".cohorts"
HRF = "https://www5.cs.fau.de/fileadmin/research/datasets/fundus-images"
SEED = 42


def build_infants(k=30):
    src = ROOT / "rop_dataset.zip"
    if not src.exists():
        print(f"  skip infants — {src.name} not present")
        return
    random.seed(SEED)
    z = zipfile.ZipFile(src)
    names = [n for n in z.namelist()
             if n.startswith("dataset_split/test/") and n.lower().endswith(".jpg")]
    for cls in ("DG0", "DG1"):
        pool = [n for n in names if f"/{cls}/" in n]
        by_patient = {}
        for n in sorted(pool):
            by_patient.setdefault(Path(n).name.split("_")[0], []).append(n)
        pids = sorted(by_patient)
        random.shuffle(pids)
        d = OUT / "infants" / cls
        d.mkdir(parents=True, exist_ok=True)
        picked = [random.choice(by_patient[p]) for p in pids[:k]]
        for n in picked:
            (d / Path(n).name).write_bytes(z.read(n))
        gas = sorted({int(m.group(1)) for n in picked
                      if (m := re.search(r"_GA(\d+)_", Path(n).name))})
        print(f"  infants/{cls}: {len(picked)} images / {len(pids)} patients, "
              f"GA {gas[0]}-{gas[-1]}w" if gas else f"  infants/{cls}: {len(picked)}")


def build_hrf():
    for key, sub in (("healthy", "healthy"), ("diabetic_retinopathy", "dr"), ("glaucoma", "glaucoma")):
        d = OUT / "hrf_test" / sub
        if d.exists() and any(d.iterdir()):
            print(f"  hrf_test/{sub}: already present")
            continue
        d.mkdir(parents=True, exist_ok=True)
        print(f"  hrf_test/{sub}: downloading…")
        with urllib.request.urlopen(f"{HRF}/{key}.zip", timeout=120) as r:
            z = zipfile.ZipFile(io.BytesIO(r.read()))
        for n in z.namelist():
            if n.lower().endswith((".jpg", ".jpeg", ".png")) and "/" not in n.strip("/"):
                (d / Path(n).name).write_bytes(z.read(n))
        print(f"  hrf_test/{sub}: {len(list(d.iterdir()))} images")


def build_nonfundus():
    import numpy as np
    from PIL import Image
    d = OUT / "nonfundus"
    d.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    Image.fromarray(rng.integers(0, 255, (800, 800, 3), dtype=np.uint8)).save(d / "noise.png")
    Image.fromarray(np.full((800, 800, 3), 128, np.uint8)).save(d / "gray.png")
    for c in ("outputs/evaluation_dashboard.png", "outputs/training_curves_unified.png"):
        p = ROOT / c
        if p.exists():
            Image.open(p).convert("RGB").save(d / p.name)
    print(f"  nonfundus: {len(list(d.iterdir()))} probes")


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    print(f"Building cohorts into {OUT}")
    build_infants()
    build_hrf()
    build_nonfundus()
    print("\nDone. Run:  .venv/bin/python tests/cohort_regression.py")
    if not (ROOT / ".gitignore").read_text().count(".cohorts"):
        print("\nWARNING: .cohorts/ is not gitignored and holds PRIVATE patient images.", file=sys.stderr)
        sys.exit(1)
