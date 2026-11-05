"""Build a unified, patient-grouped ICROP staging manifest across every ROP source on disk.

`configs/rop.yaml` trains two classes, "No ROP" vs "ROP Detected". That collapse throws away
the stage axis AND pools four non-ROP diagnoses into the positive class (see
scripts/rop_recover_staging.py). This script builds the manifest a *staging* model needs.

SOURCES (all counted from files on this machine, none from a paper):
  ostrava    data/rop/ostrava/ (hardlinks over dataset_split/)       6,004 img /   188 inf
             + labels recovered from docs/infant_retinal_database_info.csv
  shenzhen   data/rop/shenzhen/                                       1,099 img /   483 inf
  ropvl      data/rop/rop_vl/                                         2,020 img / 1,116 inf
  multiview  data/rop/multiview/                                      1,324 img /   232 inf

DEDUPLICATION. ROP-VL and Shenzhen are overlapping cohorts from the same Shenzhen hospital
(6 of 7 ROP-VL authors are on the Shenzhen paper), so a byte hash is not enough — the shared
images were re-encoded. This uses a 256-bit dHash calibrated against Shenzhen's own known
byte-duplicates: within Shenzhen, true duplicates sit at distance 0 while genuinely different
images never come closer than 6, so any threshold in 1..5 is safe. Matches are resolved in
favour of ROP-VL, which carries the higher-resolution original (616-1720px) where Shenzhen
carries a 512x512 downscale.

PATIENT GROUPING is mandatory and namespaced per source (`ropvl:412`), because patient IDs
collide across datasets. 225 of 1,116 ROP-VL infants and 20 of 483 Shenzhen infants carry
more than one class, so a patient is NOT single-label — group on the patient, label the image.

    .venv/bin/python scripts/build_rop_staging_manifest.py

Writes results/prepared/rop_staging/manifest.csv (gitignored — it carries patient filenames).
"""
import csv
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
K = ROOT / "data" / "rop"
OUT = ROOT / "results" / "prepared" / "rop_staging" / "manifest.csv"

# One vocabulary for every source. `other_pathology` is deliberately NOT a ROP class: it is
# retained so the Ostrava images are accounted for, and excluded from staging training.
NORMAL, S0, S1, S2, S3, S4, S5 = ("normal", "stage_0", "stage_1", "stage_2", "stage_3",
                                  "stage_4", "stage_5")
AROP, TREATED, OTHER = "arop", "treated", "other_pathology"

OSTRAVA_DG = {"0": NORMAL, "1": S0, "2": S1, "3": S2, "4": S3, "5": S4, "6": S4, "7": S5,
              "8": AROP, "9": TREATED, "10": OTHER, "11": OTHER, "12": OTHER, "13": OTHER}
SHENZHEN = {"Normal": NORMAL, "Stage1": S1, "Stage2": S2, "Stage3": S3,
            "laser scars": TREATED}
# Keyed lowercase: the folder says "Normal" but img_info.xlsx says "normal". Those are the
# only two spellings that differ, and they differ only in case.
ROPVL = {"normal": NORMAL, "stage 1 rop": S1, "stage 2 rop": S2, "stage 3 rop": S3,
         "stage 4 rop": S4, "stage 5 rop": S5, "a-rop": AROP, "laser-treated rop": TREATED}
MULTIVIEW = {"Normal": NORMAL, "1": S1, "2": S2, "3": S3, "4": S4, "AROP": AROP}

# Matches BOTH the original Ostrava name and the de-identified one written by
# scripts/anonymise_ostrava_tree.py, since only id/device/serie are ever read:
#   <id>_F_GA##_BW####_PA##_DG#_PF#_D3_S01_17.jpg  (original, in dataset_split/)
#   183_D3_S01_17.jpg                              (de-identified, in data/rop/ostrava/)
KEY_RE = re.compile(r"^(\d+)_(?:.*_)?D(\d)_S(\d+)_")
DEVICE = {"1": "clarity_retcam3", "2": "phoenix_icon", "3": "retcam_envision"}

# IDs 40 and 41 are the SAME infant registered twice: identical sex/GA/BW in the metadata
# and 19 byte-identical JPEGs held under both IDs (with contradictory DG codes — DG0 under
# 41, DG1 under 40 — on the same pixels). Verified by md5 scan on 2026-08-11; every
# cross-ID byte-duplicate in the corpus belongs to this one pair. Today no leak reaches the
# staging folds only because DG1 (stage 0) is excluded from the 6-class task — an accident
# of the class filter, not a guarantee. Merging the group makes patient-grouped splits
# hold under ANY future class selection. The other 13 (sex,GA,BW)-sharing ID pairs have no
# duplicate images and are consistent with twins, which are genuinely different patients.
OSTRAVA_SAME_INFANT = {"41": "40"}


def rows_ostrava():
    meta_path = ROOT / "docs" / "infant_retinal_database_info.csv"
    with open(meta_path, encoding="utf-8-sig") as fh:
        meta = {}
        for r in csv.DictReader(fh, delimiter=";"):
            meta[(str(int(r["ID"])), r["DEVICE (D)"].strip(),
                  str(int(r["SERIE NUMBER (S)"])))] = r["DIAGNOSIS CODE (DG)"].strip()
    out, miss = [], 0
    # Read from the consolidated tree, NOT the legacy ROOT/dataset_split. Those are the same
    # bytes (data/rop/ostrava is a hardlink tree over them), but the manifest path is what
    # gets committed and shipped: a dataset_split/ path leaves 4,285 images — over half the
    # corpus — outside data/rop/, so a Kaggle upload of data/rop/ would silently omit them.
    for dp, _, files in os.walk(K / "ostrava"):
        for f in files:
            if not f.lower().endswith(".jpg"):
                continue
            m = KEY_RE.match(f)
            if not m:
                miss += 1
                continue
            key = (str(int(m.group(1))), m.group(2), str(int(m.group(3))))
            dg = meta.get(key)
            if dg is None:
                miss += 1
                continue
            out.append({"image_path": os.path.join(dp, f), "source": "ostrava",
                        "patient_id": f"ostrava:{OSTRAVA_SAME_INFANT.get(key[0], key[0])}",
                        "icrop": OSTRAVA_DG[dg],
                        "device": DEVICE.get(m.group(2), "?"), "native_label": f"DG{dg}"})
    if miss:
        print(f"  ostrava: {miss} files could not be keyed", file=sys.stderr)
    return out


def rows_shenzhen():
    base = K / "shenzhen"
    if not base.is_dir():
        return []
    sx = pd.ExcelFile(base / "zip information.xlsx").parse("Sheet1")
    pid = dict(zip(sx.img_name, sx.ID))
    out = []
    for cls in os.listdir(base / "image"):
        d = base / "image" / cls
        if not d.is_dir():
            continue
        for e in os.scandir(d):
            if not e.is_file() or e.name.startswith("."):
                continue
            p = pid.get(e.name)
            out.append({"image_path": e.path, "source": "shenzhen",
                        "patient_id": f"shenzhen:{p}" if p is not None else "shenzhen:UNKNOWN",
                        "icrop": SHENZHEN[cls], "device": "unknown", "native_label": cls})
    return out


def rows_ropvl():
    base = K / "rop_vl"
    if not base.is_dir():
        return []
    info = pd.read_excel(base / "img_info.xlsx")
    lut = dict(zip(info.image, zip(info.patient_id, info.categories)))
    out = []
    for cls in os.listdir(base / "Image"):
        d = base / "Image" / cls
        if not d.is_dir():
            continue
        for e in os.scandir(d):
            if not e.is_file() or not e.name.lower().endswith(".jpg"):
                continue
            p, cat = lut.get(e.name, (None, cls))
            out.append({"image_path": e.path, "source": "ropvl",
                        "patient_id": f"ropvl:{p}" if p is not None else "ropvl:UNKNOWN",
                        "icrop": ROPVL[cls.lower()], "device": "unknown",
                        "native_label": cls})
    return out


def rows_multiview():
    base = K / "multiview" / "ROP-DATASET"
    if not base.is_dir():
        return []
    ann = pd.read_excel(base / "Dataset Annotation Table.xlsx")
    ann["Label"] = ann.Label.astype(str)
    lut = {str(r.Image_name): (r.Patient, r.Label) for r in ann.itertuples()}
    out = []
    for dp, _, files in os.walk(base / "dataset"):
        for f in files:
            if not f.lower().endswith(".jpg"):
                continue
            stem = f[:-4]
            p, lab = lut.get(stem, (None, None))
            if lab is None:
                continue
            out.append({"image_path": os.path.join(dp, f), "source": "multiview",
                        "patient_id": f"multiview:{p}", "icrop": MULTIVIEW[lab],
                        "device": "unknown", "native_label": lab})
    return out


def dhash256(path, size=16):
    with Image.open(path) as im:
        g = im.convert("L").resize((size + 1, size), Image.LANCZOS)
    a = np.asarray(g, dtype=np.int16)
    return np.packbits((a[:, 1:] > a[:, :-1]).flatten())


POP = np.unpackbits(np.arange(256, dtype=np.uint8)[:, None], axis=1).sum(1).astype(np.int32)


def drop_cross_source_dupes(rows, keep="ropvl", drop="shenzhen", thr=4):
    """Drop `drop`-source images that are the same photograph as a `keep`-source image."""
    A = [r for r in rows if r["source"] == keep]
    B = [r for r in rows if r["source"] == drop]
    if not A or not B:
        return rows, 0
    HA = np.stack([dhash256(r["image_path"]) for r in A])
    HB = np.stack([dhash256(r["image_path"]) for r in B])
    kill = set()
    for i in range(0, len(HB), 128):
        d = POP[np.bitwise_xor(HB[i:i + 128, None, :], HA[None, :, :])].sum(-1)
        for j, dist in enumerate(d.min(axis=1)):
            if dist <= thr:
                kill.add(id(B[i + j]))
    return [r for r in rows if id(r) not in kill], len(kill)


def main():
    rows = []
    for name, fn in (("ostrava", rows_ostrava), ("shenzhen", rows_shenzhen),
                     ("ropvl", rows_ropvl), ("multiview", rows_multiview)):
        got = fn()
        print(f"  {name:<10} {len(got):>6} images")
        rows += got
    if not rows:
        print("no sources found", file=sys.stderr)
        return 1

    print("\n  de-duplicating shenzhen against ropvl (same hospital, re-encoded copies)...")
    rows, n = drop_cross_source_dupes(rows)
    print(f"  dropped {n} shenzhen images that duplicate a higher-resolution ROP-VL image")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["image_path", "source", "patient_id", "icrop",
                                           "device", "native_label"])
        w.writeheader()
        w.writerows(rows)

    df = pd.DataFrame(rows)
    ORDER = [NORMAL, S0, S1, S2, S3, S4, S5, AROP, TREATED, OTHER]
    piv = pd.crosstab(df.icrop, df.source).reindex(ORDER, fill_value=0)
    for c in ["ostrava", "shenzhen", "ropvl", "multiview"]:
        if c not in piv:
            piv[c] = 0
    piv = piv[["ostrava", "shenzhen", "ropvl", "multiview"]]
    piv["TOTAL"] = piv.sum(axis=1)
    inf = df.groupby("icrop").patient_id.nunique().reindex(ORDER, fill_value=0)
    piv["infants"] = inf

    print("\n" + "=" * 84)
    print("UNIFIED ICROP STAGING MANIFEST")
    print("=" * 84)
    print(piv.to_string())
    print("-" * 84)
    print(f"{'TOTAL':<16}{len(df):>6} images   {df.patient_id.nunique()} distinct infants "
          f"across {df.source.nunique()} sources")
    trainable = df[~df.icrop.isin([OTHER])]
    print(f"{'STAGEABLE':<16}{len(trainable):>6} images   "
          f"{trainable.patient_id.nunique()} infants (excludes other_pathology)")
    print(f"\n-> {OUT}")

    thin = [c for c in ORDER if 0 < piv.loc[c, "TOTAL"] < 100]
    if thin:
        print(f"\n  WARNING - classes with under 100 images: {', '.join(thin)}. "
              f"Stage 4/5 come from one hospital only.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
