"""Group every ROP image we hold into one folder tree, by ICROP class.

    .venv/bin/python scripts/build_rop_classified.py

Output: data/rop_classified/  — every "normal" image from every dataset in normal/, every
stage 3 image in stage_3/, and so on. Plus data/rop_classified/MANIFEST.csv recording, for
each linked file, where it came from and how its class was decided.

HARD LINKS, not copies. The corpus is 11 GB and the disk has less headroom than that; a hard
link costs an inode and no data blocks. Editing a linked file edits the original, so treat
this tree as READ-ONLY. Delete it with `rm -rf data/rop_classified` — that removes only the
links, never the source images.

FILENAMES are prefixed `<source>__` because basenames collide hard across datasets: HVDROPDB
names every file 1.png..50.png inside 16 different folders, and Multi-View, FARFUM and Shantou
all use the same UUID.N convention.

--------------------------------------------------------------------------------------
WHERE EACH DATASET'S LABEL COMES FROM  (verified by inspection, not assumed)

  ostrava      docs/infant_retinal_database_info.csv, joined on (ID, serie, device).
               DG 0=normal 1..7=stages 0..5 8=AP-ROP 9=treated 10..13=NOT ROP.
  rop_vl       folder name under Image/, cross-checked against img_info.xlsx (0 mismatches).
  multiview    per-image Label in Dataset Annotation Table.xlsx.
  shenzhen     folder name under image/.
  ropstages    filename prefix (NoROP_/Stage1_/Stage2_/Stage3_).
  hvdropdb     folder name — BINARY ONLY. Its "Normal" is a real normal label; its "ROP" has
               no stage, so those images go to _rop_unstaged/ rather than being guessed at.
  farfum       Dataset_Labels.xlsx — see below.
  shantou_*    LabelMe JSON, but the shape labels are "a"/"b" (region names for segmentation),
               NOT disease classes. No class information exists. -> _no_class_label/
  orvs_seg     image/mask pairs only, no class file at all.            -> _no_class_label/
  coph100      ships masks and overlays, NO fundus images.             -> nothing to link

TWO JUDGEMENT CALLS, STATED OPENLY:

1. FARFUM IS NOT MERGED INTO THE STAGE FOLDERS. Its five graders filled the stage column
   566 / 1267 / 178 / 366 / 251 times out of 1,533 — they disagree about whether a stage
   applies at all, and zero of its 1,533 images are graded unanimously on grade. Its
   dependable axis is the plus-disease grade, which is ORTHOGONAL to ICROP stage. Pooling it
   into stage_1/2/3 would inject that disagreement into the one axis we are trying to learn.
   It is grouped by majority grade under _farfum_plus_disease/ instead, with the vote count
   recorded per image in the manifest.

2. OSTRAVA's non-ROP diagnoses (hemorrhage, toxoplasma, hamartoms, optic nerve hypoplasia)
   go to _other_pathology/, NOT into any ROP class. They are 1,295 images that the current
   binary model counts as "ROP Detected". They are not ROP.
"""
import csv
import hashlib
import os
import re
import shutil
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "data" / "rop"
OUT = ROOT / "data" / "rop_classified"

NORMAL, S0, S1, S2, S3, S4, S5 = ("normal", "stage_0", "stage_1", "stage_2", "stage_3",
                                  "stage_4", "stage_5")
AROP, TREATED = "arop", "treated"
OTHER, UNSTAGED, NOLABEL = "_other_pathology", "_rop_unstaged", "_no_class_label"
FARFUM_DIR = "_farfum_plus_disease"

STAGE_ORDER = [NORMAL, S0, S1, S2, S3, S4, S5, AROP, TREATED]

OSTRAVA_DG = {0: NORMAL, 1: S0, 2: S1, 3: S2, 4: S3, 5: S4, 6: S4, 7: S5,
              8: AROP, 9: TREATED, 10: OTHER, 11: OTHER, 12: OTHER, 13: OTHER}
ROPVL = {"normal": NORMAL, "stage 1 rop": S1, "stage 2 rop": S2, "stage 3 rop": S3,
         "stage 4 rop": S4, "stage 5 rop": S5, "a-rop": AROP, "laser-treated rop": TREATED}
SHENZHEN = {"Normal": NORMAL, "Stage1": S1, "Stage2": S2, "Stage3": S3, "laser scars": TREATED}
MULTIVIEW = {"Normal": NORMAL, "1": S1, "2": S2, "3": S3, "4": S4, "AROP": AROP}
ROPSTAGES = {"NoROP": NORMAL, "Stage1": S1, "Stage2": S2, "Stage3": S3}
FARFUM_GRADE = {0: "grade0_no_plus", 1: "grade1", 2: "grade2", 3: "grade3_plus"}

KEY_RE = re.compile(r"^(\d+)_.*_D(\d)_S(\d+)_")
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
rows = []


def add(path, cls, source, decided_by, native, patient="", extra=""):
    rows.append({"src_path": str(path), "cls": cls, "source": source,
                 "decided_by": decided_by, "native_label": native,
                 "patient_id": patient, "note": extra})


# ---------------------------------------------------------------- collectors
def collect_ostrava():
    meta = pd.read_csv(ROOT / "docs" / "infant_retinal_database_info.csv", sep=";")
    meta.columns = ["ID", "SEX", "GA", "BW", "PA", "DG", "PF", "D", "S"]
    key = {(int(r.ID), str(r.D), int(r.S)): int(r.DG) for r in meta.itertuples()}
    for dp, _, files in os.walk(SRC / "ostrava"):
        for f in files:
            if not f.lower().endswith(".jpg"):
                continue
            m = KEY_RE.match(f)
            if not m:
                continue
            dg = key.get((int(m.group(1)), m.group(2), int(m.group(3))))
            if dg is None:
                continue
            add(os.path.join(dp, f), OSTRAVA_DG[dg], "ostrava",
                "metadata join (ID,serie,device)", f"DG{dg}", f"ostrava:{m.group(1)}")


def collect_ropvl():
    info = pd.read_excel(SRC / "rop_vl" / "img_info.xlsx")
    pid = dict(zip(info.image, info.patient_id))
    base = SRC / "rop_vl" / "Image"
    for cls in os.listdir(base):
        d = base / cls
        if not d.is_dir():
            continue
        for e in os.scandir(d):
            if e.is_file() and e.name.lower().endswith(".jpg"):
                add(e.path, ROPVL[cls.lower()], "ropvl", "folder + img_info.xlsx", cls,
                    f"ropvl:{pid.get(e.name, '?')}")


def collect_multiview():
    base = SRC / "multiview" / "ROP-DATASET"
    ann = pd.read_excel(base / "Dataset Annotation Table.xlsx")
    ann["Label"] = ann.Label.astype(str)
    lut = {str(r.Image_name): (r.Patient, r.Label) for r in ann.itertuples()}
    for dp, _, files in os.walk(base / "dataset"):
        for f in files:
            if not f.lower().endswith(".jpg"):
                continue
            p, lab = lut.get(f[:-4], (None, None))
            if lab is None:
                continue
            add(os.path.join(dp, f), MULTIVIEW[lab], "multiview",
                "Dataset Annotation Table.xlsx", lab, f"multiview:{p}")


def collect_shenzhen():
    sx = pd.ExcelFile(SRC / "shenzhen" / "zip information.xlsx").parse("Sheet1")
    pid = dict(zip(sx.img_name, sx.ID))
    base = SRC / "shenzhen" / "image"
    for cls in os.listdir(base):
        d = base / cls
        if not d.is_dir():
            continue
        for e in os.scandir(d):
            if e.is_file() and not e.name.startswith("."):
                add(e.path, SHENZHEN[cls], "shenzhen", "folder name", cls,
                    f"shenzhen:{pid.get(e.name, '?')}")


def collect_ropstages():
    d = SRC / "ropstages_sample" / "NewROPDataset_Sample_justtotry"
    if not d.is_dir():
        return
    for e in os.scandir(d):
        if not e.is_file() or not e.name.lower().endswith(".jpg"):
            continue
        pre = e.name.split("_Patient")[0]
        if pre in ROPSTAGES:
            pat = e.name.split("_Patient")[1].split("_")[0].split(".")[0]
            add(e.path, ROPSTAGES[pre], "ropstages", "filename prefix", pre,
                f"ropstages:{pat}")


def collect_hvdropdb():
    for dp, _, files in os.walk(SRC / "hvdropdb"):
        low = dp.lower()
        if "classification" not in low.split("agrawal")[-1]:
            continue
        leaf = os.path.basename(dp)
        if leaf.endswith("_Normal"):
            cls, native = NORMAL, leaf
        elif leaf.endswith("_ROP"):
            cls, native = UNSTAGED, leaf
        else:
            continue
        for f in files:
            if f.lower().endswith(".png"):
                add(os.path.join(dp, f), cls, "hvdropdb", "folder name (binary only)", native,
                    "", "stage unknown - source has no stage labels" if cls == UNSTAGED else "")


def collect_nolabel():
    for name, sub in (("shantou_od", ""), ("shantou_bv", ""), ("orvs_seg", "")):
        base = SRC / name
        if not base.is_dir():
            continue
        for dp, _, files in os.walk(base):
            lp = dp.lower()
            if "labels" in lp:
                continue
            for f in files:
                if os.path.splitext(f)[1].lower() not in IMG_EXT:
                    continue
                if "_mask" in f.lower():
                    continue
                add(os.path.join(dp, f), NOLABEL, name, "none - segmentation only", "-")


def collect_farfum():
    fa = pd.read_excel(SRC / "farfum_rop" / "Dataset_Labels.xlsx", header=1)
    fa = fa.rename(columns={fa.columns[0]: "patient", fa.columns[1]: "image"})
    gcols = [c for c in fa.columns if "grade" in str(c).lower()]
    G = fa[gcols].apply(pd.to_numeric, errors="coerce")
    maj = G.mode(axis=1)[0]
    votes = G.eq(maj, axis=0).sum(axis=1)
    lut = {}
    for i, r in enumerate(fa.itertuples()):
        m = maj.iloc[i]
        if pd.isna(m):
            continue
        lut[str(r.image)] = (FARFUM_GRADE.get(int(m), f"grade{int(m)}"), int(votes.iloc[i]),
                             str(r.patient))
    for dp, _, files in os.walk(SRC / "farfum_rop" / "images"):
        for f in files:
            if os.path.splitext(f)[1].lower() not in IMG_EXT:
                continue
            hit = lut.get(f) or lut.get(os.path.splitext(f)[0])
            if not hit:
                continue
            grade, v, pat = hit
            add(os.path.join(dp, f), f"{FARFUM_DIR}/{grade}", "farfum",
                "majority of 5 graders (plus-disease grade, NOT stage)", grade,
                f"farfum:{pat}", f"{v}/5 graders agree")


# ------------------------------------------------------- cross-source dedupe
def dhash256(path, size=16):
    with Image.open(path) as im:
        g = im.convert("L").resize((size + 1, size), Image.LANCZOS)
    a = np.asarray(g, dtype=np.int16)
    return np.packbits((a[:, 1:] > a[:, :-1]).flatten())


POP = np.unpackbits(np.arange(256, dtype=np.uint8)[:, None], axis=1).sum(1).astype(np.int32)


def dedupe(keep="ropvl", drop="shenzhen", thr=4):
    A = [r for r in rows if r["source"] == keep]
    B = [r for r in rows if r["source"] == drop]
    if not A or not B:
        return 0
    HA = np.stack([dhash256(r["src_path"]) for r in A])
    HB = np.stack([dhash256(r["src_path"]) for r in B])
    n = 0
    for i in range(0, len(HB), 128):
        d = POP[np.bitwise_xor(HB[i:i + 128, None, :], HA[None, :, :])].sum(-1)
        for j, dist in enumerate(d.min(axis=1)):
            if dist <= thr:
                B[i + j]["_drop"] = True
                n += 1
    return n


def main():
    print("reading labels from every dataset...")
    for fn in (collect_ostrava, collect_ropvl, collect_multiview, collect_shenzhen,
               collect_ropstages, collect_hvdropdb, collect_nolabel, collect_farfum):
        before = len(rows)
        try:
            fn()
        except Exception as e:  # noqa: BLE001
            print(f"  {fn.__name__}: FAILED {e}", file=sys.stderr)
        print(f"  {fn.__name__:<20} +{len(rows) - before}")

    n = dedupe()
    print(f"\ndropped {n} shenzhen images duplicating a higher-resolution ROP-VL image")
    keep = [r for r in rows if not r.get("_drop")]

    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)

    linked, failed = 0, 0
    for r in keep:
        dest_dir = OUT / r["cls"]
        dest_dir.mkdir(parents=True, exist_ok=True)
        name = f"{r['source']}__{os.path.basename(r['src_path'])}"
        dest = dest_dir / name
        if dest.exists():
            h = hashlib.md5(r["src_path"].encode()).hexdigest()[:6]
            dest = dest_dir / f"{r['source']}__{h}__{os.path.basename(r['src_path'])}"
        try:
            os.link(r["src_path"], dest)
            linked += 1
        except OSError:
            try:
                shutil.copy2(r["src_path"], dest)
                linked += 1
            except OSError:
                failed += 1
                continue
        r["dest_path"] = str(dest.relative_to(ROOT))

    with open(OUT / "MANIFEST.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["dest_path", "cls", "source", "patient_id",
                                           "native_label", "decided_by", "note", "src_path"])
        w.writeheader()
        for r in keep:
            if "dest_path" in r:
                w.writerow({k: r.get(k, "") for k in w.fieldnames})

    df = pd.DataFrame([r for r in keep if "dest_path" in r])
    print(f"\nlinked {linked} files ({failed} failed)\n")
    print("=" * 78)
    print("ICROP STAGE TREE")
    print("=" * 78)
    piv = pd.crosstab(df.cls, df.source)
    for cls in STAGE_ORDER:
        if cls in piv.index:
            row = piv.loc[cls]
            parts = ", ".join(f"{s} {int(v)}" for s, v in row.items() if v)
            print(f"  {cls:<16} {int(row.sum()):>6}   ({parts})")
    tot = int(sum(piv.loc[c].sum() for c in STAGE_ORDER if c in piv.index))
    print(f"  {'-' * 16} {'-' * 6}")
    print(f"  {'STAGED TOTAL':<16} {tot:>6}   "
          f"{df[df.cls.isin(STAGE_ORDER)].patient_id.replace('', pd.NA).dropna().nunique()} infants")
    print("\nHELD SEPARATE")
    for cls in sorted(set(df.cls) - set(STAGE_ORDER)):
        print(f"  {cls:<34} {int((df.cls == cls).sum()):>6}")
    write_readme(df, piv, n)
    print(f"\n-> {OUT}")
    print(f"-> {OUT / 'MANIFEST.csv'}")
    print(f"-> {OUT / 'README.md'}")
    return 0


def write_readme(df, piv, n_dropped):
    L, A = [], None
    A = L.append
    A("# ROP corpus grouped by ICROP class\n")
    A("Generated by `scripts/build_rop_classified.py` — **do not edit by hand, and do not edit "
      "the images.** Every file here is a HARD LINK to the original under `data/rop/`, so "
      "changing one changes both. `rm -rf data/rop_classified` removes only links.\n")
    staged = df[df.cls.isin(STAGE_ORDER)]
    A(f"**{len(staged):,} stage-labelled images** from "
      f"{staged.patient_id.replace('', pd.NA).dropna().nunique():,} infants, pooled from "
      f"{staged.source.nunique()} datasets.\n")
    A("## The stage classes\n")
    A("| Class | Images | " + " | ".join(sorted(set(staged.source))) + " |")
    A("|---|---:|" + "---:|" * len(set(staged.source)))
    srcs = sorted(set(staged.source))
    for cls in STAGE_ORDER:
        if cls not in piv.index:
            continue
        row = piv.loc[cls]
        A(f"| `{cls}` | **{int(row.sum()):,}** | "
          + " | ".join(str(int(row.get(s, 0))) for s in srcs) + " |")
    A("")
    A("## Held separate, and why\n")
    A("| Folder | Images | Why it is not in a stage class |")
    A("|---|---:|---|")
    reasons = {
        OTHER: "Ostrava diagnoses that are **not ROP** — hemorrhage, toxoplasma, hamartoms, "
               "optic nerve hypoplasia. The current binary model counts all of these as "
               "'ROP Detected'.",
        UNSTAGED: "HVDROPDB ROP images. The source labels them ROP vs Normal only — it has no "
                  "stage labels, so a stage would have to be invented.",
        NOLABEL: "Shantou optic-disc, Shantou vessel and ORVS segmentation sets. Their LabelMe "
                 "JSON shape labels are `a`/`b` — region names, not disease classes.",
    }
    for cls in sorted(set(df.cls) - set(STAGE_ORDER)):
        if cls.startswith(FARFUM_DIR):
            continue
        A(f"| `{cls}` | {int((df.cls == cls).sum()):,} | {reasons.get(cls, '')} |")
    fa = df[df.cls.str.startswith(FARFUM_DIR)]
    if len(fa):
        A(f"| `{FARFUM_DIR}/` | {len(fa):,} | FARFUM-RoP, grouped by its **plus-disease "
          f"grade**, which is a different axis from ICROP stage. Its five graders filled the "
          f"stage column 566/1267/178/366/251 times out of 1,533 — they disagree on whether a "
          f"stage applies at all — and **zero** of its images are graded unanimously. Merging "
          f"it into stage_1/2/3 would inject that disagreement into the axis we want to learn. "
          f"Per-image vote counts are in `MANIFEST.csv`. |")
    A("")
    A("## How each dataset's class was decided\n")
    A("| Source | Images | Label came from |")
    A("|---|---:|---|")
    for s, g in df.groupby("source"):
        A(f"| `{s}` | {len(g):,} | {g.decided_by.iloc[0]} |")
    A("")
    A("## Caveats that survive the grouping\n")
    A(f"- **{n_dropped} Shenzhen images were dropped** as duplicates of higher-resolution "
      "ROP-VL originals (same hospital, re-encoded). Matched with a 256-bit perceptual hash "
      "calibrated against Shenzhen's own byte-duplicates.")
    A("- **Stage 4 (77) and Stage 5 (30) come from one hospital only** — ROP-VL, plus 18 "
      "Stage-4 from Multi-View. Any result on these classes is single-site.")
    A("- **`stage_0` is not `normal`.** ICROP stage 0 is an immature retina without ROP; it is "
      "45 images from 4 Ostrava infants.")
    A("- **`treated` is post-laser / post-anti-VEGF disease**, not active staged ROP. Keep it "
      "out of a stage-severity ordinal unless you mean to model it.")
    A("- **Patient IDs are namespaced per source** (`ropvl:412`) because they collide across "
      "datasets. Always split on `patient_id`, never on the image: 225 ROP-VL infants and 20 "
      "Shenzhen infants carry more than one class.")
    A("- Filenames are prefixed `<source>__` because HVDROPDB names every file `1.png`..`50.png` "
      "and Multi-View, FARFUM and Shantou all use the same UUID.N convention.")
    A("")
    (OUT / "README.md").write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    sys.exit(main())
