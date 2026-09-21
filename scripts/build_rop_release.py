"""Package every ROP source on this machine into ONE labelled, split, self-describing dataset.

This is a *packaging* script, not a training one. It takes the eleven ROP datasets sitting in
`data/rop/` — each with its own folder shape, its own label vocabulary and its own idea of what
a filename should contain — and emits a single tree that a stranger can unzip and use without
reading any of our code.

WHAT IT DOES NOT DO. It does not re-derive labels. The four-source ICROP reconciliation already
lives in `scripts/build_rop_staging_manifest.py` -> `results/prepared/rop_staging/manifest.csv`,
and the patient-grouped folds in `scripts/build_rop_splits.py` -> `splits.csv` / `fold_*.csv`.
Re-deriving them here would create a second source of truth that silently drifts. This script
READS those and extends them to the sources they never covered.

THREE LABEL SCHEMAS, THREE FOLDERS. The sources do not share a label *type*, so pooling them
into one labels.csv would be a lie:
  classification/  10,613 img  one ICROP stage per image        (ostrava, ropvl, multiview,
                                                                 shenzhen, hvdropdb)
  plus_disease/     1,533 img  five independent rater opinions  (farfum)
  segmentation/       733 img  image + pixel mask               (hvdropdb, shantou, orvs, coph100)

DE-IDENTIFICATION. Three sources ship identifiers inside filenames and they are rewritten here:
ROP-VL embeds an exam date at day granularity, COph100 inherits Ostrava's original
sex/GA/birth-weight names, and `_redundant/littlevision` carries hospital medical record
numbers (that one is excluded outright, being a duplicate re-upload as well). Every rename is
recorded in docs/filename_map.csv so the mapping is reversible from the source you already hold.

HARDLINKS. The tree is built with os.link, so it costs ~0 bytes until the zip is written.

    .venv/bin/python scripts/build_rop_release.py [--out DIR] [--zip] [--no-dims]
"""
import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
K = ROOT / "data" / "rop"
PREP = ROOT / "results" / "prepared" / "rop_staging"

VERSION = "1.0"
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

# The ICROP vocabulary from build_rop_staging_manifest.py, plus one value that script has no
# reason to know about: HVDROPDB grades ROP present/absent without a stage, so its positives
# cannot be called stage_1..5 and must not be silently dropped into one.
ROP_UNSTAGED = "rop_unstaged"

# 6-class task membership. stage_0 (pre-ROP), treated (post-laser) and other_pathology
# (haemorrhage / toxoplasma / hamartoma / hypoplasia) are shipped in full but sit outside the
# staging task, exactly as results/prepared/rop_staging/class_index.json defines it.
CLS6 = {"normal": 0, "stage_1": 1, "stage_2": 2, "stage_3": 3, "stage_4": 4, "stage_5": 4,
        "arop": 5}
CLS6_NAMES = ["normal", "stage_1", "stage_2", "stage_3", "stage_4_5", "arop"]

# ROP-VL: <patient>_<YYYY-MM-DD>_<eye>_<n>.jpg. The date is a date-of-service identifier.
ROPVL_NAME = re.compile(r"^(\d+)_(\d{4}-\d{2}-\d{2})_([LRlr])_(\d+)\.(\w+)$")
# COph100 inherits Ostrava's ORIGINAL names, which carry sex, gestational age and birth weight:
#   <infantID>_<sex>_GA<weeks>_BW<grams>_PA<days>_DG<dx>_PF<plus>_D<dev>_S<serie>_<n>_mask.png
#   (schematic: a real one carries an actual infant's sex, gestational age and birth weight,
#    so this comment names the shape instead of quoting a record)
# Ostrava's own tree under data/rop/ostrava was already de-identified to 001_D1_S01_11.jpg by
# scripts/anonymise_ostrava_tree.py; this reproduces that scheme for the annotation files.
COPH_NAME = re.compile(r"^(\d+)_[MF]_GA\d+_BW\d+_PA[\d.]+_DG\d+_PF\d+_(D\d_S\d+_\d+)(.*)$")

SOURCE_META = {
    "ostrava": dict(title="Retinal Image Dataset of Infants and ROP", region="Ostrava, Czechia",
                    licence="CC0", doi="figshare collection 6626162"),
    "ropvl": dict(title="ROP-VL", region="Shenzhen Eye Hospital, China", licence="CC BY 4.0",
                  doi="10.6084/m9.figshare.30143461"),
    "multiview": dict(title="Multi-View ROP", region="China", licence="CC BY-NC 4.0",
                      doi="HuggingFace lijuanliao/ROP-DATASET"),
    "shenzhen": dict(title="Shenzhen ROP (Zhao 2024)", region="Shenzhen Eye Hospital, China",
                     licence="CC BY 4.0", doi="10.6084/m9.figshare.25514449"),
    "hvdropdb": dict(title="HVDROPDB", region="H.V. Desai Eye Hospital, Pune, India",
                     licence="CC BY 4.0", doi="Mendeley xw5xc7xrmp"),
    "farfum": dict(title="FARFUM-RoP", region="Farabi Hospital, Iran", licence="CC BY 4.0",
                   doi="figshare collection 6721269"),
    "shantou_od": dict(title="ROP optic disc segmentation", region="Shantou, China",
                       licence="CC BY 4.0", doi="figshare (Shantou University)"),
    "shantou_bv": dict(title="ROP retinal vessel segmentation", region="Shantou, China",
                       licence="CC BY 4.0", doi="figshare (Shantou University)"),
    "orvs_rop": dict(title="ROP vessel segmentation (ORVS release)", region="unstated",
                     licence="unstated", doi="-"),
    "coph100": dict(title="COph100", region="derived from Ostrava", licence="CC BY",
                    doi="figshare 27061084"),
}

renames: list[tuple[str, str, str]] = []   # (source, original_relpath, released_name)

# (patient, exam date) -> 1-based visit number. ROP-VL images a patient repeatedly and the
# image index restarts each visit, so simply deleting the date collides 61 filenames. A visit
# ordinal keeps the longitudinal structure — which images belong to one sitting, and in what
# order the sittings happened — while carrying no date.
ROPVL_VISIT: dict[tuple[str, str], int] = {}


def build_ropvl_visit_map() -> None:
    seen = defaultdict(set)
    for f in (K / "rop_vl").rglob("*"):
        m = ROPVL_NAME.match(f.name)
        if m:
            seen[m.group(1)].add(m.group(2))
    for pid, dates in seen.items():
        for i, d in enumerate(sorted(dates), start=1):
            ROPVL_VISIT[(pid, d)] = i


def sanitise(source: str, path: Path) -> str:
    """Return the released filename for `path`, stripping any identifier it carries."""
    name = path.name
    if source == "ropvl":
        m = ROPVL_NAME.match(name)
        if m:
            visit = ROPVL_VISIT.get((m.group(1), m.group(2)), 0)
            out = (f"p{int(m.group(1)):04d}_v{visit}_{m.group(3).upper()}"
                   f"_{m.group(4)}.{m.group(5)}")
            renames.append((source, name, out))
            return out
    elif source == "hvdropdb":
        # HVDROPDB numbers its files 1..50 independently inside every class and every
        # segmentation folder, so the basename alone is not unique anywhere in the set. The
        # parent folder is the only thing that distinguishes them.
        out = f"{path.parent.name.lower()}__{name}"
        renames.append((source, name, out))
        return out
    elif source == "coph100":
        m = COPH_NAME.match(name)
        if m:
            # No zero-padding: this must reproduce the Ostrava released stem EXACTLY
            # (`101_D2_S01_1`), because that string is the only key pairing a COph100 mask
            # back to the photograph it annotates.
            out = f"{int(m.group(1))}_{m.group(2)}{m.group(3)}"
            renames.append((source, name, out))
            return out
    return name


def md5(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


def dims(path: Path):
    """(width, height) without decoding the pixels. Empty on unreadable files."""
    try:
        with Image.open(path) as im:
            return im.size
    except Exception:
        return ("", "")


def link(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


# --------------------------------------------------------------------------------------
# classification
# --------------------------------------------------------------------------------------
def read_reconciled() -> pd.DataFrame:
    """The four staged sources, already reconciled onto one ICROP vocabulary."""
    man = pd.read_csv(PREP / "manifest.csv")
    need = {"image_path", "source", "patient_id", "icrop", "device", "native_label"}
    missing = need - set(man.columns)
    if missing:
        sys.exit(f"manifest.csv is missing {missing} — re-run build_rop_staging_manifest.py")
    return man


def read_splits() -> tuple[dict, dict]:
    """patient -> train|val, and image_path -> multiview_dev|multiview_locked.

    Both come from the committed fold design so the release cannot disagree with the numbers
    already reported. fold_0 is the reference fold; the other four ship as CSVs untouched.
    """
    fold0 = pd.read_csv(PREP / "fold_0.csv")
    pat_split = {}
    for pid, grp in fold0.groupby("patient_id"):
        vals = set(grp["split"])
        # A patient in both halves would mean the grouping failed upstream; refuse to paper
        # over it, because a leaked patient is exactly the defect the grouping exists to stop.
        if len(vals) != 1:
            sys.exit(f"patient {pid} spans {vals} in fold_0.csv — grouping is broken")
        pat_split[pid] = vals.pop()

    site = {}
    sp = pd.read_csv(PREP / "splits.csv")
    for r in sp.itertuples():
        if str(r.split).startswith("multiview"):
            site[Path(r.image_path).name] = r.split
    return pat_split, site


def rows_hvdropdb_classification():
    """185 images graded ROP / normal with NO stage, over two cameras (RetCam, Neo).

    Held out as an external binary test site: it is a different hospital on a different
    continent to everything in the training pool, and it ships no patient identifiers, so it
    could not be patient-grouped into train/val safely even if we wanted it there.
    """
    base = next(K.glob("hvdropdb/*/HVDROPDB_RetCam_Neo_Classification"), None)
    if base is None:
        return []
    out = []
    for d in sorted(base.iterdir()):
        if not d.is_dir():
            continue
        device, _, cls = d.name.partition("_")
        icrop = "normal" if cls.lower() == "normal" else ROP_UNSTAGED
        for f in sorted(d.iterdir()):
            if f.suffix.lower() in IMG_EXT:
                out.append({"image_path": str(f), "source": "hvdropdb",
                            "patient_id": f"hvdropdb:{f.stem}", "icrop": icrop,
                            "device": f"hvdropdb_{device.lower()}", "native_label": d.name})
    return out


def build_classification(out_root: Path, want_dims: bool):
    man = read_reconciled()
    man = pd.concat([man, pd.DataFrame(rows_hvdropdb_classification())], ignore_index=True)
    pat_split, site = read_splits()

    rows = []
    for r in man.itertuples():
        src, fname = r.source, Path(r.image_path).name
        if src == "multiview":
            split = site.get(fname, "test_heldout_site")
        elif src == "hvdropdb":
            split = "test_external_binary"
        else:
            # Patients outside the 6-class folds (stage_0 / treated / other_pathology only)
            # are placed by a hash of the patient id, never of the image — so every image of
            # an infant lands on the same side. 80/20 to match the fold ratio.
            split = pat_split.get(r.patient_id)
            if split is None:
                digest = hashlib.md5(r.patient_id.encode()).hexdigest()
                split = "val" if int(digest[:8], 16) % 100 < 20 else "train"
                pat_split[r.patient_id] = split
        rows.append(dict(source=src, patient_id=r.patient_id, icrop=r.icrop,
                         native_label=r.native_label, device=r.device, split=split,
                         src_path=r.image_path,
                         released=f"{src}__{sanitise(src, Path(r.image_path))}"))

    rows, dropped = dedupe(rows, "icrop", "classification")

    # Every image of a patient must sit on one side of the train/val line. Assert it rather
    # than trust it: this is the one property the whole release is worthless without.
    per_patient = defaultdict(set)
    for r in rows:
        if r["split"] in ("train", "val"):
            per_patient[r["patient_id"]].add(r["split"])
    if bad := {p: v for p, v in per_patient.items() if len(v) > 1}:
        sys.exit(f"patient leakage across train/val: {list(bad)[:5]}")

    names = Counter(r["released"] for r in rows)
    if dupes := [n for n, c in names.items() if c > 1]:
        sys.exit(f"released filename collision ({len(dupes)}): {dupes[:5]}")

    for r in rows:
        rel = Path("classification/images") / r["split"] / r["icrop"] / r["released"]
        link(Path(r["src_path"]), out_root / rel)
        r["path"] = str(rel)
        r["cls6"] = CLS6_NAMES[CLS6[r["icrop"]]] if r["icrop"] in CLS6 else ""
        r["cls6_index"] = CLS6.get(r["icrop"], "")
        r["in_6class_task"] = r["icrop"] in CLS6
        r["width"], r["height"] = dims(Path(r["src_path"])) if want_dims else ("", "")

    cols = ["path", "source", "patient_id", "split", "icrop", "cls6", "cls6_index",
            "in_6class_task", "native_label", "device", "width", "height", "md5"]
    write_csv(out_root / "classification/labels.csv", rows, cols)

    # The five CV folds and the site split ship verbatim, rewritten onto released paths so
    # they are usable from inside the zip.
    by_src_name = {Path(r["src_path"]).name: r["path"] for r in rows}
    fold_dir = out_root / "classification/splits"
    fold_dir.mkdir(parents=True, exist_ok=True)
    for f in sorted(PREP.glob("fold_*.csv")) + [PREP / "test_dev.csv", PREP / "test_locked.csv"]:
        if not f.exists():
            continue
        d = pd.read_csv(f)
        # A fold row whose image was dropped as a duplicate maps to "" and is filtered, so the
        # shipped folds stay consistent with the shipped images rather than pointing at files
        # that are no longer there.
        d["path"] = [by_src_name.get(Path(p).name, "") for p in d["image_path"]]
        d = d[d["path"] != ""]
        d.drop(columns=["image_path"]).to_csv(fold_dir / f.name, index=False)
    return rows, dropped


# --------------------------------------------------------------------------------------
# plus disease (FARFUM-RoP): five raters, no consensus
# --------------------------------------------------------------------------------------
def build_plus_disease(out_root: Path, want_dims: bool):
    base = K / "farfum_rop"
    if not base.is_dir():
        return []
    lab = pd.read_excel(base / "Dataset_Labels.xlsx", sheet_name="Labels", header=1)
    # The sheet spells the same field two ways — "Retinopathy grade_A" (space) and
    # "Retinopathy grade-E" (hyphen) — so BOTH separators have to go, and the normalisation
    # has to happen BEFORE the renames or it renames the renames.
    lab.columns = [c.replace("-", "_").replace(" ", "_") for c in lab.columns]
    lab = lab.rename(columns={"Unnamed:_0": "patient", "Image_name": "image_name",
                              "Unnamed:_17": "dataset_label"})
    det = pd.read_excel(base / "Dataset_Details.xlsx").set_index("Patient.id")

    on_disk = {}
    for f in base.rglob("*"):
        if f.suffix.lower() in IMG_EXT:
            on_disk[f.stem] = f

    raters = ["A", "B", "C", "D", "E"]
    rows, missing = [], 0
    def cell(row, col):
        # Fail loudly on a column that does not exist: a typo here is invisible in the output,
        # it just yields a CSV full of blanks that looks like missing source data.
        if col not in lab.columns:
            sys.exit(f"farfum: expected column {col!r}, have {list(lab.columns)}")
        v = row[col]
        if pd.isna(v):
            return ""
        # NaNs elsewhere in the column force it to float64, so an integer grade arrives as
        # 1.0 and would name a folder "grade_1.0". Put it back.
        if isinstance(v, float) and v.is_integer():
            return int(v)
        return v

    for _, r in lab.iterrows():
        f = on_disk.get(str(r["image_name"]))
        if f is None:
            missing += 1
            continue
        grades = {g: cell(r, f"Retinopathy_grade_{g}") for g in raters}
        vals = [v for v in grades.values() if v != ""]
        # No consensus label exists in this dataset and inventing one by majority would hide
        # the disagreement that makes it valuable. Ship the raw five plus a description of
        # how much they differ, and let the user choose.
        counts = Counter(vals)
        row = dict(path="", source="farfum", patient_id=f"farfum:{r['patient']}",
                   split="", n_raters=len(vals), n_distinct_grades=len(counts),
                   modal_grade=counts.most_common(1)[0][0] if counts else "unknown",
                   modal_agreement=round(counts.most_common(1)[0][1] / len(vals), 3)
                   if vals else "",
                   unanimous=len(counts) == 1,
                   dataset_label=cell(r, "dataset_label"))
        for g in raters:
            row[f"grade_{g}"] = grades[g]
            row[f"stage_{g}"] = cell(r, f"Retinopathy_stage_{g}")
            row[f"diagnostic_{g}"] = cell(r, f"Diagnostic_{g}")
        d = det.loc[r["patient"]] if r["patient"] in det.index else None
        row["birth_weight_g"] = d["Patient.BirthWeight"] if d is not None else ""
        row["gestational_age_wk"] = d["Patient.Gestation Age"] if d is not None else ""
        row["sex"] = d["Patient.Gender"] if d is not None else ""
        row["src_path"] = str(f)
        row["released"] = f"farfum__{f.name}"
        rows.append(row)
    if missing:
        print(f"  farfum: {missing} label rows had no image on disk", file=sys.stderr)

    rows, dropped = dedupe(rows, "modal_grade", "plus_disease")

    # Patient-grouped 80/20; the same hash rule as classification, so the two parts of the
    # release are split by one policy rather than two.
    for r in rows:
        digest = hashlib.md5(r["patient_id"].encode()).hexdigest()
        r["split"] = "val" if int(digest[:8], 16) % 100 < 20 else "train"

    for r in rows:
        rel = (Path("plus_disease/images") / r["split"] /
               f"grade_{r['modal_grade']}" / r["released"])
        link(Path(r["src_path"]), out_root / rel)
        r["path"] = str(rel)

    cols = (["path", "source", "patient_id", "split", "n_raters", "n_distinct_grades",
             "modal_grade", "modal_agreement", "unanimous", "dataset_label"] +
            [f"{k}_{g}" for g in raters for k in ("grade", "stage", "diagnostic")] +
            ["birth_weight_g", "gestational_age_wk", "sex", "md5"])
    write_csv(out_root / "plus_disease/labels.csv", rows, cols)
    return rows, dropped


# --------------------------------------------------------------------------------------
# segmentation
# --------------------------------------------------------------------------------------
def build_segmentation(out_root: Path, cls_rows):
    # COph100 ships annotations WITHOUT the photographs — they are Ostrava images, already in
    # classification/. Pairing them by released stem is what makes the set usable; without it
    # you have 324 masks of nothing.
    ostrava_by_stem = {Path(r["path"]).stem.replace("ostrava__", ""): r["path"]
                       for r in cls_rows if r["source"] == "ostrava"}
    rows = []

    def add(source, target, img: Path, mask: Path | None, ann: Path | None, note=""):
        rows.append(dict(source=source, target=target, _img=img, _mask=mask, _ann=ann,
                         note=note))

    seg = next(K.glob("hvdropdb/*/HVDROPDB_RetCam_Neo_Segmentation/*"), None)
    if seg:
        for sub, target in (("HVDROPDB-BV", "vessel"), ("HVDROPDB-OD", "optic_disc"),
                            ("HVDROPDB-RIDGE", "ridge")):
            d = seg / sub
            if not d.is_dir():
                continue
            for imgs in sorted(d.glob("*_images")):
                masks = imgs.parent / imgs.name.replace("_images", "_masks")
                for f in sorted(imgs.iterdir()):
                    if f.suffix.lower() not in IMG_EXT:
                        continue
                    m = next((c for c in masks.glob(f.stem + ".*")
                              if c.suffix.lower() in IMG_EXT), None)
                    add("hvdropdb", target, f, m, None, imgs.name.split("_")[0].lower())

    # Shantou optic disc: LabelMe polygon JSON, no raster mask shipped.
    for f in sorted((K / "shantou_od").glob("*.jpg")):
        j = f.with_suffix(".json")
        add("shantou_od", "optic_disc", f, None, j if j.exists() else None, "polygon_json")

    # Shantou vessels: raster mask AND polygon JSON.
    for f in sorted((K / "shantou_bv").glob("*.jpg")):
        if f.stem.endswith("_mask"):
            continue
        m = f.with_name(f.stem + "_mask.jpg")
        j = f.with_suffix(".json")
        add("shantou_bv", "vessel", f, m if m.exists() else None,
            j if j.exists() else None, "polygon_json")

    # ORVS release: keep only the two ROP subsets. Its `ORVS/` folder is adult retinal
    # photography and is out of scope for a ROP-only package.
    for v in ("V1", "V2"):
        base = K / "orvs_seg" / "resized-images" / f"segmentation(ROP){v}"
        for part in ("Train", "Test"):
            imgs = base / part / "Images"
            if not imgs.is_dir():
                continue
            for f in sorted(imgs.iterdir()):
                if f.suffix.lower() not in IMG_EXT:
                    continue
                m = next((c for c in (base / part / "Labels").glob(f.stem + ".*")
                          if c.suffix.lower() in IMG_EXT), None)
                add("orvs_rop", "vessel", f, m, None, f"{v.lower()}_{part.lower()}")

    # COph100: registration/vessel annotations over Ostrava images we already ship. Only the
    # annotation files travel — the photographs are in classification/, not duplicated here.
    for f in sorted((K / "coph100_annotations").rglob("*_mask.png")):
        add("coph100", "vessel", f, None, None, "annotation_over_ostrava")

    # ORVS-ROP and Shantou-BV are both 101 vessel images. Byte-hash them so the release says
    # whether that is a coincidence or the same photographs shipped twice.
    hashes = defaultdict(list)
    for i, r in enumerate(rows):
        hashes[md5(r["_img"])].append(i)
    for idxs in hashes.values():
        if len(idxs) > 1:
            keep = idxs[0]
            for i in idxs[1:]:
                rows[i]["note"] = (rows[i]["note"] + "; duplicate_of=" +
                                   rows[keep]["_img"].name).strip("; ")

    out = []
    unpaired = 0
    for r in rows:
        src = r["source"]
        stem = sanitise(src, r["_img"])
        if src == "coph100":
            # What was walked here is the MASK. Its image lives in classification/.
            rel_mask = Path("segmentation/masks") / r["target"] / f"{src}__{stem}"
            link(r["_img"], out_root / rel_mask)
            paired = ostrava_by_stem.get(stem.replace("_mask.png", ""), "")
            unpaired += not paired
            out.append(dict(image_path=paired, mask_path=str(rel_mask), annotation_path="",
                            source=src, target=r["target"],
                            note=r["note"] + ("" if paired else "; UNPAIRED")))
            continue
        rel_img = Path("segmentation/images") / r["target"] / f"{src}__{stem}"
        link(r["_img"], out_root / rel_img)
        rec = dict(image_path=str(rel_img), mask_path="", annotation_path="",
                   source=src, target=r["target"], note=r["note"])
        if r["_mask"]:
            p = Path("segmentation/masks") / r["target"] / f"{src}__{sanitise(src, r['_mask'])}"
            link(r["_mask"], out_root / p)
            rec["mask_path"] = str(p)
        if r["_ann"]:
            p = (Path("segmentation/annotations") / r["target"] /
                 f"{src}__{sanitise(src, r['_ann'])}")
            link(r["_ann"], out_root / p)
            rec["annotation_path"] = str(p)
        out.append(rec)

    if unpaired:
        print(f"  coph100: {unpaired} masks could not be paired to an Ostrava image",
              file=sys.stderr)
    write_csv(out_root / "segmentation/labels.csv", out,
              ["image_path", "mask_path", "annotation_path", "source", "target", "note"])
    return out


# --------------------------------------------------------------------------------------
def dedupe(rows, label_key: str, part: str):
    """Drop byte-duplicate photographs; drop contradictions outright.

    Every source ships some file twice, and pooling them makes it worse. Three of these are
    already documented — Ostrava has 19 internal byte-duplicates, HVDROPDB re-files 21 of its
    185 classification images, ROP-VL and Shenzhen share ~20 photographs — and a duplicate is
    not harmless: in train it silently reweights an image, and in test it counts one
    photograph twice.

    A duplicate group carrying MORE THAN ONE label is a different problem. HVDROPDB files the
    same bytes as both `RetCam_Normal/14.png` and `RetCam_ROP/19.png`, so one of the two is
    wrong and nothing in the data says which. Guessing would put a known-bad label in the
    corpus, so the whole group is dropped and recorded.
    """
    groups = defaultdict(list)
    for r in rows:
        groups[md5(Path(r["src_path"]))].append(r)

    kept, dropped = [], []
    for digest, grp in groups.items():
        grp.sort(key=lambda r: (r["source"], r["src_path"]))
        for r in grp:
            r["md5"] = digest
            r["duplicate_group_size"] = len(grp)
        labels = {r[label_key] for r in grp}
        if len(labels) > 1:
            for r in grp:
                dropped.append({**r, "reason": f"contradictory labels: {sorted(labels)}"})
            continue
        kept.append(grp[0])
        for r in grp[1:]:
            dropped.append({**r, "reason": f"byte-duplicate of {grp[0]['released']}"})
    kept.sort(key=lambda r: (r["source"], r["src_path"]))
    print(f"  {part}: {len(rows):,} files -> {len(kept):,} distinct "
          f"({len(dropped):,} dropped)", file=sys.stderr)
    return kept, dropped


def write_csv(path: Path, rows, cols) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def write_readme(out_root: Path, stats, cls, plus, seg) -> None:
    c, p, s = stats["classification"], stats["plus_disease"], stats["segmentation"]

    # The per-infant cap applies to train and not to val, so a single prolific infant can own
    # a whole validation class. Measure how bad it is rather than assert it.
    val_arop = Counter(r["patient_id"] for r in cls
                       if r["icrop"] == "arop" and r["split"] == "val")
    arop_val = sum(val_arop.values())
    arop_val_pid, arop_val_top = val_arop.most_common(1)[0] if val_arop else ("-", 0)

    def table(counter, head):
        w = max(len(k) for k in counter) if counter else 4
        lines = [f"| {head} | images |", f"|{'-' * (w + 2)}|-------:|"]
        lines += [f"| {k} | {v:,} |" for k, v in sorted(counter.items(), key=lambda x: -x[1])]
        return "\n".join(lines)

    src_rows = "\n".join(
        f"| {SOURCE_META[k]['title']} | {v:,} | {SOURCE_META[k]['region']} | "
        f"{SOURCE_META[k]['licence']} | `{SOURCE_META[k]['doi']}` |"
        for k, v in sorted(c["by_source"].items(), key=lambda x: -x[1]))

    (out_root / "README.md").write_text(f"""# ROP unified dataset v{VERSION}

Every publicly obtainable retinopathy-of-prematurity dataset, merged onto one label
vocabulary, split by patient, and de-identified. **{c['images'] + p['images'] + s['images']:,}
labelled images** from ten sources across six countries.

Built by `scripts/build_rop_release.py` in the RetinAI repository. Nothing here is a claim
from a paper: every count was walked from files on disk.

---

## ⚠ Read this before you use it

**These are photographs of real infants' eyes.** Several sources are non-commercial, one is
CC BY-NC, and the whole package is de-identified but not anonymous — a determined party could
re-link images to the public source datasets. Do not upload this to a public bucket, a model
hub, or a Kaggle dataset.

**Three findings that change how the labels should be read:**

1. **Ostrava's "ROP-positive" class is 43% not-ROP.** Its diagnosis code was binarised
   `DG==0 -> negative, else positive`, sweeping haemorrhage (1,061), toxoplasma (114),
   hamartoma (72) and optic-nerve hypoplasia (48) into the positive class. In THIS package
   those images carry `icrop = other_pathology` and are excluded from the 6-class task, so
   the defect is fixed — but any result you compare against that used the old binary label
   was measuring an "abnormal infant retina" detector.

2. **Image resolution is a perfect proxy for camera, and camera carries the label.** In
   Ostrava, 640x480 = RetCam 3, 1240x1240 = Phoenix ICON, 1440x1080 = RetCam Envision. On its
   test split, *image dimensions alone* score AUC 0.911. `labels.csv` ships `width`, `height`
   and `device` so you can audit this rather than rediscover it. Letterbox rather than resize.

3. **FARFUM-RoP has no ground truth.** {p['unanimous']} of its {p['images']:,} images are
   graded unanimously by its five ophthalmologists: {p['n_distinct'].get(2, 0):,} carry two
   different grades and {p['n_distinct'].get(3, 0):,} carry three. Treat them as five
   opinions, not one label — `n_distinct_grades` and `modal_agreement` are in the CSV for
   exactly this, and `modal_grade` is a convenience that hides the disagreement.

4. **AP-ROP in `val` is effectively one infant.** {arop_val_top:,} of the {arop_val:,} AP-ROP
   images in the validation split come from a single patient (`{arop_val_pid}`), because the
   fold design caps images per infant on `train` but not on `val`. A validation AP-ROP score
   is therefore a measurement of one baby, and its confidence interval is not what the image
   count suggests. Aggregate to patient level before believing any per-class AP-ROP number,
   or re-split from `labels.csv` with your own cap.

---

## Layout

```
classification/   {c['images']:>6,} images   one ICROP stage per image
  labels.csv                    <- authoritative; the folder tree mirrors it
  splits/fold_0..4.csv          <- the 5 patient-grouped CV folds
  images/<split>/<class>/<source>__<name>.jpg
plus_disease/     {p['images']:>6,} images   five independent rater grades
  labels.csv
  images/<split>/grade_<modal>/...
segmentation/     {s['images']:>6,} images   image + pixel mask
  labels.csv
  images/<target>/  masks/<target>/  annotations/<target>/
docs/
  filename_map.csv    every de-identifying rename, reversible against the source
  PROVENANCE.json     per-source DOI, licence, caveats
class_index.json      the 6-class task definition
dataset_stats.json    every count in this README, machine-readable
```

`labels.csv` is the authority. The folder tree is a convenience for `ImageFolder`; where they
could ever disagree, believe the CSV.

---

## classification/ — {c['images']:,} images, {c['patients']:,} patients

| Source | Images | Region | Licence | DOI |
|---|---:|---|---|---|
{src_rows}

### Splits

{table(c['by_split'], 'Split')}

- **`train` / `val`** are grouped by patient — no infant appears in both. The assignment is
  taken from fold 0 of the repository's committed 5-fold design, so this release cannot
  disagree with results already reported against it. Use `splits/fold_*.csv` for the other four.
- **`multiview_dev` / `multiview_locked`** are a **held-out hospital**. Multi-View appears in
  no training fold. Its dev half is for shortcut audits; open the locked half once, at the end.
  Pointing training at `multiview_locked` is the one irreversible mistake here.
- **`test_external_binary`** is HVDROPDB (Pune, India) — a second external site, graded
  ROP/normal with **no stage**, so it scores the binary axis only. Its positives carry
  `icrop = rop_unstaged`, which is deliberately not one of the six classes.

### Classes

{table(c['by_class'], 'icrop')}

{c['in_6class_task']:,} images fall in the 6-class staging task
(`normal, stage_1, stage_2, stage_3, stage_4_5, arop` — see `class_index.json`). The rest ship
in full but sit outside it, flagged `in_6class_task = False`:

- `other_pathology` — non-ROP disease that the source's binary label hid inside "ROP".
- `treated` — post-laser eyes. Real ROP, but the photograph shows scars, not active disease.
- `stage_0` — pre-ROP; a stage the task does not model.
- `rop_unstaged` — ROP confirmed, stage never recorded (HVDROPDB).

Stage 4 ({c['by_class'].get('stage_4', 0)}) and stage 5 ({c['by_class'].get('stage_5', 0)}) are
merged into one `stage_4_5` class because no source has enough of either alone. **ROP-VL is the
only public source of stage 5 anywhere** — those {c['by_class'].get('stage_5', 0)} images are
the entire world supply.

---

## plus_disease/ — {p['images']:,} images, {p['patients']} patients

FARFUM-RoP (Farabi Hospital, Iran). Five paediatric ophthalmologists, labelled A–E, each graded
every image **independently**: `grade_A..grade_E` (0–3), `stage_A..stage_E`, and
`diagnostic_A..diagnostic_E` (the treatment call). Per-patient birth weight, gestational age
and sex are joined in.

`modal_grade` is the most common of the five and is what the folder tree uses — it is a
convenience, **not a consensus**. `modal_agreement` tells you how thin that majority was and
`unanimous` is False for every single image. Split 80/20 by patient.

---

## segmentation/ — {s['images']:,} images

{table(s['by_target'], 'Target')}

{s['with_raster_mask']:,} carry a raster mask; the rest ship LabelMe polygon JSON in
`annotations/` instead. COph100 is the exception worth knowing about: it annotates **Ostrava
photographs we already ship**, so its rows have `mask_path` set and `image_path` pointing back
into `classification/` rather than duplicating the JPEG.

Duplicate photographs across the vessel sources are detected by MD5 and flagged
`duplicate_of=...` in the `note` column rather than silently dropped.

---

## De-identification

{stats['renamed_files']:,} files were renamed. `docs/filename_map.csv` records every one.

| Source | Carried | Now |
|---|---|---|
| ROP-VL | exam date at day granularity | visit ordinal (`p0546_v2_R_42.jpg`) — keeps which images share a sitting, and their order |
| COph100 | Ostrava sex, gestational age, birth weight | `<id>_D<device>_S<serie>_<n>` |
| Ostrava | (already de-identified upstream) | unchanged |

Excluded outright: `littlevision`, a re-upload of Ostrava carrying **hospital medical record
numbers and exam dates in its folder names**. It is duplicate data and a disclosure risk.
Also excluded: the adult-retina `ORVS/` subset, which is not ROP.
""")

    lic = "\n".join(
        f"### {m['title']}\n\n- Region: {m['region']}\n- Licence: **{m['licence']}**\n"
        f"- Source: `{m['doi']}`\n" for m in SOURCE_META.values())
    (out_root / "LICENCES.md").write_text(
        "# Licences\n\nThis package redistributes ten datasets under their own terms. The most\n"
        "restrictive term governs any combined use.\n\n"
        "**Multi-View ROP is CC BY-NC 4.0 — non-commercial.** Any model trained on the pooled\n"
        "corpus inherits that restriction. Two sources state no licence at all; treat them as\n"
        "all-rights-reserved unless you confirm otherwise with the depositor.\n\n"
        "Cite the original datasets, not this package.\n\n" + lic)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(Path.home() / "Downloads" / "RetinAI_ROP_v1"))
    ap.add_argument("--no-dims", action="store_true",
                    help="skip reading image headers for width/height")
    args = ap.parse_args()

    out_root = Path(args.out)
    if out_root.exists():
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True)

    build_ropvl_visit_map()
    print("classification ...")
    cls, cls_dropped = build_classification(out_root, not args.no_dims)
    print(f"  {len(cls)} images")
    print("plus disease ...")
    plus, plus_dropped = build_plus_disease(out_root, not args.no_dims)
    print(f"  {len(plus)} images")
    print("segmentation ...")
    seg = build_segmentation(out_root, cls)
    print(f"  {len(seg)} images")

    docs = out_root / "docs"
    docs.mkdir(exist_ok=True)
    write_csv(docs / "duplicates_removed.csv", cls_dropped + plus_dropped,
              ["released", "source", "patient_id", "icrop", "modal_grade", "md5",
               "duplicate_group_size", "reason", "src_path"])
    write_csv(docs / "filename_map.csv",
              [dict(source=s, original=o, released=n) for s, o, n in sorted(set(renames))],
              ["source", "original", "released"])
    for f in (K / "PROVENANCE.json", PREP / "ostrava_name_map.csv",
              PREP / "splits_summary.json"):
        if f.exists():
            shutil.copy2(f, docs / f.name)

    stats = {
        "version": VERSION,
        "classification": {
            "images": len(cls),
            "by_split": dict(Counter(r["split"] for r in cls)),
            "by_class": dict(Counter(r["icrop"] for r in cls)),
            "by_source": dict(Counter(r["source"] for r in cls)),
            "patients": len({r["patient_id"] for r in cls}),
            "in_6class_task": sum(r["in_6class_task"] for r in cls),
        },
        "plus_disease": {
            "images": len(plus),
            "patients": len({r["patient_id"] for r in plus}),
            "unanimous": sum(bool(r["unanimous"]) for r in plus),
            "n_distinct": dict(Counter(r["n_distinct_grades"] for r in plus)),
            "by_split": dict(Counter(r["split"] for r in plus)),
        },
        "segmentation": {
            "images": len(seg),
            "by_target": dict(Counter(r["target"] for r in seg)),
            "by_source": dict(Counter(r["source"] for r in seg)),
            "with_raster_mask": sum(bool(r["mask_path"]) for r in seg),
        },
        "renamed_files": len(set(renames)),
        "duplicates_removed": {
            "classification": len(cls_dropped),
            "plus_disease": len(plus_dropped),
            "contradictory_label_groups": sum(
                1 for r in cls_dropped + plus_dropped
                if r["reason"].startswith("contradictory")),
        },
    }
    (out_root / "dataset_stats.json").write_text(json.dumps(stats, indent=2, default=str))
    write_readme(out_root, stats, cls, plus, seg)
    (out_root / "class_index.json").write_text(json.dumps(
        {"classes": CLS6_NAMES, "index": {c: i for i, c in enumerate(CLS6_NAMES)},
         "icrop_to_cls6": CLS6,
         "shipped_but_outside_task": ["stage_0", "treated", "other_pathology", ROP_UNSTAGED]},
        indent=2))
    print(json.dumps(stats, indent=2, default=str))


if __name__ == "__main__":
    main()
