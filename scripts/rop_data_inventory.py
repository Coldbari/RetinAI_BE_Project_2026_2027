"""Inventory every ROP dataset under data/rop/ and regenerate its README + PROVENANCE.json.

Counts come from walking the filesystem. Provenance (DOI, licence, citation, what the labels
mean, what is wrong with them) is curated below, because no amount of file-walking recovers
"the five graders never agree".

    .venv/bin/python scripts/rop_data_inventory.py

Writes data/rop/README.md and data/rop/PROVENANCE.json — the only two tracked files under
data/, per .gitignore. Everything else there is images and stays out of git.
"""
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "rop"
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

# folder -> curated provenance. `order` drives presentation.
META = {
    "ostrava": dict(
        order=1, title="Retinal Image Dataset of Infants and ROP", region="Ostrava, Czechia",
        doi="figshare collection 6626162", licence="CC0",
        mirror="Kaggle jananowakova/retinal-image-dataset-of-infants-and-rop",
        labels="ICROP stage 0-3, AP-ROP, treated (st.p. ROP), 4 non-ROP diagnoses; plus-form; 3 cameras",
        role="TRAINING - this is what the shipped model was built on",
        note="Symlink to ../../dataset_split (the prepared split). PRIVATE infant photographs. "
             "Stage labels are recoverable only by joining docs/infant_retinal_database_info.csv "
             "on (ID, serie, device). 43% of its 'ROP-positive' class is NOT ROP.",
        caveat="Image resolution is a perfect 1:1 proxy for camera, and camera carries most of "
               "the label signal: dimensions alone score AUC 0.911 on its test split."),
    "rop_vl": dict(
        order=2, title="ROP-VL", region="Shenzhen Eye Hospital, China",
        doi="10.6084/m9.figshare.30143461", licence="CC BY 4.0",
        mirror="Kaggle nazimriyadh1/rop-vl (unattributed re-upload - prefer figshare)",
        labels="8 classes: normal, Stage 1-5, A-ROP, laser-treated. Per-infant GA, birth weight, "
               "plurality, delivery mode. Per-class clinical text (vision-language).",
        role="THE key acquisition - only public source of Stage 4 and Stage 5 anywhere",
        note="Overlaps Shenzhen by only ~1-2% of images (20-22 shared photographs) despite the "
             "same hospital and 6 shared authors. Where they overlap ROP-VL holds the "
             "higher-resolution original. No zone axis, no plus axis.",
        caveat="Filenames embed exam dates at day granularity - a date-of-service identifier. "
               "Do not propagate those filenames into anything public."),
    "farfum_rop": dict(
        order=3, title="FARFUM-RoP", region="Farabi Hospital, Iran",
        doi="figshare collection 6721269", licence="CC BY 4.0", mirror="-",
        labels="Retinopathy grade + ICROP stage, recorded INDEPENDENTLY by five paediatric "
               "ophthalmologists (A-E), plus per-patient details.",
        role="The plus/pre-plus axis, and the only inter-rater data we have",
        note="Ships as 68 per-patient RAR5 archives. p7zip cannot decode RAR5 and will silently "
             "write zero-byte files - use `unar`.",
        caveat="ZERO of its 1,533 images are graded unanimously. Every image has 2 or 3 distinct "
               "grades among the 5 graders, and the split is systematic (C and E consistently "
               "more conservative). Treat as soft labels, never as ground truth."),
    "multiview": dict(
        order=4, title="Multi-View ROP", region="China",
        doi="HuggingFace lijuanliao/ROP-DATASET", licence="CC BY-NC 4.0", mirror="-",
        labels="Per-image ICROP stage (1-4, A-ROP, Normal) + 6 acquisition views "
               "(L/R x temporal/posterior-pole/nasal) + neonatal risk factors "
               "(oxygen therapy, ventilation, sepsis, transfusion).",
        role="The only source with acquisition-view labels - directly attacks the view confound",
        note="Discoverable ONLY via HuggingFace full-text search; absent from Kaggle, figshare "
             "and ordinary web search.",
        caveat="NON-COMMERCIAL licence - do not mirror publicly. Its zone/plus/treatment columns "
               "are nearly empty (zone 13 of 232 patients, plus 17, treatment 14); only the "
               "per-image stage labels are complete."),
    "shenzhen": dict(
        order=5, title="Shenzhen ROP (Zhao 2024)", region="Shenzhen Eye Hospital, China",
        doi="10.6084/m9.figshare.25514449", licence="CC BY 4.0",
        mirror="Kaggle errorrific/fundus-image-dataset-to-retinopathy-of-prematurity",
        labels="Normal, Stage 1-3, laser scars. zip information.xlsx maps image -> infant -> eye.",
        role="FIRST DATASET DOWNLOADED in this effort - the original external-validation target",
        note="All 512x512. Largely superseded by ROP-VL (same hospital, higher resolution), but "
             "98% of its images are NOT in ROP-VL, so it still contributes.",
        caveat="3 images are filed under two different stage folders at once. Binary labels are "
               "unaffected; staging labels contradict for those 3."),
    "hvdropdb": dict(
        order=6, title="HVDROPDB", region="H.V. Desai Eye Hospital, Pune, India",
        doi="10.17632/xw5xc7xrmp.3 (Mendeley)", licence="CC BY 4.0",
        mirror="Kaggle tonimaislamlamia/hvdropdb-datasets",
        labels="Binary ROP/Normal classification, plus segmentation masks for RIDGE "
               "(demarcation line), blood vessels and optic disc. RetCam and Neo cameras.",
        role="The ridge masks - the feature that defines ICROP stages 1 and 2",
        note="785 files but only 329 UNIQUE fundus images: 21 classification files are byte "
             "duplicates and 97 images appear in both the classification and segmentation subsets.",
        caveat="Every file is named 1.png..50.png, so basenames collide across all 16 folders. "
               "Never flatten this directory."),
    "shantou_od": dict(
        order=7, title="ROP optic disc segmentation (Shantou)", region="Shantou, China",
        doi="10.17632/tv3b2bt6dy.2 (Mendeley)", licence="CC BY 4.0", mirror="-",
        labels="Optic disc segmentation, LabelMe JSON per image.", role="Segmentation only",
        note="UUID.N filenames, same convention as Multi-View and FARFUM.", caveat="-"),
    "shantou_bv": dict(
        order=8, title="ROP retinal vessel segmentation (Shantou)", region="Shantou, China",
        doi="10.17632/prcy36j53v.2 (Mendeley)", licence="CC BY 4.0", mirror="-",
        labels="Retinal blood vessel segmentation: image + LabelMe JSON + binary mask.",
        role="Segmentation only - vessel tortuosity is the substrate of plus disease",
        note="Companion release to shantou_od, same authors and date.", caveat="-"),
    "orvs_seg": dict(
        order=9, title="ORVS + ROP vessel segmentation", region="mixed",
        doi="Kaggle ojaskundlia/orvs-rop-1-2", licence="unstated", mirror="-",
        labels="Vessel segmentation image/label pairs, ROP V1 and V2 plus the adult ORVS set.",
        role="Small segmentation supplement", note="V1 and V2 are two mask revisions of one "
             "image set.", caveat="Licence is not stated anywhere - do not redistribute."),
    "ropstages_sample": dict(
        order=10, title="ROP stages (reviewed) sample", region="unstated",
        doi="Kaggle solennollivier/ropstages-reviewed", licence="unstated", mirror="-",
        labels="Stage 1-3 and NoROP encoded in the filename, with a patient number.",
        role="Small independent stage-labelled sample",
        note="Folder is literally named NewROPDataset_Sample_justtotry - it is a sample, not a "
             "release.", caveat="Provenance and licence both unknown. Do not train on this."),
    "coph100_annotations": dict(
        order=11, title="COph100", region="derived from Ostrava",
        doi="10.6084/m9.figshare.27061084", licence="CC BY", mirror="-",
        labels="Registration control points (LabelMe JSON) + vessel masks + overlays.",
        role="ANNOTATIONS ONLY - adds no new photographs",
        note="Ships masks and overlays keyed to Ostrava filenames, which we already hold. Its "
             "filenames preserve the TRUE Ostrava DG codes.",
        caveat="Do not count these as additional images."),
    "_redundant": dict(
        order=99, title="Redundant re-uploads (quarantined)", region="-",
        doi="-", licence="-", mirror="-", labels="-",
        role="QUARANTINE - safe to delete",
        note="littlevision (Kaggle hemanathrvh13996/rop-dataset-for-little-vision) re-ships "
             "Ostrava byte-identically AND FARFUM re-encoded (all 1,533 filenames match). "
             "Kept only so nobody re-downloads it to find out.",
        caveat="Its folder names contain medical record numbers and exam dates."),
}


def scan(d):
    imgs = masks = other = 0
    exts = {}
    for dp, _, files in os.walk(d):
        for f in files:
            if f.startswith("."):
                continue
            e = os.path.splitext(f)[1].lower()
            exts[e] = exts.get(e, 0) + 1
            if e in IMG_EXT:
                p = os.path.join(dp, f).lower()
                if "mask" in p or "/labels/" in p or "overlay" in p:
                    masks += 1
                else:
                    imgs += 1
            else:
                other += 1
    return imgs, masks, other, exts


def du(d):
    t = 0
    for dp, _, files in os.walk(d):
        for f in files:
            try:
                t += os.path.getsize(os.path.join(dp, f))
            except OSError:
                pass
    return t


def main():
    if not DATA.is_dir():
        print(f"missing {DATA}", file=sys.stderr)
        return 1
    entries = []
    for name in sorted(os.listdir(DATA)):
        p = DATA / name
        if not p.is_dir() or name.startswith("."):
            continue
        m = META.get(name, dict(order=50, title=name, region="?", doi="?", licence="?",
                                mirror="-", labels="?", role="?", note="", caveat=""))
        imgs, masks, other, exts = scan(p)
        entries.append({"folder": name, "images": imgs, "masks_or_overlays": masks,
                        "other_files": other, "bytes": du(p),
                        "extensions": dict(sorted(exts.items(), key=lambda kv: -kv[1])[:6]),
                        **m})
    entries.sort(key=lambda e: e["order"])

    total_img = sum(e["images"] for e in entries if e["folder"] != "_redundant")
    total_b = sum(e["bytes"] for e in entries)

    (DATA / "PROVENANCE.json").write_text(json.dumps(
        {"generated_by": "scripts/rop_data_inventory.py",
         "total_images_excluding_redundant": total_img,
         "total_bytes": total_b, "datasets": entries}, indent=2) + "\n")

    L = []
    A = L.append
    A("# ROP corpus\n")
    A("Every public retinopathy-of-prematurity dataset we could obtain, in one place.\n")
    A("**Regenerate this file with `.venv/bin/python scripts/rop_data_inventory.py`** — "
      "counts below are walked from disk, provenance is curated in that script.\n")
    A(f"**{total_img:,} image files** across "
      f"{len([e for e in entries if e['folder'] != '_redundant'])} datasets, "
      f"{total_b / 1e9:.1f} GB on disk.\n")
    A("> That is a count of FILES, not of distinct photographs. Deduplicated by md5 and "
      "filename, the primary sources hold **~12,500 distinct images**: HVDROPDB ships 485 "
      "non-mask files but only 329 unique photographs, Ostrava has 19 internal byte-duplicates, "
      "and ~20 Shenzhen images re-appear in ROP-VL at higher resolution. "
      "`results/prepared/rop_staging/manifest.csv` carries the deduplicated set.\n")
    A("Everything here is gitignored except this file and `PROVENANCE.json`. It sits beside "
      "PRIVATE infant retinal photographs; several sets carry patient identifiers in filenames. "
      "**Do not commit, mirror or upload any of it without re-reading the caveats.**\n")
    A("## Datasets\n")
    A("| # | Dataset | Images | Masks | Size | Region | Licence |")
    A("|---|---|---:|---:|---:|---|---|")
    for e in entries:
        if e["folder"] == "_redundant":
            continue
        A(f"| {e['order']} | **{e['title']}** (`{e['folder']}`) | {e['images']:,} | "
          f"{e['masks_or_overlays']:,} | {e['bytes'] / 1e6:.0f} MB | {e['region']} | "
          f"{e['licence']} |")
    A("")
    for e in entries:
        A(f"### {e['order']}. {e['title']}  — `{e['folder']}/`\n")
        A(f"- **Role:** {e['role']}")
        A(f"- **Images:** {e['images']:,}" +
          (f" (+{e['masks_or_overlays']:,} masks/overlays)" if e["masks_or_overlays"] else ""))
        A(f"- **Source:** {e['doi']}  |  **Licence:** {e['licence']}")
        if e["mirror"] and e["mirror"] != "-":
            A(f"- **Mirror:** {e['mirror']}")
        A(f"- **Labels:** {e['labels']}")
        if e["note"]:
            A(f"- **Note:** {e['note']}")
        if e["caveat"] and e["caveat"] != "-":
            A(f"- **⚠ Caveat:** {e['caveat']}")
        A("")
    A("## Derived\n")
    A("`scripts/build_rop_staging_manifest.py` → `results/prepared/rop_staging/manifest.csv` — "
      "a unified, patient-grouped ICROP staging manifest across Ostrava, Shenzhen, ROP-VL and "
      "Multi-View.\n")
    A("`scripts/rop_recover_staging.py` → `results/rop/staging_composition.json` — what the "
      "current binary ROP label is actually made of.\n")
    (DATA / "README.md").write_text("\n".join(L) + "\n")

    print(f"{total_img:,} images, {total_b / 1e9:.2f} GB across {len(entries)} folders")
    for e in entries:
        print(f"  {e['folder']:<22} {e['images']:>6} img  {e['masks_or_overlays']:>5} mask  "
              f"{e['bytes'] / 1e6:>8.0f} MB")
    print(f"\n-> {DATA / 'README.md'}\n-> {DATA / 'PROVENANCE.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
