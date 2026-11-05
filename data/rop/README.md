# ROP corpus

Every public retinopathy-of-prematurity dataset we could obtain, in one place.

**Regenerate this file with `.venv/bin/python scripts/rop_data_inventory.py`** — counts below are walked from disk, provenance is curated in that script.

**13,037 image files** across 11 datasets, 14.5 GB on disk.

> That is a count of FILES, not of distinct photographs. Deduplicated by md5 and filename, the primary sources hold **~12,500 distinct images**: HVDROPDB ships 485 non-mask files but only 329 unique photographs, Ostrava has 19 internal byte-duplicates, and ~20 Shenzhen images re-appear in ROP-VL at higher resolution. `results/prepared/rop_staging/manifest.csv` carries the deduplicated set.

Everything here is gitignored except this file and `PROVENANCE.json`. It sits beside PRIVATE infant retinal photographs; several sets carry patient identifiers in filenames. **Do not commit, mirror or upload any of it without re-reading the caveats.**

## Datasets

| # | Dataset | Images | Masks | Size | Region | Licence |
|---|---|---:|---:|---:|---|---|
| 1 | **Retinal Image Dataset of Infants and ROP** (`ostrava`) | 6,004 | 0 | 2609 MB | Ostrava, Czechia | CC0 |
| 2 | **ROP-VL** (`rop_vl`) | 2,020 | 0 | 700 MB | Shenzhen Eye Hospital, China | CC BY 4.0 |
| 3 | **FARFUM-RoP** (`farfum_rop`) | 1,533 | 0 | 2164 MB | Farabi Hospital, Iran | CC BY 4.0 |
| 4 | **Multi-View ROP** (`multiview`) | 1,324 | 0 | 2107 MB | China | CC BY-NC 4.0 |
| 5 | **Shenzhen ROP (Zhao 2024)** (`shenzhen`) | 1,099 | 0 | 203 MB | Shenzhen Eye Hospital, China | CC BY 4.0 |
| 6 | **HVDROPDB** (`hvdropdb`) | 485 | 300 | 608 MB | H.V. Desai Eye Hospital, Pune, India | CC BY 4.0 |
| 7 | **ROP optic disc segmentation (Shantou)** (`shantou_od`) | 129 | 0 | 105 MB | Shantou, China | CC BY 4.0 |
| 8 | **ROP retinal vessel segmentation (Shantou)** (`shantou_bv`) | 101 | 101 | 114 MB | Shantou, China | CC BY 4.0 |
| 9 | **ORVS + ROP vessel segmentation** (`orvs_seg`) | 251 | 251 | 8 MB | mixed | unstated |
| 10 | **ROP stages (reviewed) sample** (`ropstages_sample`) | 91 | 0 | 136 MB | unstated | unstated |
| 11 | **COph100** (`coph100_annotations`) | 0 | 648 | 132 MB | derived from Ostrava | CC BY |

### 1. Retinal Image Dataset of Infants and ROP  — `ostrava/`

- **Role:** TRAINING - this is what the shipped model was built on
- **Images:** 6,004
- **Source:** figshare collection 6626162  |  **Licence:** CC0
- **Mirror:** Kaggle jananowakova/retinal-image-dataset-of-infants-and-rop
- **Labels:** ICROP stage 0-3, AP-ROP, treated (st.p. ROP), 4 non-ROP diagnoses; plus-form; 3 cameras
- **Note:** Symlink to ../../dataset_split (the prepared split). PRIVATE infant photographs. Stage labels are recoverable only by joining docs/infant_retinal_database_info.csv on (ID, serie, device). 43% of its 'ROP-positive' class is NOT ROP.
- **⚠ Caveat:** Image resolution is a perfect 1:1 proxy for camera, and camera carries most of the label signal: dimensions alone score AUC 0.911 on its test split.

### 2. ROP-VL  — `rop_vl/`

- **Role:** THE key acquisition - only public source of Stage 4 and Stage 5 anywhere
- **Images:** 2,020
- **Source:** 10.6084/m9.figshare.30143461  |  **Licence:** CC BY 4.0
- **Mirror:** Kaggle nazimriyadh1/rop-vl (unattributed re-upload - prefer figshare)
- **Labels:** 8 classes: normal, Stage 1-5, A-ROP, laser-treated. Per-infant GA, birth weight, plurality, delivery mode. Per-class clinical text (vision-language).
- **Note:** Overlaps Shenzhen by only ~1-2% of images (20-22 shared photographs) despite the same hospital and 6 shared authors. Where they overlap ROP-VL holds the higher-resolution original. No zone axis, no plus axis.
- **⚠ Caveat:** Filenames embed exam dates at day granularity - a date-of-service identifier. Do not propagate those filenames into anything public.

### 3. FARFUM-RoP  — `farfum_rop/`

- **Role:** The plus/pre-plus axis, and the only inter-rater data we have
- **Images:** 1,533
- **Source:** figshare collection 6721269  |  **Licence:** CC BY 4.0
- **Labels:** Retinopathy grade + ICROP stage, recorded INDEPENDENTLY by five paediatric ophthalmologists (A-E), plus per-patient details.
- **Note:** Ships as 68 per-patient RAR5 archives. p7zip cannot decode RAR5 and will silently write zero-byte files - use `unar`.
- **⚠ Caveat:** ZERO of its 1,533 images are graded unanimously. Every image has 2 or 3 distinct grades among the 5 graders, and the split is systematic (C and E consistently more conservative). Treat as soft labels, never as ground truth.

### 4. Multi-View ROP  — `multiview/`

- **Role:** The only source with acquisition-view labels - directly attacks the view confound
- **Images:** 1,324
- **Source:** HuggingFace lijuanliao/ROP-DATASET  |  **Licence:** CC BY-NC 4.0
- **Labels:** Per-image ICROP stage (1-4, A-ROP, Normal) + 6 acquisition views (L/R x temporal/posterior-pole/nasal) + neonatal risk factors (oxygen therapy, ventilation, sepsis, transfusion).
- **Note:** Discoverable ONLY via HuggingFace full-text search; absent from Kaggle, figshare and ordinary web search.
- **⚠ Caveat:** NON-COMMERCIAL licence - do not mirror publicly. Its zone/plus/treatment columns are nearly empty (zone 13 of 232 patients, plus 17, treatment 14); only the per-image stage labels are complete.

### 5. Shenzhen ROP (Zhao 2024)  — `shenzhen/`

- **Role:** FIRST DATASET DOWNLOADED in this effort - the original external-validation target
- **Images:** 1,099
- **Source:** 10.6084/m9.figshare.25514449  |  **Licence:** CC BY 4.0
- **Mirror:** Kaggle errorrific/fundus-image-dataset-to-retinopathy-of-prematurity
- **Labels:** Normal, Stage 1-3, laser scars. zip information.xlsx maps image -> infant -> eye.
- **Note:** All 512x512. Largely superseded by ROP-VL (same hospital, higher resolution), but 98% of its images are NOT in ROP-VL, so it still contributes.
- **⚠ Caveat:** 3 images are filed under two different stage folders at once. Binary labels are unaffected; staging labels contradict for those 3.

### 6. HVDROPDB  — `hvdropdb/`

- **Role:** The ridge masks - the feature that defines ICROP stages 1 and 2
- **Images:** 485 (+300 masks/overlays)
- **Source:** 10.17632/xw5xc7xrmp.3 (Mendeley)  |  **Licence:** CC BY 4.0
- **Mirror:** Kaggle tonimaislamlamia/hvdropdb-datasets
- **Labels:** Binary ROP/Normal classification, plus segmentation masks for RIDGE (demarcation line), blood vessels and optic disc. RetCam and Neo cameras.
- **Note:** 785 files but only 329 UNIQUE fundus images: 21 classification files are byte duplicates and 97 images appear in both the classification and segmentation subsets.
- **⚠ Caveat:** Every file is named 1.png..50.png, so basenames collide across all 16 folders. Never flatten this directory.

### 7. ROP optic disc segmentation (Shantou)  — `shantou_od/`

- **Role:** Segmentation only
- **Images:** 129
- **Source:** 10.17632/tv3b2bt6dy.2 (Mendeley)  |  **Licence:** CC BY 4.0
- **Labels:** Optic disc segmentation, LabelMe JSON per image.
- **Note:** UUID.N filenames, same convention as Multi-View and FARFUM.

### 8. ROP retinal vessel segmentation (Shantou)  — `shantou_bv/`

- **Role:** Segmentation only - vessel tortuosity is the substrate of plus disease
- **Images:** 101 (+101 masks/overlays)
- **Source:** 10.17632/prcy36j53v.2 (Mendeley)  |  **Licence:** CC BY 4.0
- **Labels:** Retinal blood vessel segmentation: image + LabelMe JSON + binary mask.
- **Note:** Companion release to shantou_od, same authors and date.

### 9. ORVS + ROP vessel segmentation  — `orvs_seg/`

- **Role:** Small segmentation supplement
- **Images:** 251 (+251 masks/overlays)
- **Source:** Kaggle ojaskundlia/orvs-rop-1-2  |  **Licence:** unstated
- **Labels:** Vessel segmentation image/label pairs, ROP V1 and V2 plus the adult ORVS set.
- **Note:** V1 and V2 are two mask revisions of one image set.
- **⚠ Caveat:** Licence is not stated anywhere - do not redistribute.

### 10. ROP stages (reviewed) sample  — `ropstages_sample/`

- **Role:** Small independent stage-labelled sample
- **Images:** 91
- **Source:** Kaggle solennollivier/ropstages-reviewed  |  **Licence:** unstated
- **Labels:** Stage 1-3 and NoROP encoded in the filename, with a patient number.
- **Note:** Folder is literally named NewROPDataset_Sample_justtotry - it is a sample, not a release.
- **⚠ Caveat:** Provenance and licence both unknown. Do not train on this.

### 11. COph100  — `coph100_annotations/`

- **Role:** ANNOTATIONS ONLY - adds no new photographs
- **Images:** 0 (+648 masks/overlays)
- **Source:** 10.6084/m9.figshare.27061084  |  **Licence:** CC BY
- **Labels:** Registration control points (LabelMe JSON) + vessel masks + overlays.
- **Note:** Ships masks and overlays keyed to Ostrava filenames, which we already hold. Its filenames preserve the TRUE Ostrava DG codes.
- **⚠ Caveat:** Do not count these as additional images.

### 99. Redundant re-uploads (quarantined)  — `_redundant/`

- **Role:** QUARANTINE - safe to delete
- **Images:** 8,710
- **Source:** -  |  **Licence:** -
- **Labels:** -
- **Note:** littlevision (Kaggle hemanathrvh13996/rop-dataset-for-little-vision) re-ships Ostrava byte-identically AND FARFUM re-encoded (all 1,533 filenames match). Kept only so nobody re-downloads it to find out.
- **⚠ Caveat:** Its folder names contain medical record numbers and exam dates.

## Derived

`scripts/build_rop_staging_manifest.py` → `results/prepared/rop_staging/manifest.csv` — a unified, patient-grouped ICROP staging manifest across Ostrava, Shenzhen, ROP-VL and Multi-View.

`scripts/rop_recover_staging.py` → `results/rop/staging_composition.json` — what the current binary ROP label is actually made of.

