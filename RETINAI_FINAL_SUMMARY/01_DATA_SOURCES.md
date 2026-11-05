# 1 · Where the Images Came From

**Graph:** [`graphs/01_where_the_images_came_from.png`](graphs/01_where_the_images_came_from.png)

Every photograph in this project comes from a **public research dataset**. Nothing was
collected by us, and no patient was recruited. **13,037 photographs** were acquired in total
from **11 datasets**, of which **4 datasets** were used to build and test the model.

---

## The 4 datasets the model actually uses

| # | Dataset | Hospital / country | Images | Licence | Download link |
|---|---|---|---|---|---|
| 1 | **Retinal Image Dataset of Infants and ROP** (RIDIRP / "Ostrava") | University Hospital Ostrava, **Czechia** | **6,004** | CC0 (public domain) | figshare collection **6626162** → https://figshare.com/collections/_/6626162 |
| 2 | **ROP-VL** | Shenzhen Eye Hospital, **China** | **2,020** | CC BY 4.0 | https://doi.org/10.6084/m9.figshare.30143461 |
| 3 | **Multi-View ROP** | **China** | **1,324** | **CC BY-NC 4.0** ⚠️ | https://huggingface.co/datasets/lijuanliao/ROP-DATASET |
| 4 | **Shenzhen ROP** (Zhao et al. 2024) | Shenzhen Eye Hospital, **China** | **1,099** | CC BY 4.0 | https://doi.org/10.6084/m9.figshare.25514449 |

### What each one contributes

**1 · Ostrava (6,004 images, 188 infants) — the backbone of the training set.**
The largest source. Its ICROP stage labels are not in a spreadsheet; they are encoded in the
filenames as diagnosis codes (DG0–DG13) and had to be **recovered** by joining on
`(infant ID, camera, examination series)`. All 6,004 resolved.
*Important caveat found during the audit:* **43% of what this dataset's older binary split calls
"ROP-positive" is not ROP at all** — it is haemorrhage, toxoplasmosis, hamartoma or optic-nerve
hypoplasia. That is why the binary label was abandoned and the 6-class stage label rebuilt.

**2 · ROP-VL (2,020 images, 1,116 infants) — the only public source of Stage 4 and Stage 5 anywhere.**
Without it the severe end of the scale would not exist in this project at all. It overlaps the
Shenzhen set by only ~20 photographs (same hospital, different cohort); those duplicates were
detected and removed.

**3 · Multi-View ROP (1,324 images, 232 infants) — the held-out test hospital. NEVER trained on.**
The only dataset that records **which direction the camera was pointing** for each photograph
(left/right eye × temporal / posterior pole / nasal). That single extra column is what allowed
the most important finding in this project (see [`05_RESULTS_AND_ACCURACY.md`](05_RESULTS_AND_ACCURACY.md) §5).
⚠️ Its licence is **non-commercial**.

**4 · Shenzhen ROP (1,099 images, 483 infants) — a second training site.**
The first dataset downloaded in the project. Contains no Stage 4/5 and no AP-ROP.

---

## The other 7 datasets — acquired but NOT used for staging

| Dataset | Country | Images | Licence | Link | Why not used |
|---|---|---|---|---|---|
| **FARFUM-RoP** | Farabi Hospital, Iran | 1,533 | CC BY 4.0 | figshare collection **6721269** | Records plus/pre-plus disease, **not** ICROP stage. *(Notable: 5 ophthalmologists graded it independently and **zero** of the 1,533 images got a unanimous grade — a useful reminder of how hard this task is for humans.)* |
| **HVDROPDB** | H.V. Desai Eye Hospital, Pune, India | 485 | CC BY 4.0 | https://doi.org/10.17632/xw5xc7xrmp.3 | Ridge segmentation masks, no stage label. Kept for future work. |
| **Shantou optic-disc** | Shantou, China | 129 | CC BY 4.0 | https://doi.org/10.17632/tv3b2bt6dy.2 | Segmentation only |
| **Shantou vessels** | Shantou, China | 101 | CC BY 4.0 | https://doi.org/10.17632/prcy36j53v.2 | Segmentation only |
| **ORVS + ROP vessels** | mixed | 251 | unstated | Kaggle `ojaskundlia/orvs-rop-1-2` | Segmentation supplement |
| **ROP stages (reviewed)** | unstated | 91 | unstated | Kaggle `solennollivier/ropstages-reviewed` | Too small to fold; provenance unverified |
| **COph100** | derived from Ostrava | 0 | CC BY | https://doi.org/10.6084/m9.figshare.27061084 | Annotations only — adds no new photographs |

A further **8,710 files** were quarantined as **re-uploads and mirrors** of datasets already
counted, and excluded from every total.

---

## Rules we followed on sourcing

1. **Always cite the original registry, never a Kaggle mirror.** Several of these datasets exist
   on Kaggle as unattributed re-uploads. There has been a 2025–26 wave of retractions over
   untraceable Kaggle data, so every source above is traced to its figshare / Mendeley /
   HuggingFace DOI.
2. **Every source is logged** in `data/rop/PROVENANCE.json` with its DOI, licence, image count,
   byte count, label scheme and known caveats.
3. **De-identification.** The Ostrava filenames originally encoded the baby's sex, gestational
   age, birth weight and postnatal age. Every file was renamed to `<id>_D<camera>_S<series>_<frame>.jpg`,
   and the name-map is kept outside the code repository.
4. **Licence compliance.** The Multi-View test set is **CC BY-NC**, so no commercial use or
   public re-mirroring of those images is permitted.
