# Public Retinal Image Datasets — Expansion Candidates

Research pass (Aug 2026) on freely downloadable fundus datasets for **DR**, **ROP**, and
**glaucoma**, scoped to what RetinAI could actually add to training or use as a *clean*
external test set.

Counts come from the dataset's own paper or repository unless marked **[verified]**, which
means a Kaggle file listing was paged to exhaustion or a label file was downloaded and counted.
Where a link is dead or access is gated, it says so.

> **VERIFICATION STATUS.**
> **§1 (ROP) — 2026-08-05, strongest tier.** Every ROP dataset was *downloaded and counted on
> disk*, its metadata spreadsheet parsed, and cross-dataset duplication measured by md5,
> filename and perceptual hash. Those numbers are facts about files, not claims about papers.
> **§2 (glaucoma) — out of scope**, retained for reference only, never re-verified.
> **§3 (DR) — 2026-08-04**, Kaggle slugs checked against the API and label CSVs counted; the
> figshare, Mendeley, Zenodo, IEEE DataPort and PhysioNet links there were **not** re-checked,
> so treat their counts as the source's own claim. An earlier revision of this file said
> "every entry below was link-checked"; that was too strong.

> **Read the licence column before training.** Several of the largest sets are non-commercial
> or no-derivatives. See [Licence traps](#licence-traps) at the end.

---

## What we already train on

> **SCOPE (2026-08-05): glaucoma is out.** It cannot be given a properly validated external
> test set, and `configs/glaucoma_multisource.yaml` double-mounted G1020, so ~1,020 images sat
> in the pool twice and the reported test AUC 0.973 is contaminated. §2 below is retained for
> reference only. The project is **ROP-first, DR second**.

| Disease | Sources in use | Images |
|---|---|---|
| DR | EyePACS **train split** + APTOS 2019 | **38,788** [verified] |
| ~~Glaucoma~~ | ~~SMDG-19~~ — **out of scope** | ~~12,316~~ |
| ROP | Retinal Image Dataset of Infants (Ostrava, CZ) | 6,004 / 188 patients [verified] |
| DR external | Messidor-2 (held out, no retraining) | **1,744** |
| DR external | IDRiD (held out, no retraining) | **455** |

**Corrections to earlier revisions of this table.**

- DR is **38,788**, not "~92k". 92k counts the EyePACS *test* zip (53,576), which
  `configs/dr.yaml` never mounted. `docs/PROJECT_OVERVIEW.md` already said 38,788, so the two
  docs contradicted each other. *(The test split is now obtainable with labels — see §3.)*
- Glaucoma **12,449 is a metadata ROW count**, not images used: 133 rows are `types=-1` and
  dropped, leaving 12,316 labelled. The pool the deployed checkpoint actually trained on was
  **13,336**, because `glaucoma_multisource.yaml` mounted a standalone G1020 that SMDG-19
  already contained. That duplicate source has since been removed.
- Messidor-2 is **1,744** in `results/external_dr.json`, not 1,748. IDRiD is **455** scored,
  not the 516 the source reports.

**SMDG-19 already bundles** BEH, CRFO-v4, DR-HAGIS, DRISHTI-GS1, EyePACS-AIROGS
(glaucoma-positive subset only, 3,269), FIVES, G1020, HRF, JSIEC-1000, LES-AV, OIA-ODIR,
ORIGA-light, PAPILA, REFUGE1, sjchoi86-HRF. It **excludes** ACRIMA, RIM-ONE and KEH.
That exclusion list is where a clean zero-shot glaucoma test lives.

**This is no longer the only route to zero-shot, and was never the cleanest.** We already ran a
source-level hold-out — REFUGE1 and ORIGA withheld from training, then retrained — giving
**REFUGE1 0.9364 / ORIGA 0.7354** (`results/glaucoma/heldout_source.json`). Withholding a source
we control is *stronger* evidence than testing on an outside set, because the split is ours to
verify. ACRIMA and RIM-ONE remain useful as a second, independent population.

---

## 1. ROP — the section that matters now

**Everything below was downloaded and counted on this machine on 2026-08-05.** No number here
is a paper claim. Downloads live in `kaggle_cache/` (gitignored). [LIMITATIONS.md](LIMITATIONS.md)
still says "no public ROP dataset exists" — that is badly out of date and needs fixing.

### 1a. What we now hold — 12,514 distinct images

Deduplicated on md5 **and** filename. There is zero duplication between the primary sources.

| Dataset | Images | Infants | Labels | Licence | Source |
|---|---:|---:|---|---|---|
| **Ostrava** (CZ) | **6,004** | **188** | ICROP stages 0-3, AP-ROP, treated, plus-form | **CC0** | figshare collection `6626162` (primary); Kaggle `jananowakova/retinal-image-dataset-of-infants-and-rop` |
| **ROP-VL** (CN) | **2,020** | **1,116** | **Stages 1-5**, A-ROP, laser-treated + per-class clinical text | CC BY 4.0 | figshare `10.6084/m9.figshare.30143461`, file `58329352`, 688,608,891 B, md5 `93c6cc4c838d9bbfb10f153f31bda95d` |
| **FARFUM-RoP** (IR) | **1,533** | **68** | grade + ICROP stage, **independently by 5 ophthalmologists** | CC BY 4.0 | figshare collection `6721269`, 68 per-patient `.rar` + 2 `.xlsx` |
| **Multi-View ROP** (CN) | **1,324** | **232** | stages 1-4, A-ROP, **6 acquisition views**, neonatal risk factors | CC BY-NC 4.0 | HuggingFace `lijuanliao/ROP-DATASET` |
| **Shenzhen** (CN) | **1,099** | **483** | Stages 1-3, laser scars | CC BY 4.0 | figshare `10.6084/m9.figshare.25514449` |
| **HVDROPDB** (IN) | **329 unique** (785 files) | — | binary ROP/Normal + masks for **ridge**, vessels, optic disc | CC BY 4.0 | Mendeley `xw5xc7xrmp` v3 |
| ORVS/ROP-seg | 150 | — | vessel masks | — | Kaggle `ojaskundlia/orvs-rop-1-2` |
| ropstages sample | 85 | 65 | Stages 1-3 | — | Kaggle `solennollivier/ropstages-reviewed` |
| COph100 | 324 (annotations) | 100 eyes | registration control points + vessel masks | CC BY | figshare `27061084` — **re-annotates Ostrava images, not new photographs** |

**Old = 6,004 / 188 infants. New = +6,510 images / +1,831 infants.** The image count grew 2×;
the infant count grew **11×**, which is what actually constrains this project.

### 1b. The ICROP stage axis

Built by `scripts/build_rop_staging_manifest.py` into
`results/prepared/rop_staging/manifest.csv` (10,427 images / 2,016 infants after dropping 20
Shenzhen images that duplicate a higher-resolution ROP-VL copy).

| ICROP class | Ostrava | Shenzhen | ROP-VL | Multi-View | **Total** | Infants |
|---|---:|---:|---:|---:|---:|---:|
| Normal | 2,980 | 234 | 623 | 382 | **4,219** | 716 |
| Stage 0 | 45 | 0 | 0 | 0 | 45 | 4 |
| **Stage 1** | 252 | 89 | 236 | 295 | **872** | 321 |
| **Stage 2** | 458 | 163 | 338 | 299 | **1,258** | 434 |
| **Stage 3** | 125 | 254 | 241 | 312 | **932** | 393 |
| **Stage 4** | 0 | 0 | 59 | 18 | **77** | 51 |
| **Stage 5** | 0 | 0 | 30 | 0 | **30** | 27 |
| AP-ROP | 470 | 0 | 148 | 18 | **636** | 107 |
| Treated / laser | 379 | 339 | 345 | 0 | **1,063** | 399 |
| Other pathology (**not ROP**) | 1,295 | 0 | 0 | 0 | 1,295 | 40 |

Stage 3 goes from **2 infants to 393**. Stages 4 and 5 exist at all only because of ROP-VL,
and both come from **one hospital** — a 7-class staging model is now trainable, but any
stage-4/5 result is single-site and must be reported as such.

**Ostrava carries stage labels and always did.** The codebook is the `diagnosis` sheet of
`docs/infant_retinal_database_info.xlsx`. Recovering it needs a join on
**(infant ID, serie, device)** — never on postconceptual age, which is fractional for RetCam
Envision in the CSV and integer-truncated in the filename, silently dropping 821 images. See
`scripts/rop_recover_staging.py`.

**43% of our "ROP-positive" class is not ROP.** Of 3,024 positives, 1,295 are hemorrhage
(1,061), toxoplasma (114), hamartoms (72) and optic nerve hypoplasia (48). True ROP is
**1,729 images from 20 infants**. The current model is an abnormal-retina detector.

### 1c. ROP-VL vs Shenzhen — overlapping, not duplicate

Six of ROP-VL's seven authors are on the Shenzhen paper and both are Shenzhen Eye Hospital, so
byte hashing proves nothing — the shared images were re-encoded. Measured with a 256-bit dHash
calibrated against Shenzhen's own byte-duplicates (true duplicates sit at distance 0; genuinely
different images never come closer than 6): **20-22 shared photographs, about 1-2% of either
set.** Where they overlap, ROP-VL holds the higher-resolution original (616-1720 px) and
Shenzhen a 512×512 downscale. **Prefer ROP-VL.** Patient-level overlap cannot be checked — the
two use different ID spaces.

### 1d. Label reliability — read before trusting any of these

- **FARFUM-RoP has zero unanimous labels.** Not one of its 1,533 images has all five
  ophthalmologists agreeing; every image has 2 or 3 distinct grades. It is a systematic
  grader split (graders C and E consistently more conservative), not noise. Use it for soft
  labels or label-noise modelling, **not** as ground truth.
- **Shenzhen contains 3 images filed under two different stage folders** (Stage1/Stage2 and
  Stage2/Stage3). Binary labels are unaffected; staging labels contradict for those 3.
- **Multi-View's zone, plus and treatment columns are nearly empty** — zone filled for 13 of
  232 patients, plus disease 17, treatment 14. Its per-image *stage* labels are complete.
- **No public dataset carries the full ICROP triple** (stage + zone + plus). ROP-VL has no
  zone and no plus axis at all.

### 1e. Re-uploads and dead ends — do not re-investigate

**Re-uploads of data we already hold:** `adithyalennzer/retinopathy-of-prematurity` (Shenzhen),
`vjayasoorya/rop-stage-2` + `rop-stage3` (Shenzhen subsets), `ghiffaribravia/rop-stage-classification-datasets`
(Shenzhen + HVDROPDB), HF `tsing967/ROP` (45% subsample of Shenzhen), Kaggle `nazimriyadh1/rop-vl`
(unattributed ROP-VL re-upload). `hemanathrvh13996/rop-dataset-for-little-vision` re-ships
Ostrava byte-identically **and** FARFUM re-encoded (all 1,533 filenames match) — and its folder
names contain **medical record numbers and exam dates**.

**Not obtainable:** i-ROP / i-ROP-DL (US consortium, closed), Narayana Nethralaya / KIDROP
(Bengaluru, closed), Macretina (figshare DOI `29605802` still 404s), ROPDB Mendeley
`29w6jd6tvs` (metadata-only stub, no files), IEEE DataPort DNN-ROP (paywalled, and the images
are 80×60 thumbnails).

**Portals with no ROP data at all:** Grand Challenge (all 262 challenges enumerated),
PhysioNet, TCIA. No ROP challenge has ever run at MICCAI or ISBI.

**Two licence corrections.** Ostrava is **CC0** on figshare, not the "non-commercial" its
GitHub README claims — cite the figshare grant. ROP-VL's Kaggle mirror declares its licence
"unknown"; the figshare primary is CC BY 4.0.

**PHI note.** ROP-VL filenames embed exam dates at day granularity
(`{patient}_{YYYY-MM-DD}_{L|R}_{n}.jpg`), a HIPAA date-of-service identifier. The data is
publisher-de-identified and CC BY, but do not propagate those filenames into anything public.

**Tooling note.** FARFUM's archives are **RAR5**. `p7zip` recognises the format but cannot
decode it and will silently produce zero-byte files — use `unar` (`brew install unar`).

---

## 2. Glaucoma — clean external validation, finally

The honest gap in [RESULTS.md](RESULTS.md) is that G1020 is now inside the training
distribution, so the multi-source AUC 0.957 is no longer zero-shot. These are the sets
**not** in SMDG-19, so they are legitimately unseen:

| Dataset | Images | Access | Notes |
|---|---|---|---|
| **JustRAIGS** | **101,423** [verified] | Kaggle `manit2022/jraigs-dataset` (55.29 GB), CC BY-NC-SA | ⚠️ **Gives exactly ONE new positive.** See below before considering it |
| **AIROGS (full)** | 113,893 / 60,357 subjects | [Zenodo **restricted**](https://zenodo.org/records/5745364) — record public, *files* need manual approval | Same overlap; not practically obtainable |
| **ACRIMA** | 705 | Kaggle `orvile/acrima-glaucoma-assessment-using-fundus-images`, figshare | Explicitly excluded from SMDG-19 → **clean** |
| **RIM-ONE** r1/r2/r3/DL | ~500 | medimrg.webs.ull.es | Excluded from SMDG-19 → **clean** |
| **Chákṣu** | 1,345 | figshare `10.6084/m9.figshare.20123135` | Indian population, disc+cup outlines, 5 expert decisions → **clean** |
| **LAG** | 11,760 | GitHub, by request | Large, attention-annotated → **clean** |
| **GRAPE / HYGD / RIGA / KEH** | varies | figshare / PhysioNet / Deep Blue / Harvard Dataverse | All outside SMDG-19 |

> ⚠️ **AIROGS / JustRAIGS — settled by exact ID matching, not estimated.**
>
> `JustRAIGS_Train_labels_with_paths.csv` (from `mahajantm/justraigs-with-paths`, 1.2 MB —
> labels only, no images) has **101,423 rows: RG 3,270 / NRG 98,153**, a 3.22% positive rate.
> SMDG-19's `metadata - standardized.csv` carries an **`original_name`** column holding the raw
> AIROGS IDs (`TRAIN000034.jpg`, …) for its 3,269 `EyePACS-Glaucoma` rows, so the two sets are
> directly comparable. Intersection = 3,269, with **zero** mismatches.
>
> **JustRAIGS therefore offers exactly 1 new positive image (`TRAIN057766`) and 98,153 new
> negatives.** Our pool is already ~62% negative; taking all of it makes the pool **95.7%**
> negative, and a model can score well there by answering "healthy" almost always. **Do not
> pool the full set.** The licence is also self-contradictory across the two official pages —
> `justraigs.grand-challenge.org` says CC BY-NC-**SA**, `airogs.grand-challenge.org` says
> CC BY-NC-**ND** for the same images. ND arguably forbids distributing trained weights.
>
> **Two things here ARE worth taking, and neither needs the 55 GB:**
> 1. **The labels CSV itself.** It has per-grader annotations for ten glaucoma signs (ANRS,
>    ANRI, RNFLDS, RNFLDI, BCLVS, BCLVI, NVT, DH, LD, LC) on 3,269 images **we already own**.
>    Extra supervision, zero image download, exact 1:1 join.
> 2. **A hard-negative subset** — eyes a grader called glaucoma but the final label cleared —
>    pulled by selective per-file download rather than the whole mirror.
>
> It also records something uncomfortable about data we *already* train on: **992 of the 3,270
> AIROGS positives (30.3%) needed a third grader** because the first two disagreed. Roughly a
> third of our glaucoma positives are contested calls, which no additional data will fix.
>
> For a clean zero-shot test use ACRIMA, RIM-ONE-DL or Chákṣu — not AIROGS.

Full catalogue of ~45 glaucoma sets with access status:
<https://github.com/TheBeastCoding/glaucoma-dataset-metadata>

---

## 3. DR — the biggest single win is already in our own account

### EyePACS 2015 **test** split — 53,576 labelled images, verified

We train on the EyePACS *train* split (35,126). The competition's **test** split is 53,576
images, and its labels **were released publicly** after the competition closed. They ship inside
`benjaminwarner/resized-2015-2019-blindness-detection-images` (18.6 GB, 20,668 downloads) as
`labels/testLabels15.csv`, alongside pre-resized images in `resized test 15/`.

Downloaded and counted (154 KB, labels only):

| Grade | Images |
|---|---:|
| 0 — No DR | 39,533 |
| 1 — Mild | 3,762 |
| 2 — Moderate | 7,861 |
| 3 — Severe | 1,214 |
| 4 — Proliferative | 1,206 |
| **Total** | **53,576** — referable (≥2) **10,281** |

Why this outranks everything else in this file:

- **It more than doubles the DR pool: 38,788 → 92,364.** (And it is where the old "~92k" claim
  came from — that figure described data we never mounted.)
- **It needs no new Python.** `testLabels15.csv` has columns `image, level, Usage` — the same
  schema `configs/dr.yaml` already reads. One new source block.
- **+3,762 Mild cases**, aimed straight at the project's worst DR weakness (Mild-DR recall 0.41).
- Images are pre-resized, so it is an 18.6 GB download instead of ~53 GB of original zips.
- Patient IDs are in the filenames (`<id>_left` / `<id>_right`), so it can be grouped — see the
  `group_pattern` note in `configs/dr.yaml`.

⚠️ Ships with a `Usage` column (Public/Private leaderboard). Irrelevant for training, but do not
mistake it for a data split.

### More *different* images

Raw volume is no longer the bottleneck once the above lands — **domain diversity** and
**early-DR label quality** are (Mild-DR recall is 0.41).

| Dataset | Images | Access | Why it helps us specifically |
|---|---|---|---|
| **DDR** | 13,673 | [GitHub](https://github.com/nkicsl/DDR-dataset) (Google Drive), free | 147 hospitals, 23 Chinese provinces. Includes 1,151 *ungradable* images — directly usable for the OOD/abstention gap in LIMITATIONS §3 |
| **MMRDR** | 24,460 | figshare `10.6084/m9.figshare.29423747`, **CC BY 4.0**, 18.6 GB | CFP + OCT + UWF, 5-grade DR, **7 lesion types**, DME grading. Lesion masks are the most direct fix for weak Mild-DR detection |
| **BRSET** | 16,266 | [PhysioNet](https://physionet.org/content/brazilian-ophthalmological/1.0.0/), credentialed | Brazilian; ships age/sex/comorbidity metadata → enables the **fairness audit** we currently cannot do |
| **mBRSET** | 5,164 | [PhysioNet](https://physionet.org/content/mbrset/1.0/), credentialed | **Handheld cameras, real-world LMIC screening.** Closest proxy to actual deployment conditions |
| **ODIR-5K / OIA-ODIR** | 10,000 | Kaggle `andrewmvd/ocular-disease-recognition-odir5k` | 8 conditions incl. DR *and* glaucoma — paired both-eye images |
| **RFMiD (+2.0)** | 3,200 + 860 | [IEEE DataPort open-access](https://ieee-dataport.org/open-access/retinal-fundus-multi-disease-image-dataset-rfmid) | 46 conditions → negatives for OOD rejection |
| **DeepDRiD** | 2,000 + 256 UWF | [GitHub](https://github.com/deepdrdoc/DeepDRiD) | Dual-view grading + **image-quality labels** |
| **IDRiD** | 516 | [IEEE DataPort](https://ieee-dataport.org/open-access/indian-diabetic-retinopathy-image-dataset-idrid) | Indian, pixel-level lesion masks |
| **UWF-SLO DR** | 1,630 | figshare `10.6084/m9.figshare.31259494`, **CC BY 4.0** | Normal/NPDR/PDR on ultra-widefield |
| **FGADR** | 2,842 | Email request to authors | 1,842 with pixel-level lesion annotation |

---

## Licence traps

- **Retina Image Bank (ASRS)** — ~30,000 images, looks tempting. **Do not use.** Members get
  10 images per login, and the terms forbid derivative works. Not a training corpus.
- **JustRAIGS / AIROGS** — CC BY-**NC**-SA. Fine for research; contaminates any commercial
  path, and share-alike may propagate to model weights depending on interpretation.
- **Messidor-2 and FGADR** — may not be redistributed. Keep them off any public mirror.
- **BRSET / mBRSET** — PhysioNet credentialing + signed DUA, forbids re-identification and
  sharing with non-credentialed users. That means **they cannot be uploaded to a public Kaggle
  dataset**, which conflicts with our Kaggle-mounted training flow.
- **Macretina** — not actually released yet despite the paper. Do not cite as available.

## Kaggle-mounted vs manual download

Our configs mount `/kaggle/input/...`, so mirrored sets are far cheaper to adopt:

**Already on Kaggle:** **EyePACS 2015 test split with labels** (`benjaminwarner/resized-2015-2019-blindness-detection-images` — the highest-value item in this file), Shenzhen ROP, HVDROPDB, Ostrava ROP, JustRAIGS, ODIR-5K, RFMiD,
ACRIMA, IDRiD, Messidor-2, DeepDRiD, SMDG-19.

**Manual download needed:** DDR (Google Drive), MMRDR (figshare, 10-part zip), Chákṣu, LAG,
full AIROGS (Zenodo approval), BRSET/mBRSET (PhysioNet — and cannot legally be re-mirrored).

**All ROP sources are already downloaded** to `kaggle_cache/` (see §1a) — ROP-VL, FARFUM-RoP,
Multi-View, Shenzhen, HVDROPDB, ORVS, COph100. Nothing in §1 still needs fetching. To train on
Kaggle they must first be uploaded as private Kaggle datasets; note **Multi-View is CC BY-NC**,
so a public mirror would be a licence breach.
