# RetinAI — Final Plan Before Implementation

**For:** Praharsh · **Date:** 2026-08-04 · **Branch:** `feature/clinic-grade-foundation`
**Status:** PLAN ONLY. No code has been written. Nothing has been run.

**Words explained on first use.** *Zero-shot* = we run the trained model on new images without training on them. *External test set* = images from a different hospital or country than the training images. *Checkpoint* = the saved model file (`weights.pth`). *AUC* = a score from 0 to 1 for how well the model ranks sick above healthy; 0.5 is a coin flip. *Sensitivity* = share of sick eyes the model catches. *Specificity* = share of healthy eyes it correctly leaves alone. *Prevalence* = share of sick cases in a group. *PPV* = if the model raises an alarm, the chance the eye really is sick. *TTA* = test-time augmentation, scoring one image 5 times in 5 slightly different ways and averaging. *Manifest* = the CSV list of image paths and labels the code actually reads. *sha256* = a fingerprint of a file; two different files can have the same size but never the same sha256.

---

## 1. What we are doing and why

> ## ⚠️ SUPERSEDED IN PART — read this box first (2026-08-04)
>
> Two of this plan's premises were **disproven** after it was written. The plan is kept because
> its step-by-step method, failure modes and comparator discipline are still correct, but do not
> act on the two claims below.
>
> **1. "The new ROP data has no patient IDs" — FALSE.** Three of the four candidates ship a
> recoverable patient key, so they are trainable, not test-only:
> Shenzhen (`zip information.xlsx`, `ID` + eye laterality → 483 patients),
> FARFUM-RoP (one `.rar` per patient → 68), ROP-VL (filename + `img_info.xlsx` → 1,116).
> Only HVDROPDB genuinely lacks one (its files repeat `1.png`…`50.png`).
>
> **2. "Our image-level splitter would leak" — FIXED.** Commit `0d5887b` added group-aware
> splitting (`group_pattern`) plus `audit_group_leakage`, which refuses to write a manifest
> whose groups span splits. Measured on real EyePACS labels: 4,466 patients spanned train/val
> (8,932 images, 25.4%) before; **0** after. The ROP splits were already clean — 188 infants,
> 0 spanning, in both manifests.
>
> The retrain-vs-test-only conclusion should therefore be **re-derived**, not read off this
> plan. The laser-scar caveat (31% of Shenzhen) and the downstream-cost caveat still stand.

We will test the three existing models on new public image sets that they have never seen, and we will not retrain anything. We chose this because *(see the box above — the patient-ID premise no longer holds)* the new ROP data was believed to have no patient IDs, so our image-level splitter would put sibling images of the same baby in both train and test; because 31% of the Shenzhen ROP set is post-treatment laser scars, which is a different disease state from what the model learned; and because a retrain would force 45–70 hours of hand-fixing numbers across docs and slides. The output is one honest sentence per disease: *this is how the model behaves on a hospital it has never seen*. That is the strongest evidence a thesis prototype can produce in the time we have. The whole plan is roughly **25 hours of work plus about 3–4 hours of Kaggle GPU waiting**, against 45–70 hours for the retrain path.

---

## 2. THE IMAGE COUNTS

This is the section you asked for.

### 2a. TRAINING POOL — before and after

**The honest answer: nothing changes. Not one image is added to training. We are not retraining.**

| Disease | Training pool BEFORE | Training pool AFTER | Change | Source |
|---|---:|---:|---:|---|
| DR | **38,788** | **38,788** | **0** | 35,126 counted from `trainLabels.csv` (EyePACS train split); 3,662 from `results/gate_frr_adult.json` (APTOS) |
| ROP | **6,004** | **6,004** | **0** | `results/prepared/rop/manifest.csv` (6,004 rows counted) |
| Glaucoma | **13,336** | **13,336** | **0** | reconstructed from `results/glaucoma/metrics.json:201-262` through `models/common/data_prep.py:82-84` — marked **LIKELY**, no manifest exists on disk |
| **Total** | **58,128** | **58,128** | **0** | |

Two things about these training numbers that are already wrong in our docs and must be fixed anyway:

- `docs/DATASETS.md:9` says DR is "~92k". That is wrong. 92k counts the EyePACS **test** zip (53,576), which `configs/dr.yaml:16-18` never mounts. The real corpus is **38,788**. `docs/PROJECT_OVERVIEW.md:122` already says 38,788, so our two docs contradict each other.
- `docs/DATASETS.md:10` says glaucoma is "12,449". That is the number of **metadata rows** in SMDG-19, not images used. 133 rows are dropped (`types=-1`), leaving 12,316 labelled. Then `configs/glaucoma_multisource.yaml:24-30` adds the standalone G1020 CSV a **second** time, even though SMDG-19 already contains all 1,020 G1020 images. So **about 1,020 images sit in the pool twice**, and because the splitter works at image level with no de-duplication, the same image can be in train and in test. This is real and it contaminates the reported glaucoma test AUC 0.973.

### 2b. TEST / VALIDATION POOL — before and after

**DR — no new dataset in this plan.**

| Set | Images | Positive | Negative | Change | Source |
|---|---:|---:|---:|---|---|
| Internal val (used as the test set) | 5,815 | 1,252 referable | 4,563 | unchanged | `results/dr/metrics.json:238-269` |
| Internal test | **0** | – | – | unchanged | `configs/dr.yaml` has no `test_split`; there is no `test` block in metrics.json |
| Messidor-2 external | 1,744 | 457 | 1,287 | unchanged | `results/external_dr.json:46-87` (DATASETS.md wrongly says 1,748) |
| IDRiD external | 455 | 304 | 151 | unchanged | `results/dr/external_idrid.json:8-19` (DATASETS.md wrongly says 516) |
| Manual clinic set | 40 | 20 | 20 | unchanged | `results/confusion_tables.json` DR block |
| **NEW external** | **0** | **0** | **0** | **nothing added** | no clean public DR set was in scope |
| **DR TOTAL AFTER** | **8,054** | **2,033** | **6,021** | **+0** | |

**ROP — Shenzhen is the new set.**

| Set | Images | Positive | Negative | Change | Source |
|---|---:|---:|---:|---|---|
| Internal val | 928 | 628 | 300 | unchanged | `results/prepared/rop/manifest.csv`; `results/rop/metrics.json:120-133` |
| Internal test (29 infants, **7 ROP-positive**) | 1,502 | 824 | 678 | unchanged | `results/rop_statistics.json` s4 |
| Manual clinic set (1 gate refusal) | 33 | 14 | 19 | unchanged | `results/confusion_tables.json` ROP block |
| **NEW: Shenzhen Normal** | **236** | 0 | **236** | **+236** | Kaggle file listing, 6 pages, `errorrific/fundus-image-dataset-to-retinopathy-of-prematurity` |
| **NEW: Shenzhen Stage 1** | **94** | **94** | 0 | **+94** | same listing |
| **NEW: Shenzhen Stage 2** | **165** | **165** | 0 | **+165** | same listing |
| **NEW: Shenzhen Stage 3** | **261** | **261** | 0 | **+261** | same listing |
| **NEW: Shenzhen laser scars** (treated disease) | **343** | **343** | 0 | **+343** | same listing, folder is literally `image/laser scars` |
| **NEW ROP subtotal** | **1,099** | **863** | **236** | **+1,099** | 1,100 files = 1,099 .jpg + 1 .xlsx |
| **ROP TOTAL AFTER** | **3,562** | **2,329** | **1,233** | **+1,099** | |

How the 1,099 are actually used:
- **Run A (the headline):** 236 Normal vs 520 Stage 1+2+3 = **756 images**, 68.8% positive.
- **Run B (reported separately):** the **same** 236 Normal vs 343 laser scars = **579 images**. Run B shares Run A's negatives, so the two runs are **correlated and cannot be subtracted from each other**.
- **Run C (added by this plan):** all **1,099** images, laser scars counted as positive, one confusion table. A real NICU screen cannot exclude already-treated babies — they walk in the door.

**Glaucoma — ACRIMA and RIM-ONE are the new sets.**

| Set | Images | Positive | Negative | Change | Source |
|---|---:|---:|---:|---|---|
| Internal val | 1,333 | 506 | 827 | unchanged | `results/glaucoma/metrics.json:201-227` |
| Internal test | 1,999 | 759 | 1,240 | unchanged | `results/glaucoma/metrics.json:236-262` |
| Manual clinic set | 37 | 19 | 18 | unchanged | `results/confusion_tables.json` Glaucoma block |
| G1020 (was external, now **inside** training) | 1,020 | 296 | 724 | **no longer zero-shot** | `results/glaucoma/calibration_serving.json` says "pooled into training" |
| **NEW: ACRIMA** | **705** | **396** | **309** | **+705** | Kaggle listing, 4 pages, `orvile/acrima-glaucoma-assessment-using-fundus-images` |
| **NEW: RIM-ONE-DL** (unique images) | **485** | **172** | **313** | **+485** | Kaggle listing, 5 pages, `orvile/rim-one-...`; the mirror ships 970 files that are the same 485 images twice |
| **NEW glaucoma subtotal** | **1,190** | **568** | **622** | **+1,190** | |
| **GLAUCOMA TOTAL AFTER** (excluding G1020) | **4,559** | **1,852** | **2,707** | **+1,190** | |

### 2c. The one-line answer

| | DR | ROP | Glaucoma | **All** |
|---|---:|---:|---:|---:|
| **New external images added** | **0** | **1,099** | **1,190** | **2,289** |
| of which positive | 0 | 863 | 568 | **1,431** |
| of which negative | 0 | 236 | 622 | **858** |

**Total new external images: 2,289.** None of them go into training. All of them are test-only.

### 2d. Public images we verified but are NOT using (and why)

| Set | Images | Why not used |
|---|---:|---|
| HVDROPDB (ROP) | **185 labelled** (85 pos / 100 neg) | Only 185 of its 785 files are labelled classification images; 600 are segmentation images and masks. Every file is named `1.png`…`50.png`, so basenames collide across all 16 folders. Its top folder name contains spaces **and a comma**. Worth adding later; out of scope now. |
| LAG (glaucoma) | **4,854** (1,711 / 3,143) | Only reachable through a bundle; it is 41% of the published 11,760 and **which** 41% is UNVERIFIED. |
| DRISHTI-GS (glaucoma) | 101 | Already inside SMDG-19 = already in training. **Not zero-shot.** |
| `realhaadkhan` bundle | 6,145 files | Re-ships ACRIMA (705) and RIM-ONE (485) **byte-for-byte**. Taking it as well buys 1,190 duplicates and one contaminated set. Do not attach it. |
| Macretina | — | figshare DOI returns 404. Not obtainable. |

This is why the earlier claim in `docs/DATASETS.md:58` that new ROP data "roughly **doubles** the ROP pool (6,004 → ~10,000)" is **wrong**. The honest figure for obtainable ROP data is 6,004 → **8,821 (+47%)**, and even that counts HVDROPDB and FARFUM-RoP, which this plan does not use.

---

## 3. The steps

Effort is in hours of your work. "Wall clock" is waiting for Kaggle.

### Phase 0 — Things that must be settled before any GPU runs (5.0 h)

**Step 1 — Find out what the Ostrava DG diagnosis codes mean. (1.0 h, reversible)**
Our whole reason for separating laser scars is the sentence "the model only ever saw acute ROP". That sentence is **UNVERIFIED**. `docs/infant_retinal_database_info.csv` has a `DIAGNOSIS CODE (DG)` column with 11 distinct values over 484 rows ({0:278, 11:90, 3:28, 9:27, 2:16, 8:16, 13:9, 12:7, 1:5, 10:4, 4:4}), and `docs/PROJECT_OVERVIEW.md:127` says the training label is a "binary collapse" of that code. If any of codes 1–13 means treated or regressed ROP, then the model **has** seen treated eyes and the exclusion has no basis.
*Verify:* a written table of codes 1–13 with counts and a cited meaning for each. If no codebook can be found, write "UNVERIFIED" and make Run C (all 1,099) the primary result.

**Step 2 — Create the ROP weights dataset on Kaggle. (1.0 h, reversible)**
This is **confirmed missing**, not merely unchecked: `kaggle datasets list --mine` returns only `retinal-glaucoma-weights`, `retinal-dr-weights`, `retinal-framework-code`, `rop-retinal-dataset`. Upload **both** ROP checkpoints as a private 94 MB dataset: `rop_weights.pth` (sha256 `b698afbb…`, the deployed one) and `rop_weights_devsplit.pth` (sha256 `f5c48cd1…`). They are the **same byte size**, so size proves nothing.
*Verify:* `kaggle datasets files <slug>` lists both at 94,365,746 B; the kernel prints both sha256 values and asserts them.

**Step 3 — Prove which glaucoma checkpoint is on Kaggle. (0.5 h, reversible)**
`praharshnagpure/retinal-glaucoma-weights` contains exactly two files, both 81,618,838 bytes: `weights_multisource.pth` and `weights_smdg_only.pth`. Locally they are `ac79766…` (deployed) and `f85f48c2…` (the SMDG-only model that scored **0.589** on G1020). Nothing today ties the uploaded file to the deployed one. The kernel must glob the **exact filename** `weights_multisource.pth` — never `*.pth` — and assert `ac79766…`. `kaggle_kernels/glaucoma_source_audit.py:49-52` already carries the scar from the last time a glob silently loaded the wrong checkpoint.
*Verify:* the kernel log prints `sha256 ac79766… OK`. If it prints `f85f48c2…`, stop; every number after it belongs to a different model.

**Step 4 — Freeze the comparators, in writing, before any number exists. (0.5 h, reversible)**
Picking the comparator after seeing the result is how a bad number becomes a good one.
- **ROP headline comparator = 0.8808** (`results/rop_device_audit.json` `within_device_auc`), not 0.9271. Reason: 0.9271 is a pooled number on a test split where camera D1 is 62.9% of images and carries **100%** of the ROP-positives; the pooled figure is partly camera-vs-camera separation. Shenzhen is one cohort on one camera platform, so it is device-controlled by construction. Also report 0.9271 (pooled, single view) and 0.9303 (pooled, TTA).
- **The only honest error bar on 0.9271 is the patient-clustered one, [0.8655, 0.9675]** (`results/rop_statistics.json` s4), not the image-level [0.9131, 0.9399].
- **0.6843** is the device-controlled figure for a **different** checkpoint (`weights_devsplit.pth`) on 941 images, and its own interval is **[0.5635, 0.7745]** with 841/2000 bootstrap draws degenerate. Always quote it with that interval.
- **Glaucoma comparator = a band, not a point:** `results/glaucoma/source_audit.json` offers 0.8325 (all 12 sources), 0.8784 (6 "usable"), 0.9143 (size-weighted) on the **same data**. That spread is ~0.08 wide — wider than most differences this experiment can detect. Also state the advertised 0.973 from `docs/RESULTS.md`, because that is what the thesis currently claims.

**Step 5 — Recompute the glaucoma comparator properly. (1.5 h, reversible)**
From `results/glaucoma/val_preds_by_source.npz`, which is already on disk: (a) bootstrap the mean-of-usable-sources to give 0.8784 a confidence interval it currently does not have; (b) recompute it with the two G1020 cells removed. Two of the six "usable" cells are G1020 entering twice (n=81 and n=119, 200 of 808 images) because of the double-count in §2a. The 4-source version over 608 images is the uncontaminated comparator.
*Verify:* three aggregations plus a CI, and a 4-source figure, written into a JSON file — not typed into prose.

**Step 6 — Write down the decision rule for BOTH arms. (0.5 h, reversible)**
The earlier draft had a written decision rule for glaucoma only. That asymmetry is the easiest thing for an examiner to attack. State in advance: what ROP AUC or sensitivity would force a change to `README.md`, `docs/RESULTS.md`, `docs/LIMITATIONS.md` and `webapp/registry.yaml` — not only `docs/DATASETS.md`.

### Phase 1 — Code changes, tested locally (5.5 h)

**Step 7 — Add an opt-in `--tta` flag to `models/validation/external_validation.py`. (1.5 h, reversible)**
Line 82 does a single forward pass and **ignores** `cfg.tta.enabled`, even though every external config sets it true. Every glaucoma comparator we own is 5-view TTA. Default the flag to **off** so `results/external_dr.json` and `results/external_glaucoma.json` stay reproducible. Inside the TTA branch, assert `cfg.train.loss != "ordinal"` or decode the class from logits: `predict_with_tta` returns softmax probabilities while `decode_fn` (`models/common/losses.py:75`) expects logits. It happens to be safe for argmax; it is silently wrong for the ordinal head.
*Verify:* without `--tta` on an existing set, the AUC is bit-identical to the recorded one; with `--tta`, it changes. If it does not change, the flag is not wired.

**Step 8 — Build a positive control for the new TTA path. (1.5 h, reversible)**
A positive control means: reproduce a number we already know, through the exact new code, before scoring any new image. `results/glaucoma/glaucoma_tta_val_preds.npz` and `source_audit.json` per-source AUCs already exist for this. Require the `--tta` path to reproduce them to 3 decimal places, and abort otherwise. Do the same for ROP against `results/rop/tta_test_preds.npz` (0.9303).
*Verify:* the run stops before any external image is loaded if the control fails.

**Step 9 — Write the two ROP external configs, WITHOUT `screen_threshold`. (0.5 h, reversible)**
If `screen_threshold: 0.1933` is present, `external_validation.py:140-151` writes fields named `deployed_sensitivity` and `deployed_specificity`, with exact Clopper-Pearson intervals, computed on **single-view** probabilities — while 0.1933 was derived through the **5-view TTA** path (`results/rop/operating_point.json`: `derived_on: "clean val …, 5-view TTA"`). On our own internal test set that mismatch moves specificity by 5.8 points and flips 56 of 1,502 flags. A prose warning does not survive; a reader quotes the JSON field. Omit the key, and take the operating point only from the TTA pass.
*Verify:* the produced JSON has **no** `deployed_*` keys.

**Step 10 — Write the two glaucoma external configs. (0.5 h, reversible)**
Keep `screen_threshold: 0.5` here — that is what `webapp/inference.py:258` actually serves, and with `--tta` the threshold and the probabilities come from the same path. Mirror `configs/glaucoma_multisource.yaml:32-42` exactly (the config the deployed weights were trained under). Add `.JPG` to `ext`.
*Verify:* `load_config` prints `image_size 384, circle_crop True, ben_graham False, clahe True` and an `eval` block containing `screen_threshold`.

**Step 11 — Write `scripts/external_manifest_guards.py` and break it on purpose. (1.5 h, reversible)**
Five guards, all raising, none warning: contamination (blacklist + fail-closed whitelist), duplicates (basename **and** md5), expected counts, every path readable, label polarity.
*Verify:* six deliberately broken fixtures produce six non-zero exits with the intended message; then the two real manifests exit 0. **A guard that has never been seen to fail is not a guard.**

### Phase 2 — The ROP run (4.25 h + ~1.5–2.5 h wall clock, UNVERIFIED estimate)

**Step 12 — Write `kaggle_kernels/rop_external_shenzhen.py`. (2.5 h, reversible)** It must:
1. Accept **either** ROP checkpoint by sha256 and run the whole thing twice, tagged by hash. The deployed one is the only thing comparable to 0.9271/0.8808; the devsplit one is the only thing comparable to 0.6843. Refusing to score the devsplit checkpoint permanently guarantees the one device-controlled ROP number can never be compared to anything.
2. **md5 every one of the 1,099 files** and hard-fail if one hash appears under two different class folders. Our own ROP corpus already had 39 duplicate images carrying contradictory labels. The glaucoma arm gets five guards; the ROP arm had none.
3. Preflight: open every image, run `circle_crop`, record native and cropped width/height, abort if any frame falls below 64 px on a side or below 10% retained area.
4. Run `gradability()` on all 1,099 and report the refusal count — but **the headline AUC stays on all images**, because none of our comparators (0.9271, 0.8808, 0.6843, 0.8784) was gate-filtered. A gated headline against ungated comparators changes what is being compared.
5. Pass `--internal` **only for Run A**. Passing it for Run B makes `external_validation.py:93-99` write an `internal_value`, a `drop`, and a PASS/REVIEW verdict comparing acute-ROP performance to treated-eye performance, into a permanent file, with nothing marking it meaningless.
6. Produce **Run C**: one confusion table over all 1,099 with laser scars positive.
7. Read only the **column names and row count** of `zip information.xlsx` — never cell values, it may hold infant demographics. If a subject/eye column exists, recompute the interval with `cluster_bootstrap_ci`. If it does not, write `"bootstrap": "image-level — optimistic, images are not independent"` into the JSON, the way `source_audit.json` already does for SMDG.
8. Report PPV twice: at the cohort's own 68.8% prevalence, and re-standardised to a 10% NICU prevalence.
9. Stratify the AUC by the capture-configuration clusters the preflight measures (the listing already shows two file-size clusters in the Normal folder alone) and report a within-cluster mean.

**Step 13 — Push and run. (1.0 h + wall clock, reversible)**
*Verify:* the log shows, in order: both sha lines, the five class counts 236/94/165/261/343, zero cross-class md5 collisions, both CSV lines (756 rows / 520 positive; 579 rows / 343 positive), the crop probe, the gate refusal count, both "manifest ok" lines, the positive control, then the run blocks.

**Step 14 — Read the positive control FIRST. (0.25 h, reversible)**
It must be ≥ 0.85. Do not look at any Shenzhen number before this line.

**Step 15 — Pull outputs and run the black-image check. (0.5 h, reversible)**
Count rows of `y_prob` identical to 6 decimal places. `models/common/datasets.py:33-37` catches every exception and returns a black image **with the true label**. A constant input gives a constant probability row.
*Verify:* the largest duplicate count is 1, and it agrees with the preflight's zero load failures.

### Phase 3 — The glaucoma run (5.75 h + ~30–45 min wall clock)

**Step 16 — Write `kaggle_kernels/glaucoma_zeroshot_acrima_rimone.py`. (2.0 h, reversible)** It must:
build both label CSVs from `os.scandir` (never `glob("*.jpg")`, which drops the two uppercase `.JPG` glaucoma images); use **one** RIM-ONE partitioning only; assert (705, 396) and (485, 172) in the first ten seconds; glob the exact `weights_multisource.pth` and assert sha256 `ac79766…`; run the guards; run `gradability()` too, so the two diseases are consistent about which images the app would accept; and **copy the two manifests and both label CSVs back to `/kaggle/working/`**. Without those files, Step 19 cannot run at all.
Fix the two regexes: written as `r"(r\\d+_[NGS]-\\d+)-[LR]"` they match a literal backslash and never fire, which would silently make `suspect` zero for all 485 rows and turn every "subject cluster" into a singleton.

**Step 17 — Push and run. (1.0 h + wall clock, reversible)** Do **not** attach `realhaadkhan/acrima-rim-one-drishti-gs-lag`.
*Verify:* `[guards] acrima: 705 rows, all 5 guards PASS`, `[guards] rimone: 485 rows, all 5 guards PASS`, `[guards] cross-source: 0 shared content hashes`.

**Step 18 — Pull outputs, including the manifests and label CSVs. (0.25 h, reversible)**
Note: `external_validation.py:159` writes the `_preds.npz` **unconditionally**, before the JSON overwrite guard runs. The JSON is protected; the predictions file is not, and both post-processing scripts read the `.npz`. Copy them somewhere safe immediately.

**Step 19 — Run `scripts/glaucoma_zeroshot_ci.py`. (1.5 h, reversible)**
Per-site r1/r2/r3 AUCs, subject-clustered CI, suspect-excluded AUC, and PPV re-weighted from the dataset's prevalence (56.2% / 35.5%) to the 3% screening prevalence `scripts/confusion_tables.py:39` already uses.
*Verify:* assert `suspect.sum() == 23` and `n_clusters < 485` **explicitly**. Do **not** rely on "the clustered CI must be wider" — only 137 of 485 files carry a subject code, so ~72% of clusters are singletons and a correct grouping can easily give a narrower interval by chance.

**Step 20 — Run the summary script. (1.0 h, reversible)**
One JSON where every AUC carries its file, its key, **which checkpoint** produced it, and whether it used TTA.
*Verify:* every glaucoma AUC quoted in the docs appears verbatim in that JSON.

### Phase 4 — Docs, defects, write-up (4.75 h)

**Step 21 — Fix the one executable stale-threshold bug. (1.0 h, reversible)**
`scripts/rop_statistical_analysis.py:24` hard-codes `DEPLOYED_THR = 0.371` with a comment naming `configs/rop.yaml`, which says **0.1933**. Read the value from the config instead. Then regenerate `results/rop_statistics.{md,json}` — do **not** hand-edit them.
*Verify:* `"deployed"` moves 0.371 → 0.1933 while `"test_threshold"` **stays** 0.371 and the s4 AUC values do **not** move (AUC does not depend on threshold). Then check the s6 McNemar block: `DEPLOYED_THR` also drives the ResNet50-vs-EfficientNetV2-S comparison quoted in `docs/IMPLEMENTATION_LOG.md:740`. **If the p-value crosses 0.05, the "best-in-sweep arch" claim in `README.md` needs its own correction before this step is merged.** Then fix the seven stale doc claims (README.md:30, RESULTS.md:17 and :34, PROJECT_OVERVIEW.md:92 and :129, LIMITATIONS.md:24, DATASETS.md:55), and `git rm configs/rop.yaml.bak` — a tracked file carrying a **third** threshold (0.345) and the wrong architecture, which nothing reads.
*Second verify:* after the doc edits, `grep -rIn 'spec 0.83\|0.90 / 0.83'` returns nothing outside the changelog and the `withdrawn` cells. Replacing the threshold but keeping the old sensitivity/specificity pair would be **worse** than the original.

**Step 22 — Apply the nine `docs/DATASETS.md` corrections. (1.0 h, reversible)** See §6 Decision D2 before committing it.

**Step 23 — Scope the resolution warning to the models that actually apply the filter. (0.5 h, reversible)**
`webapp/inference.py:451` calls `_resolution_notice(image)` unconditionally. On an infant ROP screen it says "Preprocessing applies a fixed-radius filter" when `configs/rop.yaml:21` sets `ben_graham: false` — and then quotes a 0.08-AUC figure measured on **adult DR** images. Pass the list of models that ran; return `None` if none of them enables `ben_graham`. `tests/test_repo_guards.py:206-215` asserts the current behaviour and **must ship in the same commit**.

**Step 24 — Stage the two HF Space files. Do not push. (0.25 h, reversible)** See Decision D8.

**Step 25 — Write the thesis section. (2.0 h, reversible)**
*Verify:* every number traces to a file and a key. Report matched pairs only: single-view AUC with single-view sensitivity/specificity, TTA AUC with TTA sensitivity/specificity. Never one from each — they are two different ROC curves.

---

## 4. What could go wrong, and how we catch it

Each row gives the **tell**: the specific thing you would see if it happened.

| # | What goes wrong | The tell | How we catch it |
|---|---|---|---|
| 1 | Labels flipped in the CSV (Normal→1, Stage→0) | **AUC near 0.3 where ~0.7 was expected.** It reads like "the model does not generalise". The true value is `1 − AUC`. | Kernel asserts `AUC ≥ 0.5`; and the output's own `per_class` support must read exactly 236 / 520. If those two are swapped, polarity is inverted. |
| 2 | Wrong checkpoint, or the softmax column is inverted | **Positive control AUC near 0.07 instead of ~0.93.** 0.07 cannot be confused with "a hard cohort". | sha256 assert on every checkpoint; `load_state_dict` runs strict; the positive control runs before any external image. |
| 3 | Glaucoma silently scored with `weights_smdg_only.pth` | **A glaucoma zero-shot AUC near 0.589** — exactly what that checkpoint scored on G1020. It would look like a real domain-shift finding and would trigger a change to the deployed model. | Exact filename `weights_multisource.pth`, plus sha256 `ac79766…` printed in the log. |
| 4 | Images fail to load, invisibly | **AUC in [0.45, 0.55] AND near-zero variance in the predicted probabilities.** Or: several rows of `y_prob` identical to 6 decimals. | Preflight opens all 1,099/1,190 before any GPU work; manifest row-count assert; post-hoc duplicate-probability count. |
| 5 | Uppercase `.JPG` dropped | **ACRIMA n = 703 and positives = 394**, instead of 705 / 396. | Count guard special-cases exactly that pair and names the two files. |
| 6 | RIM-ONE counted twice | **n = 970 instead of 485.** The AUC barely moves; the confidence interval narrows by about √2 and reports certainty the data does not support. | Basename guard, md5 guard, and a hard count assert. |
| 7 | Duplicate images across ROP classes | The same md5 under both `image/Normal/` and `image/Stage3/` — contradictory labels inside one AUC. | md5 pass over all 1,099, hard fail on any cross-class collision. Note: the old plan's "the largest duplicate count must be 1, *or a small number explained by near-identical frames*" pre-excused the exact signal. That escape clause is removed. |
| 8 | DRISHTI-GS sneaks in as "zero-shot" | **n = 1,291 instead of 1,190.** Nobody would notice. DRISHTI is inside SMDG-19 = inside training; the model has memorised it and it pulls the "zero-shot" AUC up. | Blacklist over every path component, plus a fail-closed whitelist requiring every path to start with the one allowed mount; plus the bundle is simply not attached. |
| 9 | `primary_metric` falls through | `external_value` is not equal to `external.auc`. `configs/rop.yaml:54` says `recall`, and `compute_metrics` has no `recall` key — it silently returns macro-F1, and then `drop` subtracts an F1 from an AUC. | External configs set `primary_metric: auc`; the kernel asserts equality right after each run. Do **not** "fix" `configs/rop.yaml` — `train_utils.select_metric` maps `recall` correctly and it drives early stopping. |
| 10 | Confidence intervals too narrow | **The external interval on 756 images is narrower than the internal patient-clustered [0.8655, 0.9675].** That is not better data; it is a different resampling unit. | State the internal comparator with its patient-clustered interval; label the external one "image-level, optimistic" unless a subject key is found in the xlsx. |
| 11 | Subject clustering silently dead (glaucoma) | `suspect.sum() == 0` and `n_clusters == 485`. Caused by the doubled backslashes in the regex. | Explicit asserts `suspect == 23` and `n_clusters < 485`. |
| 12 | Run B gets a meaningless verdict | A `drop` and an `internal_value` field inside `results/external_rop_shenzhen_laser.json`. | `--internal` is passed for Run A only. |
| 13 | Wrong operating point in the ROP file | A field named `deployed_sensitivity` exists in the ROP external JSON. | `screen_threshold` is omitted from the ROP external configs entirely. |
| 14 | Stale manifest short-circuit | `external_validation.py:66` skips rebuilding if a manifest file already exists — a leftover one silently scores different images under the new name. The tell: `per_class` supports are not 236 / 520. | `rmtree` before each run, then assert row count, split values and label set. |
| 15 | Gate changes what is being measured | Refusal rate above ~2%. The gate was calibrated on the Ostrava **train** split only, so on a different camera it removes a non-random slice. | Headline stays ungated (all comparators are ungated); gate-passing subset reported as a clearly labelled secondary line with the refusal count, **always**, not only when the rate is high. |
| 16 | `circle_crop` misbehaves on RetCam frames | Minimum retained area below 10%, or a **bimodal** retained-area distribution. Bimodal means two optical setups and the result should be split by cluster. | Preflight measures native and cropped size for every image and aborts on the thresholds. |
| 17 | Prevalence illusion | A flattering PPV. Run A is 68.8% positive; a NICU is nearer 10%. ACRIMA is 56.2%; a glaucoma screen is 3%. | PPV is reported twice, at the cohort prevalence and re-standardised, in the same JSON. Any write-up quoting only the first goes back. |
| 18 | Wrong comparator quoted | A "generalisation drop" measured against 0.9669 (pooled, source-mixed) or against 0.9364/0.7354 (a **different checkpoint** whose weights are not even in this repo). | The comparator is frozen in Step 4; the summary JSON carries a `checkpoint` and `tta` column on every row, so a table missing those columns is visibly incomplete. |
| 19 | Field-of-view confound (glaucoma) | RIM-ONE's r3 files are named `*_left_half.png`. If they are disc crops, not whole fundus photos, a low AUC measures framing, not glaucoma. **UNVERIFIED** — confirming needs the pixels. | Log the aspect-ratio and post-crop size distribution per source and compare with SMDG; the per-site table (r3 vs r1/r2) gives a second read for free. |
| 20 | Space push drags in PHI | `git status` inside `retinal-ai` lists more than the two intended files — especially anything under `webapp/static/gallery/`. | Before any push: exactly two paths, nothing else, then run the PHI guard test against the Space tree. |

---

## 5. What this does NOT fix

Be blunt about this in the thesis. Running new external test sets answers one question and leaves the rest untouched.

**5.1 It does not fix the ROP device confound.** `results/rop_device_audit.json` records `overall_auc 0.9271` and `within_device_auc 0.8808` for the same model on the same split, with camera D1 at 62.9% of the test set and carrying **every** ROP-positive test image. Only one camera has both classes at all (`measurable_devices: 1`). The device-controlled figure on the fixed split is **0.684 [0.564, 0.775]**, and even that interval had 841 of 2,000 bootstrap draws degenerate. A Shenzhen result cannot repair this — it is a property of our own data. Fixing it needs a multi-camera prospective cohort.

**5.2 It does not fix the 13.1% clinic PPV for ROP.** That is recorded in the repo from T17 (commit `6e3b73d`). What we can see directly is `results/confusion_tables.json`: 14 true positives, 14 false positives, 5 true negatives on 33 images. The deployed operating point 0.1933 gives sensitivity 0.990 and specificity **0.398** — about 60% of healthy infants get flagged. Adding Shenzhen images does not move the served threshold. (The exact 13.1% re-standardisation was not re-derived in this pass; treat the number as repo-recorded, not re-verified here.)

**5.3 It does not fix the 7-positive test set.** The ROP internal test split is 1,502 images but only **29 infants**, of whom **7** are ROP-positive. Every internal ROP conclusion rests on 7 babies. Shenzhen adds images, not infants to our test set — and Shenzhen's own patient count (a paper-level claim of 483) is **UNVERIFIED** from a file listing.

**5.4 It does not fix these, either:**
- **DR has no test split at all.** `configs/dr.yaml` never sets `test_split`. Every DR number — QWK 0.728, referable AUC 0.906, the calibration — is measured on the **validation** set, which is the same data the checkpoint was early-stopped on. This plan adds zero DR images, so it does not improve this.
- **The G1020 double-count** inside the glaucoma training pool. About 1,020 images are in the pool twice, and the split is image-level with no de-duplication, so the same image can be in train and test. The reported glaucoma test AUC 0.973 is contaminated by that.
- ~~**No patient-level grouping for DR or glaucoma.**~~ **FIXED for DR** in `0d5887b`: `configs/dr.yaml` now declares `group_pattern: '^(\d+)_'`, and `build_manifest` refuses to write a leaking manifest. Measured before/after on the real labels: 4,466 patients spanning train/val (8,932 images, 25.4%) → **0**. **Still open for glaucoma** — SMDG-19 has a `patient_id` column but it is blank, and its standardised filenames carry no key, so that pool is still split per image and several of its constituent collections are two-eyes-per-patient. That gap is unmeasurable with the metadata we have, not merely unmeasured.
- **Glaucoma positives are 67% source artefact.** 340 of the 506 validation positives come from sources that contain **zero** negatives (EyePACS-AIROGS is 100% positive by construction).
- **The train/serve preprocessing skew for DR** (in your memory notes) is untouched.
- **`docs/infant_retinal_database_info.csv`** — 485 rows of per-infant sex, gestational age, birth weight, diagnosis code — is still in the repo and is already inside the uploaded `retinal-framework-code` Kaggle dataset. This plan does not remove it (see Decision D3).

**5.5 Claims from the earlier draft that were wrong, and are now corrected:**
- "Adding all available ROP data roughly **doubles** the pool (6,004 → ~10,000)" → **wrong**. Obtainable data gives 8,821 (+47%), and this plan uses only 1,099 of it. The ~10,000 counted Macretina (not released, DOI 404) and counted HVDROPDB at 785 files when only 185 are labelled.
- "The external number is comparable to 0.9271" → true about the **code path**, false about the **estimand**. 0.9271 is device-confounded. Use 0.8808.
- "0.8784 is the like-for-like comparator, same shape" → **overstated**. It is an unweighted mean of six source AUCs (not a single-collection AUC), two of the six cells are the same double-counted G1020 collection, it has no confidence interval anywhere, and its own alternatives on identical data span 0.8325–0.9143.
- "Three independent gates catch a label flip" → it is **two**. The "Stage3 scores above Normal" rank check does not touch the label vector at all, so a flipped CSV passes it unchanged.
- "`build_kaggle_bundle.py` sweeps for PHI and fails closed" → **overstated**. It only rejects filenames matching a specific pattern, images under `results/`, and credential-shaped strings. A 485-row demographics CSV passes every rule, and already has.
- "`external_validation.py` refuses to overwrite" → true for the **JSON report only**. Line 159 writes the `_preds.npz` unconditionally, and both post-processing scripts read the `.npz`, not the JSON.
- "RIM-ONE `partitioned_by_hospital` preserves site provenance we need" → the `r1_`/`r2_`/`r3_` prefix is in every filename in **both** partitionings. Picking one is still correct (to avoid counting 970), but not for that reason.
- "Sharing Run A's negatives makes A and B comparable" → they share all 236 negatives, so the two estimates are **positively correlated**. They are not independent evidence and must not be subtracted.
- Effort for the ROP kernel was costed at 1.5 h with no wall-clock note, while doing strictly more GPU work than the glaucoma kernel, which was costed at 30–45 min wall clock. Corrected in §3.

---

## 6. Decision points — where you must choose

These are not defaults. A machine cannot pick them for you.

**D1 — How to handle the 343 laser-scar images.** *Blocked on Step 1.*
- **(a)** Exclude from the headline, report separately (Run B). Clean story, but drops 31% of the cohort, and an examiner will ask why.
- **(b)** Pool all 1,099 with laser scars counted positive (Run C) as the primary number. This is what a real NICU sees, because treated babies still present for screening.
- **(c)** Drop the laser images entirely.
The plan currently produces A, B **and** C, so you can choose after seeing the DG codebook. **But choose which one is the headline before you read the AUCs.** My recommendation: if any Ostrava DG code means treated/regressed disease, go with (b), because the exclusion premise collapses. If not, go with (a) and report (C) beside it.

**D2 — Do we commit `docs/DATASETS.md`?**
It is currently **untracked** and it is **not** gitignored. Every other file in `docs/` is tracked. Untracked means no diff, no review, and no coverage from the guards in `tests/test_repo_guards.py` — which is exactly how a stale threshold (line 55) and a self-contradicting ROP total (line 58 counts Macretina, which line 42 of the same file marks as not public) both survived.
- **Commit it** — but only **after** the nine corrections, and only after two claims are either checked or softened: line 6's blanket "Every entry below was link-checked" (only the Kaggle slugs were verified in this pass; figshare/Mendeley/Zenodo/PhysioNet were not), and Shenzhen's "483 patients" (a paper claim, unverifiable from a file listing). Committing it as-is publishes the errors as reviewed facts.
- **Or leave it untracked** and accept that it stays invisible to review.
Your call. My recommendation: correct, tag the two claims UNVERIFIED, then commit.

**D3 — The PHI git-history rewrite. Irreversible. Yours alone.**
Commit `a7e1c77` removed six infant patient photographs from the working tree. **They are still in the git history.** Anyone who clones the repository can check out the earlier commit and get the images back. Removing them for real means rewriting history (`git filter-repo` or similar). What that costs:
- Every commit hash after the rewrite point changes. Permanently.
- The `retinal-ai` gitlink (currently `49d912b`) and any Hugging Face Space history break and must be redone.
- Any existing clone of the repo becomes incompatible and must be re-cloned.
- **It cannot be undone.**
A second, separate PHI question sits next to it: `docs/infant_retinal_database_info.csv` holds 485 rows of per-infant sex, gestational age, birth weight, postconceptual age and diagnosis code. It is unrestricted in `docs/`, and it is **already inside the uploaded `praharshnagpure/retinal-framework-code` Kaggle dataset**. Continuing with this plan re-publishes it. Decide separately: (i) leave it, (ii) remove it from the repo going forward, (iii) remove it and rewrite history, (iv) delete or replace the Kaggle dataset. I am not doing any of these without your explicit instruction.

**D4 — Which RIM-ONE number is the headline?** RIM-ONE is itself three hospitals (r1 n=98, r2 n=250, r3 n=137). The pooled 485-image AUC has the same shape as the 0.9669 pooled number we already call invalid. The within-site mean is the honest one but lives in a secondary file. **Decide before the run.**

**D5 — Which ROP comparator is the headline?** 0.9271 (pooled, flattering, device-confounded) or 0.8808 (within-device, honest, smaller). My recommendation is 0.8808, because the glaucoma arm already applies exactly this principle by preferring source-controlled over pooled. Using pooled for ROP and source-controlled for glaucoma would mean applying opposite standards in one thesis. But it changes the story you tell, so it is yours.

**D6 — Do we also score the devsplit checkpoint on Shenzhen?** Cost: about one extra hour of GPU. Benefit: it is the **only** way anything can ever be placed beside the device-controlled 0.684. Recommended: yes.

**D7 — Gated or ungated headline?** The plan makes the ungated number the headline, because all four comparators were computed ungated. If you want the gate-passing subset as the headline instead, the comparators must be re-gated first — that is a separate piece of work, not part of this plan.

**D8 — Deploy the Hugging Face Space fix?** The live Space currently renders `DR: Not assessed (0%)` and `Glaucoma: Not assessed (0%)` on the history page — a fabricated 0% score, styled identically to a real measured result, on exactly the diseases the router declined to run. The root repo fixed this on 2026-07-31; three later Space deploys never carried it. The fix is two files. This changes a **live public page**, so it is an operator action, staged but not pushed by this plan.

---

*Every count in §2 carries its source file. Anything marked UNVERIFIED was not confirmed and must not be quoted as fact: the Shenzhen patient count, the contents of `zip information.xlsx`, the identity of the 4,854 LAG subset, RIM-ONE's field of view, the Ostrava DG codebook, whether the Shenzhen "Normal" infants are preterm, and the wall-clock estimate for the ROP kernel.*