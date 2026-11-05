# 2 · The Dataset — How Many Images of Each Stage, and How They Were Split

**Graphs:** [`graphs/02_inclusion_cascade.png`](graphs/02_inclusion_cascade.png) ·
[`graphs/03_images_per_class.png`](graphs/03_images_per_class.png) ·
[`graphs/04_how_the_data_was_split.png`](graphs/04_how_the_data_was_split.png)

---

## What the model has to tell apart

Retinopathy of prematurity (ROP) is abnormal blood-vessel growth in the retina of babies born
too early. It is a leading **preventable** cause of childhood blindness. The international
staging system (ICROP) is what the model predicts:

| Class | What it looks like |
|---|---|
| **Normal** | no disease |
| **Stage 1** | a faint flat white line where normal vessels stop |
| **Stage 2** | that line has become a raised ridge |
| **Stage 3** | abnormal vessels growing out of the ridge |
| **Stage 4/5** | the retina is partly (4) or completely (5) detached — merged, both are rare |
| **AP-ROP** | "aggressive posterior ROP", a fast severe form that is **not** a point on the 1-2-3 ladder |

AP-ROP being off the ladder is why the model has a **separate yes/no output** for it, instead of
treating it as a sixth rung.

---

## From 13,037 photographs down to 3,112 training images

Every exclusion below is counted, reasoned and reproducible.

| Step | Images left | What was removed and why |
|---|---|---|
| **0 · Acquired** | **13,037** | 11 public datasets (mirrors and re-uploads already excluded) |
| **1 · Has an ICROP stage label** | **10,427** | −2,610: segmentation-mask datasets (HVDROPDB 485, Shantou 230, ORVS 251), plus-disease-only data (FARFUM 1,533), a too-small sample (91), and 20 duplicate photographs shared between Shenzhen and ROP-VL |
| **2 · Eligible for the 6-class task** | **8,024**<br>*(1,760 infants)* | −2,403: **1,295 non-ROP pathology** (haemorrhage, toxoplasma, hamartoma, optic-nerve hypoplasia), **1,063 treated/regressed eyes** (post-treatment appearance is not an ICROP stage), **45 stage-0** (immature retina without ROP; too few for its own class) |
| **3 · Split by hospital** | **6,700** training pool<br>**1,324** external hospital | The entire Multi-View hospital (1,324 images, 232 infants) was held back and never trained on. It was halved by baby: **663 tuning** + **661 final sealed test** |
| **4 · Frame caps applied** | **3,112 used for training** | −3,588: a cap of **2 normal frames per baby** and **20 frames per baby per class**. Ostrava photographs each baby many times in one sitting, so without a cap a handful of babies would supply most of the learning signal |

> **Headline:** 13,037 acquired → 10,427 stage-labelled → 8,024 eligible → **3,112 used for training.**
> Caps apply to **training only**. Validation scores every eligible image, at natural disease rates.

---

## How many images of each stage

### Training data (3 hospitals: Ostrava · Shenzhen · ROP-VL)

| Class | Eligible in the pool | **Actually trained on** (after caps) | Removed by caps |
|---|---|---|---|
| Normal | 3,837 | **1,109** | 2,728 |
| Stage 1 | 577 | **515** | 62 |
| Stage 2 | 959 | **696** | 263 |
| Stage 3 | 620 | **535** | 85 |
| **Stage 4/5** | 89 | **89** | 0 |
| AP-ROP | 618 | **168** | 450 |
| **Total** | **6,700** | **3,112** | 3,588 |

**Imbalance in the training set: 12.5 : 1** (1,109 normal against 89 stage 4/5).

⚠️ **Only 89 Stage 4/5 images exist in the whole project, and all 89 come from a single dataset
(ROP-VL).** Any result for that class may partly reflect recognising the data source rather than
the disease. Likewise **AP-ROP has zero images from Shenzhen**.

### The final sealed test set (a hospital never seen in training)

| Class | Images | Usable? |
|---|---|---|
| Normal | 232 | ✅ |
| Stage 1 | 105 | ✅ |
| Stage 2 | 164 | ✅ |
| Stage 3 | 144 | ✅ |
| **Stage 4/5** | **4** | ❌ far too few — declared "not externally validated" in advance |
| **AP-ROP** | **12** | ❌ far too few — declared "not externally validated" in advance |
| **Total** | **661** (230 eyes, 116 babies) | |

---

## How the data was split

| Group | Images | Babies | Purpose |
|---|---|---|---|
| **Training** | 3,112 | 1,528 | the model learns from these |
| **Validation** (5-fold cross-validation) | 6,700 | 1,528 | every photo is scored by a model that never saw that baby |
| **Tuning half of the new hospital** | 663 | 116 | audits and choosing the alarm threshold |
| **Final sealed test half** | **661** | **116** | opened **exactly once**, under rules written down beforehand |

**Splits are grouped by baby, never by photograph.** This matters enormously: one baby can
contribute up to **470** photographs, so a photograph-level split would let the model recognise
the baby instead of the disease. (A published 2026 paper on this same Ostrava data split at the
image level and reported 98–100% accuracy — that is what happens without this precaution.)

**Verified independently:** **zero** babies and **zero** photographs appear in more than one
group, across every comparison; and the 5 validation folds exactly partition all 6,700 images.

### Other checks that were run on the data

| Check | Result |
|---|---|
| Duplicate photographs across splits | 0 exact, 0 near-duplicates (256-bit perceptual hash) |
| Corrupt or missing image files | 0 — all 5,489 evaluated paths resolve |
| Can the hospital be guessed from file metadata? | **Yes — 99.7% accuracy.** The shortcut exists, which is why the preprocessing and training actively fight it |
| Could a model win using metadata alone? | Only macro-F1 **0.187** — that is the floor the image model (0.692) must and does beat |
| Two Ostrava IDs that were the same baby | Found (IDs 40 and 41, 19 byte-identical photographs) and merged |
