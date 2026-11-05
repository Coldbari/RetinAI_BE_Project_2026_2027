# RetinAI — ROP Staging Model · Summary Folder

**Read this page first. Then the six documents, in order. Then the graphs.**

*Built 2026-09-21 from repository commit `f0d4508`. Every number here was recomputed from raw
per-photograph prediction files, not copied from earlier reports.*

> ⚠️ **Not a medical device.** A final-year research prototype. No regulatory clearance, no
> prospective validation. Nothing here supports clinical use.

---

## What this project is, in four sentences

A deep-learning model that looks at **one retinal photograph of a premature baby** and predicts
its **ICROP stage** of retinopathy of prematurity — a leading preventable cause of childhood
blindness. It was built from **13,037 photographs** gathered from **11 public research datasets**,
narrowed to **3,112 training images** from **1,528 babies** at **3 hospitals**, and tested on a
**4th hospital it had never seen**, whose test half was opened **exactly once**. It reaches
**79% accuracy** and **0.90 agreement with the human grader** at the hospitals it learned from,
and **55% / 0.49** at the new one. The most useful thing it produced is not the model but a
measured finding about **how ROP datasets are labelled and how such models should be scored**.

---

## The seven documents

| # | Document | What's in it |
|---|---|---|
| **1** | [`01_DATA_SOURCES.md`](01_DATA_SOURCES.md) | Every dataset used, with **download links**, licences, image counts and known caveats |
| **2** | [`02_DATASET_AND_CLASSES.md`](02_DATASET_AND_CLASSES.md) | **How many images of each stage**, how 13,037 became 3,112, and exactly how the data was split |
| **3** | [`03_PREPROCESSING.md`](03_PREPROCESSING.md) | **What we do to every photograph** before the model sees it, step by step, and why each step exists |
| **4** | [`04_HOW_WE_TRAINED.md`](04_HOW_WE_TRAINED.md) | The model's architecture and **every training parameter**, with the reasoning behind each choice |
| **5** | [`05_RESULTS_AND_ACCURACY.md`](05_RESULTS_AND_ACCURACY.md) | **All the accuracy numbers**, the mistakes, the confidence problem, and every decision that was made |
| **6** | [`06_CURRENT_STATUS.md`](06_CURRENT_STATUS.md) | Where the project stands, the five open problems, and what to do next |
| **7** | [`07_WHY_MULTIPLE_HOSPITALS.md`](07_WHY_MULTIPLE_HOSPITALS.md) | **Why four hospitals** — why not one, why these ones, and what each was used for |

## The thirteen graphs

| # | Graph | Answers |
|---|---|---|
| 01 | [Where the images came from](graphs/01_where_the_images_came_from.png) | Which datasets, how many images, which were used |
| 02 | [Inclusion cascade](graphs/02_inclusion_cascade.png) | How 13,037 photographs became 3,112 training images |
| 03 | [Images per class](graphs/03_images_per_class.png) | **How many photographs of each ICROP stage** |
| 04 | [How the data was split](graphs/04_how_the_data_was_split.png) | Train / validation / tuning / sealed test |
| 05 | [Preprocessing pipeline](graphs/05_preprocessing_pipeline.png) | The five processing steps, as a diagram |
| 06 | Preprocessing examples (infant photographs, not in this public copy) | The same five steps on a real photo of each class |
| 07 | [Training behaviour](graphs/07_training_behaviour.png) | Did it learn? Was training stable? |
| 08 | [Overall accuracy](graphs/08_overall_accuracy.png) | Home hospitals vs a new hospital |
| 09 | [Confusion matrix](graphs/09_confusion_matrix.png) | What it confuses with what |
| 10 | [Per-class performance](graphs/10_per_class_performance.png) | How good it is at each stage, with uncertainty |
| 11 | [Confidence & calibration](graphs/11_confidence_and_calibration.png) | Can the percentages be trusted? |
| 12 | [New hospital, and why](graphs/12_new_hospital_and_why.png) | Why the per-photograph score looks bad |
| 13 | [Why four hospitals](graphs/13_why_four_hospitals.png) | What each hospital contributes, and why one was not enough |

---

## The numbers at a glance

### Data
| | |
|---|---|
| Datasets acquired | **11** (13,037 photographs) |
| Datasets used | **4** |
| Eligible for the 6-class task | **8,024** photographs, 1,760 babies |
| **Used for training** | **3,112** photographs, 1,528 babies, 3 hospitals |
| Validation | 6,700 photographs (5-fold, grouped by baby) |
| **Final sealed test** | **661** photographs, 230 eyes, 116 babies, **1 new hospital** |
| Class imbalance | 12.5 : 1 (1,109 normal vs **89** stage 4/5) |
| Input size | 384 × 384 × 3 |

### Model
| | |
|---|---|
| Backbone | EfficientNetV2-S, ImageNet-pretrained |
| Heads | ordinal stage (CORN) · AP-ROP yes/no · hospital adversary (training only) |
| Parameters | **20,512,600** — all trainable |
| Training | AdamW · lr 3e-4 · OneCycleLR · batch 16 · **20 epochs** · seed 42 · natural sampling · last-epoch weights |
| Size / speed | 83 MB · **112 ms per photograph on a laptop CPU** |

### Accuracy
| | Home hospitals | New hospital |
|---|---|---|
| Accuracy | **0.786** | **0.546** |
| Agreement with grader (QWK) | **0.897** | **0.493** |
| "Any disease" AUC | **0.970** | **0.835** |
| Normal recall | 0.951 | **0.996** |
| Stage 1 / 2 / 3 recall | 0.499 / 0.631 / 0.713 | **0.276 / 0.305 / 0.292** |

---

## The five things you must not miss

1. **"Is anything wrong?" transfers to a new hospital. "Which stage?" does not.** Screening AUC
   0.835 externally, but per-stage recall around 0.28.

2. **The percentages the model shows are NOT probabilities.** Calibration error 0.109 at home
   and **0.334** at a new hospital, where it is almost equally confident whether right or wrong.
   Read them as a **ranking score**.

3. **It is most confident exactly when it makes the most dangerous mistake.** Calling a diseased
   eye "Normal" carries its highest average certainty (0.93), and 141 of those 186 cases were
   above 90% certainty. **No confidence threshold can make this safe.**

4. **A large part of the apparent failure at the new hospital is a labelling artefact, and we can
   prove it.** ROP grows at the *outer* edge of the retina. Photographs of the outer edge score
   **0.841**; nose-side 0.352; back-of-the-eye 0.229. The eye's single diagnosis gets stamped
   onto photographs that cannot possibly show the disease. **ROP models should be scored per eye,
   never per photograph** — and papers must say which unit their labels describe.

5. **The complicated ordinal model is not better than a plain softmax.** Same accuracy
   (p = 0.98), *worse* calibration (3.2×), and behind at the new hospital. Its justification is
   that it represents the disease correctly, not that it scores higher. That is reported, not hidden.

---

## Where the full detail lives

This folder is the short version. The complete audit — 275 files including every table,
confidence interval, error gallery, Grad-CAM audit, reproducibility report and privacy audit —
is in **`../FINAL_MODEL_EVIDENCE_PACKAGE/`**, starting at its `00_START_HERE/README.md`.
