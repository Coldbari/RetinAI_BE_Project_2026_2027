# Dataset Summary

*Recomputed from the committed split CSVs by `20_AUDIT/scripts/s02_dataset_leakage.py`.
Tables: [`dataset_counts.csv`](dataset_counts.csv), [`class_by_source.csv`](class_by_source.csv), [`dataset_summary.json`](dataset_summary.json).*

## The corpus this model was built from

| role | images | infants | hospitals |
|---|---|---|---|
| Training pool (unique, after per-infant caps) | **3,112** | **1,528** | 3 |
| Internal evaluation (5 val folds, exactly partitioning the corpus) | **6,700** | 1,528 | 3 |
| External dev half (audits, threshold selection) | 663 | 116 | 1 (held-out) |
| **External locked half (opened once)** | **661** | **116** | 1 (held-out) |

Sources: **Ostrava** (RIDIRP, Czech Republic), **Shenzhen** (China), **ROP-VL** (same Shenzhen hospital, different cohort), **Multi-View** (the held-out hospital, never trained on).

## Class distribution — training pool

| class | images | share |
|---|---|---|
| normal | 1,109 | 35.6% |
| stage_1 | 515 | 16.5% |
| stage_2 | 696 | 22.4% |
| stage_3 | 535 | 17.2% |
| **stage_4_5** | **89** | **2.9%** |
| arop | 168 | 5.4% |

**Imbalance ratio 12.46 : 1.** Normal was capped at 2 images per infant during split construction (which is why it is not the dominant class here); the cap applies to training only, so validation folds carry natural prevalence.

Figure: [`class_distribution.png`](class_distribution.png)

## Validation folds are deliberately not balanced with each other

Figure: [`class_by_split.png`](class_by_split.png)

Fold 0's validation set contains **476 AP-ROP images against 30–41 in every other fold**, and 1,691 images against ~1,252. **470 of those 476 frames come from a single infant** whose AP-ROP diagnosis is a per-session code stamped onto every frame of the session. Fold 0 is therefore the atypical fold in every analysis in this package, and its results should never be read as representative.

## External dev and locked halves are *not* distribution-matched

| class | dev | locked |
|---|---|---|
| normal | 150 | **232** |
| stage_1 | **190** | 105 |
| stage_2 | 135 | 164 |
| stage_3 | 168 | 144 |
| stage_4_5 | 14 | **4** |
| arop | 6 | **12** |

Because the dev half drives threshold selection and the locked half produces the final number, this mismatch propagates into the operating point. It is an inherent consequence of splitting 116 + 116 infants by patient, but it should be stated when the two halves are compared.

## Class × source confounding

Figure: [`class_by_source.png`](class_by_source.png)

| class | Ostrava | ROP-VL | Shenzhen | dominant source |
|---|---|---|---|---|
| normal | 811 | 623 | 221 | — |
| stage_1 | 218 | 236 | 89 | — |
| stage_2 | 224 | 338 | 163 | — |
| stage_3 | 40 | 241 | 254 | Shenzhen/ROP-VL — Ostrava is only 7% |
| **stage_4_5** | 0 | **89** | 0 | **100% ROP-VL** |
| **arop** | 470 | 148 | **0** | **zero Shenzhen** |

*(counts over the full fold-0 file, train + val)*

Hospital identity is linearly decodable from file metadata at **0.997 balanced accuracy** (audit gate G3). Combined with the table above, performance on `stage_4_5` — and to a lesser degree on `arop` and `stage_3` — may partly reflect source recognition rather than pathology.

The parent project declares only `stage_4_5` in `eval.unvalidated_classes`. **On this evidence the AP-ROP and Stage 3 source skews deserve the same declaration.**

## Images per infant

Figure: [`patient_eye_image_counts.png`](patient_eye_image_counts.png)

Training pool: 1,528 infants, median **2** images each (the cap), maximum **470**.
External halves: 116 infants each, median 6 images, maximum 8–9; 230 eyes per half, 1–5 acquisition views per eye (median 3).

This heavy-tailed structure is why every confidence interval in this package resamples **infants**, not images.
