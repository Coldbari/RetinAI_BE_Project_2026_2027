# ROP v4 run log

Generated 2026-11-01 by `scripts/rop_v4_runlog.py` at commit `7b259be` (HEAD when generated). Every number below is computed from the data files named beside it (run summaries, per-epoch histories, as-run configs, fold manifests, Kaggle logs, the audit and pool summaries, `v3_scores.json`); none is typed by hand. Regenerate, never edit.

Contents: 1. Why these numbers (`docs/v4/README.md`, verbatim) · 2. The data · 3. Every arm, seed, fold and epoch · 4. Results · 5. Decisions and rulings

## 1. Why these numbers

*Hand-written; `scripts/rop_v4_runlog.py` copies this file verbatim into `docs/v4/V4_RUN_LOG.md`. Sources:
`docs/paper/PROTOCOL_V2.md`, the v4 spec, the configs, and the summaries the run log reads.*

**Why 5 folds.** The pool's 1,528 infants are split into five groups. Each model trains on four groups
(about 80% of the data) and is scored on the fifth. After five models, every infant has been scored exactly
once, by a model that never saw that infant. The "CV" numbers pool those five scorings.

**Why the folds are grouped by infant.** Ostrava gives 4,285 photos of only 148 infants, about 29 per infant,
and the frames of one examination look nearly identical. If one infant's frames were in both
training and validation, the model would be marked on near-copies of photos it learned from. So every
photo of an infant sits in the same fold.

**Why 2 seeds (42 and 1337).** Near the end of its schedule a single run's macro-F1 still moves by
0.007–0.056, so one run is a noisy measurement. A second seed shows how much of a difference is noise;
the headline numbers pool both seeds.

**Why 20 epochs, OneCycle, last-epoch weights.** Every arm uses the same fixed schedule (OneCycle: the
learning rate rises over the first 10% of steps, then falls along a cosine) and keeps the weights from
the last epoch. There is no early stopping
(`eval.early_stopping: false`). A CV validation fold holds only 16–19 stage 4/5 photos, so macro-F1 jumps
from epoch to epoch. Picking the best epoch on that same validation fold would report the luckiest epoch,
not the model.

**Why batch 16 and 392 px.** Batch 16, 20 epochs and the class-balanced loss are the protocol's parity
rules with the EfficientNet baseline (arm C). DINOv2 ViT-S/14 cuts the image into 14-pixel patches, so
392 px gives a 28 × 28 patch grid (384 is not a multiple of 14). On Kaggle's Tesla T4 (16 photos per
batch fit in its memory) a 392-px fold takes about half an hour. Arm G tests 518 px, DINOv2's
pre-training size; it also kept batch 16, so the protocol's out-of-memory fallback (batch 8 with
2-step gradient accumulation) was never used.

**Why per-infant caps (20 photos per class, 2 for normal).** Without caps, Ostrava's 148 infants (about
10% of the infants) would supply 64% of the training photos. Normal is the commonest class (3,837 of the
6,700 pool photos), so it is capped at 2 per infant. After the caps, 3,112 training photos remain in v3;
v4 (the same caps plus the Ostrava quality check and the near-duplicate rule) keeps 3,083. Inside a cap,
v4 keeps the frames that pass the quality check, sharpest first.

**Why validation is not capped.** Validation scores all 6,700 pool photos at their natural class mix, so
the numbers describe the data as it is, not a balanced sample. Confidence intervals resample whole
infants, so one heavily-photographed infant cannot pass as many independent cases.

**Leave-one-source-out (LOSO) folds, and why three.** The pool has three sources: Ostrava, ROP-VL and
Shenzhen. A LOSO fold trains on two of them and validates on every photo of the third, which tests how a
model copes with a hospital and camera it never saw. There is one fold per source: `fold_site`
(Shenzhen, from v3), `fold_loso_ostrava` and `fold_loso_ropvl`. Stage 4/5 exists only in ROP-VL, so the
ROP-VL fold scores stage 4/5 with zero training examples.

**Which folds each arm trains.** A, B and D: the five CV folds and the Shenzhen fold. F and G: the five CV
folds only. B adds the two new LOSO folds in v4. H and I: all eight folds.

**Multi-View: dev and sealed.** Multi-View is a fourth hospital that never enters training, split into
two halves. The dev half (663 photos, 230 eyes, 116 infants) is scored for every arm, but only as a
description ("development-external"), never as external validation. The locked half (661 photos,
230 eyes, 116 infants) is sealed: it was scored once before (18 Aug 2026, for four EfficientNet cells)
and is opened a second and last time, for arms A–I together, only after supervisor approval.

**The v4 decision.** Arm I is adopted if its CV QWK5 beats B by at least 0.010 and its mean LOSO QWK5 is
not more than 0.010 below B's. Otherwise H is tried with the same test. Otherwise B stays.

**How to read the metrics.**
- *QWK* (quadratic weighted kappa): agreement between predicted and true stage, where a miss by two
  stages costs four times a miss by one; 1 is perfect and 0 is chance. *QWK5* first folds AP-ROP onto
  stage 4/5, giving five ordered classes.
- *Macro-F1*: the F1 score of each class, averaged with equal weight, so rare stages count as much as normal.
- *Accuracy*: the share of photos given exactly the right class. Normal is 57% of the pool, so accuracy
  says little about the rare stages.

## 2. The data

### 2.1 The pool: 6700 photos, 1528 infants

The validation rows of fold_0, fold_1, fold_2, fold_3, fold_4 (each pool photo is validated in exactly one CV fold).

| source | normal | stage_1 | stage_2 | stage_3 | stage_4_5 | arop | photos | infants |
|---|---|---|---|---|---|---|---|---|
| ostrava | 2980 | 252 | 458 | 125 | 0 | 470 | 4285 | 148 |
| ropvl | 623 | 236 | 338 | 241 | 89 | 148 | 1675 | 983 |
| shenzhen | 234 | 89 | 163 | 254 | 0 | 0 | 740 | 397 |
| **all** | 3837 | 577 | 959 | 620 | 89 | 618 | 6700 | 1528 |

### 2.2 Trained-on rows per class, v3 pool vs v4 pool (`v4_summary.json`)

| pool | normal | stage_1 | stage_2 | stage_3 | stage_4_5 | arop | total |
|---|---|---|---|---|---|---|---|
| v3_kept | 1109 | 515 | 696 | 535 | 89 | 168 | 3112 |
| v4 | 1109 | 497 | 687 | 533 | 89 | 168 | 3083 |

| source | pool | normal | stage_1 | stage_2 | stage_3 | stage_4_5 | arop |
|---|---|---|---|---|---|---|---|
| ostrava | v3_kept | 270 | 190 | 195 | 40 | 0 | 20 |
| ostrava | v4 | 270 | 173 | 191 | 40 | 0 | 20 |
| ropvl | v3_kept | 623 | 236 | 338 | 241 | 89 | 148 |
| ropvl | v4 | 623 | 236 | 338 | 241 | 89 | 148 |
| shenzhen | v3_kept | 216 | 89 | 163 | 254 | 0 | 0 |
| shenzhen | v4 | 216 | 88 | 158 | 252 | 0 | 0 |

Training photos per infant and class are capped (normal 2, other classes 20; v4 floor 2); within a cap the v4 pool keeps the best-ranked frames (blur_rank_in_device descending, ties by image_path). These counts are before the 14 exact duplicates of `exclude_duplicates.csv`, which every kernel drops from training at run time.

The v4 pool leaves out 584 Ostrava frames that fail the quality check (by class: normal 335, stage_1 34, stage_2 106, stage_3 15, stage_4_5 0, arop 94; 131 of them were v3 training rows; 1 failing frame kept by the floor) and 8 near-duplicate images (by source: ostrava 0, ropvl 0, shenzhen 8).

### 2.3 The audit (`audit_summary.json`)

Near-duplicates: dHash distance <= 4 over 6700 photos (22441650 pairs compared).

|  | pairs | cross-fold | different label | same Ostrava session | different infant |
|---|---|---|---|---|---|
| dHash <= 4 | 109 | 17 | 14 | 69 | 24 |
| true duplicates (pixel MAD <= 5) | 72 | 4 | 4 | 56 |  |

Removed from the v4 training rows: 8 images (by source: ostrava 0, ropvl 0, shenzhen 8).

Ostrava frame quality: 585 of 4285 frames fail (by reason: empty 0, blur 429, dark 108, bright 108); 131 of the 715 frames v3 trained on fail; 94 of 148 infants have a failing frame. Rule: An Ostrava frame passes if it is not empty and, against the thresholds of its own device (percentiles over that device's non-empty Ostrava frames), blur_var >= min_blur (10th percentile) and lum_lo <= luminance <= lum_hi (2.5th / 97.5th percentiles); there is no field-of-view criterion.

| class | frames | failing | blur | dark | bright | infants |
|---|---|---|---|---|---|---|
| normal | 2980 | 336 | 225 | 72 | 69 | 135 |
| stage_1 | 252 | 34 | 26 | 0 | 8 | 12 |
| stage_2 | 458 | 106 | 94 | 30 | 2 | 11 |
| stage_3 | 125 | 15 | 11 | 6 | 0 | 2 |
| stage_4_5 | 0 | 0 | 0 | 0 | 0 | 0 |
| arop | 470 | 94 | 73 | 0 | 29 | 1 |

| device | frames | failing | blur | dark | bright | failing by class |
|---|---|---|---|---|---|---|
| clarity_retcam3 | 1428 | 205 | 143 | 36 | 36 | normal 68/653, stage_1 19/85, stage_2 24/220, arop 94/470 |
| phoenix_icon | 1946 | 253 | 195 | 49 | 49 | normal 182/1711, stage_1 14/83, stage_2 57/152 |
| retcam_envision | 911 | 127 | 91 | 23 | 23 | normal 86/616, stage_1 1/84, stage_2 25/86, stage_3 15/125 |

Thresholds per device (percentiles: min_blur 10, lum_lo 2.5, lum_hi 97.5; work size 512):

| device | min_blur | lum_lo | lum_hi | frames |
|---|---|---|---|---|
| clarity_retcam3 | 23.94 | 77.20 | 162.40 | 1428 |
| phoenix_icon | 19.39 | 51.71 | 165.09 | 1946 |
| retcam_envision | 59.76 | 62.52 | 130.65 | 911 |

### 2.4 Multi-View (the held-out hospital)

Dev half (`test_dev.csv`): 663 photos, 230 eyes (v3_scores.json), 116 infants; per class normal 150, stage_1 190, stage_2 135, stage_3 168, stage_4_5 14, arop 6.
Locked half: 661 photos, 116 infants (from `splits_summary.json`; the locked file itself is never read).

## 3. Every arm, seed, fold and epoch

Arms A, B, C, D, E, F, G, H, I in that order; seeds [42, 1337]. Per fold: rows per class, images and optimizer steps per epoch, time, one table row per epoch, and the final epoch's per-class recall.

### Arm A: RETFound-Green ViT-S

Run dirs: `results/rop/v3/runs/retfound_green_s42`, `results/rop/v3/runs/retfound_green_s1337`. Missing seeds of [42, 1337]: none.

Recipe (the as-run `config.yaml` of fold_0, seed 42; any fold whose config differs says so):

| parameter | value |
|---|---|
| model.arch | retfound_green |
| preprocess.image_size | 392 |
| preprocess.normalize | half |
| preprocess.clahe | True |
| train.lr | 0.0002 |
| train.layer_decay | 0.8 |
| train.weight_decay | 0.05 |
| train.epochs | 20 |
| train.batch_size | 16 |
| train.grad_accum | 1 (default) |
| train.label_smoothing | 0.05 |
| train.loss | ce |
| train.class_balanced | True |
| train.class_balanced_beta | 0.999 |
| train.weighted_sampler | False |
| train.strong_aug | True |
| train.site_randomization | True |
| train.amp | True |
| train.ema_decay | not set |
| train.head_warmup_epochs | not set |
| model.pretrained | True |
| model.dropout | 0.3 |
| model.drop_path | 0.1 |
| eval.early_stopping | False |
| eval.primary_metric | macro_f1 |
| train.aug_v4 | not set |
| train.bag_loss | not set |
| train.uda | not set |
| pool (summary.json) | v3 |
| config source | configs/rop_staging_rfg.yaml |

<details><summary>Every key of that as-run config.yaml</summary>

| key | value |
|---|---|
| disease | rop_staging |
| seed | 42 |
| data.num_classes | 6 |
| data.class_names | [Normal, Stage 1, Stage 2, Stage 3, Stage 4/5, AP-ROP] |
| data.manifest | results/prepared/rop_staging/_v3_fold0.csv |
| data.presplit | True |
| data.group_col | patient_id |
| preprocess.image_size | 392 |
| preprocess.circle_crop | True |
| preprocess.letterbox | True |
| preprocess.ben_graham | False |
| preprocess.clahe | True |
| preprocess.normalize | half |
| preprocess.cached | True |
| preprocess.cache_root | /tmp/_cache768 |
| model.arch | retfound_green |
| model.weights_path | /kaggle/input/datasets/praharshnagpure/rop-v3-weights/retfoundgreen_statedict.pth |
| model.pretrained | True |
| model.head | classification |
| model.dropout | 0.3 |
| model.drop_path | 0.1 |
| train.epochs | 20 |
| train.batch_size | 16 |
| train.lr | 0.0002 |
| train.layer_decay | 0.8 |
| train.weight_decay | 0.05 |
| train.label_smoothing | 0.05 |
| train.amp | True |
| train.strong_aug | True |
| train.site_randomization | True |
| train.loss | ce |
| train.class_balanced | True |
| train.class_balanced_beta | 0.999 |
| train.weighted_sampler | False |
| train.num_workers | 4 |
| eval.early_stopping | False |
| eval.primary_metric | macro_f1 |
| eval.qwk_classes | [Normal, Stage 1, Stage 2, Stage 3, Stage 4/5] |
| eval.report_per_class_recall | True |
| eval.unvalidated_classes | [Stage 4/5, AP-ROP] |
| _source | configs/rop_staging_rfg.yaml |

</details>

#### Arm A · seed 42: overview (6 folds: fold_0, fold_1, fold_2, fold_3, fold_4, fold_site)

| fold | train rows | val rows | images/epoch | steps/epoch | minutes | img/s | final acc | final macro-F1 | final QWK (6) | final QWK5 | AUC |
|---|---|---|---|---|---|---|---|---|---|---|---|
| fold_0 | 2462 | 1691 | 2462 | 154 | 34.0 | 32.1 | 0.6470 | 0.5845 | 0.5291 | 0.6231 | 0.8820 |
| fold_1 | 2477 | 1252 | 2477 | 155 | 31.7 | 40.1 | 0.8107 | 0.7406 | 0.8744 | 0.8684 | 0.9327 |
| fold_2 | 2542 | 1252 | 2542 | 159 | 32.1 | 39.5 | 0.7380 | 0.6248 | 0.8718 | 0.8716 | 0.9021 |
| fold_3 | 2459 | 1252 | 2459 | 154 | 31.6 | 39.7 | 0.7891 | 0.7545 | 0.9065 | 0.9075 | 0.9533 |
| fold_4 | 2452 | 1253 | 2452 | 154 | 31.6 | 39.4 | 0.8532 | 0.7776 | 0.9081 | 0.9183 | 0.9512 |
| fold_site | 2388 | 740 | 2388 | 150 | 27.7 | 40.1 | 0.8608 | 0.5505 | 0.9435 | 0.9537 | n/a |

#### Arm A · seed 42 · fold_0 (CV fold 0)

Kernel `rop-v3-retfound-green-s42-lr2e-4` · run dir `results/rop/v3/runs/retfound_green_s42/fold_0` · as-run manifest `results/prepared/rop_staging/_v3_fold0.csv`

Fold file `results/prepared/rop_staging/fold_0.csv`: validation = 1691 photos of 297 infants (ostrava 1219, ropvl 310, shenzhen 162); training = 2462 photos of 1231 infants (ostrava 544, ropvl 1363, shenzhen 555).

| rows | normal | stage_1 | stage_2 | stage_3 | stage_4_5 | arop | total |
|---|---|---|---|---|---|---|---|
| training rows | 888 | 403 | 556 | 404 | 70 | 141 | 2462 |
| validation rows | 767 | 137 | 163 | 129 | 19 | 476 | 1691 |

- images per epoch = 2462 (the training rows: the fold file's 2474 less 12 exact duplicate(s) dropped at run time)
- steps per epoch = 154 (= ceil(2462 / (16 x 1)), batch_size x grad_accum)
- minutes 34.0 (fold start to next fold start, kernel log) · 32.1 img/s · GPU Tesla T4 · kernel log reports train 2462 val 1691

Per epoch (`metrics.json` history; lr = the first optimizer group, i.e. the lowest layer under layer-wise decay, at the end of the epoch):

| Epoch | train loss | lr | val acc | val macro-F1 | val macro-P | val macro-R |
|---|---|---|---|---|---|---|
| Epoch 1 | 1.8758 | 5.744e-06 | 0.6304 | 0.3251 | 0.4883 | 0.3519 |
| Epoch 2 | 1.5992 | 1.100e-05 | 0.6990 | 0.5178 | 0.5030 | 0.6242 |
| Epoch 3 | 1.3755 | 1.091e-05 | 0.7138 | 0.5510 | 0.5439 | 0.5862 |
| Epoch 4 | 1.2931 | 1.066e-05 | 0.4956 | 0.4351 | 0.5350 | 0.5818 |
| Epoch 5 | 1.1949 | 1.026e-05 | 0.3655 | 0.3722 | 0.5596 | 0.5422 |
| Epoch 6 | 1.1316 | 9.705e-06 | 0.5103 | 0.4598 | 0.5356 | 0.6226 |
| Epoch 7 | 1.0679 | 9.027e-06 | 0.6541 | 0.5574 | 0.6432 | 0.5813 |
| Epoch 8 | 1.0514 | 8.241e-06 | 0.5210 | 0.4699 | 0.5624 | 0.5196 |
| Epoch 9 | 1.0184 | 7.372e-06 | 0.6280 | 0.5182 | 0.5879 | 0.6041 |
| Epoch 10 | 0.9793 | 6.446e-06 | 0.6280 | 0.5496 | 0.6571 | 0.6044 |
| Epoch 11 | 0.9495 | 5.491e-06 | 0.6600 | 0.6271 | 0.6801 | 0.6761 |
| Epoch 12 | 0.8926 | 4.537e-06 | 0.6422 | 0.5582 | 0.6139 | 0.6579 |
| Epoch 13 | 0.8720 | 3.611e-06 | 0.6387 | 0.5562 | 0.6199 | 0.6351 |
| Epoch 14 | 0.8427 | 2.743e-06 | 0.5890 | 0.5623 | 0.6470 | 0.6550 |
| Epoch 15 | 0.8040 | 1.959e-06 | 0.6718 | 0.6064 | 0.6637 | 0.6695 |
| Epoch 16 | 0.7979 | 1.282e-06 | 0.6487 | 0.5760 | 0.6343 | 0.6591 |
| Epoch 17 | 0.7541 | 7.335e-07 | 0.6328 | 0.5708 | 0.6415 | 0.6626 |
| Epoch 18 | 0.7477 | 3.295e-07 | 0.6410 | 0.5822 | 0.6428 | 0.6692 |
| Epoch 19 | 0.7531 | 8.249e-08 | 0.6434 | 0.5819 | 0.6494 | 0.6634 |
| Epoch 20 | 0.7348 | 4.751e-11 | 0.6470 | 0.5845 | 0.6497 | 0.6633 |

Final epoch (Epoch 20), all 1691 validation rows: QWK (6 labels, summary.json) 0.5291 · QWK5 (AP-ROP folded onto stage 4/5, val_predictions.npz) 0.6231 · accuracy 0.6470 · macro-F1 0.5845 · balanced accuracy 0.6633 · AUC 0.8820

| final epoch | Normal | Stage 1 | Stage 2 | Stage 3 | Stage 4/5 | AP-ROP |
|---|---|---|---|---|---|---|
| recall | 0.9635 | 0.5547 | 0.5153 | 0.8605 | 0.9474 | 0.1387 |
| support | 767 | 137 | 163 | 129 | 19 | 476 |

#### Arm A · seed 42 · fold_1 (CV fold 1)

Kernel `rop-v3-retfound-green-s42-lr2e-4` · run dir `results/rop/v3/runs/retfound_green_s42/fold_1` · as-run manifest `results/prepared/rop_staging/_v3_fold1.csv`

Fold file `results/prepared/rop_staging/fold_1.csv`: validation = 1252 photos of 299 infants (ostrava 812, ropvl 327, shenzhen 113); training = 2477 photos of 1228 infants (ostrava 527, ropvl 1346, shenzhen 604).

| rows | normal | stage_1 | stage_2 | stage_3 | stage_4_5 | arop | total |
|---|---|---|---|---|---|---|---|
| training rows | 890 | 401 | 534 | 442 | 73 | 137 | 2477 |
| validation rows | 756 | 114 | 220 | 116 | 16 | 30 | 1252 |

- images per epoch = 2477 (the training rows: the fold file's 2486 less 9 exact duplicate(s) dropped at run time)
- steps per epoch = 155 (= ceil(2477 / (16 x 1)), batch_size x grad_accum)
- minutes 31.7 (fold start to next fold start, kernel log) · 40.1 img/s · GPU Tesla T4 · kernel log reports train 2477 val 1252

Per epoch (`metrics.json` history; lr = the first optimizer group, i.e. the lowest layer under layer-wise decay, at the end of the epoch):

| Epoch | train loss | lr | val acc | val macro-F1 | val macro-P | val macro-R |
|---|---|---|---|---|---|---|
| Epoch 1 | 1.8738 | 5.744e-06 | 0.6885 | 0.4255 | 0.4480 | 0.4622 |
| Epoch 2 | 1.5970 | 1.100e-05 | 0.7332 | 0.5173 | 0.5558 | 0.6030 |
| Epoch 3 | 1.3381 | 1.091e-05 | 0.7133 | 0.5572 | 0.6621 | 0.5773 |
| Epoch 4 | 1.2747 | 1.066e-05 | 0.7812 | 0.6637 | 0.7720 | 0.6133 |
| Epoch 5 | 1.1743 | 1.026e-05 | 0.7468 | 0.6115 | 0.7064 | 0.5826 |
| Epoch 6 | 1.1003 | 9.705e-06 | 0.7859 | 0.6835 | 0.7123 | 0.6809 |
| Epoch 7 | 1.0548 | 9.027e-06 | 0.8211 | 0.7207 | 0.7478 | 0.7052 |
| Epoch 8 | 0.9994 | 8.241e-06 | 0.7149 | 0.6618 | 0.6997 | 0.6809 |
| Epoch 9 | 0.9622 | 7.372e-06 | 0.7812 | 0.6757 | 0.7267 | 0.6769 |
| Epoch 10 | 0.9343 | 6.446e-06 | 0.8035 | 0.7013 | 0.7163 | 0.7046 |
| Epoch 11 | 0.9320 | 5.491e-06 | 0.6709 | 0.6526 | 0.6983 | 0.6366 |
| Epoch 12 | 0.8772 | 4.537e-06 | 0.7979 | 0.7012 | 0.7205 | 0.7253 |
| Epoch 13 | 0.8387 | 3.611e-06 | 0.7716 | 0.6990 | 0.7244 | 0.6926 |
| Epoch 14 | 0.7977 | 2.743e-06 | 0.8139 | 0.7309 | 0.7715 | 0.7085 |
| Epoch 15 | 0.7713 | 1.959e-06 | 0.7923 | 0.7351 | 0.7565 | 0.7340 |
| Epoch 16 | 0.7701 | 1.282e-06 | 0.8099 | 0.7379 | 0.7604 | 0.7246 |
| Epoch 17 | 0.7417 | 7.335e-07 | 0.8027 | 0.7357 | 0.7493 | 0.7322 |
| Epoch 18 | 0.7312 | 3.295e-07 | 0.8147 | 0.7508 | 0.7766 | 0.7370 |
| Epoch 19 | 0.7222 | 8.249e-08 | 0.8107 | 0.7395 | 0.7563 | 0.7341 |
| Epoch 20 | 0.7216 | 4.747e-11 | 0.8107 | 0.7406 | 0.7585 | 0.7340 |

Final epoch (Epoch 20), all 1252 validation rows: QWK (6 labels, summary.json) 0.8744 · QWK5 (AP-ROP folded onto stage 4/5, val_predictions.npz) 0.8684 · accuracy 0.8107 · macro-F1 0.7406 · balanced accuracy 0.7340 · AUC 0.9327

| final epoch | Normal | Stage 1 | Stage 2 | Stage 3 | Stage 4/5 | AP-ROP |
|---|---|---|---|---|---|---|
| recall | 0.9590 | 0.4912 | 0.5136 | 0.6983 | 0.8750 | 0.8667 |
| support | 756 | 114 | 220 | 116 | 16 | 30 |

#### Arm A · seed 42 · fold_2 (CV fold 2)

Kernel `rop-v3-retfound-green-s42-lr2e-4` · run dir `results/rop/v3/runs/retfound_green_s42/fold_2` · as-run manifest `results/prepared/rop_staging/_v3_fold2.csv`

Fold file `results/prepared/rop_staging/fold_2.csv`: validation = 1252 photos of 272 infants (ostrava 830, ropvl 295, shenzhen 127); training = 2542 photos of 1255 infants (ostrava 573, ropvl 1378, shenzhen 591).

| rows | normal | stage_1 | stage_2 | stage_3 | stage_4_5 | arop | total |
|---|---|---|---|---|---|---|---|
| training rows | 888 | 415 | 556 | 481 | 71 | 131 | 2542 |
| validation rows | 772 | 123 | 193 | 110 | 18 | 36 | 1252 |

- images per epoch = 2542 (the training rows: the fold file's 2553 less 11 exact duplicate(s) dropped at run time)
- steps per epoch = 159 (= ceil(2542 / (16 x 1)), batch_size x grad_accum)
- minutes 32.1 (fold start to next fold start, kernel log) · 39.5 img/s · GPU Tesla T4 · kernel log reports train 2542 val 1252

Per epoch (`metrics.json` history; lr = the first optimizer group, i.e. the lowest layer under layer-wise decay, at the end of the epoch):

| Epoch | train loss | lr | val acc | val macro-F1 | val macro-P | val macro-R |
|---|---|---|---|---|---|---|
| Epoch 1 | 1.8747 | 5.744e-06 | 0.2324 | 0.1483 | 0.2937 | 0.2695 |
| Epoch 2 | 1.6014 | 1.100e-05 | 0.7196 | 0.3210 | 0.3210 | 0.3480 |
| Epoch 3 | 1.3573 | 1.091e-05 | 0.3706 | 0.3927 | 0.4846 | 0.4357 |
| Epoch 4 | 1.2312 | 1.066e-05 | 0.7021 | 0.4123 | 0.5186 | 0.4170 |
| Epoch 5 | 1.1906 | 1.026e-05 | 0.4002 | 0.4064 | 0.5270 | 0.4661 |
| Epoch 6 | 1.1417 | 9.705e-06 | 0.6302 | 0.5095 | 0.5291 | 0.6049 |
| Epoch 7 | 1.0959 | 9.027e-06 | 0.7835 | 0.5454 | 0.5450 | 0.6321 |
| Epoch 8 | 1.0127 | 8.241e-06 | 0.5232 | 0.4866 | 0.5700 | 0.5846 |
| Epoch 9 | 0.9975 | 7.372e-06 | 0.7780 | 0.5964 | 0.6207 | 0.6295 |
| Epoch 10 | 0.9871 | 6.446e-06 | 0.7452 | 0.6194 | 0.6552 | 0.6808 |
| Epoch 11 | 0.9596 | 5.492e-06 | 0.6558 | 0.6164 | 0.6748 | 0.6403 |
| Epoch 12 | 0.9279 | 4.537e-06 | 0.6398 | 0.6211 | 0.7184 | 0.6604 |
| Epoch 13 | 0.8674 | 3.612e-06 | 0.7324 | 0.6343 | 0.6597 | 0.6641 |
| Epoch 14 | 0.8571 | 2.744e-06 | 0.7029 | 0.6420 | 0.6996 | 0.6688 |
| Epoch 15 | 0.8209 | 1.959e-06 | 0.7212 | 0.6267 | 0.6390 | 0.6559 |
| Epoch 16 | 0.7832 | 1.282e-06 | 0.7684 | 0.6609 | 0.6897 | 0.6842 |
| Epoch 17 | 0.7684 | 7.336e-07 | 0.6957 | 0.6007 | 0.6112 | 0.6510 |
| Epoch 18 | 0.7710 | 3.295e-07 | 0.7548 | 0.6352 | 0.6533 | 0.6786 |
| Epoch 19 | 0.7533 | 8.252e-08 | 0.7332 | 0.6223 | 0.6444 | 0.6651 |
| Epoch 20 | 0.7536 | 4.729e-11 | 0.7380 | 0.6248 | 0.6474 | 0.6653 |

Final epoch (Epoch 20), all 1252 validation rows: QWK (6 labels, summary.json) 0.8718 · QWK5 (AP-ROP folded onto stage 4/5, val_predictions.npz) 0.8716 · accuracy 0.7380 · macro-F1 0.6248 · balanced accuracy 0.6653 · AUC 0.9021

| final epoch | Normal | Stage 1 | Stage 2 | Stage 3 | Stage 4/5 | AP-ROP |
|---|---|---|---|---|---|---|
| recall | 0.8692 | 0.5610 | 0.5751 | 0.2364 | 0.8889 | 0.8611 |
| support | 772 | 123 | 193 | 110 | 18 | 36 |

#### Arm A · seed 42 · fold_3 (CV fold 3)

Kernel `rop-v3-retfound-green-s42-lr2e-4` · run dir `results/rop/v3/runs/retfound_green_s42/fold_3` · as-run manifest `results/prepared/rop_staging/_v3_fold3.csv`

Fold file `results/prepared/rop_staging/fold_3.csv`: validation = 1252 photos of 327 infants (ostrava 713, ropvl 362, shenzhen 177); training = 2459 photos of 1200 infants (ostrava 609, ropvl 1313, shenzhen 537).

| rows | normal | stage_1 | stage_2 | stage_3 | stage_4_5 | arop | total |
|---|---|---|---|---|---|---|---|
| training rows | 888 | 409 | 564 | 400 | 71 | 127 | 2459 |
| validation rows | 766 | 111 | 182 | 134 | 18 | 41 | 1252 |

- images per epoch = 2459 (the training rows: the fold file's 2469 less 10 exact duplicate(s) dropped at run time)
- steps per epoch = 154 (= ceil(2459 / (16 x 1)), batch_size x grad_accum)
- minutes 31.6 (fold start to next fold start, kernel log) · 39.7 img/s · GPU Tesla T4 · kernel log reports train 2459 val 1252

Per epoch (`metrics.json` history; lr = the first optimizer group, i.e. the lowest layer under layer-wise decay, at the end of the epoch):

| Epoch | train loss | lr | val acc | val macro-F1 | val macro-P | val macro-R |
|---|---|---|---|---|---|---|
| Epoch 1 | 1.8773 | 5.744e-06 | 0.6765 | 0.3739 | 0.4864 | 0.4448 |
| Epoch 2 | 1.6159 | 1.100e-05 | 0.7564 | 0.5680 | 0.6717 | 0.5395 |
| Epoch 3 | 1.3642 | 1.091e-05 | 0.6637 | 0.5933 | 0.5657 | 0.6542 |
| Epoch 4 | 1.2868 | 1.066e-05 | 0.8027 | 0.6794 | 0.6948 | 0.6706 |
| Epoch 5 | 1.1841 | 1.026e-05 | 0.7963 | 0.7114 | 0.7336 | 0.7074 |
| Epoch 6 | 1.1433 | 9.705e-06 | 0.8267 | 0.6441 | 0.6702 | 0.6863 |
| Epoch 7 | 1.0931 | 9.027e-06 | 0.8562 | 0.7492 | 0.8195 | 0.7019 |
| Epoch 8 | 1.0327 | 8.241e-06 | 0.7947 | 0.7234 | 0.7175 | 0.7358 |
| Epoch 9 | 1.0018 | 7.372e-06 | 0.6214 | 0.6670 | 0.7007 | 0.6787 |
| Epoch 10 | 0.9591 | 6.446e-06 | 0.5319 | 0.6629 | 0.7260 | 0.6856 |
| Epoch 11 | 0.9327 | 5.491e-06 | 0.8570 | 0.7186 | 0.7988 | 0.6811 |
| Epoch 12 | 0.9235 | 4.537e-06 | 0.8554 | 0.7680 | 0.7680 | 0.7717 |
| Epoch 13 | 0.8854 | 3.611e-06 | 0.8387 | 0.7355 | 0.7328 | 0.7413 |
| Epoch 14 | 0.8339 | 2.743e-06 | 0.8538 | 0.7814 | 0.8073 | 0.7690 |
| Epoch 15 | 0.7963 | 1.959e-06 | 0.7540 | 0.7352 | 0.7747 | 0.7397 |
| Epoch 16 | 0.7748 | 1.282e-06 | 0.7899 | 0.7571 | 0.7744 | 0.7612 |
| Epoch 17 | 0.7627 | 7.335e-07 | 0.8411 | 0.7764 | 0.7825 | 0.7790 |
| Epoch 18 | 0.7873 | 3.295e-07 | 0.7851 | 0.7571 | 0.7743 | 0.7653 |
| Epoch 19 | 0.7424 | 8.249e-08 | 0.7812 | 0.7492 | 0.7599 | 0.7616 |
| Epoch 20 | 0.7373 | 4.751e-11 | 0.7891 | 0.7545 | 0.7641 | 0.7667 |

Final epoch (Epoch 20), all 1252 validation rows: QWK (6 labels, summary.json) 0.9065 · QWK5 (AP-ROP folded onto stage 4/5, val_predictions.npz) 0.9075 · accuracy 0.7891 · macro-F1 0.7545 · balanced accuracy 0.7667 · AUC 0.9533

| final epoch | Normal | Stage 1 | Stage 2 | Stage 3 | Stage 4/5 | AP-ROP |
|---|---|---|---|---|---|---|
| recall | 0.8159 | 0.5586 | 0.7527 | 0.8657 | 0.7778 | 0.8293 |
| support | 766 | 111 | 182 | 134 | 18 | 41 |

#### Arm A · seed 42 · fold_4 (CV fold 4)

Kernel `rop-v3-retfound-green-s42-lr2e-4` · run dir `results/rop/v3/runs/retfound_green_s42/fold_4` · as-run manifest `results/prepared/rop_staging/_v3_fold4.csv`

Fold file `results/prepared/rop_staging/fold_4.csv`: validation = 1253 photos of 333 infants (ostrava 711, ropvl 381, shenzhen 161); training = 2452 photos of 1194 infants (ostrava 607, ropvl 1292, shenzhen 553).

| rows | normal | stage_1 | stage_2 | stage_3 | stage_4_5 | arop | total |
|---|---|---|---|---|---|---|---|
| training rows | 882 | 420 | 546 | 401 | 71 | 132 | 2452 |
| validation rows | 776 | 92 | 201 | 131 | 18 | 35 | 1253 |

- images per epoch = 2452 (the training rows: the fold file's 2466 less 14 exact duplicate(s) dropped at run time)
- steps per epoch = 154 (= ceil(2452 / (16 x 1)), batch_size x grad_accum)
- minutes 31.6 (fold start to next fold start, kernel log) · 39.4 img/s · GPU Tesla T4 · kernel log reports train 2452 val 1253

Per epoch (`metrics.json` history; lr = the first optimizer group, i.e. the lowest layer under layer-wise decay, at the end of the epoch):

| Epoch | train loss | lr | val acc | val macro-F1 | val macro-P | val macro-R |
|---|---|---|---|---|---|---|
| Epoch 1 | 1.9008 | 5.744e-06 | 0.6161 | 0.1610 | 0.1756 | 0.2671 |
| Epoch 2 | 1.6191 | 1.100e-05 | 0.7390 | 0.5005 | 0.5495 | 0.5353 |
| Epoch 3 | 1.3789 | 1.091e-05 | 0.7678 | 0.4884 | 0.5000 | 0.5977 |
| Epoch 4 | 1.2615 | 1.066e-05 | 0.6864 | 0.5347 | 0.5114 | 0.6148 |
| Epoch 5 | 1.1794 | 1.026e-05 | 0.6872 | 0.5767 | 0.5631 | 0.6455 |
| Epoch 6 | 1.1217 | 9.705e-06 | 0.8140 | 0.6308 | 0.6572 | 0.6810 |
| Epoch 7 | 1.1133 | 9.027e-06 | 0.6808 | 0.6670 | 0.6954 | 0.6985 |
| Epoch 8 | 1.0311 | 8.241e-06 | 0.8220 | 0.6566 | 0.6572 | 0.6975 |
| Epoch 9 | 0.9796 | 7.372e-06 | 0.8236 | 0.6744 | 0.6610 | 0.7337 |
| Epoch 10 | 0.9788 | 6.446e-06 | 0.8532 | 0.7156 | 0.7051 | 0.7379 |
| Epoch 11 | 0.9175 | 5.491e-06 | 0.6552 | 0.6821 | 0.7079 | 0.7111 |
| Epoch 12 | 0.9035 | 4.537e-06 | 0.6816 | 0.7021 | 0.7353 | 0.7437 |
| Epoch 13 | 0.8869 | 3.611e-06 | 0.7694 | 0.7478 | 0.7507 | 0.7788 |
| Epoch 14 | 0.8324 | 2.743e-06 | 0.8532 | 0.7671 | 0.7949 | 0.7637 |
| Epoch 15 | 0.8159 | 1.959e-06 | 0.8468 | 0.7507 | 0.7374 | 0.7916 |
| Epoch 16 | 0.7860 | 1.282e-06 | 0.8516 | 0.7634 | 0.7509 | 0.7943 |
| Epoch 17 | 0.7653 | 7.335e-07 | 0.8547 | 0.7749 | 0.7665 | 0.7960 |
| Epoch 18 | 0.7420 | 3.295e-07 | 0.8500 | 0.7721 | 0.7612 | 0.8055 |
| Epoch 19 | 0.7345 | 8.249e-08 | 0.8500 | 0.7766 | 0.7656 | 0.8089 |
| Epoch 20 | 0.7345 | 4.751e-11 | 0.8532 | 0.7776 | 0.7670 | 0.8082 |

Final epoch (Epoch 20), all 1253 validation rows: QWK (6 labels, summary.json) 0.9081 · QWK5 (AP-ROP folded onto stage 4/5, val_predictions.npz) 0.9183 · accuracy 0.8532 · macro-F1 0.7776 · balanced accuracy 0.8082 · AUC 0.9512

| final epoch | Normal | Stage 1 | Stage 2 | Stage 3 | Stage 4/5 | AP-ROP |
|---|---|---|---|---|---|---|
| recall | 0.9227 | 0.6739 | 0.6418 | 0.8931 | 0.8889 | 0.8286 |
| support | 776 | 92 | 201 | 131 | 18 | 35 |

#### Arm A · seed 42 · fold_site (held-out source Shenzhen (the v3 site fold))

Kernel `rop-v3-retfound-green-s42-lr2e-4` · run dir `results/rop/v3/runs/retfound_green_s42/fold_site` · as-run manifest `results/prepared/rop_staging/_v3_foldsite.csv`

Fold file `results/prepared/rop_staging/fold_site.csv`: validation = 740 photos of 397 infants (shenzhen 740); training = 2388 photos of 1131 infants (ostrava 715, ropvl 1673).

| rows | normal | stage_1 | stage_2 | stage_3 | stage_4_5 | arop | total |
|---|---|---|---|---|---|---|---|
| training rows | 893 | 426 | 533 | 280 | 89 | 167 | 2388 |
| validation rows | 234 | 89 | 163 | 254 | 0 | 0 | 740 |

- images per epoch = 2388 (the training rows: the fold file's 2390 less 2 exact duplicate(s) dropped at run time)
- steps per epoch = 150 (= ceil(2388 / (16 x 1)), batch_size x grad_accum)
- minutes 27.7 (fold start to next fold start, kernel log) · 40.1 img/s · GPU Tesla T4 · kernel log reports train 2388 val 740

Per epoch (`metrics.json` history; lr = the first optimizer group, i.e. the lowest layer under layer-wise decay, at the end of the epoch):

| Epoch | train loss | lr | val acc | val macro-F1 | val macro-P | val macro-R |
|---|---|---|---|---|---|---|
| Epoch 1 | 1.8663 | 5.745e-06 | 0.3568 | 0.1734 | 0.1737 | 0.1983 |
| Epoch 2 | 1.5844 | 1.100e-05 | 0.6324 | 0.3356 | 0.3319 | 0.3638 |
| Epoch 3 | 1.4212 | 1.091e-05 | 0.7446 | 0.4731 | 0.5229 | 0.4705 |
| Epoch 4 | 1.2757 | 1.066e-05 | 0.6851 | 0.3687 | 0.4603 | 0.3729 |
| Epoch 5 | 1.2232 | 1.026e-05 | 0.7284 | 0.3807 | 0.4386 | 0.3950 |
| Epoch 6 | 1.1677 | 9.705e-06 | 0.8122 | 0.5059 | 0.5715 | 0.4928 |
| Epoch 7 | 1.0889 | 9.026e-06 | 0.7811 | 0.4896 | 0.5234 | 0.4732 |
| Epoch 8 | 1.0494 | 8.241e-06 | 0.8541 | 0.5447 | 0.5586 | 0.5404 |
| Epoch 9 | 1.0177 | 7.372e-06 | 0.8405 | 0.5390 | 0.5672 | 0.5256 |
| Epoch 10 | 0.9859 | 6.446e-06 | 0.8270 | 0.5312 | 0.5492 | 0.5206 |
| Epoch 11 | 0.9517 | 5.491e-06 | 0.8595 | 0.6582 | 0.6846 | 0.6469 |
| Epoch 12 | 0.9084 | 4.537e-06 | 0.8378 | 0.5468 | 0.5482 | 0.5505 |
| Epoch 13 | 0.8890 | 3.611e-06 | 0.8446 | 0.5453 | 0.5578 | 0.5366 |
| Epoch 14 | 0.8145 | 2.743e-06 | 0.8554 | 0.5531 | 0.5738 | 0.5393 |
| Epoch 15 | 0.8021 | 1.959e-06 | 0.8541 | 0.5399 | 0.5712 | 0.5274 |
| Epoch 16 | 0.7729 | 1.282e-06 | 0.8432 | 0.5338 | 0.5726 | 0.5196 |
| Epoch 17 | 0.7685 | 7.334e-07 | 0.8635 | 0.5559 | 0.5741 | 0.5461 |
| Epoch 18 | 0.7589 | 3.294e-07 | 0.8608 | 0.5504 | 0.5693 | 0.5398 |
| Epoch 19 | 0.7438 | 8.246e-08 | 0.8568 | 0.5467 | 0.5696 | 0.5350 |
| Epoch 20 | 0.7453 | 4.770e-11 | 0.8608 | 0.5505 | 0.5725 | 0.5393 |

Final epoch (Epoch 20), all 740 validation rows: QWK (6 labels, summary.json) 0.9435 · QWK5 (AP-ROP folded onto stage 4/5, val_predictions.npz) 0.9537 · accuracy 0.8608 · macro-F1 0.5505 · balanced accuracy 0.5393 · AUC n/a

| final epoch | Normal | Stage 1 | Stage 2 | Stage 3 | Stage 4/5 | AP-ROP |
|---|---|---|---|---|---|---|
| recall | 0.9829 | 0.5618 | 0.7975 | 0.8937 | 0.0000 | 0.0000 |
| support | 234 | 89 | 163 | 254 | 0 | 0 |

#### Arm A · seed 1337: overview (6 folds: fold_0, fold_1, fold_2, fold_3, fold_4, fold_site)

| fold | train rows | val rows | images/epoch | steps/epoch | minutes | img/s | final acc | final macro-F1 | final QWK (6) | final QWK5 | AUC |
|---|---|---|---|---|---|---|---|---|---|---|---|
| fold_0 | 2462 | 1691 | 2462 | 154 | 34.5 | 32.5 | 0.6251 | 0.5356 | 0.4857 | 0.5953 | 0.8365 |
| fold_1 | 2477 | 1252 | 2477 | 155 | 30.6 | 41.5 | 0.8115 | 0.7156 | 0.8732 | 0.8681 | 0.9407 |
| fold_2 | 2542 | 1252 | 2542 | 159 | 32.1 | 41.1 | 0.7748 | 0.6100 | 0.8574 | 0.8818 | 0.8860 |
| fold_3 | 2459 | 1252 | 2459 | 154 | 31.2 | 40.0 | 0.7796 | 0.7558 | 0.9123 | 0.9104 | 0.9502 |
| fold_4 | 2452 | 1253 | 2452 | 154 | 31.1 | 40.2 | 0.8388 | 0.7705 | 0.9082 | 0.9164 | 0.9562 |
| fold_site | 2388 | 740 | 2388 | 150 | 27.9 | 40.5 | 0.8689 | 0.5612 | 0.9479 | 0.9592 | n/a |

#### Arm A · seed 1337 · fold_0 (CV fold 0)

Kernel `rop-v3-retfound-green-s1337-lr2e-4` · run dir `results/rop/v3/runs/retfound_green_s1337/fold_0` · as-run manifest `results/prepared/rop_staging/_v3_fold0.csv`

Fold file `results/prepared/rop_staging/fold_0.csv`: validation = 1691 photos of 297 infants (ostrava 1219, ropvl 310, shenzhen 162); training = 2462 photos of 1231 infants (ostrava 544, ropvl 1363, shenzhen 555).

| rows | normal | stage_1 | stage_2 | stage_3 | stage_4_5 | arop | total |
|---|---|---|---|---|---|---|---|
| training rows | 888 | 403 | 556 | 404 | 70 | 141 | 2462 |
| validation rows | 767 | 137 | 163 | 129 | 19 | 476 | 1691 |

- images per epoch = 2462 (the training rows: the fold file's 2474 less 12 exact duplicate(s) dropped at run time)
- steps per epoch = 154 (= ceil(2462 / (16 x 1)), batch_size x grad_accum)
- minutes 34.5 (fold start to next fold start, kernel log) · 32.5 img/s · GPU Tesla T4 · kernel log reports train 2462 val 1691

Per epoch (`metrics.json` history; lr = the first optimizer group, i.e. the lowest layer under layer-wise decay, at the end of the epoch):

| Epoch | train loss | lr | val acc | val macro-F1 | val macro-P | val macro-R |
|---|---|---|---|---|---|---|
| Epoch 1 | 1.8461 | 5.744e-06 | 0.4335 | 0.2782 | 0.5718 | 0.3759 |
| Epoch 2 | 1.6204 | 1.100e-05 | 0.6369 | 0.5204 | 0.5912 | 0.6190 |
| Epoch 3 | 1.4105 | 1.091e-05 | 0.6168 | 0.5521 | 0.6435 | 0.5990 |
| Epoch 4 | 1.2514 | 1.066e-05 | 0.6138 | 0.5329 | 0.5754 | 0.6332 |
| Epoch 5 | 1.1584 | 1.026e-05 | 0.6393 | 0.5085 | 0.5503 | 0.6035 |
| Epoch 6 | 1.1605 | 9.705e-06 | 0.7788 | 0.6755 | 0.7036 | 0.6632 |
| Epoch 7 | 1.0738 | 9.027e-06 | 0.5884 | 0.5127 | 0.6334 | 0.5649 |
| Epoch 8 | 1.0195 | 8.241e-06 | 0.5955 | 0.4274 | 0.4963 | 0.5847 |
| Epoch 9 | 1.0330 | 7.372e-06 | 0.5535 | 0.4885 | 0.5457 | 0.6000 |
| Epoch 10 | 0.9722 | 6.446e-06 | 0.6168 | 0.4987 | 0.5817 | 0.6147 |
| Epoch 11 | 0.9463 | 5.491e-06 | 0.6014 | 0.5197 | 0.5837 | 0.6370 |
| Epoch 12 | 0.9142 | 4.537e-06 | 0.6085 | 0.5505 | 0.6510 | 0.6218 |
| Epoch 13 | 0.8708 | 3.611e-06 | 0.6062 | 0.5396 | 0.6154 | 0.6352 |
| Epoch 14 | 0.8487 | 2.743e-06 | 0.6570 | 0.5760 | 0.6330 | 0.6591 |
| Epoch 15 | 0.8188 | 1.959e-06 | 0.6233 | 0.5197 | 0.5891 | 0.6405 |
| Epoch 16 | 0.7823 | 1.282e-06 | 0.6209 | 0.5200 | 0.5983 | 0.6415 |
| Epoch 17 | 0.7772 | 7.335e-07 | 0.6245 | 0.5281 | 0.6040 | 0.6596 |
| Epoch 18 | 0.7466 | 3.295e-07 | 0.6263 | 0.5359 | 0.6103 | 0.6439 |
| Epoch 19 | 0.7486 | 8.249e-08 | 0.6239 | 0.5333 | 0.6026 | 0.6464 |
| Epoch 20 | 0.7522 | 4.751e-11 | 0.6251 | 0.5356 | 0.6047 | 0.6468 |

Final epoch (Epoch 20), all 1691 validation rows: QWK (6 labels, summary.json) 0.4857 · QWK5 (AP-ROP folded onto stage 4/5, val_predictions.npz) 0.5953 · accuracy 0.6251 · macro-F1 0.5356 · balanced accuracy 0.6468 · AUC 0.8365

| final epoch | Normal | Stage 1 | Stage 2 | Stage 3 | Stage 4/5 | AP-ROP |
|---|---|---|---|---|---|---|
| recall | 0.9635 | 0.6204 | 0.4969 | 0.8527 | 0.8947 | 0.0525 |
| support | 767 | 137 | 163 | 129 | 19 | 476 |

#### Arm A · seed 1337 · fold_1 (CV fold 1)

Kernel `rop-v3-retfound-green-s1337-lr2e-4` · run dir `results/rop/v3/runs/retfound_green_s1337/fold_1` · as-run manifest `results/prepared/rop_staging/_v3_fold1.csv`

Fold file `results/prepared/rop_staging/fold_1.csv`: validation = 1252 photos of 299 infants (ostrava 812, ropvl 327, shenzhen 113); training = 2477 photos of 1228 infants (ostrava 527, ropvl 1346, shenzhen 604).

| rows | normal | stage_1 | stage_2 | stage_3 | stage_4_5 | arop | total |
|---|---|---|---|---|---|---|---|
| training rows | 890 | 401 | 534 | 442 | 73 | 137 | 2477 |
| validation rows | 756 | 114 | 220 | 116 | 16 | 30 | 1252 |

- images per epoch = 2477 (the training rows: the fold file's 2486 less 9 exact duplicate(s) dropped at run time)
- steps per epoch = 155 (= ceil(2477 / (16 x 1)), batch_size x grad_accum)
- minutes 30.6 (fold start to next fold start, kernel log) · 41.5 img/s · GPU Tesla T4 · kernel log reports train 2477 val 1252

Per epoch (`metrics.json` history; lr = the first optimizer group, i.e. the lowest layer under layer-wise decay, at the end of the epoch):

| Epoch | train loss | lr | val acc | val macro-F1 | val macro-P | val macro-R |
|---|---|---|---|---|---|---|
| Epoch 1 | 1.8630 | 5.744e-06 | 0.6765 | 0.4054 | 0.3853 | 0.4658 |
| Epoch 2 | 1.5677 | 1.100e-05 | 0.7069 | 0.5091 | 0.4584 | 0.6379 |
| Epoch 3 | 1.3609 | 1.091e-05 | 0.7292 | 0.5828 | 0.6131 | 0.6494 |
| Epoch 4 | 1.2348 | 1.066e-05 | 0.7796 | 0.5935 | 0.5920 | 0.6736 |
| Epoch 5 | 1.1465 | 1.026e-05 | 0.7644 | 0.6052 | 0.6388 | 0.6150 |
| Epoch 6 | 1.1009 | 9.705e-06 | 0.8043 | 0.6771 | 0.6928 | 0.6851 |
| Epoch 7 | 1.0755 | 9.027e-06 | 0.7812 | 0.6689 | 0.6858 | 0.7060 |
| Epoch 8 | 1.0177 | 8.241e-06 | 0.8059 | 0.7252 | 0.7892 | 0.6866 |
| Epoch 9 | 0.9754 | 7.372e-06 | 0.7947 | 0.6874 | 0.6833 | 0.7130 |
| Epoch 10 | 0.9270 | 6.446e-06 | 0.7891 | 0.6832 | 0.6743 | 0.7044 |
| Epoch 11 | 0.9056 | 5.491e-06 | 0.8035 | 0.6990 | 0.7116 | 0.7005 |
| Epoch 12 | 0.8697 | 4.537e-06 | 0.8043 | 0.7103 | 0.7251 | 0.7136 |
| Epoch 13 | 0.8197 | 3.611e-06 | 0.8131 | 0.7237 | 0.7336 | 0.7216 |
| Epoch 14 | 0.8093 | 2.743e-06 | 0.7947 | 0.7034 | 0.7030 | 0.7325 |
| Epoch 15 | 0.7622 | 1.959e-06 | 0.8147 | 0.7423 | 0.7822 | 0.7127 |
| Epoch 16 | 0.7671 | 1.282e-06 | 0.7923 | 0.7175 | 0.7228 | 0.7245 |
| Epoch 17 | 0.7449 | 7.335e-07 | 0.8059 | 0.7159 | 0.7336 | 0.7219 |
| Epoch 18 | 0.7403 | 3.295e-07 | 0.8123 | 0.7243 | 0.7396 | 0.7299 |
| Epoch 19 | 0.7420 | 8.249e-08 | 0.8139 | 0.7176 | 0.7279 | 0.7271 |
| Epoch 20 | 0.7293 | 4.747e-11 | 0.8115 | 0.7156 | 0.7261 | 0.7245 |

Final epoch (Epoch 20), all 1252 validation rows: QWK (6 labels, summary.json) 0.8732 · QWK5 (AP-ROP folded onto stage 4/5, val_predictions.npz) 0.8681 · accuracy 0.8115 · macro-F1 0.7156 · balanced accuracy 0.7245 · AUC 0.9407

| final epoch | Normal | Stage 1 | Stage 2 | Stage 3 | Stage 4/5 | AP-ROP |
|---|---|---|---|---|---|---|
| recall | 0.9775 | 0.4737 | 0.4773 | 0.6810 | 0.9375 | 0.8000 |
| support | 756 | 114 | 220 | 116 | 16 | 30 |

#### Arm A · seed 1337 · fold_2 (CV fold 2)

Kernel `rop-v3-retfound-green-s1337-lr2e-4` · run dir `results/rop/v3/runs/retfound_green_s1337/fold_2` · as-run manifest `results/prepared/rop_staging/_v3_fold2.csv`

Fold file `results/prepared/rop_staging/fold_2.csv`: validation = 1252 photos of 272 infants (ostrava 830, ropvl 295, shenzhen 127); training = 2542 photos of 1255 infants (ostrava 573, ropvl 1378, shenzhen 591).

| rows | normal | stage_1 | stage_2 | stage_3 | stage_4_5 | arop | total |
|---|---|---|---|---|---|---|---|
| training rows | 888 | 415 | 556 | 481 | 71 | 131 | 2542 |
| validation rows | 772 | 123 | 193 | 110 | 18 | 36 | 1252 |

- images per epoch = 2542 (the training rows: the fold file's 2553 less 11 exact duplicate(s) dropped at run time)
- steps per epoch = 159 (= ceil(2542 / (16 x 1)), batch_size x grad_accum)
- minutes 32.1 (fold start to next fold start, kernel log) · 41.1 img/s · GPU Tesla T4 · kernel log reports train 2542 val 1252

Per epoch (`metrics.json` history; lr = the first optimizer group, i.e. the lowest layer under layer-wise decay, at the end of the epoch):

| Epoch | train loss | lr | val acc | val macro-F1 | val macro-P | val macro-R |
|---|---|---|---|---|---|---|
| Epoch 1 | 1.8807 | 5.744e-06 | 0.4936 | 0.3244 | 0.3162 | 0.4142 |
| Epoch 2 | 1.5435 | 1.100e-05 | 0.7428 | 0.4599 | 0.5263 | 0.5217 |
| Epoch 3 | 1.3695 | 1.091e-05 | 0.6374 | 0.4676 | 0.5047 | 0.6006 |
| Epoch 4 | 1.2288 | 1.066e-05 | 0.7676 | 0.5615 | 0.5438 | 0.6340 |
| Epoch 5 | 1.1720 | 1.026e-05 | 0.7053 | 0.5638 | 0.6461 | 0.5654 |
| Epoch 6 | 1.1162 | 9.705e-06 | 0.5639 | 0.5696 | 0.6826 | 0.5625 |
| Epoch 7 | 1.0773 | 9.027e-06 | 0.7069 | 0.5407 | 0.5369 | 0.5834 |
| Epoch 8 | 1.0359 | 8.241e-06 | 0.7979 | 0.6563 | 0.7204 | 0.6699 |
| Epoch 9 | 0.9773 | 7.372e-06 | 0.7005 | 0.5805 | 0.6391 | 0.5920 |
| Epoch 10 | 0.9695 | 6.446e-06 | 0.6510 | 0.5915 | 0.6630 | 0.5997 |
| Epoch 11 | 0.9337 | 5.492e-06 | 0.7780 | 0.5855 | 0.6175 | 0.6474 |
| Epoch 12 | 0.8790 | 4.537e-06 | 0.7125 | 0.5483 | 0.6083 | 0.6448 |
| Epoch 13 | 0.8607 | 3.612e-06 | 0.7021 | 0.5703 | 0.5892 | 0.6553 |
| Epoch 14 | 0.8377 | 2.744e-06 | 0.7556 | 0.5909 | 0.6009 | 0.6640 |
| Epoch 15 | 0.8251 | 1.959e-06 | 0.7907 | 0.5989 | 0.6342 | 0.6672 |
| Epoch 16 | 0.8081 | 1.282e-06 | 0.7867 | 0.6130 | 0.6124 | 0.6683 |
| Epoch 17 | 0.7885 | 7.336e-07 | 0.7524 | 0.6034 | 0.6168 | 0.6718 |
| Epoch 18 | 0.7648 | 3.295e-07 | 0.7780 | 0.6149 | 0.6133 | 0.6784 |
| Epoch 19 | 0.7453 | 8.252e-08 | 0.7748 | 0.6094 | 0.6187 | 0.6778 |
| Epoch 20 | 0.7566 | 4.729e-11 | 0.7748 | 0.6100 | 0.6165 | 0.6786 |

Final epoch (Epoch 20), all 1252 validation rows: QWK (6 labels, summary.json) 0.8574 · QWK5 (AP-ROP folded onto stage 4/5, val_predictions.npz) 0.8818 · accuracy 0.7748 · macro-F1 0.6100 · balanced accuracy 0.6786 · AUC 0.8860

| final epoch | Normal | Stage 1 | Stage 2 | Stage 3 | Stage 4/5 | AP-ROP |
|---|---|---|---|---|---|---|
| recall | 0.9301 | 0.5447 | 0.5648 | 0.2545 | 0.8889 | 0.8889 |
| support | 772 | 123 | 193 | 110 | 18 | 36 |

#### Arm A · seed 1337 · fold_3 (CV fold 3)

Kernel `rop-v3-retfound-green-s1337-lr2e-4` · run dir `results/rop/v3/runs/retfound_green_s1337/fold_3` · as-run manifest `results/prepared/rop_staging/_v3_fold3.csv`

Fold file `results/prepared/rop_staging/fold_3.csv`: validation = 1252 photos of 327 infants (ostrava 713, ropvl 362, shenzhen 177); training = 2459 photos of 1200 infants (ostrava 609, ropvl 1313, shenzhen 537).

| rows | normal | stage_1 | stage_2 | stage_3 | stage_4_5 | arop | total |
|---|---|---|---|---|---|---|---|
| training rows | 888 | 409 | 564 | 400 | 71 | 127 | 2459 |
| validation rows | 766 | 111 | 182 | 134 | 18 | 41 | 1252 |

- images per epoch = 2459 (the training rows: the fold file's 2469 less 10 exact duplicate(s) dropped at run time)
- steps per epoch = 154 (= ceil(2459 / (16 x 1)), batch_size x grad_accum)
- minutes 31.2 (fold start to next fold start, kernel log) · 40.0 img/s · GPU Tesla T4 · kernel log reports train 2459 val 1252

Per epoch (`metrics.json` history; lr = the first optimizer group, i.e. the lowest layer under layer-wise decay, at the end of the epoch):

| Epoch | train loss | lr | val acc | val macro-F1 | val macro-P | val macro-R |
|---|---|---|---|---|---|---|
| Epoch 1 | 1.8813 | 5.744e-06 | 0.5743 | 0.2880 | 0.4735 | 0.3218 |
| Epoch 2 | 1.6337 | 1.100e-05 | 0.7141 | 0.6008 | 0.6041 | 0.6378 |
| Epoch 3 | 1.3986 | 1.091e-05 | 0.7516 | 0.6341 | 0.6412 | 0.6681 |
| Epoch 4 | 1.2648 | 1.066e-05 | 0.8003 | 0.6640 | 0.6402 | 0.7339 |
| Epoch 5 | 1.1738 | 1.026e-05 | 0.5831 | 0.6206 | 0.7079 | 0.6444 |
| Epoch 6 | 1.1069 | 9.705e-06 | 0.4481 | 0.5142 | 0.6586 | 0.5180 |
| Epoch 7 | 1.0968 | 9.027e-06 | 0.7157 | 0.6694 | 0.6857 | 0.6939 |
| Epoch 8 | 1.0329 | 8.241e-06 | 0.6494 | 0.6924 | 0.6985 | 0.7489 |
| Epoch 9 | 1.0124 | 7.372e-06 | 0.8586 | 0.7335 | 0.7927 | 0.7002 |
| Epoch 10 | 0.9779 | 6.446e-06 | 0.4497 | 0.5970 | 0.7048 | 0.6304 |
| Epoch 11 | 0.9496 | 5.491e-06 | 0.8355 | 0.7702 | 0.7841 | 0.7659 |
| Epoch 12 | 0.8934 | 4.537e-06 | 0.7292 | 0.7208 | 0.7411 | 0.7394 |
| Epoch 13 | 0.8789 | 3.611e-06 | 0.5439 | 0.6770 | 0.7370 | 0.7171 |
| Epoch 14 | 0.8363 | 2.743e-06 | 0.8075 | 0.7685 | 0.7832 | 0.7737 |
| Epoch 15 | 0.8067 | 1.959e-06 | 0.7835 | 0.7554 | 0.7643 | 0.7784 |
| Epoch 16 | 0.7844 | 1.282e-06 | 0.8251 | 0.7711 | 0.7822 | 0.7754 |
| Epoch 17 | 0.7778 | 7.335e-07 | 0.7676 | 0.7504 | 0.7667 | 0.7643 |
| Epoch 18 | 0.7427 | 3.295e-07 | 0.7708 | 0.7507 | 0.7563 | 0.7725 |
| Epoch 19 | 0.7481 | 8.249e-08 | 0.7796 | 0.7554 | 0.7683 | 0.7701 |
| Epoch 20 | 0.7370 | 4.751e-11 | 0.7796 | 0.7558 | 0.7682 | 0.7705 |

Final epoch (Epoch 20), all 1252 validation rows: QWK (6 labels, summary.json) 0.9123 · QWK5 (AP-ROP folded onto stage 4/5, val_predictions.npz) 0.9104 · accuracy 0.7796 · macro-F1 0.7558 · balanced accuracy 0.7705 · AUC 0.9502

| final epoch | Normal | Stage 1 | Stage 2 | Stage 3 | Stage 4/5 | AP-ROP |
|---|---|---|---|---|---|---|
| recall | 0.8003 | 0.5946 | 0.7088 | 0.8881 | 0.7778 | 0.8537 |
| support | 766 | 111 | 182 | 134 | 18 | 41 |

#### Arm A · seed 1337 · fold_4 (CV fold 4)

Kernel `rop-v3-retfound-green-s1337-lr2e-4` · run dir `results/rop/v3/runs/retfound_green_s1337/fold_4` · as-run manifest `results/prepared/rop_staging/_v3_fold4.csv`

Fold file `results/prepared/rop_staging/fold_4.csv`: validation = 1253 photos of 333 infants (ostrava 711, ropvl 381, shenzhen 161); training = 2452 photos of 1194 infants (ostrava 607, ropvl 1292, shenzhen 553).

| rows | normal | stage_1 | stage_2 | stage_3 | stage_4_5 | arop | total |
|---|---|---|---|---|---|---|---|
| training rows | 882 | 420 | 546 | 401 | 71 | 132 | 2452 |
| validation rows | 776 | 92 | 201 | 131 | 18 | 35 | 1253 |

- images per epoch = 2452 (the training rows: the fold file's 2466 less 14 exact duplicate(s) dropped at run time)
- steps per epoch = 154 (= ceil(2452 / (16 x 1)), batch_size x grad_accum)
- minutes 31.1 (fold start to next fold start, kernel log) · 40.2 img/s · GPU Tesla T4 · kernel log reports train 2452 val 1253

Per epoch (`metrics.json` history; lr = the first optimizer group, i.e. the lowest layer under layer-wise decay, at the end of the epoch):

| Epoch | train loss | lr | val acc | val macro-F1 | val macro-P | val macro-R |
|---|---|---|---|---|---|---|
| Epoch 1 | 1.8606 | 5.744e-06 | 0.6760 | 0.3850 | 0.4612 | 0.4971 |
| Epoch 2 | 1.5757 | 1.100e-05 | 0.4876 | 0.4430 | 0.5856 | 0.4835 |
| Epoch 3 | 1.3475 | 1.091e-05 | 0.7606 | 0.5596 | 0.5802 | 0.6092 |
| Epoch 4 | 1.2555 | 1.066e-05 | 0.8172 | 0.6380 | 0.6600 | 0.6845 |
| Epoch 5 | 1.1949 | 1.026e-05 | 0.7749 | 0.6091 | 0.6098 | 0.6466 |
| Epoch 6 | 1.1465 | 9.705e-06 | 0.7981 | 0.6239 | 0.5885 | 0.6913 |
| Epoch 7 | 1.1047 | 9.027e-06 | 0.8188 | 0.6643 | 0.6670 | 0.7092 |
| Epoch 8 | 1.0574 | 8.241e-06 | 0.8468 | 0.7082 | 0.7172 | 0.7051 |
| Epoch 9 | 1.0068 | 7.372e-06 | 0.7255 | 0.6620 | 0.6628 | 0.7224 |
| Epoch 10 | 0.9457 | 6.446e-06 | 0.8101 | 0.7239 | 0.7233 | 0.7541 |
| Epoch 11 | 0.9340 | 5.491e-06 | 0.8061 | 0.7438 | 0.7451 | 0.7658 |
| Epoch 12 | 0.9224 | 4.537e-06 | 0.8420 | 0.7239 | 0.7136 | 0.7393 |
| Epoch 13 | 0.8556 | 3.611e-06 | 0.6903 | 0.6759 | 0.6923 | 0.7441 |
| Epoch 14 | 0.8327 | 2.743e-06 | 0.8380 | 0.7613 | 0.7603 | 0.7894 |
| Epoch 15 | 0.8134 | 1.959e-06 | 0.8292 | 0.7460 | 0.7475 | 0.7854 |
| Epoch 16 | 0.7760 | 1.282e-06 | 0.8444 | 0.7658 | 0.7808 | 0.7801 |
| Epoch 17 | 0.7674 | 7.335e-07 | 0.8292 | 0.7606 | 0.7600 | 0.7860 |
| Epoch 18 | 0.7726 | 3.295e-07 | 0.8460 | 0.7648 | 0.7620 | 0.7945 |
| Epoch 19 | 0.7537 | 8.249e-08 | 0.8388 | 0.7705 | 0.7685 | 0.7956 |
| Epoch 20 | 0.7610 | 4.751e-11 | 0.8388 | 0.7705 | 0.7685 | 0.7956 |

Final epoch (Epoch 20), all 1253 validation rows: QWK (6 labels, summary.json) 0.9082 · QWK5 (AP-ROP folded onto stage 4/5, val_predictions.npz) 0.9164 · accuracy 0.8388 · macro-F1 0.7705 · balanced accuracy 0.7956 · AUC 0.9562

| final epoch | Normal | Stage 1 | Stage 2 | Stage 3 | Stage 4/5 | AP-ROP |
|---|---|---|---|---|---|---|
| recall | 0.9162 | 0.6522 | 0.6119 | 0.8473 | 0.8889 | 0.8571 |
| support | 776 | 92 | 201 | 131 | 18 | 35 |

#### Arm A · seed 1337 · fold_site (held-out source Shenzhen (the v3 site fold))

Kernel `rop-v3-retfound-green-s1337-lr2e-4` · run dir `results/rop/v3/runs/retfound_green_s1337/fold_site` · as-run manifest `results/prepared/rop_staging/_v3_foldsite.csv`

Fold file `results/prepared/rop_staging/fold_site.csv`: validation = 740 photos of 397 infants (shenzhen 740); training = 2388 photos of 1131 infants (ostrava 715, ropvl 1673).

| rows | normal | stage_1 | stage_2 | stage_3 | stage_4_5 | arop | total |
|---|---|---|---|---|---|---|---|
| training rows | 893 | 426 | 533 | 280 | 89 | 167 | 2388 |
| validation rows | 234 | 89 | 163 | 254 | 0 | 0 | 740 |

- images per epoch = 2388 (the training rows: the fold file's 2390 less 2 exact duplicate(s) dropped at run time)
- steps per epoch = 150 (= ceil(2388 / (16 x 1)), batch_size x grad_accum)
- minutes 27.9 (fold start to next fold start, kernel log) · 40.5 img/s · GPU Tesla T4 · kernel log reports train 2388 val 740

Per epoch (`metrics.json` history; lr = the first optimizer group, i.e. the lowest layer under layer-wise decay, at the end of the epoch):

| Epoch | train loss | lr | val acc | val macro-F1 | val macro-P | val macro-R |
|---|---|---|---|---|---|---|
| Epoch 1 | 1.8709 | 5.745e-06 | 0.2932 | 0.1537 | 0.2297 | 0.1557 |
| Epoch 2 | 1.6075 | 1.100e-05 | 0.7095 | 0.4499 | 0.5076 | 0.4651 |
| Epoch 3 | 1.3682 | 1.091e-05 | 0.6932 | 0.3645 | 0.4096 | 0.3740 |
| Epoch 4 | 1.3273 | 1.066e-05 | 0.7689 | 0.5233 | 0.5852 | 0.5436 |
| Epoch 5 | 1.2090 | 1.026e-05 | 0.7324 | 0.4051 | 0.5373 | 0.4073 |
| Epoch 6 | 1.1517 | 9.705e-06 | 0.7595 | 0.4811 | 0.5363 | 0.5004 |
| Epoch 7 | 1.1018 | 9.026e-06 | 0.7905 | 0.5014 | 0.5357 | 0.4834 |
| Epoch 8 | 1.0506 | 8.241e-06 | 0.8338 | 0.5292 | 0.5617 | 0.5151 |
| Epoch 9 | 1.0098 | 7.372e-06 | 0.8162 | 0.5297 | 0.5428 | 0.5265 |
| Epoch 10 | 0.9597 | 6.446e-06 | 0.8284 | 0.5067 | 0.5291 | 0.5033 |
| Epoch 11 | 0.9453 | 5.491e-06 | 0.8176 | 0.5168 | 0.5538 | 0.4996 |
| Epoch 12 | 0.9059 | 4.537e-06 | 0.8014 | 0.5043 | 0.5251 | 0.4993 |
| Epoch 13 | 0.8854 | 3.611e-06 | 0.8027 | 0.4779 | 0.5529 | 0.4688 |
| Epoch 14 | 0.8315 | 2.743e-06 | 0.8595 | 0.5548 | 0.5583 | 0.5541 |
| Epoch 15 | 0.8185 | 1.959e-06 | 0.8541 | 0.5404 | 0.5718 | 0.5265 |
| Epoch 16 | 0.7707 | 1.282e-06 | 0.8568 | 0.5472 | 0.5777 | 0.5344 |
| Epoch 17 | 0.7601 | 7.334e-07 | 0.8662 | 0.5578 | 0.5743 | 0.5462 |
| Epoch 18 | 0.7570 | 3.294e-07 | 0.8595 | 0.5506 | 0.5725 | 0.5372 |
| Epoch 19 | 0.7225 | 8.246e-08 | 0.8676 | 0.5594 | 0.5805 | 0.5466 |
| Epoch 20 | 0.7415 | 4.770e-11 | 0.8689 | 0.5612 | 0.5815 | 0.5485 |

Final epoch (Epoch 20), all 740 validation rows: QWK (6 labels, summary.json) 0.9479 · QWK5 (AP-ROP folded onto stage 4/5, val_predictions.npz) 0.9592 · accuracy 0.8689 · macro-F1 0.5612 · balanced accuracy 0.5485 · AUC n/a

| final epoch | Normal | Stage 1 | Stage 2 | Stage 3 | Stage 4/5 | AP-ROP |
|---|---|---|---|---|---|---|
| recall | 1.0000 | 0.6180 | 0.7791 | 0.8937 | 0.0000 | 0.0000 |
| support | 234 | 89 | 163 | 254 | 0 | 0 |

### Arm B: DINOv2 ViT-S

Run dirs: `results/rop/v3/runs/dinov2_s_s42`, `results/rop/v3/runs/dinov2_s_s1337`. Missing seeds of [42, 1337]: none.

Recipe (the as-run `config.yaml` of fold_0, seed 42; any fold whose config differs says so):

| parameter | value |
|---|---|
| model.arch | dinov2_s |
| preprocess.image_size | 392 |
| preprocess.normalize | imagenet |
| preprocess.clahe | True |
| train.lr | 0.0001 |
| train.layer_decay | 0.8 |
| train.weight_decay | 0.05 |
| train.epochs | 20 |
| train.batch_size | 16 |
| train.grad_accum | 1 (default) |
| train.label_smoothing | 0.05 |
| train.loss | ce |
| train.class_balanced | True |
| train.class_balanced_beta | 0.999 |
| train.weighted_sampler | False |
| train.strong_aug | True |
| train.site_randomization | True |
| train.amp | True |
| train.ema_decay | not set |
| train.head_warmup_epochs | not set |
| model.pretrained | True |
| model.dropout | 0.3 |
| model.drop_path | 0.1 |
| eval.early_stopping | False |
| eval.primary_metric | macro_f1 |
| train.aug_v4 | not set |
| train.bag_loss | not set |
| train.uda | not set |
| pool (summary.json) | v3 |
| config source | configs/rop_staging_dinov2.yaml |

<details><summary>Every key of that as-run config.yaml</summary>

| key | value |
|---|---|
| disease | rop_staging |
| seed | 42 |
| data.num_classes | 6 |
| data.class_names | [Normal, Stage 1, Stage 2, Stage 3, Stage 4/5, AP-ROP] |
| data.manifest | results/prepared/rop_staging/_v3_fold0.csv |
| data.presplit | True |
| data.group_col | patient_id |
| preprocess.image_size | 392 |
| preprocess.circle_crop | True |
| preprocess.letterbox | True |
| preprocess.ben_graham | False |
| preprocess.clahe | True |
| preprocess.normalize | imagenet |
| preprocess.cached | True |
| preprocess.cache_root | /tmp/_cache768 |
| model.arch | dinov2_s |
| model.weights_path | None |
| model.pretrained | True |
| model.head | classification |
| model.dropout | 0.3 |
| model.drop_path | 0.1 |
| train.epochs | 20 |
| train.batch_size | 16 |
| train.lr | 0.0001 |
| train.layer_decay | 0.8 |
| train.weight_decay | 0.05 |
| train.label_smoothing | 0.05 |
| train.amp | True |
| train.strong_aug | True |
| train.site_randomization | True |
| train.loss | ce |
| train.class_balanced | True |
| train.class_balanced_beta | 0.999 |
| train.weighted_sampler | False |
| train.num_workers | 4 |
| eval.early_stopping | False |
| eval.primary_metric | macro_f1 |
| eval.qwk_classes | [Normal, Stage 1, Stage 2, Stage 3, Stage 4/5] |
| eval.report_per_class_recall | True |
| eval.unvalidated_classes | [Stage 4/5, AP-ROP] |
| _source | configs/rop_staging_dinov2.yaml |

</details>

#### Arm B · seed 42: overview (8 folds: fold_0, fold_1, fold_2, fold_3, fold_4, fold_site, fold_loso_ostrava, fold_loso_ropvl)

| fold | train rows | val rows | images/epoch | steps/epoch | minutes | img/s | final acc | final macro-F1 | final QWK (6) | final QWK5 | AUC |
|---|---|---|---|---|---|---|---|---|---|---|---|
| fold_0 | 2462 | 1691 | 2462 | 154 | 36.7 | 29.7 | 0.6984 | 0.5971 | 0.6783 | 0.7429 | 0.9197 |
| fold_1 | 2477 | 1252 | 2477 | 155 | 32.7 | 32.7 | 0.8051 | 0.7390 | 0.8753 | 0.8702 | 0.9409 |
| fold_2 | 2542 | 1252 | 2542 | 159 | 33.6 | 32.7 | 0.7204 | 0.5976 | 0.8426 | 0.8534 | 0.8803 |
| fold_3 | 2459 | 1252 | 2459 | 154 | 33.3 | 32.6 | 0.7907 | 0.7637 | 0.8998 | 0.9040 | 0.9529 |
| fold_4 | 2452 | 1253 | 2452 | 154 | 33.7 | 31.4 | 0.8356 | 0.7242 | 0.8727 | 0.8977 | 0.9454 |
| fold_site | 2388 | 740 | 2388 | 150 | 30.0 | 31.2 | 0.8770 | 0.5578 | 0.9530 | 0.9600 | n/a |
| fold_loso_ostrava | 2383 | 4285 | 2383 | 149 | 47.2 | 30.9 | 0.7832 | 0.3996 | 0.6944 | 0.7219 | n/a |
| fold_loso_ropvl | 1425 | 1675 | 1425 | 90 | 24.3 | 32.0 | 0.6788 | 0.4543 | 0.7450 | 0.8079 | 0.7974 |

#### Arm B · seed 42 · fold_0 (CV fold 0)

Kernel `rop-v3-dinov2-s-s42-lr1e-4` · run dir `results/rop/v3/runs/dinov2_s_s42/fold_0` · as-run manifest `results/prepared/rop_staging/_v3_fold0.csv`

Fold file `results/prepared/rop_staging/fold_0.csv`: validation = 1691 photos of 297 infants (ostrava 1219, ropvl 310, shenzhen 162); training = 2462 photos of 1231 infants (ostrava 544, ropvl 1363, shenzhen 555).

| rows | normal | stage_1 | stage_2 | stage_3 | stage_4_5 | arop | total |
|---|---|---|---|---|---|---|---|
| training rows | 888 | 403 | 556 | 404 | 70 | 141 | 2462 |
| validation rows | 767 | 137 | 163 | 129 | 19 | 476 | 1691 |

- images per epoch = 2462 (the training rows: the fold file's 2474 less 12 exact duplicate(s) dropped at run time)
- steps per epoch = 154 (= ceil(2462 / (16 x 1)), batch_size x grad_accum)
- minutes 36.7 (fold start to next fold start, kernel log) · 29.7 img/s · GPU Tesla T4 · kernel log reports train 2462 val 1691

Per epoch (`metrics.json` history; lr = the first optimizer group, i.e. the lowest layer under layer-wise decay, at the end of the epoch):

| Epoch | train loss | lr | val acc | val macro-F1 | val macro-P | val macro-R |
|---|---|---|---|---|---|---|
| Epoch 1 | 1.9016 | 2.872e-06 | 0.5281 | 0.3273 | 0.3699 | 0.3595 |
| Epoch 2 | 1.7213 | 5.498e-06 | 0.7652 | 0.5861 | 0.5625 | 0.6743 |
| Epoch 3 | 1.3921 | 5.455e-06 | 0.7883 | 0.6387 | 0.6263 | 0.6769 |
| Epoch 4 | 1.2445 | 5.331e-06 | 0.4772 | 0.4634 | 0.5677 | 0.5997 |
| Epoch 5 | 1.1419 | 5.128e-06 | 0.5784 | 0.5228 | 0.6372 | 0.6493 |
| Epoch 6 | 1.0794 | 4.852e-06 | 0.6824 | 0.5724 | 0.6106 | 0.6899 |
| Epoch 7 | 1.0384 | 4.513e-06 | 0.6635 | 0.5749 | 0.6377 | 0.6282 |
| Epoch 8 | 0.9772 | 4.120e-06 | 0.6079 | 0.5644 | 0.6400 | 0.6347 |
| Epoch 9 | 0.9721 | 3.686e-06 | 0.6357 | 0.5257 | 0.5822 | 0.6385 |
| Epoch 10 | 0.9313 | 3.223e-06 | 0.6854 | 0.5769 | 0.6116 | 0.6805 |
| Epoch 11 | 0.8984 | 2.746e-06 | 0.7008 | 0.6255 | 0.6376 | 0.6859 |
| Epoch 12 | 0.8690 | 2.268e-06 | 0.6954 | 0.5863 | 0.6101 | 0.6985 |
| Epoch 13 | 0.8407 | 1.806e-06 | 0.6860 | 0.5823 | 0.6173 | 0.6879 |
| Epoch 14 | 0.8297 | 1.372e-06 | 0.6221 | 0.5697 | 0.6235 | 0.6676 |
| Epoch 15 | 0.8032 | 9.795e-07 | 0.7380 | 0.6442 | 0.6513 | 0.7180 |
| Epoch 16 | 0.7981 | 6.411e-07 | 0.6990 | 0.5950 | 0.6172 | 0.6769 |
| Epoch 17 | 0.7685 | 3.667e-07 | 0.6824 | 0.5870 | 0.6247 | 0.6929 |
| Epoch 18 | 0.7679 | 1.647e-07 | 0.6925 | 0.5963 | 0.6195 | 0.7033 |
| Epoch 19 | 0.7663 | 4.124e-08 | 0.6978 | 0.5978 | 0.6187 | 0.7013 |
| Epoch 20 | 0.7561 | 2.376e-11 | 0.6984 | 0.5971 | 0.6175 | 0.7007 |

Final epoch (Epoch 20), all 1691 validation rows: QWK (6 labels, summary.json) 0.6783 · QWK5 (AP-ROP folded onto stage 4/5, val_predictions.npz) 0.7429 · accuracy 0.6984 · macro-F1 0.5971 · balanced accuracy 0.7007 · AUC 0.9197

| final epoch | Normal | Stage 1 | Stage 2 | Stage 3 | Stage 4/5 | AP-ROP |
|---|---|---|---|---|---|---|
| recall | 0.9439 | 0.7007 | 0.4785 | 0.8605 | 0.8947 | 0.3256 |
| support | 767 | 137 | 163 | 129 | 19 | 476 |

#### Arm B · seed 42 · fold_1 (CV fold 1)

Kernel `rop-v3-dinov2-s-s42-lr1e-4` · run dir `results/rop/v3/runs/dinov2_s_s42/fold_1` · as-run manifest `results/prepared/rop_staging/_v3_fold1.csv`

Fold file `results/prepared/rop_staging/fold_1.csv`: validation = 1252 photos of 299 infants (ostrava 812, ropvl 327, shenzhen 113); training = 2477 photos of 1228 infants (ostrava 527, ropvl 1346, shenzhen 604).

| rows | normal | stage_1 | stage_2 | stage_3 | stage_4_5 | arop | total |
|---|---|---|---|---|---|---|---|
| training rows | 890 | 401 | 534 | 442 | 73 | 137 | 2477 |
| validation rows | 756 | 114 | 220 | 116 | 16 | 30 | 1252 |

- images per epoch = 2477 (the training rows: the fold file's 2486 less 9 exact duplicate(s) dropped at run time)
- steps per epoch = 155 (= ceil(2477 / (16 x 1)), batch_size x grad_accum)
- minutes 32.7 (fold start to next fold start, kernel log) · 32.7 img/s · GPU Tesla T4 · kernel log reports train 2477 val 1252

Per epoch (`metrics.json` history; lr = the first optimizer group, i.e. the lowest layer under layer-wise decay, at the end of the epoch):

| Epoch | train loss | lr | val acc | val macro-F1 | val macro-P | val macro-R |
|---|---|---|---|---|---|---|
| Epoch 1 | 1.8888 | 2.872e-06 | 0.6086 | 0.3431 | 0.4809 | 0.3632 |
| Epoch 2 | 1.6632 | 5.498e-06 | 0.7109 | 0.4632 | 0.5235 | 0.6022 |
| Epoch 3 | 1.3570 | 5.455e-06 | 0.7260 | 0.4922 | 0.5070 | 0.6232 |
| Epoch 4 | 1.2362 | 5.331e-06 | 0.7532 | 0.5792 | 0.7057 | 0.5968 |
| Epoch 5 | 1.1303 | 5.128e-06 | 0.7923 | 0.6837 | 0.7257 | 0.6682 |
| Epoch 6 | 1.0798 | 4.852e-06 | 0.8067 | 0.7031 | 0.7286 | 0.6960 |
| Epoch 7 | 1.0374 | 4.513e-06 | 0.8219 | 0.7276 | 0.7496 | 0.7216 |
| Epoch 8 | 0.9876 | 4.120e-06 | 0.7724 | 0.7078 | 0.7319 | 0.7102 |
| Epoch 9 | 0.9235 | 3.686e-06 | 0.7772 | 0.7007 | 0.7296 | 0.6979 |
| Epoch 10 | 0.9231 | 3.223e-06 | 0.8011 | 0.6848 | 0.6847 | 0.7117 |
| Epoch 11 | 0.9084 | 2.746e-06 | 0.7995 | 0.6978 | 0.7011 | 0.7000 |
| Epoch 12 | 0.8598 | 2.268e-06 | 0.7923 | 0.7094 | 0.7132 | 0.7319 |
| Epoch 13 | 0.8176 | 1.806e-06 | 0.7939 | 0.7259 | 0.7728 | 0.7075 |
| Epoch 14 | 0.7903 | 1.372e-06 | 0.8035 | 0.7260 | 0.7427 | 0.7303 |
| Epoch 15 | 0.7877 | 9.795e-07 | 0.8043 | 0.7373 | 0.7671 | 0.7216 |
| Epoch 16 | 0.7798 | 6.411e-07 | 0.8115 | 0.7420 | 0.7667 | 0.7292 |
| Epoch 17 | 0.7571 | 3.667e-07 | 0.8059 | 0.7392 | 0.7645 | 0.7322 |
| Epoch 18 | 0.7429 | 1.647e-07 | 0.8019 | 0.7355 | 0.7618 | 0.7276 |
| Epoch 19 | 0.7385 | 4.125e-08 | 0.8035 | 0.7344 | 0.7579 | 0.7288 |
| Epoch 20 | 0.7481 | 2.373e-11 | 0.8051 | 0.7390 | 0.7667 | 0.7296 |

Final epoch (Epoch 20), all 1252 validation rows: QWK (6 labels, summary.json) 0.8753 · QWK5 (AP-ROP folded onto stage 4/5, val_predictions.npz) 0.8702 · accuracy 0.8051 · macro-F1 0.7390 · balanced accuracy 0.7296 · AUC 0.9409

| final epoch | Normal | Stage 1 | Stage 2 | Stage 3 | Stage 4/5 | AP-ROP |
|---|---|---|---|---|---|---|
| recall | 0.9603 | 0.5175 | 0.4727 | 0.6897 | 0.9375 | 0.8000 |
| support | 756 | 114 | 220 | 116 | 16 | 30 |

#### Arm B · seed 42 · fold_2 (CV fold 2)

Kernel `rop-v3-dinov2-s-s42-lr1e-4` · run dir `results/rop/v3/runs/dinov2_s_s42/fold_2` · as-run manifest `results/prepared/rop_staging/_v3_fold2.csv`

Fold file `results/prepared/rop_staging/fold_2.csv`: validation = 1252 photos of 272 infants (ostrava 830, ropvl 295, shenzhen 127); training = 2542 photos of 1255 infants (ostrava 573, ropvl 1378, shenzhen 591).

| rows | normal | stage_1 | stage_2 | stage_3 | stage_4_5 | arop | total |
|---|---|---|---|---|---|---|---|
| training rows | 888 | 415 | 556 | 481 | 71 | 131 | 2542 |
| validation rows | 772 | 123 | 193 | 110 | 18 | 36 | 1252 |

- images per epoch = 2542 (the training rows: the fold file's 2553 less 11 exact duplicate(s) dropped at run time)
- steps per epoch = 159 (= ceil(2542 / (16 x 1)), batch_size x grad_accum)
- minutes 33.6 (fold start to next fold start, kernel log) · 32.7 img/s · GPU Tesla T4 · kernel log reports train 2542 val 1252

Per epoch (`metrics.json` history; lr = the first optimizer group, i.e. the lowest layer under layer-wise decay, at the end of the epoch):

| Epoch | train loss | lr | val acc | val macro-F1 | val macro-P | val macro-R |
|---|---|---|---|---|---|---|
| Epoch 1 | 1.9064 | 2.872e-06 | 0.5823 | 0.2241 | 0.2982 | 0.2632 |
| Epoch 2 | 1.6565 | 5.498e-06 | 0.7204 | 0.4321 | 0.5254 | 0.4163 |
| Epoch 3 | 1.3657 | 5.455e-06 | 0.5296 | 0.3923 | 0.4498 | 0.4186 |
| Epoch 4 | 1.1987 | 5.331e-06 | 0.6869 | 0.4823 | 0.5871 | 0.5489 |
| Epoch 5 | 1.1463 | 5.128e-06 | 0.5511 | 0.4857 | 0.5815 | 0.5570 |
| Epoch 6 | 1.0629 | 4.853e-06 | 0.6398 | 0.5441 | 0.5787 | 0.6322 |
| Epoch 7 | 1.0541 | 4.513e-06 | 0.7540 | 0.5821 | 0.5568 | 0.6576 |
| Epoch 8 | 0.9785 | 4.121e-06 | 0.6118 | 0.5547 | 0.6123 | 0.6059 |
| Epoch 9 | 0.9518 | 3.686e-06 | 0.6957 | 0.5918 | 0.6646 | 0.6607 |
| Epoch 10 | 0.9507 | 3.223e-06 | 0.6446 | 0.6064 | 0.6546 | 0.6561 |
| Epoch 11 | 0.9172 | 2.746e-06 | 0.7436 | 0.6261 | 0.6381 | 0.6762 |
| Epoch 12 | 0.8944 | 2.269e-06 | 0.5559 | 0.5300 | 0.5968 | 0.6200 |
| Epoch 13 | 0.8580 | 1.806e-06 | 0.6677 | 0.5994 | 0.6094 | 0.6532 |
| Epoch 14 | 0.8323 | 1.372e-06 | 0.7324 | 0.5839 | 0.5715 | 0.6646 |
| Epoch 15 | 0.8245 | 9.796e-07 | 0.7165 | 0.5909 | 0.5828 | 0.6591 |
| Epoch 16 | 0.7788 | 6.412e-07 | 0.7692 | 0.6263 | 0.6137 | 0.6779 |
| Epoch 17 | 0.7769 | 3.668e-07 | 0.7093 | 0.5939 | 0.5795 | 0.6687 |
| Epoch 18 | 0.7895 | 1.648e-07 | 0.7276 | 0.5924 | 0.5770 | 0.6775 |
| Epoch 19 | 0.7705 | 4.126e-08 | 0.7165 | 0.5944 | 0.5840 | 0.6730 |
| Epoch 20 | 0.7733 | 2.365e-11 | 0.7204 | 0.5976 | 0.5867 | 0.6741 |

Final epoch (Epoch 20), all 1252 validation rows: QWK (6 labels, summary.json) 0.8426 · QWK5 (AP-ROP folded onto stage 4/5, val_predictions.npz) 0.8534 · accuracy 0.7204 · macro-F1 0.5976 · balanced accuracy 0.6741 · AUC 0.8803

| final epoch | Normal | Stage 1 | Stage 2 | Stage 3 | Stage 4/5 | AP-ROP |
|---|---|---|---|---|---|---|
| recall | 0.8368 | 0.5122 | 0.5803 | 0.2818 | 0.8889 | 0.9444 |
| support | 772 | 123 | 193 | 110 | 18 | 36 |

#### Arm B · seed 42 · fold_3 (CV fold 3)

Kernel `rop-v3-dinov2-s-s42-lr1e-4` · run dir `results/rop/v3/runs/dinov2_s_s42/fold_3` · as-run manifest `results/prepared/rop_staging/_v3_fold3.csv`

Fold file `results/prepared/rop_staging/fold_3.csv`: validation = 1252 photos of 327 infants (ostrava 713, ropvl 362, shenzhen 177); training = 2459 photos of 1200 infants (ostrava 609, ropvl 1313, shenzhen 537).

| rows | normal | stage_1 | stage_2 | stage_3 | stage_4_5 | arop | total |
|---|---|---|---|---|---|---|---|
| training rows | 888 | 409 | 564 | 400 | 71 | 127 | 2459 |
| validation rows | 766 | 111 | 182 | 134 | 18 | 41 | 1252 |

- images per epoch = 2459 (the training rows: the fold file's 2469 less 10 exact duplicate(s) dropped at run time)
- steps per epoch = 154 (= ceil(2459 / (16 x 1)), batch_size x grad_accum)
- minutes 33.3 (fold start to next fold start, kernel log) · 32.6 img/s · GPU Tesla T4 · kernel log reports train 2459 val 1252

Per epoch (`metrics.json` history; lr = the first optimizer group, i.e. the lowest layer under layer-wise decay, at the end of the epoch):

| Epoch | train loss | lr | val acc | val macro-F1 | val macro-P | val macro-R |
|---|---|---|---|---|---|---|
| Epoch 1 | 1.9045 | 2.872e-06 | 0.6773 | 0.2552 | 0.4373 | 0.3059 |
| Epoch 2 | 1.7244 | 5.498e-06 | 0.7604 | 0.5459 | 0.6016 | 0.6134 |
| Epoch 3 | 1.3887 | 5.455e-06 | 0.7580 | 0.6339 | 0.6183 | 0.6945 |
| Epoch 4 | 1.2415 | 5.331e-06 | 0.6813 | 0.6391 | 0.6379 | 0.6734 |
| Epoch 5 | 1.1452 | 5.128e-06 | 0.7899 | 0.7371 | 0.7601 | 0.7441 |
| Epoch 6 | 1.0996 | 4.852e-06 | 0.7292 | 0.5843 | 0.5641 | 0.6693 |
| Epoch 7 | 1.0693 | 4.513e-06 | 0.8339 | 0.7493 | 0.7676 | 0.7415 |
| Epoch 8 | 0.9969 | 4.120e-06 | 0.8187 | 0.7584 | 0.7516 | 0.7840 |
| Epoch 9 | 0.9703 | 3.686e-06 | 0.5958 | 0.6780 | 0.6981 | 0.7221 |
| Epoch 10 | 0.9443 | 3.223e-06 | 0.7596 | 0.7307 | 0.7248 | 0.7748 |
| Epoch 11 | 0.8822 | 2.746e-06 | 0.8554 | 0.7640 | 0.7778 | 0.7589 |
| Epoch 12 | 0.8904 | 2.268e-06 | 0.8578 | 0.7726 | 0.7670 | 0.7824 |
| Epoch 13 | 0.8726 | 1.806e-06 | 0.8195 | 0.7382 | 0.7188 | 0.7631 |
| Epoch 14 | 0.8211 | 1.372e-06 | 0.8315 | 0.7794 | 0.7987 | 0.7797 |
| Epoch 15 | 0.7983 | 9.795e-07 | 0.7420 | 0.7476 | 0.7698 | 0.7734 |
| Epoch 16 | 0.7981 | 6.411e-07 | 0.7843 | 0.7677 | 0.7871 | 0.7817 |
| Epoch 17 | 0.7834 | 3.667e-07 | 0.7995 | 0.7677 | 0.7749 | 0.7870 |
| Epoch 18 | 0.8145 | 1.647e-07 | 0.7883 | 0.7644 | 0.7755 | 0.7803 |
| Epoch 19 | 0.7671 | 4.124e-08 | 0.7899 | 0.7623 | 0.7695 | 0.7806 |
| Epoch 20 | 0.7634 | 2.376e-11 | 0.7907 | 0.7637 | 0.7709 | 0.7824 |

Final epoch (Epoch 20), all 1252 validation rows: QWK (6 labels, summary.json) 0.8998 · QWK5 (AP-ROP folded onto stage 4/5, val_predictions.npz) 0.9040 · accuracy 0.7907 · macro-F1 0.7637 · balanced accuracy 0.7824 · AUC 0.9529

| final epoch | Normal | Stage 1 | Stage 2 | Stage 3 | Stage 4/5 | AP-ROP |
|---|---|---|---|---|---|---|
| recall | 0.8120 | 0.6126 | 0.7418 | 0.8657 | 0.8333 | 0.8293 |
| support | 766 | 111 | 182 | 134 | 18 | 41 |

#### Arm B · seed 42 · fold_4 (CV fold 4)

Kernel `rop-v3-dinov2-s-s42-lr1e-4` · run dir `results/rop/v3/runs/dinov2_s_s42/fold_4` · as-run manifest `results/prepared/rop_staging/_v3_fold4.csv`

Fold file `results/prepared/rop_staging/fold_4.csv`: validation = 1253 photos of 333 infants (ostrava 711, ropvl 381, shenzhen 161); training = 2452 photos of 1194 infants (ostrava 607, ropvl 1292, shenzhen 553).

| rows | normal | stage_1 | stage_2 | stage_3 | stage_4_5 | arop | total |
|---|---|---|---|---|---|---|---|
| training rows | 882 | 420 | 546 | 401 | 71 | 132 | 2452 |
| validation rows | 776 | 92 | 201 | 131 | 18 | 35 | 1253 |

- images per epoch = 2452 (the training rows: the fold file's 2466 less 14 exact duplicate(s) dropped at run time)
- steps per epoch = 154 (= ceil(2452 / (16 x 1)), batch_size x grad_accum)
- minutes 33.7 (fold start to next fold start, kernel log) · 31.4 img/s · GPU Tesla T4 · kernel log reports train 2452 val 1253

Per epoch (`metrics.json` history; lr = the first optimizer group, i.e. the lowest layer under layer-wise decay, at the end of the epoch):

| Epoch | train loss | lr | val acc | val macro-F1 | val macro-P | val macro-R |
|---|---|---|---|---|---|---|
| Epoch 1 | 1.9227 | 2.872e-06 | 0.6449 | 0.3119 | 0.3777 | 0.2980 |
| Epoch 2 | 1.7217 | 5.498e-06 | 0.7686 | 0.5219 | 0.5162 | 0.6157 |
| Epoch 3 | 1.4024 | 5.455e-06 | 0.7717 | 0.5466 | 0.5706 | 0.6492 |
| Epoch 4 | 1.2225 | 5.331e-06 | 0.7310 | 0.5445 | 0.5266 | 0.6205 |
| Epoch 5 | 1.1380 | 5.128e-06 | 0.6935 | 0.5569 | 0.5395 | 0.6819 |
| Epoch 6 | 1.0860 | 4.852e-06 | 0.7702 | 0.5942 | 0.6387 | 0.6621 |
| Epoch 7 | 1.0755 | 4.513e-06 | 0.6864 | 0.6252 | 0.6260 | 0.6999 |
| Epoch 8 | 0.9863 | 4.120e-06 | 0.8037 | 0.6727 | 0.6697 | 0.7386 |
| Epoch 9 | 0.9581 | 3.686e-06 | 0.8292 | 0.6833 | 0.6780 | 0.7475 |
| Epoch 10 | 0.9131 | 3.223e-06 | 0.8388 | 0.6822 | 0.6614 | 0.7445 |
| Epoch 11 | 0.9112 | 2.746e-06 | 0.6568 | 0.6577 | 0.6604 | 0.7015 |
| Epoch 12 | 0.9093 | 2.268e-06 | 0.7590 | 0.6821 | 0.6807 | 0.7540 |
| Epoch 13 | 0.8808 | 1.806e-06 | 0.8109 | 0.6974 | 0.6769 | 0.7618 |
| Epoch 14 | 0.8290 | 1.372e-06 | 0.8468 | 0.7189 | 0.6940 | 0.7590 |
| Epoch 15 | 0.8142 | 9.795e-07 | 0.8412 | 0.7079 | 0.6862 | 0.7700 |
| Epoch 16 | 0.7875 | 6.411e-07 | 0.8396 | 0.7120 | 0.6892 | 0.7684 |
| Epoch 17 | 0.7733 | 3.667e-07 | 0.8260 | 0.7247 | 0.7029 | 0.7674 |
| Epoch 18 | 0.7541 | 1.647e-07 | 0.8316 | 0.7194 | 0.6928 | 0.7772 |
| Epoch 19 | 0.7672 | 4.124e-08 | 0.8332 | 0.7230 | 0.6971 | 0.7788 |
| Epoch 20 | 0.7482 | 2.376e-11 | 0.8356 | 0.7242 | 0.6978 | 0.7795 |

Final epoch (Epoch 20), all 1253 validation rows: QWK (6 labels, summary.json) 0.8727 · QWK5 (AP-ROP folded onto stage 4/5, val_predictions.npz) 0.8977 · accuracy 0.8356 · macro-F1 0.7242 · balanced accuracy 0.7795 · AUC 0.9454

| final epoch | Normal | Stage 1 | Stage 2 | Stage 3 | Stage 4/5 | AP-ROP |
|---|---|---|---|---|---|---|
| recall | 0.9278 | 0.5761 | 0.5871 | 0.8397 | 0.8889 | 0.8571 |
| support | 776 | 92 | 201 | 131 | 18 | 35 |

#### Arm B · seed 42 · fold_site (held-out source Shenzhen (the v3 site fold))

Kernel `rop-v3-dinov2-s-s42-lr1e-4` · run dir `results/rop/v3/runs/dinov2_s_s42/fold_site` · as-run manifest `results/prepared/rop_staging/_v3_foldsite.csv`

Fold file `results/prepared/rop_staging/fold_site.csv`: validation = 740 photos of 397 infants (shenzhen 740); training = 2388 photos of 1131 infants (ostrava 715, ropvl 1673).

| rows | normal | stage_1 | stage_2 | stage_3 | stage_4_5 | arop | total |
|---|---|---|---|---|---|---|---|
| training rows | 893 | 426 | 533 | 280 | 89 | 167 | 2388 |
| validation rows | 234 | 89 | 163 | 254 | 0 | 0 | 740 |

- images per epoch = 2388 (the training rows: the fold file's 2390 less 2 exact duplicate(s) dropped at run time)
- steps per epoch = 150 (= ceil(2388 / (16 x 1)), batch_size x grad_accum)
- minutes 30.0 (fold start to next fold start, kernel log) · 31.2 img/s · GPU Tesla T4 · kernel log reports train 2388 val 740

Per epoch (`metrics.json` history; lr = the first optimizer group, i.e. the lowest layer under layer-wise decay, at the end of the epoch):

| Epoch | train loss | lr | val acc | val macro-F1 | val macro-P | val macro-R |
|---|---|---|---|---|---|---|
| Epoch 1 | 1.8897 | 2.873e-06 | 0.4527 | 0.2693 | 0.2327 | 0.3303 |
| Epoch 2 | 1.6874 | 5.498e-06 | 0.5338 | 0.2498 | 0.2841 | 0.2731 |
| Epoch 3 | 1.3942 | 5.455e-06 | 0.7689 | 0.5416 | 0.5890 | 0.5516 |
| Epoch 4 | 1.2495 | 5.331e-06 | 0.7608 | 0.5074 | 0.6346 | 0.5142 |
| Epoch 5 | 1.1732 | 5.128e-06 | 0.8216 | 0.4941 | 0.5265 | 0.4900 |
| Epoch 6 | 1.1106 | 4.852e-06 | 0.8176 | 0.4898 | 0.5759 | 0.4912 |
| Epoch 7 | 1.0648 | 4.513e-06 | 0.6608 | 0.3915 | 0.4912 | 0.3745 |
| Epoch 8 | 0.9965 | 4.120e-06 | 0.8392 | 0.5359 | 0.5379 | 0.5351 |
| Epoch 9 | 0.9626 | 3.686e-06 | 0.8595 | 0.5464 | 0.5681 | 0.5346 |
| Epoch 10 | 0.9382 | 3.223e-06 | 0.8203 | 0.5229 | 0.5679 | 0.5047 |
| Epoch 11 | 0.9218 | 2.746e-06 | 0.8595 | 0.6530 | 0.6776 | 0.6458 |
| Epoch 12 | 0.8941 | 2.268e-06 | 0.8703 | 0.5583 | 0.5740 | 0.5514 |
| Epoch 13 | 0.8579 | 1.806e-06 | 0.8703 | 0.6756 | 0.6830 | 0.6693 |
| Epoch 14 | 0.8090 | 1.372e-06 | 0.8824 | 0.5674 | 0.5793 | 0.5584 |
| Epoch 15 | 0.8044 | 9.795e-07 | 0.8527 | 0.5395 | 0.5687 | 0.5261 |
| Epoch 16 | 0.7944 | 6.411e-07 | 0.8595 | 0.5292 | 0.5823 | 0.5192 |
| Epoch 17 | 0.7720 | 3.667e-07 | 0.8797 | 0.5592 | 0.5861 | 0.5486 |
| Epoch 18 | 0.7724 | 1.647e-07 | 0.8703 | 0.5564 | 0.5721 | 0.5462 |
| Epoch 19 | 0.7679 | 4.123e-08 | 0.8811 | 0.5611 | 0.5858 | 0.5500 |
| Epoch 20 | 0.7609 | 2.385e-11 | 0.8770 | 0.5578 | 0.5764 | 0.5482 |

Final epoch (Epoch 20), all 740 validation rows: QWK (6 labels, summary.json) 0.9530 · QWK5 (AP-ROP folded onto stage 4/5, val_predictions.npz) 0.9600 · accuracy 0.8770 · macro-F1 0.5578 · balanced accuracy 0.5482 · AUC n/a

| final epoch | Normal | Stage 1 | Stage 2 | Stage 3 | Stage 4/5 | AP-ROP |
|---|---|---|---|---|---|---|
| recall | 1.0000 | 0.5618 | 0.8098 | 0.9173 | 0.0000 | 0.0000 |
| support | 234 | 89 | 163 | 254 | 0 | 0 |

#### Arm B · seed 42 · fold_loso_ostrava (held-out source Ostrava (leave-one-source-out))

Kernel `rop-v3-dinov2-s-s42-lr1e-4-loso` · run dir `results/rop/v3/runs/dinov2_s_s42/fold_loso_ostrava` · as-run manifest `results/prepared/rop_staging/_v3_foldloso_ostrava.csv`

Fold file `results/prepared/rop_staging/fold_loso_ostrava.csv`: validation = 4285 photos of 148 infants (ostrava 4285); training = 2383 photos of 1379 infants (ropvl 1673, shenzhen 710).

| rows | normal | stage_1 | stage_2 | stage_3 | stage_4_5 | arop | total |
|---|---|---|---|---|---|---|---|
| training rows | 839 | 322 | 494 | 492 | 89 | 147 | 2383 |
| validation rows | 2980 | 252 | 458 | 125 | 0 | 470 | 4285 |

- images per epoch = 2383 (the training rows: the fold file's 2397 less 14 exact duplicate(s) dropped at run time)
- steps per epoch = 149 (= ceil(2383 / (16 x 1)), batch_size x grad_accum)
- minutes 47.2 (fold start to next fold start, kernel log) · 30.9 img/s · GPU Tesla T4 · kernel log reports train 2383 val 4285

Per epoch (`metrics.json` history; lr = the first optimizer group, i.e. the lowest layer under layer-wise decay, at the end of the epoch):

| Epoch | train loss | lr | val acc | val macro-F1 | val macro-P | val macro-R |
|---|---|---|---|---|---|---|
| Epoch 1 | 1.8717 | 2.873e-06 | 0.6828 | 0.2417 | 0.2949 | 0.2449 |
| Epoch 2 | 1.5674 | 5.498e-06 | 0.7617 | 0.3341 | 0.4351 | 0.3578 |
| Epoch 3 | 1.2065 | 5.455e-06 | 0.7916 | 0.3887 | 0.4798 | 0.3883 |
| Epoch 4 | 1.1304 | 5.331e-06 | 0.7771 | 0.3655 | 0.4483 | 0.3782 |
| Epoch 5 | 1.0431 | 5.128e-06 | 0.7865 | 0.3705 | 0.4428 | 0.3707 |
| Epoch 6 | 0.9802 | 4.852e-06 | 0.7942 | 0.3731 | 0.4809 | 0.3579 |
| Epoch 7 | 0.9162 | 4.513e-06 | 0.8077 | 0.4099 | 0.4860 | 0.4061 |
| Epoch 8 | 0.9162 | 4.120e-06 | 0.7508 | 0.3639 | 0.4674 | 0.3990 |
| Epoch 9 | 0.8952 | 3.686e-06 | 0.8012 | 0.4169 | 0.5018 | 0.4156 |
| Epoch 10 | 0.8531 | 3.223e-06 | 0.8068 | 0.4154 | 0.4921 | 0.4195 |
| Epoch 11 | 0.8485 | 2.746e-06 | 0.7937 | 0.3948 | 0.4903 | 0.4051 |
| Epoch 12 | 0.8196 | 2.268e-06 | 0.7888 | 0.3905 | 0.4857 | 0.4063 |
| Epoch 13 | 0.7832 | 1.806e-06 | 0.7764 | 0.3743 | 0.4736 | 0.3871 |
| Epoch 14 | 0.7503 | 1.372e-06 | 0.7797 | 0.3913 | 0.4808 | 0.3998 |
| Epoch 15 | 0.7459 | 9.795e-07 | 0.7874 | 0.3937 | 0.4739 | 0.4048 |
| Epoch 16 | 0.7312 | 6.410e-07 | 0.7853 | 0.3996 | 0.4808 | 0.4031 |
| Epoch 17 | 0.7416 | 3.667e-07 | 0.7736 | 0.3875 | 0.4741 | 0.4027 |
| Epoch 18 | 0.6977 | 1.647e-07 | 0.7834 | 0.3971 | 0.4812 | 0.4058 |
| Epoch 19 | 0.7033 | 4.122e-08 | 0.7832 | 0.3986 | 0.4801 | 0.4083 |
| Epoch 20 | 0.7134 | 2.388e-11 | 0.7832 | 0.3996 | 0.4817 | 0.4098 |

Final epoch (Epoch 20), all 4285 validation rows: QWK (6 labels, summary.json) 0.6944 · QWK5 (AP-ROP folded onto stage 4/5, val_predictions.npz) 0.7219 · accuracy 0.7832 · macro-F1 0.3996 · balanced accuracy 0.4098 · AUC n/a

| final epoch | Normal | Stage 1 | Stage 2 | Stage 3 | Stage 4/5 | AP-ROP |
|---|---|---|---|---|---|---|
| recall | 0.9453 | 0.1468 | 0.2031 | 0.4000 | 0.0000 | 0.7638 |
| support | 2980 | 252 | 458 | 125 | 0 | 470 |

#### Arm B · seed 42 · fold_loso_ropvl (held-out source ROP-VL (leave-one-source-out))

Kernel `rop-v3-dinov2-s-s42-lr1e-4-loso` · run dir `results/rop/v3/runs/dinov2_s_s42/fold_loso_ropvl` · as-run manifest `results/prepared/rop_staging/_v3_foldloso_ropvl.csv`

Fold file `results/prepared/rop_staging/fold_loso_ropvl.csv`: validation = 1675 photos of 983 infants (ropvl 1675); training = 1425 photos of 544 infants (ostrava 715, shenzhen 710).

| rows | normal | stage_1 | stage_2 | stage_3 | stage_4_5 | arop | total |
|---|---|---|---|---|---|---|---|
| training rows | 486 | 276 | 351 | 292 | 0 | 20 | 1425 |
| validation rows | 623 | 236 | 338 | 241 | 89 | 148 | 1675 |

- validated with zero training rows: stage_4_5
- images per epoch = 1425 (the training rows: the fold file's 1437 less 12 exact duplicate(s) dropped at run time)
- steps per epoch = 90 (= ceil(1425 / (16 x 1)), batch_size x grad_accum)
- minutes 24.3 (fold start to next fold start, kernel log) · 32.0 img/s · GPU Tesla T4 · kernel log reports train 1425 val 1675

Per epoch (`metrics.json` history; lr = the first optimizer group, i.e. the lowest layer under layer-wise decay, at the end of the epoch):

| Epoch | train loss | lr | val acc | val macro-F1 | val macro-P | val macro-R |
|---|---|---|---|---|---|---|
| Epoch 1 | 1.8905 | 2.882e-06 | 0.1701 | 0.0649 | 0.1841 | 0.1784 |
| Epoch 2 | 1.7278 | 5.498e-06 | 0.3499 | 0.1839 | 0.2857 | 0.2620 |
| Epoch 3 | 1.4753 | 5.455e-06 | 0.6072 | 0.3412 | 0.3549 | 0.4075 |
| Epoch 4 | 1.2463 | 5.330e-06 | 0.5803 | 0.3161 | 0.2823 | 0.3910 |
| Epoch 5 | 1.2036 | 5.127e-06 | 0.6287 | 0.4300 | 0.4529 | 0.4624 |
| Epoch 6 | 1.0847 | 4.851e-06 | 0.6418 | 0.4250 | 0.3955 | 0.4684 |
| Epoch 7 | 1.0784 | 4.512e-06 | 0.6328 | 0.4333 | 0.4463 | 0.4752 |
| Epoch 8 | 1.0273 | 4.119e-06 | 0.6197 | 0.3625 | 0.3671 | 0.4203 |
| Epoch 9 | 0.9989 | 3.684e-06 | 0.6531 | 0.4403 | 0.4075 | 0.4849 |
| Epoch 10 | 0.9448 | 3.221e-06 | 0.6800 | 0.4512 | 0.4187 | 0.4945 |
| Epoch 11 | 0.9262 | 2.743e-06 | 0.6770 | 0.4532 | 0.5074 | 0.4941 |
| Epoch 12 | 0.9007 | 2.266e-06 | 0.6710 | 0.4368 | 0.4207 | 0.4804 |
| Epoch 13 | 0.8761 | 1.804e-06 | 0.6812 | 0.4545 | 0.4304 | 0.5066 |
| Epoch 14 | 0.8690 | 1.370e-06 | 0.6788 | 0.4554 | 0.4271 | 0.5034 |
| Epoch 15 | 0.8495 | 9.778e-07 | 0.6507 | 0.4344 | 0.4122 | 0.4911 |
| Epoch 16 | 0.8160 | 6.397e-07 | 0.6818 | 0.4575 | 0.4274 | 0.5054 |
| Epoch 17 | 0.8201 | 3.656e-07 | 0.6758 | 0.4530 | 0.4247 | 0.4987 |
| Epoch 18 | 0.7910 | 1.640e-07 | 0.6782 | 0.4524 | 0.4218 | 0.5018 |
| Epoch 19 | 0.8020 | 4.086e-08 | 0.6782 | 0.4538 | 0.4227 | 0.5019 |
| Epoch 20 | 0.8114 | 2.716e-11 | 0.6788 | 0.4543 | 0.4232 | 0.5024 |

Final epoch (Epoch 20), all 1675 validation rows: QWK (6 labels, summary.json) 0.7450 · QWK5 (AP-ROP folded onto stage 4/5, val_predictions.npz) 0.8079 · accuracy 0.6788 · macro-F1 0.4543 · balanced accuracy 0.5024 · AUC 0.7974

| final epoch | Normal | Stage 1 | Stage 2 | Stage 3 | Stage 4/5 | AP-ROP |
|---|---|---|---|---|---|---|
| recall | 0.9326 | 0.7331 | 0.5976 | 0.7510 | 0.0000 | 0.0000 |
| support | 623 | 236 | 338 | 241 | 89 | 148 |

#### Arm B · seed 1337: overview (8 folds: fold_0, fold_1, fold_2, fold_3, fold_4, fold_site, fold_loso_ostrava, fold_loso_ropvl)

| fold | train rows | val rows | images/epoch | steps/epoch | minutes | img/s | final acc | final macro-F1 | final QWK (6) | final QWK5 | AUC |
|---|---|---|---|---|---|---|---|---|---|---|---|
| fold_0 | 2462 | 1691 | 2462 | 154 | 34.3 | 30.8 | 0.6570 | 0.5602 | 0.5775 | 0.6664 | 0.8922 |
| fold_1 | 2477 | 1252 | 2477 | 155 | 32.3 | 32.9 | 0.8259 | 0.7430 | 0.8841 | 0.8788 | 0.9440 |
| fold_2 | 2542 | 1252 | 2542 | 159 | 32.7 | 32.4 | 0.7204 | 0.5617 | 0.8148 | 0.8571 | 0.8758 |
| fold_3 | 2459 | 1252 | 2459 | 154 | 31.8 | 32.2 | 0.8139 | 0.7688 | 0.9213 | 0.9224 | 0.9599 |
| fold_4 | 2452 | 1253 | 2452 | 154 | 31.7 | 32.0 | 0.8516 | 0.7455 | 0.8862 | 0.9085 | 0.9571 |
| fold_site | 2388 | 740 | 2388 | 150 | 28.6 | 30.9 | 0.8824 | 0.5655 | 0.9538 | 0.9619 | n/a |
| fold_loso_ostrava | 2383 | 4285 | 2383 | 149 | 49.7 | 29.2 | 0.7986 | 0.4050 | 0.7291 | 0.7574 | n/a |
| fold_loso_ropvl | 1425 | 1675 | 1425 | 90 | 26.1 | 29.4 | 0.6740 | 0.4506 | 0.7275 | 0.7916 | 0.7524 |

#### Arm B · seed 1337 · fold_0 (CV fold 0)

Kernel `rop-v3-dinov2-s-s1337-lr1e-4` · run dir `results/rop/v3/runs/dinov2_s_s1337/fold_0` · as-run manifest `results/prepared/rop_staging/_v3_fold0.csv`

Fold file `results/prepared/rop_staging/fold_0.csv`: validation = 1691 photos of 297 infants (ostrava 1219, ropvl 310, shenzhen 162); training = 2462 photos of 1231 infants (ostrava 544, ropvl 1363, shenzhen 555).

| rows | normal | stage_1 | stage_2 | stage_3 | stage_4_5 | arop | total |
|---|---|---|---|---|---|---|---|
| training rows | 888 | 403 | 556 | 404 | 70 | 141 | 2462 |
| validation rows | 767 | 137 | 163 | 129 | 19 | 476 | 1691 |

- images per epoch = 2462 (the training rows: the fold file's 2474 less 12 exact duplicate(s) dropped at run time)
- steps per epoch = 154 (= ceil(2462 / (16 x 1)), batch_size x grad_accum)
- minutes 34.3 (fold start to next fold start, kernel log) · 30.8 img/s · GPU Tesla T4 · kernel log reports train 2462 val 1691

Per epoch (`metrics.json` history; lr = the first optimizer group, i.e. the lowest layer under layer-wise decay, at the end of the epoch):

| Epoch | train loss | lr | val acc | val macro-F1 | val macro-P | val macro-R |
|---|---|---|---|---|---|---|
| Epoch 1 | 1.8674 | 2.872e-06 | 0.5103 | 0.2834 | 0.4355 | 0.3333 |
| Epoch 2 | 1.7142 | 5.498e-06 | 0.6576 | 0.4706 | 0.5804 | 0.5430 |
| Epoch 3 | 1.3863 | 5.455e-06 | 0.7150 | 0.6101 | 0.6133 | 0.6721 |
| Epoch 4 | 1.2184 | 5.331e-06 | 0.5517 | 0.5086 | 0.5743 | 0.6297 |
| Epoch 5 | 1.1487 | 5.128e-06 | 0.7025 | 0.6055 | 0.6263 | 0.6882 |
| Epoch 6 | 1.1233 | 4.852e-06 | 0.7853 | 0.7037 | 0.7128 | 0.7288 |
| Epoch 7 | 1.0480 | 4.513e-06 | 0.6062 | 0.4874 | 0.5437 | 0.5821 |
| Epoch 8 | 0.9868 | 4.120e-06 | 0.6091 | 0.4684 | 0.5369 | 0.6175 |
| Epoch 9 | 0.9860 | 3.686e-06 | 0.6523 | 0.5453 | 0.5935 | 0.6342 |
| Epoch 10 | 0.9427 | 3.223e-06 | 0.6582 | 0.4991 | 0.5502 | 0.6410 |
| Epoch 11 | 0.9374 | 2.746e-06 | 0.6251 | 0.5518 | 0.6202 | 0.6691 |
| Epoch 12 | 0.9115 | 2.268e-06 | 0.6339 | 0.5638 | 0.6195 | 0.6559 |
| Epoch 13 | 0.8666 | 1.806e-06 | 0.6511 | 0.5675 | 0.6210 | 0.6667 |
| Epoch 14 | 0.8450 | 1.372e-06 | 0.6677 | 0.5869 | 0.6127 | 0.6689 |
| Epoch 15 | 0.8111 | 9.795e-07 | 0.6629 | 0.5706 | 0.6023 | 0.6710 |
| Epoch 16 | 0.8157 | 6.411e-07 | 0.6434 | 0.5547 | 0.6054 | 0.6601 |
| Epoch 17 | 0.8058 | 3.667e-07 | 0.6357 | 0.5435 | 0.6031 | 0.6504 |
| Epoch 18 | 0.7657 | 1.647e-07 | 0.6582 | 0.5629 | 0.6075 | 0.6661 |
| Epoch 19 | 0.7640 | 4.124e-08 | 0.6546 | 0.5624 | 0.6079 | 0.6648 |
| Epoch 20 | 0.7801 | 2.376e-11 | 0.6570 | 0.5602 | 0.6038 | 0.6640 |

Final epoch (Epoch 20), all 1691 validation rows: QWK (6 labels, summary.json) 0.5775 · QWK5 (AP-ROP folded onto stage 4/5, val_predictions.npz) 0.6664 · accuracy 0.6570 · macro-F1 0.5602 · balanced accuracy 0.6640 · AUC 0.8922

| final epoch | Normal | Stage 1 | Stage 2 | Stage 3 | Stage 4/5 | AP-ROP |
|---|---|---|---|---|---|---|
| recall | 0.9648 | 0.6058 | 0.5153 | 0.8372 | 0.8947 | 0.1660 |
| support | 767 | 137 | 163 | 129 | 19 | 476 |

#### Arm B · seed 1337 · fold_1 (CV fold 1)

Kernel `rop-v3-dinov2-s-s1337-lr1e-4` · run dir `results/rop/v3/runs/dinov2_s_s1337/fold_1` · as-run manifest `results/prepared/rop_staging/_v3_fold1.csv`

Fold file `results/prepared/rop_staging/fold_1.csv`: validation = 1252 photos of 299 infants (ostrava 812, ropvl 327, shenzhen 113); training = 2477 photos of 1228 infants (ostrava 527, ropvl 1346, shenzhen 604).

| rows | normal | stage_1 | stage_2 | stage_3 | stage_4_5 | arop | total |
|---|---|---|---|---|---|---|---|
| training rows | 890 | 401 | 534 | 442 | 73 | 137 | 2477 |
| validation rows | 756 | 114 | 220 | 116 | 16 | 30 | 1252 |

- images per epoch = 2477 (the training rows: the fold file's 2486 less 9 exact duplicate(s) dropped at run time)
- steps per epoch = 155 (= ceil(2477 / (16 x 1)), batch_size x grad_accum)
- minutes 32.3 (fold start to next fold start, kernel log) · 32.9 img/s · GPU Tesla T4 · kernel log reports train 2477 val 1252

Per epoch (`metrics.json` history; lr = the first optimizer group, i.e. the lowest layer under layer-wise decay, at the end of the epoch):

| Epoch | train loss | lr | val acc | val macro-F1 | val macro-P | val macro-R |
|---|---|---|---|---|---|---|
| Epoch 1 | 1.8808 | 2.872e-06 | 0.6845 | 0.3856 | 0.3937 | 0.4232 |
| Epoch 2 | 1.6546 | 5.498e-06 | 0.6917 | 0.5393 | 0.6183 | 0.6049 |
| Epoch 3 | 1.3630 | 5.455e-06 | 0.7676 | 0.5985 | 0.6074 | 0.6640 |
| Epoch 4 | 1.2426 | 5.331e-06 | 0.6997 | 0.5636 | 0.5643 | 0.6660 |
| Epoch 5 | 1.1439 | 5.128e-06 | 0.7700 | 0.6337 | 0.6335 | 0.6443 |
| Epoch 6 | 1.0472 | 4.852e-06 | 0.8075 | 0.7276 | 0.7527 | 0.7169 |
| Epoch 7 | 1.0325 | 4.513e-06 | 0.7931 | 0.6163 | 0.6764 | 0.6381 |
| Epoch 8 | 0.9948 | 4.120e-06 | 0.8003 | 0.7010 | 0.6925 | 0.7279 |
| Epoch 9 | 0.9361 | 3.686e-06 | 0.7907 | 0.6760 | 0.6557 | 0.7045 |
| Epoch 10 | 0.9029 | 3.223e-06 | 0.8059 | 0.7031 | 0.6881 | 0.7356 |
| Epoch 11 | 0.8744 | 2.746e-06 | 0.8179 | 0.7202 | 0.7201 | 0.7362 |
| Epoch 12 | 0.8513 | 2.268e-06 | 0.8123 | 0.7295 | 0.7505 | 0.7218 |
| Epoch 13 | 0.8131 | 1.806e-06 | 0.8147 | 0.7096 | 0.7056 | 0.7282 |
| Epoch 14 | 0.8087 | 1.372e-06 | 0.8075 | 0.7112 | 0.7110 | 0.7323 |
| Epoch 15 | 0.7789 | 9.795e-07 | 0.8291 | 0.7419 | 0.7557 | 0.7393 |
| Epoch 16 | 0.7765 | 6.411e-07 | 0.8107 | 0.7223 | 0.7164 | 0.7423 |
| Epoch 17 | 0.7513 | 3.667e-07 | 0.8235 | 0.7360 | 0.7459 | 0.7494 |
| Epoch 18 | 0.7553 | 1.647e-07 | 0.8235 | 0.7416 | 0.7506 | 0.7493 |
| Epoch 19 | 0.7578 | 4.125e-08 | 0.8243 | 0.7370 | 0.7427 | 0.7489 |
| Epoch 20 | 0.7399 | 2.373e-11 | 0.8259 | 0.7430 | 0.7506 | 0.7518 |

Final epoch (Epoch 20), all 1252 validation rows: QWK (6 labels, summary.json) 0.8841 · QWK5 (AP-ROP folded onto stage 4/5, val_predictions.npz) 0.8788 · accuracy 0.8259 · macro-F1 0.7430 · balanced accuracy 0.7518 · AUC 0.9440

| final epoch | Normal | Stage 1 | Stage 2 | Stage 3 | Stage 4/5 | AP-ROP |
|---|---|---|---|---|---|---|
| recall | 0.9828 | 0.5175 | 0.4909 | 0.7155 | 0.9375 | 0.8667 |
| support | 756 | 114 | 220 | 116 | 16 | 30 |

#### Arm B · seed 1337 · fold_2 (CV fold 2)

Kernel `rop-v3-dinov2-s-s1337-lr1e-4` · run dir `results/rop/v3/runs/dinov2_s_s1337/fold_2` · as-run manifest `results/prepared/rop_staging/_v3_fold2.csv`

Fold file `results/prepared/rop_staging/fold_2.csv`: validation = 1252 photos of 272 infants (ostrava 830, ropvl 295, shenzhen 127); training = 2542 photos of 1255 infants (ostrava 573, ropvl 1378, shenzhen 591).

| rows | normal | stage_1 | stage_2 | stage_3 | stage_4_5 | arop | total |
|---|---|---|---|---|---|---|---|
| training rows | 888 | 415 | 556 | 481 | 71 | 131 | 2542 |
| validation rows | 772 | 123 | 193 | 110 | 18 | 36 | 1252 |

- images per epoch = 2542 (the training rows: the fold file's 2553 less 11 exact duplicate(s) dropped at run time)
- steps per epoch = 159 (= ceil(2542 / (16 x 1)), batch_size x grad_accum)
- minutes 32.7 (fold start to next fold start, kernel log) · 32.4 img/s · GPU Tesla T4 · kernel log reports train 2542 val 1252

Per epoch (`metrics.json` history; lr = the first optimizer group, i.e. the lowest layer under layer-wise decay, at the end of the epoch):

| Epoch | train loss | lr | val acc | val macro-F1 | val macro-P | val macro-R |
|---|---|---|---|---|---|---|
| Epoch 1 | 1.9037 | 2.872e-06 | 0.5391 | 0.2252 | 0.2927 | 0.3415 |
| Epoch 2 | 1.5976 | 5.498e-06 | 0.7452 | 0.4719 | 0.5548 | 0.5711 |
| Epoch 3 | 1.3050 | 5.455e-06 | 0.7540 | 0.5132 | 0.5020 | 0.6152 |
| Epoch 4 | 1.2007 | 5.331e-06 | 0.7468 | 0.5480 | 0.5738 | 0.6186 |
| Epoch 5 | 1.1210 | 5.128e-06 | 0.7220 | 0.5460 | 0.5588 | 0.5778 |
| Epoch 6 | 1.0684 | 4.853e-06 | 0.4944 | 0.4915 | 0.5419 | 0.5449 |
| Epoch 7 | 1.0309 | 4.513e-06 | 0.6869 | 0.5311 | 0.5133 | 0.5888 |
| Epoch 8 | 0.9962 | 4.121e-06 | 0.7268 | 0.5558 | 0.5925 | 0.5886 |
| Epoch 9 | 0.9653 | 3.686e-06 | 0.7460 | 0.6018 | 0.6473 | 0.6171 |
| Epoch 10 | 0.9517 | 3.223e-06 | 0.6326 | 0.5326 | 0.5559 | 0.5826 |
| Epoch 11 | 0.9268 | 2.746e-06 | 0.7620 | 0.5846 | 0.5866 | 0.6593 |
| Epoch 12 | 0.8692 | 2.269e-06 | 0.6142 | 0.5438 | 0.6068 | 0.6316 |
| Epoch 13 | 0.8643 | 1.806e-06 | 0.6677 | 0.5260 | 0.5612 | 0.6134 |
| Epoch 14 | 0.8192 | 1.372e-06 | 0.7268 | 0.5588 | 0.5612 | 0.6508 |
| Epoch 15 | 0.8275 | 9.796e-07 | 0.7276 | 0.5517 | 0.5708 | 0.6397 |
| Epoch 16 | 0.8062 | 6.412e-07 | 0.7348 | 0.5630 | 0.5786 | 0.6438 |
| Epoch 17 | 0.7936 | 3.668e-07 | 0.7029 | 0.5471 | 0.5552 | 0.6370 |
| Epoch 18 | 0.7691 | 1.648e-07 | 0.7133 | 0.5509 | 0.5609 | 0.6447 |
| Epoch 19 | 0.7571 | 4.126e-08 | 0.7236 | 0.5616 | 0.5743 | 0.6483 |
| Epoch 20 | 0.7632 | 2.365e-11 | 0.7204 | 0.5617 | 0.5746 | 0.6483 |

Final epoch (Epoch 20), all 1252 validation rows: QWK (6 labels, summary.json) 0.8148 · QWK5 (AP-ROP folded onto stage 4/5, val_predictions.npz) 0.8571 · accuracy 0.7204 · macro-F1 0.5617 · balanced accuracy 0.6483 · AUC 0.8758

| final epoch | Normal | Stage 1 | Stage 2 | Stage 3 | Stage 4/5 | AP-ROP |
|---|---|---|---|---|---|---|
| recall | 0.8744 | 0.5528 | 0.4301 | 0.2545 | 0.8889 | 0.8889 |
| support | 772 | 123 | 193 | 110 | 18 | 36 |

#### Arm B · seed 1337 · fold_3 (CV fold 3)

Kernel `rop-v3-dinov2-s-s1337-lr1e-4` · run dir `results/rop/v3/runs/dinov2_s_s1337/fold_3` · as-run manifest `results/prepared/rop_staging/_v3_fold3.csv`

Fold file `results/prepared/rop_staging/fold_3.csv`: validation = 1252 photos of 327 infants (ostrava 713, ropvl 362, shenzhen 177); training = 2459 photos of 1200 infants (ostrava 609, ropvl 1313, shenzhen 537).

| rows | normal | stage_1 | stage_2 | stage_3 | stage_4_5 | arop | total |
|---|---|---|---|---|---|---|---|
| training rows | 888 | 409 | 564 | 400 | 71 | 127 | 2459 |
| validation rows | 766 | 111 | 182 | 134 | 18 | 41 | 1252 |

- images per epoch = 2459 (the training rows: the fold file's 2469 less 10 exact duplicate(s) dropped at run time)
- steps per epoch = 154 (= ceil(2459 / (16 x 1)), batch_size x grad_accum)
- minutes 31.8 (fold start to next fold start, kernel log) · 32.2 img/s · GPU Tesla T4 · kernel log reports train 2459 val 1252

Per epoch (`metrics.json` history; lr = the first optimizer group, i.e. the lowest layer under layer-wise decay, at the end of the epoch):

| Epoch | train loss | lr | val acc | val macro-F1 | val macro-P | val macro-R |
|---|---|---|---|---|---|---|
| Epoch 1 | 1.8930 | 2.872e-06 | 0.5966 | 0.1878 | 0.2453 | 0.2557 |
| Epoch 2 | 1.7301 | 5.498e-06 | 0.6877 | 0.5018 | 0.4829 | 0.6134 |
| Epoch 3 | 1.3823 | 5.455e-06 | 0.7748 | 0.6798 | 0.6731 | 0.7188 |
| Epoch 4 | 1.2550 | 5.331e-06 | 0.7788 | 0.6495 | 0.6767 | 0.7300 |
| Epoch 5 | 1.1614 | 5.128e-06 | 0.7947 | 0.6913 | 0.6967 | 0.7364 |
| Epoch 6 | 1.0766 | 4.852e-06 | 0.6821 | 0.6880 | 0.7171 | 0.7119 |
| Epoch 7 | 1.0720 | 4.513e-06 | 0.7931 | 0.7367 | 0.7704 | 0.7361 |
| Epoch 8 | 1.0105 | 4.120e-06 | 0.8427 | 0.7605 | 0.7541 | 0.7860 |
| Epoch 9 | 0.9729 | 3.686e-06 | 0.8474 | 0.7741 | 0.7732 | 0.7781 |
| Epoch 10 | 0.9512 | 3.223e-06 | 0.6422 | 0.7009 | 0.7518 | 0.7260 |
| Epoch 11 | 0.9238 | 2.746e-06 | 0.8219 | 0.7578 | 0.7674 | 0.7597 |
| Epoch 12 | 0.8962 | 2.268e-06 | 0.7077 | 0.7325 | 0.7721 | 0.7399 |
| Epoch 13 | 0.8647 | 1.806e-06 | 0.6406 | 0.7086 | 0.7442 | 0.7446 |
| Epoch 14 | 0.8506 | 1.372e-06 | 0.7955 | 0.7615 | 0.7672 | 0.7813 |
| Epoch 15 | 0.8147 | 9.795e-07 | 0.8283 | 0.7739 | 0.7683 | 0.7953 |
| Epoch 16 | 0.7947 | 6.411e-07 | 0.8051 | 0.7729 | 0.7830 | 0.7861 |
| Epoch 17 | 0.7924 | 3.667e-07 | 0.7987 | 0.7666 | 0.7712 | 0.7846 |
| Epoch 18 | 0.7687 | 1.647e-07 | 0.8139 | 0.7648 | 0.7652 | 0.7838 |
| Epoch 19 | 0.7555 | 4.124e-08 | 0.8139 | 0.7686 | 0.7748 | 0.7835 |
| Epoch 20 | 0.7631 | 2.376e-11 | 0.8139 | 0.7688 | 0.7747 | 0.7842 |

Final epoch (Epoch 20), all 1252 validation rows: QWK (6 labels, summary.json) 0.9213 · QWK5 (AP-ROP folded onto stage 4/5, val_predictions.npz) 0.9224 · accuracy 0.8139 · macro-F1 0.7688 · balanced accuracy 0.7842 · AUC 0.9599

| final epoch | Normal | Stage 1 | Stage 2 | Stage 3 | Stage 4/5 | AP-ROP |
|---|---|---|---|---|---|---|
| recall | 0.8525 | 0.6396 | 0.7088 | 0.8731 | 0.7778 | 0.8537 |
| support | 766 | 111 | 182 | 134 | 18 | 41 |

#### Arm B · seed 1337 · fold_4 (CV fold 4)

Kernel `rop-v3-dinov2-s-s1337-lr1e-4` · run dir `results/rop/v3/runs/dinov2_s_s1337/fold_4` · as-run manifest `results/prepared/rop_staging/_v3_fold4.csv`

Fold file `results/prepared/rop_staging/fold_4.csv`: validation = 1253 photos of 333 infants (ostrava 711, ropvl 381, shenzhen 161); training = 2452 photos of 1194 infants (ostrava 607, ropvl 1292, shenzhen 553).

| rows | normal | stage_1 | stage_2 | stage_3 | stage_4_5 | arop | total |
|---|---|---|---|---|---|---|---|
| training rows | 882 | 420 | 546 | 401 | 71 | 132 | 2452 |
| validation rows | 776 | 92 | 201 | 131 | 18 | 35 | 1253 |

- images per epoch = 2452 (the training rows: the fold file's 2466 less 14 exact duplicate(s) dropped at run time)
- steps per epoch = 154 (= ceil(2452 / (16 x 1)), batch_size x grad_accum)
- minutes 31.7 (fold start to next fold start, kernel log) · 32.0 img/s · GPU Tesla T4 · kernel log reports train 2452 val 1253

Per epoch (`metrics.json` history; lr = the first optimizer group, i.e. the lowest layer under layer-wise decay, at the end of the epoch):

| Epoch | train loss | lr | val acc | val macro-F1 | val macro-P | val macro-R |
|---|---|---|---|---|---|---|
| Epoch 1 | 1.8840 | 2.872e-06 | 0.6409 | 0.4150 | 0.4610 | 0.4711 |
| Epoch 2 | 1.6646 | 5.498e-06 | 0.6480 | 0.4907 | 0.5355 | 0.4973 |
| Epoch 3 | 1.3970 | 5.455e-06 | 0.7845 | 0.5598 | 0.5998 | 0.6207 |
| Epoch 4 | 1.2056 | 5.331e-06 | 0.8013 | 0.6277 | 0.6348 | 0.6844 |
| Epoch 5 | 1.1708 | 5.128e-06 | 0.8101 | 0.6450 | 0.6513 | 0.6948 |
| Epoch 6 | 1.1177 | 4.852e-06 | 0.7023 | 0.5885 | 0.5641 | 0.6907 |
| Epoch 7 | 1.0643 | 4.513e-06 | 0.8117 | 0.6406 | 0.6221 | 0.7226 |
| Epoch 8 | 1.0225 | 4.120e-06 | 0.8595 | 0.7291 | 0.7056 | 0.7731 |
| Epoch 9 | 0.9718 | 3.686e-06 | 0.7941 | 0.7160 | 0.7111 | 0.7462 |
| Epoch 10 | 0.9124 | 3.223e-06 | 0.8476 | 0.7085 | 0.6802 | 0.7652 |
| Epoch 11 | 0.8981 | 2.746e-06 | 0.8348 | 0.7224 | 0.6969 | 0.7711 |
| Epoch 12 | 0.9069 | 2.268e-06 | 0.8579 | 0.7201 | 0.6946 | 0.7637 |
| Epoch 13 | 0.8469 | 1.806e-06 | 0.8053 | 0.6719 | 0.6539 | 0.7522 |
| Epoch 14 | 0.8345 | 1.372e-06 | 0.8005 | 0.7022 | 0.6845 | 0.7701 |
| Epoch 15 | 0.8127 | 9.795e-07 | 0.8508 | 0.7347 | 0.7085 | 0.7974 |
| Epoch 16 | 0.7740 | 6.411e-07 | 0.8308 | 0.7056 | 0.6803 | 0.7777 |
| Epoch 17 | 0.7927 | 3.667e-07 | 0.8436 | 0.7262 | 0.6983 | 0.7847 |
| Epoch 18 | 0.7732 | 1.647e-07 | 0.8571 | 0.7424 | 0.7144 | 0.7974 |
| Epoch 19 | 0.7523 | 4.124e-08 | 0.8500 | 0.7450 | 0.7161 | 0.7989 |
| Epoch 20 | 0.7733 | 2.376e-11 | 0.8516 | 0.7455 | 0.7168 | 0.7987 |

Final epoch (Epoch 20), all 1253 validation rows: QWK (6 labels, summary.json) 0.8862 · QWK5 (AP-ROP folded onto stage 4/5, val_predictions.npz) 0.9085 · accuracy 0.8516 · macro-F1 0.7455 · balanced accuracy 0.7987 · AUC 0.9571

| final epoch | Normal | Stage 1 | Stage 2 | Stage 3 | Stage 4/5 | AP-ROP |
|---|---|---|---|---|---|---|
| recall | 0.9356 | 0.5978 | 0.6368 | 0.8473 | 0.8889 | 0.8857 |
| support | 776 | 92 | 201 | 131 | 18 | 35 |

#### Arm B · seed 1337 · fold_site (held-out source Shenzhen (the v3 site fold))

Kernel `rop-v3-dinov2-s-s1337-lr1e-4` · run dir `results/rop/v3/runs/dinov2_s_s1337/fold_site` · as-run manifest `results/prepared/rop_staging/_v3_foldsite.csv`

Fold file `results/prepared/rop_staging/fold_site.csv`: validation = 740 photos of 397 infants (shenzhen 740); training = 2388 photos of 1131 infants (ostrava 715, ropvl 1673).

| rows | normal | stage_1 | stage_2 | stage_3 | stage_4_5 | arop | total |
|---|---|---|---|---|---|---|---|
| training rows | 893 | 426 | 533 | 280 | 89 | 167 | 2388 |
| validation rows | 234 | 89 | 163 | 254 | 0 | 0 | 740 |

- images per epoch = 2388 (the training rows: the fold file's 2390 less 2 exact duplicate(s) dropped at run time)
- steps per epoch = 150 (= ceil(2388 / (16 x 1)), batch_size x grad_accum)
- minutes 28.6 (fold start to next fold start, kernel log) · 30.9 img/s · GPU Tesla T4 · kernel log reports train 2388 val 740

Per epoch (`metrics.json` history; lr = the first optimizer group, i.e. the lowest layer under layer-wise decay, at the end of the epoch):

| Epoch | train loss | lr | val acc | val macro-F1 | val macro-P | val macro-R |
|---|---|---|---|---|---|---|
| Epoch 1 | 1.8791 | 2.873e-06 | 0.2959 | 0.1664 | 0.2631 | 0.1587 |
| Epoch 2 | 1.6719 | 5.498e-06 | 0.6973 | 0.4206 | 0.5234 | 0.4439 |
| Epoch 3 | 1.3685 | 5.455e-06 | 0.6108 | 0.3287 | 0.4547 | 0.3303 |
| Epoch 4 | 1.2599 | 5.331e-06 | 0.8176 | 0.5824 | 0.6473 | 0.5830 |
| Epoch 5 | 1.2061 | 5.128e-06 | 0.7216 | 0.3990 | 0.4989 | 0.3991 |
| Epoch 6 | 1.0987 | 4.852e-06 | 0.8230 | 0.5809 | 0.6507 | 0.5767 |
| Epoch 7 | 1.0362 | 4.513e-06 | 0.8405 | 0.5444 | 0.5595 | 0.5322 |
| Epoch 8 | 1.0365 | 4.120e-06 | 0.8595 | 0.6528 | 0.6915 | 0.6436 |
| Epoch 9 | 0.9773 | 3.686e-06 | 0.8541 | 0.5535 | 0.5653 | 0.5437 |
| Epoch 10 | 0.9411 | 3.223e-06 | 0.8500 | 0.5197 | 0.5668 | 0.5103 |
| Epoch 11 | 0.9230 | 2.746e-06 | 0.8500 | 0.5351 | 0.5666 | 0.5211 |
| Epoch 12 | 0.8776 | 2.268e-06 | 0.8365 | 0.5406 | 0.5562 | 0.5312 |
| Epoch 13 | 0.8698 | 1.806e-06 | 0.8662 | 0.5598 | 0.5786 | 0.5464 |
| Epoch 14 | 0.8398 | 1.372e-06 | 0.8878 | 0.6873 | 0.6899 | 0.6851 |
| Epoch 15 | 0.8234 | 9.795e-07 | 0.8676 | 0.5504 | 0.5730 | 0.5386 |
| Epoch 16 | 0.7793 | 6.411e-07 | 0.8838 | 0.5640 | 0.5803 | 0.5545 |
| Epoch 17 | 0.7696 | 3.667e-07 | 0.8797 | 0.5603 | 0.5778 | 0.5500 |
| Epoch 18 | 0.7828 | 1.647e-07 | 0.8770 | 0.5614 | 0.5790 | 0.5502 |
| Epoch 19 | 0.7492 | 4.123e-08 | 0.8838 | 0.5648 | 0.5790 | 0.5555 |
| Epoch 20 | 0.7678 | 2.385e-11 | 0.8824 | 0.5655 | 0.5793 | 0.5560 |

Final epoch (Epoch 20), all 740 validation rows: QWK (6 labels, summary.json) 0.9538 · QWK5 (AP-ROP folded onto stage 4/5, val_predictions.npz) 0.9619 · accuracy 0.8824 · macro-F1 0.5655 · balanced accuracy 0.5560 · AUC n/a

| final epoch | Normal | Stage 1 | Stage 2 | Stage 3 | Stage 4/5 | AP-ROP |
|---|---|---|---|---|---|---|
| recall | 0.9915 | 0.6180 | 0.7975 | 0.9291 | 0.0000 | 0.0000 |
| support | 234 | 89 | 163 | 254 | 0 | 0 |

#### Arm B · seed 1337 · fold_loso_ostrava (held-out source Ostrava (leave-one-source-out))

Kernel `rop-v3-dinov2-s-s1337-lr1e-4-loso` · run dir `results/rop/v3/runs/dinov2_s_s1337/fold_loso_ostrava` · as-run manifest `results/prepared/rop_staging/_v3_foldloso_ostrava.csv`

Fold file `results/prepared/rop_staging/fold_loso_ostrava.csv`: validation = 4285 photos of 148 infants (ostrava 4285); training = 2383 photos of 1379 infants (ropvl 1673, shenzhen 710).

| rows | normal | stage_1 | stage_2 | stage_3 | stage_4_5 | arop | total |
|---|---|---|---|---|---|---|---|
| training rows | 839 | 322 | 494 | 492 | 89 | 147 | 2383 |
| validation rows | 2980 | 252 | 458 | 125 | 0 | 470 | 4285 |

- images per epoch = 2383 (the training rows: the fold file's 2397 less 14 exact duplicate(s) dropped at run time)
- steps per epoch = 149 (= ceil(2383 / (16 x 1)), batch_size x grad_accum)
- minutes 49.7 (fold start to next fold start, kernel log) · 29.2 img/s · GPU Tesla T4 · kernel log reports train 2383 val 4285

Per epoch (`metrics.json` history; lr = the first optimizer group, i.e. the lowest layer under layer-wise decay, at the end of the epoch):

| Epoch | train loss | lr | val acc | val macro-F1 | val macro-P | val macro-R |
|---|---|---|---|---|---|---|
| Epoch 1 | 1.8625 | 2.873e-06 | 0.6964 | 0.1735 | 0.1872 | 0.2121 |
| Epoch 2 | 1.5706 | 5.498e-06 | 0.8058 | 0.3578 | 0.4000 | 0.3639 |
| Epoch 3 | 1.1928 | 5.455e-06 | 0.7846 | 0.3565 | 0.4827 | 0.3914 |
| Epoch 4 | 1.0692 | 5.331e-06 | 0.7617 | 0.4005 | 0.4832 | 0.4171 |
| Epoch 5 | 1.0283 | 5.128e-06 | 0.7281 | 0.3365 | 0.4556 | 0.3683 |
| Epoch 6 | 0.9644 | 4.852e-06 | 0.7139 | 0.3582 | 0.4497 | 0.3993 |
| Epoch 7 | 0.9231 | 4.513e-06 | 0.7781 | 0.3791 | 0.4386 | 0.3935 |
| Epoch 8 | 0.9046 | 4.120e-06 | 0.7881 | 0.3900 | 0.4669 | 0.3878 |
| Epoch 9 | 0.9034 | 3.686e-06 | 0.7904 | 0.3767 | 0.4432 | 0.3609 |
| Epoch 10 | 0.8607 | 3.223e-06 | 0.7804 | 0.3887 | 0.4840 | 0.3982 |
| Epoch 11 | 0.8240 | 2.746e-06 | 0.8019 | 0.4009 | 0.4965 | 0.4041 |
| Epoch 12 | 0.8105 | 2.268e-06 | 0.7876 | 0.3838 | 0.4857 | 0.3973 |
| Epoch 13 | 0.7819 | 1.806e-06 | 0.7872 | 0.3926 | 0.4805 | 0.3937 |
| Epoch 14 | 0.7487 | 1.372e-06 | 0.8051 | 0.4048 | 0.4893 | 0.4050 |
| Epoch 15 | 0.7569 | 9.795e-07 | 0.7928 | 0.4021 | 0.5050 | 0.4185 |
| Epoch 16 | 0.7186 | 6.410e-07 | 0.8098 | 0.4109 | 0.5004 | 0.4097 |
| Epoch 17 | 0.7185 | 3.667e-07 | 0.8030 | 0.4071 | 0.4975 | 0.4103 |
| Epoch 18 | 0.7394 | 1.647e-07 | 0.8000 | 0.4048 | 0.5033 | 0.4086 |
| Epoch 19 | 0.7047 | 4.122e-08 | 0.7977 | 0.4044 | 0.5016 | 0.4090 |
| Epoch 20 | 0.6961 | 2.388e-11 | 0.7986 | 0.4050 | 0.5021 | 0.4095 |

Final epoch (Epoch 20), all 4285 validation rows: QWK (6 labels, summary.json) 0.7291 · QWK5 (AP-ROP folded onto stage 4/5, val_predictions.npz) 0.7574 · accuracy 0.7986 · macro-F1 0.4050 · balanced accuracy 0.4095 · AUC n/a

| final epoch | Normal | Stage 1 | Stage 2 | Stage 3 | Stage 4/5 | AP-ROP |
|---|---|---|---|---|---|---|
| recall | 0.9614 | 0.1389 | 0.2052 | 0.3280 | 0.0000 | 0.8234 |
| support | 2980 | 252 | 458 | 125 | 0 | 470 |

#### Arm B · seed 1337 · fold_loso_ropvl (held-out source ROP-VL (leave-one-source-out))

Kernel `rop-v3-dinov2-s-s1337-lr1e-4-loso` · run dir `results/rop/v3/runs/dinov2_s_s1337/fold_loso_ropvl` · as-run manifest `results/prepared/rop_staging/_v3_foldloso_ropvl.csv`

Fold file `results/prepared/rop_staging/fold_loso_ropvl.csv`: validation = 1675 photos of 983 infants (ropvl 1675); training = 1425 photos of 544 infants (ostrava 715, shenzhen 710).

| rows | normal | stage_1 | stage_2 | stage_3 | stage_4_5 | arop | total |
|---|---|---|---|---|---|---|---|
| training rows | 486 | 276 | 351 | 292 | 0 | 20 | 1425 |
| validation rows | 623 | 236 | 338 | 241 | 89 | 148 | 1675 |

- validated with zero training rows: stage_4_5
- images per epoch = 1425 (the training rows: the fold file's 1437 less 12 exact duplicate(s) dropped at run time)
- steps per epoch = 90 (= ceil(1425 / (16 x 1)), batch_size x grad_accum)
- minutes 26.1 (fold start to next fold start, kernel log) · 29.4 img/s · GPU Tesla T4 · kernel log reports train 1425 val 1675

Per epoch (`metrics.json` history; lr = the first optimizer group, i.e. the lowest layer under layer-wise decay, at the end of the epoch):

| Epoch | train loss | lr | val acc | val macro-F1 | val macro-P | val macro-R |
|---|---|---|---|---|---|---|
| Epoch 1 | 1.8306 | 2.882e-06 | 0.2591 | 0.1325 | 0.1991 | 0.2161 |
| Epoch 2 | 1.6387 | 5.498e-06 | 0.4627 | 0.2011 | 0.2348 | 0.2981 |
| Epoch 3 | 1.4556 | 5.455e-06 | 0.5206 | 0.3202 | 0.3191 | 0.3687 |
| Epoch 4 | 1.3155 | 5.330e-06 | 0.6328 | 0.3907 | 0.4366 | 0.4426 |
| Epoch 5 | 1.1759 | 5.127e-06 | 0.6430 | 0.3826 | 0.4346 | 0.4317 |
| Epoch 6 | 1.1101 | 4.851e-06 | 0.6328 | 0.4114 | 0.3856 | 0.4651 |
| Epoch 7 | 1.0823 | 4.512e-06 | 0.6263 | 0.3914 | 0.4272 | 0.4414 |
| Epoch 8 | 1.0253 | 4.119e-06 | 0.6722 | 0.4478 | 0.5011 | 0.4907 |
| Epoch 9 | 0.9896 | 3.684e-06 | 0.6621 | 0.4433 | 0.4710 | 0.4816 |
| Epoch 10 | 0.9619 | 3.221e-06 | 0.6322 | 0.4261 | 0.4758 | 0.4655 |
| Epoch 11 | 0.9296 | 2.743e-06 | 0.6710 | 0.4467 | 0.4687 | 0.4877 |
| Epoch 12 | 0.9169 | 2.266e-06 | 0.6621 | 0.4476 | 0.4640 | 0.4981 |
| Epoch 13 | 0.9198 | 1.804e-06 | 0.6597 | 0.4370 | 0.4736 | 0.4868 |
| Epoch 14 | 0.8699 | 1.370e-06 | 0.6579 | 0.4376 | 0.5193 | 0.4914 |
| Epoch 15 | 0.8291 | 9.778e-07 | 0.6704 | 0.4472 | 0.4708 | 0.4956 |
| Epoch 16 | 0.8185 | 6.397e-07 | 0.6603 | 0.4411 | 0.4493 | 0.4877 |
| Epoch 17 | 0.8102 | 3.656e-07 | 0.6651 | 0.4439 | 0.4527 | 0.4898 |
| Epoch 18 | 0.8135 | 1.640e-07 | 0.6687 | 0.4456 | 0.4528 | 0.4904 |
| Epoch 19 | 0.7948 | 4.086e-08 | 0.6722 | 0.4491 | 0.4487 | 0.4938 |
| Epoch 20 | 0.8079 | 2.716e-11 | 0.6740 | 0.4506 | 0.4498 | 0.4950 |

Final epoch (Epoch 20), all 1675 validation rows: QWK (6 labels, summary.json) 0.7275 · QWK5 (AP-ROP folded onto stage 4/5, val_predictions.npz) 0.7916 · accuracy 0.6740 · macro-F1 0.4506 · balanced accuracy 0.4950 · AUC 0.7524

| final epoch | Normal | Stage 1 | Stage 2 | Stage 3 | Stage 4/5 | AP-ROP |
|---|---|---|---|---|---|---|
| recall | 0.9390 | 0.7373 | 0.6065 | 0.6805 | 0.0000 | 0.0068 |
| support | 623 | 236 | 338 | 241 | 89 | 148 |

### Arm C: EfficientNetV2-S (baseline)

No run directory: `flat_natural EfficientNetV2-S (saved npz)` (seeds [42, 1337]), scored by `scripts/rop_v3_score.py` from saved predictions; its numbers are in section 4.

### Arm D: DINOv2 + exam-level loss

Run dirs: `results/rop/v3/runs/dinov2_s_bag_s42`, `results/rop/v3/runs/dinov2_s_bag_s1337`. Missing seeds of [42, 1337]: none.

Recipe (the as-run `config.yaml` of fold_0, seed 42; any fold whose config differs says so):

| parameter | value |
|---|---|
| model.arch | dinov2_s |
| preprocess.image_size | 392 |
| preprocess.normalize | imagenet |
| preprocess.clahe | True |
| train.lr | 0.0001 |
| train.layer_decay | 0.8 |
| train.weight_decay | 0.05 |
| train.epochs | 20 |
| train.batch_size | 16 |
| train.grad_accum | 1 (default) |
| train.label_smoothing | 0.05 |
| train.loss | ce |
| train.class_balanced | True |
| train.class_balanced_beta | 0.999 |
| train.weighted_sampler | False |
| train.strong_aug | True |
| train.site_randomization | True |
| train.amp | True |
| train.ema_decay | not set |
| train.head_warmup_epochs | not set |
| model.pretrained | True |
| model.dropout | 0.3 |
| model.drop_path | 0.1 |
| eval.early_stopping | False |
| eval.primary_metric | macro_f1 |
| train.aug_v4 | not set |
| train.bag_loss.enabled | True |
| train.bag_loss.source | ostrava |
| train.bag_loss.k | 8 |
| train.bag_loss.tau | 1 |
| train.uda | not set |
| pool (summary.json) | v3 |
| config source | configs/rop_staging_dinov2_bag.yaml |

<details><summary>Every key of that as-run config.yaml</summary>

| key | value |
|---|---|
| disease | rop_staging |
| seed | 42 |
| data.num_classes | 6 |
| data.class_names | [Normal, Stage 1, Stage 2, Stage 3, Stage 4/5, AP-ROP] |
| data.manifest | results/prepared/rop_staging/_v3_fold0.csv |
| data.presplit | True |
| data.group_col | patient_id |
| preprocess.image_size | 392 |
| preprocess.circle_crop | True |
| preprocess.letterbox | True |
| preprocess.ben_graham | False |
| preprocess.clahe | True |
| preprocess.normalize | imagenet |
| preprocess.cached | True |
| preprocess.cache_root | /tmp/_cache768 |
| model.arch | dinov2_s |
| model.weights_path | None |
| model.pretrained | True |
| model.head | classification |
| model.dropout | 0.3 |
| model.drop_path | 0.1 |
| train.epochs | 20 |
| train.batch_size | 16 |
| train.lr | 0.0001 |
| train.layer_decay | 0.8 |
| train.weight_decay | 0.05 |
| train.label_smoothing | 0.05 |
| train.amp | True |
| train.strong_aug | True |
| train.site_randomization | True |
| train.loss | ce |
| train.class_balanced | True |
| train.class_balanced_beta | 0.999 |
| train.weighted_sampler | False |
| train.num_workers | 4 |
| train.bag_loss.enabled | True |
| train.bag_loss.source | ostrava |
| train.bag_loss.k | 8 |
| train.bag_loss.tau | 1 |
| eval.early_stopping | False |
| eval.primary_metric | macro_f1 |
| eval.qwk_classes | [Normal, Stage 1, Stage 2, Stage 3, Stage 4/5] |
| eval.report_per_class_recall | True |
| eval.unvalidated_classes | [Stage 4/5, AP-ROP] |
| _source | configs/rop_staging_dinov2_bag.yaml |

</details>

#### Arm D · seed 42: overview (6 folds: fold_0, fold_1, fold_2, fold_3, fold_4, fold_site)

| fold | train rows | val rows | images/epoch | steps/epoch | minutes | img/s | final acc | final macro-F1 | final QWK (6) | final QWK5 | AUC |
|---|---|---|---|---|---|---|---|---|---|---|---|
| fold_0 | 2462 | 1691 | 2317 | 187 | 35.4 | 25.3 | 0.7156 | 0.5850 | 0.6522 | 0.6910 | 0.8971 |
| fold_1 | 2477 | 1252 | 2339 | 188 | 31.0 | 30.2 | 0.8243 | 0.7381 | 0.8570 | 0.8511 | 0.9193 |
| fold_2 | 2542 | 1252 | 2386 | 192 | 31.5 | 31.8 | 0.7716 | 0.5505 | 0.7400 | 0.8102 | 0.8445 |
| fold_3 | 2459 | 1252 | 2289 | 185 | 31.8 | 28.9 | 0.8650 | 0.7627 | 0.8905 | 0.8974 | 0.9467 |
| fold_4 | 2452 | 1253 | 2269 | 184 | 30.6 | 30.2 | 0.8579 | 0.6981 | 0.8367 | 0.8824 | 0.9045 |
| fold_site | 2388 | 740 | 2190 | 189 | 27.8 | 31.4 | 0.8608 | 0.5468 | 0.9401 | 0.9530 | n/a |

#### Arm D · seed 42 · fold_0 (CV fold 0)

Kernel `rop-v3-dinov2-s-bag-s42-lr1e-4` · run dir `results/rop/v3/runs/dinov2_s_bag_s42/fold_0` · as-run manifest `results/prepared/rop_staging/_v3_fold0.csv`

Fold file `results/prepared/rop_staging/fold_0.csv`: validation = 1691 photos of 297 infants (ostrava 1219, ropvl 310, shenzhen 162); training = 2462 photos of 1231 infants (ostrava 544, ropvl 1363, shenzhen 555).

| rows | normal | stage_1 | stage_2 | stage_3 | stage_4_5 | arop | total |
|---|---|---|---|---|---|---|---|
| training rows | 888 | 403 | 556 | 404 | 70 | 141 | 2462 |
| validation rows | 767 | 137 | 163 | 129 | 19 | 476 | 1691 |

- images per epoch = 2317 under the bag loss (1918 single-image rows + 399 frames drawn from 133 multi-frame Ostrava sessions, at most k = 8 per session; kernel log); training rows = 2462
- steps per epoch = 187 (120 image batches + 67 bag batches; kernel log)
- minutes 35.4 (fold start to next fold start, kernel log) · 25.3 img/s · GPU Tesla T4 · kernel log reports train 2462 val 1691

Per epoch (`metrics.json` history; lr = the first optimizer group, i.e. the lowest layer under layer-wise decay, at the end of the epoch):

| Epoch | train loss | lr | val acc | val macro-F1 | val macro-P | val macro-R |
|---|---|---|---|---|---|---|
| Epoch 1 | 1.7783 | 2.870e-06 | 0.4926 | 0.1924 | 0.1576 | 0.2562 |
| Epoch 2 | 1.5619 | 5.498e-06 | 0.6895 | 0.4953 | 0.6396 | 0.5238 |
| Epoch 3 | 1.2829 | 5.455e-06 | 0.7357 | 0.5869 | 0.6618 | 0.6533 |
| Epoch 4 | 1.1454 | 5.331e-06 | 0.7286 | 0.5485 | 0.7214 | 0.5810 |
| Epoch 5 | 1.1338 | 5.128e-06 | 0.7351 | 0.5464 | 0.6915 | 0.6488 |
| Epoch 6 | 1.0355 | 4.853e-06 | 0.7179 | 0.5298 | 0.6247 | 0.6422 |
| Epoch 7 | 0.9845 | 4.514e-06 | 0.7889 | 0.6285 | 0.6550 | 0.6982 |
| Epoch 8 | 0.9539 | 4.121e-06 | 0.7510 | 0.6158 | 0.6350 | 0.6612 |
| Epoch 9 | 0.9303 | 3.687e-06 | 0.7297 | 0.6000 | 0.6201 | 0.6650 |
| Epoch 10 | 0.9242 | 3.224e-06 | 0.7357 | 0.5970 | 0.6310 | 0.6818 |
| Epoch 11 | 0.8721 | 2.746e-06 | 0.7700 | 0.6361 | 0.6558 | 0.6910 |
| Epoch 12 | 0.8586 | 2.269e-06 | 0.7747 | 0.6520 | 0.6790 | 0.7131 |
| Epoch 13 | 0.8568 | 1.806e-06 | 0.7433 | 0.6171 | 0.6679 | 0.6887 |
| Epoch 14 | 0.8109 | 1.372e-06 | 0.7410 | 0.6142 | 0.6431 | 0.6940 |
| Epoch 15 | 0.8091 | 9.800e-07 | 0.7345 | 0.6041 | 0.6349 | 0.6884 |
| Epoch 16 | 0.7766 | 6.415e-07 | 0.7191 | 0.5949 | 0.6340 | 0.6892 |
| Epoch 17 | 0.7673 | 3.670e-07 | 0.7250 | 0.5952 | 0.6314 | 0.6720 |
| Epoch 18 | 0.7505 | 1.649e-07 | 0.7161 | 0.5871 | 0.6248 | 0.6680 |
| Epoch 19 | 0.7531 | 4.134e-08 | 0.7156 | 0.5859 | 0.6202 | 0.6687 |
| Epoch 20 | 0.7555 | 2.319e-11 | 0.7156 | 0.5850 | 0.6198 | 0.6687 |

Final epoch (Epoch 20), all 1691 validation rows: QWK (6 labels, summary.json) 0.6522 · QWK5 (AP-ROP folded onto stage 4/5, val_predictions.npz) 0.6910 · accuracy 0.7156 · macro-F1 0.5850 · balanced accuracy 0.6687 · AUC 0.8971

| final epoch | Normal | Stage 1 | Stage 2 | Stage 3 | Stage 4/5 | AP-ROP |
|---|---|---|---|---|---|---|
| recall | 0.9909 | 0.4015 | 0.5644 | 0.8372 | 0.8421 | 0.3761 |
| support | 767 | 137 | 163 | 129 | 19 | 476 |

#### Arm D · seed 42 · fold_1 (CV fold 1)

Kernel `rop-v3-dinov2-s-bag-s42-lr1e-4` · run dir `results/rop/v3/runs/dinov2_s_bag_s42/fold_1` · as-run manifest `results/prepared/rop_staging/_v3_fold1.csv`

Fold file `results/prepared/rop_staging/fold_1.csv`: validation = 1252 photos of 299 infants (ostrava 812, ropvl 327, shenzhen 113); training = 2477 photos of 1228 infants (ostrava 527, ropvl 1346, shenzhen 604).

| rows | normal | stage_1 | stage_2 | stage_3 | stage_4_5 | arop | total |
|---|---|---|---|---|---|---|---|
| training rows | 890 | 401 | 534 | 442 | 73 | 137 | 2477 |
| validation rows | 756 | 114 | 220 | 116 | 16 | 30 | 1252 |

- images per epoch = 2339 under the bag loss (1950 single-image rows + 389 frames drawn from 131 multi-frame Ostrava sessions, at most k = 8 per session; kernel log); training rows = 2477
- steps per epoch = 188 (122 image batches + 66 bag batches; kernel log)
- minutes 31.0 (fold start to next fold start, kernel log) · 30.2 img/s · GPU Tesla T4 · kernel log reports train 2477 val 1252

Per epoch (`metrics.json` history; lr = the first optimizer group, i.e. the lowest layer under layer-wise decay, at the end of the epoch):

| Epoch | train loss | lr | val acc | val macro-F1 | val macro-P | val macro-R |
|---|---|---|---|---|---|---|
| Epoch 1 | 1.7864 | 2.870e-06 | 0.6494 | 0.2028 | 0.2643 | 0.2266 |
| Epoch 2 | 1.5892 | 5.498e-06 | 0.7069 | 0.4489 | 0.5860 | 0.5096 |
| Epoch 3 | 1.2885 | 5.455e-06 | 0.7460 | 0.5112 | 0.5147 | 0.6371 |
| Epoch 4 | 1.1668 | 5.331e-06 | 0.7867 | 0.6367 | 0.6923 | 0.6291 |
| Epoch 5 | 1.0772 | 5.128e-06 | 0.8075 | 0.6861 | 0.7217 | 0.6757 |
| Epoch 6 | 1.0270 | 4.853e-06 | 0.7971 | 0.6794 | 0.7020 | 0.6771 |
| Epoch 7 | 0.9826 | 4.514e-06 | 0.7955 | 0.6401 | 0.6581 | 0.6820 |
| Epoch 8 | 0.9901 | 4.121e-06 | 0.7947 | 0.6661 | 0.7744 | 0.6333 |
| Epoch 9 | 0.9415 | 3.687e-06 | 0.7843 | 0.6449 | 0.6827 | 0.6661 |
| Epoch 10 | 0.9118 | 3.224e-06 | 0.8123 | 0.6927 | 0.7075 | 0.7055 |
| Epoch 11 | 0.8686 | 2.746e-06 | 0.8083 | 0.7005 | 0.7123 | 0.7356 |
| Epoch 12 | 0.8496 | 2.269e-06 | 0.7708 | 0.6694 | 0.7099 | 0.7008 |
| Epoch 13 | 0.8011 | 1.806e-06 | 0.8179 | 0.7213 | 0.7448 | 0.7159 |
| Epoch 14 | 0.7948 | 1.372e-06 | 0.8235 | 0.7376 | 0.7688 | 0.7296 |
| Epoch 15 | 0.7805 | 9.800e-07 | 0.8099 | 0.7100 | 0.7303 | 0.7231 |
| Epoch 16 | 0.7622 | 6.415e-07 | 0.8187 | 0.7271 | 0.7319 | 0.7512 |
| Epoch 17 | 0.7619 | 3.670e-07 | 0.8179 | 0.7325 | 0.7457 | 0.7418 |
| Epoch 18 | 0.7535 | 1.649e-07 | 0.8267 | 0.7357 | 0.7506 | 0.7384 |
| Epoch 19 | 0.7488 | 4.134e-08 | 0.8243 | 0.7383 | 0.7506 | 0.7424 |
| Epoch 20 | 0.7362 | 2.317e-11 | 0.8243 | 0.7381 | 0.7502 | 0.7424 |

Final epoch (Epoch 20), all 1252 validation rows: QWK (6 labels, summary.json) 0.8570 · QWK5 (AP-ROP folded onto stage 4/5, val_predictions.npz) 0.8511 · accuracy 0.8243 · macro-F1 0.7381 · balanced accuracy 0.7424 · AUC 0.9193

| final epoch | Normal | Stage 1 | Stage 2 | Stage 3 | Stage 4/5 | AP-ROP |
|---|---|---|---|---|---|---|
| recall | 0.9921 | 0.4737 | 0.5045 | 0.6466 | 0.9375 | 0.9000 |
| support | 756 | 114 | 220 | 116 | 16 | 30 |

#### Arm D · seed 42 · fold_2 (CV fold 2)

Kernel `rop-v3-dinov2-s-bag-s42-lr1e-4` · run dir `results/rop/v3/runs/dinov2_s_bag_s42/fold_2` · as-run manifest `results/prepared/rop_staging/_v3_fold2.csv`

Fold file `results/prepared/rop_staging/fold_2.csv`: validation = 1252 photos of 272 infants (ostrava 830, ropvl 295, shenzhen 127); training = 2542 photos of 1255 infants (ostrava 573, ropvl 1378, shenzhen 591).

| rows | normal | stage_1 | stage_2 | stage_3 | stage_4_5 | arop | total |
|---|---|---|---|---|---|---|---|
| training rows | 888 | 415 | 556 | 481 | 71 | 131 | 2542 |
| validation rows | 772 | 123 | 193 | 110 | 18 | 36 | 1252 |

- images per epoch = 2386 under the bag loss (1969 single-image rows + 417 frames drawn from 135 multi-frame Ostrava sessions, at most k = 8 per session; kernel log); training rows = 2542
- steps per epoch = 192 (124 image batches + 68 bag batches; kernel log)
- minutes 31.5 (fold start to next fold start, kernel log) · 31.8 img/s · GPU Tesla T4 · kernel log reports train 2542 val 1252

Per epoch (`metrics.json` history; lr = the first optimizer group, i.e. the lowest layer under layer-wise decay, at the end of the epoch):

| Epoch | train loss | lr | val acc | val macro-F1 | val macro-P | val macro-R |
|---|---|---|---|---|---|---|
| Epoch 1 | 1.7837 | 2.870e-06 | 0.5319 | 0.1706 | 0.2035 | 0.3184 |
| Epoch 2 | 1.5520 | 5.498e-06 | 0.5847 | 0.3406 | 0.3917 | 0.4639 |
| Epoch 3 | 1.2939 | 5.455e-06 | 0.7460 | 0.4828 | 0.4793 | 0.5835 |
| Epoch 4 | 1.1920 | 5.331e-06 | 0.7540 | 0.5223 | 0.6294 | 0.6311 |
| Epoch 5 | 1.0875 | 5.128e-06 | 0.7572 | 0.4814 | 0.5553 | 0.5799 |
| Epoch 6 | 1.0518 | 4.853e-06 | 0.7636 | 0.5143 | 0.5268 | 0.6389 |
| Epoch 7 | 1.0104 | 4.514e-06 | 0.7500 | 0.5052 | 0.5417 | 0.5897 |
| Epoch 8 | 0.9681 | 4.121e-06 | 0.7604 | 0.5361 | 0.6340 | 0.5798 |
| Epoch 9 | 0.9300 | 3.687e-06 | 0.7596 | 0.5315 | 0.6203 | 0.5933 |
| Epoch 10 | 0.9061 | 3.224e-06 | 0.7764 | 0.5700 | 0.6019 | 0.6363 |
| Epoch 11 | 0.8932 | 2.746e-06 | 0.7644 | 0.5594 | 0.6279 | 0.6175 |
| Epoch 12 | 0.8958 | 2.269e-06 | 0.7620 | 0.5407 | 0.5906 | 0.6292 |
| Epoch 13 | 0.8393 | 1.806e-06 | 0.7676 | 0.5551 | 0.6286 | 0.6359 |
| Epoch 14 | 0.8200 | 1.372e-06 | 0.7668 | 0.5553 | 0.6098 | 0.6285 |
| Epoch 15 | 0.8184 | 9.800e-07 | 0.7724 | 0.5586 | 0.6042 | 0.6328 |
| Epoch 16 | 0.8340 | 6.415e-07 | 0.7708 | 0.5494 | 0.5987 | 0.6351 |
| Epoch 17 | 0.7683 | 3.670e-07 | 0.7716 | 0.5516 | 0.6000 | 0.6403 |
| Epoch 18 | 0.7994 | 1.649e-07 | 0.7740 | 0.5606 | 0.6041 | 0.6388 |
| Epoch 19 | 0.7717 | 4.135e-08 | 0.7716 | 0.5505 | 0.5887 | 0.6351 |
| Epoch 20 | 0.7813 | 2.313e-11 | 0.7716 | 0.5505 | 0.5887 | 0.6351 |

Final epoch (Epoch 20), all 1252 validation rows: QWK (6 labels, summary.json) 0.7400 · QWK5 (AP-ROP folded onto stage 4/5, val_predictions.npz) 0.8102 · accuracy 0.7716 · macro-F1 0.5505 · balanced accuracy 0.6351 · AUC 0.8445

| final epoch | Normal | Stage 1 | Stage 2 | Stage 3 | Stage 4/5 | AP-ROP |
|---|---|---|---|---|---|---|
| recall | 0.9961 | 0.4797 | 0.3212 | 0.2636 | 0.8889 | 0.8611 |
| support | 772 | 123 | 193 | 110 | 18 | 36 |

#### Arm D · seed 42 · fold_3 (CV fold 3)

Kernel `rop-v3-dinov2-s-bag-s42-lr1e-4` · run dir `results/rop/v3/runs/dinov2_s_bag_s42/fold_3` · as-run manifest `results/prepared/rop_staging/_v3_fold3.csv`

Fold file `results/prepared/rop_staging/fold_3.csv`: validation = 1252 photos of 327 infants (ostrava 713, ropvl 362, shenzhen 177); training = 2459 photos of 1200 infants (ostrava 609, ropvl 1313, shenzhen 537).

| rows | normal | stage_1 | stage_2 | stage_3 | stage_4_5 | arop | total |
|---|---|---|---|---|---|---|---|
| training rows | 888 | 409 | 564 | 400 | 71 | 127 | 2459 |
| validation rows | 766 | 111 | 182 | 134 | 18 | 41 | 1252 |

- images per epoch = 2289 under the bag loss (1850 single-image rows + 439 frames drawn from 138 multi-frame Ostrava sessions, at most k = 8 per session; kernel log); training rows = 2459
- steps per epoch = 185 (116 image batches + 69 bag batches; kernel log)
- minutes 31.8 (fold start to next fold start, kernel log) · 28.9 img/s · GPU Tesla T4 · kernel log reports train 2459 val 1252

Per epoch (`metrics.json` history; lr = the first optimizer group, i.e. the lowest layer under layer-wise decay, at the end of the epoch):

| Epoch | train loss | lr | val acc | val macro-F1 | val macro-P | val macro-R |
|---|---|---|---|---|---|---|
| Epoch 1 | 1.8130 | 2.870e-06 | 0.6837 | 0.2442 | 0.3575 | 0.3102 |
| Epoch 2 | 1.5550 | 5.498e-06 | 0.7612 | 0.4656 | 0.5234 | 0.5418 |
| Epoch 3 | 1.3171 | 5.455e-06 | 0.7987 | 0.5902 | 0.6107 | 0.6147 |
| Epoch 4 | 1.1631 | 5.331e-06 | 0.8267 | 0.6220 | 0.6149 | 0.6480 |
| Epoch 5 | 1.1174 | 5.128e-06 | 0.7732 | 0.5289 | 0.5940 | 0.5489 |
| Epoch 6 | 1.0896 | 4.853e-06 | 0.8450 | 0.7183 | 0.7244 | 0.7365 |
| Epoch 7 | 1.0224 | 4.514e-06 | 0.8387 | 0.7069 | 0.7579 | 0.7067 |
| Epoch 8 | 0.9826 | 4.121e-06 | 0.8435 | 0.7200 | 0.7572 | 0.7146 |
| Epoch 9 | 0.9905 | 3.686e-06 | 0.8554 | 0.7739 | 0.7861 | 0.7657 |
| Epoch 10 | 0.9103 | 3.224e-06 | 0.8435 | 0.7312 | 0.7551 | 0.7306 |
| Epoch 11 | 0.9135 | 2.746e-06 | 0.8506 | 0.7298 | 0.7150 | 0.7701 |
| Epoch 12 | 0.8978 | 2.269e-06 | 0.8570 | 0.7435 | 0.7345 | 0.7733 |
| Epoch 13 | 0.8452 | 1.806e-06 | 0.8706 | 0.7768 | 0.7757 | 0.7830 |
| Epoch 14 | 0.8496 | 1.372e-06 | 0.8674 | 0.7738 | 0.7941 | 0.7617 |
| Epoch 15 | 0.7986 | 9.799e-07 | 0.8674 | 0.7771 | 0.7969 | 0.7638 |
| Epoch 16 | 0.8285 | 6.414e-07 | 0.8642 | 0.7620 | 0.7762 | 0.7518 |
| Epoch 17 | 0.7763 | 3.670e-07 | 0.8666 | 0.7689 | 0.7855 | 0.7683 |
| Epoch 18 | 0.7573 | 1.649e-07 | 0.8634 | 0.7606 | 0.7767 | 0.7491 |
| Epoch 19 | 0.8182 | 4.133e-08 | 0.8650 | 0.7627 | 0.7790 | 0.7512 |
| Epoch 20 | 0.7894 | 2.321e-11 | 0.8650 | 0.7627 | 0.7790 | 0.7512 |

Final epoch (Epoch 20), all 1252 validation rows: QWK (6 labels, summary.json) 0.8905 · QWK5 (AP-ROP folded onto stage 4/5, val_predictions.npz) 0.8974 · accuracy 0.8650 · macro-F1 0.7627 · balanced accuracy 0.7512 · AUC 0.9467

| final epoch | Normal | Stage 1 | Stage 2 | Stage 3 | Stage 4/5 | AP-ROP |
|---|---|---|---|---|---|---|
| recall | 0.9856 | 0.4324 | 0.6484 | 0.8582 | 0.7778 | 0.8049 |
| support | 766 | 111 | 182 | 134 | 18 | 41 |

#### Arm D · seed 42 · fold_4 (CV fold 4)

Kernel `rop-v3-dinov2-s-bag-s42-lr1e-4` · run dir `results/rop/v3/runs/dinov2_s_bag_s42/fold_4` · as-run manifest `results/prepared/rop_staging/_v3_fold4.csv`

Fold file `results/prepared/rop_staging/fold_4.csv`: validation = 1253 photos of 333 infants (ostrava 711, ropvl 381, shenzhen 161); training = 2452 photos of 1194 infants (ostrava 607, ropvl 1292, shenzhen 553).

| rows | normal | stage_1 | stage_2 | stage_3 | stage_4_5 | arop | total |
|---|---|---|---|---|---|---|---|
| training rows | 882 | 420 | 546 | 401 | 71 | 132 | 2452 |
| validation rows | 776 | 92 | 201 | 131 | 18 | 35 | 1253 |

- images per epoch = 2269 under the bag loss (1845 single-image rows + 424 frames drawn from 135 multi-frame Ostrava sessions, at most k = 8 per session; kernel log); training rows = 2452
- steps per epoch = 184 (116 image batches + 68 bag batches; kernel log)
- minutes 30.6 (fold start to next fold start, kernel log) · 30.2 img/s · GPU Tesla T4 · kernel log reports train 2452 val 1253

Per epoch (`metrics.json` history; lr = the first optimizer group, i.e. the lowest layer under layer-wise decay, at the end of the epoch):

| Epoch | train loss | lr | val acc | val macro-F1 | val macro-P | val macro-R |
|---|---|---|---|---|---|---|
| Epoch 1 | 1.8034 | 2.870e-06 | 0.6512 | 0.2466 | 0.2921 | 0.3036 |
| Epoch 2 | 1.5955 | 5.498e-06 | 0.7334 | 0.3877 | 0.4181 | 0.4631 |
| Epoch 3 | 1.2990 | 5.455e-06 | 0.6895 | 0.4508 | 0.4461 | 0.5927 |
| Epoch 4 | 1.1639 | 5.331e-06 | 0.8268 | 0.6286 | 0.6275 | 0.6836 |
| Epoch 5 | 1.1095 | 5.128e-06 | 0.8109 | 0.5721 | 0.6133 | 0.6713 |
| Epoch 6 | 1.0429 | 4.853e-06 | 0.8101 | 0.6099 | 0.6204 | 0.7148 |
| Epoch 7 | 1.0343 | 4.514e-06 | 0.7997 | 0.5625 | 0.5774 | 0.6237 |
| Epoch 8 | 0.9883 | 4.121e-06 | 0.8228 | 0.6153 | 0.6118 | 0.6716 |
| Epoch 9 | 0.9575 | 3.686e-06 | 0.8236 | 0.6176 | 0.6023 | 0.7072 |
| Epoch 10 | 0.9165 | 3.224e-06 | 0.8276 | 0.6405 | 0.6535 | 0.7028 |
| Epoch 11 | 0.8728 | 2.746e-06 | 0.8236 | 0.6322 | 0.6342 | 0.7102 |
| Epoch 12 | 0.8751 | 2.269e-06 | 0.8428 | 0.6799 | 0.6605 | 0.7551 |
| Epoch 13 | 0.7987 | 1.806e-06 | 0.8436 | 0.6583 | 0.6468 | 0.7359 |
| Epoch 14 | 0.8185 | 1.372e-06 | 0.8476 | 0.6788 | 0.6715 | 0.7444 |
| Epoch 15 | 0.7935 | 9.799e-07 | 0.8547 | 0.6964 | 0.6959 | 0.7503 |
| Epoch 16 | 0.7868 | 6.414e-07 | 0.8436 | 0.6743 | 0.6671 | 0.7434 |
| Epoch 17 | 0.7857 | 3.670e-07 | 0.8484 | 0.6778 | 0.6704 | 0.7437 |
| Epoch 18 | 0.7651 | 1.649e-07 | 0.8579 | 0.6967 | 0.6868 | 0.7558 |
| Epoch 19 | 0.7717 | 4.133e-08 | 0.8563 | 0.6954 | 0.6833 | 0.7656 |
| Epoch 20 | 0.7637 | 2.323e-11 | 0.8579 | 0.6981 | 0.6874 | 0.7673 |

Final epoch (Epoch 20), all 1253 validation rows: QWK (6 labels, summary.json) 0.8367 · QWK5 (AP-ROP folded onto stage 4/5, val_predictions.npz) 0.8824 · accuracy 0.8579 · macro-F1 0.6981 · balanced accuracy 0.7673 · AUC 0.9045

| final epoch | Normal | Stage 1 | Stage 2 | Stage 3 | Stage 4/5 | AP-ROP |
|---|---|---|---|---|---|---|
| recall | 0.9936 | 0.6196 | 0.4527 | 0.8473 | 0.8333 | 0.8571 |
| support | 776 | 92 | 201 | 131 | 18 | 35 |

#### Arm D · seed 42 · fold_site (held-out source Shenzhen (the v3 site fold))

Kernel `rop-v3-dinov2-s-bag-s42-lr1e-4` · run dir `results/rop/v3/runs/dinov2_s_bag_s42/fold_site` · as-run manifest `results/prepared/rop_staging/_v3_foldsite.csv`

Fold file `results/prepared/rop_staging/fold_site.csv`: validation = 740 photos of 397 infants (shenzhen 740); training = 2388 photos of 1131 infants (ostrava 715, ropvl 1673).

| rows | normal | stage_1 | stage_2 | stage_3 | stage_4_5 | arop | total |
|---|---|---|---|---|---|---|---|
| training rows | 893 | 426 | 533 | 280 | 89 | 167 | 2388 |
| validation rows | 234 | 89 | 163 | 254 | 0 | 0 | 740 |

- images per epoch = 2190 under the bag loss (1673 single-image rows + 517 frames drawn from 168 multi-frame Ostrava sessions, at most k = 8 per session; kernel log); training rows = 2388
- steps per epoch = 189 (105 image batches + 84 bag batches; kernel log)
- minutes 27.8 (fold start to next fold start, kernel log) · 31.4 img/s · GPU Tesla T4 · kernel log reports train 2388 val 740

Per epoch (`metrics.json` history; lr = the first optimizer group, i.e. the lowest layer under layer-wise decay, at the end of the epoch):

| Epoch | train loss | lr | val acc | val macro-F1 | val macro-P | val macro-R |
|---|---|---|---|---|---|---|
| Epoch 1 | 1.7338 | 2.870e-06 | 0.4162 | 0.2985 | 0.3112 | 0.4031 |
| Epoch 2 | 1.6080 | 5.498e-06 | 0.4973 | 0.2964 | 0.2643 | 0.3819 |
| Epoch 3 | 1.4088 | 5.455e-06 | 0.6554 | 0.3575 | 0.4553 | 0.3910 |
| Epoch 4 | 1.1879 | 5.331e-06 | 0.4838 | 0.2761 | 0.4215 | 0.2607 |
| Epoch 5 | 1.1009 | 5.128e-06 | 0.5257 | 0.2697 | 0.4405 | 0.2742 |
| Epoch 6 | 1.0496 | 4.853e-06 | 0.7486 | 0.4428 | 0.5776 | 0.4472 |
| Epoch 7 | 0.9819 | 4.514e-06 | 0.7838 | 0.5068 | 0.5220 | 0.5062 |
| Epoch 8 | 0.9637 | 4.121e-06 | 0.6946 | 0.4127 | 0.5373 | 0.4390 |
| Epoch 9 | 0.9343 | 3.687e-06 | 0.7919 | 0.4819 | 0.5643 | 0.4827 |
| Epoch 10 | 0.8833 | 3.224e-06 | 0.7554 | 0.4562 | 0.5545 | 0.4402 |
| Epoch 11 | 0.8626 | 2.746e-06 | 0.8365 | 0.5391 | 0.5666 | 0.5242 |
| Epoch 12 | 0.8644 | 2.269e-06 | 0.8135 | 0.4927 | 0.5543 | 0.4845 |
| Epoch 13 | 0.8263 | 1.806e-06 | 0.8054 | 0.5105 | 0.5389 | 0.4939 |
| Epoch 14 | 0.8330 | 1.372e-06 | 0.8608 | 0.5345 | 0.5745 | 0.5273 |
| Epoch 15 | 0.7811 | 9.800e-07 | 0.8649 | 0.5515 | 0.5737 | 0.5385 |
| Epoch 16 | 0.7391 | 6.415e-07 | 0.8608 | 0.5439 | 0.5772 | 0.5307 |
| Epoch 17 | 0.7784 | 3.670e-07 | 0.8595 | 0.5476 | 0.5743 | 0.5351 |
| Epoch 18 | 0.7657 | 1.649e-07 | 0.8581 | 0.5434 | 0.5725 | 0.5312 |
| Epoch 19 | 0.7576 | 4.134e-08 | 0.8622 | 0.5477 | 0.5747 | 0.5363 |
| Epoch 20 | 0.7427 | 2.316e-11 | 0.8608 | 0.5468 | 0.5738 | 0.5353 |

Final epoch (Epoch 20), all 740 validation rows: QWK (6 labels, summary.json) 0.9401 · QWK5 (AP-ROP folded onto stage 4/5, val_predictions.npz) 0.9530 · accuracy 0.8608 · macro-F1 0.5468 · balanced accuracy 0.5353 · AUC n/a

| final epoch | Normal | Stage 1 | Stage 2 | Stage 3 | Stage 4/5 | AP-ROP |
|---|---|---|---|---|---|---|
| recall | 1.0000 | 0.5056 | 0.8282 | 0.8780 | 0.0000 | 0.0000 |
| support | 234 | 89 | 163 | 254 | 0 | 0 |

#### Arm D · seed 1337: overview (6 folds: fold_0, fold_1, fold_2, fold_3, fold_4, fold_site)

| fold | train rows | val rows | images/epoch | steps/epoch | minutes | img/s | final acc | final macro-F1 | final QWK (6) | final QWK5 | AUC |
|---|---|---|---|---|---|---|---|---|---|---|---|
| fold_0 | 2462 | 1691 | 2317 | 187 | 33.2 | 27.8 | 0.7179 | 0.6040 | 0.6126 | 0.6506 | 0.8785 |
| fold_1 | 2477 | 1252 | 2339 | 188 | 32.0 | 31.9 | 0.8291 | 0.7499 | 0.8659 | 0.8587 | 0.9157 |
| fold_2 | 2542 | 1252 | 2386 | 192 | 33.5 | 27.6 | 0.7740 | 0.5700 | 0.7425 | 0.8162 | 0.8404 |
| fold_3 | 2459 | 1252 | 2289 | 185 | 33.3 | 27.9 | 0.8746 | 0.7914 | 0.9157 | 0.9197 | 0.9387 |
| fold_4 | 2452 | 1253 | 2269 | 184 | 32.9 | 28.4 | 0.8532 | 0.7104 | 0.8397 | 0.8834 | 0.9323 |
| fold_site | 2388 | 740 | 2190 | 189 | 28.0 | 29.5 | 0.8865 | 0.5657 | 0.9508 | 0.9594 | n/a |

#### Arm D · seed 1337 · fold_0 (CV fold 0)

Kernel `rop-v3-dinov2-s-bag-s1337-lr1e-4` · run dir `results/rop/v3/runs/dinov2_s_bag_s1337/fold_0` · as-run manifest `results/prepared/rop_staging/_v3_fold0.csv`

Fold file `results/prepared/rop_staging/fold_0.csv`: validation = 1691 photos of 297 infants (ostrava 1219, ropvl 310, shenzhen 162); training = 2462 photos of 1231 infants (ostrava 544, ropvl 1363, shenzhen 555).

| rows | normal | stage_1 | stage_2 | stage_3 | stage_4_5 | arop | total |
|---|---|---|---|---|---|---|---|
| training rows | 888 | 403 | 556 | 404 | 70 | 141 | 2462 |
| validation rows | 767 | 137 | 163 | 129 | 19 | 476 | 1691 |

- images per epoch = 2317 under the bag loss (1918 single-image rows + 399 frames drawn from 133 multi-frame Ostrava sessions, at most k = 8 per session; kernel log); training rows = 2462
- steps per epoch = 187 (120 image batches + 67 bag batches; kernel log)
- minutes 33.2 (fold start to next fold start, kernel log) · 27.8 img/s · GPU Tesla T4 · kernel log reports train 2462 val 1691

Per epoch (`metrics.json` history; lr = the first optimizer group, i.e. the lowest layer under layer-wise decay, at the end of the epoch):

| Epoch | train loss | lr | val acc | val macro-F1 | val macro-P | val macro-R |
|---|---|---|---|---|---|---|
| Epoch 1 | 1.7766 | 2.870e-06 | 0.4601 | 0.1743 | 0.4142 | 0.2177 |
| Epoch 2 | 1.5161 | 5.498e-06 | 0.6682 | 0.4366 | 0.4966 | 0.5341 |
| Epoch 3 | 1.3083 | 5.455e-06 | 0.7292 | 0.5544 | 0.6823 | 0.6212 |
| Epoch 4 | 1.1415 | 5.331e-06 | 0.7309 | 0.5995 | 0.6250 | 0.6452 |
| Epoch 5 | 1.1073 | 5.128e-06 | 0.7528 | 0.6174 | 0.6660 | 0.6516 |
| Epoch 6 | 1.0634 | 4.853e-06 | 0.7469 | 0.6185 | 0.6591 | 0.6542 |
| Epoch 7 | 1.0085 | 4.514e-06 | 0.7368 | 0.6064 | 0.6396 | 0.6594 |
| Epoch 8 | 0.9981 | 4.121e-06 | 0.6978 | 0.5900 | 0.6506 | 0.6096 |
| Epoch 9 | 0.9582 | 3.687e-06 | 0.6736 | 0.5708 | 0.6308 | 0.6050 |
| Epoch 10 | 0.8998 | 3.224e-06 | 0.7363 | 0.6339 | 0.6665 | 0.6559 |
| Epoch 11 | 0.8739 | 2.746e-06 | 0.6931 | 0.6014 | 0.6560 | 0.6079 |
| Epoch 12 | 0.8485 | 2.269e-06 | 0.7067 | 0.5884 | 0.6250 | 0.6371 |
| Epoch 13 | 0.8229 | 1.806e-06 | 0.7156 | 0.6154 | 0.6550 | 0.6513 |
| Epoch 14 | 0.8110 | 1.372e-06 | 0.7191 | 0.6157 | 0.6430 | 0.6438 |
| Epoch 15 | 0.8002 | 9.800e-07 | 0.7244 | 0.6113 | 0.6368 | 0.6598 |
| Epoch 16 | 0.7851 | 6.415e-07 | 0.7002 | 0.5953 | 0.6331 | 0.6410 |
| Epoch 17 | 0.7819 | 3.670e-07 | 0.7085 | 0.6004 | 0.6382 | 0.6500 |
| Epoch 18 | 0.7810 | 1.649e-07 | 0.7126 | 0.6056 | 0.6372 | 0.6606 |
| Epoch 19 | 0.7459 | 4.134e-08 | 0.7167 | 0.6035 | 0.6346 | 0.6568 |
| Epoch 20 | 0.7713 | 2.319e-11 | 0.7179 | 0.6040 | 0.6352 | 0.6573 |

Final epoch (Epoch 20), all 1691 validation rows: QWK (6 labels, summary.json) 0.6126 · QWK5 (AP-ROP folded onto stage 4/5, val_predictions.npz) 0.6506 · accuracy 0.7179 · macro-F1 0.6040 · balanced accuracy 0.6573 · AUC 0.8785

| final epoch | Normal | Stage 1 | Stage 2 | Stage 3 | Stage 4/5 | AP-ROP |
|---|---|---|---|---|---|---|
| recall | 0.9974 | 0.3942 | 0.5215 | 0.8527 | 0.7895 | 0.3887 |
| support | 767 | 137 | 163 | 129 | 19 | 476 |

#### Arm D · seed 1337 · fold_1 (CV fold 1)

Kernel `rop-v3-dinov2-s-bag-s1337-lr1e-4` · run dir `results/rop/v3/runs/dinov2_s_bag_s1337/fold_1` · as-run manifest `results/prepared/rop_staging/_v3_fold1.csv`

Fold file `results/prepared/rop_staging/fold_1.csv`: validation = 1252 photos of 299 infants (ostrava 812, ropvl 327, shenzhen 113); training = 2477 photos of 1228 infants (ostrava 527, ropvl 1346, shenzhen 604).

| rows | normal | stage_1 | stage_2 | stage_3 | stage_4_5 | arop | total |
|---|---|---|---|---|---|---|---|
| training rows | 890 | 401 | 534 | 442 | 73 | 137 | 2477 |
| validation rows | 756 | 114 | 220 | 116 | 16 | 30 | 1252 |

- images per epoch = 2339 under the bag loss (1950 single-image rows + 389 frames drawn from 131 multi-frame Ostrava sessions, at most k = 8 per session; kernel log); training rows = 2477
- steps per epoch = 188 (122 image batches + 66 bag batches; kernel log)
- minutes 32.0 (fold start to next fold start, kernel log) · 31.9 img/s · GPU Tesla T4 · kernel log reports train 2477 val 1252

Per epoch (`metrics.json` history; lr = the first optimizer group, i.e. the lowest layer under layer-wise decay, at the end of the epoch):

| Epoch | train loss | lr | val acc | val macro-F1 | val macro-P | val macro-R |
|---|---|---|---|---|---|---|
| Epoch 1 | 1.7788 | 2.870e-06 | 0.6310 | 0.2291 | 0.1995 | 0.3411 |
| Epoch 2 | 1.5288 | 5.498e-06 | 0.7396 | 0.4315 | 0.4632 | 0.5478 |
| Epoch 3 | 1.2923 | 5.455e-06 | 0.7452 | 0.5479 | 0.5682 | 0.6091 |
| Epoch 4 | 1.1224 | 5.331e-06 | 0.7636 | 0.6365 | 0.6669 | 0.6447 |
| Epoch 5 | 1.0700 | 5.128e-06 | 0.7772 | 0.6068 | 0.6843 | 0.6297 |
| Epoch 6 | 1.0509 | 4.853e-06 | 0.7947 | 0.6529 | 0.6892 | 0.6691 |
| Epoch 7 | 0.9915 | 4.514e-06 | 0.7875 | 0.6735 | 0.7389 | 0.6396 |
| Epoch 8 | 0.9431 | 4.121e-06 | 0.7995 | 0.7046 | 0.7463 | 0.6894 |
| Epoch 9 | 0.9170 | 3.687e-06 | 0.8035 | 0.6759 | 0.7127 | 0.6733 |
| Epoch 10 | 0.8967 | 3.224e-06 | 0.7995 | 0.6859 | 0.7094 | 0.6901 |
| Epoch 11 | 0.8804 | 2.746e-06 | 0.7987 | 0.6634 | 0.6560 | 0.7202 |
| Epoch 12 | 0.8347 | 2.269e-06 | 0.8139 | 0.7103 | 0.7380 | 0.7038 |
| Epoch 13 | 0.8074 | 1.806e-06 | 0.8179 | 0.7360 | 0.7720 | 0.7170 |
| Epoch 14 | 0.8123 | 1.372e-06 | 0.8235 | 0.7523 | 0.7748 | 0.7459 |
| Epoch 15 | 0.7830 | 9.800e-07 | 0.8227 | 0.7372 | 0.7485 | 0.7428 |
| Epoch 16 | 0.7599 | 6.415e-07 | 0.8275 | 0.7473 | 0.7774 | 0.7425 |
| Epoch 17 | 0.7519 | 3.670e-07 | 0.8283 | 0.7288 | 0.7405 | 0.7477 |
| Epoch 18 | 0.7701 | 1.649e-07 | 0.8323 | 0.7505 | 0.7792 | 0.7342 |
| Epoch 19 | 0.7485 | 4.134e-08 | 0.8275 | 0.7473 | 0.7778 | 0.7304 |
| Epoch 20 | 0.7343 | 2.317e-11 | 0.8291 | 0.7499 | 0.7809 | 0.7327 |

Final epoch (Epoch 20), all 1252 validation rows: QWK (6 labels, summary.json) 0.8659 · QWK5 (AP-ROP folded onto stage 4/5, val_predictions.npz) 0.8587 · accuracy 0.8291 · macro-F1 0.7499 · balanced accuracy 0.7327 · AUC 0.9157

| final epoch | Normal | Stage 1 | Stage 2 | Stage 3 | Stage 4/5 | AP-ROP |
|---|---|---|---|---|---|---|
| recall | 0.9934 | 0.5000 | 0.5045 | 0.6897 | 0.8750 | 0.8333 |
| support | 756 | 114 | 220 | 116 | 16 | 30 |

#### Arm D · seed 1337 · fold_2 (CV fold 2)

Kernel `rop-v3-dinov2-s-bag-s1337-lr1e-4` · run dir `results/rop/v3/runs/dinov2_s_bag_s1337/fold_2` · as-run manifest `results/prepared/rop_staging/_v3_fold2.csv`

Fold file `results/prepared/rop_staging/fold_2.csv`: validation = 1252 photos of 272 infants (ostrava 830, ropvl 295, shenzhen 127); training = 2542 photos of 1255 infants (ostrava 573, ropvl 1378, shenzhen 591).

| rows | normal | stage_1 | stage_2 | stage_3 | stage_4_5 | arop | total |
|---|---|---|---|---|---|---|---|
| training rows | 888 | 415 | 556 | 481 | 71 | 131 | 2542 |
| validation rows | 772 | 123 | 193 | 110 | 18 | 36 | 1252 |

- images per epoch = 2386 under the bag loss (1969 single-image rows + 417 frames drawn from 135 multi-frame Ostrava sessions, at most k = 8 per session; kernel log); training rows = 2542
- steps per epoch = 192 (124 image batches + 68 bag batches; kernel log)
- minutes 33.5 (fold start to next fold start, kernel log) · 27.6 img/s · GPU Tesla T4 · kernel log reports train 2542 val 1252

Per epoch (`metrics.json` history; lr = the first optimizer group, i.e. the lowest layer under layer-wise decay, at the end of the epoch):

| Epoch | train loss | lr | val acc | val macro-F1 | val macro-P | val macro-R |
|---|---|---|---|---|---|---|
| Epoch 1 | 1.7604 | 2.870e-06 | 0.6198 | 0.2016 | 0.1771 | 0.2362 |
| Epoch 2 | 1.5825 | 5.498e-06 | 0.6861 | 0.3018 | 0.4350 | 0.3832 |
| Epoch 3 | 1.3025 | 5.455e-06 | 0.7452 | 0.4647 | 0.5608 | 0.5585 |
| Epoch 4 | 1.1600 | 5.331e-06 | 0.7668 | 0.5398 | 0.6435 | 0.5880 |
| Epoch 5 | 1.0855 | 5.128e-06 | 0.7540 | 0.4942 | 0.6202 | 0.5349 |
| Epoch 6 | 1.0453 | 4.853e-06 | 0.7372 | 0.4893 | 0.5197 | 0.5750 |
| Epoch 7 | 0.9786 | 4.514e-06 | 0.7740 | 0.5438 | 0.5715 | 0.6014 |
| Epoch 8 | 0.9493 | 4.121e-06 | 0.7812 | 0.5907 | 0.6560 | 0.6540 |
| Epoch 9 | 0.9433 | 3.687e-06 | 0.7564 | 0.5388 | 0.6806 | 0.6094 |
| Epoch 10 | 0.9176 | 3.224e-06 | 0.7692 | 0.5478 | 0.6129 | 0.6309 |
| Epoch 11 | 0.8881 | 2.746e-06 | 0.7612 | 0.5517 | 0.5843 | 0.5775 |
| Epoch 12 | 0.8355 | 2.269e-06 | 0.7596 | 0.5477 | 0.5937 | 0.6253 |
| Epoch 13 | 0.8356 | 1.806e-06 | 0.7692 | 0.5463 | 0.6530 | 0.6038 |
| Epoch 14 | 0.8203 | 1.372e-06 | 0.7756 | 0.5615 | 0.6057 | 0.6033 |
| Epoch 15 | 0.8128 | 9.800e-07 | 0.7740 | 0.5594 | 0.6176 | 0.6152 |
| Epoch 16 | 0.7852 | 6.415e-07 | 0.7772 | 0.5727 | 0.6239 | 0.6391 |
| Epoch 17 | 0.7749 | 3.670e-07 | 0.7756 | 0.5663 | 0.6087 | 0.6229 |
| Epoch 18 | 0.7782 | 1.649e-07 | 0.7740 | 0.5645 | 0.6157 | 0.6195 |
| Epoch 19 | 0.7816 | 4.135e-08 | 0.7740 | 0.5706 | 0.6193 | 0.6286 |
| Epoch 20 | 0.7632 | 2.313e-11 | 0.7740 | 0.5700 | 0.6171 | 0.6286 |

Final epoch (Epoch 20), all 1252 validation rows: QWK (6 labels, summary.json) 0.7425 · QWK5 (AP-ROP folded onto stage 4/5, val_predictions.npz) 0.8162 · accuracy 0.7740 · macro-F1 0.5700 · balanced accuracy 0.6286 · AUC 0.8404

| final epoch | Normal | Stage 1 | Stage 2 | Stage 3 | Stage 4/5 | AP-ROP |
|---|---|---|---|---|---|---|
| recall | 0.9974 | 0.4959 | 0.3472 | 0.2364 | 0.8889 | 0.8056 |
| support | 772 | 123 | 193 | 110 | 18 | 36 |

#### Arm D · seed 1337 · fold_3 (CV fold 3)

Kernel `rop-v3-dinov2-s-bag-s1337-lr1e-4` · run dir `results/rop/v3/runs/dinov2_s_bag_s1337/fold_3` · as-run manifest `results/prepared/rop_staging/_v3_fold3.csv`

Fold file `results/prepared/rop_staging/fold_3.csv`: validation = 1252 photos of 327 infants (ostrava 713, ropvl 362, shenzhen 177); training = 2459 photos of 1200 infants (ostrava 609, ropvl 1313, shenzhen 537).

| rows | normal | stage_1 | stage_2 | stage_3 | stage_4_5 | arop | total |
|---|---|---|---|---|---|---|---|
| training rows | 888 | 409 | 564 | 400 | 71 | 127 | 2459 |
| validation rows | 766 | 111 | 182 | 134 | 18 | 41 | 1252 |

- images per epoch = 2289 under the bag loss (1850 single-image rows + 439 frames drawn from 138 multi-frame Ostrava sessions, at most k = 8 per session; kernel log); training rows = 2459
- steps per epoch = 185 (116 image batches + 69 bag batches; kernel log)
- minutes 33.3 (fold start to next fold start, kernel log) · 27.9 img/s · GPU Tesla T4 · kernel log reports train 2459 val 1252

Per epoch (`metrics.json` history; lr = the first optimizer group, i.e. the lowest layer under layer-wise decay, at the end of the epoch):

| Epoch | train loss | lr | val acc | val macro-F1 | val macro-P | val macro-R |
|---|---|---|---|---|---|---|
| Epoch 1 | 1.7889 | 2.870e-06 | 0.6454 | 0.2547 | 0.2339 | 0.2973 |
| Epoch 2 | 1.6093 | 5.498e-06 | 0.6789 | 0.3643 | 0.3712 | 0.4189 |
| Epoch 3 | 1.3022 | 5.455e-06 | 0.7835 | 0.5166 | 0.6454 | 0.6081 |
| Epoch 4 | 1.2103 | 5.331e-06 | 0.7875 | 0.5918 | 0.7379 | 0.5833 |
| Epoch 5 | 1.1103 | 5.128e-06 | 0.8474 | 0.7582 | 0.7849 | 0.7487 |
| Epoch 6 | 1.0628 | 4.853e-06 | 0.8458 | 0.7372 | 0.7693 | 0.7136 |
| Epoch 7 | 1.0408 | 4.514e-06 | 0.8387 | 0.7081 | 0.7090 | 0.7564 |
| Epoch 8 | 0.9972 | 4.121e-06 | 0.8506 | 0.7306 | 0.7708 | 0.7194 |
| Epoch 9 | 0.9270 | 3.686e-06 | 0.8618 | 0.7585 | 0.7859 | 0.7447 |
| Epoch 10 | 0.9043 | 3.224e-06 | 0.8562 | 0.7716 | 0.7930 | 0.7544 |
| Epoch 11 | 0.8750 | 2.746e-06 | 0.8411 | 0.6958 | 0.7563 | 0.7017 |
| Epoch 12 | 0.8563 | 2.269e-06 | 0.8538 | 0.7425 | 0.7434 | 0.7498 |
| Epoch 13 | 0.8692 | 1.806e-06 | 0.8722 | 0.7852 | 0.8259 | 0.7558 |
| Epoch 14 | 0.8419 | 1.372e-06 | 0.8722 | 0.7823 | 0.8080 | 0.7601 |
| Epoch 15 | 0.8148 | 9.799e-07 | 0.8754 | 0.7837 | 0.7850 | 0.7904 |
| Epoch 16 | 0.7986 | 6.414e-07 | 0.8754 | 0.7909 | 0.7957 | 0.7918 |
| Epoch 17 | 0.7686 | 3.670e-07 | 0.8706 | 0.7824 | 0.7897 | 0.7841 |
| Epoch 18 | 0.7955 | 1.649e-07 | 0.8730 | 0.7873 | 0.8022 | 0.7790 |
| Epoch 19 | 0.8036 | 4.133e-08 | 0.8754 | 0.7915 | 0.7958 | 0.7935 |
| Epoch 20 | 0.7919 | 2.321e-11 | 0.8746 | 0.7914 | 0.7980 | 0.7912 |

Final epoch (Epoch 20), all 1252 validation rows: QWK (6 labels, summary.json) 0.9157 · QWK5 (AP-ROP folded onto stage 4/5, val_predictions.npz) 0.9197 · accuracy 0.8746 · macro-F1 0.7914 · balanced accuracy 0.7912 · AUC 0.9387

| final epoch | Normal | Stage 1 | Stage 2 | Stage 3 | Stage 4/5 | AP-ROP |
|---|---|---|---|---|---|---|
| recall | 0.9922 | 0.4505 | 0.6484 | 0.8582 | 0.9444 | 0.8537 |
| support | 766 | 111 | 182 | 134 | 18 | 41 |

#### Arm D · seed 1337 · fold_4 (CV fold 4)

Kernel `rop-v3-dinov2-s-bag-s1337-lr1e-4` · run dir `results/rop/v3/runs/dinov2_s_bag_s1337/fold_4` · as-run manifest `results/prepared/rop_staging/_v3_fold4.csv`

Fold file `results/prepared/rop_staging/fold_4.csv`: validation = 1253 photos of 333 infants (ostrava 711, ropvl 381, shenzhen 161); training = 2452 photos of 1194 infants (ostrava 607, ropvl 1292, shenzhen 553).

| rows | normal | stage_1 | stage_2 | stage_3 | stage_4_5 | arop | total |
|---|---|---|---|---|---|---|---|
| training rows | 882 | 420 | 546 | 401 | 71 | 132 | 2452 |
| validation rows | 776 | 92 | 201 | 131 | 18 | 35 | 1253 |

- images per epoch = 2269 under the bag loss (1845 single-image rows + 424 frames drawn from 135 multi-frame Ostrava sessions, at most k = 8 per session; kernel log); training rows = 2452
- steps per epoch = 184 (116 image batches + 68 bag batches; kernel log)
- minutes 32.9 (fold start to next fold start, kernel log) · 28.4 img/s · GPU Tesla T4 · kernel log reports train 2452 val 1253

Per epoch (`metrics.json` history; lr = the first optimizer group, i.e. the lowest layer under layer-wise decay, at the end of the epoch):

| Epoch | train loss | lr | val acc | val macro-F1 | val macro-P | val macro-R |
|---|---|---|---|---|---|---|
| Epoch 1 | 1.7772 | 2.870e-06 | 0.6329 | 0.1912 | 0.2729 | 0.3085 |
| Epoch 2 | 1.5588 | 5.498e-06 | 0.7478 | 0.4465 | 0.5562 | 0.5393 |
| Epoch 3 | 1.2951 | 5.455e-06 | 0.7821 | 0.5046 | 0.5059 | 0.5861 |
| Epoch 4 | 1.1807 | 5.331e-06 | 0.8156 | 0.6080 | 0.5857 | 0.7138 |
| Epoch 5 | 1.0893 | 5.128e-06 | 0.8276 | 0.6385 | 0.6380 | 0.7036 |
| Epoch 6 | 1.0285 | 4.853e-06 | 0.7941 | 0.5964 | 0.6519 | 0.6331 |
| Epoch 7 | 1.0007 | 4.514e-06 | 0.8268 | 0.6655 | 0.6860 | 0.7184 |
| Epoch 8 | 0.9660 | 4.121e-06 | 0.8292 | 0.6639 | 0.6754 | 0.7429 |
| Epoch 9 | 0.9342 | 3.686e-06 | 0.7997 | 0.5957 | 0.6039 | 0.7224 |
| Epoch 10 | 0.9094 | 3.224e-06 | 0.8188 | 0.6427 | 0.6244 | 0.7446 |
| Epoch 11 | 0.8833 | 2.746e-06 | 0.8212 | 0.6387 | 0.6649 | 0.6872 |
| Epoch 12 | 0.8396 | 2.269e-06 | 0.8220 | 0.6503 | 0.6505 | 0.7261 |
| Epoch 13 | 0.8380 | 1.806e-06 | 0.8420 | 0.6871 | 0.6908 | 0.7418 |
| Epoch 14 | 0.8239 | 1.372e-06 | 0.8436 | 0.6881 | 0.6834 | 0.7518 |
| Epoch 15 | 0.8012 | 9.799e-07 | 0.8476 | 0.6977 | 0.6899 | 0.7650 |
| Epoch 16 | 0.7711 | 6.414e-07 | 0.8492 | 0.7043 | 0.6969 | 0.7766 |
| Epoch 17 | 0.7612 | 3.670e-07 | 0.8524 | 0.7102 | 0.7027 | 0.7775 |
| Epoch 18 | 0.7682 | 1.649e-07 | 0.8524 | 0.7071 | 0.6993 | 0.7661 |
| Epoch 19 | 0.7477 | 4.133e-08 | 0.8532 | 0.7104 | 0.7037 | 0.7764 |
| Epoch 20 | 0.7502 | 2.323e-11 | 0.8532 | 0.7104 | 0.7037 | 0.7764 |

Final epoch (Epoch 20), all 1253 validation rows: QWK (6 labels, summary.json) 0.8397 · QWK5 (AP-ROP folded onto stage 4/5, val_predictions.npz) 0.8834 · accuracy 0.8532 · macro-F1 0.7104 · balanced accuracy 0.7764 · AUC 0.9323

| final epoch | Normal | Stage 1 | Stage 2 | Stage 3 | Stage 4/5 | AP-ROP |
|---|---|---|---|---|---|---|
| recall | 0.9897 | 0.6087 | 0.4229 | 0.8626 | 0.8889 | 0.8857 |
| support | 776 | 92 | 201 | 131 | 18 | 35 |

#### Arm D · seed 1337 · fold_site (held-out source Shenzhen (the v3 site fold))

Kernel `rop-v3-dinov2-s-bag-s1337-lr1e-4` · run dir `results/rop/v3/runs/dinov2_s_bag_s1337/fold_site` · as-run manifest `results/prepared/rop_staging/_v3_foldsite.csv`

Fold file `results/prepared/rop_staging/fold_site.csv`: validation = 740 photos of 397 infants (shenzhen 740); training = 2388 photos of 1131 infants (ostrava 715, ropvl 1673).

| rows | normal | stage_1 | stage_2 | stage_3 | stage_4_5 | arop | total |
|---|---|---|---|---|---|---|---|
| training rows | 893 | 426 | 533 | 280 | 89 | 167 | 2388 |
| validation rows | 234 | 89 | 163 | 254 | 0 | 0 | 740 |

- images per epoch = 2190 under the bag loss (1673 single-image rows + 517 frames drawn from 168 multi-frame Ostrava sessions, at most k = 8 per session; kernel log); training rows = 2388
- steps per epoch = 189 (105 image batches + 84 bag batches; kernel log)
- minutes 28.0 (fold start to next fold start, kernel log) · 29.5 img/s · GPU Tesla T4 · kernel log reports train 2388 val 740

Per epoch (`metrics.json` history; lr = the first optimizer group, i.e. the lowest layer under layer-wise decay, at the end of the epoch):

| Epoch | train loss | lr | val acc | val macro-F1 | val macro-P | val macro-R |
|---|---|---|---|---|---|---|
| Epoch 1 | 1.6778 | 2.870e-06 | 0.2878 | 0.1434 | 0.1360 | 0.1517 |
| Epoch 2 | 1.5120 | 5.498e-06 | 0.4946 | 0.3657 | 0.3332 | 0.4238 |
| Epoch 3 | 1.3251 | 5.455e-06 | 0.5689 | 0.2957 | 0.5297 | 0.3551 |
| Epoch 4 | 1.1487 | 5.331e-06 | 0.7865 | 0.5179 | 0.5335 | 0.5104 |
| Epoch 5 | 1.0746 | 5.128e-06 | 0.8392 | 0.5396 | 0.5422 | 0.5395 |
| Epoch 6 | 1.0309 | 4.853e-06 | 0.7730 | 0.4216 | 0.4684 | 0.4376 |
| Epoch 7 | 1.0215 | 4.514e-06 | 0.8297 | 0.6355 | 0.6394 | 0.6475 |
| Epoch 8 | 0.9837 | 4.121e-06 | 0.7730 | 0.4142 | 0.4107 | 0.4448 |
| Epoch 9 | 0.9277 | 3.687e-06 | 0.8338 | 0.5246 | 0.5592 | 0.5127 |
| Epoch 10 | 0.8854 | 3.224e-06 | 0.8554 | 0.5336 | 0.5533 | 0.5258 |
| Epoch 11 | 0.8526 | 2.746e-06 | 0.8338 | 0.5240 | 0.5587 | 0.5102 |
| Epoch 12 | 0.8473 | 2.269e-06 | 0.8757 | 0.5586 | 0.5632 | 0.5545 |
| Epoch 13 | 0.8310 | 1.806e-06 | 0.8541 | 0.5365 | 0.5726 | 0.5255 |
| Epoch 14 | 0.8276 | 1.372e-06 | 0.8527 | 0.5492 | 0.5654 | 0.5432 |
| Epoch 15 | 0.7783 | 9.800e-07 | 0.8770 | 0.5602 | 0.5725 | 0.5530 |
| Epoch 16 | 0.7755 | 6.415e-07 | 0.8649 | 0.5402 | 0.5780 | 0.5309 |
| Epoch 17 | 0.7392 | 3.670e-07 | 0.8797 | 0.5633 | 0.5843 | 0.5525 |
| Epoch 18 | 0.7673 | 1.649e-07 | 0.8689 | 0.5485 | 0.5777 | 0.5376 |
| Epoch 19 | 0.7538 | 4.134e-08 | 0.8878 | 0.5681 | 0.5852 | 0.5592 |
| Epoch 20 | 0.7226 | 2.316e-11 | 0.8865 | 0.5657 | 0.5845 | 0.5565 |

Final epoch (Epoch 20), all 740 validation rows: QWK (6 labels, summary.json) 0.9508 · QWK5 (AP-ROP folded onto stage 4/5, val_predictions.npz) 0.9594 · accuracy 0.8865 · macro-F1 0.5657 · balanced accuracy 0.5565 · AUC n/a

| final epoch | Normal | Stage 1 | Stage 2 | Stage 3 | Stage 4/5 | AP-ROP |
|---|---|---|---|---|---|---|
| recall | 0.9957 | 0.5730 | 0.8528 | 0.9173 | 0.0000 | 0.0000 |
| support | 234 | 89 | 163 | 254 | 0 | 0 |

### Arm E: Ensemble A+B+C

No run directory: `ensemble(A,B,C)` (seeds [42, 1337]), scored by `scripts/rop_v3_score.py` from saved predictions; its numbers are in section 4.

### Arm F: DINOv2 + adaptation

Run dirs: `results/rop/v3/runs/dinov2_s_uda_s42`, `results/rop/v3/runs/dinov2_s_uda_s1337`. Missing seeds of [42, 1337]: none.

Recipe (the as-run `config.yaml` of fold_0, seed 42; any fold whose config differs says so):

| parameter | value |
|---|---|
| model.arch | dinov2_s |
| preprocess.image_size | 392 |
| preprocess.normalize | imagenet |
| preprocess.clahe | True |
| train.lr | 0.0001 |
| train.layer_decay | 0.8 |
| train.weight_decay | 0.05 |
| train.epochs | 20 |
| train.batch_size | 16 |
| train.grad_accum | 1 (default) |
| train.label_smoothing | 0.05 |
| train.loss | ce |
| train.class_balanced | True |
| train.class_balanced_beta | 0.999 |
| train.weighted_sampler | False |
| train.strong_aug | True |
| train.site_randomization | True |
| train.amp | True |
| train.ema_decay | not set |
| train.head_warmup_epochs | not set |
| model.pretrained | True |
| model.dropout | 0.3 |
| model.drop_path | 0.1 |
| eval.early_stopping | False |
| eval.primary_metric | macro_f1 |
| train.aug_v4 | not set |
| train.bag_loss | not set |
| train.uda.enabled | True |
| train.uda.method | self_ensembling |
| train.uda.ema_decay | 0.99 |
| train.uda.conf_threshold | 0.9 |
| train.uda.weight | 1 |
| train.uda.ramp_epochs | 2 |
| train.uda.target_manifest | results/prepared/rop_staging/test_dev.csv |
| train.uda.target_batch | 16 |
| pool (summary.json) | v3 |
| config source | configs/rop_staging_dinov2_uda.yaml |

<details><summary>Every key of that as-run config.yaml</summary>

| key | value |
|---|---|
| disease | rop_staging |
| seed | 42 |
| data.num_classes | 6 |
| data.class_names | [Normal, Stage 1, Stage 2, Stage 3, Stage 4/5, AP-ROP] |
| data.manifest | results/prepared/rop_staging/_v3_fold0.csv |
| data.presplit | True |
| data.group_col | patient_id |
| preprocess.image_size | 392 |
| preprocess.circle_crop | True |
| preprocess.letterbox | True |
| preprocess.ben_graham | False |
| preprocess.clahe | True |
| preprocess.normalize | imagenet |
| preprocess.cached | True |
| preprocess.cache_root | /tmp/_cache768 |
| model.arch | dinov2_s |
| model.weights_path | None |
| model.pretrained | True |
| model.head | classification |
| model.dropout | 0.3 |
| model.drop_path | 0.1 |
| train.epochs | 20 |
| train.batch_size | 16 |
| train.lr | 0.0001 |
| train.layer_decay | 0.8 |
| train.weight_decay | 0.05 |
| train.label_smoothing | 0.05 |
| train.amp | True |
| train.strong_aug | True |
| train.site_randomization | True |
| train.loss | ce |
| train.class_balanced | True |
| train.class_balanced_beta | 0.999 |
| train.weighted_sampler | False |
| train.num_workers | 4 |
| train.uda.enabled | True |
| train.uda.method | self_ensembling |
| train.uda.ema_decay | 0.99 |
| train.uda.conf_threshold | 0.9 |
| train.uda.weight | 1 |
| train.uda.ramp_epochs | 2 |
| train.uda.target_manifest | results/prepared/rop_staging/test_dev.csv |
| train.uda.target_batch | 16 |
| eval.early_stopping | False |
| eval.primary_metric | macro_f1 |
| eval.qwk_classes | [Normal, Stage 1, Stage 2, Stage 3, Stage 4/5] |
| eval.report_per_class_recall | True |
| eval.unvalidated_classes | [Stage 4/5, AP-ROP] |
| _source | configs/rop_staging_dinov2_uda.yaml |

</details>

#### Arm F · seed 42: overview (5 folds: fold_0, fold_1, fold_2, fold_3, fold_4)

| fold | train rows | val rows | images/epoch | steps/epoch | minutes | img/s | final acc | final macro-F1 | final QWK (6) | final QWK5 | AUC |
|---|---|---|---|---|---|---|---|---|---|---|---|
| fold_0 | 2462 | 1691 | 2462 | 154 | 60.8 | 14.6 | 0.6617 | 0.5995 | 0.5911 | 0.6732 | 0.9070 |
| fold_1 | 2477 | 1252 | 2477 | 155 | 57.8 | 15.0 | 0.8243 | 0.7505 | 0.8913 | 0.8858 | 0.9432 |
| fold_2 | 2542 | 1252 | 2542 | 159 | 58.8 | 15.3 | 0.7260 | 0.5866 | 0.8151 | 0.8406 | 0.9098 |
| fold_3 | 2459 | 1252 | 2459 | 154 | 61.9 | 13.8 | 0.8251 | 0.7899 | 0.9286 | 0.9272 | 0.9581 |
| fold_4 | 2452 | 1253 | 2452 | 154 | 62.3 | 14.1 | 0.8013 | 0.6714 | 0.8464 | 0.8783 | 0.9348 |

#### Arm F · seed 42 · fold_0 (CV fold 0)

Kernel `rop-v3-dinov2-s-uda-s42-lr1e-4-a` · run dir `results/rop/v3/runs/dinov2_s_uda_s42/fold_0` · as-run manifest `results/prepared/rop_staging/_v3_fold0.csv`

Fold file `results/prepared/rop_staging/fold_0.csv`: validation = 1691 photos of 297 infants (ostrava 1219, ropvl 310, shenzhen 162); training = 2462 photos of 1231 infants (ostrava 544, ropvl 1363, shenzhen 555).

| rows | normal | stage_1 | stage_2 | stage_3 | stage_4_5 | arop | total |
|---|---|---|---|---|---|---|---|
| training rows | 888 | 403 | 556 | 404 | 70 | 141 | 2462 |
| validation rows | 767 | 137 | 163 | 129 | 19 | 476 | 1691 |

- images per epoch = 2462 (the training rows: the fold file's 2474 less 12 exact duplicate(s) dropped at run time)
- steps per epoch = 154 (= ceil(2462 / (16 x 1)), batch_size x grad_accum)
- plus 16 unlabelled Multi-View dev photos per step (self-ensembling) = 2464 unlabelled images per epoch
- minutes 60.8 (fold start to next fold start, kernel log) · 14.6 img/s · GPU Tesla T4 · kernel log reports train 2462 val 1691

Per epoch (`metrics.json` history; lr = the first optimizer group, i.e. the lowest layer under layer-wise decay, at the end of the epoch):

| Epoch | train loss | lr | val acc | val macro-F1 | val macro-P | val macro-R |
|---|---|---|---|---|---|---|
| Epoch 1 | 1.9016 | 2.872e-06 | 0.3690 | 0.2250 | 0.3527 | 0.2698 |
| Epoch 2 | 1.7148 | 5.498e-06 | 0.6304 | 0.5300 | 0.5889 | 0.5260 |
| Epoch 3 | 1.3777 | 5.455e-06 | 0.7475 | 0.6266 | 0.6258 | 0.6820 |
| Epoch 4 | 1.2031 | 5.331e-06 | 0.7351 | 0.6215 | 0.6208 | 0.6951 |
| Epoch 5 | 1.1691 | 5.128e-06 | 0.6919 | 0.5974 | 0.6213 | 0.6632 |
| Epoch 6 | 1.0843 | 4.852e-06 | 0.6730 | 0.5683 | 0.6042 | 0.6453 |
| Epoch 7 | 1.0114 | 4.513e-06 | 0.6724 | 0.5687 | 0.6013 | 0.6541 |
| Epoch 8 | 0.9629 | 4.120e-06 | 0.6523 | 0.5528 | 0.5937 | 0.6389 |
| Epoch 9 | 0.9853 | 3.686e-06 | 0.6464 | 0.5446 | 0.5920 | 0.6501 |
| Epoch 10 | 0.9642 | 3.223e-06 | 0.6523 | 0.5577 | 0.6063 | 0.6564 |
| Epoch 11 | 0.9109 | 2.746e-06 | 0.6718 | 0.5719 | 0.6086 | 0.6525 |
| Epoch 12 | 0.8859 | 2.268e-06 | 0.6570 | 0.5807 | 0.6331 | 0.6590 |
| Epoch 13 | 0.8419 | 1.806e-06 | 0.6552 | 0.5895 | 0.6448 | 0.6636 |
| Epoch 14 | 0.8424 | 1.372e-06 | 0.6582 | 0.5883 | 0.6362 | 0.6515 |
| Epoch 15 | 0.8330 | 9.795e-07 | 0.6535 | 0.5872 | 0.6348 | 0.6620 |
| Epoch 16 | 0.7999 | 6.411e-07 | 0.6682 | 0.6045 | 0.6492 | 0.6739 |
| Epoch 17 | 0.7687 | 3.667e-07 | 0.6576 | 0.6013 | 0.6471 | 0.6776 |
| Epoch 18 | 0.7773 | 1.647e-07 | 0.6623 | 0.5995 | 0.6422 | 0.6816 |
| Epoch 19 | 0.7376 | 4.124e-08 | 0.6623 | 0.6000 | 0.6416 | 0.6846 |
| Epoch 20 | 0.7610 | 2.376e-11 | 0.6617 | 0.5995 | 0.6421 | 0.6856 |

Final epoch (Epoch 20), all 1691 validation rows: QWK (6 labels, summary.json) 0.5911 · QWK5 (AP-ROP folded onto stage 4/5, val_predictions.npz) 0.6732 · accuracy 0.6617 · macro-F1 0.5995 · balanced accuracy 0.6856 · AUC 0.9070

| final epoch | Normal | Stage 1 | Stage 2 | Stage 3 | Stage 4/5 | AP-ROP |
|---|---|---|---|---|---|---|
| recall | 0.9335 | 0.6204 | 0.4969 | 0.8992 | 0.9474 | 0.2164 |
| support | 767 | 137 | 163 | 129 | 19 | 476 |

#### Arm F · seed 42 · fold_1 (CV fold 1)

Kernel `rop-v3-dinov2-s-uda-s42-lr1e-4-a` · run dir `results/rop/v3/runs/dinov2_s_uda_s42/fold_1` · as-run manifest `results/prepared/rop_staging/_v3_fold1.csv`

Fold file `results/prepared/rop_staging/fold_1.csv`: validation = 1252 photos of 299 infants (ostrava 812, ropvl 327, shenzhen 113); training = 2477 photos of 1228 infants (ostrava 527, ropvl 1346, shenzhen 604).

| rows | normal | stage_1 | stage_2 | stage_3 | stage_4_5 | arop | total |
|---|---|---|---|---|---|---|---|
| training rows | 890 | 401 | 534 | 442 | 73 | 137 | 2477 |
| validation rows | 756 | 114 | 220 | 116 | 16 | 30 | 1252 |

- images per epoch = 2477 (the training rows: the fold file's 2486 less 9 exact duplicate(s) dropped at run time)
- steps per epoch = 155 (= ceil(2477 / (16 x 1)), batch_size x grad_accum)
- plus 16 unlabelled Multi-View dev photos per step (self-ensembling) = 2480 unlabelled images per epoch
- minutes 57.8 (fold start to next fold start, kernel log) · 15.0 img/s · GPU Tesla T4 · kernel log reports train 2477 val 1252

Per epoch (`metrics.json` history; lr = the first optimizer group, i.e. the lowest layer under layer-wise decay, at the end of the epoch):

| Epoch | train loss | lr | val acc | val macro-F1 | val macro-P | val macro-R |
|---|---|---|---|---|---|---|
| Epoch 1 | 1.8889 | 2.872e-06 | 0.5998 | 0.2666 | 0.3446 | 0.2633 |
| Epoch 2 | 1.6786 | 5.498e-06 | 0.7492 | 0.5569 | 0.6602 | 0.5339 |
| Epoch 3 | 1.3692 | 5.455e-06 | 0.7780 | 0.6173 | 0.6297 | 0.6216 |
| Epoch 4 | 1.2351 | 5.331e-06 | 0.7907 | 0.6669 | 0.6804 | 0.6721 |
| Epoch 5 | 1.1455 | 5.128e-06 | 0.8019 | 0.6890 | 0.7034 | 0.6901 |
| Epoch 6 | 1.0520 | 4.852e-06 | 0.8171 | 0.7142 | 0.7220 | 0.7157 |
| Epoch 7 | 1.0493 | 4.513e-06 | 0.8115 | 0.7289 | 0.7405 | 0.7287 |
| Epoch 8 | 0.9782 | 4.120e-06 | 0.8123 | 0.7273 | 0.7503 | 0.7220 |
| Epoch 9 | 0.9514 | 3.686e-06 | 0.8235 | 0.7391 | 0.7579 | 0.7369 |
| Epoch 10 | 0.9013 | 3.223e-06 | 0.8203 | 0.7403 | 0.7499 | 0.7511 |
| Epoch 11 | 0.9005 | 2.746e-06 | 0.8187 | 0.7456 | 0.7660 | 0.7485 |
| Epoch 12 | 0.8909 | 2.268e-06 | 0.8243 | 0.7475 | 0.7735 | 0.7459 |
| Epoch 13 | 0.8455 | 1.806e-06 | 0.8251 | 0.7494 | 0.7640 | 0.7490 |
| Epoch 14 | 0.8097 | 1.372e-06 | 0.8259 | 0.7472 | 0.7633 | 0.7449 |
| Epoch 15 | 0.7979 | 9.795e-07 | 0.8331 | 0.7563 | 0.7772 | 0.7528 |
| Epoch 16 | 0.7880 | 6.411e-07 | 0.8267 | 0.7447 | 0.7589 | 0.7454 |
| Epoch 17 | 0.7750 | 3.667e-07 | 0.8219 | 0.7521 | 0.7762 | 0.7491 |
| Epoch 18 | 0.7647 | 1.647e-07 | 0.8243 | 0.7528 | 0.7731 | 0.7480 |
| Epoch 19 | 0.7503 | 4.125e-08 | 0.8227 | 0.7523 | 0.7753 | 0.7479 |
| Epoch 20 | 0.7458 | 2.373e-11 | 0.8243 | 0.7505 | 0.7726 | 0.7446 |

Final epoch (Epoch 20), all 1252 validation rows: QWK (6 labels, summary.json) 0.8913 · QWK5 (AP-ROP folded onto stage 4/5, val_predictions.npz) 0.8858 · accuracy 0.8243 · macro-F1 0.7505 · balanced accuracy 0.7446 · AUC 0.9432

| final epoch | Normal | Stage 1 | Stage 2 | Stage 3 | Stage 4/5 | AP-ROP |
|---|---|---|---|---|---|---|
| recall | 0.9749 | 0.5175 | 0.5136 | 0.7241 | 0.9375 | 0.8000 |
| support | 756 | 114 | 220 | 116 | 16 | 30 |

#### Arm F · seed 42 · fold_2 (CV fold 2)

Kernel `rop-v3-dinov2-s-uda-s42-lr1e-4-a` · run dir `results/rop/v3/runs/dinov2_s_uda_s42/fold_2` · as-run manifest `results/prepared/rop_staging/_v3_fold2.csv`

Fold file `results/prepared/rop_staging/fold_2.csv`: validation = 1252 photos of 272 infants (ostrava 830, ropvl 295, shenzhen 127); training = 2542 photos of 1255 infants (ostrava 573, ropvl 1378, shenzhen 591).

| rows | normal | stage_1 | stage_2 | stage_3 | stage_4_5 | arop | total |
|---|---|---|---|---|---|---|---|
| training rows | 888 | 415 | 556 | 481 | 71 | 131 | 2542 |
| validation rows | 772 | 123 | 193 | 110 | 18 | 36 | 1252 |

- images per epoch = 2542 (the training rows: the fold file's 2553 less 11 exact duplicate(s) dropped at run time)
- steps per epoch = 159 (= ceil(2542 / (16 x 1)), batch_size x grad_accum)
- plus 16 unlabelled Multi-View dev photos per step (self-ensembling) = 2544 unlabelled images per epoch
- minutes 58.8 (fold start to next fold start, kernel log) · 15.3 img/s · GPU Tesla T4 · kernel log reports train 2542 val 1252

Per epoch (`metrics.json` history; lr = the first optimizer group, i.e. the lowest layer under layer-wise decay, at the end of the epoch):

| Epoch | train loss | lr | val acc | val macro-F1 | val macro-P | val macro-R |
|---|---|---|---|---|---|---|
| Epoch 1 | 1.9063 | 2.872e-06 | 0.6038 | 0.2649 | 0.3980 | 0.2570 |
| Epoch 2 | 1.6877 | 5.498e-06 | 0.7204 | 0.4531 | 0.5210 | 0.4613 |
| Epoch 3 | 1.3710 | 5.455e-06 | 0.7444 | 0.4886 | 0.5146 | 0.5471 |
| Epoch 4 | 1.1988 | 5.331e-06 | 0.7532 | 0.5271 | 0.5302 | 0.6210 |
| Epoch 5 | 1.1582 | 5.128e-06 | 0.7620 | 0.5672 | 0.5658 | 0.6580 |
| Epoch 6 | 1.0672 | 4.853e-06 | 0.7364 | 0.5415 | 0.5438 | 0.6311 |
| Epoch 7 | 1.0321 | 4.513e-06 | 0.7300 | 0.5524 | 0.5456 | 0.6415 |
| Epoch 8 | 0.9577 | 4.121e-06 | 0.7556 | 0.5925 | 0.5997 | 0.6624 |
| Epoch 9 | 0.9584 | 3.686e-06 | 0.7364 | 0.5723 | 0.5599 | 0.6527 |
| Epoch 10 | 0.9305 | 3.223e-06 | 0.7173 | 0.5502 | 0.5383 | 0.6341 |
| Epoch 11 | 0.9294 | 2.746e-06 | 0.7308 | 0.5692 | 0.5589 | 0.6471 |
| Epoch 12 | 0.8853 | 2.269e-06 | 0.7548 | 0.6185 | 0.6191 | 0.6637 |
| Epoch 13 | 0.8557 | 1.806e-06 | 0.7252 | 0.5813 | 0.5812 | 0.6420 |
| Epoch 14 | 0.8325 | 1.372e-06 | 0.7484 | 0.6069 | 0.6101 | 0.6595 |
| Epoch 15 | 0.8281 | 9.796e-07 | 0.7324 | 0.5852 | 0.5802 | 0.6437 |
| Epoch 16 | 0.8173 | 6.412e-07 | 0.7228 | 0.5757 | 0.5694 | 0.6304 |
| Epoch 17 | 0.7971 | 3.668e-07 | 0.7252 | 0.5820 | 0.5763 | 0.6429 |
| Epoch 18 | 0.7841 | 1.648e-07 | 0.7220 | 0.5880 | 0.5828 | 0.6453 |
| Epoch 19 | 0.7614 | 4.126e-08 | 0.7220 | 0.5846 | 0.5808 | 0.6413 |
| Epoch 20 | 0.7730 | 2.365e-11 | 0.7260 | 0.5866 | 0.5816 | 0.6420 |

Final epoch (Epoch 20), all 1252 validation rows: QWK (6 labels, summary.json) 0.8151 · QWK5 (AP-ROP folded onto stage 4/5, val_predictions.npz) 0.8406 · accuracy 0.7260 · macro-F1 0.5866 · balanced accuracy 0.6420 · AUC 0.9098

| final epoch | Normal | Stage 1 | Stage 2 | Stage 3 | Stage 4/5 | AP-ROP |
|---|---|---|---|---|---|---|
| recall | 0.8692 | 0.4715 | 0.5440 | 0.2727 | 0.8889 | 0.8056 |
| support | 772 | 123 | 193 | 110 | 18 | 36 |

#### Arm F · seed 42 · fold_3 (CV fold 3)

Kernel `rop-v3-dinov2-s-uda-s42-lr1e-4-b` · run dir `results/rop/v3/runs/dinov2_s_uda_s42/fold_3` · as-run manifest `results/prepared/rop_staging/_v3_fold3.csv`

Fold file `results/prepared/rop_staging/fold_3.csv`: validation = 1252 photos of 327 infants (ostrava 713, ropvl 362, shenzhen 177); training = 2459 photos of 1200 infants (ostrava 609, ropvl 1313, shenzhen 537).

| rows | normal | stage_1 | stage_2 | stage_3 | stage_4_5 | arop | total |
|---|---|---|---|---|---|---|---|
| training rows | 888 | 409 | 564 | 400 | 71 | 127 | 2459 |
| validation rows | 766 | 111 | 182 | 134 | 18 | 41 | 1252 |

- images per epoch = 2459 (the training rows: the fold file's 2469 less 10 exact duplicate(s) dropped at run time)
- steps per epoch = 154 (= ceil(2459 / (16 x 1)), batch_size x grad_accum)
- plus 16 unlabelled Multi-View dev photos per step (self-ensembling) = 2464 unlabelled images per epoch
- minutes 61.9 (fold start to next fold start, kernel log) · 13.8 img/s · GPU Tesla T4 · kernel log reports train 2459 val 1252

Per epoch (`metrics.json` history; lr = the first optimizer group, i.e. the lowest layer under layer-wise decay, at the end of the epoch):

| Epoch | train loss | lr | val acc | val macro-F1 | val macro-P | val macro-R |
|---|---|---|---|---|---|---|
| Epoch 1 | 1.9046 | 2.872e-06 | 0.5000 | 0.2895 | 0.3877 | 0.2840 |
| Epoch 2 | 1.7107 | 5.498e-06 | 0.7947 | 0.6433 | 0.6512 | 0.6538 |
| Epoch 3 | 1.4142 | 5.455e-06 | 0.7859 | 0.6638 | 0.6465 | 0.6934 |
| Epoch 4 | 1.2408 | 5.331e-06 | 0.8099 | 0.7152 | 0.6922 | 0.7458 |
| Epoch 5 | 1.1566 | 5.128e-06 | 0.7875 | 0.7218 | 0.7148 | 0.7478 |
| Epoch 6 | 1.1191 | 4.852e-06 | 0.7931 | 0.7465 | 0.7471 | 0.7812 |
| Epoch 7 | 1.0542 | 4.513e-06 | 0.7612 | 0.7245 | 0.7257 | 0.7623 |
| Epoch 8 | 0.9931 | 4.120e-06 | 0.8043 | 0.7603 | 0.7604 | 0.7777 |
| Epoch 9 | 0.9893 | 3.686e-06 | 0.7875 | 0.7574 | 0.7676 | 0.7740 |
| Epoch 10 | 0.9484 | 3.223e-06 | 0.8179 | 0.7781 | 0.7884 | 0.7892 |
| Epoch 11 | 0.9217 | 2.746e-06 | 0.8179 | 0.7697 | 0.7683 | 0.7867 |
| Epoch 12 | 0.9247 | 2.268e-06 | 0.8179 | 0.7760 | 0.7837 | 0.7864 |
| Epoch 13 | 0.8699 | 1.806e-06 | 0.8243 | 0.7850 | 0.7918 | 0.7949 |
| Epoch 14 | 0.8433 | 1.372e-06 | 0.8139 | 0.7826 | 0.7903 | 0.7985 |
| Epoch 15 | 0.8216 | 9.795e-07 | 0.8171 | 0.7841 | 0.7886 | 0.8008 |
| Epoch 16 | 0.7994 | 6.411e-07 | 0.8299 | 0.7840 | 0.7864 | 0.7957 |
| Epoch 17 | 0.7828 | 3.667e-07 | 0.8267 | 0.7888 | 0.7913 | 0.8044 |
| Epoch 18 | 0.7801 | 1.647e-07 | 0.8107 | 0.7791 | 0.7787 | 0.8030 |
| Epoch 19 | 0.7647 | 4.124e-08 | 0.8267 | 0.7909 | 0.7941 | 0.8060 |
| Epoch 20 | 0.7692 | 2.376e-11 | 0.8251 | 0.7899 | 0.7929 | 0.8050 |

Final epoch (Epoch 20), all 1252 validation rows: QWK (6 labels, summary.json) 0.9286 · QWK5 (AP-ROP folded onto stage 4/5, val_predictions.npz) 0.9272 · accuracy 0.8251 · macro-F1 0.7899 · balanced accuracy 0.8050 · AUC 0.9581

| final epoch | Normal | Stage 1 | Stage 2 | Stage 3 | Stage 4/5 | AP-ROP |
|---|---|---|---|---|---|---|
| recall | 0.8642 | 0.6486 | 0.7308 | 0.8507 | 0.8333 | 0.9024 |
| support | 766 | 111 | 182 | 134 | 18 | 41 |

#### Arm F · seed 42 · fold_4 (CV fold 4)

Kernel `rop-v3-dinov2-s-uda-s42-lr1e-4-b` · run dir `results/rop/v3/runs/dinov2_s_uda_s42/fold_4` · as-run manifest `results/prepared/rop_staging/_v3_fold4.csv`

Fold file `results/prepared/rop_staging/fold_4.csv`: validation = 1253 photos of 333 infants (ostrava 711, ropvl 381, shenzhen 161); training = 2452 photos of 1194 infants (ostrava 607, ropvl 1292, shenzhen 553).

| rows | normal | stage_1 | stage_2 | stage_3 | stage_4_5 | arop | total |
|---|---|---|---|---|---|---|---|
| training rows | 882 | 420 | 546 | 401 | 71 | 132 | 2452 |
| validation rows | 776 | 92 | 201 | 131 | 18 | 35 | 1253 |

- images per epoch = 2452 (the training rows: the fold file's 2466 less 14 exact duplicate(s) dropped at run time)
- steps per epoch = 154 (= ceil(2452 / (16 x 1)), batch_size x grad_accum)
- plus 16 unlabelled Multi-View dev photos per step (self-ensembling) = 2464 unlabelled images per epoch
- minutes 62.3 (fold start to next fold start, kernel log) · 14.1 img/s · GPU Tesla T4 · kernel log reports train 2452 val 1253

Per epoch (`metrics.json` history; lr = the first optimizer group, i.e. the lowest layer under layer-wise decay, at the end of the epoch):

| Epoch | train loss | lr | val acc | val macro-F1 | val macro-P | val macro-R |
|---|---|---|---|---|---|---|
| Epoch 1 | 1.9228 | 2.872e-06 | 0.5642 | 0.2976 | 0.4202 | 0.2728 |
| Epoch 2 | 1.7231 | 5.498e-06 | 0.7654 | 0.5660 | 0.5656 | 0.6314 |
| Epoch 3 | 1.3638 | 5.455e-06 | 0.7989 | 0.5771 | 0.5535 | 0.6766 |
| Epoch 4 | 1.2164 | 5.331e-06 | 0.7925 | 0.6109 | 0.5789 | 0.6936 |
| Epoch 5 | 1.1619 | 5.128e-06 | 0.8077 | 0.6185 | 0.5872 | 0.7169 |
| Epoch 6 | 1.0912 | 4.852e-06 | 0.8156 | 0.6470 | 0.6225 | 0.7310 |
| Epoch 7 | 1.0870 | 4.513e-06 | 0.7941 | 0.6407 | 0.6165 | 0.7287 |
| Epoch 8 | 0.9976 | 4.120e-06 | 0.8029 | 0.6558 | 0.6324 | 0.7373 |
| Epoch 9 | 0.9360 | 3.686e-06 | 0.8132 | 0.6645 | 0.6387 | 0.7362 |
| Epoch 10 | 0.9450 | 3.223e-06 | 0.8284 | 0.6842 | 0.6577 | 0.7508 |
| Epoch 11 | 0.9239 | 2.746e-06 | 0.8172 | 0.6728 | 0.6488 | 0.7394 |
| Epoch 12 | 0.8644 | 2.268e-06 | 0.8093 | 0.6809 | 0.6535 | 0.7505 |
| Epoch 13 | 0.8543 | 1.806e-06 | 0.8164 | 0.6849 | 0.6565 | 0.7508 |
| Epoch 14 | 0.8409 | 1.372e-06 | 0.8085 | 0.6838 | 0.6582 | 0.7536 |
| Epoch 15 | 0.8173 | 9.795e-07 | 0.8172 | 0.7005 | 0.6732 | 0.7648 |
| Epoch 16 | 0.8005 | 6.411e-07 | 0.8188 | 0.6845 | 0.6560 | 0.7459 |
| Epoch 17 | 0.7836 | 3.667e-07 | 0.8125 | 0.6822 | 0.6545 | 0.7538 |
| Epoch 18 | 0.7579 | 1.647e-07 | 0.8164 | 0.6857 | 0.6577 | 0.7577 |
| Epoch 19 | 0.7613 | 4.124e-08 | 0.8109 | 0.6795 | 0.6518 | 0.7531 |
| Epoch 20 | 0.7799 | 2.376e-11 | 0.8013 | 0.6714 | 0.6454 | 0.7448 |

Final epoch (Epoch 20), all 1253 validation rows: QWK (6 labels, summary.json) 0.8464 · QWK5 (AP-ROP folded onto stage 4/5, val_predictions.npz) 0.8783 · accuracy 0.8013 · macro-F1 0.6714 · balanced accuracy 0.7448 · AUC 0.9348

| final epoch | Normal | Stage 1 | Stage 2 | Stage 3 | Stage 4/5 | AP-ROP |
|---|---|---|---|---|---|---|
| recall | 0.9072 | 0.5761 | 0.4627 | 0.8321 | 0.8333 | 0.8571 |
| support | 776 | 92 | 201 | 131 | 18 | 35 |

#### Arm F · seed 1337: overview (5 folds: fold_0, fold_1, fold_2, fold_3, fold_4)

| fold | train rows | val rows | images/epoch | steps/epoch | minutes | img/s | final acc | final macro-F1 | final QWK (6) | final QWK5 | AUC |
|---|---|---|---|---|---|---|---|---|---|---|---|
| fold_0 | 2462 | 1691 | 2462 | 154 | 65.9 | 13.2 | 0.6363 | 0.5244 | 0.5516 | 0.6482 | 0.8813 |
| fold_1 | 2477 | 1252 | 2477 | 155 | 63.0 | 13.6 | 0.8331 | 0.7621 | 0.8943 | 0.8867 | 0.9417 |
| fold_2 | 2542 | 1252 | 2542 | 159 | 64.6 | 14.0 | 0.7556 | 0.6247 | 0.8688 | 0.8789 | 0.9126 |
| fold_3 | 2459 | 1252 | 2459 | 154 | 60.4 | 14.2 | 0.7843 | 0.7557 | 0.9111 | 0.9124 | 0.9477 |
| fold_4 | 2452 | 1253 | 2452 | 154 | 61.7 | 14.7 | 0.8324 | 0.7437 | 0.9005 | 0.9097 | 0.9433 |

#### Arm F · seed 1337 · fold_0 (CV fold 0)

Kernel `rop-v3-dinov2-s-uda-s1337-lr1e-4-a` · run dir `results/rop/v3/runs/dinov2_s_uda_s1337/fold_0` · as-run manifest `results/prepared/rop_staging/_v3_fold0.csv`

Fold file `results/prepared/rop_staging/fold_0.csv`: validation = 1691 photos of 297 infants (ostrava 1219, ropvl 310, shenzhen 162); training = 2462 photos of 1231 infants (ostrava 544, ropvl 1363, shenzhen 555).

| rows | normal | stage_1 | stage_2 | stage_3 | stage_4_5 | arop | total |
|---|---|---|---|---|---|---|---|
| training rows | 888 | 403 | 556 | 404 | 70 | 141 | 2462 |
| validation rows | 767 | 137 | 163 | 129 | 19 | 476 | 1691 |

- images per epoch = 2462 (the training rows: the fold file's 2474 less 12 exact duplicate(s) dropped at run time)
- steps per epoch = 154 (= ceil(2462 / (16 x 1)), batch_size x grad_accum)
- plus 16 unlabelled Multi-View dev photos per step (self-ensembling) = 2464 unlabelled images per epoch
- minutes 65.9 (fold start to next fold start, kernel log) · 13.2 img/s · GPU Tesla T4 · kernel log reports train 2462 val 1691

Per epoch (`metrics.json` history; lr = the first optimizer group, i.e. the lowest layer under layer-wise decay, at the end of the epoch):

| Epoch | train loss | lr | val acc | val macro-F1 | val macro-P | val macro-R |
|---|---|---|---|---|---|---|
| Epoch 1 | 1.8674 | 2.872e-06 | 0.4891 | 0.2417 | 0.4136 | 0.3019 |
| Epoch 2 | 1.6824 | 5.498e-06 | 0.6706 | 0.4962 | 0.5183 | 0.5574 |
| Epoch 3 | 1.4134 | 5.455e-06 | 0.7534 | 0.6528 | 0.6510 | 0.6935 |
| Epoch 4 | 1.2361 | 5.331e-06 | 0.7339 | 0.6206 | 0.6267 | 0.6982 |
| Epoch 5 | 1.1363 | 5.128e-06 | 0.6996 | 0.5783 | 0.5972 | 0.6844 |
| Epoch 6 | 1.0957 | 4.852e-06 | 0.6759 | 0.5579 | 0.5859 | 0.6802 |
| Epoch 7 | 1.0153 | 4.513e-06 | 0.6901 | 0.5907 | 0.6238 | 0.6865 |
| Epoch 8 | 0.9933 | 4.120e-06 | 0.6712 | 0.5891 | 0.6324 | 0.6867 |
| Epoch 9 | 0.9687 | 3.686e-06 | 0.6943 | 0.5944 | 0.6240 | 0.6973 |
| Epoch 10 | 0.9274 | 3.223e-06 | 0.6759 | 0.5624 | 0.5972 | 0.6738 |
| Epoch 11 | 0.9284 | 2.746e-06 | 0.6665 | 0.5675 | 0.6122 | 0.6761 |
| Epoch 12 | 0.8522 | 2.268e-06 | 0.6600 | 0.5661 | 0.6118 | 0.6679 |
| Epoch 13 | 0.8917 | 1.806e-06 | 0.6647 | 0.5812 | 0.6248 | 0.6812 |
| Epoch 14 | 0.8083 | 1.372e-06 | 0.6671 | 0.5785 | 0.6183 | 0.6884 |
| Epoch 15 | 0.8175 | 9.795e-07 | 0.6535 | 0.5536 | 0.5982 | 0.6737 |
| Epoch 16 | 0.7958 | 6.411e-07 | 0.6523 | 0.5390 | 0.5823 | 0.6664 |
| Epoch 17 | 0.7883 | 3.667e-07 | 0.6529 | 0.5370 | 0.5765 | 0.6604 |
| Epoch 18 | 0.7746 | 1.647e-07 | 0.6404 | 0.5203 | 0.5676 | 0.6516 |
| Epoch 19 | 0.7731 | 4.124e-08 | 0.6334 | 0.5196 | 0.5715 | 0.6578 |
| Epoch 20 | 0.7657 | 2.376e-11 | 0.6363 | 0.5244 | 0.5749 | 0.6584 |

Final epoch (Epoch 20), all 1691 validation rows: QWK (6 labels, summary.json) 0.5516 · QWK5 (AP-ROP folded onto stage 4/5, val_predictions.npz) 0.6482 · accuracy 0.6363 · macro-F1 0.5244 · balanced accuracy 0.6584 · AUC 0.8813

| final epoch | Normal | Stage 1 | Stage 2 | Stage 3 | Stage 4/5 | AP-ROP |
|---|---|---|---|---|---|---|
| recall | 0.9531 | 0.5912 | 0.5215 | 0.8217 | 0.9474 | 0.1155 |
| support | 767 | 137 | 163 | 129 | 19 | 476 |

#### Arm F · seed 1337 · fold_1 (CV fold 1)

Kernel `rop-v3-dinov2-s-uda-s1337-lr1e-4-a` · run dir `results/rop/v3/runs/dinov2_s_uda_s1337/fold_1` · as-run manifest `results/prepared/rop_staging/_v3_fold1.csv`

Fold file `results/prepared/rop_staging/fold_1.csv`: validation = 1252 photos of 299 infants (ostrava 812, ropvl 327, shenzhen 113); training = 2477 photos of 1228 infants (ostrava 527, ropvl 1346, shenzhen 604).

| rows | normal | stage_1 | stage_2 | stage_3 | stage_4_5 | arop | total |
|---|---|---|---|---|---|---|---|
| training rows | 890 | 401 | 534 | 442 | 73 | 137 | 2477 |
| validation rows | 756 | 114 | 220 | 116 | 16 | 30 | 1252 |

- images per epoch = 2477 (the training rows: the fold file's 2486 less 9 exact duplicate(s) dropped at run time)
- steps per epoch = 155 (= ceil(2477 / (16 x 1)), batch_size x grad_accum)
- plus 16 unlabelled Multi-View dev photos per step (self-ensembling) = 2480 unlabelled images per epoch
- minutes 63.0 (fold start to next fold start, kernel log) · 13.6 img/s · GPU Tesla T4 · kernel log reports train 2477 val 1252

Per epoch (`metrics.json` history; lr = the first optimizer group, i.e. the lowest layer under layer-wise decay, at the end of the epoch):

| Epoch | train loss | lr | val acc | val macro-F1 | val macro-P | val macro-R |
|---|---|---|---|---|---|---|
| Epoch 1 | 1.8808 | 2.872e-06 | 0.6550 | 0.3153 | 0.3263 | 0.3371 |
| Epoch 2 | 1.6763 | 5.498e-06 | 0.7308 | 0.5708 | 0.6462 | 0.5453 |
| Epoch 3 | 1.3566 | 5.455e-06 | 0.8011 | 0.6730 | 0.6870 | 0.6765 |
| Epoch 4 | 1.2062 | 5.331e-06 | 0.7971 | 0.6735 | 0.6846 | 0.6779 |
| Epoch 5 | 1.1281 | 5.128e-06 | 0.8067 | 0.6951 | 0.7070 | 0.6968 |
| Epoch 6 | 1.0644 | 4.852e-06 | 0.8123 | 0.7208 | 0.7344 | 0.7167 |
| Epoch 7 | 1.0128 | 4.513e-06 | 0.8123 | 0.7100 | 0.7187 | 0.7161 |
| Epoch 8 | 0.9943 | 4.120e-06 | 0.8115 | 0.7155 | 0.7304 | 0.7131 |
| Epoch 9 | 0.9484 | 3.686e-06 | 0.8195 | 0.7212 | 0.7382 | 0.7241 |
| Epoch 10 | 0.9071 | 3.223e-06 | 0.8187 | 0.7221 | 0.7368 | 0.7259 |
| Epoch 11 | 0.8854 | 2.746e-06 | 0.8211 | 0.7387 | 0.7500 | 0.7393 |
| Epoch 12 | 0.8819 | 2.268e-06 | 0.8203 | 0.7365 | 0.7470 | 0.7411 |
| Epoch 13 | 0.8284 | 1.806e-06 | 0.8187 | 0.7308 | 0.7365 | 0.7367 |
| Epoch 14 | 0.7916 | 1.372e-06 | 0.8251 | 0.7441 | 0.7575 | 0.7427 |
| Epoch 15 | 0.7994 | 9.795e-07 | 0.8275 | 0.7544 | 0.7751 | 0.7478 |
| Epoch 16 | 0.7695 | 6.411e-07 | 0.8275 | 0.7558 | 0.7774 | 0.7496 |
| Epoch 17 | 0.7631 | 3.667e-07 | 0.8291 | 0.7596 | 0.7802 | 0.7555 |
| Epoch 18 | 0.7639 | 1.647e-07 | 0.8339 | 0.7619 | 0.7818 | 0.7553 |
| Epoch 19 | 0.7429 | 4.125e-08 | 0.8363 | 0.7683 | 0.7824 | 0.7699 |
| Epoch 20 | 0.7526 | 2.373e-11 | 0.8331 | 0.7621 | 0.7773 | 0.7626 |

Final epoch (Epoch 20), all 1252 validation rows: QWK (6 labels, summary.json) 0.8943 · QWK5 (AP-ROP folded onto stage 4/5, val_predictions.npz) 0.8867 · accuracy 0.8331 · macro-F1 0.7621 · balanced accuracy 0.7626 · AUC 0.9417

| final epoch | Normal | Stage 1 | Stage 2 | Stage 3 | Stage 4/5 | AP-ROP |
|---|---|---|---|---|---|---|
| recall | 0.9868 | 0.5526 | 0.5091 | 0.6897 | 0.9375 | 0.9000 |
| support | 756 | 114 | 220 | 116 | 16 | 30 |

#### Arm F · seed 1337 · fold_2 (CV fold 2)

Kernel `rop-v3-dinov2-s-uda-s1337-lr1e-4-a` · run dir `results/rop/v3/runs/dinov2_s_uda_s1337/fold_2` · as-run manifest `results/prepared/rop_staging/_v3_fold2.csv`

Fold file `results/prepared/rop_staging/fold_2.csv`: validation = 1252 photos of 272 infants (ostrava 830, ropvl 295, shenzhen 127); training = 2542 photos of 1255 infants (ostrava 573, ropvl 1378, shenzhen 591).

| rows | normal | stage_1 | stage_2 | stage_3 | stage_4_5 | arop | total |
|---|---|---|---|---|---|---|---|
| training rows | 888 | 415 | 556 | 481 | 71 | 131 | 2542 |
| validation rows | 772 | 123 | 193 | 110 | 18 | 36 | 1252 |

- images per epoch = 2542 (the training rows: the fold file's 2553 less 11 exact duplicate(s) dropped at run time)
- steps per epoch = 159 (= ceil(2542 / (16 x 1)), batch_size x grad_accum)
- plus 16 unlabelled Multi-View dev photos per step (self-ensembling) = 2544 unlabelled images per epoch
- minutes 64.6 (fold start to next fold start, kernel log) · 14.0 img/s · GPU Tesla T4 · kernel log reports train 2542 val 1252

Per epoch (`metrics.json` history; lr = the first optimizer group, i.e. the lowest layer under layer-wise decay, at the end of the epoch):

| Epoch | train loss | lr | val acc | val macro-F1 | val macro-P | val macro-R |
|---|---|---|---|---|---|---|
| Epoch 1 | 1.9037 | 2.872e-06 | 0.6334 | 0.2674 | 0.3322 | 0.2965 |
| Epoch 2 | 1.6691 | 5.498e-06 | 0.6997 | 0.4610 | 0.4981 | 0.4905 |
| Epoch 3 | 1.3786 | 5.455e-06 | 0.7356 | 0.5095 | 0.5398 | 0.5634 |
| Epoch 4 | 1.2036 | 5.331e-06 | 0.7356 | 0.5292 | 0.5647 | 0.5827 |
| Epoch 5 | 1.1481 | 5.128e-06 | 0.7444 | 0.5528 | 0.5795 | 0.6294 |
| Epoch 6 | 1.0839 | 4.853e-06 | 0.7212 | 0.5536 | 0.5604 | 0.6506 |
| Epoch 7 | 1.0434 | 4.513e-06 | 0.7324 | 0.5551 | 0.5796 | 0.6440 |
| Epoch 8 | 1.0059 | 4.121e-06 | 0.6957 | 0.5496 | 0.5867 | 0.6443 |
| Epoch 9 | 0.9756 | 3.686e-06 | 0.6518 | 0.5407 | 0.5672 | 0.6353 |
| Epoch 10 | 0.9299 | 3.223e-06 | 0.6693 | 0.5692 | 0.5796 | 0.6532 |
| Epoch 11 | 0.9244 | 2.746e-06 | 0.6965 | 0.5812 | 0.5916 | 0.6608 |
| Epoch 12 | 0.8926 | 2.269e-06 | 0.7181 | 0.5996 | 0.6143 | 0.6563 |
| Epoch 13 | 0.8526 | 1.806e-06 | 0.7204 | 0.6290 | 0.6528 | 0.6669 |
| Epoch 14 | 0.8509 | 1.372e-06 | 0.7692 | 0.6207 | 0.6138 | 0.6797 |
| Epoch 15 | 0.8156 | 9.796e-07 | 0.7812 | 0.6361 | 0.6440 | 0.6813 |
| Epoch 16 | 0.8272 | 6.412e-07 | 0.7883 | 0.6487 | 0.6578 | 0.6848 |
| Epoch 17 | 0.8028 | 3.668e-07 | 0.7708 | 0.6418 | 0.6499 | 0.6829 |
| Epoch 18 | 0.7795 | 1.648e-07 | 0.7548 | 0.6288 | 0.6384 | 0.6760 |
| Epoch 19 | 0.7777 | 4.126e-08 | 0.7556 | 0.6244 | 0.6319 | 0.6762 |
| Epoch 20 | 0.7716 | 2.365e-11 | 0.7556 | 0.6247 | 0.6323 | 0.6767 |

Final epoch (Epoch 20), all 1252 validation rows: QWK (6 labels, summary.json) 0.8688 · QWK5 (AP-ROP folded onto stage 4/5, val_predictions.npz) 0.8789 · accuracy 0.7556 · macro-F1 0.6247 · balanced accuracy 0.6767 · AUC 0.9126

| final epoch | Normal | Stage 1 | Stage 2 | Stage 3 | Stage 4/5 | AP-ROP |
|---|---|---|---|---|---|---|
| recall | 0.8886 | 0.5447 | 0.6218 | 0.2273 | 0.8889 | 0.8889 |
| support | 772 | 123 | 193 | 110 | 18 | 36 |

#### Arm F · seed 1337 · fold_3 (CV fold 3)

Kernel `rop-v3-dinov2-s-uda-s1337-lr1e-4-b` · run dir `results/rop/v3/runs/dinov2_s_uda_s1337/fold_3` · as-run manifest `results/prepared/rop_staging/_v3_fold3.csv`

Fold file `results/prepared/rop_staging/fold_3.csv`: validation = 1252 photos of 327 infants (ostrava 713, ropvl 362, shenzhen 177); training = 2459 photos of 1200 infants (ostrava 609, ropvl 1313, shenzhen 537).

| rows | normal | stage_1 | stage_2 | stage_3 | stage_4_5 | arop | total |
|---|---|---|---|---|---|---|---|
| training rows | 888 | 409 | 564 | 400 | 71 | 127 | 2459 |
| validation rows | 766 | 111 | 182 | 134 | 18 | 41 | 1252 |

- images per epoch = 2459 (the training rows: the fold file's 2469 less 10 exact duplicate(s) dropped at run time)
- steps per epoch = 154 (= ceil(2459 / (16 x 1)), batch_size x grad_accum)
- plus 16 unlabelled Multi-View dev photos per step (self-ensembling) = 2464 unlabelled images per epoch
- minutes 60.4 (fold start to next fold start, kernel log) · 14.2 img/s · GPU Tesla T4 · kernel log reports train 2459 val 1252

Per epoch (`metrics.json` history; lr = the first optimizer group, i.e. the lowest layer under layer-wise decay, at the end of the epoch):

| Epoch | train loss | lr | val acc | val macro-F1 | val macro-P | val macro-R |
|---|---|---|---|---|---|---|
| Epoch 1 | 1.8930 | 2.872e-06 | 0.6462 | 0.2383 | 0.2515 | 0.3049 |
| Epoch 2 | 1.7182 | 5.498e-06 | 0.7676 | 0.5828 | 0.5875 | 0.5969 |
| Epoch 3 | 1.3779 | 5.455e-06 | 0.8019 | 0.6211 | 0.5956 | 0.6751 |
| Epoch 4 | 1.2250 | 5.331e-06 | 0.8107 | 0.7035 | 0.6953 | 0.7144 |
| Epoch 5 | 1.1357 | 5.128e-06 | 0.7532 | 0.7186 | 0.7143 | 0.7466 |
| Epoch 6 | 1.0943 | 4.852e-06 | 0.7684 | 0.7315 | 0.7251 | 0.7667 |
| Epoch 7 | 1.0520 | 4.513e-06 | 0.7987 | 0.7555 | 0.7430 | 0.7953 |
| Epoch 8 | 1.0299 | 4.120e-06 | 0.8115 | 0.7729 | 0.7716 | 0.7951 |
| Epoch 9 | 0.9739 | 3.686e-06 | 0.7764 | 0.7516 | 0.7578 | 0.7703 |
| Epoch 10 | 0.9514 | 3.223e-06 | 0.7843 | 0.7618 | 0.7607 | 0.7917 |
| Epoch 11 | 0.9248 | 2.746e-06 | 0.7780 | 0.7584 | 0.7593 | 0.7857 |
| Epoch 12 | 0.8697 | 2.268e-06 | 0.7995 | 0.7648 | 0.7577 | 0.7897 |
| Epoch 13 | 0.8408 | 1.806e-06 | 0.7835 | 0.7442 | 0.7437 | 0.7626 |
| Epoch 14 | 0.8341 | 1.372e-06 | 0.8067 | 0.7611 | 0.7577 | 0.7840 |
| Epoch 15 | 0.8313 | 9.795e-07 | 0.7963 | 0.7671 | 0.7749 | 0.7835 |
| Epoch 16 | 0.7971 | 6.411e-07 | 0.7748 | 0.7538 | 0.7589 | 0.7754 |
| Epoch 17 | 0.7958 | 3.667e-07 | 0.7947 | 0.7658 | 0.7669 | 0.7877 |
| Epoch 18 | 0.7813 | 1.647e-07 | 0.7931 | 0.7586 | 0.7597 | 0.7776 |
| Epoch 19 | 0.7644 | 4.124e-08 | 0.7899 | 0.7615 | 0.7655 | 0.7808 |
| Epoch 20 | 0.7891 | 2.376e-11 | 0.7843 | 0.7557 | 0.7580 | 0.7761 |

Final epoch (Epoch 20), all 1252 validation rows: QWK (6 labels, summary.json) 0.9111 · QWK5 (AP-ROP folded onto stage 4/5, val_predictions.npz) 0.9124 · accuracy 0.7843 · macro-F1 0.7557 · balanced accuracy 0.7761 · AUC 0.9477

| final epoch | Normal | Stage 1 | Stage 2 | Stage 3 | Stage 4/5 | AP-ROP |
|---|---|---|---|---|---|---|
| recall | 0.8159 | 0.5586 | 0.6978 | 0.8731 | 0.8333 | 0.8780 |
| support | 766 | 111 | 182 | 134 | 18 | 41 |

#### Arm F · seed 1337 · fold_4 (CV fold 4)

Kernel `rop-v3-dinov2-s-uda-s1337-lr1e-4-b` · run dir `results/rop/v3/runs/dinov2_s_uda_s1337/fold_4` · as-run manifest `results/prepared/rop_staging/_v3_fold4.csv`

Fold file `results/prepared/rop_staging/fold_4.csv`: validation = 1253 photos of 333 infants (ostrava 711, ropvl 381, shenzhen 161); training = 2452 photos of 1194 infants (ostrava 607, ropvl 1292, shenzhen 553).

| rows | normal | stage_1 | stage_2 | stage_3 | stage_4_5 | arop | total |
|---|---|---|---|---|---|---|---|
| training rows | 882 | 420 | 546 | 401 | 71 | 132 | 2452 |
| validation rows | 776 | 92 | 201 | 131 | 18 | 35 | 1253 |

- images per epoch = 2452 (the training rows: the fold file's 2466 less 14 exact duplicate(s) dropped at run time)
- steps per epoch = 154 (= ceil(2452 / (16 x 1)), batch_size x grad_accum)
- plus 16 unlabelled Multi-View dev photos per step (self-ensembling) = 2464 unlabelled images per epoch
- minutes 61.7 (fold start to next fold start, kernel log) · 14.7 img/s · GPU Tesla T4 · kernel log reports train 2452 val 1253

Per epoch (`metrics.json` history; lr = the first optimizer group, i.e. the lowest layer under layer-wise decay, at the end of the epoch):

| Epoch | train loss | lr | val acc | val macro-F1 | val macro-P | val macro-R |
|---|---|---|---|---|---|---|
| Epoch 1 | 1.8840 | 2.872e-06 | 0.6680 | 0.3241 | 0.3278 | 0.3374 |
| Epoch 2 | 1.6738 | 5.498e-06 | 0.7614 | 0.5118 | 0.5102 | 0.6235 |
| Epoch 3 | 1.3876 | 5.455e-06 | 0.7885 | 0.5777 | 0.5471 | 0.6592 |
| Epoch 4 | 1.2293 | 5.331e-06 | 0.7981 | 0.6096 | 0.5861 | 0.6769 |
| Epoch 5 | 1.1401 | 5.128e-06 | 0.7965 | 0.6211 | 0.5916 | 0.7031 |
| Epoch 6 | 1.0827 | 4.852e-06 | 0.8013 | 0.6378 | 0.6097 | 0.7010 |
| Epoch 7 | 1.0469 | 4.513e-06 | 0.8053 | 0.6453 | 0.6247 | 0.7196 |
| Epoch 8 | 1.0141 | 4.120e-06 | 0.8069 | 0.6743 | 0.6636 | 0.7307 |
| Epoch 9 | 0.9575 | 3.686e-06 | 0.8125 | 0.6957 | 0.6787 | 0.7450 |
| Epoch 10 | 0.9416 | 3.223e-06 | 0.8276 | 0.7135 | 0.6881 | 0.7721 |
| Epoch 11 | 0.9483 | 2.746e-06 | 0.8292 | 0.7356 | 0.7193 | 0.7790 |
| Epoch 12 | 0.8831 | 2.268e-06 | 0.8364 | 0.7307 | 0.7130 | 0.7828 |
| Epoch 13 | 0.8541 | 1.806e-06 | 0.8468 | 0.7475 | 0.7286 | 0.7924 |
| Epoch 14 | 0.8497 | 1.372e-06 | 0.8300 | 0.7436 | 0.7278 | 0.7895 |
| Epoch 15 | 0.8146 | 9.795e-07 | 0.8356 | 0.7431 | 0.7352 | 0.7807 |
| Epoch 16 | 0.7924 | 6.411e-07 | 0.8428 | 0.7530 | 0.7435 | 0.7859 |
| Epoch 17 | 0.7766 | 3.667e-07 | 0.8316 | 0.7399 | 0.7273 | 0.7755 |
| Epoch 18 | 0.7554 | 1.647e-07 | 0.8460 | 0.7485 | 0.7338 | 0.7813 |
| Epoch 19 | 0.7623 | 4.124e-08 | 0.8372 | 0.7562 | 0.7478 | 0.7878 |
| Epoch 20 | 0.7851 | 2.376e-11 | 0.8324 | 0.7437 | 0.7299 | 0.7820 |

Final epoch (Epoch 20), all 1253 validation rows: QWK (6 labels, summary.json) 0.9005 · QWK5 (AP-ROP folded onto stage 4/5, val_predictions.npz) 0.9097 · accuracy 0.8324 · macro-F1 0.7437 · balanced accuracy 0.7820 · AUC 0.9433

| final epoch | Normal | Stage 1 | Stage 2 | Stage 3 | Stage 4/5 | AP-ROP |
|---|---|---|---|---|---|---|
| recall | 0.9201 | 0.6304 | 0.5522 | 0.8702 | 0.8333 | 0.8857 |
| support | 776 | 92 | 201 | 131 | 18 | 35 |

### Arm G: DINOv2 at 518 px

Run dirs: `results/rop/v3/runs/dinov2_s_r518_s42`, `results/rop/v3/runs/dinov2_s_r518_s1337`. Missing seeds of [42, 1337]: none.

Recipe (the as-run `config.yaml` of fold_0, seed 42; any fold whose config differs says so):

| parameter | value |
|---|---|
| model.arch | dinov2_s |
| preprocess.image_size | 518 |
| preprocess.normalize | imagenet |
| preprocess.clahe | True |
| train.lr | 0.0001 |
| train.layer_decay | 0.8 |
| train.weight_decay | 0.05 |
| train.epochs | 20 |
| train.batch_size | 16 |
| train.grad_accum | 1 (default) |
| train.label_smoothing | 0.05 |
| train.loss | ce |
| train.class_balanced | True |
| train.class_balanced_beta | 0.999 |
| train.weighted_sampler | False |
| train.strong_aug | True |
| train.site_randomization | True |
| train.amp | True |
| train.ema_decay | not set |
| train.head_warmup_epochs | not set |
| model.pretrained | True |
| model.dropout | 0.3 |
| model.drop_path | 0.1 |
| eval.early_stopping | False |
| eval.primary_metric | macro_f1 |
| train.aug_v4 | not set |
| train.bag_loss | not set |
| train.uda | not set |
| pool (summary.json) | v3 |
| config source | configs/rop_staging_dinov2_518.yaml |

<details><summary>Every key of that as-run config.yaml</summary>

| key | value |
|---|---|
| disease | rop_staging |
| seed | 42 |
| data.num_classes | 6 |
| data.class_names | [Normal, Stage 1, Stage 2, Stage 3, Stage 4/5, AP-ROP] |
| data.manifest | results/prepared/rop_staging/_v3_fold0.csv |
| data.presplit | True |
| data.group_col | patient_id |
| preprocess.image_size | 518 |
| preprocess.circle_crop | True |
| preprocess.letterbox | True |
| preprocess.ben_graham | False |
| preprocess.clahe | True |
| preprocess.normalize | imagenet |
| preprocess.cached | True |
| preprocess.cache_root | /tmp/_cache768 |
| model.arch | dinov2_s |
| model.weights_path | None |
| model.pretrained | True |
| model.head | classification |
| model.dropout | 0.3 |
| model.drop_path | 0.1 |
| train.epochs | 20 |
| train.batch_size | 16 |
| train.lr | 0.0001 |
| train.layer_decay | 0.8 |
| train.weight_decay | 0.05 |
| train.label_smoothing | 0.05 |
| train.amp | True |
| train.strong_aug | True |
| train.site_randomization | True |
| train.loss | ce |
| train.class_balanced | True |
| train.class_balanced_beta | 0.999 |
| train.weighted_sampler | False |
| train.num_workers | 4 |
| eval.early_stopping | False |
| eval.primary_metric | macro_f1 |
| eval.qwk_classes | [Normal, Stage 1, Stage 2, Stage 3, Stage 4/5] |
| eval.report_per_class_recall | True |
| eval.unvalidated_classes | [Stage 4/5, AP-ROP] |
| _source | configs/rop_staging_dinov2_518.yaml |

</details>

#### Arm G · seed 42: overview (5 folds: fold_0, fold_1, fold_2, fold_3, fold_4)

| fold | train rows | val rows | images/epoch | steps/epoch | minutes | img/s | final acc | final macro-F1 | final QWK (6) | final QWK5 | AUC |
|---|---|---|---|---|---|---|---|---|---|---|---|
| fold_0 | 2462 | 1691 | 2462 | 154 | 62.2 | 18.3 | 0.6410 | 0.5582 | 0.5750 | 0.6657 | 0.8887 |
| fold_1 | 2477 | 1252 | 2477 | 155 | 57.1 | 19.0 | 0.8227 | 0.7535 | 0.8775 | 0.8757 | 0.9476 |
| fold_2 | 2542 | 1252 | 2542 | 159 | 58.1 | 18.9 | 0.7452 | 0.6274 | 0.8621 | 0.8763 | 0.9177 |
| fold_3 | 2459 | 1252 | 2459 | 154 | 58.2 | 18.5 | 0.8107 | 0.7765 | 0.9170 | 0.9193 | 0.9633 |
| fold_4 | 2452 | 1253 | 2452 | 154 | 58.1 | 19.3 | 0.8404 | 0.7421 | 0.8724 | 0.8948 | 0.9473 |

#### Arm G · seed 42 · fold_0 (CV fold 0)

Kernel `rop-v3-dinov2-s-r518-s42-lr1e-4-a` · run dir `results/rop/v3/runs/dinov2_s_r518_s42/fold_0` · as-run manifest `results/prepared/rop_staging/_v3_fold0.csv`

Fold file `results/prepared/rop_staging/fold_0.csv`: validation = 1691 photos of 297 infants (ostrava 1219, ropvl 310, shenzhen 162); training = 2462 photos of 1231 infants (ostrava 544, ropvl 1363, shenzhen 555).

| rows | normal | stage_1 | stage_2 | stage_3 | stage_4_5 | arop | total |
|---|---|---|---|---|---|---|---|
| training rows | 888 | 403 | 556 | 404 | 70 | 141 | 2462 |
| validation rows | 767 | 137 | 163 | 129 | 19 | 476 | 1691 |

- images per epoch = 2462 (the training rows: the fold file's 2474 less 12 exact duplicate(s) dropped at run time)
- steps per epoch = 154 (= ceil(2462 / (16 x 1)), batch_size x grad_accum)
- minutes 62.2 (fold start to next fold start, kernel log) · 18.3 img/s · GPU Tesla T4 · kernel log reports train 2462 val 1691

Per epoch (`metrics.json` history; lr = the first optimizer group, i.e. the lowest layer under layer-wise decay, at the end of the epoch):

| Epoch | train loss | lr | val acc | val macro-F1 | val macro-P | val macro-R |
|---|---|---|---|---|---|---|
| Epoch 1 | 1.8725 | 2.872e-06 | 0.5393 | 0.3634 | 0.5002 | 0.3461 |
| Epoch 2 | 1.6777 | 5.498e-06 | 0.6724 | 0.4467 | 0.4439 | 0.5668 |
| Epoch 3 | 1.3665 | 5.455e-06 | 0.6487 | 0.4985 | 0.5660 | 0.5783 |
| Epoch 4 | 1.2408 | 5.331e-06 | 0.6682 | 0.5872 | 0.6304 | 0.6620 |
| Epoch 5 | 1.1338 | 5.128e-06 | 0.6552 | 0.5640 | 0.6037 | 0.6400 |
| Epoch 6 | 1.0720 | 4.852e-06 | 0.5931 | 0.5314 | 0.6341 | 0.6150 |
| Epoch 7 | 1.0215 | 4.513e-06 | 0.6558 | 0.5627 | 0.5946 | 0.6333 |
| Epoch 8 | 1.0027 | 4.120e-06 | 0.5719 | 0.5488 | 0.5891 | 0.6068 |
| Epoch 9 | 0.9557 | 3.686e-06 | 0.6653 | 0.5792 | 0.6337 | 0.6360 |
| Epoch 10 | 0.9223 | 3.223e-06 | 0.6712 | 0.6099 | 0.6567 | 0.6851 |
| Epoch 11 | 0.8865 | 2.746e-06 | 0.6174 | 0.5387 | 0.6188 | 0.6424 |
| Epoch 12 | 0.8751 | 2.268e-06 | 0.6286 | 0.5615 | 0.6337 | 0.6494 |
| Epoch 13 | 0.8610 | 1.806e-06 | 0.6736 | 0.5863 | 0.6272 | 0.6538 |
| Epoch 14 | 0.8430 | 1.372e-06 | 0.6570 | 0.5517 | 0.5999 | 0.6438 |
| Epoch 15 | 0.8127 | 9.795e-07 | 0.6410 | 0.5701 | 0.6256 | 0.6655 |
| Epoch 16 | 0.7881 | 6.411e-07 | 0.6268 | 0.5726 | 0.6242 | 0.6738 |
| Epoch 17 | 0.7538 | 3.667e-07 | 0.6339 | 0.5550 | 0.6125 | 0.6653 |
| Epoch 18 | 0.7479 | 1.647e-07 | 0.6404 | 0.5564 | 0.6113 | 0.6671 |
| Epoch 19 | 0.7774 | 4.124e-08 | 0.6404 | 0.5576 | 0.6124 | 0.6672 |
| Epoch 20 | 0.7662 | 2.376e-11 | 0.6410 | 0.5582 | 0.6123 | 0.6669 |

Final epoch (Epoch 20), all 1691 validation rows: QWK (6 labels, summary.json) 0.5750 · QWK5 (AP-ROP folded onto stage 4/5, val_predictions.npz) 0.6657 · accuracy 0.6410 · macro-F1 0.5582 · balanced accuracy 0.6669 · AUC 0.8887

| final epoch | Normal | Stage 1 | Stage 2 | Stage 3 | Stage 4/5 | AP-ROP |
|---|---|---|---|---|---|---|
| recall | 0.9100 | 0.7518 | 0.4785 | 0.8527 | 0.8421 | 0.1660 |
| support | 767 | 137 | 163 | 129 | 19 | 476 |

#### Arm G · seed 42 · fold_1 (CV fold 1)

Kernel `rop-v3-dinov2-s-r518-s42-lr1e-4-a` · run dir `results/rop/v3/runs/dinov2_s_r518_s42/fold_1` · as-run manifest `results/prepared/rop_staging/_v3_fold1.csv`

Fold file `results/prepared/rop_staging/fold_1.csv`: validation = 1252 photos of 299 infants (ostrava 812, ropvl 327, shenzhen 113); training = 2477 photos of 1228 infants (ostrava 527, ropvl 1346, shenzhen 604).

| rows | normal | stage_1 | stage_2 | stage_3 | stage_4_5 | arop | total |
|---|---|---|---|---|---|---|---|
| training rows | 890 | 401 | 534 | 442 | 73 | 137 | 2477 |
| validation rows | 756 | 114 | 220 | 116 | 16 | 30 | 1252 |

- images per epoch = 2477 (the training rows: the fold file's 2486 less 9 exact duplicate(s) dropped at run time)
- steps per epoch = 155 (= ceil(2477 / (16 x 1)), batch_size x grad_accum)
- minutes 57.1 (fold start to next fold start, kernel log) · 19.0 img/s · GPU Tesla T4 · kernel log reports train 2477 val 1252

Per epoch (`metrics.json` history; lr = the first optimizer group, i.e. the lowest layer under layer-wise decay, at the end of the epoch):

| Epoch | train loss | lr | val acc | val macro-F1 | val macro-P | val macro-R |
|---|---|---|---|---|---|---|
| Epoch 1 | 1.8872 | 2.872e-06 | 0.5855 | 0.1635 | 0.3131 | 0.3080 |
| Epoch 2 | 1.6958 | 5.498e-06 | 0.7580 | 0.6004 | 0.6127 | 0.6211 |
| Epoch 3 | 1.3689 | 5.455e-06 | 0.7021 | 0.5754 | 0.5939 | 0.6734 |
| Epoch 4 | 1.2420 | 5.331e-06 | 0.7604 | 0.6440 | 0.6240 | 0.6820 |
| Epoch 5 | 1.1241 | 5.128e-06 | 0.7668 | 0.6760 | 0.6850 | 0.7442 |
| Epoch 6 | 1.0534 | 4.852e-06 | 0.8019 | 0.6764 | 0.7101 | 0.6868 |
| Epoch 7 | 1.0410 | 4.513e-06 | 0.7764 | 0.6354 | 0.6906 | 0.6914 |
| Epoch 8 | 0.9645 | 4.120e-06 | 0.8027 | 0.6903 | 0.6999 | 0.7146 |
| Epoch 9 | 0.9539 | 3.686e-06 | 0.8099 | 0.7342 | 0.7661 | 0.7198 |
| Epoch 10 | 0.9167 | 3.223e-06 | 0.8123 | 0.7128 | 0.7170 | 0.7288 |
| Epoch 11 | 0.9063 | 2.746e-06 | 0.8075 | 0.7297 | 0.7239 | 0.7464 |
| Epoch 12 | 0.8625 | 2.268e-06 | 0.7899 | 0.7273 | 0.7390 | 0.7566 |
| Epoch 13 | 0.8314 | 1.806e-06 | 0.7684 | 0.7120 | 0.7197 | 0.7266 |
| Epoch 14 | 0.8101 | 1.372e-06 | 0.8211 | 0.7319 | 0.7309 | 0.7467 |
| Epoch 15 | 0.8184 | 9.795e-07 | 0.8091 | 0.7463 | 0.7757 | 0.7483 |
| Epoch 16 | 0.7846 | 6.411e-07 | 0.8243 | 0.7524 | 0.7672 | 0.7571 |
| Epoch 17 | 0.7556 | 3.667e-07 | 0.8067 | 0.7337 | 0.7511 | 0.7448 |
| Epoch 18 | 0.7375 | 1.647e-07 | 0.8251 | 0.7596 | 0.7797 | 0.7626 |
| Epoch 19 | 0.7604 | 4.125e-08 | 0.8227 | 0.7535 | 0.7656 | 0.7612 |
| Epoch 20 | 0.7376 | 2.373e-11 | 0.8227 | 0.7535 | 0.7659 | 0.7612 |

Final epoch (Epoch 20), all 1252 validation rows: QWK (6 labels, summary.json) 0.8775 · QWK5 (AP-ROP folded onto stage 4/5, val_predictions.npz) 0.8757 · accuracy 0.8227 · macro-F1 0.7535 · balanced accuracy 0.7612 · AUC 0.9476

| final epoch | Normal | Stage 1 | Stage 2 | Stage 3 | Stage 4/5 | AP-ROP |
|---|---|---|---|---|---|---|
| recall | 0.9709 | 0.5263 | 0.4955 | 0.7414 | 1.0000 | 0.8333 |
| support | 756 | 114 | 220 | 116 | 16 | 30 |

#### Arm G · seed 42 · fold_2 (CV fold 2)

Kernel `rop-v3-dinov2-s-r518-s42-lr1e-4-a` · run dir `results/rop/v3/runs/dinov2_s_r518_s42/fold_2` · as-run manifest `results/prepared/rop_staging/_v3_fold2.csv`

Fold file `results/prepared/rop_staging/fold_2.csv`: validation = 1252 photos of 272 infants (ostrava 830, ropvl 295, shenzhen 127); training = 2542 photos of 1255 infants (ostrava 573, ropvl 1378, shenzhen 591).

| rows | normal | stage_1 | stage_2 | stage_3 | stage_4_5 | arop | total |
|---|---|---|---|---|---|---|---|
| training rows | 888 | 415 | 556 | 481 | 71 | 131 | 2542 |
| validation rows | 772 | 123 | 193 | 110 | 18 | 36 | 1252 |

- images per epoch = 2542 (the training rows: the fold file's 2553 less 11 exact duplicate(s) dropped at run time)
- steps per epoch = 159 (= ceil(2542 / (16 x 1)), batch_size x grad_accum)
- minutes 58.1 (fold start to next fold start, kernel log) · 18.9 img/s · GPU Tesla T4 · kernel log reports train 2542 val 1252

Per epoch (`metrics.json` history; lr = the first optimizer group, i.e. the lowest layer under layer-wise decay, at the end of the epoch):

| Epoch | train loss | lr | val acc | val macro-F1 | val macro-P | val macro-R |
|---|---|---|---|---|---|---|
| Epoch 1 | 1.8773 | 2.872e-06 | 0.6462 | 0.2849 | 0.4322 | 0.2688 |
| Epoch 2 | 1.6641 | 5.498e-06 | 0.7276 | 0.4101 | 0.4948 | 0.4888 |
| Epoch 3 | 1.4221 | 5.455e-06 | 0.6222 | 0.4458 | 0.4950 | 0.5022 |
| Epoch 4 | 1.2241 | 5.331e-06 | 0.5192 | 0.4454 | 0.5184 | 0.5798 |
| Epoch 5 | 1.1708 | 5.128e-06 | 0.4696 | 0.4329 | 0.5568 | 0.5106 |
| Epoch 6 | 1.0917 | 4.853e-06 | 0.6198 | 0.5317 | 0.5883 | 0.6019 |
| Epoch 7 | 1.0951 | 4.513e-06 | 0.6286 | 0.5213 | 0.5627 | 0.6008 |
| Epoch 8 | 0.9916 | 4.121e-06 | 0.7181 | 0.5844 | 0.6570 | 0.6580 |
| Epoch 9 | 0.9543 | 3.686e-06 | 0.7827 | 0.5688 | 0.5964 | 0.6448 |
| Epoch 10 | 0.9534 | 3.223e-06 | 0.6741 | 0.5916 | 0.6035 | 0.6568 |
| Epoch 11 | 0.9020 | 2.746e-06 | 0.7628 | 0.5931 | 0.6151 | 0.6518 |
| Epoch 12 | 0.8653 | 2.269e-06 | 0.7308 | 0.5964 | 0.6184 | 0.6532 |
| Epoch 13 | 0.8600 | 1.806e-06 | 0.7604 | 0.6264 | 0.6209 | 0.6848 |
| Epoch 14 | 0.8479 | 1.372e-06 | 0.7452 | 0.6399 | 0.6692 | 0.6590 |
| Epoch 15 | 0.8184 | 9.796e-07 | 0.7085 | 0.6147 | 0.6523 | 0.6483 |
| Epoch 16 | 0.7791 | 6.412e-07 | 0.7412 | 0.6153 | 0.6196 | 0.6709 |
| Epoch 17 | 0.7868 | 3.668e-07 | 0.7404 | 0.6365 | 0.6530 | 0.6729 |
| Epoch 18 | 0.7756 | 1.648e-07 | 0.7548 | 0.6354 | 0.6391 | 0.6782 |
| Epoch 19 | 0.7972 | 4.126e-08 | 0.7444 | 0.6320 | 0.6423 | 0.6717 |
| Epoch 20 | 0.7608 | 2.365e-11 | 0.7452 | 0.6274 | 0.6371 | 0.6700 |

Final epoch (Epoch 20), all 1252 validation rows: QWK (6 labels, summary.json) 0.8621 · QWK5 (AP-ROP folded onto stage 4/5, val_predictions.npz) 0.8763 · accuracy 0.7452 · macro-F1 0.6274 · balanced accuracy 0.6700 · AUC 0.9177

| final epoch | Normal | Stage 1 | Stage 2 | Stage 3 | Stage 4/5 | AP-ROP |
|---|---|---|---|---|---|---|
| recall | 0.8782 | 0.5203 | 0.5803 | 0.2909 | 0.8889 | 0.8611 |
| support | 772 | 123 | 193 | 110 | 18 | 36 |

#### Arm G · seed 42 · fold_3 (CV fold 3)

Kernel `rop-v3-dinov2-s-r518-s42-lr1e-4-b` · run dir `results/rop/v3/runs/dinov2_s_r518_s42/fold_3` · as-run manifest `results/prepared/rop_staging/_v3_fold3.csv`

Fold file `results/prepared/rop_staging/fold_3.csv`: validation = 1252 photos of 327 infants (ostrava 713, ropvl 362, shenzhen 177); training = 2459 photos of 1200 infants (ostrava 609, ropvl 1313, shenzhen 537).

| rows | normal | stage_1 | stage_2 | stage_3 | stage_4_5 | arop | total |
|---|---|---|---|---|---|---|---|
| training rows | 888 | 409 | 564 | 400 | 71 | 127 | 2459 |
| validation rows | 766 | 111 | 182 | 134 | 18 | 41 | 1252 |

- images per epoch = 2459 (the training rows: the fold file's 2469 less 10 exact duplicate(s) dropped at run time)
- steps per epoch = 154 (= ceil(2459 / (16 x 1)), batch_size x grad_accum)
- minutes 58.2 (fold start to next fold start, kernel log) · 18.5 img/s · GPU Tesla T4 · kernel log reports train 2459 val 1252

Per epoch (`metrics.json` history; lr = the first optimizer group, i.e. the lowest layer under layer-wise decay, at the end of the epoch):

| Epoch | train loss | lr | val acc | val macro-F1 | val macro-P | val macro-R |
|---|---|---|---|---|---|---|
| Epoch 1 | 1.8807 | 2.872e-06 | 0.4920 | 0.3169 | 0.4600 | 0.3593 |
| Epoch 2 | 1.7002 | 5.498e-06 | 0.6725 | 0.5565 | 0.5166 | 0.6623 |
| Epoch 3 | 1.4021 | 5.455e-06 | 0.7692 | 0.5913 | 0.6525 | 0.5788 |
| Epoch 4 | 1.2740 | 5.331e-06 | 0.6222 | 0.5980 | 0.5871 | 0.6426 |
| Epoch 5 | 1.1817 | 5.128e-06 | 0.7556 | 0.6617 | 0.6717 | 0.7258 |
| Epoch 6 | 1.1260 | 4.852e-06 | 0.8283 | 0.6871 | 0.7286 | 0.6998 |
| Epoch 7 | 1.0701 | 4.513e-06 | 0.6390 | 0.6539 | 0.6889 | 0.7404 |
| Epoch 8 | 1.0269 | 4.120e-06 | 0.7516 | 0.6939 | 0.6936 | 0.7321 |
| Epoch 9 | 0.9995 | 3.686e-06 | 0.7883 | 0.7433 | 0.7845 | 0.7428 |
| Epoch 10 | 0.9484 | 3.223e-06 | 0.7724 | 0.7220 | 0.7558 | 0.7302 |
| Epoch 11 | 0.9130 | 2.746e-06 | 0.6382 | 0.6902 | 0.7323 | 0.7301 |
| Epoch 12 | 0.8984 | 2.268e-06 | 0.7316 | 0.7272 | 0.7639 | 0.7481 |
| Epoch 13 | 0.8630 | 1.806e-06 | 0.7276 | 0.7321 | 0.7439 | 0.7676 |
| Epoch 14 | 0.8310 | 1.372e-06 | 0.7404 | 0.7392 | 0.7480 | 0.7872 |
| Epoch 15 | 0.8007 | 9.795e-07 | 0.7851 | 0.7720 | 0.7873 | 0.7860 |
| Epoch 16 | 0.8096 | 6.411e-07 | 0.7708 | 0.7636 | 0.7767 | 0.7800 |
| Epoch 17 | 0.7673 | 3.667e-07 | 0.7724 | 0.7572 | 0.7560 | 0.7911 |
| Epoch 18 | 0.7638 | 1.647e-07 | 0.8139 | 0.7737 | 0.7761 | 0.7893 |
| Epoch 19 | 0.7598 | 4.124e-08 | 0.8147 | 0.7779 | 0.7837 | 0.7925 |
| Epoch 20 | 0.7583 | 2.376e-11 | 0.8107 | 0.7765 | 0.7821 | 0.7915 |

Final epoch (Epoch 20), all 1252 validation rows: QWK (6 labels, summary.json) 0.9170 · QWK5 (AP-ROP folded onto stage 4/5, val_predictions.npz) 0.9193 · accuracy 0.8107 · macro-F1 0.7765 · balanced accuracy 0.7915 · AUC 0.9633

| final epoch | Normal | Stage 1 | Stage 2 | Stage 3 | Stage 4/5 | AP-ROP |
|---|---|---|---|---|---|---|
| recall | 0.8577 | 0.6577 | 0.6648 | 0.8507 | 0.8889 | 0.8293 |
| support | 766 | 111 | 182 | 134 | 18 | 41 |

#### Arm G · seed 42 · fold_4 (CV fold 4)

Kernel `rop-v3-dinov2-s-r518-s42-lr1e-4-b` · run dir `results/rop/v3/runs/dinov2_s_r518_s42/fold_4` · as-run manifest `results/prepared/rop_staging/_v3_fold4.csv`

Fold file `results/prepared/rop_staging/fold_4.csv`: validation = 1253 photos of 333 infants (ostrava 711, ropvl 381, shenzhen 161); training = 2452 photos of 1194 infants (ostrava 607, ropvl 1292, shenzhen 553).

| rows | normal | stage_1 | stage_2 | stage_3 | stage_4_5 | arop | total |
|---|---|---|---|---|---|---|---|
| training rows | 882 | 420 | 546 | 401 | 71 | 132 | 2452 |
| validation rows | 776 | 92 | 201 | 131 | 18 | 35 | 1253 |

- images per epoch = 2452 (the training rows: the fold file's 2466 less 14 exact duplicate(s) dropped at run time)
- steps per epoch = 154 (= ceil(2452 / (16 x 1)), batch_size x grad_accum)
- minutes 58.1 (fold start to next fold start, kernel log) · 19.3 img/s · GPU Tesla T4 · kernel log reports train 2452 val 1253

Per epoch (`metrics.json` history; lr = the first optimizer group, i.e. the lowest layer under layer-wise decay, at the end of the epoch):

| Epoch | train loss | lr | val acc | val macro-F1 | val macro-P | val macro-R |
|---|---|---|---|---|---|---|
| Epoch 1 | 1.8871 | 2.872e-06 | 0.6640 | 0.4254 | 0.5300 | 0.4090 |
| Epoch 2 | 1.7125 | 5.498e-06 | 0.7103 | 0.4264 | 0.5000 | 0.5710 |
| Epoch 3 | 1.3869 | 5.455e-06 | 0.7654 | 0.5166 | 0.5333 | 0.5602 |
| Epoch 4 | 1.2533 | 5.331e-06 | 0.7207 | 0.5880 | 0.5629 | 0.6978 |
| Epoch 5 | 1.1405 | 5.128e-06 | 0.6864 | 0.5760 | 0.5590 | 0.6726 |
| Epoch 6 | 1.0842 | 4.852e-06 | 0.8220 | 0.6274 | 0.6090 | 0.7135 |
| Epoch 7 | 1.0186 | 4.513e-06 | 0.6959 | 0.6185 | 0.6146 | 0.6533 |
| Epoch 8 | 1.0251 | 4.120e-06 | 0.6026 | 0.5688 | 0.6280 | 0.6625 |
| Epoch 9 | 0.9612 | 3.686e-06 | 0.7302 | 0.6228 | 0.6054 | 0.6977 |
| Epoch 10 | 0.9335 | 3.223e-06 | 0.8069 | 0.6972 | 0.6691 | 0.7498 |
| Epoch 11 | 0.8957 | 2.746e-06 | 0.8380 | 0.6689 | 0.6565 | 0.7422 |
| Epoch 12 | 0.8653 | 2.268e-06 | 0.8388 | 0.7298 | 0.7177 | 0.7817 |
| Epoch 13 | 0.8285 | 1.806e-06 | 0.7909 | 0.7050 | 0.6992 | 0.7577 |
| Epoch 14 | 0.8482 | 1.372e-06 | 0.8340 | 0.7303 | 0.7151 | 0.7846 |
| Epoch 15 | 0.7951 | 9.795e-07 | 0.7574 | 0.7086 | 0.6982 | 0.7798 |
| Epoch 16 | 0.8026 | 6.411e-07 | 0.8723 | 0.7591 | 0.7489 | 0.7863 |
| Epoch 17 | 0.7800 | 3.667e-07 | 0.8037 | 0.7158 | 0.6962 | 0.7700 |
| Epoch 18 | 0.7592 | 1.647e-07 | 0.8396 | 0.7407 | 0.7225 | 0.7913 |
| Epoch 19 | 0.7736 | 4.124e-08 | 0.8428 | 0.7428 | 0.7245 | 0.7921 |
| Epoch 20 | 0.7344 | 2.376e-11 | 0.8404 | 0.7421 | 0.7236 | 0.7915 |

Final epoch (Epoch 20), all 1253 validation rows: QWK (6 labels, summary.json) 0.8724 · QWK5 (AP-ROP folded onto stage 4/5, val_predictions.npz) 0.8948 · accuracy 0.8404 · macro-F1 0.7421 · balanced accuracy 0.7915 · AUC 0.9473

| final epoch | Normal | Stage 1 | Stage 2 | Stage 3 | Stage 4/5 | AP-ROP |
|---|---|---|---|---|---|---|
| recall | 0.9201 | 0.6630 | 0.6070 | 0.8397 | 0.8333 | 0.8857 |
| support | 776 | 92 | 201 | 131 | 18 | 35 |

#### Arm G · seed 1337: overview (5 folds: fold_0, fold_1, fold_2, fold_3, fold_4)

| fold | train rows | val rows | images/epoch | steps/epoch | minutes | img/s | final acc | final macro-F1 | final QWK (6) | final QWK5 | AUC |
|---|---|---|---|---|---|---|---|---|---|---|---|
| fold_0 | 2462 | 1691 | 2462 | 154 | 60.3 | 19.4 | 0.6363 | 0.5467 | 0.5454 | 0.6458 | 0.8867 |
| fold_1 | 2477 | 1252 | 2477 | 155 | 54.9 | 20.1 | 0.8083 | 0.7244 | 0.8816 | 0.8824 | 0.9480 |
| fold_2 | 2542 | 1252 | 2542 | 159 | 56.1 | 19.7 | 0.7500 | 0.5943 | 0.8505 | 0.8753 | 0.8985 |
| fold_3 | 2459 | 1252 | 2459 | 154 | 58.2 | 19.6 | 0.8211 | 0.7721 | 0.9113 | 0.9165 | 0.9644 |
| fold_4 | 2452 | 1253 | 2452 | 154 | 57.8 | 19.5 | 0.8484 | 0.7597 | 0.9023 | 0.9152 | 0.9426 |

#### Arm G · seed 1337 · fold_0 (CV fold 0)

Kernel `rop-v3-dinov2-s-r518-s1337-lr1e-4-a` · run dir `results/rop/v3/runs/dinov2_s_r518_s1337/fold_0` · as-run manifest `results/prepared/rop_staging/_v3_fold0.csv`

Fold file `results/prepared/rop_staging/fold_0.csv`: validation = 1691 photos of 297 infants (ostrava 1219, ropvl 310, shenzhen 162); training = 2462 photos of 1231 infants (ostrava 544, ropvl 1363, shenzhen 555).

| rows | normal | stage_1 | stage_2 | stage_3 | stage_4_5 | arop | total |
|---|---|---|---|---|---|---|---|
| training rows | 888 | 403 | 556 | 404 | 70 | 141 | 2462 |
| validation rows | 767 | 137 | 163 | 129 | 19 | 476 | 1691 |

- images per epoch = 2462 (the training rows: the fold file's 2474 less 12 exact duplicate(s) dropped at run time)
- steps per epoch = 154 (= ceil(2462 / (16 x 1)), batch_size x grad_accum)
- minutes 60.3 (fold start to next fold start, kernel log) · 19.4 img/s · GPU Tesla T4 · kernel log reports train 2462 val 1691

Per epoch (`metrics.json` history; lr = the first optimizer group, i.e. the lowest layer under layer-wise decay, at the end of the epoch):

| Epoch | train loss | lr | val acc | val macro-F1 | val macro-P | val macro-R |
|---|---|---|---|---|---|---|
| Epoch 1 | 1.9017 | 2.872e-06 | 0.4808 | 0.1965 | 0.1531 | 0.3317 |
| Epoch 2 | 1.6992 | 5.498e-06 | 0.6050 | 0.4278 | 0.6262 | 0.5170 |
| Epoch 3 | 1.3819 | 5.455e-06 | 0.6304 | 0.5682 | 0.6533 | 0.6341 |
| Epoch 4 | 1.2654 | 5.331e-06 | 0.6594 | 0.4996 | 0.5981 | 0.5996 |
| Epoch 5 | 1.1391 | 5.128e-06 | 0.7014 | 0.6180 | 0.6795 | 0.6645 |
| Epoch 6 | 1.0791 | 4.852e-06 | 0.6541 | 0.5357 | 0.5770 | 0.6664 |
| Epoch 7 | 1.0285 | 4.513e-06 | 0.6736 | 0.5791 | 0.5994 | 0.6947 |
| Epoch 8 | 1.0225 | 4.120e-06 | 0.6174 | 0.5231 | 0.6324 | 0.5627 |
| Epoch 9 | 0.9696 | 3.686e-06 | 0.6481 | 0.5296 | 0.5710 | 0.6713 |
| Epoch 10 | 0.9613 | 3.223e-06 | 0.6174 | 0.5047 | 0.5657 | 0.5922 |
| Epoch 11 | 0.9072 | 2.746e-06 | 0.6428 | 0.5365 | 0.5822 | 0.6612 |
| Epoch 12 | 0.9091 | 2.268e-06 | 0.6109 | 0.5078 | 0.5724 | 0.6558 |
| Epoch 13 | 0.8578 | 1.806e-06 | 0.6257 | 0.5366 | 0.5932 | 0.6746 |
| Epoch 14 | 0.8326 | 1.372e-06 | 0.6511 | 0.5605 | 0.6005 | 0.6691 |
| Epoch 15 | 0.8157 | 9.795e-07 | 0.6097 | 0.5249 | 0.5696 | 0.6500 |
| Epoch 16 | 0.7873 | 6.411e-07 | 0.6162 | 0.5335 | 0.5835 | 0.6659 |
| Epoch 17 | 0.7761 | 3.667e-07 | 0.6286 | 0.5485 | 0.5977 | 0.6633 |
| Epoch 18 | 0.7536 | 1.647e-07 | 0.6345 | 0.5433 | 0.5928 | 0.6643 |
| Epoch 19 | 0.7691 | 4.124e-08 | 0.6381 | 0.5474 | 0.5951 | 0.6655 |
| Epoch 20 | 0.7759 | 2.376e-11 | 0.6363 | 0.5467 | 0.5951 | 0.6665 |

Final epoch (Epoch 20), all 1691 validation rows: QWK (6 labels, summary.json) 0.5454 · QWK5 (AP-ROP folded onto stage 4/5, val_predictions.npz) 0.6458 · accuracy 0.6363 · macro-F1 0.5467 · balanced accuracy 0.6665 · AUC 0.8867

| final epoch | Normal | Stage 1 | Stage 2 | Stage 3 | Stage 4/5 | AP-ROP |
|---|---|---|---|---|---|---|
| recall | 0.9374 | 0.6788 | 0.5031 | 0.8760 | 0.8947 | 0.1092 |
| support | 767 | 137 | 163 | 129 | 19 | 476 |

#### Arm G · seed 1337 · fold_1 (CV fold 1)

Kernel `rop-v3-dinov2-s-r518-s1337-lr1e-4-a` · run dir `results/rop/v3/runs/dinov2_s_r518_s1337/fold_1` · as-run manifest `results/prepared/rop_staging/_v3_fold1.csv`

Fold file `results/prepared/rop_staging/fold_1.csv`: validation = 1252 photos of 299 infants (ostrava 812, ropvl 327, shenzhen 113); training = 2477 photos of 1228 infants (ostrava 527, ropvl 1346, shenzhen 604).

| rows | normal | stage_1 | stage_2 | stage_3 | stage_4_5 | arop | total |
|---|---|---|---|---|---|---|---|
| training rows | 890 | 401 | 534 | 442 | 73 | 137 | 2477 |
| validation rows | 756 | 114 | 220 | 116 | 16 | 30 | 1252 |

- images per epoch = 2477 (the training rows: the fold file's 2486 less 9 exact duplicate(s) dropped at run time)
- steps per epoch = 155 (= ceil(2477 / (16 x 1)), batch_size x grad_accum)
- minutes 54.9 (fold start to next fold start, kernel log) · 20.1 img/s · GPU Tesla T4 · kernel log reports train 2477 val 1252

Per epoch (`metrics.json` history; lr = the first optimizer group, i.e. the lowest layer under layer-wise decay, at the end of the epoch):

| Epoch | train loss | lr | val acc | val macro-F1 | val macro-P | val macro-R |
|---|---|---|---|---|---|---|
| Epoch 1 | 1.9088 | 2.872e-06 | 0.6701 | 0.2510 | 0.2569 | 0.2687 |
| Epoch 2 | 1.6752 | 5.498e-06 | 0.6765 | 0.5711 | 0.6410 | 0.5632 |
| Epoch 3 | 1.3757 | 5.455e-06 | 0.7093 | 0.5715 | 0.6348 | 0.6206 |
| Epoch 4 | 1.2163 | 5.331e-06 | 0.7468 | 0.5539 | 0.6078 | 0.6503 |
| Epoch 5 | 1.1452 | 5.128e-06 | 0.7899 | 0.6398 | 0.7085 | 0.6340 |
| Epoch 6 | 1.0910 | 4.852e-06 | 0.7851 | 0.6824 | 0.6981 | 0.7053 |
| Epoch 7 | 1.0044 | 4.513e-06 | 0.6813 | 0.6454 | 0.6645 | 0.6964 |
| Epoch 8 | 0.9982 | 4.120e-06 | 0.7772 | 0.6940 | 0.7195 | 0.7142 |
| Epoch 9 | 0.9467 | 3.686e-06 | 0.8035 | 0.7216 | 0.7385 | 0.7304 |
| Epoch 10 | 0.9278 | 3.223e-06 | 0.8075 | 0.7260 | 0.7750 | 0.7006 |
| Epoch 11 | 0.9112 | 2.746e-06 | 0.8003 | 0.6969 | 0.7209 | 0.6839 |
| Epoch 12 | 0.8525 | 2.268e-06 | 0.7923 | 0.7084 | 0.7630 | 0.6859 |
| Epoch 13 | 0.8611 | 1.806e-06 | 0.8155 | 0.7259 | 0.7534 | 0.7092 |
| Epoch 14 | 0.7917 | 1.372e-06 | 0.8075 | 0.7191 | 0.7237 | 0.7317 |
| Epoch 15 | 0.7951 | 9.795e-07 | 0.8003 | 0.7062 | 0.7388 | 0.6924 |
| Epoch 16 | 0.7898 | 6.411e-07 | 0.8123 | 0.7266 | 0.7517 | 0.7154 |
| Epoch 17 | 0.7630 | 3.667e-07 | 0.8099 | 0.7216 | 0.7402 | 0.7132 |
| Epoch 18 | 0.7438 | 1.647e-07 | 0.8091 | 0.7211 | 0.7491 | 0.7097 |
| Epoch 19 | 0.7607 | 4.125e-08 | 0.8083 | 0.7242 | 0.7490 | 0.7189 |
| Epoch 20 | 0.7627 | 2.373e-11 | 0.8083 | 0.7244 | 0.7486 | 0.7195 |

Final epoch (Epoch 20), all 1252 validation rows: QWK (6 labels, summary.json) 0.8816 · QWK5 (AP-ROP folded onto stage 4/5, val_predictions.npz) 0.8824 · accuracy 0.8083 · macro-F1 0.7244 · balanced accuracy 0.7195 · AUC 0.9480

| final epoch | Normal | Stage 1 | Stage 2 | Stage 3 | Stage 4/5 | AP-ROP |
|---|---|---|---|---|---|---|
| recall | 0.9643 | 0.5351 | 0.5000 | 0.6466 | 0.9375 | 0.7333 |
| support | 756 | 114 | 220 | 116 | 16 | 30 |

#### Arm G · seed 1337 · fold_2 (CV fold 2)

Kernel `rop-v3-dinov2-s-r518-s1337-lr1e-4-a` · run dir `results/rop/v3/runs/dinov2_s_r518_s1337/fold_2` · as-run manifest `results/prepared/rop_staging/_v3_fold2.csv`

Fold file `results/prepared/rop_staging/fold_2.csv`: validation = 1252 photos of 272 infants (ostrava 830, ropvl 295, shenzhen 127); training = 2542 photos of 1255 infants (ostrava 573, ropvl 1378, shenzhen 591).

| rows | normal | stage_1 | stage_2 | stage_3 | stage_4_5 | arop | total |
|---|---|---|---|---|---|---|---|
| training rows | 888 | 415 | 556 | 481 | 71 | 131 | 2542 |
| validation rows | 772 | 123 | 193 | 110 | 18 | 36 | 1252 |

- images per epoch = 2542 (the training rows: the fold file's 2553 less 11 exact duplicate(s) dropped at run time)
- steps per epoch = 159 (= ceil(2542 / (16 x 1)), batch_size x grad_accum)
- minutes 56.1 (fold start to next fold start, kernel log) · 19.7 img/s · GPU Tesla T4 · kernel log reports train 2542 val 1252

Per epoch (`metrics.json` history; lr = the first optimizer group, i.e. the lowest layer under layer-wise decay, at the end of the epoch):

| Epoch | train loss | lr | val acc | val macro-F1 | val macro-P | val macro-R |
|---|---|---|---|---|---|---|
| Epoch 1 | 1.9043 | 2.872e-06 | 0.6446 | 0.1820 | 0.2398 | 0.1995 |
| Epoch 2 | 1.6531 | 5.498e-06 | 0.6669 | 0.4339 | 0.4772 | 0.5326 |
| Epoch 3 | 1.3697 | 5.455e-06 | 0.7308 | 0.4355 | 0.4799 | 0.5224 |
| Epoch 4 | 1.2536 | 5.331e-06 | 0.7620 | 0.5219 | 0.5235 | 0.5662 |
| Epoch 5 | 1.1620 | 5.128e-06 | 0.7764 | 0.5642 | 0.6005 | 0.5972 |
| Epoch 6 | 1.0891 | 4.853e-06 | 0.7236 | 0.4920 | 0.6024 | 0.6154 |
| Epoch 7 | 1.0347 | 4.513e-06 | 0.7244 | 0.4980 | 0.4893 | 0.5999 |
| Epoch 8 | 1.0282 | 4.121e-06 | 0.7292 | 0.5454 | 0.5560 | 0.6586 |
| Epoch 9 | 0.9426 | 3.686e-06 | 0.7053 | 0.5209 | 0.5332 | 0.6141 |
| Epoch 10 | 0.9342 | 3.223e-06 | 0.6893 | 0.5561 | 0.5591 | 0.6251 |
| Epoch 11 | 0.9086 | 2.746e-06 | 0.7308 | 0.5634 | 0.5633 | 0.6596 |
| Epoch 12 | 0.8965 | 2.269e-06 | 0.6749 | 0.5290 | 0.5324 | 0.6196 |
| Epoch 13 | 0.8487 | 1.806e-06 | 0.7620 | 0.6006 | 0.5999 | 0.6654 |
| Epoch 14 | 0.8295 | 1.372e-06 | 0.7708 | 0.5963 | 0.5823 | 0.6692 |
| Epoch 15 | 0.8064 | 9.796e-07 | 0.7532 | 0.5867 | 0.5756 | 0.6604 |
| Epoch 16 | 0.7992 | 6.412e-07 | 0.7724 | 0.5938 | 0.5872 | 0.6622 |
| Epoch 17 | 0.7820 | 3.668e-07 | 0.7452 | 0.5885 | 0.5887 | 0.6686 |
| Epoch 18 | 0.7564 | 1.648e-07 | 0.7684 | 0.5968 | 0.5942 | 0.6678 |
| Epoch 19 | 0.7680 | 4.126e-08 | 0.7516 | 0.5948 | 0.5961 | 0.6649 |
| Epoch 20 | 0.7848 | 2.365e-11 | 0.7500 | 0.5943 | 0.5953 | 0.6639 |

Final epoch (Epoch 20), all 1252 validation rows: QWK (6 labels, summary.json) 0.8505 · QWK5 (AP-ROP folded onto stage 4/5, val_predictions.npz) 0.8753 · accuracy 0.7500 · macro-F1 0.5943 · balanced accuracy 0.6639 · AUC 0.8985

| final epoch | Normal | Stage 1 | Stage 2 | Stage 3 | Stage 4/5 | AP-ROP |
|---|---|---|---|---|---|---|
| recall | 0.8951 | 0.4959 | 0.5699 | 0.2727 | 0.8889 | 0.8611 |
| support | 772 | 123 | 193 | 110 | 18 | 36 |

#### Arm G · seed 1337 · fold_3 (CV fold 3)

Kernel `rop-v3-dinov2-s-r518-s1337-lr1e-4-b` · run dir `results/rop/v3/runs/dinov2_s_r518_s1337/fold_3` · as-run manifest `results/prepared/rop_staging/_v3_fold3.csv`

Fold file `results/prepared/rop_staging/fold_3.csv`: validation = 1252 photos of 327 infants (ostrava 713, ropvl 362, shenzhen 177); training = 2459 photos of 1200 infants (ostrava 609, ropvl 1313, shenzhen 537).

| rows | normal | stage_1 | stage_2 | stage_3 | stage_4_5 | arop | total |
|---|---|---|---|---|---|---|---|
| training rows | 888 | 409 | 564 | 400 | 71 | 127 | 2459 |
| validation rows | 766 | 111 | 182 | 134 | 18 | 41 | 1252 |

- images per epoch = 2459 (the training rows: the fold file's 2469 less 10 exact duplicate(s) dropped at run time)
- steps per epoch = 154 (= ceil(2459 / (16 x 1)), batch_size x grad_accum)
- minutes 58.2 (fold start to next fold start, kernel log) · 19.6 img/s · GPU Tesla T4 · kernel log reports train 2459 val 1252

Per epoch (`metrics.json` history; lr = the first optimizer group, i.e. the lowest layer under layer-wise decay, at the end of the epoch):

| Epoch | train loss | lr | val acc | val macro-F1 | val macro-P | val macro-R |
|---|---|---|---|---|---|---|
| Epoch 1 | 1.9095 | 2.872e-06 | 0.6254 | 0.2000 | 0.4672 | 0.2701 |
| Epoch 2 | 1.7061 | 5.498e-06 | 0.6797 | 0.3656 | 0.4076 | 0.4821 |
| Epoch 3 | 1.3986 | 5.455e-06 | 0.7756 | 0.6901 | 0.7017 | 0.6965 |
| Epoch 4 | 1.2212 | 5.331e-06 | 0.8099 | 0.6399 | 0.6308 | 0.6843 |
| Epoch 5 | 1.1824 | 5.128e-06 | 0.6206 | 0.5354 | 0.5109 | 0.6468 |
| Epoch 6 | 1.0575 | 4.852e-06 | 0.8195 | 0.7079 | 0.7234 | 0.7108 |
| Epoch 7 | 1.0414 | 4.513e-06 | 0.8059 | 0.7367 | 0.7404 | 0.7528 |
| Epoch 8 | 1.0416 | 4.120e-06 | 0.7819 | 0.6886 | 0.6711 | 0.7573 |
| Epoch 9 | 1.0059 | 3.686e-06 | 0.8371 | 0.7589 | 0.7597 | 0.7679 |
| Epoch 10 | 0.9573 | 3.223e-06 | 0.6789 | 0.6864 | 0.6986 | 0.7509 |
| Epoch 11 | 0.9219 | 2.746e-06 | 0.7939 | 0.7534 | 0.7479 | 0.7852 |
| Epoch 12 | 0.8908 | 2.268e-06 | 0.8514 | 0.7667 | 0.7730 | 0.7681 |
| Epoch 13 | 0.8534 | 1.806e-06 | 0.7196 | 0.7238 | 0.7239 | 0.7798 |
| Epoch 14 | 0.8440 | 1.372e-06 | 0.8099 | 0.7741 | 0.7749 | 0.7925 |
| Epoch 15 | 0.8340 | 9.795e-07 | 0.8251 | 0.7832 | 0.7891 | 0.7992 |
| Epoch 16 | 0.7879 | 6.411e-07 | 0.8482 | 0.7879 | 0.7918 | 0.8006 |
| Epoch 17 | 0.7973 | 3.667e-07 | 0.8163 | 0.7717 | 0.7493 | 0.8126 |
| Epoch 18 | 0.7795 | 1.647e-07 | 0.8267 | 0.7734 | 0.7591 | 0.8073 |
| Epoch 19 | 0.7894 | 4.124e-08 | 0.8227 | 0.7745 | 0.7578 | 0.8132 |
| Epoch 20 | 0.7722 | 2.376e-11 | 0.8211 | 0.7721 | 0.7544 | 0.8094 |

Final epoch (Epoch 20), all 1252 validation rows: QWK (6 labels, summary.json) 0.9113 · QWK5 (AP-ROP folded onto stage 4/5, val_predictions.npz) 0.9165 · accuracy 0.8211 · macro-F1 0.7721 · balanced accuracy 0.8094 · AUC 0.9644

| final epoch | Normal | Stage 1 | Stage 2 | Stage 3 | Stage 4/5 | AP-ROP |
|---|---|---|---|---|---|---|
| recall | 0.8747 | 0.6306 | 0.6758 | 0.8284 | 0.9444 | 0.9024 |
| support | 766 | 111 | 182 | 134 | 18 | 41 |

#### Arm G · seed 1337 · fold_4 (CV fold 4)

Kernel `rop-v3-dinov2-s-r518-s1337-lr1e-4-b` · run dir `results/rop/v3/runs/dinov2_s_r518_s1337/fold_4` · as-run manifest `results/prepared/rop_staging/_v3_fold4.csv`

Fold file `results/prepared/rop_staging/fold_4.csv`: validation = 1253 photos of 333 infants (ostrava 711, ropvl 381, shenzhen 161); training = 2452 photos of 1194 infants (ostrava 607, ropvl 1292, shenzhen 553).

| rows | normal | stage_1 | stage_2 | stage_3 | stage_4_5 | arop | total |
|---|---|---|---|---|---|---|---|
| training rows | 882 | 420 | 546 | 401 | 71 | 132 | 2452 |
| validation rows | 776 | 92 | 201 | 131 | 18 | 35 | 1253 |

- images per epoch = 2452 (the training rows: the fold file's 2466 less 14 exact duplicate(s) dropped at run time)
- steps per epoch = 154 (= ceil(2452 / (16 x 1)), batch_size x grad_accum)
- minutes 57.8 (fold start to next fold start, kernel log) · 19.5 img/s · GPU Tesla T4 · kernel log reports train 2452 val 1253

Per epoch (`metrics.json` history; lr = the first optimizer group, i.e. the lowest layer under layer-wise decay, at the end of the epoch):

| Epoch | train loss | lr | val acc | val macro-F1 | val macro-P | val macro-R |
|---|---|---|---|---|---|---|
| Epoch 1 | 1.9123 | 2.872e-06 | 0.6640 | 0.3857 | 0.4383 | 0.4012 |
| Epoch 2 | 1.7100 | 5.498e-06 | 0.6824 | 0.4909 | 0.5382 | 0.5933 |
| Epoch 3 | 1.3669 | 5.455e-06 | 0.7709 | 0.5422 | 0.5514 | 0.6240 |
| Epoch 4 | 1.2483 | 5.331e-06 | 0.8164 | 0.6265 | 0.6359 | 0.6914 |
| Epoch 5 | 1.1736 | 5.128e-06 | 0.7941 | 0.5973 | 0.5778 | 0.6852 |
| Epoch 6 | 1.1293 | 4.852e-06 | 0.7550 | 0.6410 | 0.6461 | 0.7231 |
| Epoch 7 | 1.0486 | 4.513e-06 | 0.6887 | 0.6119 | 0.6027 | 0.7207 |
| Epoch 8 | 1.0131 | 4.120e-06 | 0.8212 | 0.6583 | 0.6383 | 0.7281 |
| Epoch 9 | 0.9584 | 3.686e-06 | 0.8563 | 0.7173 | 0.7146 | 0.7373 |
| Epoch 10 | 0.9450 | 3.223e-06 | 0.7885 | 0.6613 | 0.6453 | 0.7411 |
| Epoch 11 | 0.9083 | 2.746e-06 | 0.8675 | 0.7432 | 0.7293 | 0.7705 |
| Epoch 12 | 0.9132 | 2.268e-06 | 0.8053 | 0.6961 | 0.6761 | 0.7428 |
| Epoch 13 | 0.8472 | 1.806e-06 | 0.8476 | 0.7568 | 0.7584 | 0.7772 |
| Epoch 14 | 0.8384 | 1.372e-06 | 0.8093 | 0.7133 | 0.6942 | 0.7664 |
| Epoch 15 | 0.8107 | 9.795e-07 | 0.7598 | 0.6989 | 0.7060 | 0.7549 |
| Epoch 16 | 0.7955 | 6.411e-07 | 0.7957 | 0.7023 | 0.6892 | 0.7653 |
| Epoch 17 | 0.7614 | 3.667e-07 | 0.8404 | 0.7619 | 0.7579 | 0.7923 |
| Epoch 18 | 0.7867 | 1.647e-07 | 0.8348 | 0.7460 | 0.7324 | 0.7884 |
| Epoch 19 | 0.7527 | 4.124e-08 | 0.8484 | 0.7595 | 0.7522 | 0.7883 |
| Epoch 20 | 0.7678 | 2.376e-11 | 0.8484 | 0.7597 | 0.7517 | 0.7889 |

Final epoch (Epoch 20), all 1253 validation rows: QWK (6 labels, summary.json) 0.9023 · QWK5 (AP-ROP folded onto stage 4/5, val_predictions.npz) 0.9152 · accuracy 0.8484 · macro-F1 0.7597 · balanced accuracy 0.7889 · AUC 0.9426

| final epoch | Normal | Stage 1 | Stage 2 | Stage 3 | Stage 4/5 | AP-ROP |
|---|---|---|---|---|---|---|
| recall | 0.9356 | 0.6196 | 0.6119 | 0.8473 | 0.8333 | 0.8857 |
| support | 776 | 92 | 201 | 131 | 18 | 35 |

### Arm H: DINOv2 on the v4 pool

Run dirs: `results/rop/v3/runs/dinov2_s_v4data_s42`, `results/rop/v3/runs/dinov2_s_v4data_s1337`. Missing seeds of [42, 1337]: none.

Recipe (the as-run `config.yaml` of fold_0, seed 42; any fold whose config differs says so):

| parameter | value |
|---|---|
| model.arch | dinov2_s |
| preprocess.image_size | 392 |
| preprocess.normalize | imagenet |
| preprocess.clahe | True |
| train.lr | 0.0001 |
| train.layer_decay | 0.8 |
| train.weight_decay | 0.05 |
| train.epochs | 20 |
| train.batch_size | 16 |
| train.grad_accum | 1 (default) |
| train.label_smoothing | 0.05 |
| train.loss | ce |
| train.class_balanced | True |
| train.class_balanced_beta | 0.999 |
| train.weighted_sampler | False |
| train.strong_aug | True |
| train.site_randomization | True |
| train.amp | True |
| train.ema_decay | not set |
| train.head_warmup_epochs | not set |
| model.pretrained | True |
| model.dropout | 0.3 |
| model.drop_path | 0.1 |
| eval.early_stopping | False |
| eval.primary_metric | macro_f1 |
| train.aug_v4 | not set |
| train.bag_loss | not set |
| train.uda | not set |
| pool (summary.json) | v4 |
| config source | configs/rop_staging_dinov2_v4data.yaml |

<details><summary>Every key of that as-run config.yaml</summary>

| key | value |
|---|---|
| disease | rop_staging |
| seed | 42 |
| data.num_classes | 6 |
| data.class_names | [Normal, Stage 1, Stage 2, Stage 3, Stage 4/5, AP-ROP] |
| data.manifest | results/prepared/rop_staging/_v3_fold0.csv |
| data.presplit | True |
| data.group_col | patient_id |
| preprocess.image_size | 392 |
| preprocess.circle_crop | True |
| preprocess.letterbox | True |
| preprocess.ben_graham | False |
| preprocess.clahe | True |
| preprocess.normalize | imagenet |
| preprocess.cached | True |
| preprocess.cache_root | /tmp/_cache768 |
| model.arch | dinov2_s |
| model.weights_path | None |
| model.pretrained | True |
| model.head | classification |
| model.dropout | 0.3 |
| model.drop_path | 0.1 |
| train.arm_tag | v4data |
| train.epochs | 20 |
| train.batch_size | 16 |
| train.lr | 0.0001 |
| train.layer_decay | 0.8 |
| train.weight_decay | 0.05 |
| train.label_smoothing | 0.05 |
| train.amp | True |
| train.strong_aug | True |
| train.site_randomization | True |
| train.loss | ce |
| train.class_balanced | True |
| train.class_balanced_beta | 0.999 |
| train.weighted_sampler | False |
| train.num_workers | 4 |
| eval.early_stopping | False |
| eval.primary_metric | macro_f1 |
| eval.qwk_classes | [Normal, Stage 1, Stage 2, Stage 3, Stage 4/5] |
| eval.report_per_class_recall | True |
| eval.unvalidated_classes | [Stage 4/5, AP-ROP] |
| _source | configs/rop_staging_dinov2_v4data.yaml |

</details>

#### Arm H · seed 42: overview (8 folds: fold_0, fold_1, fold_2, fold_3, fold_4, fold_site, fold_loso_ostrava, fold_loso_ropvl)

| fold | train rows | val rows | images/epoch | steps/epoch | minutes | img/s | final acc | final macro-F1 | final QWK (6) | final QWK5 | AUC |
|---|---|---|---|---|---|---|---|---|---|---|---|
| fold_0 | 2447 | 1691 | 2447 | 153 | 37.2 | 29.3 | 0.6410 | 0.5566 | 0.5628 | 0.6628 | 0.8879 |
| fold_1 | 2465 | 1252 | 2465 | 155 | 34.3 | 30.4 | 0.8195 | 0.7523 | 0.8861 | 0.8801 | 0.9489 |
| fold_2 | 2524 | 1252 | 2524 | 158 | 34.9 | 30.2 | 0.7452 | 0.5870 | 0.7973 | 0.8317 | 0.8474 |
| fold_3 | 2438 | 1252 | 2438 | 153 | 34.0 | 31.1 | 0.8147 | 0.7652 | 0.9185 | 0.9154 | 0.9593 |
| fold_4 | 2434 | 1253 | 2434 | 153 | 34.2 | 30.6 | 0.8372 | 0.7689 | 0.9087 | 0.9145 | 0.9490 |
| fold_site | 2367 | 740 | 2367 | 148 | 27.7 | 31.9 | 0.8662 | 0.5539 | 0.9461 | 0.9585 | n/a |
| fold_loso_ostrava | 2383 | 4285 | 2383 | 149 | 47.1 | 33.5 | 0.7853 | 0.3953 | 0.7213 | 0.7468 | n/a |
| fold_loso_ropvl | 1404 | 1675 | 1404 | 88 | 24.3 | 32.7 | 0.6573 | 0.4432 | 0.7520 | 0.8102 | 0.7848 |

#### Arm H · seed 42 · fold_0 (CV fold 0)

Kernel `rop-v4-dinov2-s-v4data-s42-lr1e-4-a` · run dir `results/rop/v3/runs/dinov2_s_v4data_s42/fold_0` · as-run manifest `results/prepared/rop_staging/_v3_fold0.csv`

Fold file `results/prepared/rop_staging/v4/fold_0.csv`: validation = 1691 photos of 297 infants (ostrava 1219, ropvl 310, shenzhen 162); training = 2447 photos of 1231 infants (ostrava 529, ropvl 1363, shenzhen 555).

| rows | normal | stage_1 | stage_2 | stage_3 | stage_4_5 | arop | total |
|---|---|---|---|---|---|---|---|
| training rows | 888 | 392 | 552 | 404 | 70 | 141 | 2447 |
| validation rows | 767 | 137 | 163 | 129 | 19 | 476 | 1691 |

- images per epoch = 2447 (the training rows: the fold file's 2453 less 6 exact duplicate(s) dropped at run time)
- steps per epoch = 153 (= ceil(2447 / (16 x 1)), batch_size x grad_accum)
- minutes 37.2 (fold start to next fold start, kernel log) · 29.3 img/s · GPU Tesla T4 · kernel log reports train 2447 val 1691

Per epoch (`metrics.json` history; lr = the first optimizer group, i.e. the lowest layer under layer-wise decay, at the end of the epoch):

| Epoch | train loss | lr | val acc | val macro-F1 | val macro-P | val macro-R |
|---|---|---|---|---|---|---|
| Epoch 1 | 1.9135 | 2.872e-06 | 0.3489 | 0.1501 | 0.1583 | 0.2377 |
| Epoch 2 | 1.7003 | 5.498e-06 | 0.6736 | 0.5205 | 0.5464 | 0.6014 |
| Epoch 3 | 1.3767 | 5.455e-06 | 0.6895 | 0.4701 | 0.5231 | 0.5739 |
| Epoch 4 | 1.2475 | 5.331e-06 | 0.6428 | 0.5441 | 0.5986 | 0.6107 |
| Epoch 5 | 1.1630 | 5.128e-06 | 0.6464 | 0.5059 | 0.5578 | 0.6588 |
| Epoch 6 | 1.0868 | 4.852e-06 | 0.7522 | 0.6479 | 0.6489 | 0.6809 |
| Epoch 7 | 1.0427 | 4.513e-06 | 0.6470 | 0.5257 | 0.5789 | 0.6735 |
| Epoch 8 | 1.0263 | 4.120e-06 | 0.6558 | 0.5477 | 0.5803 | 0.6541 |
| Epoch 9 | 0.9817 | 3.686e-06 | 0.6322 | 0.5164 | 0.5592 | 0.6291 |
| Epoch 10 | 0.9424 | 3.223e-06 | 0.6127 | 0.5067 | 0.5666 | 0.6107 |
| Epoch 11 | 0.9436 | 2.746e-06 | 0.5683 | 0.5160 | 0.5940 | 0.6290 |
| Epoch 12 | 0.9007 | 2.268e-06 | 0.6174 | 0.4939 | 0.5453 | 0.6306 |
| Epoch 13 | 0.8568 | 1.806e-06 | 0.6192 | 0.5199 | 0.5815 | 0.6478 |
| Epoch 14 | 0.8380 | 1.372e-06 | 0.6286 | 0.5499 | 0.6194 | 0.6423 |
| Epoch 15 | 0.8184 | 9.795e-07 | 0.6588 | 0.5538 | 0.5971 | 0.6660 |
| Epoch 16 | 0.8042 | 6.411e-07 | 0.6481 | 0.5549 | 0.6050 | 0.6603 |
| Epoch 17 | 0.7791 | 3.667e-07 | 0.6546 | 0.5680 | 0.6199 | 0.6660 |
| Epoch 18 | 0.7687 | 1.647e-07 | 0.6475 | 0.5639 | 0.6162 | 0.6599 |
| Epoch 19 | 0.7711 | 4.124e-08 | 0.6434 | 0.5574 | 0.6116 | 0.6614 |
| Epoch 20 | 0.7678 | 2.378e-11 | 0.6410 | 0.5566 | 0.6127 | 0.6601 |

Final epoch (Epoch 20), all 1691 validation rows: QWK (6 labels, summary.json) 0.5628 · QWK5 (AP-ROP folded onto stage 4/5, val_predictions.npz) 0.6628 · accuracy 0.6410 · macro-F1 0.5566 · balanced accuracy 0.6601 · AUC 0.8879

| final epoch | Normal | Stage 1 | Stage 2 | Stage 3 | Stage 4/5 | AP-ROP |
|---|---|---|---|---|---|---|
| recall | 0.9648 | 0.5693 | 0.5031 | 0.8605 | 0.9474 | 0.1155 |
| support | 767 | 137 | 163 | 129 | 19 | 476 |

#### Arm H · seed 42 · fold_1 (CV fold 1)

Kernel `rop-v4-dinov2-s-v4data-s42-lr1e-4-a` · run dir `results/rop/v3/runs/dinov2_s_v4data_s42/fold_1` · as-run manifest `results/prepared/rop_staging/_v3_fold1.csv`

Fold file `results/prepared/rop_staging/v4/fold_1.csv`: validation = 1252 photos of 299 infants (ostrava 812, ropvl 327, shenzhen 113); training = 2465 photos of 1228 infants (ostrava 515, ropvl 1346, shenzhen 604).

| rows | normal | stage_1 | stage_2 | stage_3 | stage_4_5 | arop | total |
|---|---|---|---|---|---|---|---|
| training rows | 890 | 393 | 530 | 442 | 73 | 137 | 2465 |
| validation rows | 756 | 114 | 220 | 116 | 16 | 30 | 1252 |

- images per epoch = 2465 (the training rows: the fold file's 2469 less 4 exact duplicate(s) dropped at run time)
- steps per epoch = 155 (= ceil(2465 / (16 x 1)), batch_size x grad_accum)
- minutes 34.3 (fold start to next fold start, kernel log) · 30.4 img/s · GPU Tesla T4 · kernel log reports train 2465 val 1252

Per epoch (`metrics.json` history; lr = the first optimizer group, i.e. the lowest layer under layer-wise decay, at the end of the epoch):

| Epoch | train loss | lr | val acc | val macro-F1 | val macro-P | val macro-R |
|---|---|---|---|---|---|---|
| Epoch 1 | 1.9260 | 2.872e-06 | 0.2292 | 0.0793 | 0.1952 | 0.1814 |
| Epoch 2 | 1.6853 | 5.498e-06 | 0.7316 | 0.5418 | 0.5707 | 0.6113 |
| Epoch 3 | 1.3591 | 5.455e-06 | 0.7684 | 0.6174 | 0.6609 | 0.6257 |
| Epoch 4 | 1.2223 | 5.331e-06 | 0.7404 | 0.6188 | 0.6383 | 0.6603 |
| Epoch 5 | 1.1288 | 5.128e-06 | 0.7955 | 0.6678 | 0.7281 | 0.6618 |
| Epoch 6 | 1.0564 | 4.852e-06 | 0.8131 | 0.6658 | 0.6874 | 0.6875 |
| Epoch 7 | 1.0612 | 4.513e-06 | 0.7907 | 0.6566 | 0.7671 | 0.6171 |
| Epoch 8 | 0.9914 | 4.120e-06 | 0.8083 | 0.7015 | 0.7139 | 0.7162 |
| Epoch 9 | 0.9830 | 3.686e-06 | 0.8131 | 0.7351 | 0.7596 | 0.7309 |
| Epoch 10 | 0.9369 | 3.223e-06 | 0.7859 | 0.7002 | 0.6944 | 0.7210 |
| Epoch 11 | 0.9019 | 2.746e-06 | 0.8059 | 0.7371 | 0.7651 | 0.7449 |
| Epoch 12 | 0.8804 | 2.268e-06 | 0.8131 | 0.7202 | 0.7576 | 0.7087 |
| Epoch 13 | 0.8581 | 1.806e-06 | 0.8291 | 0.7574 | 0.7735 | 0.7536 |
| Epoch 14 | 0.8227 | 1.372e-06 | 0.8187 | 0.7254 | 0.7268 | 0.7478 |
| Epoch 15 | 0.8113 | 9.795e-07 | 0.8251 | 0.7589 | 0.7704 | 0.7580 |
| Epoch 16 | 0.7987 | 6.411e-07 | 0.8275 | 0.7565 | 0.7819 | 0.7420 |
| Epoch 17 | 0.7878 | 3.667e-07 | 0.8179 | 0.7525 | 0.7771 | 0.7471 |
| Epoch 18 | 0.7789 | 1.647e-07 | 0.8227 | 0.7546 | 0.7825 | 0.7453 |
| Epoch 19 | 0.7667 | 4.125e-08 | 0.8203 | 0.7537 | 0.7774 | 0.7448 |
| Epoch 20 | 0.7452 | 2.373e-11 | 0.8195 | 0.7523 | 0.7764 | 0.7436 |

Final epoch (Epoch 20), all 1252 validation rows: QWK (6 labels, summary.json) 0.8861 · QWK5 (AP-ROP folded onto stage 4/5, val_predictions.npz) 0.8801 · accuracy 0.8195 · macro-F1 0.7523 · balanced accuracy 0.7436 · AUC 0.9489

| final epoch | Normal | Stage 1 | Stage 2 | Stage 3 | Stage 4/5 | AP-ROP |
|---|---|---|---|---|---|---|
| recall | 0.9696 | 0.5088 | 0.4955 | 0.7500 | 0.9375 | 0.8000 |
| support | 756 | 114 | 220 | 116 | 16 | 30 |

#### Arm H · seed 42 · fold_2 (CV fold 2)

Kernel `rop-v4-dinov2-s-v4data-s42-lr1e-4-a` · run dir `results/rop/v3/runs/dinov2_s_v4data_s42/fold_2` · as-run manifest `results/prepared/rop_staging/_v3_fold2.csv`

Fold file `results/prepared/rop_staging/v4/fold_2.csv`: validation = 1252 photos of 272 infants (ostrava 830, ropvl 295, shenzhen 127); training = 2524 photos of 1255 infants (ostrava 555, ropvl 1378, shenzhen 591).

| rows | normal | stage_1 | stage_2 | stage_3 | stage_4_5 | arop | total |
|---|---|---|---|---|---|---|---|
| training rows | 888 | 400 | 553 | 481 | 71 | 131 | 2524 |
| validation rows | 772 | 123 | 193 | 110 | 18 | 36 | 1252 |

- images per epoch = 2524 (the training rows: the fold file's 2528 less 4 exact duplicate(s) dropped at run time)
- steps per epoch = 158 (= ceil(2524 / (16 x 1)), batch_size x grad_accum)
- minutes 34.9 (fold start to next fold start, kernel log) · 30.2 img/s · GPU Tesla T4 · kernel log reports train 2524 val 1252

Per epoch (`metrics.json` history; lr = the first optimizer group, i.e. the lowest layer under layer-wise decay, at the end of the epoch):

| Epoch | train loss | lr | val acc | val macro-F1 | val macro-P | val macro-R |
|---|---|---|---|---|---|---|
| Epoch 1 | 1.9094 | 2.872e-06 | 0.6214 | 0.2053 | 0.1937 | 0.2828 |
| Epoch 2 | 1.7075 | 5.498e-06 | 0.5551 | 0.3551 | 0.4104 | 0.4356 |
| Epoch 3 | 1.3836 | 5.455e-06 | 0.5128 | 0.3854 | 0.5256 | 0.4606 |
| Epoch 4 | 1.2572 | 5.331e-06 | 0.5535 | 0.4941 | 0.5337 | 0.5852 |
| Epoch 5 | 1.1572 | 5.128e-06 | 0.6909 | 0.5222 | 0.5715 | 0.5867 |
| Epoch 6 | 1.0901 | 4.853e-06 | 0.6573 | 0.5030 | 0.5171 | 0.6083 |
| Epoch 7 | 1.0754 | 4.513e-06 | 0.6318 | 0.5328 | 0.6379 | 0.6130 |
| Epoch 8 | 1.0328 | 4.121e-06 | 0.7077 | 0.5260 | 0.5565 | 0.6004 |
| Epoch 9 | 0.9913 | 3.686e-06 | 0.7013 | 0.5266 | 0.5478 | 0.6290 |
| Epoch 10 | 0.9594 | 3.223e-06 | 0.6158 | 0.4914 | 0.5149 | 0.5904 |
| Epoch 11 | 0.9425 | 2.746e-06 | 0.6390 | 0.5509 | 0.5967 | 0.6160 |
| Epoch 12 | 0.9021 | 2.268e-06 | 0.7748 | 0.5881 | 0.6230 | 0.6545 |
| Epoch 13 | 0.8521 | 1.806e-06 | 0.6925 | 0.5458 | 0.5877 | 0.6263 |
| Epoch 14 | 0.8613 | 1.372e-06 | 0.7196 | 0.5740 | 0.5972 | 0.6392 |
| Epoch 15 | 0.8370 | 9.796e-07 | 0.7708 | 0.5873 | 0.6085 | 0.6424 |
| Epoch 16 | 0.8179 | 6.412e-07 | 0.7540 | 0.5608 | 0.5742 | 0.6244 |
| Epoch 17 | 0.7922 | 3.668e-07 | 0.7516 | 0.5979 | 0.6290 | 0.6520 |
| Epoch 18 | 0.7894 | 1.648e-07 | 0.7348 | 0.5804 | 0.6096 | 0.6441 |
| Epoch 19 | 0.7781 | 4.126e-08 | 0.7468 | 0.5871 | 0.6146 | 0.6486 |
| Epoch 20 | 0.7833 | 2.367e-11 | 0.7452 | 0.5870 | 0.6144 | 0.6489 |

Final epoch (Epoch 20), all 1252 validation rows: QWK (6 labels, summary.json) 0.7973 · QWK5 (AP-ROP folded onto stage 4/5, val_predictions.npz) 0.8317 · accuracy 0.7452 · macro-F1 0.5870 · balanced accuracy 0.6489 · AUC 0.8474

| final epoch | Normal | Stage 1 | Stage 2 | Stage 3 | Stage 4/5 | AP-ROP |
|---|---|---|---|---|---|---|
| recall | 0.9041 | 0.5447 | 0.5130 | 0.2091 | 0.8889 | 0.8333 |
| support | 772 | 123 | 193 | 110 | 18 | 36 |

#### Arm H · seed 42 · fold_3 (CV fold 3)

Kernel `rop-v4-dinov2-s-v4data-s42-lr1e-4-a` · run dir `results/rop/v3/runs/dinov2_s_v4data_s42/fold_3` · as-run manifest `results/prepared/rop_staging/_v3_fold3.csv`

Fold file `results/prepared/rop_staging/v4/fold_3.csv`: validation = 1252 photos of 327 infants (ostrava 713, ropvl 362, shenzhen 177); training = 2438 photos of 1200 infants (ostrava 588, ropvl 1313, shenzhen 537).

| rows | normal | stage_1 | stage_2 | stage_3 | stage_4_5 | arop | total |
|---|---|---|---|---|---|---|---|
| training rows | 888 | 392 | 560 | 400 | 71 | 127 | 2438 |
| validation rows | 766 | 111 | 182 | 134 | 18 | 41 | 1252 |

- images per epoch = 2438 (the training rows: the fold file's 2442 less 4 exact duplicate(s) dropped at run time)
- steps per epoch = 153 (= ceil(2438 / (16 x 1)), batch_size x grad_accum)
- minutes 34.0 (fold start to next fold start, kernel log) · 31.1 img/s · GPU Tesla T4 · kernel log reports train 2438 val 1252

Per epoch (`metrics.json` history; lr = the first optimizer group, i.e. the lowest layer under layer-wise decay, at the end of the epoch):

| Epoch | train loss | lr | val acc | val macro-F1 | val macro-P | val macro-R |
|---|---|---|---|---|---|---|
| Epoch 1 | 1.9241 | 2.872e-06 | 0.5487 | 0.2442 | 0.4727 | 0.2661 |
| Epoch 2 | 1.6857 | 5.498e-06 | 0.5184 | 0.4553 | 0.4622 | 0.6058 |
| Epoch 3 | 1.4208 | 5.455e-06 | 0.7268 | 0.5443 | 0.5350 | 0.6539 |
| Epoch 4 | 1.2391 | 5.331e-06 | 0.6829 | 0.6131 | 0.5771 | 0.6925 |
| Epoch 5 | 1.1590 | 5.128e-06 | 0.8035 | 0.6924 | 0.7254 | 0.6956 |
| Epoch 6 | 1.1211 | 4.852e-06 | 0.7899 | 0.7334 | 0.7575 | 0.7433 |
| Epoch 7 | 1.0528 | 4.513e-06 | 0.7939 | 0.7200 | 0.7082 | 0.7574 |
| Epoch 8 | 1.0084 | 4.120e-06 | 0.8387 | 0.7115 | 0.7135 | 0.7247 |
| Epoch 9 | 1.0012 | 3.686e-06 | 0.7875 | 0.7312 | 0.7426 | 0.7549 |
| Epoch 10 | 0.9493 | 3.223e-06 | 0.8618 | 0.7672 | 0.7721 | 0.7678 |
| Epoch 11 | 0.9281 | 2.746e-06 | 0.7819 | 0.7571 | 0.7771 | 0.7602 |
| Epoch 12 | 0.8987 | 2.268e-06 | 0.8355 | 0.7654 | 0.7658 | 0.7731 |
| Epoch 13 | 0.8597 | 1.806e-06 | 0.7979 | 0.7525 | 0.7546 | 0.7637 |
| Epoch 14 | 0.8283 | 1.372e-06 | 0.7835 | 0.7562 | 0.7669 | 0.7747 |
| Epoch 15 | 0.8189 | 9.795e-07 | 0.7867 | 0.7236 | 0.7247 | 0.7336 |
| Epoch 16 | 0.8003 | 6.411e-07 | 0.8195 | 0.7624 | 0.7584 | 0.7815 |
| Epoch 17 | 0.7845 | 3.667e-07 | 0.8147 | 0.7596 | 0.7567 | 0.7752 |
| Epoch 18 | 0.7973 | 1.647e-07 | 0.8171 | 0.7611 | 0.7552 | 0.7787 |
| Epoch 19 | 0.7739 | 4.124e-08 | 0.8099 | 0.7605 | 0.7579 | 0.7764 |
| Epoch 20 | 0.7760 | 2.378e-11 | 0.8147 | 0.7652 | 0.7640 | 0.7790 |

Final epoch (Epoch 20), all 1252 validation rows: QWK (6 labels, summary.json) 0.9185 · QWK5 (AP-ROP folded onto stage 4/5, val_predictions.npz) 0.9154 · accuracy 0.8147 · macro-F1 0.7652 · balanced accuracy 0.7790 · AUC 0.9593

| final epoch | Normal | Stage 1 | Stage 2 | Stage 3 | Stage 4/5 | AP-ROP |
|---|---|---|---|---|---|---|
| recall | 0.8695 | 0.5495 | 0.6923 | 0.8582 | 0.7778 | 0.9268 |
| support | 766 | 111 | 182 | 134 | 18 | 41 |

#### Arm H · seed 42 · fold_4 (CV fold 4)

Kernel `rop-v4-dinov2-s-v4data-s42-lr1e-4-a` · run dir `results/rop/v3/runs/dinov2_s_v4data_s42/fold_4` · as-run manifest `results/prepared/rop_staging/_v3_fold4.csv`

Fold file `results/prepared/rop_staging/v4/fold_4.csv`: validation = 1253 photos of 333 infants (ostrava 711, ropvl 381, shenzhen 161); training = 2434 photos of 1194 infants (ostrava 589, ropvl 1292, shenzhen 553).

| rows | normal | stage_1 | stage_2 | stage_3 | stage_4_5 | arop | total |
|---|---|---|---|---|---|---|---|
| training rows | 882 | 403 | 545 | 401 | 71 | 132 | 2434 |
| validation rows | 776 | 92 | 201 | 131 | 18 | 35 | 1253 |

- images per epoch = 2434 (the training rows: the fold file's 2440 less 6 exact duplicate(s) dropped at run time)
- steps per epoch = 153 (= ceil(2434 / (16 x 1)), batch_size x grad_accum)
- minutes 34.2 (fold start to next fold start, kernel log) · 30.6 img/s · GPU Tesla T4 · kernel log reports train 2434 val 1253

Per epoch (`metrics.json` history; lr = the first optimizer group, i.e. the lowest layer under layer-wise decay, at the end of the epoch):

| Epoch | train loss | lr | val acc | val macro-F1 | val macro-P | val macro-R |
|---|---|---|---|---|---|---|
| Epoch 1 | 1.8971 | 2.872e-06 | 0.6305 | 0.3198 | 0.3048 | 0.3530 |
| Epoch 2 | 1.7482 | 5.498e-06 | 0.6648 | 0.3752 | 0.4335 | 0.5560 |
| Epoch 3 | 1.4204 | 5.455e-06 | 0.7965 | 0.6093 | 0.6283 | 0.6478 |
| Epoch 4 | 1.2312 | 5.331e-06 | 0.8172 | 0.6182 | 0.5991 | 0.7054 |
| Epoch 5 | 1.1700 | 5.128e-06 | 0.7757 | 0.6003 | 0.5669 | 0.7082 |
| Epoch 6 | 1.1235 | 4.852e-06 | 0.6161 | 0.5483 | 0.5354 | 0.6339 |
| Epoch 7 | 1.0636 | 4.513e-06 | 0.8332 | 0.6553 | 0.6506 | 0.7403 |
| Epoch 8 | 1.0080 | 4.120e-06 | 0.8013 | 0.6714 | 0.6404 | 0.7164 |
| Epoch 9 | 0.9866 | 3.686e-06 | 0.7350 | 0.6434 | 0.6563 | 0.7140 |
| Epoch 10 | 0.9402 | 3.223e-06 | 0.7981 | 0.6701 | 0.6599 | 0.7479 |
| Epoch 11 | 0.9091 | 2.746e-06 | 0.7949 | 0.6659 | 0.6303 | 0.7558 |
| Epoch 12 | 0.8946 | 2.268e-06 | 0.7518 | 0.6912 | 0.6697 | 0.7543 |
| Epoch 13 | 0.8668 | 1.806e-06 | 0.8356 | 0.7280 | 0.7114 | 0.7768 |
| Epoch 14 | 0.8388 | 1.372e-06 | 0.8659 | 0.7682 | 0.7486 | 0.7975 |
| Epoch 15 | 0.8306 | 9.795e-07 | 0.7686 | 0.7023 | 0.6858 | 0.7747 |
| Epoch 16 | 0.8183 | 6.411e-07 | 0.8452 | 0.7590 | 0.7368 | 0.8043 |
| Epoch 17 | 0.7756 | 3.667e-07 | 0.8244 | 0.7553 | 0.7406 | 0.8000 |
| Epoch 18 | 0.7779 | 1.647e-07 | 0.8484 | 0.7705 | 0.7533 | 0.8080 |
| Epoch 19 | 0.7732 | 4.124e-08 | 0.8348 | 0.7680 | 0.7499 | 0.8073 |
| Epoch 20 | 0.7874 | 2.378e-11 | 0.8372 | 0.7689 | 0.7503 | 0.8080 |

Final epoch (Epoch 20), all 1253 validation rows: QWK (6 labels, summary.json) 0.9087 · QWK5 (AP-ROP folded onto stage 4/5, val_predictions.npz) 0.9145 · accuracy 0.8372 · macro-F1 0.7689 · balanced accuracy 0.8080 · AUC 0.9490

| final epoch | Normal | Stage 1 | Stage 2 | Stage 3 | Stage 4/5 | AP-ROP |
|---|---|---|---|---|---|---|
| recall | 0.9059 | 0.6413 | 0.6119 | 0.8855 | 0.8889 | 0.9143 |
| support | 776 | 92 | 201 | 131 | 18 | 35 |

#### Arm H · seed 42 · fold_site (held-out source Shenzhen (the v3 site fold))

Kernel `rop-v4-dinov2-s-v4data-s42-lr1e-4-b` · run dir `results/rop/v3/runs/dinov2_s_v4data_s42/fold_site` · as-run manifest `results/prepared/rop_staging/_v3_foldsite.csv`

Fold file `results/prepared/rop_staging/v4/fold_site.csv`: validation = 740 photos of 397 infants (shenzhen 740); training = 2367 photos of 1131 infants (ostrava 694, ropvl 1673).

| rows | normal | stage_1 | stage_2 | stage_3 | stage_4_5 | arop | total |
|---|---|---|---|---|---|---|---|
| training rows | 893 | 409 | 529 | 280 | 89 | 167 | 2367 |
| validation rows | 234 | 89 | 163 | 254 | 0 | 0 | 740 |

- images per epoch = 2367 (the training rows: the fold file's 2369 less 2 exact duplicate(s) dropped at run time)
- steps per epoch = 148 (= ceil(2367 / (16 x 1)), batch_size x grad_accum)
- minutes 27.7 (fold start to next fold start, kernel log) · 31.9 img/s · GPU Tesla T4 · kernel log reports train 2367 val 740

Per epoch (`metrics.json` history; lr = the first optimizer group, i.e. the lowest layer under layer-wise decay, at the end of the epoch):

| Epoch | train loss | lr | val acc | val macro-F1 | val macro-P | val macro-R |
|---|---|---|---|---|---|---|
| Epoch 1 | 1.8737 | 2.873e-06 | 0.5284 | 0.3382 | 0.3875 | 0.3978 |
| Epoch 2 | 1.7108 | 5.498e-06 | 0.6514 | 0.4533 | 0.5743 | 0.4654 |
| Epoch 3 | 1.4298 | 5.455e-06 | 0.7608 | 0.4375 | 0.5223 | 0.4452 |
| Epoch 4 | 1.2625 | 5.331e-06 | 0.7797 | 0.4889 | 0.5251 | 0.5005 |
| Epoch 5 | 1.2136 | 5.128e-06 | 0.8459 | 0.6249 | 0.6731 | 0.6165 |
| Epoch 6 | 1.1309 | 4.852e-06 | 0.8176 | 0.5167 | 0.5544 | 0.5042 |
| Epoch 7 | 1.0810 | 4.513e-06 | 0.8446 | 0.6233 | 0.6695 | 0.6119 |
| Epoch 8 | 1.0286 | 4.120e-06 | 0.8230 | 0.5121 | 0.5646 | 0.5011 |
| Epoch 9 | 0.9709 | 3.686e-06 | 0.7932 | 0.4906 | 0.5683 | 0.4874 |
| Epoch 10 | 0.9643 | 3.223e-06 | 0.7959 | 0.4894 | 0.5538 | 0.4719 |
| Epoch 11 | 0.9242 | 2.746e-06 | 0.8514 | 0.5457 | 0.5584 | 0.5363 |
| Epoch 12 | 0.9043 | 2.268e-06 | 0.8392 | 0.5192 | 0.5800 | 0.5048 |
| Epoch 13 | 0.8608 | 1.806e-06 | 0.8622 | 0.5515 | 0.5670 | 0.5435 |
| Epoch 14 | 0.8232 | 1.372e-06 | 0.8635 | 0.5533 | 0.5747 | 0.5422 |
| Epoch 15 | 0.8260 | 9.794e-07 | 0.8689 | 0.5543 | 0.5671 | 0.5453 |
| Epoch 16 | 0.7818 | 6.410e-07 | 0.8514 | 0.5438 | 0.5680 | 0.5301 |
| Epoch 17 | 0.7826 | 3.667e-07 | 0.8730 | 0.5547 | 0.5735 | 0.5444 |
| Epoch 18 | 0.7818 | 1.647e-07 | 0.8662 | 0.5546 | 0.5716 | 0.5435 |
| Epoch 19 | 0.7730 | 4.122e-08 | 0.8676 | 0.5552 | 0.5705 | 0.5446 |
| Epoch 20 | 0.7678 | 2.390e-11 | 0.8662 | 0.5539 | 0.5707 | 0.5428 |

Final epoch (Epoch 20), all 740 validation rows: QWK (6 labels, summary.json) 0.9461 · QWK5 (AP-ROP folded onto stage 4/5, val_predictions.npz) 0.9585 · accuracy 0.8662 · macro-F1 0.5539 · balanced accuracy 0.5428 · AUC n/a

| final epoch | Normal | Stage 1 | Stage 2 | Stage 3 | Stage 4/5 | AP-ROP |
|---|---|---|---|---|---|---|
| recall | 1.0000 | 0.5843 | 0.7669 | 0.9055 | 0.0000 | 0.0000 |
| support | 234 | 89 | 163 | 254 | 0 | 0 |

#### Arm H · seed 42 · fold_loso_ostrava (held-out source Ostrava (leave-one-source-out))

Kernel `rop-v4-dinov2-s-v4data-s42-lr1e-4-b` · run dir `results/rop/v3/runs/dinov2_s_v4data_s42/fold_loso_ostrava` · as-run manifest `results/prepared/rop_staging/_v3_foldloso_ostrava.csv`

Fold file `results/prepared/rop_staging/v4/fold_loso_ostrava.csv`: validation = 4285 photos of 148 infants (ostrava 4285); training = 2383 photos of 1379 infants (ropvl 1673, shenzhen 710).

| rows | normal | stage_1 | stage_2 | stage_3 | stage_4_5 | arop | total |
|---|---|---|---|---|---|---|---|
| training rows | 839 | 322 | 494 | 492 | 89 | 147 | 2383 |
| validation rows | 2980 | 252 | 458 | 125 | 0 | 470 | 4285 |

- images per epoch = 2383 (the training rows: the fold file's 2389 less 6 exact duplicate(s) dropped at run time)
- steps per epoch = 149 (= ceil(2383 / (16 x 1)), batch_size x grad_accum)
- minutes 47.1 (fold start to next fold start, kernel log) · 33.5 img/s · GPU Tesla T4 · kernel log reports train 2383 val 4285

Per epoch (`metrics.json` history; lr = the first optimizer group, i.e. the lowest layer under layer-wise decay, at the end of the epoch):

| Epoch | train loss | lr | val acc | val macro-F1 | val macro-P | val macro-R |
|---|---|---|---|---|---|---|
| Epoch 1 | 1.8709 | 2.873e-06 | 0.6805 | 0.2410 | 0.2927 | 0.2471 |
| Epoch 2 | 1.5688 | 5.498e-06 | 0.7594 | 0.3382 | 0.4216 | 0.3692 |
| Epoch 3 | 1.2020 | 5.455e-06 | 0.7743 | 0.3690 | 0.4505 | 0.3788 |
| Epoch 4 | 1.1261 | 5.331e-06 | 0.7757 | 0.3665 | 0.4560 | 0.3819 |
| Epoch 5 | 1.0378 | 5.128e-06 | 0.7685 | 0.3805 | 0.4563 | 0.3766 |
| Epoch 6 | 0.9935 | 4.852e-06 | 0.7944 | 0.3874 | 0.4947 | 0.3780 |
| Epoch 7 | 0.9296 | 4.513e-06 | 0.8014 | 0.4075 | 0.5037 | 0.3930 |
| Epoch 8 | 0.9156 | 4.120e-06 | 0.7907 | 0.3878 | 0.4761 | 0.3954 |
| Epoch 9 | 0.8926 | 3.686e-06 | 0.7988 | 0.4005 | 0.4963 | 0.3954 |
| Epoch 10 | 0.8622 | 3.223e-06 | 0.7932 | 0.4019 | 0.4796 | 0.4030 |
| Epoch 11 | 0.8572 | 2.746e-06 | 0.7967 | 0.3962 | 0.4932 | 0.3918 |
| Epoch 12 | 0.8175 | 2.268e-06 | 0.7911 | 0.3918 | 0.4868 | 0.3974 |
| Epoch 13 | 0.7829 | 1.806e-06 | 0.7764 | 0.3728 | 0.4722 | 0.3843 |
| Epoch 14 | 0.7555 | 1.372e-06 | 0.7760 | 0.3845 | 0.4715 | 0.3958 |
| Epoch 15 | 0.7507 | 9.795e-07 | 0.7909 | 0.3937 | 0.4688 | 0.4015 |
| Epoch 16 | 0.7302 | 6.410e-07 | 0.7804 | 0.3902 | 0.4653 | 0.3998 |
| Epoch 17 | 0.7377 | 3.667e-07 | 0.7734 | 0.3851 | 0.4652 | 0.3954 |
| Epoch 18 | 0.6951 | 1.647e-07 | 0.7879 | 0.3946 | 0.4748 | 0.4000 |
| Epoch 19 | 0.7107 | 4.122e-08 | 0.7860 | 0.3946 | 0.4706 | 0.4025 |
| Epoch 20 | 0.7144 | 2.388e-11 | 0.7853 | 0.3953 | 0.4713 | 0.4045 |

Final epoch (Epoch 20), all 4285 validation rows: QWK (6 labels, summary.json) 0.7213 · QWK5 (AP-ROP folded onto stage 4/5, val_predictions.npz) 0.7468 · accuracy 0.7853 · macro-F1 0.3953 · balanced accuracy 0.4045 · AUC n/a

| final epoch | Normal | Stage 1 | Stage 2 | Stage 3 | Stage 4/5 | AP-ROP |
|---|---|---|---|---|---|---|
| recall | 0.9530 | 0.1389 | 0.1965 | 0.3920 | 0.0000 | 0.7468 |
| support | 2980 | 252 | 458 | 125 | 0 | 470 |

#### Arm H · seed 42 · fold_loso_ropvl (held-out source ROP-VL (leave-one-source-out))

Kernel `rop-v4-dinov2-s-v4data-s42-lr1e-4-b` · run dir `results/rop/v3/runs/dinov2_s_v4data_s42/fold_loso_ropvl` · as-run manifest `results/prepared/rop_staging/_v3_foldloso_ropvl.csv`

Fold file `results/prepared/rop_staging/v4/fold_loso_ropvl.csv`: validation = 1675 photos of 983 infants (ropvl 1675); training = 1404 photos of 544 infants (ostrava 694, shenzhen 710).

| rows | normal | stage_1 | stage_2 | stage_3 | stage_4_5 | arop | total |
|---|---|---|---|---|---|---|---|
| training rows | 486 | 259 | 347 | 292 | 0 | 20 | 1404 |
| validation rows | 623 | 236 | 338 | 241 | 89 | 148 | 1675 |

- validated with zero training rows: stage_4_5
- images per epoch = 1404 (the training rows: the fold file's 1408 less 4 exact duplicate(s) dropped at run time)
- steps per epoch = 88 (= ceil(1404 / (16 x 1)), batch_size x grad_accum)
- minutes 24.3 (fold start to next fold start, kernel log) · 32.7 img/s · GPU Tesla T4 · kernel log reports train 1404 val 1675

Per epoch (`metrics.json` history; lr = the first optimizer group, i.e. the lowest layer under layer-wise decay, at the end of the epoch):

| Epoch | train loss | lr | val acc | val macro-F1 | val macro-P | val macro-R |
|---|---|---|---|---|---|---|
| Epoch 1 | 1.8905 | 2.882e-06 | 0.3116 | 0.1560 | 0.1757 | 0.2299 |
| Epoch 2 | 1.6833 | 5.498e-06 | 0.4878 | 0.3157 | 0.2954 | 0.3690 |
| Epoch 3 | 1.4877 | 5.455e-06 | 0.5445 | 0.3482 | 0.5040 | 0.3982 |
| Epoch 4 | 1.3137 | 5.330e-06 | 0.5940 | 0.3503 | 0.4428 | 0.4028 |
| Epoch 5 | 1.1725 | 5.127e-06 | 0.6328 | 0.4053 | 0.4797 | 0.4454 |
| Epoch 6 | 1.1297 | 4.851e-06 | 0.6531 | 0.4396 | 0.4891 | 0.4845 |
| Epoch 7 | 1.0936 | 4.511e-06 | 0.6143 | 0.3863 | 0.4697 | 0.4660 |
| Epoch 8 | 1.0443 | 4.118e-06 | 0.6728 | 0.4476 | 0.4955 | 0.4853 |
| Epoch 9 | 1.0014 | 3.684e-06 | 0.6143 | 0.4117 | 0.5145 | 0.4544 |
| Epoch 10 | 0.9798 | 3.221e-06 | 0.6716 | 0.4485 | 0.4524 | 0.4882 |
| Epoch 11 | 0.9477 | 2.743e-06 | 0.6621 | 0.4367 | 0.4038 | 0.4814 |
| Epoch 12 | 0.9272 | 2.266e-06 | 0.6651 | 0.4382 | 0.4150 | 0.4816 |
| Epoch 13 | 0.8849 | 1.804e-06 | 0.6519 | 0.4302 | 0.4121 | 0.4723 |
| Epoch 14 | 0.8565 | 1.370e-06 | 0.6472 | 0.4331 | 0.4114 | 0.4802 |
| Epoch 15 | 0.8351 | 9.777e-07 | 0.6436 | 0.4241 | 0.3961 | 0.4803 |
| Epoch 16 | 0.8289 | 6.396e-07 | 0.6633 | 0.4480 | 0.4596 | 0.4936 |
| Epoch 17 | 0.8202 | 3.656e-07 | 0.6573 | 0.4429 | 0.4453 | 0.4909 |
| Epoch 18 | 0.8113 | 1.639e-07 | 0.6573 | 0.4424 | 0.4526 | 0.4863 |
| Epoch 19 | 0.8069 | 4.084e-08 | 0.6567 | 0.4431 | 0.4536 | 0.4872 |
| Epoch 20 | 0.8106 | 2.740e-11 | 0.6573 | 0.4432 | 0.4535 | 0.4870 |

Final epoch (Epoch 20), all 1675 validation rows: QWK (6 labels, summary.json) 0.7520 · QWK5 (AP-ROP folded onto stage 4/5, val_predictions.npz) 0.8102 · accuracy 0.6573 · macro-F1 0.4432 · balanced accuracy 0.4870 · AUC 0.7848

| final epoch | Normal | Stage 1 | Stage 2 | Stage 3 | Stage 4/5 | AP-ROP |
|---|---|---|---|---|---|---|
| recall | 0.8925 | 0.6907 | 0.6183 | 0.7137 | 0.0000 | 0.0068 |
| support | 623 | 236 | 338 | 241 | 89 | 148 |

#### Arm H · seed 1337: overview (8 folds: fold_0, fold_1, fold_2, fold_3, fold_4, fold_site, fold_loso_ostrava, fold_loso_ropvl)

| fold | train rows | val rows | images/epoch | steps/epoch | minutes | img/s | final acc | final macro-F1 | final QWK (6) | final QWK5 | AUC |
|---|---|---|---|---|---|---|---|---|---|---|---|
| fold_0 | 2447 | 1691 | 2447 | 153 | 35.2 | 32.3 | 0.6493 | 0.5643 | 0.5812 | 0.6720 | 0.8958 |
| fold_1 | 2465 | 1252 | 2465 | 155 | 32.5 | 31.7 | 0.8211 | 0.7338 | 0.8857 | 0.8833 | 0.9494 |
| fold_2 | 2524 | 1252 | 2524 | 158 | 33.3 | 32.5 | 0.7548 | 0.5842 | 0.7970 | 0.8416 | 0.8869 |
| fold_3 | 2438 | 1252 | 2438 | 153 | 32.5 | 31.9 | 0.8131 | 0.7600 | 0.9089 | 0.9125 | 0.9602 |
| fold_4 | 2434 | 1253 | 2434 | 153 | 32.6 | 32.1 | 0.8404 | 0.7473 | 0.8934 | 0.9121 | 0.9460 |
| fold_site | 2367 | 740 | 2367 | 148 | 29.4 | 32.2 | 0.8811 | 0.5629 | 0.9493 | 0.9579 | n/a |
| fold_loso_ostrava | 2383 | 4285 | 2383 | 149 | 51.6 | 32.0 | 0.8077 | 0.4034 | 0.7582 | 0.7859 | n/a |
| fold_loso_ropvl | 1404 | 1675 | 1404 | 88 | 26.6 | 30.8 | 0.6585 | 0.4432 | 0.7268 | 0.7916 | 0.8051 |

#### Arm H · seed 1337 · fold_0 (CV fold 0)

Kernel `rop-v4-dinov2-s-v4data-s1337-lr1e-4-a` · run dir `results/rop/v3/runs/dinov2_s_v4data_s1337/fold_0` · as-run manifest `results/prepared/rop_staging/_v3_fold0.csv`

Fold file `results/prepared/rop_staging/v4/fold_0.csv`: validation = 1691 photos of 297 infants (ostrava 1219, ropvl 310, shenzhen 162); training = 2447 photos of 1231 infants (ostrava 529, ropvl 1363, shenzhen 555).

| rows | normal | stage_1 | stage_2 | stage_3 | stage_4_5 | arop | total |
|---|---|---|---|---|---|---|---|
| training rows | 888 | 392 | 552 | 404 | 70 | 141 | 2447 |
| validation rows | 767 | 137 | 163 | 129 | 19 | 476 | 1691 |

- images per epoch = 2447 (the training rows: the fold file's 2453 less 6 exact duplicate(s) dropped at run time)
- steps per epoch = 153 (= ceil(2447 / (16 x 1)), batch_size x grad_accum)
- minutes 35.2 (fold start to next fold start, kernel log) · 32.3 img/s · GPU Tesla T4 · kernel log reports train 2447 val 1691

Per epoch (`metrics.json` history; lr = the first optimizer group, i.e. the lowest layer under layer-wise decay, at the end of the epoch):

| Epoch | train loss | lr | val acc | val macro-F1 | val macro-P | val macro-R |
|---|---|---|---|---|---|---|
| Epoch 1 | 1.8903 | 2.872e-06 | 0.6127 | 0.3206 | 0.3430 | 0.3302 |
| Epoch 2 | 1.7224 | 5.498e-06 | 0.6943 | 0.5435 | 0.6245 | 0.6217 |
| Epoch 3 | 1.4251 | 5.455e-06 | 0.6872 | 0.5612 | 0.6062 | 0.6334 |
| Epoch 4 | 1.2191 | 5.331e-06 | 0.7126 | 0.5829 | 0.6008 | 0.6721 |
| Epoch 5 | 1.1403 | 5.128e-06 | 0.7002 | 0.5657 | 0.5931 | 0.6731 |
| Epoch 6 | 1.0761 | 4.852e-06 | 0.7061 | 0.5936 | 0.5943 | 0.6696 |
| Epoch 7 | 1.0567 | 4.513e-06 | 0.6978 | 0.5690 | 0.5937 | 0.6558 |
| Epoch 8 | 1.0117 | 4.120e-06 | 0.6487 | 0.5607 | 0.6196 | 0.6552 |
| Epoch 9 | 0.9377 | 3.686e-06 | 0.7102 | 0.6139 | 0.6244 | 0.6783 |
| Epoch 10 | 0.9429 | 3.223e-06 | 0.6044 | 0.5291 | 0.5984 | 0.6369 |
| Epoch 11 | 0.9203 | 2.746e-06 | 0.5452 | 0.5319 | 0.6315 | 0.6208 |
| Epoch 12 | 0.8845 | 2.268e-06 | 0.6677 | 0.5785 | 0.6038 | 0.6815 |
| Epoch 13 | 0.8770 | 1.806e-06 | 0.6470 | 0.5597 | 0.6106 | 0.6836 |
| Epoch 14 | 0.8338 | 1.372e-06 | 0.6363 | 0.5373 | 0.5843 | 0.6555 |
| Epoch 15 | 0.8214 | 9.795e-07 | 0.6535 | 0.5705 | 0.6118 | 0.6691 |
| Epoch 16 | 0.7764 | 6.411e-07 | 0.6446 | 0.5542 | 0.5992 | 0.6593 |
| Epoch 17 | 0.7824 | 3.667e-07 | 0.6002 | 0.5350 | 0.6053 | 0.6355 |
| Epoch 18 | 0.7794 | 1.647e-07 | 0.6529 | 0.5703 | 0.6212 | 0.6749 |
| Epoch 19 | 0.7675 | 4.124e-08 | 0.6529 | 0.5666 | 0.6174 | 0.6629 |
| Epoch 20 | 0.7792 | 2.378e-11 | 0.6493 | 0.5643 | 0.6164 | 0.6613 |

Final epoch (Epoch 20), all 1691 validation rows: QWK (6 labels, summary.json) 0.5812 · QWK5 (AP-ROP folded onto stage 4/5, val_predictions.npz) 0.6720 · accuracy 0.6493 · macro-F1 0.5643 · balanced accuracy 0.6613 · AUC 0.8958

| final epoch | Normal | Stage 1 | Stage 2 | Stage 3 | Stage 4/5 | AP-ROP |
|---|---|---|---|---|---|---|
| recall | 0.9361 | 0.6350 | 0.5337 | 0.8527 | 0.8421 | 0.1681 |
| support | 767 | 137 | 163 | 129 | 19 | 476 |

#### Arm H · seed 1337 · fold_1 (CV fold 1)

Kernel `rop-v4-dinov2-s-v4data-s1337-lr1e-4-a` · run dir `results/rop/v3/runs/dinov2_s_v4data_s1337/fold_1` · as-run manifest `results/prepared/rop_staging/_v3_fold1.csv`

Fold file `results/prepared/rop_staging/v4/fold_1.csv`: validation = 1252 photos of 299 infants (ostrava 812, ropvl 327, shenzhen 113); training = 2465 photos of 1228 infants (ostrava 515, ropvl 1346, shenzhen 604).

| rows | normal | stage_1 | stage_2 | stage_3 | stage_4_5 | arop | total |
|---|---|---|---|---|---|---|---|
| training rows | 890 | 393 | 530 | 442 | 73 | 137 | 2465 |
| validation rows | 756 | 114 | 220 | 116 | 16 | 30 | 1252 |

- images per epoch = 2465 (the training rows: the fold file's 2469 less 4 exact duplicate(s) dropped at run time)
- steps per epoch = 155 (= ceil(2465 / (16 x 1)), batch_size x grad_accum)
- minutes 32.5 (fold start to next fold start, kernel log) · 31.7 img/s · GPU Tesla T4 · kernel log reports train 2465 val 1252

Per epoch (`metrics.json` history; lr = the first optimizer group, i.e. the lowest layer under layer-wise decay, at the end of the epoch):

| Epoch | train loss | lr | val acc | val macro-F1 | val macro-P | val macro-R |
|---|---|---|---|---|---|---|
| Epoch 1 | 1.8737 | 2.872e-06 | 0.6637 | 0.4245 | 0.4914 | 0.4160 |
| Epoch 2 | 1.6607 | 5.498e-06 | 0.7580 | 0.5166 | 0.6812 | 0.4768 |
| Epoch 3 | 1.3413 | 5.455e-06 | 0.7867 | 0.6043 | 0.7033 | 0.6123 |
| Epoch 4 | 1.2036 | 5.331e-06 | 0.7636 | 0.5457 | 0.5714 | 0.5828 |
| Epoch 5 | 1.1643 | 5.128e-06 | 0.7676 | 0.6847 | 0.6869 | 0.7105 |
| Epoch 6 | 1.0728 | 4.852e-06 | 0.7660 | 0.6363 | 0.6211 | 0.6842 |
| Epoch 7 | 1.0427 | 4.513e-06 | 0.7891 | 0.6322 | 0.6591 | 0.6602 |
| Epoch 8 | 0.9700 | 4.120e-06 | 0.8011 | 0.6497 | 0.6905 | 0.6598 |
| Epoch 9 | 0.9652 | 3.686e-06 | 0.8155 | 0.7334 | 0.7687 | 0.7099 |
| Epoch 10 | 0.9265 | 3.223e-06 | 0.8163 | 0.7421 | 0.7552 | 0.7372 |
| Epoch 11 | 0.9033 | 2.746e-06 | 0.7923 | 0.7074 | 0.7459 | 0.7136 |
| Epoch 12 | 0.8684 | 2.268e-06 | 0.8251 | 0.7292 | 0.7266 | 0.7452 |
| Epoch 13 | 0.8565 | 1.806e-06 | 0.8171 | 0.7400 | 0.7641 | 0.7427 |
| Epoch 14 | 0.8442 | 1.372e-06 | 0.8107 | 0.7132 | 0.7115 | 0.7448 |
| Epoch 15 | 0.7955 | 9.795e-07 | 0.8187 | 0.7189 | 0.7134 | 0.7387 |
| Epoch 16 | 0.7812 | 6.411e-07 | 0.8091 | 0.7223 | 0.7312 | 0.7357 |
| Epoch 17 | 0.7661 | 3.667e-07 | 0.8179 | 0.7372 | 0.7464 | 0.7387 |
| Epoch 18 | 0.7827 | 1.647e-07 | 0.8139 | 0.7190 | 0.7301 | 0.7283 |
| Epoch 19 | 0.7668 | 4.125e-08 | 0.8179 | 0.7296 | 0.7312 | 0.7438 |
| Epoch 20 | 0.7605 | 2.373e-11 | 0.8211 | 0.7338 | 0.7380 | 0.7427 |

Final epoch (Epoch 20), all 1252 validation rows: QWK (6 labels, summary.json) 0.8857 · QWK5 (AP-ROP folded onto stage 4/5, val_predictions.npz) 0.8833 · accuracy 0.8211 · macro-F1 0.7338 · balanced accuracy 0.7427 · AUC 0.9494

| final epoch | Normal | Stage 1 | Stage 2 | Stage 3 | Stage 4/5 | AP-ROP |
|---|---|---|---|---|---|---|
| recall | 0.9669 | 0.5000 | 0.5364 | 0.7155 | 0.9375 | 0.8000 |
| support | 756 | 114 | 220 | 116 | 16 | 30 |

#### Arm H · seed 1337 · fold_2 (CV fold 2)

Kernel `rop-v4-dinov2-s-v4data-s1337-lr1e-4-a` · run dir `results/rop/v3/runs/dinov2_s_v4data_s1337/fold_2` · as-run manifest `results/prepared/rop_staging/_v3_fold2.csv`

Fold file `results/prepared/rop_staging/v4/fold_2.csv`: validation = 1252 photos of 272 infants (ostrava 830, ropvl 295, shenzhen 127); training = 2524 photos of 1255 infants (ostrava 555, ropvl 1378, shenzhen 591).

| rows | normal | stage_1 | stage_2 | stage_3 | stage_4_5 | arop | total |
|---|---|---|---|---|---|---|---|
| training rows | 888 | 400 | 553 | 481 | 71 | 131 | 2524 |
| validation rows | 772 | 123 | 193 | 110 | 18 | 36 | 1252 |

- images per epoch = 2524 (the training rows: the fold file's 2528 less 4 exact duplicate(s) dropped at run time)
- steps per epoch = 158 (= ceil(2524 / (16 x 1)), batch_size x grad_accum)
- minutes 33.3 (fold start to next fold start, kernel log) · 32.5 img/s · GPU Tesla T4 · kernel log reports train 2524 val 1252

Per epoch (`metrics.json` history; lr = the first optimizer group, i.e. the lowest layer under layer-wise decay, at the end of the epoch):

| Epoch | train loss | lr | val acc | val macro-F1 | val macro-P | val macro-R |
|---|---|---|---|---|---|---|
| Epoch 1 | 1.8883 | 2.872e-06 | 0.4177 | 0.2280 | 0.4310 | 0.2646 |
| Epoch 2 | 1.6175 | 5.498e-06 | 0.7260 | 0.4476 | 0.4602 | 0.5238 |
| Epoch 3 | 1.3350 | 5.455e-06 | 0.6741 | 0.4623 | 0.4665 | 0.5584 |
| Epoch 4 | 1.2023 | 5.331e-06 | 0.7316 | 0.4692 | 0.6072 | 0.5166 |
| Epoch 5 | 1.1845 | 5.128e-06 | 0.6677 | 0.4848 | 0.5241 | 0.5511 |
| Epoch 6 | 1.1057 | 4.853e-06 | 0.6550 | 0.4499 | 0.4527 | 0.5663 |
| Epoch 7 | 1.0355 | 4.513e-06 | 0.7676 | 0.5337 | 0.6178 | 0.5713 |
| Epoch 8 | 1.0551 | 4.121e-06 | 0.6821 | 0.5219 | 0.5585 | 0.5948 |
| Epoch 9 | 0.9952 | 3.686e-06 | 0.6853 | 0.5467 | 0.5679 | 0.6090 |
| Epoch 10 | 0.9519 | 3.223e-06 | 0.7316 | 0.5742 | 0.6133 | 0.6235 |
| Epoch 11 | 0.9356 | 2.746e-06 | 0.7524 | 0.5740 | 0.6056 | 0.6311 |
| Epoch 12 | 0.8684 | 2.268e-06 | 0.7188 | 0.5494 | 0.5985 | 0.6063 |
| Epoch 13 | 0.8641 | 1.806e-06 | 0.7692 | 0.5884 | 0.6245 | 0.6475 |
| Epoch 14 | 0.8274 | 1.372e-06 | 0.7396 | 0.5576 | 0.5990 | 0.5957 |
| Epoch 15 | 0.8168 | 9.796e-07 | 0.7356 | 0.5777 | 0.6076 | 0.6437 |
| Epoch 16 | 0.8056 | 6.412e-07 | 0.7772 | 0.5826 | 0.6129 | 0.6349 |
| Epoch 17 | 0.7983 | 3.668e-07 | 0.7548 | 0.5831 | 0.6053 | 0.6434 |
| Epoch 18 | 0.7800 | 1.648e-07 | 0.7556 | 0.5799 | 0.6044 | 0.6458 |
| Epoch 19 | 0.7699 | 4.126e-08 | 0.7516 | 0.5805 | 0.6071 | 0.6449 |
| Epoch 20 | 0.7846 | 2.367e-11 | 0.7548 | 0.5842 | 0.6096 | 0.6477 |

Final epoch (Epoch 20), all 1252 validation rows: QWK (6 labels, summary.json) 0.7970 · QWK5 (AP-ROP folded onto stage 4/5, val_predictions.npz) 0.8416 · accuracy 0.7548 · macro-F1 0.5842 · balanced accuracy 0.6477 · AUC 0.8869

| final epoch | Normal | Stage 1 | Stage 2 | Stage 3 | Stage 4/5 | AP-ROP |
|---|---|---|---|---|---|---|
| recall | 0.9262 | 0.5772 | 0.4611 | 0.2273 | 0.8889 | 0.8056 |
| support | 772 | 123 | 193 | 110 | 18 | 36 |

#### Arm H · seed 1337 · fold_3 (CV fold 3)

Kernel `rop-v4-dinov2-s-v4data-s1337-lr1e-4-a` · run dir `results/rop/v3/runs/dinov2_s_v4data_s1337/fold_3` · as-run manifest `results/prepared/rop_staging/_v3_fold3.csv`

Fold file `results/prepared/rop_staging/v4/fold_3.csv`: validation = 1252 photos of 327 infants (ostrava 713, ropvl 362, shenzhen 177); training = 2438 photos of 1200 infants (ostrava 588, ropvl 1313, shenzhen 537).

| rows | normal | stage_1 | stage_2 | stage_3 | stage_4_5 | arop | total |
|---|---|---|---|---|---|---|---|
| training rows | 888 | 392 | 560 | 400 | 71 | 127 | 2438 |
| validation rows | 766 | 111 | 182 | 134 | 18 | 41 | 1252 |

- images per epoch = 2438 (the training rows: the fold file's 2442 less 4 exact duplicate(s) dropped at run time)
- steps per epoch = 153 (= ceil(2438 / (16 x 1)), batch_size x grad_accum)
- minutes 32.5 (fold start to next fold start, kernel log) · 31.9 img/s · GPU Tesla T4 · kernel log reports train 2438 val 1252

Per epoch (`metrics.json` history; lr = the first optimizer group, i.e. the lowest layer under layer-wise decay, at the end of the epoch):

| Epoch | train loss | lr | val acc | val macro-F1 | val macro-P | val macro-R |
|---|---|---|---|---|---|---|
| Epoch 1 | 1.8889 | 2.872e-06 | 0.6398 | 0.4206 | 0.5112 | 0.4365 |
| Epoch 2 | 1.7213 | 5.498e-06 | 0.6885 | 0.5616 | 0.7467 | 0.5829 |
| Epoch 3 | 1.3844 | 5.455e-06 | 0.6781 | 0.5785 | 0.5665 | 0.6484 |
| Epoch 4 | 1.2580 | 5.331e-06 | 0.7644 | 0.6759 | 0.6991 | 0.7249 |
| Epoch 5 | 1.1709 | 5.128e-06 | 0.7109 | 0.6978 | 0.7041 | 0.7365 |
| Epoch 6 | 1.0935 | 4.852e-06 | 0.6374 | 0.6519 | 0.6554 | 0.7088 |
| Epoch 7 | 1.0330 | 4.513e-06 | 0.8259 | 0.7368 | 0.7639 | 0.7292 |
| Epoch 8 | 1.0391 | 4.120e-06 | 0.7013 | 0.7066 | 0.7249 | 0.7425 |
| Epoch 9 | 0.9909 | 3.686e-06 | 0.8466 | 0.7822 | 0.7994 | 0.7820 |
| Epoch 10 | 0.9392 | 3.223e-06 | 0.8227 | 0.7376 | 0.7559 | 0.7277 |
| Epoch 11 | 0.9179 | 2.746e-06 | 0.6781 | 0.6854 | 0.6765 | 0.7390 |
| Epoch 12 | 0.8746 | 2.268e-06 | 0.7204 | 0.7160 | 0.7308 | 0.7571 |
| Epoch 13 | 0.8764 | 1.806e-06 | 0.7780 | 0.7377 | 0.7520 | 0.7563 |
| Epoch 14 | 0.8200 | 1.372e-06 | 0.8395 | 0.7469 | 0.7814 | 0.7298 |
| Epoch 15 | 0.8033 | 9.795e-07 | 0.7788 | 0.7361 | 0.7377 | 0.7634 |
| Epoch 16 | 0.7844 | 6.411e-07 | 0.8059 | 0.7586 | 0.7686 | 0.7681 |
| Epoch 17 | 0.7796 | 3.667e-07 | 0.7796 | 0.7462 | 0.7662 | 0.7552 |
| Epoch 18 | 0.7781 | 1.647e-07 | 0.8115 | 0.7601 | 0.7623 | 0.7770 |
| Epoch 19 | 0.7651 | 4.124e-08 | 0.8123 | 0.7578 | 0.7591 | 0.7716 |
| Epoch 20 | 0.7426 | 2.378e-11 | 0.8131 | 0.7600 | 0.7630 | 0.7727 |

Final epoch (Epoch 20), all 1252 validation rows: QWK (6 labels, summary.json) 0.9089 · QWK5 (AP-ROP folded onto stage 4/5, val_predictions.npz) 0.9125 · accuracy 0.8131 · macro-F1 0.7600 · balanced accuracy 0.7727 · AUC 0.9602

| final epoch | Normal | Stage 1 | Stage 2 | Stage 3 | Stage 4/5 | AP-ROP |
|---|---|---|---|---|---|---|
| recall | 0.8629 | 0.5766 | 0.7143 | 0.8507 | 0.7778 | 0.8537 |
| support | 766 | 111 | 182 | 134 | 18 | 41 |

#### Arm H · seed 1337 · fold_4 (CV fold 4)

Kernel `rop-v4-dinov2-s-v4data-s1337-lr1e-4-a` · run dir `results/rop/v3/runs/dinov2_s_v4data_s1337/fold_4` · as-run manifest `results/prepared/rop_staging/_v3_fold4.csv`

Fold file `results/prepared/rop_staging/v4/fold_4.csv`: validation = 1253 photos of 333 infants (ostrava 711, ropvl 381, shenzhen 161); training = 2434 photos of 1194 infants (ostrava 589, ropvl 1292, shenzhen 553).

| rows | normal | stage_1 | stage_2 | stage_3 | stage_4_5 | arop | total |
|---|---|---|---|---|---|---|---|
| training rows | 882 | 403 | 545 | 401 | 71 | 132 | 2434 |
| validation rows | 776 | 92 | 201 | 131 | 18 | 35 | 1253 |

- images per epoch = 2434 (the training rows: the fold file's 2440 less 6 exact duplicate(s) dropped at run time)
- steps per epoch = 153 (= ceil(2434 / (16 x 1)), batch_size x grad_accum)
- minutes 32.6 (fold start to next fold start, kernel log) · 32.1 img/s · GPU Tesla T4 · kernel log reports train 2434 val 1253

Per epoch (`metrics.json` history; lr = the first optimizer group, i.e. the lowest layer under layer-wise decay, at the end of the epoch):

| Epoch | train loss | lr | val acc | val macro-F1 | val macro-P | val macro-R |
|---|---|---|---|---|---|---|
| Epoch 1 | 1.8879 | 2.872e-06 | 0.6305 | 0.2831 | 0.2628 | 0.4077 |
| Epoch 2 | 1.7161 | 5.498e-06 | 0.7007 | 0.5076 | 0.4838 | 0.5825 |
| Epoch 3 | 1.3867 | 5.455e-06 | 0.7662 | 0.5689 | 0.5758 | 0.6238 |
| Epoch 4 | 1.2198 | 5.331e-06 | 0.6161 | 0.5051 | 0.4932 | 0.6651 |
| Epoch 5 | 1.1611 | 5.128e-06 | 0.8053 | 0.6158 | 0.6206 | 0.7029 |
| Epoch 6 | 1.1001 | 4.852e-06 | 0.7765 | 0.6215 | 0.6271 | 0.7181 |
| Epoch 7 | 1.0449 | 4.513e-06 | 0.7885 | 0.6609 | 0.6470 | 0.7179 |
| Epoch 8 | 1.0503 | 4.120e-06 | 0.8180 | 0.6785 | 0.6915 | 0.7530 |
| Epoch 9 | 0.9855 | 3.686e-06 | 0.8172 | 0.6548 | 0.6325 | 0.7321 |
| Epoch 10 | 0.9865 | 3.223e-06 | 0.8396 | 0.7271 | 0.7144 | 0.7766 |
| Epoch 11 | 0.9317 | 2.746e-06 | 0.8013 | 0.6599 | 0.6618 | 0.7353 |
| Epoch 12 | 0.8973 | 2.268e-06 | 0.8444 | 0.7250 | 0.7117 | 0.7740 |
| Epoch 13 | 0.8576 | 1.806e-06 | 0.7797 | 0.6893 | 0.6852 | 0.7601 |
| Epoch 14 | 0.8485 | 1.372e-06 | 0.8372 | 0.7072 | 0.6947 | 0.7761 |
| Epoch 15 | 0.8198 | 9.795e-07 | 0.8412 | 0.7577 | 0.7464 | 0.7965 |
| Epoch 16 | 0.7968 | 6.411e-07 | 0.8420 | 0.7352 | 0.7149 | 0.7937 |
| Epoch 17 | 0.7833 | 3.667e-07 | 0.8252 | 0.7287 | 0.7136 | 0.7920 |
| Epoch 18 | 0.7717 | 1.647e-07 | 0.8316 | 0.7345 | 0.7246 | 0.7889 |
| Epoch 19 | 0.7764 | 4.124e-08 | 0.8340 | 0.7431 | 0.7266 | 0.7921 |
| Epoch 20 | 0.7725 | 2.378e-11 | 0.8404 | 0.7473 | 0.7306 | 0.7948 |

Final epoch (Epoch 20), all 1253 validation rows: QWK (6 labels, summary.json) 0.8934 · QWK5 (AP-ROP folded onto stage 4/5, val_predictions.npz) 0.9121 · accuracy 0.8404 · macro-F1 0.7473 · balanced accuracy 0.7948 · AUC 0.9460

| final epoch | Normal | Stage 1 | Stage 2 | Stage 3 | Stage 4/5 | AP-ROP |
|---|---|---|---|---|---|---|
| recall | 0.9356 | 0.6304 | 0.5373 | 0.8626 | 0.8889 | 0.9143 |
| support | 776 | 92 | 201 | 131 | 18 | 35 |

#### Arm H · seed 1337 · fold_site (held-out source Shenzhen (the v3 site fold))

Kernel `rop-v4-dinov2-s-v4data-s1337-lr1e-4-b` · run dir `results/rop/v3/runs/dinov2_s_v4data_s1337/fold_site` · as-run manifest `results/prepared/rop_staging/_v3_foldsite.csv`

Fold file `results/prepared/rop_staging/v4/fold_site.csv`: validation = 740 photos of 397 infants (shenzhen 740); training = 2367 photos of 1131 infants (ostrava 694, ropvl 1673).

| rows | normal | stage_1 | stage_2 | stage_3 | stage_4_5 | arop | total |
|---|---|---|---|---|---|---|---|
| training rows | 893 | 409 | 529 | 280 | 89 | 167 | 2367 |
| validation rows | 234 | 89 | 163 | 254 | 0 | 0 | 740 |

- images per epoch = 2367 (the training rows: the fold file's 2369 less 2 exact duplicate(s) dropped at run time)
- steps per epoch = 148 (= ceil(2367 / (16 x 1)), batch_size x grad_accum)
- minutes 29.4 (fold start to next fold start, kernel log) · 32.2 img/s · GPU Tesla T4 · kernel log reports train 2367 val 740

Per epoch (`metrics.json` history; lr = the first optimizer group, i.e. the lowest layer under layer-wise decay, at the end of the epoch):

| Epoch | train loss | lr | val acc | val macro-F1 | val macro-P | val macro-R |
|---|---|---|---|---|---|---|
| Epoch 1 | 1.8803 | 2.873e-06 | 0.4743 | 0.2582 | 0.3141 | 0.3091 |
| Epoch 2 | 1.6897 | 5.498e-06 | 0.6054 | 0.2781 | 0.3845 | 0.3105 |
| Epoch 3 | 1.4151 | 5.455e-06 | 0.8189 | 0.5936 | 0.6191 | 0.5934 |
| Epoch 4 | 1.2028 | 5.331e-06 | 0.7905 | 0.5011 | 0.5246 | 0.4883 |
| Epoch 5 | 1.1557 | 5.128e-06 | 0.7851 | 0.5175 | 0.5341 | 0.5113 |
| Epoch 6 | 1.1418 | 4.852e-06 | 0.8176 | 0.5364 | 0.5546 | 0.5312 |
| Epoch 7 | 1.0591 | 4.513e-06 | 0.8297 | 0.5241 | 0.5298 | 0.5318 |
| Epoch 8 | 1.0222 | 4.120e-06 | 0.8189 | 0.5232 | 0.5226 | 0.5325 |
| Epoch 9 | 0.9866 | 3.686e-06 | 0.8446 | 0.5431 | 0.5490 | 0.5405 |
| Epoch 10 | 0.9383 | 3.223e-06 | 0.8459 | 0.5328 | 0.5795 | 0.5231 |
| Epoch 11 | 0.9274 | 2.746e-06 | 0.8770 | 0.5590 | 0.5852 | 0.5498 |
| Epoch 12 | 0.8956 | 2.268e-06 | 0.8635 | 0.5579 | 0.5623 | 0.5555 |
| Epoch 13 | 0.8637 | 1.806e-06 | 0.8365 | 0.5152 | 0.5590 | 0.5033 |
| Epoch 14 | 0.8412 | 1.372e-06 | 0.8797 | 0.5614 | 0.5680 | 0.5573 |
| Epoch 15 | 0.8407 | 9.794e-07 | 0.8797 | 0.5606 | 0.5743 | 0.5534 |
| Epoch 16 | 0.7835 | 6.410e-07 | 0.8824 | 0.5624 | 0.5873 | 0.5510 |
| Epoch 17 | 0.7701 | 3.667e-07 | 0.8757 | 0.5612 | 0.5736 | 0.5530 |
| Epoch 18 | 0.7624 | 1.647e-07 | 0.8770 | 0.5614 | 0.5745 | 0.5526 |
| Epoch 19 | 0.7941 | 4.122e-08 | 0.8797 | 0.5613 | 0.5721 | 0.5539 |
| Epoch 20 | 0.7703 | 2.390e-11 | 0.8811 | 0.5629 | 0.5753 | 0.5546 |

Final epoch (Epoch 20), all 740 validation rows: QWK (6 labels, summary.json) 0.9493 · QWK5 (AP-ROP folded onto stage 4/5, val_predictions.npz) 0.9579 · accuracy 0.8811 · macro-F1 0.5629 · balanced accuracy 0.5546 · AUC n/a

| final epoch | Normal | Stage 1 | Stage 2 | Stage 3 | Stage 4/5 | AP-ROP |
|---|---|---|---|---|---|---|
| recall | 0.9957 | 0.6067 | 0.8037 | 0.9213 | 0.0000 | 0.0000 |
| support | 234 | 89 | 163 | 254 | 0 | 0 |

#### Arm H · seed 1337 · fold_loso_ostrava (held-out source Ostrava (leave-one-source-out))

Kernel `rop-v4-dinov2-s-v4data-s1337-lr1e-4-b` · run dir `results/rop/v3/runs/dinov2_s_v4data_s1337/fold_loso_ostrava` · as-run manifest `results/prepared/rop_staging/_v3_foldloso_ostrava.csv`

Fold file `results/prepared/rop_staging/v4/fold_loso_ostrava.csv`: validation = 4285 photos of 148 infants (ostrava 4285); training = 2383 photos of 1379 infants (ropvl 1673, shenzhen 710).

| rows | normal | stage_1 | stage_2 | stage_3 | stage_4_5 | arop | total |
|---|---|---|---|---|---|---|---|
| training rows | 839 | 322 | 494 | 492 | 89 | 147 | 2383 |
| validation rows | 2980 | 252 | 458 | 125 | 0 | 470 | 4285 |

- images per epoch = 2383 (the training rows: the fold file's 2389 less 6 exact duplicate(s) dropped at run time)
- steps per epoch = 149 (= ceil(2383 / (16 x 1)), batch_size x grad_accum)
- minutes 51.6 (fold start to next fold start, kernel log) · 32.0 img/s · GPU Tesla T4 · kernel log reports train 2383 val 4285

Per epoch (`metrics.json` history; lr = the first optimizer group, i.e. the lowest layer under layer-wise decay, at the end of the epoch):

| Epoch | train loss | lr | val acc | val macro-F1 | val macro-P | val macro-R |
|---|---|---|---|---|---|---|
| Epoch 1 | 1.8622 | 2.873e-06 | 0.6966 | 0.1733 | 0.1871 | 0.2109 |
| Epoch 2 | 1.5745 | 5.498e-06 | 0.7939 | 0.3334 | 0.3649 | 0.3573 |
| Epoch 3 | 1.1989 | 5.455e-06 | 0.7757 | 0.3504 | 0.4944 | 0.4102 |
| Epoch 4 | 1.0759 | 5.331e-06 | 0.7456 | 0.3903 | 0.4712 | 0.4108 |
| Epoch 5 | 1.0319 | 5.128e-06 | 0.7946 | 0.3788 | 0.4852 | 0.3966 |
| Epoch 6 | 0.9649 | 4.852e-06 | 0.7629 | 0.3718 | 0.4678 | 0.4026 |
| Epoch 7 | 0.9125 | 4.513e-06 | 0.7895 | 0.3845 | 0.4540 | 0.4005 |
| Epoch 8 | 0.9091 | 4.120e-06 | 0.8037 | 0.3977 | 0.4867 | 0.3915 |
| Epoch 9 | 0.8969 | 3.686e-06 | 0.7799 | 0.3533 | 0.4157 | 0.3425 |
| Epoch 10 | 0.8760 | 3.223e-06 | 0.8016 | 0.4001 | 0.4929 | 0.4075 |
| Epoch 11 | 0.8376 | 2.746e-06 | 0.7981 | 0.3920 | 0.5009 | 0.3972 |
| Epoch 12 | 0.8036 | 2.268e-06 | 0.7965 | 0.3848 | 0.4840 | 0.3955 |
| Epoch 13 | 0.7788 | 1.806e-06 | 0.7930 | 0.3855 | 0.4806 | 0.3833 |
| Epoch 14 | 0.7487 | 1.372e-06 | 0.8044 | 0.4028 | 0.4978 | 0.4012 |
| Epoch 15 | 0.7565 | 9.795e-07 | 0.8007 | 0.3977 | 0.4987 | 0.4092 |
| Epoch 16 | 0.7102 | 6.410e-07 | 0.8105 | 0.4041 | 0.4942 | 0.4008 |
| Epoch 17 | 0.7243 | 3.667e-07 | 0.8051 | 0.3952 | 0.4906 | 0.3941 |
| Epoch 18 | 0.7370 | 1.647e-07 | 0.8084 | 0.4076 | 0.5124 | 0.4052 |
| Epoch 19 | 0.7024 | 4.122e-08 | 0.8079 | 0.4033 | 0.5021 | 0.4012 |
| Epoch 20 | 0.6992 | 2.388e-11 | 0.8077 | 0.4034 | 0.5018 | 0.4018 |

Final epoch (Epoch 20), all 4285 validation rows: QWK (6 labels, summary.json) 0.7582 · QWK5 (AP-ROP folded onto stage 4/5, val_predictions.npz) 0.7859 · accuracy 0.8077 · macro-F1 0.4034 · balanced accuracy 0.4018 · AUC n/a

| final epoch | Normal | Stage 1 | Stage 2 | Stage 3 | Stage 4/5 | AP-ROP |
|---|---|---|---|---|---|---|
| recall | 0.9812 | 0.1270 | 0.2052 | 0.3040 | 0.0000 | 0.7936 |
| support | 2980 | 252 | 458 | 125 | 0 | 470 |

#### Arm H · seed 1337 · fold_loso_ropvl (held-out source ROP-VL (leave-one-source-out))

Kernel `rop-v4-dinov2-s-v4data-s1337-lr1e-4-b` · run dir `results/rop/v3/runs/dinov2_s_v4data_s1337/fold_loso_ropvl` · as-run manifest `results/prepared/rop_staging/_v3_foldloso_ropvl.csv`

Fold file `results/prepared/rop_staging/v4/fold_loso_ropvl.csv`: validation = 1675 photos of 983 infants (ropvl 1675); training = 1404 photos of 544 infants (ostrava 694, shenzhen 710).

| rows | normal | stage_1 | stage_2 | stage_3 | stage_4_5 | arop | total |
|---|---|---|---|---|---|---|---|
| training rows | 486 | 259 | 347 | 292 | 0 | 20 | 1404 |
| validation rows | 623 | 236 | 338 | 241 | 89 | 148 | 1675 |

- validated with zero training rows: stage_4_5
- images per epoch = 1404 (the training rows: the fold file's 1408 less 4 exact duplicate(s) dropped at run time)
- steps per epoch = 88 (= ceil(1404 / (16 x 1)), batch_size x grad_accum)
- minutes 26.6 (fold start to next fold start, kernel log) · 30.8 img/s · GPU Tesla T4 · kernel log reports train 1404 val 1675

Per epoch (`metrics.json` history; lr = the first optimizer group, i.e. the lowest layer under layer-wise decay, at the end of the epoch):

| Epoch | train loss | lr | val acc | val macro-F1 | val macro-P | val macro-R |
|---|---|---|---|---|---|---|
| Epoch 1 | 1.8322 | 2.882e-06 | 0.3940 | 0.1936 | 0.1741 | 0.2626 |
| Epoch 2 | 1.6733 | 5.498e-06 | 0.4275 | 0.2366 | 0.2979 | 0.3059 |
| Epoch 3 | 1.4650 | 5.455e-06 | 0.6275 | 0.4069 | 0.5181 | 0.4475 |
| Epoch 4 | 1.2682 | 5.330e-06 | 0.5009 | 0.2990 | 0.2980 | 0.3717 |
| Epoch 5 | 1.1970 | 5.127e-06 | 0.5791 | 0.3716 | 0.3676 | 0.4419 |
| Epoch 6 | 1.1814 | 4.851e-06 | 0.6543 | 0.4351 | 0.5161 | 0.4720 |
| Epoch 7 | 1.1249 | 4.511e-06 | 0.5409 | 0.3638 | 0.3578 | 0.4171 |
| Epoch 8 | 1.0757 | 4.118e-06 | 0.6060 | 0.4109 | 0.3906 | 0.4593 |
| Epoch 9 | 1.0100 | 3.684e-06 | 0.6322 | 0.4261 | 0.3980 | 0.4765 |
| Epoch 10 | 0.9995 | 3.221e-06 | 0.6484 | 0.4374 | 0.4114 | 0.4776 |
| Epoch 11 | 0.9497 | 2.743e-06 | 0.6221 | 0.4195 | 0.3951 | 0.4734 |
| Epoch 12 | 0.9189 | 2.266e-06 | 0.6448 | 0.4359 | 0.4070 | 0.4876 |
| Epoch 13 | 0.8922 | 1.804e-06 | 0.6227 | 0.4174 | 0.3926 | 0.4707 |
| Epoch 14 | 0.8660 | 1.370e-06 | 0.6549 | 0.4445 | 0.4540 | 0.4967 |
| Epoch 15 | 0.8521 | 9.777e-07 | 0.6752 | 0.4511 | 0.4219 | 0.4928 |
| Epoch 16 | 0.8296 | 6.396e-07 | 0.6561 | 0.4451 | 0.4409 | 0.4972 |
| Epoch 17 | 0.8157 | 3.656e-07 | 0.6699 | 0.4496 | 0.4181 | 0.4981 |
| Epoch 18 | 0.8141 | 1.639e-07 | 0.6633 | 0.4458 | 0.4150 | 0.4967 |
| Epoch 19 | 0.7984 | 4.084e-08 | 0.6591 | 0.4437 | 0.4135 | 0.4937 |
| Epoch 20 | 0.8092 | 2.740e-11 | 0.6585 | 0.4432 | 0.4130 | 0.4928 |

Final epoch (Epoch 20), all 1675 validation rows: QWK (6 labels, summary.json) 0.7268 · QWK5 (AP-ROP folded onto stage 4/5, val_predictions.npz) 0.7916 · accuracy 0.6585 · macro-F1 0.4432 · balanced accuracy 0.4928 · AUC 0.8051

| final epoch | Normal | Stage 1 | Stage 2 | Stage 3 | Stage 4/5 | AP-ROP |
|---|---|---|---|---|---|---|
| recall | 0.8828 | 0.7288 | 0.5858 | 0.7593 | 0.0000 | 0.0000 |
| support | 623 | 236 | 338 | 241 | 89 | 148 |

### Arm I: DINOv2, v4 pool + v4 recipe

Run dirs: `results/rop/v3/runs/dinov2_s_v4recipe_s42`, `results/rop/v3/runs/dinov2_s_v4recipe_s1337`. Missing seeds of [42, 1337]: none.

Recipe (the as-run `config.yaml` of fold_0, seed 42; any fold whose config differs says so):

| parameter | value |
|---|---|
| model.arch | dinov2_s |
| preprocess.image_size | 392 |
| preprocess.normalize | imagenet |
| preprocess.clahe | True |
| train.lr | 0.0001 |
| train.layer_decay | 0.8 |
| train.weight_decay | 0.05 |
| train.epochs | 20 |
| train.batch_size | 16 |
| train.grad_accum | 1 (default) |
| train.label_smoothing | 0.1 |
| train.loss | ce |
| train.class_balanced | True |
| train.class_balanced_beta | 0.999 |
| train.weighted_sampler | False |
| train.strong_aug | True |
| train.site_randomization | True |
| train.amp | True |
| train.ema_decay | 0.999 |
| train.head_warmup_epochs | 1 |
| model.pretrained | True |
| model.dropout | 0.3 |
| model.drop_path | 0.1 |
| eval.early_stopping | False |
| eval.primary_metric | macro_f1 |
| train.aug_v4.enabled | True |
| train.aug_v4.rotation | 180 |
| train.aug_v4.min_crop_scale | 0.85 |
| train.aug_v4.gamma | [0.7, 1.4] |
| train.aug_v4.gamma_p | 0.5 |
| train.aug_v4.noise_sigma | 0.02 |
| train.aug_v4.noise_p | 0.3 |
| train.aug_v4.fourier_block | 3 |
| train.aug_v4.fourier_p | 0.5 |
| train.bag_loss | not set |
| train.uda | not set |
| pool (summary.json) | v4 |
| config source | configs/rop_staging_dinov2_v4recipe.yaml |

<details><summary>Every key of that as-run config.yaml</summary>

| key | value |
|---|---|
| disease | rop_staging |
| seed | 42 |
| data.num_classes | 6 |
| data.class_names | [Normal, Stage 1, Stage 2, Stage 3, Stage 4/5, AP-ROP] |
| data.manifest | results/prepared/rop_staging/_v3_fold0.csv |
| data.presplit | True |
| data.group_col | patient_id |
| preprocess.image_size | 392 |
| preprocess.circle_crop | True |
| preprocess.letterbox | True |
| preprocess.ben_graham | False |
| preprocess.clahe | True |
| preprocess.normalize | imagenet |
| preprocess.cached | True |
| preprocess.cache_root | /tmp/_cache768 |
| model.arch | dinov2_s |
| model.weights_path | None |
| model.pretrained | True |
| model.head | classification |
| model.dropout | 0.3 |
| model.drop_path | 0.1 |
| train.arm_tag | v4recipe |
| train.epochs | 20 |
| train.batch_size | 16 |
| train.lr | 0.0001 |
| train.layer_decay | 0.8 |
| train.weight_decay | 0.05 |
| train.label_smoothing | 0.1 |
| train.amp | True |
| train.strong_aug | True |
| train.site_randomization | True |
| train.loss | ce |
| train.class_balanced | True |
| train.class_balanced_beta | 0.999 |
| train.weighted_sampler | False |
| train.num_workers | 4 |
| train.ema_decay | 0.999 |
| train.head_warmup_epochs | 1 |
| train.aug_v4.enabled | True |
| train.aug_v4.rotation | 180 |
| train.aug_v4.min_crop_scale | 0.85 |
| train.aug_v4.gamma | [0.7, 1.4] |
| train.aug_v4.gamma_p | 0.5 |
| train.aug_v4.noise_sigma | 0.02 |
| train.aug_v4.noise_p | 0.3 |
| train.aug_v4.fourier_block | 3 |
| train.aug_v4.fourier_p | 0.5 |
| eval.early_stopping | False |
| eval.primary_metric | macro_f1 |
| eval.qwk_classes | [Normal, Stage 1, Stage 2, Stage 3, Stage 4/5] |
| eval.report_per_class_recall | True |
| eval.unvalidated_classes | [Stage 4/5, AP-ROP] |
| _source | configs/rop_staging_dinov2_v4recipe.yaml |

</details>

#### Arm I · seed 42: overview (8 folds: fold_0, fold_1, fold_2, fold_3, fold_4, fold_site, fold_loso_ostrava, fold_loso_ropvl)

| fold | train rows | val rows | images/epoch | steps/epoch | minutes | img/s | final acc | final macro-F1 | final QWK (6) | final QWK5 | AUC |
|---|---|---|---|---|---|---|---|---|---|---|---|
| fold_0 | 2447 | 1691 | 2447 | 153 | 39.6 | 27.9 | 0.6251 | 0.5237 | 0.5655 | 0.6774 | 0.9047 |
| fold_1 | 2465 | 1252 | 2465 | 155 | 36.8 | 30.9 | 0.8171 | 0.7362 | 0.8869 | 0.8851 | 0.9446 |
| fold_2 | 2524 | 1252 | 2524 | 158 | 37.3 | 30.6 | 0.7540 | 0.6095 | 0.8478 | 0.8726 | 0.9171 |
| fold_3 | 2438 | 1252 | 2438 | 153 | 36.3 | 31.0 | 0.7883 | 0.7479 | 0.9132 | 0.9119 | 0.9561 |
| fold_4 | 2434 | 1253 | 2434 | 153 | 36.6 | 30.9 | 0.8148 | 0.7056 | 0.8924 | 0.9099 | 0.9349 |
| fold_site | 2367 | 740 | 2367 | 148 | 29.4 | 30.6 | 0.8378 | 0.5239 | 0.9221 | 0.9380 | n/a |
| fold_loso_ostrava | 2383 | 4285 | 2383 | 149 | 48.6 | 33.1 | 0.8002 | 0.4035 | 0.7442 | 0.7709 | n/a |
| fold_loso_ropvl | 1404 | 1675 | 1404 | 88 | 25.0 | 33.1 | 0.6788 | 0.4609 | 0.7445 | 0.8071 | 0.8157 |

#### Arm I · seed 42 · fold_0 (CV fold 0)

Kernel `rop-v4-dinov2-s-v4recipe-s42-lr1e-4-a` · run dir `results/rop/v3/runs/dinov2_s_v4recipe_s42/fold_0` · as-run manifest `results/prepared/rop_staging/_v3_fold0.csv`

Fold file `results/prepared/rop_staging/v4/fold_0.csv`: validation = 1691 photos of 297 infants (ostrava 1219, ropvl 310, shenzhen 162); training = 2447 photos of 1231 infants (ostrava 529, ropvl 1363, shenzhen 555).

| rows | normal | stage_1 | stage_2 | stage_3 | stage_4_5 | arop | total |
|---|---|---|---|---|---|---|---|
| training rows | 888 | 392 | 552 | 404 | 70 | 141 | 2447 |
| validation rows | 767 | 137 | 163 | 129 | 19 | 476 | 1691 |

- images per epoch = 2447 (the training rows: the fold file's 2453 less 6 exact duplicate(s) dropped at run time)
- steps per epoch = 153 (= ceil(2447 / (16 x 1)), batch_size x grad_accum)
- epochs 1..1 train the classification head only (train.head_warmup_epochs)
- minutes 39.6 (fold start to next fold start, kernel log) · 27.9 img/s · GPU Tesla T4 · kernel log reports train 2447 val 1691

Per epoch (`metrics.json` history; lr = the first optimizer group, i.e. the lowest layer under layer-wise decay, at the end of the epoch):

| Epoch | train loss | lr | val acc | val macro-F1 | val macro-P | val macro-R |
|---|---|---|---|---|---|---|
| Epoch 1 | 2.0345 | 2.872e-06 | 0.0763 | 0.0450 | 0.0440 | 0.1226 |
| Epoch 2 | 1.9760 | 5.498e-06 | 0.1632 | 0.1107 | 0.2931 | 0.1556 |
| Epoch 3 | 1.6702 | 5.455e-06 | 0.4009 | 0.2319 | 0.5142 | 0.2517 |
| Epoch 4 | 1.4679 | 5.331e-06 | 0.5210 | 0.3587 | 0.6145 | 0.3542 |
| Epoch 5 | 1.4032 | 5.128e-06 | 0.6062 | 0.4478 | 0.5908 | 0.4683 |
| Epoch 6 | 1.3249 | 4.852e-06 | 0.6854 | 0.5617 | 0.6269 | 0.5789 |
| Epoch 7 | 1.3103 | 4.513e-06 | 0.7368 | 0.6528 | 0.6817 | 0.6662 |
| Epoch 8 | 1.2433 | 4.120e-06 | 0.7487 | 0.6671 | 0.6854 | 0.6883 |
| Epoch 9 | 1.2335 | 3.686e-06 | 0.7410 | 0.6629 | 0.6826 | 0.6921 |
| Epoch 10 | 1.2204 | 3.223e-06 | 0.7244 | 0.6452 | 0.6686 | 0.6900 |
| Epoch 11 | 1.1784 | 2.746e-06 | 0.7055 | 0.6149 | 0.6452 | 0.6737 |
| Epoch 12 | 1.1609 | 2.268e-06 | 0.6830 | 0.5805 | 0.6144 | 0.6564 |
| Epoch 13 | 1.1343 | 1.806e-06 | 0.6606 | 0.5568 | 0.5984 | 0.6520 |
| Epoch 14 | 1.1312 | 1.372e-06 | 0.6535 | 0.5491 | 0.5961 | 0.6504 |
| Epoch 15 | 1.1074 | 9.795e-07 | 0.6422 | 0.5390 | 0.5908 | 0.6522 |
| Epoch 16 | 1.0850 | 6.411e-07 | 0.6363 | 0.5331 | 0.5875 | 0.6481 |
| Epoch 17 | 1.0684 | 3.667e-07 | 0.6298 | 0.5262 | 0.5866 | 0.6457 |
| Epoch 18 | 1.0593 | 1.647e-07 | 0.6286 | 0.5247 | 0.5840 | 0.6483 |
| Epoch 19 | 1.0429 | 4.124e-08 | 0.6268 | 0.5239 | 0.5837 | 0.6485 |
| Epoch 20 | 1.0543 | 2.378e-11 | 0.6251 | 0.5237 | 0.5837 | 0.6498 |

Final epoch (Epoch 20), all 1691 validation rows: QWK (6 labels, summary.json) 0.5655 · QWK5 (AP-ROP folded onto stage 4/5, val_predictions.npz) 0.6774 · accuracy 0.6251 · macro-F1 0.5237 · balanced accuracy 0.6498 · AUC 0.9047

| final epoch | Normal | Stage 1 | Stage 2 | Stage 3 | Stage 4/5 | AP-ROP |
|---|---|---|---|---|---|---|
| recall | 0.9492 | 0.5109 | 0.5951 | 0.8837 | 0.8947 | 0.0651 |
| support | 767 | 137 | 163 | 129 | 19 | 476 |

#### Arm I · seed 42 · fold_1 (CV fold 1)

Kernel `rop-v4-dinov2-s-v4recipe-s42-lr1e-4-a` · run dir `results/rop/v3/runs/dinov2_s_v4recipe_s42/fold_1` · as-run manifest `results/prepared/rop_staging/_v3_fold1.csv`

Fold file `results/prepared/rop_staging/v4/fold_1.csv`: validation = 1252 photos of 299 infants (ostrava 812, ropvl 327, shenzhen 113); training = 2465 photos of 1228 infants (ostrava 515, ropvl 1346, shenzhen 604).

| rows | normal | stage_1 | stage_2 | stage_3 | stage_4_5 | arop | total |
|---|---|---|---|---|---|---|---|
| training rows | 890 | 393 | 530 | 442 | 73 | 137 | 2465 |
| validation rows | 756 | 114 | 220 | 116 | 16 | 30 | 1252 |

- images per epoch = 2465 (the training rows: the fold file's 2469 less 4 exact duplicate(s) dropped at run time)
- steps per epoch = 155 (= ceil(2465 / (16 x 1)), batch_size x grad_accum)
- epochs 1..1 train the classification head only (train.head_warmup_epochs)
- minutes 36.8 (fold start to next fold start, kernel log) · 30.9 img/s · GPU Tesla T4 · kernel log reports train 2465 val 1252

Per epoch (`metrics.json` history; lr = the first optimizer group, i.e. the lowest layer under layer-wise decay, at the end of the epoch):

| Epoch | train loss | lr | val acc | val macro-F1 | val macro-P | val macro-R |
|---|---|---|---|---|---|---|
| Epoch 1 | 2.0351 | 2.872e-06 | 0.1358 | 0.0686 | 0.0738 | 0.1357 |
| Epoch 2 | 1.9578 | 5.498e-06 | 0.2077 | 0.0894 | 0.1421 | 0.1425 |
| Epoch 3 | 1.6069 | 5.455e-06 | 0.6198 | 0.2980 | 0.4461 | 0.3001 |
| Epoch 4 | 1.4189 | 5.331e-06 | 0.7045 | 0.4409 | 0.4582 | 0.4559 |
| Epoch 5 | 1.3405 | 5.128e-06 | 0.7260 | 0.4716 | 0.4667 | 0.5293 |
| Epoch 6 | 1.3299 | 4.852e-06 | 0.7524 | 0.5182 | 0.6215 | 0.5853 |
| Epoch 7 | 1.2830 | 4.513e-06 | 0.7788 | 0.5840 | 0.6215 | 0.6199 |
| Epoch 8 | 1.2397 | 4.120e-06 | 0.7963 | 0.6174 | 0.6328 | 0.6454 |
| Epoch 9 | 1.2366 | 3.686e-06 | 0.8059 | 0.6605 | 0.6651 | 0.6835 |
| Epoch 10 | 1.1803 | 3.223e-06 | 0.8131 | 0.6906 | 0.6936 | 0.7020 |
| Epoch 11 | 1.1591 | 2.746e-06 | 0.8131 | 0.6915 | 0.6918 | 0.7041 |
| Epoch 12 | 1.1293 | 2.268e-06 | 0.8147 | 0.7045 | 0.7059 | 0.7130 |
| Epoch 13 | 1.0935 | 1.806e-06 | 0.8155 | 0.7124 | 0.7173 | 0.7162 |
| Epoch 14 | 1.0542 | 1.372e-06 | 0.8155 | 0.7200 | 0.7258 | 0.7235 |
| Epoch 15 | 1.0805 | 9.795e-07 | 0.8179 | 0.7271 | 0.7371 | 0.7261 |
| Epoch 16 | 1.0572 | 6.411e-07 | 0.8179 | 0.7307 | 0.7384 | 0.7321 |
| Epoch 17 | 1.0663 | 3.667e-07 | 0.8171 | 0.7305 | 0.7381 | 0.7326 |
| Epoch 18 | 1.0387 | 1.647e-07 | 0.8163 | 0.7319 | 0.7412 | 0.7330 |
| Epoch 19 | 1.0427 | 4.125e-08 | 0.8179 | 0.7345 | 0.7442 | 0.7366 |
| Epoch 20 | 1.0356 | 2.373e-11 | 0.8171 | 0.7362 | 0.7486 | 0.7378 |

Final epoch (Epoch 20), all 1252 validation rows: QWK (6 labels, summary.json) 0.8869 · QWK5 (AP-ROP folded onto stage 4/5, val_predictions.npz) 0.8851 · accuracy 0.8171 · macro-F1 0.7362 · balanced accuracy 0.7378 · AUC 0.9446

| final epoch | Normal | Stage 1 | Stage 2 | Stage 3 | Stage 4/5 | AP-ROP |
|---|---|---|---|---|---|---|
| recall | 0.9696 | 0.4912 | 0.5045 | 0.7241 | 0.9375 | 0.8000 |
| support | 756 | 114 | 220 | 116 | 16 | 30 |

#### Arm I · seed 42 · fold_2 (CV fold 2)

Kernel `rop-v4-dinov2-s-v4recipe-s42-lr1e-4-a` · run dir `results/rop/v3/runs/dinov2_s_v4recipe_s42/fold_2` · as-run manifest `results/prepared/rop_staging/_v3_fold2.csv`

Fold file `results/prepared/rop_staging/v4/fold_2.csv`: validation = 1252 photos of 272 infants (ostrava 830, ropvl 295, shenzhen 127); training = 2524 photos of 1255 infants (ostrava 555, ropvl 1378, shenzhen 591).

| rows | normal | stage_1 | stage_2 | stage_3 | stage_4_5 | arop | total |
|---|---|---|---|---|---|---|---|
| training rows | 888 | 400 | 553 | 481 | 71 | 131 | 2524 |
| validation rows | 772 | 123 | 193 | 110 | 18 | 36 | 1252 |

- images per epoch = 2524 (the training rows: the fold file's 2528 less 4 exact duplicate(s) dropped at run time)
- steps per epoch = 158 (= ceil(2524 / (16 x 1)), batch_size x grad_accum)
- epochs 1..1 train the classification head only (train.head_warmup_epochs)
- minutes 37.3 (fold start to next fold start, kernel log) · 30.6 img/s · GPU Tesla T4 · kernel log reports train 2524 val 1252

Per epoch (`metrics.json` history; lr = the first optimizer group, i.e. the lowest layer under layer-wise decay, at the end of the epoch):

| Epoch | train loss | lr | val acc | val macro-F1 | val macro-P | val macro-R |
|---|---|---|---|---|---|---|
| Epoch 1 | 2.0550 | 2.872e-06 | 0.1262 | 0.0683 | 0.0839 | 0.1314 |
| Epoch 2 | 1.9454 | 5.498e-06 | 0.1981 | 0.1215 | 0.3237 | 0.1572 |
| Epoch 3 | 1.5806 | 5.455e-06 | 0.5359 | 0.3177 | 0.3765 | 0.3217 |
| Epoch 4 | 1.4613 | 5.331e-06 | 0.6342 | 0.3483 | 0.3958 | 0.4096 |
| Epoch 5 | 1.4015 | 5.128e-06 | 0.6589 | 0.3805 | 0.5631 | 0.4917 |
| Epoch 6 | 1.3488 | 4.853e-06 | 0.7005 | 0.4828 | 0.6084 | 0.5634 |
| Epoch 7 | 1.2953 | 4.513e-06 | 0.7220 | 0.5270 | 0.5995 | 0.6011 |
| Epoch 8 | 1.2461 | 4.121e-06 | 0.7332 | 0.5464 | 0.5898 | 0.6083 |
| Epoch 9 | 1.2530 | 3.686e-06 | 0.7420 | 0.5626 | 0.6016 | 0.6188 |
| Epoch 10 | 1.2200 | 3.223e-06 | 0.7508 | 0.5816 | 0.6095 | 0.6422 |
| Epoch 11 | 1.1952 | 2.746e-06 | 0.7444 | 0.5736 | 0.5993 | 0.6319 |
| Epoch 12 | 1.1642 | 2.268e-06 | 0.7444 | 0.5735 | 0.5959 | 0.6291 |
| Epoch 13 | 1.1290 | 1.806e-06 | 0.7500 | 0.5805 | 0.6005 | 0.6352 |
| Epoch 14 | 1.1338 | 1.372e-06 | 0.7516 | 0.5849 | 0.6044 | 0.6385 |
| Epoch 15 | 1.1017 | 9.796e-07 | 0.7500 | 0.5818 | 0.5918 | 0.6425 |
| Epoch 16 | 1.1166 | 6.412e-07 | 0.7404 | 0.5816 | 0.5885 | 0.6481 |
| Epoch 17 | 1.0903 | 3.668e-07 | 0.7468 | 0.5884 | 0.5835 | 0.6568 |
| Epoch 18 | 1.0986 | 1.648e-07 | 0.7492 | 0.5935 | 0.5856 | 0.6619 |
| Epoch 19 | 1.0815 | 4.126e-08 | 0.7540 | 0.6017 | 0.5903 | 0.6684 |
| Epoch 20 | 1.0613 | 2.367e-11 | 0.7540 | 0.6095 | 0.5993 | 0.6735 |

Final epoch (Epoch 20), all 1252 validation rows: QWK (6 labels, summary.json) 0.8478 · QWK5 (AP-ROP folded onto stage 4/5, val_predictions.npz) 0.8726 · accuracy 0.7540 · macro-F1 0.6095 · balanced accuracy 0.6735 · AUC 0.9171

| final epoch | Normal | Stage 1 | Stage 2 | Stage 3 | Stage 4/5 | AP-ROP |
|---|---|---|---|---|---|---|
| recall | 0.8912 | 0.4472 | 0.6425 | 0.2545 | 0.8889 | 0.9167 |
| support | 772 | 123 | 193 | 110 | 18 | 36 |

#### Arm I · seed 42 · fold_3 (CV fold 3)

Kernel `rop-v4-dinov2-s-v4recipe-s42-lr1e-4-a` · run dir `results/rop/v3/runs/dinov2_s_v4recipe_s42/fold_3` · as-run manifest `results/prepared/rop_staging/_v3_fold3.csv`

Fold file `results/prepared/rop_staging/v4/fold_3.csv`: validation = 1252 photos of 327 infants (ostrava 713, ropvl 362, shenzhen 177); training = 2438 photos of 1200 infants (ostrava 588, ropvl 1313, shenzhen 537).

| rows | normal | stage_1 | stage_2 | stage_3 | stage_4_5 | arop | total |
|---|---|---|---|---|---|---|---|
| training rows | 888 | 392 | 560 | 400 | 71 | 127 | 2438 |
| validation rows | 766 | 111 | 182 | 134 | 18 | 41 | 1252 |

- images per epoch = 2438 (the training rows: the fold file's 2442 less 4 exact duplicate(s) dropped at run time)
- steps per epoch = 153 (= ceil(2438 / (16 x 1)), batch_size x grad_accum)
- epochs 1..1 train the classification head only (train.head_warmup_epochs)
- minutes 36.3 (fold start to next fold start, kernel log) · 31.0 img/s · GPU Tesla T4 · kernel log reports train 2438 val 1252

Per epoch (`metrics.json` history; lr = the first optimizer group, i.e. the lowest layer under layer-wise decay, at the end of the epoch):

| Epoch | train loss | lr | val acc | val macro-F1 | val macro-P | val macro-R |
|---|---|---|---|---|---|---|
| Epoch 1 | 2.0552 | 2.872e-06 | 0.1278 | 0.0831 | 0.0939 | 0.1498 |
| Epoch 2 | 1.9590 | 5.498e-06 | 0.2157 | 0.1227 | 0.1775 | 0.1592 |
| Epoch 3 | 1.6506 | 5.455e-06 | 0.5895 | 0.2729 | 0.4962 | 0.2763 |
| Epoch 4 | 1.4682 | 5.331e-06 | 0.7021 | 0.4691 | 0.5123 | 0.4566 |
| Epoch 5 | 1.3957 | 5.128e-06 | 0.7524 | 0.5466 | 0.6555 | 0.5923 |
| Epoch 6 | 1.3450 | 4.852e-06 | 0.7987 | 0.6519 | 0.6849 | 0.6878 |
| Epoch 7 | 1.2816 | 4.513e-06 | 0.8235 | 0.6999 | 0.7009 | 0.7193 |
| Epoch 8 | 1.2761 | 4.120e-06 | 0.8299 | 0.7348 | 0.7319 | 0.7431 |
| Epoch 9 | 1.2549 | 3.686e-06 | 0.8331 | 0.7398 | 0.7333 | 0.7500 |
| Epoch 10 | 1.2172 | 3.223e-06 | 0.8363 | 0.7546 | 0.7437 | 0.7669 |
| Epoch 11 | 1.1809 | 2.746e-06 | 0.8275 | 0.7476 | 0.7365 | 0.7606 |
| Epoch 12 | 1.1855 | 2.268e-06 | 0.8283 | 0.7558 | 0.7470 | 0.7673 |
| Epoch 13 | 1.1589 | 1.806e-06 | 0.8235 | 0.7468 | 0.7415 | 0.7555 |
| Epoch 14 | 1.1449 | 1.372e-06 | 0.8171 | 0.7437 | 0.7383 | 0.7543 |
| Epoch 15 | 1.0999 | 9.795e-07 | 0.8067 | 0.7439 | 0.7348 | 0.7598 |
| Epoch 16 | 1.1200 | 6.411e-07 | 0.8003 | 0.7442 | 0.7349 | 0.7636 |
| Epoch 17 | 1.0857 | 3.667e-07 | 0.7979 | 0.7486 | 0.7417 | 0.7679 |
| Epoch 18 | 1.0841 | 1.647e-07 | 0.7915 | 0.7488 | 0.7443 | 0.7691 |
| Epoch 19 | 1.0623 | 4.124e-08 | 0.7923 | 0.7504 | 0.7462 | 0.7709 |
| Epoch 20 | 1.0864 | 2.378e-11 | 0.7883 | 0.7479 | 0.7445 | 0.7684 |

Final epoch (Epoch 20), all 1252 validation rows: QWK (6 labels, summary.json) 0.9132 · QWK5 (AP-ROP folded onto stage 4/5, val_predictions.npz) 0.9119 · accuracy 0.7883 · macro-F1 0.7479 · balanced accuracy 0.7684 · AUC 0.9561

| final epoch | Normal | Stage 1 | Stage 2 | Stage 3 | Stage 4/5 | AP-ROP |
|---|---|---|---|---|---|---|
| recall | 0.8394 | 0.5315 | 0.6484 | 0.8731 | 0.8889 | 0.8293 |
| support | 766 | 111 | 182 | 134 | 18 | 41 |

#### Arm I · seed 42 · fold_4 (CV fold 4)

Kernel `rop-v4-dinov2-s-v4recipe-s42-lr1e-4-a` · run dir `results/rop/v3/runs/dinov2_s_v4recipe_s42/fold_4` · as-run manifest `results/prepared/rop_staging/_v3_fold4.csv`

Fold file `results/prepared/rop_staging/v4/fold_4.csv`: validation = 1253 photos of 333 infants (ostrava 711, ropvl 381, shenzhen 161); training = 2434 photos of 1194 infants (ostrava 589, ropvl 1292, shenzhen 553).

| rows | normal | stage_1 | stage_2 | stage_3 | stage_4_5 | arop | total |
|---|---|---|---|---|---|---|---|
| training rows | 882 | 403 | 545 | 401 | 71 | 132 | 2434 |
| validation rows | 776 | 92 | 201 | 131 | 18 | 35 | 1253 |

- images per epoch = 2434 (the training rows: the fold file's 2440 less 6 exact duplicate(s) dropped at run time)
- steps per epoch = 153 (= ceil(2434 / (16 x 1)), batch_size x grad_accum)
- epochs 1..1 train the classification head only (train.head_warmup_epochs)
- minutes 36.6 (fold start to next fold start, kernel log) · 30.9 img/s · GPU Tesla T4 · kernel log reports train 2434 val 1253

Per epoch (`metrics.json` history; lr = the first optimizer group, i.e. the lowest layer under layer-wise decay, at the end of the epoch):

| Epoch | train loss | lr | val acc | val macro-F1 | val macro-P | val macro-R |
|---|---|---|---|---|---|---|
| Epoch 1 | 2.0367 | 2.872e-06 | 0.1317 | 0.0772 | 0.0890 | 0.1437 |
| Epoch 2 | 1.9701 | 5.498e-06 | 0.2346 | 0.1203 | 0.1837 | 0.1691 |
| Epoch 3 | 1.6809 | 5.455e-06 | 0.5954 | 0.2802 | 0.4055 | 0.2917 |
| Epoch 4 | 1.5033 | 5.331e-06 | 0.6824 | 0.3750 | 0.4352 | 0.4175 |
| Epoch 5 | 1.3924 | 5.128e-06 | 0.7167 | 0.4344 | 0.5938 | 0.5093 |
| Epoch 6 | 1.3395 | 4.852e-06 | 0.7749 | 0.5595 | 0.6160 | 0.6089 |
| Epoch 7 | 1.2827 | 4.513e-06 | 0.7909 | 0.5926 | 0.6096 | 0.6417 |
| Epoch 8 | 1.2583 | 4.120e-06 | 0.8053 | 0.6168 | 0.6079 | 0.6763 |
| Epoch 9 | 1.2336 | 3.686e-06 | 0.8053 | 0.6281 | 0.6117 | 0.6904 |
| Epoch 10 | 1.2052 | 3.223e-06 | 0.8061 | 0.6367 | 0.6185 | 0.6995 |
| Epoch 11 | 1.1948 | 2.746e-06 | 0.8140 | 0.6564 | 0.6353 | 0.7204 |
| Epoch 12 | 1.1771 | 2.268e-06 | 0.8132 | 0.6667 | 0.6537 | 0.7207 |
| Epoch 13 | 1.1532 | 1.806e-06 | 0.8093 | 0.6667 | 0.6526 | 0.7218 |
| Epoch 14 | 1.1170 | 1.372e-06 | 0.8125 | 0.6794 | 0.6637 | 0.7328 |
| Epoch 15 | 1.0948 | 9.795e-07 | 0.8140 | 0.6862 | 0.6616 | 0.7436 |
| Epoch 16 | 1.1045 | 6.411e-07 | 0.8196 | 0.6956 | 0.6672 | 0.7525 |
| Epoch 17 | 1.0724 | 3.667e-07 | 0.8132 | 0.6954 | 0.6667 | 0.7528 |
| Epoch 18 | 1.0703 | 1.647e-07 | 0.8156 | 0.7050 | 0.6768 | 0.7569 |
| Epoch 19 | 1.0580 | 4.124e-08 | 0.8172 | 0.7054 | 0.6751 | 0.7592 |
| Epoch 20 | 1.0534 | 2.378e-11 | 0.8148 | 0.7056 | 0.6770 | 0.7575 |

Final epoch (Epoch 20), all 1253 validation rows: QWK (6 labels, summary.json) 0.8924 · QWK5 (AP-ROP folded onto stage 4/5, val_predictions.npz) 0.9099 · accuracy 0.8148 · macro-F1 0.7056 · balanced accuracy 0.7575 · AUC 0.9349

| final epoch | Normal | Stage 1 | Stage 2 | Stage 3 | Stage 4/5 | AP-ROP |
|---|---|---|---|---|---|---|
| recall | 0.9111 | 0.5217 | 0.5473 | 0.8473 | 0.8889 | 0.8286 |
| support | 776 | 92 | 201 | 131 | 18 | 35 |

#### Arm I · seed 42 · fold_site (held-out source Shenzhen (the v3 site fold))

Kernel `rop-v4-dinov2-s-v4recipe-s42-lr1e-4-b` · run dir `results/rop/v3/runs/dinov2_s_v4recipe_s42/fold_site` · as-run manifest `results/prepared/rop_staging/_v3_foldsite.csv`

Fold file `results/prepared/rop_staging/v4/fold_site.csv`: validation = 740 photos of 397 infants (shenzhen 740); training = 2367 photos of 1131 infants (ostrava 694, ropvl 1673).

| rows | normal | stage_1 | stage_2 | stage_3 | stage_4_5 | arop | total |
|---|---|---|---|---|---|---|---|
| training rows | 893 | 409 | 529 | 280 | 89 | 167 | 2367 |
| validation rows | 234 | 89 | 163 | 254 | 0 | 0 | 740 |

- images per epoch = 2367 (the training rows: the fold file's 2369 less 2 exact duplicate(s) dropped at run time)
- steps per epoch = 148 (= ceil(2367 / (16 x 1)), batch_size x grad_accum)
- epochs 1..1 train the classification head only (train.head_warmup_epochs)
- minutes 29.4 (fold start to next fold start, kernel log) · 30.6 img/s · GPU Tesla T4 · kernel log reports train 2367 val 740

Per epoch (`metrics.json` history; lr = the first optimizer group, i.e. the lowest layer under layer-wise decay, at the end of the epoch):

| Epoch | train loss | lr | val acc | val macro-F1 | val macro-P | val macro-R |
|---|---|---|---|---|---|---|
| Epoch 1 | 1.9932 | 2.873e-06 | 0.1757 | 0.1393 | 0.1730 | 0.2094 |
| Epoch 2 | 1.9330 | 5.498e-06 | 0.2311 | 0.1874 | 0.2388 | 0.2649 |
| Epoch 3 | 1.6748 | 5.455e-06 | 0.4176 | 0.3413 | 0.3612 | 0.4126 |
| Epoch 4 | 1.4527 | 5.331e-06 | 0.4568 | 0.2541 | 0.2997 | 0.3086 |
| Epoch 5 | 1.3845 | 5.128e-06 | 0.4919 | 0.2985 | 0.4041 | 0.3439 |
| Epoch 6 | 1.3173 | 4.852e-06 | 0.6122 | 0.4078 | 0.4759 | 0.4132 |
| Epoch 7 | 1.2672 | 4.513e-06 | 0.7108 | 0.4755 | 0.5329 | 0.4572 |
| Epoch 8 | 1.2372 | 4.120e-06 | 0.7649 | 0.4906 | 0.5456 | 0.4730 |
| Epoch 9 | 1.2495 | 3.686e-06 | 0.7851 | 0.4882 | 0.5440 | 0.4744 |
| Epoch 10 | 1.1940 | 3.223e-06 | 0.8041 | 0.4924 | 0.5463 | 0.4815 |
| Epoch 11 | 1.1669 | 2.746e-06 | 0.8230 | 0.5107 | 0.5554 | 0.4981 |
| Epoch 12 | 1.1514 | 2.268e-06 | 0.8338 | 0.5116 | 0.5591 | 0.5013 |
| Epoch 13 | 1.1309 | 1.806e-06 | 0.8338 | 0.5131 | 0.5596 | 0.5022 |
| Epoch 14 | 1.1075 | 1.372e-06 | 0.8365 | 0.5171 | 0.5613 | 0.5055 |
| Epoch 15 | 1.0861 | 9.794e-07 | 0.8324 | 0.5148 | 0.5594 | 0.5024 |
| Epoch 16 | 1.0558 | 6.410e-07 | 0.8284 | 0.5121 | 0.5569 | 0.4993 |
| Epoch 17 | 1.0403 | 3.667e-07 | 0.8284 | 0.5121 | 0.5569 | 0.4993 |
| Epoch 18 | 1.0377 | 1.647e-07 | 0.8284 | 0.5130 | 0.5543 | 0.5002 |
| Epoch 19 | 1.0532 | 4.122e-08 | 0.8351 | 0.5196 | 0.5586 | 0.5066 |
| Epoch 20 | 1.0426 | 2.390e-11 | 0.8378 | 0.5239 | 0.5611 | 0.5104 |

Final epoch (Epoch 20), all 740 validation rows: QWK (6 labels, summary.json) 0.9221 · QWK5 (AP-ROP folded onto stage 4/5, val_predictions.npz) 0.9380 · accuracy 0.8378 · macro-F1 0.5239 · balanced accuracy 0.5104 · AUC n/a

| final epoch | Normal | Stage 1 | Stage 2 | Stage 3 | Stage 4/5 | AP-ROP |
|---|---|---|---|---|---|---|
| recall | 0.9915 | 0.4270 | 0.7423 | 0.9016 | 0.0000 | 0.0000 |
| support | 234 | 89 | 163 | 254 | 0 | 0 |

#### Arm I · seed 42 · fold_loso_ostrava (held-out source Ostrava (leave-one-source-out))

Kernel `rop-v4-dinov2-s-v4recipe-s42-lr1e-4-b` · run dir `results/rop/v3/runs/dinov2_s_v4recipe_s42/fold_loso_ostrava` · as-run manifest `results/prepared/rop_staging/_v3_foldloso_ostrava.csv`

Fold file `results/prepared/rop_staging/v4/fold_loso_ostrava.csv`: validation = 4285 photos of 148 infants (ostrava 4285); training = 2383 photos of 1379 infants (ropvl 1673, shenzhen 710).

| rows | normal | stage_1 | stage_2 | stage_3 | stage_4_5 | arop | total |
|---|---|---|---|---|---|---|---|
| training rows | 839 | 322 | 494 | 492 | 89 | 147 | 2383 |
| validation rows | 2980 | 252 | 458 | 125 | 0 | 470 | 4285 |

- images per epoch = 2383 (the training rows: the fold file's 2389 less 6 exact duplicate(s) dropped at run time)
- steps per epoch = 149 (= ceil(2383 / (16 x 1)), batch_size x grad_accum)
- epochs 1..1 train the classification head only (train.head_warmup_epochs)
- minutes 48.6 (fold start to next fold start, kernel log) · 33.1 img/s · GPU Tesla T4 · kernel log reports train 2383 val 4285

Per epoch (`metrics.json` history; lr = the first optimizer group, i.e. the lowest layer under layer-wise decay, at the end of the epoch):

| Epoch | train loss | lr | val acc | val macro-F1 | val macro-P | val macro-R |
|---|---|---|---|---|---|---|
| Epoch 1 | 2.0080 | 2.873e-06 | 0.0901 | 0.0530 | 0.0527 | 0.1690 |
| Epoch 2 | 1.8655 | 5.498e-06 | 0.1830 | 0.0949 | 0.1350 | 0.1540 |
| Epoch 3 | 1.4676 | 5.455e-06 | 0.7004 | 0.2453 | 0.2887 | 0.2345 |
| Epoch 4 | 1.3020 | 5.331e-06 | 0.7673 | 0.2764 | 0.3241 | 0.2911 |
| Epoch 5 | 1.2614 | 5.128e-06 | 0.7848 | 0.3014 | 0.4771 | 0.3220 |
| Epoch 6 | 1.1618 | 4.852e-06 | 0.7979 | 0.3432 | 0.4806 | 0.3524 |
| Epoch 7 | 1.1537 | 4.513e-06 | 0.8054 | 0.3694 | 0.4885 | 0.3749 |
| Epoch 8 | 1.1306 | 4.120e-06 | 0.8110 | 0.3951 | 0.5099 | 0.3963 |
| Epoch 9 | 1.0925 | 3.686e-06 | 0.8126 | 0.4017 | 0.4991 | 0.4042 |
| Epoch 10 | 1.0760 | 3.223e-06 | 0.8126 | 0.4082 | 0.4972 | 0.4106 |
| Epoch 11 | 1.0542 | 2.746e-06 | 0.8135 | 0.4156 | 0.5024 | 0.4190 |
| Epoch 12 | 1.0274 | 2.268e-06 | 0.8105 | 0.4119 | 0.4928 | 0.4175 |
| Epoch 13 | 1.0167 | 1.806e-06 | 0.8082 | 0.4096 | 0.4910 | 0.4164 |
| Epoch 14 | 0.9961 | 1.372e-06 | 0.8075 | 0.4101 | 0.4905 | 0.4176 |
| Epoch 15 | 0.9956 | 9.795e-07 | 0.8063 | 0.4070 | 0.4863 | 0.4155 |
| Epoch 16 | 0.9701 | 6.410e-07 | 0.8044 | 0.4064 | 0.4830 | 0.4148 |
| Epoch 17 | 0.9732 | 3.667e-07 | 0.8037 | 0.4064 | 0.4826 | 0.4150 |
| Epoch 18 | 0.9655 | 1.647e-07 | 0.8019 | 0.4049 | 0.4807 | 0.4146 |
| Epoch 19 | 0.9552 | 4.122e-08 | 0.8016 | 0.4048 | 0.4807 | 0.4142 |
| Epoch 20 | 0.9429 | 2.388e-11 | 0.8002 | 0.4035 | 0.4793 | 0.4129 |

Final epoch (Epoch 20), all 4285 validation rows: QWK (6 labels, summary.json) 0.7442 · QWK5 (AP-ROP folded onto stage 4/5, val_predictions.npz) 0.7709 · accuracy 0.8002 · macro-F1 0.4035 · balanced accuracy 0.4129 · AUC n/a

| final epoch | Normal | Stage 1 | Stage 2 | Stage 3 | Stage 4/5 | AP-ROP |
|---|---|---|---|---|---|---|
| recall | 0.9685 | 0.1706 | 0.1812 | 0.3680 | 0.0000 | 0.7894 |
| support | 2980 | 252 | 458 | 125 | 0 | 470 |

#### Arm I · seed 42 · fold_loso_ropvl (held-out source ROP-VL (leave-one-source-out))

Kernel `rop-v4-dinov2-s-v4recipe-s42-lr1e-4-b` · run dir `results/rop/v3/runs/dinov2_s_v4recipe_s42/fold_loso_ropvl` · as-run manifest `results/prepared/rop_staging/_v3_foldloso_ropvl.csv`

Fold file `results/prepared/rop_staging/v4/fold_loso_ropvl.csv`: validation = 1675 photos of 983 infants (ropvl 1675); training = 1404 photos of 544 infants (ostrava 694, shenzhen 710).

| rows | normal | stage_1 | stage_2 | stage_3 | stage_4_5 | arop | total |
|---|---|---|---|---|---|---|---|
| training rows | 486 | 259 | 347 | 292 | 0 | 20 | 1404 |
| validation rows | 623 | 236 | 338 | 241 | 89 | 148 | 1675 |

- validated with zero training rows: stage_4_5
- images per epoch = 1404 (the training rows: the fold file's 1408 less 4 exact duplicate(s) dropped at run time)
- steps per epoch = 88 (= ceil(1404 / (16 x 1)), batch_size x grad_accum)
- epochs 1..1 train the classification head only (train.head_warmup_epochs)
- minutes 25.0 (fold start to next fold start, kernel log) · 33.1 img/s · GPU Tesla T4 · kernel log reports train 1404 val 1675

Per epoch (`metrics.json` history; lr = the first optimizer group, i.e. the lowest layer under layer-wise decay, at the end of the epoch):

| Epoch | train loss | lr | val acc | val macro-F1 | val macro-P | val macro-R |
|---|---|---|---|---|---|---|
| Epoch 1 | 2.2170 | 2.882e-06 | 0.1588 | 0.0940 | 0.0951 | 0.1303 |
| Epoch 2 | 1.9776 | 5.498e-06 | 0.1815 | 0.0973 | 0.1121 | 0.1436 |
| Epoch 3 | 1.8035 | 5.455e-06 | 0.2209 | 0.0994 | 0.1525 | 0.1701 |
| Epoch 4 | 1.6262 | 5.330e-06 | 0.2884 | 0.1368 | 0.2200 | 0.2035 |
| Epoch 5 | 1.4872 | 5.127e-06 | 0.4239 | 0.2297 | 0.3073 | 0.2792 |
| Epoch 6 | 1.4724 | 4.851e-06 | 0.5063 | 0.2937 | 0.3767 | 0.3393 |
| Epoch 7 | 1.3643 | 4.511e-06 | 0.5576 | 0.3478 | 0.4182 | 0.3880 |
| Epoch 8 | 1.3437 | 4.118e-06 | 0.5934 | 0.3985 | 0.4393 | 0.4223 |
| Epoch 9 | 1.2958 | 3.684e-06 | 0.6167 | 0.4276 | 0.4454 | 0.4404 |
| Epoch 10 | 1.2833 | 3.221e-06 | 0.6352 | 0.4415 | 0.4500 | 0.4529 |
| Epoch 11 | 1.2683 | 2.743e-06 | 0.6490 | 0.4487 | 0.4556 | 0.4642 |
| Epoch 12 | 1.2607 | 2.266e-06 | 0.6537 | 0.4467 | 0.4541 | 0.4666 |
| Epoch 13 | 1.2211 | 1.804e-06 | 0.6663 | 0.4552 | 0.4680 | 0.4787 |
| Epoch 14 | 1.2090 | 1.370e-06 | 0.6687 | 0.4531 | 0.4690 | 0.4807 |
| Epoch 15 | 1.2172 | 9.777e-07 | 0.6687 | 0.4534 | 0.4783 | 0.4819 |
| Epoch 16 | 1.1646 | 6.396e-07 | 0.6710 | 0.4525 | 0.4721 | 0.4845 |
| Epoch 17 | 1.1707 | 3.656e-07 | 0.6734 | 0.4538 | 0.4699 | 0.4873 |
| Epoch 18 | 1.1435 | 1.639e-07 | 0.6728 | 0.4530 | 0.4758 | 0.4873 |
| Epoch 19 | 1.1678 | 4.084e-08 | 0.6764 | 0.4590 | 0.4820 | 0.4934 |
| Epoch 20 | 1.1395 | 2.740e-11 | 0.6788 | 0.4609 | 0.4849 | 0.4962 |

Final epoch (Epoch 20), all 1675 validation rows: QWK (6 labels, summary.json) 0.7445 · QWK5 (AP-ROP folded onto stage 4/5, val_predictions.npz) 0.8071 · accuracy 0.6788 · macro-F1 0.4609 · balanced accuracy 0.4962 · AUC 0.8157

| final epoch | Normal | Stage 1 | Stage 2 | Stage 3 | Stage 4/5 | AP-ROP |
|---|---|---|---|---|---|---|
| recall | 0.9486 | 0.4958 | 0.6598 | 0.8257 | 0.0000 | 0.0473 |
| support | 623 | 236 | 338 | 241 | 89 | 148 |

#### Arm I · seed 1337: overview (8 folds: fold_0, fold_1, fold_2, fold_3, fold_4, fold_site, fold_loso_ostrava, fold_loso_ropvl)

| fold | train rows | val rows | images/epoch | steps/epoch | minutes | img/s | final acc | final macro-F1 | final QWK (6) | final QWK5 | AUC |
|---|---|---|---|---|---|---|---|---|---|---|---|
| fold_0 | 2447 | 1691 | 2447 | 153 | 34.1 | 32.7 | 0.6410 | 0.5322 | 0.6067 | 0.7038 | 0.9143 |
| fold_1 | 2465 | 1252 | 2465 | 155 | 31.7 | 34.2 | 0.8131 | 0.7275 | 0.8778 | 0.8767 | 0.9417 |
| fold_2 | 2524 | 1252 | 2524 | 158 | 32.4 | 34.6 | 0.7292 | 0.5693 | 0.7711 | 0.8237 | 0.9152 |
| fold_3 | 2438 | 1252 | 2438 | 153 | 31.7 | 34.1 | 0.7979 | 0.7459 | 0.9143 | 0.9151 | 0.9509 |
| fold_4 | 2434 | 1253 | 2434 | 153 | 31.6 | 34.2 | 0.7933 | 0.6785 | 0.8590 | 0.8877 | 0.9307 |
| fold_site | 2367 | 740 | 2367 | 148 | 29.6 | 32.5 | 0.8392 | 0.5175 | 0.9343 | 0.9447 | n/a |
| fold_loso_ostrava | 2383 | 4285 | 2383 | 149 | 52.9 | 33.6 | 0.8079 | 0.4087 | 0.7621 | 0.7873 | n/a |
| fold_loso_ropvl | 1404 | 1675 | 1404 | 88 | 26.8 | 33.3 | 0.6722 | 0.4632 | 0.7180 | 0.7852 | 0.7731 |

#### Arm I · seed 1337 · fold_0 (CV fold 0)

Kernel `rop-v4-dinov2-s-v4recipe-s1337-lr1e-4-a` · run dir `results/rop/v3/runs/dinov2_s_v4recipe_s1337/fold_0` · as-run manifest `results/prepared/rop_staging/_v3_fold0.csv`

Fold file `results/prepared/rop_staging/v4/fold_0.csv`: validation = 1691 photos of 297 infants (ostrava 1219, ropvl 310, shenzhen 162); training = 2447 photos of 1231 infants (ostrava 529, ropvl 1363, shenzhen 555).

| rows | normal | stage_1 | stage_2 | stage_3 | stage_4_5 | arop | total |
|---|---|---|---|---|---|---|---|
| training rows | 888 | 392 | 552 | 404 | 70 | 141 | 2447 |
| validation rows | 767 | 137 | 163 | 129 | 19 | 476 | 1691 |

- images per epoch = 2447 (the training rows: the fold file's 2453 less 6 exact duplicate(s) dropped at run time)
- steps per epoch = 153 (= ceil(2447 / (16 x 1)), batch_size x grad_accum)
- epochs 1..1 train the classification head only (train.head_warmup_epochs)
- minutes 34.1 (fold start to next fold start, kernel log) · 32.7 img/s · GPU Tesla T4 · kernel log reports train 2447 val 1691

Per epoch (`metrics.json` history; lr = the first optimizer group, i.e. the lowest layer under layer-wise decay, at the end of the epoch):

| Epoch | train loss | lr | val acc | val macro-F1 | val macro-P | val macro-R |
|---|---|---|---|---|---|---|
| Epoch 1 | 2.0370 | 2.872e-06 | 0.3578 | 0.2067 | 0.2179 | 0.2475 |
| Epoch 2 | 1.9579 | 5.498e-06 | 0.4240 | 0.2331 | 0.4267 | 0.2906 |
| Epoch 3 | 1.6758 | 5.455e-06 | 0.5328 | 0.3014 | 0.5204 | 0.3457 |
| Epoch 4 | 1.5166 | 5.331e-06 | 0.6517 | 0.4690 | 0.5205 | 0.4650 |
| Epoch 5 | 1.3837 | 5.128e-06 | 0.7274 | 0.5876 | 0.6306 | 0.5712 |
| Epoch 6 | 1.2954 | 4.852e-06 | 0.7564 | 0.6303 | 0.6465 | 0.6372 |
| Epoch 7 | 1.2907 | 4.513e-06 | 0.7688 | 0.6561 | 0.6700 | 0.6569 |
| Epoch 8 | 1.2795 | 4.120e-06 | 0.7705 | 0.6625 | 0.6658 | 0.6819 |
| Epoch 9 | 1.2236 | 3.686e-06 | 0.7569 | 0.6422 | 0.6420 | 0.6814 |
| Epoch 10 | 1.2314 | 3.223e-06 | 0.7487 | 0.6287 | 0.6284 | 0.6797 |
| Epoch 11 | 1.1953 | 2.746e-06 | 0.7439 | 0.6219 | 0.6217 | 0.6797 |
| Epoch 12 | 1.1652 | 2.268e-06 | 0.7339 | 0.6092 | 0.6106 | 0.6781 |
| Epoch 13 | 1.1482 | 1.806e-06 | 0.7126 | 0.5874 | 0.5989 | 0.6656 |
| Epoch 14 | 1.1227 | 1.372e-06 | 0.6978 | 0.5729 | 0.5931 | 0.6580 |
| Epoch 15 | 1.0919 | 9.795e-07 | 0.6878 | 0.5702 | 0.5942 | 0.6624 |
| Epoch 16 | 1.0897 | 6.411e-07 | 0.6759 | 0.5616 | 0.5943 | 0.6591 |
| Epoch 17 | 1.0859 | 3.667e-07 | 0.6629 | 0.5504 | 0.5894 | 0.6536 |
| Epoch 18 | 1.0729 | 1.647e-07 | 0.6535 | 0.5428 | 0.5825 | 0.6580 |
| Epoch 19 | 1.0525 | 4.124e-08 | 0.6464 | 0.5361 | 0.5789 | 0.6536 |
| Epoch 20 | 1.0408 | 2.378e-11 | 0.6410 | 0.5322 | 0.5789 | 0.6528 |

Final epoch (Epoch 20), all 1691 validation rows: QWK (6 labels, summary.json) 0.6067 · QWK5 (AP-ROP folded onto stage 4/5, val_predictions.npz) 0.7038 · accuracy 0.6410 · macro-F1 0.5322 · balanced accuracy 0.6528 · AUC 0.9143

| final epoch | Normal | Stage 1 | Stage 2 | Stage 3 | Stage 4/5 | AP-ROP |
|---|---|---|---|---|---|---|
| recall | 0.9374 | 0.5036 | 0.6442 | 0.8527 | 0.8421 | 0.1366 |
| support | 767 | 137 | 163 | 129 | 19 | 476 |

#### Arm I · seed 1337 · fold_1 (CV fold 1)

Kernel `rop-v4-dinov2-s-v4recipe-s1337-lr1e-4-a` · run dir `results/rop/v3/runs/dinov2_s_v4recipe_s1337/fold_1` · as-run manifest `results/prepared/rop_staging/_v3_fold1.csv`

Fold file `results/prepared/rop_staging/v4/fold_1.csv`: validation = 1252 photos of 299 infants (ostrava 812, ropvl 327, shenzhen 113); training = 2465 photos of 1228 infants (ostrava 515, ropvl 1346, shenzhen 604).

| rows | normal | stage_1 | stage_2 | stage_3 | stage_4_5 | arop | total |
|---|---|---|---|---|---|---|---|
| training rows | 890 | 393 | 530 | 442 | 73 | 137 | 2465 |
| validation rows | 756 | 114 | 220 | 116 | 16 | 30 | 1252 |

- images per epoch = 2465 (the training rows: the fold file's 2469 less 4 exact duplicate(s) dropped at run time)
- steps per epoch = 155 (= ceil(2465 / (16 x 1)), batch_size x grad_accum)
- epochs 1..1 train the classification head only (train.head_warmup_epochs)
- minutes 31.7 (fold start to next fold start, kernel log) · 34.2 img/s · GPU Tesla T4 · kernel log reports train 2465 val 1252

Per epoch (`metrics.json` history; lr = the first optimizer group, i.e. the lowest layer under layer-wise decay, at the end of the epoch):

| Epoch | train loss | lr | val acc | val macro-F1 | val macro-P | val macro-R |
|---|---|---|---|---|---|---|
| Epoch 1 | 2.0202 | 2.872e-06 | 0.3131 | 0.1558 | 0.2126 | 0.1932 |
| Epoch 2 | 1.9378 | 5.498e-06 | 0.5447 | 0.2159 | 0.2259 | 0.2714 |
| Epoch 3 | 1.6056 | 5.455e-06 | 0.6757 | 0.3073 | 0.4526 | 0.3080 |
| Epoch 4 | 1.4340 | 5.331e-06 | 0.7037 | 0.4722 | 0.5747 | 0.4413 |
| Epoch 5 | 1.3835 | 5.128e-06 | 0.7292 | 0.5050 | 0.5575 | 0.4919 |
| Epoch 6 | 1.3111 | 4.852e-06 | 0.7540 | 0.5805 | 0.6049 | 0.5750 |
| Epoch 7 | 1.2844 | 4.513e-06 | 0.7740 | 0.6091 | 0.6200 | 0.6149 |
| Epoch 8 | 1.2545 | 4.120e-06 | 0.7812 | 0.6227 | 0.6297 | 0.6299 |
| Epoch 9 | 1.2050 | 3.686e-06 | 0.7891 | 0.6462 | 0.6449 | 0.6641 |
| Epoch 10 | 1.1796 | 3.223e-06 | 0.7955 | 0.6610 | 0.6601 | 0.6761 |
| Epoch 11 | 1.1334 | 2.746e-06 | 0.7971 | 0.6774 | 0.6765 | 0.6874 |
| Epoch 12 | 1.1372 | 2.268e-06 | 0.7971 | 0.6777 | 0.6738 | 0.6905 |
| Epoch 13 | 1.1151 | 1.806e-06 | 0.8075 | 0.6983 | 0.6965 | 0.7081 |
| Epoch 14 | 1.1116 | 1.372e-06 | 0.8083 | 0.7006 | 0.6965 | 0.7126 |
| Epoch 15 | 1.0700 | 9.795e-07 | 0.8051 | 0.7001 | 0.6966 | 0.7108 |
| Epoch 16 | 1.0571 | 6.411e-07 | 0.8083 | 0.7050 | 0.7022 | 0.7157 |
| Epoch 17 | 1.0589 | 3.667e-07 | 0.8075 | 0.7110 | 0.7097 | 0.7219 |
| Epoch 18 | 1.0389 | 1.647e-07 | 0.8083 | 0.7162 | 0.7145 | 0.7281 |
| Epoch 19 | 1.0249 | 4.125e-08 | 0.8123 | 0.7242 | 0.7254 | 0.7340 |
| Epoch 20 | 1.0391 | 2.373e-11 | 0.8131 | 0.7275 | 0.7318 | 0.7347 |

Final epoch (Epoch 20), all 1252 validation rows: QWK (6 labels, summary.json) 0.8778 · QWK5 (AP-ROP folded onto stage 4/5, val_predictions.npz) 0.8767 · accuracy 0.8131 · macro-F1 0.7275 · balanced accuracy 0.7347 · AUC 0.9417

| final epoch | Normal | Stage 1 | Stage 2 | Stage 3 | Stage 4/5 | AP-ROP |
|---|---|---|---|---|---|---|
| recall | 0.9590 | 0.4123 | 0.5409 | 0.7586 | 0.9375 | 0.8000 |
| support | 756 | 114 | 220 | 116 | 16 | 30 |

#### Arm I · seed 1337 · fold_2 (CV fold 2)

Kernel `rop-v4-dinov2-s-v4recipe-s1337-lr1e-4-a` · run dir `results/rop/v3/runs/dinov2_s_v4recipe_s1337/fold_2` · as-run manifest `results/prepared/rop_staging/_v3_fold2.csv`

Fold file `results/prepared/rop_staging/v4/fold_2.csv`: validation = 1252 photos of 272 infants (ostrava 830, ropvl 295, shenzhen 127); training = 2524 photos of 1255 infants (ostrava 555, ropvl 1378, shenzhen 591).

| rows | normal | stage_1 | stage_2 | stage_3 | stage_4_5 | arop | total |
|---|---|---|---|---|---|---|---|
| training rows | 888 | 400 | 553 | 481 | 71 | 131 | 2524 |
| validation rows | 772 | 123 | 193 | 110 | 18 | 36 | 1252 |

- images per epoch = 2524 (the training rows: the fold file's 2528 less 4 exact duplicate(s) dropped at run time)
- steps per epoch = 158 (= ceil(2524 / (16 x 1)), batch_size x grad_accum)
- epochs 1..1 train the classification head only (train.head_warmup_epochs)
- minutes 32.4 (fold start to next fold start, kernel log) · 34.6 img/s · GPU Tesla T4 · kernel log reports train 2524 val 1252

Per epoch (`metrics.json` history; lr = the first optimizer group, i.e. the lowest layer under layer-wise decay, at the end of the epoch):

| Epoch | train loss | lr | val acc | val macro-F1 | val macro-P | val macro-R |
|---|---|---|---|---|---|---|
| Epoch 1 | 2.0552 | 2.872e-06 | 0.3634 | 0.1772 | 0.2117 | 0.2334 |
| Epoch 2 | 1.9565 | 5.498e-06 | 0.5727 | 0.2284 | 0.2320 | 0.2843 |
| Epoch 3 | 1.6251 | 5.455e-06 | 0.6581 | 0.3121 | 0.3451 | 0.3274 |
| Epoch 4 | 1.4686 | 5.331e-06 | 0.6877 | 0.3563 | 0.4252 | 0.4164 |
| Epoch 5 | 1.4033 | 5.128e-06 | 0.6973 | 0.3888 | 0.4354 | 0.4622 |
| Epoch 6 | 1.3535 | 4.853e-06 | 0.7292 | 0.4777 | 0.5259 | 0.5386 |
| Epoch 7 | 1.3050 | 4.513e-06 | 0.7476 | 0.5293 | 0.5835 | 0.5799 |
| Epoch 8 | 1.2670 | 4.121e-06 | 0.7500 | 0.5364 | 0.5719 | 0.6020 |
| Epoch 9 | 1.2330 | 3.686e-06 | 0.7492 | 0.5344 | 0.5609 | 0.6006 |
| Epoch 10 | 1.2206 | 3.223e-06 | 0.7492 | 0.5362 | 0.5567 | 0.6089 |
| Epoch 11 | 1.1776 | 2.746e-06 | 0.7532 | 0.5468 | 0.5658 | 0.6227 |
| Epoch 12 | 1.1894 | 2.268e-06 | 0.7508 | 0.5590 | 0.5791 | 0.6334 |
| Epoch 13 | 1.1624 | 1.806e-06 | 0.7508 | 0.5605 | 0.5804 | 0.6383 |
| Epoch 14 | 1.1444 | 1.372e-06 | 0.7508 | 0.5646 | 0.5817 | 0.6461 |
| Epoch 15 | 1.1099 | 9.796e-07 | 0.7428 | 0.5616 | 0.5761 | 0.6455 |
| Epoch 16 | 1.0921 | 6.412e-07 | 0.7340 | 0.5624 | 0.5725 | 0.6452 |
| Epoch 17 | 1.0873 | 3.668e-07 | 0.7340 | 0.5639 | 0.5738 | 0.6468 |
| Epoch 18 | 1.0739 | 1.648e-07 | 0.7300 | 0.5662 | 0.5761 | 0.6494 |
| Epoch 19 | 1.0701 | 4.126e-08 | 0.7292 | 0.5685 | 0.5776 | 0.6512 |
| Epoch 20 | 1.0630 | 2.367e-11 | 0.7292 | 0.5693 | 0.5720 | 0.6543 |

Final epoch (Epoch 20), all 1252 validation rows: QWK (6 labels, summary.json) 0.7711 · QWK5 (AP-ROP folded onto stage 4/5, val_predictions.npz) 0.8237 · accuracy 0.7292 · macro-F1 0.5693 · balanced accuracy 0.6543 · AUC 0.9152

| final epoch | Normal | Stage 1 | Stage 2 | Stage 3 | Stage 4/5 | AP-ROP |
|---|---|---|---|---|---|---|
| recall | 0.8860 | 0.4715 | 0.4715 | 0.2909 | 0.9444 | 0.8611 |
| support | 772 | 123 | 193 | 110 | 18 | 36 |

#### Arm I · seed 1337 · fold_3 (CV fold 3)

Kernel `rop-v4-dinov2-s-v4recipe-s1337-lr1e-4-a` · run dir `results/rop/v3/runs/dinov2_s_v4recipe_s1337/fold_3` · as-run manifest `results/prepared/rop_staging/_v3_fold3.csv`

Fold file `results/prepared/rop_staging/v4/fold_3.csv`: validation = 1252 photos of 327 infants (ostrava 713, ropvl 362, shenzhen 177); training = 2438 photos of 1200 infants (ostrava 588, ropvl 1313, shenzhen 537).

| rows | normal | stage_1 | stage_2 | stage_3 | stage_4_5 | arop | total |
|---|---|---|---|---|---|---|---|
| training rows | 888 | 392 | 560 | 400 | 71 | 127 | 2438 |
| validation rows | 766 | 111 | 182 | 134 | 18 | 41 | 1252 |

- images per epoch = 2438 (the training rows: the fold file's 2442 less 4 exact duplicate(s) dropped at run time)
- steps per epoch = 153 (= ceil(2438 / (16 x 1)), batch_size x grad_accum)
- epochs 1..1 train the classification head only (train.head_warmup_epochs)
- minutes 31.7 (fold start to next fold start, kernel log) · 34.1 img/s · GPU Tesla T4 · kernel log reports train 2438 val 1252

Per epoch (`metrics.json` history; lr = the first optimizer group, i.e. the lowest layer under layer-wise decay, at the end of the epoch):

| Epoch | train loss | lr | val acc | val macro-F1 | val macro-P | val macro-R |
|---|---|---|---|---|---|---|
| Epoch 1 | 2.0530 | 2.872e-06 | 0.3043 | 0.1559 | 0.2036 | 0.1984 |
| Epoch 2 | 1.9521 | 5.498e-06 | 0.5423 | 0.2210 | 0.2238 | 0.2670 |
| Epoch 3 | 1.6338 | 5.455e-06 | 0.6821 | 0.3598 | 0.3829 | 0.3726 |
| Epoch 4 | 1.4930 | 5.331e-06 | 0.7181 | 0.4356 | 0.4738 | 0.4838 |
| Epoch 5 | 1.4468 | 5.128e-06 | 0.7580 | 0.5316 | 0.5529 | 0.5819 |
| Epoch 6 | 1.3206 | 4.852e-06 | 0.7923 | 0.6190 | 0.6312 | 0.6481 |
| Epoch 7 | 1.3043 | 4.513e-06 | 0.8139 | 0.6664 | 0.6747 | 0.6791 |
| Epoch 8 | 1.2792 | 4.120e-06 | 0.8219 | 0.6780 | 0.6785 | 0.6928 |
| Epoch 9 | 1.2577 | 3.686e-06 | 0.8267 | 0.6899 | 0.6894 | 0.6997 |
| Epoch 10 | 1.2101 | 3.223e-06 | 0.8251 | 0.7052 | 0.7033 | 0.7121 |
| Epoch 11 | 1.1986 | 2.746e-06 | 0.8211 | 0.7113 | 0.7044 | 0.7205 |
| Epoch 12 | 1.1855 | 2.268e-06 | 0.8211 | 0.7197 | 0.7134 | 0.7280 |
| Epoch 13 | 1.1471 | 1.806e-06 | 0.8187 | 0.7260 | 0.7177 | 0.7362 |
| Epoch 14 | 1.1220 | 1.372e-06 | 0.8163 | 0.7313 | 0.7263 | 0.7387 |
| Epoch 15 | 1.1041 | 9.795e-07 | 0.8179 | 0.7432 | 0.7405 | 0.7491 |
| Epoch 16 | 1.0668 | 6.411e-07 | 0.8203 | 0.7471 | 0.7444 | 0.7532 |
| Epoch 17 | 1.0793 | 3.667e-07 | 0.8211 | 0.7516 | 0.7478 | 0.7595 |
| Epoch 18 | 1.0729 | 1.647e-07 | 0.8155 | 0.7551 | 0.7498 | 0.7667 |
| Epoch 19 | 1.0842 | 4.124e-08 | 0.8083 | 0.7540 | 0.7507 | 0.7653 |
| Epoch 20 | 1.0725 | 2.378e-11 | 0.7979 | 0.7459 | 0.7464 | 0.7567 |

Final epoch (Epoch 20), all 1252 validation rows: QWK (6 labels, summary.json) 0.9143 · QWK5 (AP-ROP folded onto stage 4/5, val_predictions.npz) 0.9151 · accuracy 0.7979 · macro-F1 0.7459 · balanced accuracy 0.7567 · AUC 0.9509

| final epoch | Normal | Stage 1 | Stage 2 | Stage 3 | Stage 4/5 | AP-ROP |
|---|---|---|---|---|---|---|
| recall | 0.8551 | 0.4775 | 0.6813 | 0.8881 | 0.8333 | 0.8049 |
| support | 766 | 111 | 182 | 134 | 18 | 41 |

#### Arm I · seed 1337 · fold_4 (CV fold 4)

Kernel `rop-v4-dinov2-s-v4recipe-s1337-lr1e-4-a` · run dir `results/rop/v3/runs/dinov2_s_v4recipe_s1337/fold_4` · as-run manifest `results/prepared/rop_staging/_v3_fold4.csv`

Fold file `results/prepared/rop_staging/v4/fold_4.csv`: validation = 1253 photos of 333 infants (ostrava 711, ropvl 381, shenzhen 161); training = 2434 photos of 1194 infants (ostrava 589, ropvl 1292, shenzhen 553).

| rows | normal | stage_1 | stage_2 | stage_3 | stage_4_5 | arop | total |
|---|---|---|---|---|---|---|---|
| training rows | 882 | 403 | 545 | 401 | 71 | 132 | 2434 |
| validation rows | 776 | 92 | 201 | 131 | 18 | 35 | 1253 |

- images per epoch = 2434 (the training rows: the fold file's 2440 less 6 exact duplicate(s) dropped at run time)
- steps per epoch = 153 (= ceil(2434 / (16 x 1)), batch_size x grad_accum)
- epochs 1..1 train the classification head only (train.head_warmup_epochs)
- minutes 31.6 (fold start to next fold start, kernel log) · 34.2 img/s · GPU Tesla T4 · kernel log reports train 2434 val 1253

Per epoch (`metrics.json` history; lr = the first optimizer group, i.e. the lowest layer under layer-wise decay, at the end of the epoch):

| Epoch | train loss | lr | val acc | val macro-F1 | val macro-P | val macro-R |
|---|---|---|---|---|---|---|
| Epoch 1 | 2.0439 | 2.872e-06 | 0.3232 | 0.1695 | 0.2291 | 0.2177 |
| Epoch 2 | 1.9629 | 5.498e-06 | 0.5834 | 0.2705 | 0.3560 | 0.2945 |
| Epoch 3 | 1.6301 | 5.455e-06 | 0.6975 | 0.3767 | 0.3749 | 0.3938 |
| Epoch 4 | 1.4699 | 5.331e-06 | 0.7079 | 0.3905 | 0.4479 | 0.4594 |
| Epoch 5 | 1.3951 | 5.128e-06 | 0.7502 | 0.4853 | 0.5112 | 0.5895 |
| Epoch 6 | 1.3329 | 4.852e-06 | 0.7877 | 0.5675 | 0.5651 | 0.6718 |
| Epoch 7 | 1.2664 | 4.513e-06 | 0.8156 | 0.6277 | 0.6150 | 0.7156 |
| Epoch 8 | 1.2571 | 4.120e-06 | 0.8188 | 0.6376 | 0.6186 | 0.7318 |
| Epoch 9 | 1.2441 | 3.686e-06 | 0.8244 | 0.6463 | 0.6272 | 0.7269 |
| Epoch 10 | 1.2079 | 3.223e-06 | 0.8204 | 0.6463 | 0.6235 | 0.7229 |
| Epoch 11 | 1.1890 | 2.746e-06 | 0.8093 | 0.6428 | 0.6206 | 0.7169 |
| Epoch 12 | 1.1862 | 2.268e-06 | 0.8045 | 0.6439 | 0.6171 | 0.7216 |
| Epoch 13 | 1.1303 | 1.806e-06 | 0.7989 | 0.6457 | 0.6185 | 0.7241 |
| Epoch 14 | 1.1130 | 1.372e-06 | 0.7949 | 0.6462 | 0.6184 | 0.7269 |
| Epoch 15 | 1.0958 | 9.795e-07 | 0.7941 | 0.6528 | 0.6264 | 0.7298 |
| Epoch 16 | 1.0980 | 6.411e-07 | 0.7941 | 0.6584 | 0.6316 | 0.7358 |
| Epoch 17 | 1.0818 | 3.667e-07 | 0.7933 | 0.6579 | 0.6323 | 0.7336 |
| Epoch 18 | 1.0788 | 1.647e-07 | 0.7909 | 0.6647 | 0.6408 | 0.7376 |
| Epoch 19 | 1.0577 | 4.124e-08 | 0.7933 | 0.6714 | 0.6475 | 0.7471 |
| Epoch 20 | 1.0678 | 2.378e-11 | 0.7933 | 0.6785 | 0.6524 | 0.7486 |

Final epoch (Epoch 20), all 1253 validation rows: QWK (6 labels, summary.json) 0.8590 · QWK5 (AP-ROP folded onto stage 4/5, val_predictions.npz) 0.8877 · accuracy 0.7933 · macro-F1 0.6785 · balanced accuracy 0.7486 · AUC 0.9307

| final epoch | Normal | Stage 1 | Stage 2 | Stage 3 | Stage 4/5 | AP-ROP |
|---|---|---|---|---|---|---|
| recall | 0.8918 | 0.5543 | 0.4577 | 0.8702 | 0.8889 | 0.8286 |
| support | 776 | 92 | 201 | 131 | 18 | 35 |

#### Arm I · seed 1337 · fold_site (held-out source Shenzhen (the v3 site fold))

Kernel `rop-v4-dinov2-s-v4recipe-s1337-lr1e-4-b` · run dir `results/rop/v3/runs/dinov2_s_v4recipe_s1337/fold_site` · as-run manifest `results/prepared/rop_staging/_v3_foldsite.csv`

Fold file `results/prepared/rop_staging/v4/fold_site.csv`: validation = 740 photos of 397 infants (shenzhen 740); training = 2367 photos of 1131 infants (ostrava 694, ropvl 1673).

| rows | normal | stage_1 | stage_2 | stage_3 | stage_4_5 | arop | total |
|---|---|---|---|---|---|---|---|
| training rows | 893 | 409 | 529 | 280 | 89 | 167 | 2367 |
| validation rows | 234 | 89 | 163 | 254 | 0 | 0 | 740 |

- images per epoch = 2367 (the training rows: the fold file's 2369 less 2 exact duplicate(s) dropped at run time)
- steps per epoch = 148 (= ceil(2367 / (16 x 1)), batch_size x grad_accum)
- epochs 1..1 train the classification head only (train.head_warmup_epochs)
- minutes 29.6 (fold start to next fold start, kernel log) · 32.5 img/s · GPU Tesla T4 · kernel log reports train 2367 val 740

Per epoch (`metrics.json` history; lr = the first optimizer group, i.e. the lowest layer under layer-wise decay, at the end of the epoch):

| Epoch | train loss | lr | val acc | val macro-F1 | val macro-P | val macro-R |
|---|---|---|---|---|---|---|
| Epoch 1 | 1.9874 | 2.873e-06 | 0.3257 | 0.2323 | 0.2843 | 0.2093 |
| Epoch 2 | 1.9351 | 5.498e-06 | 0.4662 | 0.2972 | 0.3194 | 0.2985 |
| Epoch 3 | 1.6073 | 5.455e-06 | 0.5784 | 0.3585 | 0.3523 | 0.3726 |
| Epoch 4 | 1.4561 | 5.331e-06 | 0.6338 | 0.3352 | 0.4847 | 0.3451 |
| Epoch 5 | 1.3796 | 5.128e-06 | 0.6797 | 0.3850 | 0.4639 | 0.3809 |
| Epoch 6 | 1.3309 | 4.852e-06 | 0.7243 | 0.4137 | 0.5111 | 0.4115 |
| Epoch 7 | 1.2688 | 4.513e-06 | 0.7635 | 0.4389 | 0.5314 | 0.4383 |
| Epoch 8 | 1.2607 | 4.120e-06 | 0.7878 | 0.4611 | 0.5480 | 0.4586 |
| Epoch 9 | 1.2023 | 3.686e-06 | 0.8000 | 0.4687 | 0.5531 | 0.4660 |
| Epoch 10 | 1.2047 | 3.223e-06 | 0.8095 | 0.4823 | 0.5604 | 0.4759 |
| Epoch 11 | 1.1572 | 2.746e-06 | 0.8189 | 0.4932 | 0.5662 | 0.4849 |
| Epoch 12 | 1.1454 | 2.268e-06 | 0.8243 | 0.5009 | 0.5687 | 0.4908 |
| Epoch 13 | 1.1066 | 1.806e-06 | 0.8311 | 0.5042 | 0.5694 | 0.4942 |
| Epoch 14 | 1.0888 | 1.372e-06 | 0.8351 | 0.5072 | 0.5703 | 0.4970 |
| Epoch 15 | 1.0793 | 9.794e-07 | 0.8378 | 0.5136 | 0.5722 | 0.5016 |
| Epoch 16 | 1.0450 | 6.410e-07 | 0.8405 | 0.5163 | 0.5732 | 0.5041 |
| Epoch 17 | 1.0224 | 3.667e-07 | 0.8405 | 0.5171 | 0.5725 | 0.5046 |
| Epoch 18 | 1.0367 | 1.647e-07 | 0.8392 | 0.5171 | 0.5680 | 0.5045 |
| Epoch 19 | 1.0296 | 4.122e-08 | 0.8378 | 0.5168 | 0.5679 | 0.5038 |
| Epoch 20 | 1.0329 | 2.390e-11 | 0.8392 | 0.5175 | 0.5686 | 0.5045 |

Final epoch (Epoch 20), all 740 validation rows: QWK (6 labels, summary.json) 0.9343 · QWK5 (AP-ROP folded onto stage 4/5, val_predictions.npz) 0.9447 · accuracy 0.8392 · macro-F1 0.5175 · balanced accuracy 0.5045 · AUC n/a

| final epoch | Normal | Stage 1 | Stage 2 | Stage 3 | Stage 4/5 | AP-ROP |
|---|---|---|---|---|---|---|
| recall | 1.0000 | 0.3820 | 0.7117 | 0.9331 | 0.0000 | 0.0000 |
| support | 234 | 89 | 163 | 254 | 0 | 0 |

#### Arm I · seed 1337 · fold_loso_ostrava (held-out source Ostrava (leave-one-source-out))

Kernel `rop-v4-dinov2-s-v4recipe-s1337-lr1e-4-b` · run dir `results/rop/v3/runs/dinov2_s_v4recipe_s1337/fold_loso_ostrava` · as-run manifest `results/prepared/rop_staging/_v3_foldloso_ostrava.csv`

Fold file `results/prepared/rop_staging/v4/fold_loso_ostrava.csv`: validation = 4285 photos of 148 infants (ostrava 4285); training = 2383 photos of 1379 infants (ropvl 1673, shenzhen 710).

| rows | normal | stage_1 | stage_2 | stage_3 | stage_4_5 | arop | total |
|---|---|---|---|---|---|---|---|
| training rows | 839 | 322 | 494 | 492 | 89 | 147 | 2383 |
| validation rows | 2980 | 252 | 458 | 125 | 0 | 470 | 4285 |

- images per epoch = 2383 (the training rows: the fold file's 2389 less 6 exact duplicate(s) dropped at run time)
- steps per epoch = 149 (= ceil(2383 / (16 x 1)), batch_size x grad_accum)
- epochs 1..1 train the classification head only (train.head_warmup_epochs)
- minutes 52.9 (fold start to next fold start, kernel log) · 33.6 img/s · GPU Tesla T4 · kernel log reports train 2383 val 4285

Per epoch (`metrics.json` history; lr = the first optimizer group, i.e. the lowest layer under layer-wise decay, at the end of the epoch):

| Epoch | train loss | lr | val acc | val macro-F1 | val macro-P | val macro-R |
|---|---|---|---|---|---|---|
| Epoch 1 | 1.9900 | 2.873e-06 | 0.3669 | 0.1859 | 0.2511 | 0.3158 |
| Epoch 2 | 1.8738 | 5.498e-06 | 0.6030 | 0.2553 | 0.3088 | 0.3278 |
| Epoch 3 | 1.4669 | 5.455e-06 | 0.7505 | 0.3521 | 0.4927 | 0.3601 |
| Epoch 4 | 1.3160 | 5.331e-06 | 0.7762 | 0.3762 | 0.6139 | 0.3829 |
| Epoch 5 | 1.2614 | 5.128e-06 | 0.7858 | 0.3284 | 0.5047 | 0.3419 |
| Epoch 6 | 1.1913 | 4.852e-06 | 0.7981 | 0.3567 | 0.4665 | 0.3685 |
| Epoch 7 | 1.1565 | 4.513e-06 | 0.8093 | 0.3881 | 0.4941 | 0.3975 |
| Epoch 8 | 1.1223 | 4.120e-06 | 0.8133 | 0.3965 | 0.4882 | 0.4122 |
| Epoch 9 | 1.0940 | 3.686e-06 | 0.8170 | 0.4063 | 0.5030 | 0.4233 |
| Epoch 10 | 1.0863 | 3.223e-06 | 0.8189 | 0.4130 | 0.5033 | 0.4316 |
| Epoch 11 | 1.0583 | 2.746e-06 | 0.8168 | 0.4086 | 0.4939 | 0.4263 |
| Epoch 12 | 1.0392 | 2.268e-06 | 0.8145 | 0.4067 | 0.4897 | 0.4255 |
| Epoch 13 | 1.0057 | 1.806e-06 | 0.8131 | 0.4056 | 0.4898 | 0.4226 |
| Epoch 14 | 1.0049 | 1.372e-06 | 0.8140 | 0.4108 | 0.4940 | 0.4237 |
| Epoch 15 | 0.9901 | 9.795e-07 | 0.8140 | 0.4119 | 0.4918 | 0.4243 |
| Epoch 16 | 0.9922 | 6.410e-07 | 0.8131 | 0.4133 | 0.4928 | 0.4253 |
| Epoch 17 | 0.9642 | 3.667e-07 | 0.8119 | 0.4131 | 0.4929 | 0.4246 |
| Epoch 18 | 0.9584 | 1.647e-07 | 0.8105 | 0.4116 | 0.4909 | 0.4233 |
| Epoch 19 | 0.9561 | 4.122e-08 | 0.8093 | 0.4090 | 0.4878 | 0.4207 |
| Epoch 20 | 0.9631 | 2.388e-11 | 0.8079 | 0.4087 | 0.4890 | 0.4207 |

Final epoch (Epoch 20), all 4285 validation rows: QWK (6 labels, summary.json) 0.7621 · QWK5 (AP-ROP folded onto stage 4/5, val_predictions.npz) 0.7873 · accuracy 0.8079 · macro-F1 0.4087 · balanced accuracy 0.4207 · AUC n/a

| final epoch | Normal | Stage 1 | Stage 2 | Stage 3 | Stage 4/5 | AP-ROP |
|---|---|---|---|---|---|---|
| recall | 0.9688 | 0.1349 | 0.2074 | 0.3600 | 0.0000 | 0.8532 |
| support | 2980 | 252 | 458 | 125 | 0 | 470 |

#### Arm I · seed 1337 · fold_loso_ropvl (held-out source ROP-VL (leave-one-source-out))

Kernel `rop-v4-dinov2-s-v4recipe-s1337-lr1e-4-b` · run dir `results/rop/v3/runs/dinov2_s_v4recipe_s1337/fold_loso_ropvl` · as-run manifest `results/prepared/rop_staging/_v3_foldloso_ropvl.csv`

Fold file `results/prepared/rop_staging/v4/fold_loso_ropvl.csv`: validation = 1675 photos of 983 infants (ropvl 1675); training = 1404 photos of 544 infants (ostrava 694, shenzhen 710).

| rows | normal | stage_1 | stage_2 | stage_3 | stage_4_5 | arop | total |
|---|---|---|---|---|---|---|---|
| training rows | 486 | 259 | 347 | 292 | 0 | 20 | 1404 |
| validation rows | 623 | 236 | 338 | 241 | 89 | 148 | 1675 |

- validated with zero training rows: stage_4_5
- images per epoch = 1404 (the training rows: the fold file's 1408 less 4 exact duplicate(s) dropped at run time)
- steps per epoch = 88 (= ceil(1404 / (16 x 1)), batch_size x grad_accum)
- epochs 1..1 train the classification head only (train.head_warmup_epochs)
- minutes 26.8 (fold start to next fold start, kernel log) · 33.3 img/s · GPU Tesla T4 · kernel log reports train 1404 val 1675

Per epoch (`metrics.json` history; lr = the first optimizer group, i.e. the lowest layer under layer-wise decay, at the end of the epoch):

| Epoch | train loss | lr | val acc | val macro-F1 | val macro-P | val macro-R |
|---|---|---|---|---|---|---|
| Epoch 1 | 2.1042 | 2.882e-06 | 0.2430 | 0.1647 | 0.1902 | 0.2080 |
| Epoch 2 | 1.9270 | 5.498e-06 | 0.2800 | 0.1781 | 0.2049 | 0.2272 |
| Epoch 3 | 1.7158 | 5.455e-06 | 0.3493 | 0.2052 | 0.2237 | 0.2679 |
| Epoch 4 | 1.6124 | 5.330e-06 | 0.4012 | 0.2268 | 0.2424 | 0.3026 |
| Epoch 5 | 1.4761 | 5.127e-06 | 0.4382 | 0.2453 | 0.2812 | 0.3301 |
| Epoch 6 | 1.4434 | 4.851e-06 | 0.4776 | 0.2682 | 0.3392 | 0.3552 |
| Epoch 7 | 1.3983 | 4.511e-06 | 0.5116 | 0.3005 | 0.3903 | 0.3813 |
| Epoch 8 | 1.3512 | 4.118e-06 | 0.5552 | 0.3568 | 0.4052 | 0.4174 |
| Epoch 9 | 1.3213 | 3.684e-06 | 0.5910 | 0.3985 | 0.4382 | 0.4441 |
| Epoch 10 | 1.2837 | 3.221e-06 | 0.6281 | 0.4378 | 0.4722 | 0.4702 |
| Epoch 11 | 1.2778 | 2.743e-06 | 0.6448 | 0.4535 | 0.4843 | 0.4792 |
| Epoch 12 | 1.2387 | 2.266e-06 | 0.6484 | 0.4545 | 0.4823 | 0.4769 |
| Epoch 13 | 1.2073 | 1.804e-06 | 0.6525 | 0.4529 | 0.4764 | 0.4762 |
| Epoch 14 | 1.1973 | 1.370e-06 | 0.6645 | 0.4653 | 0.4930 | 0.4875 |
| Epoch 15 | 1.1917 | 9.777e-07 | 0.6699 | 0.4700 | 0.4955 | 0.4922 |
| Epoch 16 | 1.1714 | 6.396e-07 | 0.6681 | 0.4656 | 0.4889 | 0.4896 |
| Epoch 17 | 1.1512 | 3.656e-07 | 0.6722 | 0.4682 | 0.4936 | 0.4934 |
| Epoch 18 | 1.1540 | 1.639e-07 | 0.6716 | 0.4650 | 0.4853 | 0.4921 |
| Epoch 19 | 1.1611 | 4.084e-08 | 0.6693 | 0.4609 | 0.4794 | 0.4895 |
| Epoch 20 | 1.1561 | 2.740e-11 | 0.6722 | 0.4632 | 0.4802 | 0.4926 |

Final epoch (Epoch 20), all 1675 validation rows: QWK (6 labels, summary.json) 0.7180 · QWK5 (AP-ROP folded onto stage 4/5, val_predictions.npz) 0.7852 · accuracy 0.6722 · macro-F1 0.4632 · balanced accuracy 0.4926 · AUC 0.7731

| final epoch | Normal | Stage 1 | Stage 2 | Stage 3 | Stage 4/5 | AP-ROP |
|---|---|---|---|---|---|---|
| recall | 0.9374 | 0.4661 | 0.6746 | 0.7967 | 0.0000 | 0.0811 |
| support | 623 | 236 | 338 | 241 | 89 | 148 |

## 4. Results (`v3_scores.json`)

### 4.1 Selection table

Rule: CV pooled out-of-fold image-level QWK, 5-class map (AP-ROP folded onto stage 4/5), computed once per arm, no dev involvement; AP-ROP scored separately as the binary branch metric (mean over the arm's seeds)

| arm | CV QWK5 | 95% CI | seeds pooled | AP-ROP binary recall | AP-ROP binary F1 |
|---|---|---|---|---|---|
| A | 0.7944 | [0.6636, 0.9153] | 2 | 0.2682 | 0.3965 |
| B | 0.8252 | [0.7365, 0.9111] | 2 | 0.3890 | 0.5012 |
| C | 0.8163 | [0.7134, 0.9159] | 2 | 0.3817 | 0.5155 |
| D | 0.7967 | [0.6989, 0.9094] | 2 | 0.4903 | 0.5319 |
| E | 0.8208 | [0.7079, 0.9271] | 2 | 0.3282 | 0.4561 |
| F | 0.8117 | [0.7040, 0.9146] | 2 | 0.3274 | 0.4527 |
| G | 0.8139 | [0.7059, 0.9148] | 2 | 0.3015 | 0.4223 |
| H | 0.8128 | [0.7066, 0.9142] | 2 | 0.3063 | 0.4218 |
| I | 0.8214 | [0.7271, 0.9124] | 2 | 0.2690 | 0.3718 |

### 4.2 Headline per arm (pooled over seeds)

CV = pooled out-of-fold, image level (14 exact duplicates excluded); dev = Multi-View dev half, eye level (descriptive, development-external).

| arm | CV lists | CV QWK5 [CI] | CV QWK (6) [CI] | CV macro-F1 [CI] | dev lists | dev eye QWK5 [CI] | dev eye macro-F1 [CI] | note |
|---|---|---|---|---|---|---|---|---|
| A | 2 | 0.7944 [0.6636, 0.9153] | 0.7363 [0.5773, 0.9129] | 0.6430 [0.5610, 0.7708] | 10 | 0.7742 [0.7067, 0.8281] | 0.5632 [0.4610, 0.7107] |  |
| B | 2 | 0.8252 [0.7365, 0.9111] | 0.7787 [0.6704, 0.9052] | 0.6528 [0.5754, 0.7603] | 10 | 0.7556 [0.6790, 0.8211] | 0.5657 [0.4610, 0.7184] |  |
| C | 2 | 0.8163 [0.7134, 0.9159] | 0.7731 [0.6526, 0.9113] | 0.6562 [0.5698, 0.7703] | 10 | 0.7599 [0.6798, 0.8221] | 0.5726 [0.5081, 0.6946] |  |
| D | 2 | 0.7967 [0.6989, 0.9094] | 0.7539 [0.6546, 0.8968] | 0.6606 [0.5900, 0.7534] | 10 | 0.7464 [0.6708, 0.8102] | 0.5416 [0.4399, 0.6935] |  |
| E | 2 | 0.8208 [0.7079, 0.9271] | 0.7680 [0.6316, 0.9224] | 0.6642 [0.5748, 0.7866] | 10 | 0.7699 [0.6970, 0.8274] | 0.5675 [0.4683, 0.7183] |  |
| F | 2 | 0.8117 [0.7040, 0.9146] | 0.7598 [0.6272, 0.9109] | 0.6441 [0.5612, 0.7627] | 10 | 0.7441 [0.6648, 0.8120] | 0.5515 [0.4527, 0.7096] | transductive: dev images seen unlabelled during training |
| G | 2 | 0.8139 [0.7059, 0.9148] | 0.7583 [0.6236, 0.9095] | 0.6473 [0.5601, 0.7686] | 10 | 0.7674 [0.6967, 0.8265] | 0.5514 [0.4542, 0.7115] |  |
| H | 2 | 0.8128 [0.7066, 0.9142] | 0.7569 [0.6238, 0.9113] | 0.6463 [0.5639, 0.7648] | 10 | 0.7501 [0.6742, 0.8136] | 0.5562 [0.4478, 0.7135] |  |
| I | 2 | 0.8214 [0.7271, 0.9124] | 0.7622 [0.6350, 0.9081] | 0.6181 [0.5305, 0.7440] | 10 | 0.7758 [0.7045, 0.8335] | 0.5605 [0.4751, 0.7090] |  |

### 4.3 Held-out sources (leave-one-source-out)

| arm | pool | held-out source | photos | infants | QWK5 | QWK5 95% CI | macro-F1 | accuracy | stage_4_5 training rows | seeds |
|---|---|---|---|---|---|---|---|---|---|---|
| A | v3 | shenzhen | 728 | 396 | 0.9575 | [0.9475, 0.9662] | 0.8324 | 0.8668 | 89 | 2 |
| B | v3 | shenzhen | 728 | 396 | 0.9620 | [0.9526, 0.9708] | 0.8412 | 0.8819 | 89 | 2 |
| B | v3 | ostrava | 4285 | 148 | 0.7397 | [0.3982, 0.8451] | 0.4828 | 0.7909 | 89 | 2 |
| B | v3 | ropvl | 1673 | 983 | 0.7995 | [0.7710, 0.8271] | 0.4524 | 0.6766 | 0 | 2 |
| D | v3 | shenzhen | 728 | 396 | 0.9577 | [0.9451, 0.9683] | 0.8357 | 0.8771 | 89 | 2 |
| H | v4 | shenzhen | 728 | 396 | 0.9597 | [0.9496, 0.9688] | 0.8390 | 0.8771 | 89 | 2 |
| H | v4 | ostrava | 4285 | 148 | 0.7664 | [0.4518, 0.8581] | 0.4792 | 0.7965 | 89 | 2 |
| H | v4 | ropvl | 1673 | 983 | 0.8006 | [0.7740, 0.8241] | 0.4431 | 0.6581 | 0 | 2 |
| I | v4 | shenzhen | 728 | 396 | 0.9423 | [0.9262, 0.9561] | 0.7774 | 0.8400 | 89 | 2 |
| I | v4 | ostrava | 4285 | 148 | 0.7791 | [0.4706, 0.8747] | 0.4873 | 0.8041 | 89 | 2 |
| I | v4 | ropvl | 1673 | 983 | 0.7959 | [0.7684, 0.8204] | 0.4621 | 0.6757 | 0 | 2 |

| arm | mean QWK5 over sources | sources | all three |
|---|---|---|---|
| A | 0.9575 | shenzhen | False |
| B | 0.8337 | shenzhen, ostrava, ropvl | True |
| D | 0.9577 | shenzhen | False |
| H | 0.8422 | shenzhen, ostrava, ropvl | True |
| I | 0.8391 | shenzhen, ostrava, ropvl | True |

### 4.4 Paired contrasts (infant-clustered bootstrap)

**cv_image** (CV pooled out-of-fold, image level, 14 duplicates excluded; selection-rule context, not a pre-registered endpoint (no 'significant' language); 6686 images)

| pair | QWK5 a | QWK5 b | QWK5 diff | diff 95% CI | macro-F1 diff | diff 95% CI | lists | note |
|---|---|---|---|---|---|---|---|---|
| A-C | 0.7944 | 0.8163 | -0.0219 | [-0.0517, 0.0084] | -0.0132 | [-0.0254, 0.0118] | A 2, C 2 |  |
| A-B | 0.7944 | 0.8252 | -0.0309 | [-0.0773, 0.0148] | -0.0098 | [-0.0225, 0.0355] | A 2, B 2 |  |
| B-C | 0.8252 | 0.8163 | 0.0090 | [-0.0181, 0.0273] | -0.0034 | [-0.0412, 0.0181] | B 2, C 2 |  |
| E-C | 0.8208 | 0.8163 | 0.0045 | [-0.0073, 0.0186] | 0.0079 | [-0.0014, 0.0240] | E 2, C 2 |  |
| E-A | 0.8208 | 0.7944 | 0.0264 | [0.0082, 0.0455] | 0.0212 | [0.0073, 0.0286] | E 2, A 2 |  |
| D-A | 0.7967 | 0.7944 | 0.0024 | [-0.0573, 0.0460] | 0.0176 | [-0.0740, 0.0552] | D 2, A 2 |  |
| D-B | 0.7967 | 0.8252 | -0.0285 | [-0.0507, 0.0026] | 0.0078 | [-0.0420, 0.0377] | D 2, B 2 |  |
| F-B | 0.8117 | 0.8252 | -0.0135 | [-0.0348, 0.0075] | -0.0087 | [-0.0210, 0.0213] | F 2, B 2 |  |
| G-B | 0.8139 | 0.8252 | -0.0113 | [-0.0357, 0.0124] | -0.0055 | [-0.0197, 0.0259] | G 2, B 2 |  |
| H-B | 0.8128 | 0.8252 | -0.0125 | [-0.0310, 0.0085] | -0.0065 | [-0.0170, 0.0190] | H 2, B 2 |  |
| I-B | 0.8214 | 0.8252 | -0.0038 | [-0.0145, 0.0070] | -0.0347 | [-0.0530, -0.0014] | I 2, B 2 |  |
| I-H | 0.8214 | 0.8128 | 0.0087 | [-0.0069, 0.0196] | -0.0282 | [-0.0428, -0.0099] | I 2, H 2 |  |

**dev_eye** (descriptive, not an endpoint (dev is development-external); 230 eyes)

| pair | QWK5 a | QWK5 b | QWK5 diff | diff 95% CI | macro-F1 diff | diff 95% CI | lists | note |
|---|---|---|---|---|---|---|---|---|
| A-C | 0.7742 | 0.7599 | 0.0142 | [-0.0119, 0.0482] | -0.0094 | [-0.0632, 0.0313] | A 10, C 10 |  |
| A-B | 0.7742 | 0.7556 | 0.0185 | [0.0002, 0.0406] | -0.0025 | [-0.0324, 0.0163] | A 10, B 10 |  |
| B-C | 0.7556 | 0.7599 | -0.0043 | [-0.0376, 0.0304] | -0.0068 | [-0.0656, 0.0418] | B 10, C 10 |  |
| E-C | 0.7699 | 0.7599 | 0.0099 | [-0.0116, 0.0399] | -0.0051 | [-0.0575, 0.0344] | E 10, C 10 |  |
| E-A | 0.7699 | 0.7742 | -0.0043 | [-0.0196, 0.0080] | 0.0043 | [-0.0095, 0.0217] | E 10, A 10 |  |
| D-A | 0.7464 | 0.7742 | -0.0278 | [-0.0477, -0.0105] | -0.0216 | [-0.0465, 0.0191] | D 10, A 10 |  |
| D-B | 0.7464 | 0.7556 | -0.0092 | [-0.0192, 0.0012] | -0.0241 | [-0.0423, -0.0020] | D 10, B 10 |  |
| F-B | 0.7441 | 0.7556 | -0.0115 | [-0.0219, -0.0020] | -0.0143 | [-0.0274, 0.0061] | F 10, B 10 | transductive: dev images seen unlabelled during training |
| G-B | 0.7674 | 0.7556 | 0.0118 | [-0.0055, 0.0315] | -0.0144 | [-0.0366, 0.0162] | G 10, B 10 |  |
| H-B | 0.7501 | 0.7556 | -0.0055 | [-0.0149, 0.0028] | -0.0095 | [-0.0227, 0.0095] | H 10, B 10 |  |
| I-B | 0.7758 | 0.7556 | 0.0202 | [0.0029, 0.0405] | -0.0052 | [-0.0569, 0.0332] | I 10, B 10 |  |
| I-H | 0.7758 | 0.7501 | 0.0258 | [0.0062, 0.0463] | 0.0043 | [-0.0501, 0.0452] | I 10, H 10 |  |

### 4.5 The v4 comparison rule (Amendment 5c)

Rule: Amendment 5c: adopt I if CV QWK5(I) - QWK5(B) >= +0.010 and LOSO(I) >= LOSO(B) - 0.010; else adopt H under the same two conditions; else B stays. CV QWK5 = the selection table (pooled out-of-fold, both seeds); LOSO = mean QWK5 over the three held-out sources (Shenzhen site fold, Ostrava, ROP-VL), pooled over both seeds; differences of the reported 4-dp numbers. An arm without both seeds or without all three held-out sources is not adoptable.

**Chosen arm: B**

| arm | CV QWK5 | mean LOSO QWK5 |
|---|---|---|
| B | 0.8252 | 0.8337 |
| H | 0.8128 | 0.8422 |
| I | 0.8214 | 0.8391 |

- I: CV QWK5 I-B = -0.0038 (needs >= +0.010), LOSO I-B = +0.0054 (needs >= -0.010) -> not adopted
- H: CV QWK5 H-B = -0.0124 (needs >= +0.010), LOSO H-B = +0.0085 (needs >= -0.010) -> not adopted
- B stays

## 5. Decisions and rulings

Every line of the SDD ledger `.superpowers/sdd/2026-10-31-rop-v4-implementation/progress.md` that starts with "Ruling", verbatim (16 lines):

- Ruling: no git worktree — the audit and pool builder read data/rop/* which only exists in the main tree — cost if wrong: an in-flight task could touch an uncommitted file (tree was clean at start).
- Ruling: `fourier_swap_batch` runs on the normalised GPU batch (after `Normalize`); FFT is linear so the amplitude swap is equivalent up to the per-channel affine — cost if wrong: a slightly different swap strength than SABPI's pixel-space version; disclosed in the run log.
- Ruling: head warm-up freezes every parameter outside optim's "head" group (norm + head); the OneCycle schedule still steps every batch — cost if wrong: epoch 1 of arm I spends 1/20 of the schedule on the head only (that is the intent).
- Ruling (plan defect, spec 5a.2): hospital-derived thresholds measure the camera, not the frame. Revised rule, to be written into the protocol as "Amendment 5a, revision (after the audit, before the pool)": quality is judged WITHIN each Ostrava device (D1/D2/D3): a frame fails if empty, or blur_var < that device's 10th percentile, or luminance outside that device's 2.5–97.5 percentiles; the FOV criterion is dropped (a camera constant). Cap ranking uses the blur_var percentile rank within device. Floor: an (infant, class) cell with fewer than 2 passing frames keeps its 2 best-ranked frames (all if fewer exist). — why: the audit's purpose is uninformative frames, and the spec's rule would delete a camera and every Ostrava stage_3 frame — cost if wrong: ~10% of Ostrava frames per camera removed on a within-camera rule that may remove some gradable frames; disclosed.
- Ruling (concern 2): cross-fold near-duplicate exclusion requires Hamming ≤ 4 AND pixel_mad ≤ 5.0 grey levels (mean absolute difference, both images grey at 256×256); the audit adds `pixel_mad` and `true_duplicate` columns; all ≤4 pairs stay in the clinician list — why: 256-bit dHash at ≤4 collides across different Ostrava retinas (pixel diff 9–47 vs 3.2 for same-session frames) — cost if wrong: a few genuine duplicates at MAD 5–9 stay in training.
- Ruling (concern 1): Shenzhen's 11 over-cap normal infants are re-ranked by blur_rank_in_device like every other source (10 photos differ from v3's filename-order pick) — protocol 5a.3 states quality ranking for every (infant, class) cap, and arm H is defined as "the v4 pool"; v3's `kept` flag is not resurrected — cost if wrong: 10 of 3,083 training photos differ from a pure "Ostrava-only" data arm; disclosed in the run log.
- Ruling (floor ordering): the floor keeps the 2 best-ranked rows overall, exactly as the protocol revision states ("2 best-ranked frames"); passing rows are not privileged — why: the protocol text is what was pre-registered; on today's data the single floor cell keeps its passing frame either way — cost if wrong: none today; a future cell could keep a sharp-but-dark frame over a dim passing one.
- Ruling (reverses the Task 7 dispatch's "LOSO always from v4"): arm B's LOSO folds must train on the v3 pool rows (B's recipe AND B's data), or the H−B LOSO guard compares data arms against themselves. v3-pool LOSO files `results/prepared/rop_staging/fold_loso_ostrava.csv` / `fold_loso_ropvl.csv` are emitted from the v3 `kept` rows by the v4 pool builder; `fold_file` routes LOSO by pool like every other fold; val sides must be identical across the v3 and v4 LOSO files — cost if wrong: none (strictly more correct); the kernel comment about "LOSO only under v4" goes.
- Ruling (concern 1): the collector's train-row check (kernel's trained rows == pool fold-file train rows minus duplicates) is accepted — it is the only check that tells the two pools apart, since val sides are identical by design — cost if wrong: none (56/56 real v3 folds pass).
- Ruling (concern 2): pusher guard accepted — a `--loso` launch of variant "" must be dinov2_s with `--folds ""` and `--no-site-fold` (B's LOSO-only kernel); A/D/F/G may not run LOSO — cost if wrong: an extra flag to type.
- Ruling (rounding): select_v4 compares differences of the 4-dp numbers written to the selection table — the same numbers the protocol's readers see; the margins 0.010 are two orders above the rounding — cost if wrong: a decision could flip only when a true difference lies within 1e-4 of a margin; disclosed in the JSON rule text.
- Ruling (bundle contents): the dry-run bundle (27 MB) would publish docs/ (report + paper drafts, fonts, figures), tmp/ and output/ to the PUBLIC code dataset. build_kaggle_bundle.py EXCLUDE gains "docs", "output", "tmp" and the patterns "*.pdf", "*.docx", "*.pptx" — no kernel imports any of them — cost if wrong: none for training; the protocol text stays in git, not on Kaggle.
- Ruling (publish): re-versioning the existing public dataset `retinal-framework-code` is part of the approved plan and was done for every v3 round; contents are swept for PHI; the owner's pending decision on making it private is unchanged and restated in the final summary.
- Ruling (corrects the Fourier ruling above): the swap copies the partner's COMPLEX zero-frequency value and the amplitudes of the other 8 block entries; this equals the pixel-space swap exactly — cost if wrong: none.
- Ruling: a class with zero training rows gets zero class-balanced weight (pre-run protocol note; no v3 fold is affected since every class is present in every v3 fold) — cost if wrong: none for A–G; LOSO-ropvl scores stage 4/5 at recall 0 by construction, which is the honest value.
- Ruling: test_locked.csv joins the bundle EXCLUDE (no kernel reads it; the dataset is public) — cost if wrong: none.
