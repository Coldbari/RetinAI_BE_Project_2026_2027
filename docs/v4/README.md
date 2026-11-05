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
