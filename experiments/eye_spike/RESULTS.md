# Eye/examination-level spike: attention-MIL vs "most severe photo" on frozen features

Throwaway spike, run 2026-10-18. Not committed. **test_locked was never read.** Only the 6,700-image CV
pool (fold_0..4 val rows) and the Multi-View **dev** half were used.

**Question.** If every method gets the same frozen image features, does a learned attention model over
all the photos of one eye or examination beat taking the most severe photo? And which retinal encoder
gives the better features?

**Answer.**
1. **No.** On CV, attention ties with "photo, then max". On the only clean multi-view-eye test (Multi-View
   dev), it is significantly worse.
2. RETFound-Green with CLAHE is the best encoder, but that ranking is partly a preprocessing effect (see
   caveat 5).
3. Frozen features plus the best head lose clearly to the fine-tuned EfficientNetV2-S on CV. On dev QWK
   they tie, and on dev macro-F1 they lose.

## Pipeline (all scripts in this folder, run in order)

| script | does |
|---|---|
| `s1_bags.py` | `bags.csv`, `bags_summary.json`, `cache/membership.npz` (image row → bag). Bag ids and patient ids are salted HMAC hashes (salt in `cache/salt.bin`). No filenames or dates are written anywhere. |
| `s2_features.py` | Frozen embeddings → `cache/feat_*.npz` (float16), `timing_features.json` |
| `s3_heads.py` | All heads, 5-fold CV over bags plus a train-on-all model scored on dev → `cache/preds_*.npz` |
| `s4_reference.py` | Fine-tuned reference, from committed npz only → `ref_results.json` (includes the reproduction check) |
| `s5_report.py` | Metrics, per-source breakdown, bootstraps → `results.json` and the tables below |

**Bags.** Each bag is one source's natural unit:
- Ostrava: (infant, device, serie) session. Ostrava has no laterality, so this is an "examination", not an eye.
- Shenzhen: (infant, OD/OS), from `zip information.xlsx`.
- ROP-VL: (infant, exam date, L/R), parsed from the filename and then hashed.
- Multi-View dev: (patient, Left/Right), via `rop_locked_eval.eye_frame`.

Bag label = max class index (0 normal … 4 stage_4_5, 5 arop). Integrity checks: 0 bags span folds, 0 bags
span patients, 0 dev patients appear in the pool.

| | bags | images | single-image bags | median / p90 / max size | bags >16 imgs |
|---|---|---|---|---|---|
| CV pool, all | 2,512 | 6,700 | **1,952 (78%)** | 1 / 8 / 58 | 60 |
| Ostrava sessions | 342 (64 diseased) | 4,285 | 0 | 11.5 / 19.9 / 58 | 60 |
| ROP-VL eye-exams | 1,557 | 1,675 | 1,443 | 1 / 1 / 3 | 0 |
| Shenzhen eyes | 613 | 740 | 509 | 1 / 2 / 6 | 0 |
| Multi-View dev eyes | 230 | 663 | 7 | 3 / 3 / 4 | 0 |

- Bags per fold: 497 / 479 / 448 / 536 / 552.
- Multi-image diseased CV bags: **263** in total, about 210 per training split.
- CV bag labels: normal 1,094, stage_1 274, stage_2 446, stage_3 453, stage_4_5 81, arop 164.
- Dev eye labels: 50 / 68 / 46 / 59 / 5 / 2.

**Encoders.** Every encoder gets the repo's `circle_crop` and `letterbox_square`, then a resize. fp16
autocast on MPS (minimum cosine against fp32 ≥ 0.99999).

| key | model | input / norm | feature |
|---|---|---|---|
| rfg | RETFound-Green (GitHub release v0.1, `weights_only=True`, strict load into timm `vit_small_patch14_reg4_dinov2`) | 392, **no CLAHE**, mean/std 0.5 | CLS ⊕ mean of patch tokens (768) |
| rfg_clahe | same | 392, **+ repo CLAHE** | 768 |
| dino | timm `vit_small_patch14_reg4_dinov2.lvd142m` (safetensors) | 392, no CLAHE, ImageNet | 768 |
| effnet | torchvision EfficientNetV2-S ImageNet. This is the repo backbone's own init (`weights_only=True`). | 384, + CLAHE (the repo staging pipeline), ImageNet | avg-pool (1280) |

**Heads.** Features are z-scored on training images only. All heads use **natural sampling and
unweighted cross-entropy**, fixed before any result was seen. No early stopping. Torch heads are run with
2 seeds and the metrics averaged.
- `img_lr_max`: multinomial LR on single images. C is picked from {0.01, 0.1, 1} by log-loss on an inner
  patient-disjoint fold. The bag gets the most severe predicted class over **all** its images.
- `img_mlp_max`: same as above, but with the φ-MLP the MIL heads use: Linear(d,128), ReLU, Dropout 0.25,
  then Linear(128,6).
- `meanpool`, `maxpool`, `attn`: φ, then masked mean, max, or gated attention (Ilse 2018, 64-d), then a
  linear layer trained on bag labels.
  - Training: every bag shows exactly K=4 instances (drawn without replacement if the bag has ≥4 images,
    with replacement otherwise), so bag size cannot be read.
  - Evaluation: up to 16 instances, evenly spaced over frame order, deterministic.
  - Optimiser: AdamW 1e-3, weight decay 1e-2, cosine schedule, 30 epochs.
- Reference: the fine-tuned EfficientNetV2-S structured head (seed 42).
  - CV: its committed out-of-fold `structured_5fold/fold_k.npz`, max per bag.
  - Dev: `dev_eval/structured_s42_ckpt{0..4}.npz`, mean over the 5 checkpoints.
  - Alignment check: `y_true == label` row by row.
  - **Reproduction check: dev eye QWK 0.7206, which matches `preprint_stats.json` B exactly.** Pooled
    image-level CV macro-F1 0.652 and QWK 0.761 match ROP_PROJECT_STATE §5.

**Metrics.**
- QWK over the 6 labels, with labels fixed at 0..5 (arop = 5), as the repo's eye metric does.
- Macro-F1 over the classes present in y.
- CV metrics are pooled over the 5 out-of-fold folds.

## Main table

"±" = seed SD of CV QWK (2 seeds).

| encoder · method | CV bag QWK | CV bag mF1 | CV acc | Ostrava sess QWK / mF1 | multi-img CV bags (560) QWK / mF1 | **dev eye QWK** | **dev eye mF1** |
|---|---|---|---|---|---|---|---|
| RETFound-Green (no CLAHE) · img_lr_max | 0.823 | 0.672 | 0.744 | 0.648 / 0.456 | 0.802 / 0.637 | 0.629 | 0.384 |
| RETFound-Green (no CLAHE) · img_mlp_max | 0.841 ±0.001 | 0.689 | 0.754 | 0.677 / 0.483 | 0.829 / 0.651 | 0.630 | 0.431 |
| RETFound-Green (no CLAHE) · meanpool | 0.831 ±0.002 | 0.676 | 0.748 | 0.665 / 0.494 | 0.827 / 0.638 | 0.449 | 0.429 |
| RETFound-Green (no CLAHE) · maxpool | 0.822 ±0.007 | 0.675 | 0.745 | 0.563 / 0.385 | 0.798 / 0.617 | 0.572 | 0.452 |
| RETFound-Green (no CLAHE) · attn | 0.838 ±0.001 | 0.684 | 0.754 | 0.675 / 0.453 | 0.835 / 0.638 | 0.613 | 0.411 |
| DINOv2 ViT-S/14-reg · img_lr_max | 0.816 | 0.670 | 0.739 | 0.620 / 0.430 | 0.791 / 0.612 | 0.705 | 0.449 |
| DINOv2 ViT-S/14-reg · img_mlp_max | 0.835 ±0.002 | 0.693 | 0.751 | 0.652 / 0.437 | 0.818 / 0.644 | 0.691 | 0.460 |
| DINOv2 ViT-S/14-reg · meanpool | 0.833 ±0.003 | 0.690 | 0.752 | 0.651 / 0.424 | 0.814 / 0.630 | 0.419 | 0.436 |
| DINOv2 ViT-S/14-reg · maxpool | 0.834 ±0.003 | 0.685 | 0.752 | 0.646 / 0.452 | 0.808 / 0.580 | 0.502 | 0.448 |
| DINOv2 ViT-S/14-reg · attn | 0.833 ±0.008 | 0.687 | 0.755 | 0.671 / 0.443 | 0.823 / 0.607 | 0.648 | 0.439 |
| EffNetV2-S ImageNet (+CLAHE) · img_lr_max | 0.783 | 0.627 | 0.713 | 0.596 / 0.348 | 0.761 / 0.538 | 0.631 | 0.394 |
| EffNetV2-S ImageNet (+CLAHE) · img_mlp_max | 0.796 ±0.001 | 0.636 | 0.717 | 0.576 / 0.386 | 0.761 / 0.556 | 0.597 | 0.362 |
| EffNetV2-S ImageNet (+CLAHE) · meanpool | 0.798 ±0.002 | 0.637 | 0.719 | 0.459 / 0.346 | 0.772 / 0.557 | 0.524 | 0.383 |
| EffNetV2-S ImageNet (+CLAHE) · maxpool | 0.794 ±0.002 | 0.635 | 0.721 | 0.431 / 0.339 | 0.768 / 0.560 | 0.643 | 0.425 |
| EffNetV2-S ImageNet (+CLAHE) · attn | 0.805 ±0.001 | 0.639 | 0.725 | 0.494 / 0.376 | 0.795 / 0.593 | 0.649 | 0.370 |
| RETFound-Green (+CLAHE) · img_lr_max | 0.840 | 0.683 | 0.754 | 0.677 / 0.434 | 0.826 / 0.624 | **0.739** | 0.423 |
| **RETFound-Green (+CLAHE) · img_mlp_max** | **0.859** ±0.001 | **0.707** | 0.770 | **0.720** / 0.476 | **0.858** / 0.655 | 0.730 | 0.443 |
| RETFound-Green (+CLAHE) · meanpool | 0.852 ±0.001 | 0.690 | 0.756 | 0.691 / 0.509 | 0.832 / 0.616 | 0.525 | 0.390 |
| RETFound-Green (+CLAHE) · maxpool | 0.841 ±0.001 | 0.689 | 0.758 | 0.593 / 0.407 | 0.797 / 0.610 | 0.611 | 0.340 |
| RETFound-Green (+CLAHE) · attn | 0.854 ±0.001 | 0.702 | 0.767 | 0.713 / 0.500 | 0.850 / 0.657 | 0.651 | 0.372 |
| **Fine-tuned EffNetV2-S structured s42, max of photos (reference)** | **0.917** | **0.790** | 0.828 | **0.830 / 0.544** | **0.901** / 0.655 | 0.721 (5-ckpt mean) | **0.551** |
| Size-only control (LR on one-hot bag-size bins) | 0.096 | 0.142 | 0.451 | 0.253 / 0.181 | 0.584 / 0.243 | 0.018 | 0.076 |

**Size shortcut.** On its own, raw bag size gives these AUCs for normal vs any disease:
- Ostrava sessions: **0.774**, which reproduces the audit.
- Non-Ostrava bags: 0.562.
- All CV bags: 0.443. This is *inverted*, because the singleton sources are the disease-rich ones.
- Dev: 0.481.

As a 6-class predictor, size alone reaches CV macro-F1 0.142 and QWK 0.096, below the G4 metadata bar of
0.187. The MIL heads could not use size, because training fixed K=4.

**Per source, best encoder (RETFound-Green + CLAHE), CV, QWK / mF1:**

| method | Ostrava sessions (342) | ROP-VL (1,557) | Shenzhen (613) | single-image bags (1,952) | multi-image bags (560) |
|---|---|---|---|---|---|
| img_lr_max | 0.677/0.434 | 0.835/0.671 | 0.912/0.732 | 0.844/0.693 | 0.826/0.624 |
| img_mlp_max | 0.720/0.476 | 0.853/0.695 | 0.923/0.754 | 0.859/0.716 | 0.858/0.655 |
| meanpool | 0.691/0.509 | 0.837/0.668 | 0.910/0.714 | 0.855/0.702 | 0.832/0.616 |
| maxpool | 0.593/0.407 | 0.829/0.672 | 0.910/0.720 | 0.850/0.703 | 0.797/0.610 |
| attn | 0.713/0.500 | 0.837/0.683 | 0.918/0.716 | 0.853/0.709 | 0.850/0.657 |
| reference (fine-tuned) | 0.830/0.544 | 0.911/0.793 | 0.968/0.859 | 0.921/0.822 | 0.901/0.655 |

**Per-class recall, CV pooled / dev (RETFound-Green + CLAHE):**

| method | normal | stage_1 | stage_2 | stage_3 | stage_4_5 | arop |
|---|---|---|---|---|---|---|
| img_mlp_max | 0.94/0.88 | 0.47/**0.12** | 0.60/0.59 | 0.73/0.55 | 0.72/0.40 | 0.71/0.50 |
| attn | 0.96/1.00 | 0.41/**0.02** | 0.57/0.57 | 0.74/**0.23** | 0.77/0.30 | 0.67/0.50 |
| meanpool | 0.96/1.00 | 0.37/0.01 | 0.56/0.53 | 0.72/0.14 | 0.75/0.60 | 0.70/0.50 |
| reference | 0.92/0.97 | 0.66/**0.41** | 0.73/0.67 | 0.83/0.59 | 0.78/0.44 | 0.80/0.50 |

Dev has 5 stage_4_5 eyes and 2 arop eyes, so those two recalls are anecdotes.

## Bootstraps

Paired and patient-clustered, 1,000 replicates. Seeds (or checkpoints) are averaged inside each
replicate. Each row is a − b.

| comparison | metric | a | b | diff [95% CI] |
|---|---|---|---|---|
| CV: attn − img_mlp_max (RFG+CLAHE) | QWK | 0.854 | 0.859 | −0.006 [−0.028, +0.015] |
| | mF1 | 0.702 | 0.707 | −0.006 [−0.021, +0.011] |
| CV multi-image bags only: attn − img_mlp_max | QWK | 0.851 | 0.858 | −0.008 [−0.078, +0.067] |
| | mF1 | 0.657 | 0.655 | +0.002 [−0.041, +0.054] |
| **Dev: attn − img_mlp_max** | QWK | 0.651 | 0.730 | **−0.080 [−0.132, −0.029]** |
| | mF1 | 0.372 | 0.443 | **−0.071 [−0.176, −0.024]** |
| CV: best frozen (RFG+CLAHE img_mlp_max) − fine-tuned | QWK | 0.859 | 0.917 | **−0.057 [−0.080, −0.033]** |
| | mF1 | 0.707 | 0.790 | **−0.082 [−0.110, −0.059]** |
| Dev: best frozen − fine-tuned | QWK | 0.730 | 0.721 | +0.010 [−0.040, +0.068] |
| | mF1 | 0.443 | 0.551 | **−0.108 [−0.209, −0.011]** |
| CV: attn (RFG+CLAHE) − fine-tuned | QWK | 0.854 | 0.917 | −0.063 [−0.097, −0.035] |
| Dev: attn (RFG+CLAHE) − fine-tuned | QWK | 0.651 | 0.721 | −0.070 [−0.123, −0.017] |

## Answers

**(a) Attention vs the most severe photo, same features.**
- Attention does not help.
- On CV it ties (−0.006 QWK, CI spans 0). That holds even on the 560 multi-image bags, where the
  aggregator actually matters (−0.008, CI ±0.07).
- On Multi-View dev it is worse for 3 of the 4 encoders. For RFG+CLAHE the gap is −0.080 QWK, with a CI
  that excludes 0. The exception is ImageNet EffNet, where attn scored 0.649 and img_mlp_max 0.597.
- Mechanism: on dev, attention and mean-pooling systematically under-call. Mean (pred − true) is −0.55
  for attn and −0.69 for meanpool, against −0.01 for max. In seed 0, 52 of 68 stage-1 eyes are called
  normal, against 26 under max.
- The only multi-image supervision the MIL heads get is mostly Ostrava sessions. Those are 278 of 342
  normal, with sparse disease frames. So the heads learn that a bag is normal unless the evidence is
  broad. Three-view eyes where one view shows the ridge then get diluted.
- Max hard-codes "any view counts". That is also how the labels are defined, and it transfers.
- Learned max-pooling is the worst pooled head on Ostrava sessions. Its pooled statistic shifts between
  K=4 in training and up to 16 at eval.

**(b) Best encoder.**
- On these frozen heads, RETFound-Green with CLAHE is best: CV QWK 0.84–0.86 and dev QWK up to 0.74.
- At matched preprocessing (no CLAHE), RETFound-Green and DINOv2 are indistinguishable on CV: mean
  head score 0.755 vs 0.758. DINOv2 is *better* on dev with the image heads: 0.69–0.71 vs 0.63.
- DINOv2 + CLAHE was not extracted. So "RETFound-Green > DINOv2" cannot be separated from "CLAHE helps".
  For RETFound-Green, CLAHE added about +0.02 CV QWK and +0.10 dev QWK with the image heads.
- Frozen ImageNet EfficientNetV2-S is clearly the worst, at −0.04 to −0.06 CV QWK.

**(c) Frozen + best head vs the fine-tuned reference.**
- CV: the fine-tuned model wins clearly, with QWK 0.917 vs 0.859 and macro-F1 0.790 vs 0.707. Both CIs
  exclude 0. It wins on every source.
- Dev: QWK ties (0.730 vs 0.721), but macro-F1 is worse (0.443 vs 0.551, CI excludes 0). The gap is
  driven by stage_1 eyes, with recall 0.12 frozen vs 0.41 fine-tuned.

## Caveats and fairness issues

1. **CV mostly cannot test aggregation.** 78% of CV bags are single images. For those, attn is just an
   image classifier trained on fewer units: 2,010 bags vs 5,360 images. The multi-image CV subset (560
   bags, 342 of them Ostrava) and dev (230 eyes) are the real tests, and both are small.
2. **Dev confounds aggregation with hospital shift.** It is the only clean 3-view-per-eye data, but every
   head there was trained on other hospitals. Attention's loss there is a transfer failure of a *learned*
   aggregator, not proof that attention is wrong in principle. Also, 217 of 230 dev eyes carry one label
   across all views.
3. **The training data differ from the reference's.**
   - Same patient partition (no leakage).
   - The frozen heads train on *all* images of the 4 training folds: uncapped, about 5,360 images,
     dominated by Ostrava session-stamped labels.
   - The reference trained on the capped training rows, with augmentation, CORN, an AP-ROP branch and a
     site adversary.
4. **Dev protocols differ.** Reference dev = mean over 5 fold checkpoints, each trained on 4/5 of the
   data. Frozen heads = one train-on-all model, averaged over 2 seeds. This slightly *favours* the frozen
   heads.
5. **Preprocessing deviation.** The brief said no CLAHE for the foundation encoders. RETFound-Green with
   CLAHE (an added ablation) was better, and became "best" by the CV selection rule. That selection was
   made on CV and then compared with the reference, which is mildly optimistic for the frozen side. The
   conclusion does not change, since frozen still loses on CV.
6. **Hyperparameters are fixed and untuned** (the MIL heads have no clean inner validation set: about 210
   multi-image diseased training bags per fold). The LR picked C=0.01, the grid edge, in every fold. The
   MLP image head beat the LR anyway. A tuned MIL head might close the CV tie, but the dev under-calling
   is systematic rather than noise.
7. **Image counts differ by method.** IMG+MAX uses all of a bag's images. The MIL heads see at most 16
   (this matters for 60 Ostrava sessions). The pooled heads train at K=4 and evaluate at up to 16.
8. **QWK convention.** QWK treats arop as ordinal level 5, with labels fixed at 0..5, matching the repo's
   eye metric. The per-source QWKs are not comparable across sources, because the class mixes differ.
9. **Odd timing.** DINOv2's forward pass was about 2× slower than RETFound-Green's on the same timm
   architecture. Not investigated, and it does not affect the results.

## Timings

Apple M5, MPS, fp16 autocast.
- Feature extraction, one decode pass for all 4 encoders over 7,363 images: 2,596 s wall.
- Forward time per encoder, and throughput:

  | encoder | forward time | images/s |
  |---|---|---|
  | rfg | 537 s | 13.7 |
  | rfg_clahe | 475 s | 15.5 |
  | dino | 995 s | 7.4 |
  | effnet | 446 s | 16.5 |

- Heads: 4 parallel single-thread processes, at 954 s (dino), 993 s (rfg), 1,028 s (rfg_clahe) and
  1,427 s (effnet) wall each.
- Bootstrap report: about 7 min.
- Untracked artefacts:
  - `weights/` holds the RETFound-Green weights (83 MB). SHA-256:
    `431de5dbc1bebbb32f60e2c0bcf8daa4f8bcbf06f7cb1e1dc97ec589713942e1`.
  - `cache/` holds about 54 MB.
