# Pre-registered protocol v2: v3 arms, selection, and the second opening of `test_locked`

*Committed BEFORE any v3 training run. Nothing below may be changed after the first v3
training run; anything not specified here may not be decided after seeing locked or dev
results. Source of truth for the design: `docs/superpowers/specs/2026-10-19-rop-v3-design.md`
(§4, §6); consensus: `docs/superpowers/panel/2026-10-19/review_methodologist.md`. The commit
hash of this file is recorded in the paper.*

## Arms

- **A** = `retfound_green`: timm `vit_small_patch14_reg4_dinov2`, img_size 392,
  `global_pool=avg`, state dict loaded strictly on backbone keys from the released
  `retfoundgreen_statedict.pth`, fresh 6-way head. `preprocess.normalize: half`.
- **B** = `dinov2_s`: timm `vit_small_patch14_reg4_dinov2.lvd142m`, img_size 392,
  `global_pool=avg`, HF-cache weights, fresh head. CLAHE on, like A. `preprocess.normalize: imagenet`.
- **C** = existing EfficientNetV2-S flat-natural, from saved npz under
  `results/rop/staging_runs/flat_natural_5fold*/`, `dev_eval/`, `locked_eval/`; never re-inferred.
- **D** (exam-level bag loss on Ostrava sessions): optional, 6-week scenario only.
- Training uses seeds 42 and 1337 for A and B.

## Parity rules

Identical to C (`configs/rop_staging.yaml` as run by `kaggle_kernels/rop_staging_benchmark.py`
with `weighted_sampler=false`): committed folds; natural sampler; normal cap 2/infant;
`class_balanced: true, beta 0.999`; `label_smoothing 0.05`; `strong_aug`; `site_randomization`;
batch 16; 20 epochs; OneCycle (pct 0.1, cosine); AMP; last-epoch weights; no TTA for paper arms.

Allowed to differ: backbone; `preprocess.image_size: 392`; `preprocess.normalize` (above, read in
the train AND the TTA/serving paths); lr recipe: AdamW, per-group lr with `layer_decay` 0.8,
peak lr chosen by the probe below from {3e-5, 1e-4, 2e-4} (default 1e-4), drop-path 0.1, head
dropout 0.3, weight decay 0.05 on all params (including 1-D params, to match C), grad-clip 1.0,
optional head warm-start from a linear probe on the same fold's train features.

## Peak-lr probe

Grid {3e-5, 1e-4, 2e-4}, run on an inner patient-disjoint split of fold-0 train only, for A and B.
Fold-0 val, dev and `test_locked` take no part. The recipe is frozen after the probe.

## Selection rule

CV pooled out-of-fold image-level QWK on the 5-class map (AP-ROP folded onto Stage 4/5 on both
sides, per `FINAL_MODEL_EVIDENCE_PACKAGE/06_VALIDATION/QWK_DEFINITION_CORRECTION.md`; AP-ROP
scored separately as the binary branch metric), computed once per arm. No dev involvement. Dev
is "development-external: unseen in training, used for design decisions in paper 1 and the
spike", never "external validation".

## Units

Multi-View = eye (patient, L/R) via the acquisition-view table; ROP-VL = eye-examination;
Shenzhen = image (no bagging); Ostrava = examination (both eyes). Eye/exam prediction = most
severe predicted class over the unit's images; label = most severe label. Report per source;
no pooled bag metric.

## Primary endpoint

Locked eye-level QWK (5-class map), contrast **A − C**, paired on the same infant resample
(2,000 replicates), C from saved npz (no re-inference). This is the only primary endpoint.

## Secondary endpoints (Holm-corrected)

A − B (also reported as a 90 % CI against ±0.05); D − A if D exists.

## Descriptive endpoints (no "significant" language)

Any-ROP and severe (stage 3/4-5/AP-ROP) sensitivity and specificity at the CV-OOF eye operating
point; macro-F1 over present classes; per-class recall; raw eye confusion matrices;
ECE/Brier/NLL after one CV-OOF temperature (eye probability = the driving photo's calibrated
vector, defined before any fitting); per-view miss table.

## Minimum detectable difference

Minimum detectable difference at 80 % power: ≈ 0.08 eye QWK on 230 eyes / 116 infants (661
images) of the locked half, from the paired infant-clustered bootstrap on saved dev predictions
(`review_methodologist.md`, Cons 1). Expected gains at or below this are not confirmable.

## Deduplication

The 20 images of the 10 exact-duplicate groups in the audit
(`FINAL_MODEL_EVIDENCE_PACKAGE/20_AUDIT/verification_pass_2/03_DATA_LEAKAGE/`) are examined
against the 6,700-image CV pool. The 14 images (7 groups) inside the pool are dropped from
training and from every CV metric (`results/prepared/rop_staging/exclude_duplicates.csv`); folds
are unchanged; C is re-scored on the reduced set from its npz; the change from paper 1 is
disclosed. 2 exact-duplicate pairs inside the locked half and 1 inside the dev half are
retained to keep paper 1's counts comparable; they do not cross any split.

## Second opening of `test_locked`

The second opening is ONE more scoring, last, only for A and B (5 checkpoints × 2 seeds each), with C from saved npz,
**only after the supervisor approves and the log-book row is amended**. Methods wording
(chair's consensus, quoted):

> "The locked half was scored once before (18 Aug 2026) for four pre-declared EfficientNet cells.
> Its aggregate results were known when the present models were designed; no model,
> hyperparameter, checkpoint or decision rule was selected using it. Protocol v2 (commit <hash>,
> before the first v2 training run) fixed arms, seeds, units, metrics, the primary contrast, the
> selection rule and the minimum detectable difference. The locked half was scored a second time
> on <date> and will not be scored again for these models."

A Limitations sentence states analyst exposure. Protocol v1's "no second shot" binds the 2026
EfficientNet cells, not these new families. **Fallback** if not approved: the dev half is the
final external set, with full disclosure.


## Amendment 1 — 2026-10-21, before any v3 arm result exists

Clarifications of items the protocol left open; no pre-registered endpoint, arm, unit, metric or
selection rule is changed. Made while scoring baseline C from saved npz (`scripts/rop_v3_score.py`).

a. **Temperature.** One scalar T per arm-seed, fitted by minimising NLL on the CV-OOF
   *unit-level driving-photo* probability vectors (the same object the eye-level calibration
   endpoint is computed on). The same T is applied to dev/locked eye vectors (primary) and to
   image-level probabilities (descriptive). T is reported per seed, never averaged.
   (Driving photo = the unit's photo with the highest predicted class, ties by that class's probability.)
b. **Any-ROP operating point.** The threshold on max-over-photos(1 - p_normal) is fitted on all CV
   units pooled over sources (target unit-level sensitivity 0.90, the largest threshold meeting it)
   in UNCALIBRATED probability space. Temperature scaling preserves each photo's argmax but not the
   cross-unit ordering of p_normal, so calibrated-space and uncalibrated-space thresholds are not
   decision-equivalent (tested); no equivalence is claimed.
c. **Headline pooling.** Headline numbers pool every prediction list of every seed (CV: one OOF list
   per seed; dev/locked: seeds x 5 fold checkpoints = 10 lists): the point estimate is the mean over
   lists, and one infant-clustered percentile 95 % CI (2,000 replicates, seed 42) averages the lists
   within each replicate. Per-seed numbers are also reported.
d. **Lr-probe kernels.** The peak-lr probe kernels also wrote dev predictions. Those are never
   opened or scored and are deleted after download.


## Amendment 2 — 2026-10-21, after the peak-lr probe, before any main v3 run

No pre-registered endpoint, arm, unit, metric or selection rule is changed.

a. **Peak lr, fixed by the probe.** Fold-0 train rows only, split by patient 80/20
   (`GroupShuffleSplit`, seed 0; 497 inner-val images), seed 42, 20 epochs, last-epoch weights, T4.
   Rule, written in the implementation plan before the probe ran and quoted here because the body
   above did not state it: keep 1e-4 unless another grid value's inner macro-F1 beats it by more than
   0.03 (the fold-0 seed SD); if several do, take the best. Record: `results/rop/v3/lr_probe.json`.

   | Arm | 3e-5 | 1e-4 | 2e-4 | Chosen |
   |---|---|---|---|---|
   | retfound_green (A) | 0.717 | 0.755 | 0.812 | **2e-4** (+0.057) |
   | dinov2_s (B) | 0.747 | 0.764 | 0.772 | **1e-4** (+0.008 is not > 0.03) |

   Inner macro-F1. A's best value sits at the upper edge of the grid; the grid is not extended
   (sweeps are a non-goal). A and B therefore train at different peak lrs, which the parity rules
   allow (the lr recipe is one of the four permitted differences).
b. **Duplicates in the probe.** The probes ran before the training-side de-duplication was wired
   into the kernel. Fold-0 train holds 12 of the 14 in-pool duplicates: 7 fell in the inner train
   and 5 in the inner val, and two duplicate groups straddle the two (about 2 of 497 inner-val
   images had an exact copy in training). This touches all six probe runs alike; the probe is not
   re-run. Every main run drops all 14 from training, as pre-registered (kernel
   `kaggle_kernels/rop_v3_finetune.py`, in the commit that adds this amendment).
c. **Probe dev predictions.** Stricter than Amendment 1(d): the probe kernels' dev predictions were
   never downloaded (only `summary.json`, the log, `metrics.json` and `config.yaml` were fetched).
   Three 1–2-epoch smoke runs in probe mode had been downloaded whole; their dev predictions were
   deleted unopened.
d. **Launch layout.** One kernel per arm-seed trains the five CV folds and the site fold. A kernel
   stopped by its time budget is completed by relaunching only the missing folds; every fold comes
   from exactly one kernel (`scripts/collect_v3_run.py` enforces it, and refuses a run whose lr is
   not the one above).


## Amendment 3 — 2026-10-22, after the A/B/C CV and dev results, before any locked scoring

Disclosure: this amendment was written after the A, B and C cross-validation and dev-half results
(`results/rop/v3/v3_scores.json`, commit e8afe42) were seen. Nothing below is fitted on dev or
locked; every fitted quantity uses CV out-of-fold predictions only. The primary endpoint and its
selection rule are unchanged. The additions are pre-registered here before any of them is scored
on dev, and all of them are scored once, together with A, B and C, at the second locked opening.

a. **Arm E (ensemble).** The mean of the per-image probability vectors of A, B and C, matched by
   seed (E-s42 = mean of A-s42, B-s42, C-s42; likewise s1337; dev/locked: fold checkpoint k of
   each arm averaged, 10 lists). No weights are fitted. Contrast E − C is added as a Holm
   secondary; E − A descriptive.
b. **Ordinal decoder (every arm).** Per photo, the expected stage on the 5-class map,
   s = Σ_k k·p_k (AP-ROP folded onto 4), is cut by four thresholds into stages 0–4; AP-ROP is kept
   whenever it is the photo's argmax. Thresholds maximise image QWK5 and are cross-fitted on the CV
   OOF predictions (fitted on four folds' OOF, applied to the fifth, per arm-seed), so the reported
   CV number is out-of-sample; dev/locked use thresholds fitted on all five folds' OOF. The eye
   rule stays "most severe decoded photo". Reported beside argmax for every arm, as the
   "decoded" variant; argmax remains the pre-registered primary decode.
c. **Arm D (exam-level bag loss).** Encoder = the CV-selected arm, which is B (`dinov2_s`,
   CV image QWK5 0.825 vs A 0.794), same recipe, lr 1e-4, seeds 42 and 1337. For Ostrava rows the
   loss unit is the examination session (the protocol's Ostrava unit): per step the K = 8 (or all,
   if fewer) frames of a session are pooled by log-sum-exp (τ = 1) over per-frame logits into one
   6-way logit, scored by the same class-balanced, label-smoothed CE; every other row keeps
   per-image CE. Singleton sessions are plain CE. Inference is unchanged (per-image predictions;
   units aggregated by the scorer). Contrast D − A stays the pre-registered secondary; D − B is
   descriptive. Expected (panel): CV image QWK down slightly, Ostrava exam QWK up, dev tie.
d. **Reporting.** The pre-registered primary remains locked eye QWK5, A − C, argmax decode. The
   paper reports the decoded variant and E as pre-registered additions made after the dev results,
   stating that.


## Amendment 4 — 2026-10-25, after the A–E CV and dev results, before any locked scoring

Disclosure as in Amendment 3: written after the A–E results (tag `v3-frozen`, commit 937430d)
were seen; nothing below is fitted on dev labels or touches `test_locked`. Arms A–E are frozen:
no retraining, no re-selection. Both new arms use encoder B (`dinov2_s`, the CV-selected arm),
B's recipe and lr 1e-4, seeds 42 and 1337, the five CV folds (no site fold), last-epoch weights,
and are scored once with A–E at the single second opening.

a. **Arm G (native resolution).** B at `preprocess.image_size: 518` (DINOv2's pre-training
   resolution, 37×37 patches; B ran at 392 with interpolated position embeddings). Batch 16; if
   the smoke run shows a T4 out-of-memory, batch 8 with gradient accumulation 2 (same effective
   batch), decided before any main run. Rationale: stage 1 is a 1–2 px line at 392.
b. **Arm F (unsupervised adaptation to the new hospital).** B plus self-ensembling (French et
   al., ICLR 2018): an EMA teacher (decay 0.99) of the student; every step adds 16 unlabelled
   Multi-View DEV photos to the 16 labelled source photos; target loss = mean squared difference
   between the teacher's softmax on a weakly augmented view (resize + flips) and the student's
   softmax on a strongly augmented view (B's training augmentation), counted only where the
   teacher's max probability ≥ 0.90; weight 1.0, ramped linearly from 0 over the first two epochs.
   Dev LABELS are never read by training. Predictions come from the teacher. Because F sees the dev
   images, F's dev numbers are TRANSDUCTIVE (images seen unlabelled, labels unseen) and are labelled
   so; F's locked numbers are inductive (locked images never seen). No hyperparameter of F is tuned.
c. **Endpoints.** The primary (locked eye QWK5, A − C, argmax) is unchanged. The Holm secondary
   family becomes A − B, D − A, E − C, F − B, G − B. Everything else is descriptive.


## Amendment 5 — 2026-10-31, after the A–G results, before any v4 run or locked scoring

Disclosure as in Amendments 3 and 4: written after the A–G results were seen, before any v4 run
or locked scoring; arms A–G are frozen (no retraining, no re-selection). Nothing below is fitted
on dev labels or touches `test_locked`. This is one final improvement round on the CV-selected
arm B (DINOv2 ViT-S/14-reg, 392 px). Out of scope: new backbones, resolution above 392, synthetic
images, mixup/CutMix, pseudo-labelling, relabelling without a clinician, any change to the
Multi-View halves.

a. **Data audit and the v4 training pool.**
   1. *Near-duplicate sweep.* 256-bit dHash (`scripts/build_rop_staging_manifest.py:dhash256`)
      over all 6,700 pool photos, pairs at Hamming distance ≤ 4 (the threshold already calibrated
      on Shenzhen's byte-duplicates). Report: pairs crossing CV folds, pairs with different
      labels, pairs within one Ostrava session. Rule: both members of a cross-fold pair are
      removed from **training** rows of every fold (`exclude_near_duplicates.csv`); validation
      rows are untouched so CV remains comparable with v3. Different-label pairs go to the
      clinician list; nothing is relabelled.
   2. *Ostrava frame quality.* `models/common/data_quality.gradability_metrics` (work size 512) on
      every pool photo. Thresholds are fixed from the two single-photo hospitals (ROP-VL ∪
      Shenzhen, whose photos were clinician-selected): `min_blur` = 5th percentile of `blur_var`,
      `lum_lo`/`lum_hi` = 2.5th/97.5th percentiles of `luminance`, `min_fov` = 5th percentile of
      `fov_ratio`. An **Ostrava** frame passes if `not empty`, `blur_var ≥ min_blur`,
      `lum_lo ≤ luminance ≤ lum_hi`, `fov_ratio ≥ min_fov`. ROP-VL and Shenzhen rows are not
      filtered (their haze is often the disease). Applies to training rows only.
   3. *Cap re-ranking.* The per-infant caps are unchanged (normal 2, other classes 20) but, within
      each (infant, class), the kept frames are the quality-passing frames with the highest
      `blur_var` instead of the first by filename.
   4. *Folds unchanged.* The same infant-to-fold assignment (`splits.csv`) and the same
      validation rows. Validation always scores all 6,700 photos (uncapped, natural prevalence).
   5. *Leave-one-source-out (LOSO).* Two new fold files beside the existing Shenzhen site fold:
      `fold_loso_ostrava` (train = ROP-VL + Shenzhen training rows, val = all 4,285 Ostrava rows)
      and `fold_loso_ropvl` (train = Ostrava + Shenzhen training rows, val = all 1,675 ROP-VL
      rows). `fold_site` is the third (Shenzhen held out). Stage 4/5 exists only in ROP-VL, so
      `fold_loso_ropvl` scores it with zero training examples; reported as such.
   6. The audit report (`results/rop/v4/audit/AUDIT.md`) with every count is committed **before**
      the v4 pool is built, and the pool summary before any kernel is pushed.
b. **Arms.**

   | Arm | Pool | Recipe | Folds | Seeds |
   |---|---|---|---|---|
   | B (frozen) | v3 | B | plus the two LOSO folds only (new runs) | 42, 1337 |
   | H | v4 | B's recipe unchanged | 0–4, site, loso_ostrava, loso_ropvl | 42, 1337 |
   | I | v4 | B plus the v4 recipe below | same 8 | 42, 1337 |

   v4 recipe (every value fixed here, nothing tuned): weight EMA, ema_decay 0.999 per optimizer
   step, evaluated and saved in place of the student (reuses `uda.EMATeacher`); head warm-up:
   epoch 1 trains the classification head only (all other parameters frozen), the OneCycle
   schedule is unchanged; augmentation: rotation ±180°, RandomResizedCrop scale (0.85, 1.0),
   gamma U(0.7, 1.4) with p 0.5, Gaussian pixel noise σ 0.02 with p 0.3, Fourier amplitude swap
   of the centred 3×3 low-frequency block with a random batch partner, p 0.5; label smoothing
   0.10. Everything else equals B.
c. **Comparison rule.** Primary: CV pooled out-of-fold image-level QWK5 (same scorer as v3;
   metrics exclude only the 14 exact duplicates, as before, so v3 and v4 stay comparable; the
   near-duplicate exclusions of 5a apply to training rows only). Paired infant-clustered
   bootstrap, 2,000 reps, seed 42. Guard: mean LOSO QWK5 over the three held-out sources (pooled
   seeds). Adopt I if QWK5(I) − QWK5(B) ≥ +0.010 **and** LOSO(I) ≥ LOSO(B) − 0.010. Else adopt H
   under the same two conditions. Else B stays. Dev results are descriptive; `test_locked` is
   opened once for arms A–I together, after supervisor approval. Pairs added to the
   paired-contrast set: H − B, I − B, I − H.
d. **Budget.** B LOSO: 2 folds × 2 seeds ≈ 1.5 GPU-h. H and I: 8 folds × 2 seeds × 2 arms ≈
   17 GPU-h. Smoke ≈ 1 h. Total ≈ 20 h of the 30-h weekly Kaggle allowance. Kernels:
   `rop-v4-<arm>-s<seed>-a` (folds 0–4) and `-b` (site + LOSO).

### Amendment 5a, revision — 2026-10-31, after the audit, before the v4 pool

Written after the 5a audit (`results/rop/v4/audit/`) and before the v4 pool is built or any
v4 run; no model result was seen. The audit found that the hospital-derived thresholds of 5a.2
measure the camera, not the frame: 903 of 911 RetCam Envision frames, and all 125 Ostrava
stage-3 frames (all from that camera), fell below `min_fov`, which is a camera constant (the
aperture). 5a.2 is therefore replaced for Ostrava. Quality is judged **within each Ostrava
device**, over that device's non-empty Ostrava frames: `min_blur` = 10th percentile of
`blur_var`, `lum_lo`/`lum_hi` = 2.5th/97.5th percentiles of `luminance`, and no FOV rule
(`fov_ratio` stays in `quality.csv` for the record). A frame fails if it is empty, below its
device's `min_blur`, or outside its device's [`lum_lo`, `lum_hi`]. 5a.3 gains an (infant, class)
floor: when fewer than 2 frames of an (infant, class) pass, its 2 best-ranked frames (by
`blur_rank_in_device`, the `blur_var` percentile within the frame's device) are kept, so no
infant loses a class. 5a.1's cross-fold exclusion now requires Hamming distance ≤ 4 **and**
pixel MAD ≤ 5.0 (mean absolute grey-level difference, both images in greyscale resized to
256×256 bilinear), because dHash ≤ 4 collides across distinct Ostrava retinas: the
cross-infant Ostrava pairs differed by 9–47 grey levels, against 3.2 within a session (audit
diagnostic: median, 128 px, inside the field of view). Every Hamming ≤ 4 pair stays on the
clinician list. The rest of Amendment 5 is unchanged.

### Amendment 5, pre-run note — 2026-10-31, before any v4 kernel

Written after the whole-branch code review and before any v4 kernel was pushed; no v4 result
exists. Implementation details the amendment did not fix, stated so the runs are read correctly:

1. *Absent class.* A class with zero training rows in a fold receives zero class-balanced weight
   (`models/common/losses.py:class_balanced_weights`); the present classes are normalised among
   themselves by the unchanged formula. Previously the zero count was clamped to 1, which made the
   absent class the largest weight, and because the class weights also scale the label-smoothing
   term of every sample, the loss was minimised by predicting that class for every true class.
   Only `fold_loso_ropvl` is affected (no stage 4/5 training row). No v3 fold and no result of arms
   A–G changes, because every class is present in every v3 fold; with every class present the
   weights are bit-identical to the old ones (`tests/test_losses.py`).
2. *Fourier swap DC term.* The swap runs on the normalised batch, where the zero-frequency term is
   real with the sign of the centred channel mean. The swap therefore copies the partner's complex
   zero-frequency term and the amplitudes (own phase kept) of the other eight entries of the 3×3
   block, which equals the amplitude swap done in pixel space and then normalised, exactly
   (`tests/test_v4_aug.py`). Taking only the partner's DC amplitude with the image's own sign
   would have mirrored the colour cast whenever the two signs differ.
3. *Cap ranking.* The per-infant cap (5a.3) ranks frames by `blur_rank_in_device`, the
   within-camera percentile rank of `blur_var`. On the v4 pool this selects the identical row set
   as ranking by raw `blur_var`, because no Ostrava (infant, class) cell spans two cameras.
4. *EMA start.* The weight EMA of arm I (decay 0.999, no warm-up, no bias correction) leaves a
   fraction ≈ 0.999^steps of the initial weights in the final average: ≈ 5% on a CV fold
   (~153 optimizer steps × 20 epochs) and ≈ 17% on `fold_loso_ropvl` (~88 × 20). Disclosed, not
   changed.
