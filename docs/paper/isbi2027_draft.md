# What the Label Unit Hides: An Audited Multi-Hospital Benchmark for ICROP Retinopathy of Prematurity Staging

**Working draft for IEEE ISBI 2027 (4-page limit; deadline 2026-10-26).**
Every number in this draft traces to a committed artifact in the repository; the pointer
follows each claim in [brackets]. Numbers that await the seed repeats are marked (s42).
Convert to the IEEE LaTeX template only after `test_locked` is scored — the external
column must be written exactly once.

Authors: P. Nagpure, R. Bait, P. Rangnekar, Y. Shengale, [pediatric ophthalmologist
co-author — REQUIRED before submission], A. Kumar. VESIT, Mumbai.

---

## Abstract (target ≤ 200 words)

Deep models for retinopathy of prematurity (ROP) staging report strong image-level
metrics, but the labels those metrics consume are frequently not image-level facts.
We assemble a 6-class ICROP staging corpus of 8,024 fundus images from 1,760 infants
across four cohorts from three hospitals, gate it with a five-part shortcut audit (patient and duplicate
leakage, metadata decodability, metadata-only baselines, and disease-controlled site
probes), and evaluate with patient-grouped 5-fold cross-validation plus a fully held-out
hospital. We report three findings. (1) **The evaluation unit dominates**: AP-ROP labels
are per-examination-session, and the same predictions score 0.345 image-level recall but
0.840 (CI 0.76–0.90) session-level at a 0.25% false-alarm rate; a patient-clustered
bootstrap puts the image-level figure's interval at [0.25, 0.84], wide enough to be
meaningless. A single fold's "architectural failure," reproduced identically across five
backbones, dissolved under this analysis. (2) **Naive probes overstate adversarial
debiasing**: a DANN site branch improves the standard linear probe while a
disease-controlled probe is never improved at any adversary strength — and at the
strongest setting is significantly *worsened* (+0.066, patient-clustered CI [+0.02,
+0.12]) while AP-ROP recall and QWK collapse. (3) A **taxonomy-faithful head** (ordinal CORN stages plus an
independent AP-ROP branch, as ICROP-3 defines the aggressive form) **matches** flat softmax
once both are trained at the same class prevalence, and the sign of the difference *reverses
between random seeds* (−0.0055 at one, +0.0224 at another, against per-fold spreads of
±0.09); the advantages we first reported were a sampling confound. What survives is not
accuracy but error *structure* — 284 of 506 missed severe cases are called the adjacent
stage rather than normal. On the locked half of the held-out hospital — scored once,
under a pre-registered protocol — the two heads at two seeds are indistinguishable
(eye-level QWK 0.791–0.801), and the per-eye label-unit effect reappears with anatomical
precision: peripheral-disease frames are "missed" at 64–72% on posterior/nasal views but
13% on temporal views, where the ridge actually sits.

---

## 1. Introduction

Retinopathy of prematurity is a leading preventable cause of childhood blindness, and the
constraint on screening it is not imaging capacity but the number of clinicians able to
read the images inside the narrow window where treatment still works. Automated ICROP
staging is therefore an attractive target, and a steady stream of deep-learning systems now
report strong image-level metrics on it.

We argue that a large part of those metrics is decided before any architecture is chosen,
by three methodological choices that are rarely stated and never ablated: the unit the
labels actually describe, the probe used to claim site-invariance, and the class-balancing
sampler. Each is a place where a reported number can move further than any architectural
contribution in the same paper.

The label unit matters because ROP annotations are frequently not image-level facts. In
several public and institutional corpora, including the one we use, a diagnosis code is
recorded per *examination session* and inherited by every frame captured in it, many of
which are peripheral or poor views that do not show the finding. Evaluating per image
against such labels charges the model for annotations that were never about those frames.
The consequences are visible in the literature at both extremes: ROPDeepX [C7] splits
RIDIRP at image level with roughly 32 frames per infant and reports 99.7% accuracy — the
canonical illustration of what patient leakage buys — while VEA-Net [C6], which uses honest
subject-level folds on the same Ostrava data we use, folds AP-ROP into Stage 4 (contrary to
ICROP-3 [C3], which defines the aggressive form as explicitly outside the stage ladder) and
reports accuracy alone, at 90.4% against a majority-class prior of 89.5%.

Site invariance is claimed the same loose way. Hospitals in these corpora differ by camera,
so hospital identity is written into every image; the standard remedy is domain-adversarial
training [C5] and the standard evidence is a linear probe whose accuracy falls. That probe
is confounded by disease prevalence, which also varies by hospital, and we show it reports
success where a disease-controlled probe reports none.

Our contributions are: (i) an audited multi-hospital ICROP staging corpus with
patient-grouped folds, a held-out hospital, and a metadata-only performance floor;
(ii) a quantification of how far the evaluation unit alone moves reported AP-ROP detection
(0.345 → 0.840 from identical predictions), shown to be independent of the model;
(iii) a dose–response measurement, with patient-clustered intervals, showing adversarial
debiasing that the naive probe credits and a disease-controlled probe never does — the
controlled probe rising significantly at the strongest adversary; and (iv) a
taxonomy-faithful head — ordinal stages plus an independent AP-ROP branch — that matches
flat softmax under 5-fold cross-validation, with a seven-arm ablation attributing the
result to its components. We deliberately do *not* claim an architectural advance: our own
single-fold result suggested one, and full cross-validation withdrew it. That withdrawal is
part of the contribution, because the same single-fold pattern reproduced across five
backbones and would have been reported by a less paranoid protocol.

## 2. Data and audit

**Inclusion cascade** [inclusion_flow.json — every layer recomputed from the manifest and
the script refuses to emit a flow whose reasons do not sum to the drop]:

```
13,037  images acquired (10 public datasets + one hospital corpus = 11 sources)
 2,590  − sources with no ICROP stage label (segmentation masks, plus-only grades, n=91 sample)
    20  − Shenzhen images that are perceptual-hash duplicates of higher-resolution ROP-VL copies
10,427  images with an ICROP stage label (2,015 infants)
 1,295  − not ROP (hemorrhage, toxoplasma, hamartoma, optic-nerve hypoplasia)
 1,063  − st.p. ROP: treated/regressed — a post-treatment appearance is not a stage
    45  − ICROP stage 0 (immature retina without ROP; cannot support its own class)
 8,024  eligible for 6-class staging (1,760 infants)
        → 6,700 CV pool (1,528) · 663 held-out-hospital dev · 661 held-out-hospital LOCKED
 3,588  − per-infant frame caps, TRAINING ONLY (≤20/infant/class, ≤2 normals)
 3,112  images used to fit each model
```

The two largest exclusions are clinical judgements, not hygiene: `other_pathology` is the
same 1,295 images that make 43% of this corpus's binary "ROP-positive" class not ROP, and
`treated` would otherwise teach the model to call a successfully treated eye diseased.
Frame caps apply to training only — validation scores every eligible image, which is
precisely why one heavily-imaged infant can dominate an AP-ROP validation metric (§4.3).

**Corpus** [splits_summary.json, fold CSVs]: 6,700 CV images / 1,528 infants — Ostrava
4,285/148 (de-identified; per-session DG codes), ROP-VL 1,675/983, Shenzhen 740/397 —
plus held-out hospital "multiview" 1,324/232, split into dev (663) and a locked half
(661) scored exactly once [state whether done]. Classes: Normal 3,837 · S1 577 · S2 959
· S3 620 · S4/5 89 · AP-ROP 618. Ostrava supplies 64% of images from 10% of infants
(~29 frames/infant) → all statistics patient-grouped or patient-clustered. Training
frame-capped (≤20/infant, 2 for normals); validation uncapped.

**Five audit gates** [staging_shortcut_audit.json]: G1 patient leakage 0 · G2 duplicate
leakage 0 (MD5 + calibrated dHash) · G3 site decodable from trivial metadata at 0.997
(names the confound) · G4 metadata-only classifier macro-F1 0.187 = the floor every model
must clear (all clear it 2.6×) · G5 disease-controlled embedding probe (see §4.3).

## 3. Methods

Shared backbone (EfficientNetV2-S, ImageNet), identical preprocessing for every arm
(circle-crop, CLAHE, letterbox, 384²). Structured head: (a) CORN ordinal stage branch,
K−1 conditional logits, AP-ROP images *masked* from the ordinal loss (ICROP-3: the form
has no stage); (b) independent AP-ROP sigmoid branch, √-damped pos-weight, threshold 0.5
fixed a priori; (c) DANN site adversary behind a gradient-reversal layer, λ ramped.
6-class decode: AP-ROP if p_a>0.5 else rank; probabilities compose to sum to 1 → scored
by the same metrics code as every baseline. Last-epoch checkpoints (no argmax-over-
epochs); natural sampling (§4.2 shows why); 20 epochs OneCycle.

## 4. Experiments

### 4.1 Equivalence, measured at one sampling regime [matched_equivalence.json]

**The confound is now closed.** Our first version of this comparison varied *two* things:
the committed flat baseline trained with a class-balancing sampler while the structured arm
trained at natural prevalence (a weighted sampler destroys the CORN head, §4.2). Fold-0
evidence put the sampling effect on flat macro-F1 at +0.047 — five times the +0.009
"architecture effect" we were reporting from it. We therefore trained the missing cell:
flat softmax at natural prevalence, same folds, seed, backbone and preprocessing.

| 5-fold CV | flat, **natural** | structured, natural | flat, weighted *(what we used to report)* |
|---|---|---|---|
| macro-F1, **seed 42** | **0.698 ± 0.099** | **0.692 ± 0.086** | 0.683 ± 0.118 |
| macro-F1, **seed 1337** | **0.686 ± 0.109** | **0.708 ± 0.097** | — |
| per-fold macro-F1 (s42) | .545 / .752 / .656 / .745 / .790 | .553 / .757 / .671 / .718 / .762 | — |
| AP-ROP recall (s42) | 0.723 | 0.671 | — |
| Stage 2 recall (s42) | 0.615 | 0.632 | — |
| held-out hospital macro-F1† (s42 / s1337) | 0.8182 / 0.8242 | 0.8180 / 0.8201 | 0.792 / — |
| held-out hospital QWK (s42 / s1337) | 0.9470 / 0.9540 | 0.9462 / 0.9600 | 0.930 / — |

†over the 4 classes the held-out hospital has; a 6-label macro adds two structural zeros.

**The head difference changes sign between seeds.** Structured minus flat is **−0.0055** at
seed 42 and **+0.0224** at seed 1337 — the two seeds disagree about which architecture is
better, at a magnitude an order of magnitude below the per-fold spread (±0.09 to ±0.11) in
both cases. Per fold the sign is mixed within each seed as well (structured ahead on 3 of 5
folds at seed 42, 3 of 5 at seed 1337, but not the same three). A difference that reverses
direction when only the random seed changes is not an architecture effect. Reporting either
seed alone — the near-universal practice under compute constraints — would have yielded a
confident claim in whichever direction that seed happened to land, and we would have had the
CV table to back it.

**Two claims we previously made do not survive.** "Structured wins Stage 2 recall on every
fold" was measured against the weighted arm; against the matched arm it holds on 3 of 5.
"Structured transfers better to a new hospital" is gone entirely — at seed 42 the two arms
score 0.8180 and 0.8182 macro-F1, and at seed 1337 flat is *ahead* on macro-F1 (0.8242 vs
0.8201) while structured leads on QWK (0.9600 vs 0.9540). Both apparent advantages were the
sampler, not the head. What remains for the structured head is what §4.6 measures and no
aggregate captures: taxonomy-faithful outputs, and errors that land *next door* on the
severity ladder rather than at the far end.

Five-backbone fold-0 benchmark: spread 0.028 macro-F1 across 7× parameters — no ranking
claimed; every model clears the 0.187 metadata bar. The fold-0 "softmax failure on AP-ROP"
(recall 0.16–0.22 across all five backbones) did not generalize — flat reaches 0.83–0.97 on
folds 1–4. Fold 0's validation split holds the one heavily-imaged, session-labelled AP-ROP
infant (§4.3); five architectures "failed" identically because they shared an evaluation
unit, not a defect. Fold-0 ECE is 0.196 vs 0.057–0.092 elsewhere
[structured_bootstrap_calibration.json] — the miscalibrated fold is the mislabelled fold.

### 4.2 Seven-arm component ablation (fold 0, s42) [ablations_fold0.json]

Table: flat-weighted 0.603 QWK · flat-natural 0.580 · full 0.569 · w=0 0.648 · −branch
0.575 · −ordinal 0.394 · +sampler 0.129. Readings: the sampling confound is a non-issue
for flat but fatal for CORN (conditional thresholds calibrate to training prevalence —
a class-balanced sampler feeds them a false one; QWK 0.129, S4/5 precision 0.043);
removing the ordinal head is the one large head effect; nothing else separates within
single-fold noise.

### 4.3 The evaluation unit is the story [session_level_arop.json, ablation_session_level.json]

⚠️ **REWRITE REQUIRED — the previous version of this section overclaimed. Audited
2026-08-14; do not submit the old wording.** What the data actually supports:

**Composition first, because it governs everything else.** The 618 AP-ROP images in the CV
pool come from 104 infants, but **470 of them (76%) are one infant** (`ostrava:43`) imaged
across 16 sessions (median 24 frames/session). The remaining 103 infants are ROP-VL with
1–2 frames each, where max-over-frames aggregation is a no-op. So the 119 "session" units
are 16 genuine multi-frame sessions plus 103 single-patient units.

**Decomposing the 0.345 → 0.840 jump** (pooled, seed 42): weighting each unit equally with
NO aggregation already gives 0.693; max-over-frames adds the remaining 0.147. So ~70% of
the headline movement is a change of *population weighting*, not of evaluation unit. The
two must be reported separately.

**Where aggregation genuinely acts** — Ostrava, the only source with real session structure:
image-level recall 0.204 → session-level 0.875 over 16 sessions. That is the clean
demonstration, and its n is 16 sessions from **one infant**: a case study, not a benchmark.

**The "structural ceiling" framing is wrong.** Image-level AP-ROP **ROC-AUC is 0.917** — the
model ranks AP-ROP well per frame; 0.345 is an artifact of the fixed 0.5 threshold against a
population dominated by one infant's unrepresentative frames. The defensible claim is
therefore *not* "image-level metrics cannot detect AP-ROP", but: **image-level recall at a
fixed threshold is not a stable quantity on session-labelled data** — its patient-clustered
bootstrap interval is [0.25, 0.84], and its value is set by corpus composition rather than
by detection ability.

**Supporting, and it carries its own control** [session_level_by_seed.json]: pooled
out-of-fold, **70.4%** of Ostrava's AP-ROP-labelled frames score p<0.1 (470 frames, 16
sessions; the median session has 74.6% of its frames below 0.1) — while in the other
sources, which label 1–2 frames per infant rather than stamping a session, only **10.1%**
do (148 frames). Same model, same class, seven times the "label carried, appearance absent"
rate precisely where the annotation is per-session. The frames are not mysteriously hard;
they inherited a label that was never about them. ⚠️ Earlier drafts quoted 47% from
`session_level_arop.json`; that artifact was computed on a superseded prediction set (its
fold-0 image recall is 0.311 against the canonical 0.212) and is now marked as such.
Session scoring lifts every ablation arm (0.71–0.95, overlapping CIs), so the effect is
model-independent.

**Open decision for the authors:** either (a) reframe this as a methodological caution
supported by one well-imaged case plus the bootstrap instability, or (b) obtain a corpus
with multi-session AP-ROP imaging in more than one infant. (a) is honest and defensible
at 4 pages; (b) is not achievable before the deadline.

### 4.4 Naive probes overstate adversarial debiasing [dann_probe_ci.json]

**Measured 2026-08-15 with patient-clustered bootstrap CIs** (2000 replicates resampling
the 128 normal-class patients; the same resample scores every arm, so difference CIs are
paired). Dose–response, disease-controlled linear probe on normal-only images (prevalence
held constant), all three cells the SAME structured head, all three checkpoints verified
by hash against their kernel of origin:

| | w=0 | w=0.5 | w=2.0 | flat no-DANN | untrained |
|---|---|---|---|---|---|
| naive probe | 0.790 [.73,.88] | 0.809 [.75,.89] | 0.867 [.82,.93] | 0.859 | 0.890 |
| disease-controlled | **0.866** [.81,.92] | **0.852** [.79,.91] | **0.917** [.87,.96] | 0.882 | 0.894 |

Paired differences, controlled probe: w=0→0.5 is noise (−0.015, CI [−0.073, +0.047]);
**w=2.0 is a real increase** over both w=0.5 (+0.066, CI [+0.022, +0.115]) and w=0
(+0.051, CI [+0.006, +0.100]). So the defensible claim is not "monotonic" but sharper:
across its whole dose range the adversary **never reduces** disease-controlled site
decodability, and at the strength that visibly moves the naive probe it *significantly
increases* it — while destroying AP-ROP recall (0.168) and QWK (0.525). The naive probe
cannot see any of this. Cross-site transfer without the adversary: QWK 0.942 vs 0.946
with it. DANN removed the disease–site correlation, not site appearance; the standard
probe cannot tell the difference.

Two provenance notes the CIs forced into the open, kept because they are evidence: (i)
earlier drafts quoted 0.867→0.885→0.921 and called it monotonic — the w=0.5 cell's 0.885
came from a sibling training run of the same config whose checkpoint a later kernel
version overwrote; on the canonical checkpoint (the one served publicly, hash-verified)
the cell is 0.852. The 0.033 sibling-run spread is a direct measurement of training-run
variability in the probe — the same size as the "step" it had been read as. (ii) The w=0
row had earlier been carried over from w=0.5; an audit caught it and the probe was run on
the recovered w=0 checkpoint. Both corrections made the finding stronger.

### 4.5 Per-hospital calibration — and the control that shrinks it [per_source_calibration.json]

Pooled out-of-fold calibration (6,700 images, every image scored by the model that did not
train on it) is ECE **0.109**, the model over-confident by the same margin. Split by
hospital, the raw numbers look alarming:

| | Ostrava | ROP-VL | Shenzhen | pooled |
|---|---|---|---|---|
| n images / infants | 4,285 / 148 | 1,675 / 983 | 740 / 397 | 6,700 / 1,528 |
| accuracy | 0.757 | 0.812 | 0.897 | 0.786 |
| ECE | **0.153** [.08,.24] | 0.049 [.03,.07] | 0.018 [.01,.04] | 0.109 |
| confidence − accuracy | **+0.153** | +0.049 | −0.011 (ns) | +0.109 |

An 8.4× spread, and the pooled figure describes no site. **We do not report it that way**,
because ECE confounds calibration with case difficulty and the sites do not share a class
distribution. Holding the class constant — normal, the only class present at all three —
collapses the spread to 0.012–0.027 (2.3×), and every site is well calibrated on the class
they share (Ostrava 0.027, ROP-VL 0.022, Shenzhen 0.012).

So the defensible claims are narrower and more useful than the raw table: the model is
over-confident *where the cases are hard*, not *at a particular hospital*; and a small
residual site effect does survive the control at Ostrava (+0.027, CI excludes 0), an order
of magnitude below what the uncontrolled number implied. This is the third place in this
work where an uncontrolled aggregate overstated a site effect — after the naive site probe
(§4.4) and image-level AP-ROP recall (§4.3). The control was cheap every time.

### 4.6 The decision a clinician actually makes [treatment_endpoint.json]

No clinician acts on a macro-F1. We therefore re-score the *same* out-of-fold predictions as
the binary decisions a screening service feeds. ICROP-3 Type 1 is defined on zone × stage ×
plus and this corpus has no usable zone axis (Multi-View records zone for 13 of 232
patients), so we do not claim Type 1 detection. We report two collapses we can defend:
**referral-warranted** (any ROP) and **severe** (stage 3, stage 4/5, or AP-ROP — the end of
the ladder where treatment decisions live). *Severe* deliberately over-includes, since a
stage 3 in zone III without plus is not Type 1; we name the direction of the bias so the
numbers are not read as Type 1 performance.

| endpoint | unit | n (pos) | sensitivity [clustered CI] | specificity | PPV |
|---|---|---|---|---|---|
| referral-warranted | image | 6,700 (2,863) | 0.903 [0.870, 0.944] | 0.951 | 0.932 |
| referral-warranted | **session** | 1,722 (963) | **0.989** [0.982, 0.995] | 0.884 | 0.915 |
| severe | image | 6,700 (1,327) | 0.619 [0.469, 0.876] | 0.981 | 0.888 |
| severe | **session** | 1,722 (491) | **0.904** [0.878, 0.929] | 0.970 | 0.923 |

The unit effect reappears at the endpoint that matters, and larger: severe-disease
sensitivity is 0.619 per image and 0.904 per session, and the image-level interval
([0.469, 0.876]) is too wide to support any claim at all. This is the same phenomenon as
§4.3, now expressed in the currency a screening programme is commissioned in.

**Miss anatomy.** Reporting a miss count hides the only thing that matters about a miss:
what the model said instead. Of the 506 severe images missed, **284 were called stage 2** —
an adjacent error that any graded referral protocol still escalates — and **106 were called
normal**, the failure that actually costs an eye. At the session unit, 47 of 491
severe-disease sessions were missed entirely. The concentration of errors *next door* on
the severity ladder is the one place the ordinal (CORN) head earns its keep: the equivalence
result of §4.1 says it buys no accuracy, and this says it buys error *structure*, which is
worth more clinically than the macro-F1 it does not move. We know of no ROP paper that
reports this decomposition; a confusion matrix contains it, and none we surveyed reads it
out.

### 4.7 Held-out hospital, locked — opened once [locked_eval/locked_table.json]

The locked half (661 images, 116 infants) was scored on 2026-08-18, exactly once, under a
protocol committed before any locked inference existed [LOCKED_EVAL_PROTOCOL.md] and
amended only on dev-half evidence; the evaluation script refuses to run twice. All four
matched arm-seed cells, each as mean ± sd over its five fold checkpoints:

| arm, seed | image-level QWK [clustered CI] | eye-level QWK | stage 1–3 eyes missed |
|---|---|---|---|
| structured, 42 | 0.523 ± 0.018 [0.433, 0.594] | **0.801 ± 0.028** | 17.8% |
| structured, 1337 | 0.509 ± 0.016 [0.417, 0.581] | **0.791 ± 0.013** | 20.0% |
| flat, 42 | 0.561 ± 0.023 [0.464, 0.635] | **0.795 ± 0.007** | 17.2% |
| flat, 1337 | 0.553 ± 0.018 [0.456, 0.629] | **0.793 ± 0.021** | 19.3% |

Three results. **(1) Equivalence replicates externally.** At the eye unit the four cells
span 0.791–0.801 — two heads, two seeds, indistinguishable at a hospital none of them ever
saw. **(2) The label-unit mechanism replicates, with retinal anatomy as the witness.** The
dev half revealed (and the locked half confirms) that Multi-View's labels are per-eye
diagnoses stamped onto every acquisition view: stage 1–3 frames are predicted normal at
71.7% for posterior-pole views and 63.4% for nasal, but **12.7% for temporal views — where
the ROP ridge anatomically sits**. Normal recall is 0.996, so this is not over-calling.
Image-level QWK ≈ 0.53 against eye-level ≈ 0.80 is §4.3's session effect at a new
hospital, in per-eye form, on data that was sealed while the claim was written. **(3) Our
own "cross-site QWK 0.95" was never leave-hospital-out.** The site fold holds out the
Shenzhen *cohort* while ROP-VL — the same hospital (§2) — stays in training; the first
genuinely-new-hospital number is 0.79–0.80 at the eye unit, and the gap between those two
figures is the price of that distinction, which most multi-"site" medical-imaging
evaluations do not draw.

Pre-registered flags hold regardless of their values: stage 4/5 (n=4, recall 0.75) and
AP-ROP (n=12, recall 0.50) are **not externally validated** at these n. Dev and locked
halves agree in structure (dev: image 0.40–0.44, eye 0.71–0.73), so the amendment was not
steered by a lucky half.

### 4.8 Replication across seeds

Structured 5-fold at seed 1337: macro-F1 0.708±0.097, QWK 0.839, cross-site QWK 0.960,
session-level AP-ROP 0.832 (vs 0.840 at s42, identical 119 units). Fold 0 is the lowest
fold at both seeds — the fold-0 pattern is a property of the split, not seed luck.
[s2024 pending.]

## 5. Discussion & limitations

**Read our agreement against the right benchmark.** ROP *stage* is the least reliable axis
of ICROP among human experts: 7 experts reading the same wide-field photographs reach
Fleiss κ = 0.24 on stage, and a single expert re-reading 8 weeks later reaches only
κ = 0.56 [C1]; across 1,553 expert-vs-expert comparisons, stage disagreement runs 40% —
while agreement on treatment-requiring (type 1) disease exceeds 95% [C2]. We therefore
report **unweighted** κ alongside QWK, because the human literature is unweighted and
quadratic weighting inflates ordinal agreement: our pooled out-of-fold κ is 0.655 (s42) /
0.681 (s1337) unweighted against 0.761 / 0.783 quadratic. We explicitly do **not** claim
superhuman agreement — Fleiss κ among seven mutually-disagreeing experts is a different
quantity from Cohen κ between one model and one institution's coding practice, and part of
what our model has learned is that practice. Two consequences follow. First, our
label-noise finding is not a defect peculiar to this corpus; unreliable staging labels are
the field's baseline condition, which is why an evaluation-unit error propagates so widely.
Second, the axis where the reference standard *is* trustworthy — treatment-relevant,
aggressive disease — is exactly the axis our session-level analysis measures, which argues
for weighting AP-ROP detection above 6-class QWK when judging clinical utility.

**What generalises beyond this corpus.** Two of our three measurement findings are
properties of protocol, not of data. The evaluation-unit effect appears in every one of
seven ablation arms, including arms whose training collapsed, so it cannot be attributed to
a particular head, sampler or adversary; any corpus whose labels are recorded per session
or per eye and evaluated per frame will understate detection by a similar mechanism. The
probe result is likewise about instrumentation: any study claiming adversarial site removal
from an all-class probe is measuring a quantity that disease prevalence can satisfy. The
fold-0 episode is the third: a defect reproduced across five backbones on one fold was
still a property of the split, which argues that single-fold ablations — common in this
literature under compute constraints — are not evidence about architectures at all.

**Limitations.** One camera system per hospital, so site and device are not separable here
and our "site" adversary is equally a device adversary. Ostrava staging labels are
machine-translated diagnosis codes pending clinician adjudication of ~300 stratified images
[status]; the expected κ should be judged against the [C1] inter-expert benchmark rather
than against 1.0. Two seeds so far, with the third pending. Stage 4/5 rests on 89 training
images and 4 in the locked half and supports no claim whatever its score, as does AP-ROP
externally at 12. Session-level aggregation by maximum is a screening-appropriate rule but
inflates sensitivity relative to a per-frame reading, and we report its false-alarm rate on
normal sessions alongside; the normal-session denominator must be stated explicitly, since
sessions defined only where session structure exists (Ostrava) and sessions defined by
patient fallback give 0.25% and 1.16% respectively.

Equivalence is itself the result we would ask the field to take seriously: a
taxonomy-faithful head costs nothing, so clinically structured outputs are available for
free — but the reported architecture gains that motivate such heads should be re-measured
at the correct label unit before any of them is credited.

## Compliance with ethical standards

**Human subjects.** This work is a retrospective secondary analysis of four previously
published, de-identified retinal image datasets. No new patient data was collected, no
participant was contacted, and no clinical decision was influenced. Ethical approval for
the original acquisitions rests with the releasing institutions and is documented in their
source publications; no additional approval was sought, consistent with secondary use of
openly licensed de-identified data.

**Data sources and licences.** Ostrava — *Retinal Image Dataset of Infants and ROP*
(Ostrava, Czechia), figshare collection 6626162, **CC0**. ROP-VL (Shenzhen Eye Hospital,
China), doi 10.6084/m9.figshare.30143461, **CC BY 4.0**. Shenzhen ROP (Zhao et al., 2024),
doi 10.6084/m9.figshare.25514449, **CC BY 4.0**. Multi-View ROP (China), HuggingFace
`lijuanliao/ROP-DATASET`, **CC BY-NC 4.0**. The held-out hospital is the non-commercial
one: we neither redistribute it nor make any commercial claim for a model evaluated on it,
and readers reusing this protocol commercially must substitute a different held-out site.
Datasets were obtained from their primary repositories; third-party mirrors exist for
several and are deliberately not cited, because at least one is an unattributed re-upload.

**Re-identification risk, and what we did about it.** Two of the four corpora encode
patient attributes in image *filenames*: Ostrava embeds patient number, sex, gestational
age, birth weight and postnatal age; ROP-VL embeds examination dates at day granularity.
Sex, gestational age and birth weight for a named hospital's neonatal cohort is plausibly
re-identifying, and a date of service is a direct identifier under most frameworks —
regardless of the permissive licence, which grants reuse rights but does not make the
content non-identifying. We therefore de-identified the working image tree, and no
artifact released with this paper contains these filename forms.

We report, because it is methodologically relevant rather than merely embarrassing, that
achieving this took three attempts and each miss had a different shape: filenames were
first stripped from paths while byte-identical images remained published; then result JSONs
were found carrying the original filenames as record keys, invisible to a guard that
matched only on paths; then the same identifiers surfaced inside a *docstring* of the
script written to remove them. Each was found by widening the check rather than by
re-reading the code. Our repository therefore verifies de-identification by file content
and by image hash, not by filename, and treats a commit message asserting removal as
unverified until a fetch proves it. Any group publishing infant retinal data should assume
the same class of miss.

**Intended use.** The models described here are research prototypes. They have no
regulatory clearance, were never validated prospectively, and are not intended for clinical
use. All reported operating points are research measurements, not deployment settings.

**Data and code availability.** All four datasets are publicly available at the DOIs above.
Fold assignments, out-of-fold predictions, audit reports and every analysis script are
released with the paper; the corpus itself is not redistributed. The locked held-out half
was scored exactly once, after all other results were fixed.

## References (working list — full details in `human_agreement_ceiling.md`)

- **[C1]** Gschließer A, et al. Inter-expert and Intra-expert Agreement on the Diagnosis
  and Treatment of Retinopathy of Prematurity. *Am J Ophthalmol* 2015;160(3):553–560.
- **[C2]** Campbell JP, et al. Diagnostic Discrepancies in Retinopathy of Prematurity
  Classification. *Ophthalmology* 2016;123(8):1795–1801.
- **[C3]** ICROP3 Committee. International Classification of Retinopathy of Prematurity,
  Third Edition. *Ophthalmology* 2021.
- **[C4]** Shi X, Cao W, Raschka S. Deep neural networks for rank-consistent ordinal
  regression (CORN). *Pattern Anal Appl* 2023;26:941–955.
- **[C5]** Ganin Y, et al. Domain-Adversarial Training of Neural Networks. *JMLR* 2016.
- **[C6]** VEA-Net (Frontiers 2026) · **[C7]** ROPDeepX (Discover AI 2026) ·
  **[C8]** Peng et al. *IEEE TMI* 2021 — the three closest rivals (see §1).

## Checklist before submission

- [ ] CLAIM checklist filled and cited
- [ ] Unweighted-κ column added to the §4.1 table (0.655 s42 / 0.681 s1337 pooled)
- [ ] Seed repeats folded into §4.1 (mean ± SD over 3 seeds)
- [ ] test_locked scored once; §4.5 written
- [ ] White-background figure variants (architecture, session-level, dose–response)
- [ ] Clinician co-author on the author list with adjudication κ in §5
- [x] Data-use terms checked and licences cited for all four sources (see *Compliance with
      ethical standards*). ⚠️ Multi-View — the **held-out test hospital** — is CC BY-**NC**:
      no redistribution, and no commercial claim may rest on the held-out result.
