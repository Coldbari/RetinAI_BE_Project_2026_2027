# What the Label Unit Hides: An Audited Multi-Hospital Benchmark for ICROP Staging in Retinopathy of Prematurity

**Target:** IEEE ISBI 2027 · 4 pages technical + optional 5th (non-technical only) · single-blind · deadline 26 October 2026
**Status:** complete draft, ready for LaTeX conversion. Every number traces to a committed artifact; pointers in `[brackets]` are for the authors and are stripped at conversion.

**Authors:** P. Nagpure, R. Bait, P. Rangnekar, Y. Shengale, **[paediatric ophthalmologist co-author — REQUIRED before submission]**, A. Kumar.
Vivekanand Education Society's Institute of Technology, Mumbai, India.

---

## Abstract

*(196 words)*

Automated ICROP staging for retinopathy of prematurity (ROP) is usually reported as an accuracy number, and that number is usually decided before the architecture is. We assemble a 6-class staging corpus of 8,024 images from 1,760 infants across four cohorts, gate it with a five-part shortcut audit, and evaluate under patient-grouped 5-fold cross-validation plus a held-out hospital whose second half was scored exactly once under a pre-registered protocol. Three results follow. First, the evaluation unit dominates: on the locked set the same predictions score quadratic-weighted kappa 0.51–0.56 per image and 0.79–0.80 per eye, a gap that holds across two architectures and two seeds and is therefore a property of the labels, not the model. The misses are anatomically directed — posterior-pole frames are called normal 71.7% of the time against 12.7% for temporal frames — because a frame that does not contain the ridge is, as an image, normal. Second, a taxonomy-faithful ordinal head and a flat softmax are indistinguishable, and the sign of their difference reverses between seeds. Third, a screening head with 0.970 internal AUC collapses to 0.691 externally and flags 663 of 663 images. We report the negative results because they are the finding.

**Index Terms** — Retinopathy of prematurity, ICROP staging, evaluation methodology, external validation, shortcut learning.

---

## 1. Introduction

Retinopathy of prematurity is a leading preventable cause of childhood blindness. The bottleneck in screening it is not imaging capacity but the number of clinicians able to read the images inside the narrow window where treatment still works, which is why automated ICROP staging has attracted a steady stream of deep-learning systems reporting strong image-level metrics.

We think a large part of those metrics is decided before any architecture is chosen. Three methodological choices — the unit the labels actually describe, the probe used to claim site-invariance, and the split granularity — each move a reported number further than the architectural contribution in the same paper, and none of them is routinely ablated.

The label unit is the most consequential. ROP annotations are frequently not image-level facts: in several corpora, including the one we use, a diagnosis is recorded per examination session and inherited by every frame captured in it, many of them peripheral or poorly-centred views that do not show the finding. Scoring per image against such labels charges the model for annotations that were never about those frames. The literature contains the symptom without the diagnosis. Wang et al. [C8] report referral-warranted ROP at image, eye and patient level and observe F1 falling 0.956 → 0.915 → 0.898; the unit is acknowledged, the effect is small, and it points downward. We find the opposite sign and roughly three times the magnitude, and we show why: when the label's native unit is the eye, per-image scoring does not merely lose precision, it systematically understates agreement.

Split granularity is the second. A recent study [C1] applies EfficientNetB4 to the same Ostrava corpus we use and reports 0.98 stage accuracy and 1.000 test accuracy for plus disease under an 80/16/4 **image-level** split of a dataset averaging roughly 32 frames per infant, with no external validation and no device-stratified analysis despite three camera systems. Its own limitations section names the risk. We take that as the control condition rather than as a competitor: it is what our corpus produces when the audit is skipped.

Site invariance is the third. Hospitals differ by camera, so hospital identity is written into every image; the standard remedy is domain-adversarial training [C5] and the standard evidence is a linear probe whose accuracy falls. That probe is confounded by disease prevalence, which also varies by site.

Our contributions: (i) an audited multi-hospital ICROP staging benchmark with patient-grouped folds, a metadata-only performance floor, and a locked external half scored once; (ii) a quantification of the evaluation-unit effect, decomposed into population re-weighting and frame aggregation, and shown to be model-independent; (iii) two negative results — architectural equivalence, and a disease-controlled site probe that never improves — reported at the same weight as the positive ones.

---

## 2. Materials and Methods

### 2.1 Corpus and inclusion

We pooled 13,037 images from eleven sources and reduced them to 10,427 stage-labelled images from 2,015 infants, then applied clinical exclusions: 1,295 images labelled ROP-positive that record other pathology (haemorrhage, toxoplasmosis, hamartoma, optic-nerve hypoplasia), 1,063 treated or regressed eyes whose post-treatment appearance is not a stage, and 45 stage-0 images too few to support a class. The eligible corpus is **8,024 images from 1,760 infants** in six classes (Normal 3,837; S1 577; S2 959; S3 620; S4/5 89; AP-ROP 618) [`inclusion_flow.json`]. Every exclusion is counted and the reasons are required to sum to the observed drop; the script refuses to emit a flow that does not balance.

That first exclusion deserves emphasis. **43% of this corpus's binary "ROP-positive" class is not ROP.** A model trained on the uncorrected labels is an abnormal-retina detector, and reports it as an ROP result.

### 2.2 Reference standard

Stages follow ICROP-3 [C3], with AP-ROP treated as an aggressive *form* outside the stage ladder rather than as a rung on it. Labels are as released by the source institutions; we did not re-grade, and we state plainly that this is the study's principal limitation. Independent adjudication of a stratified subsample by a paediatric retina specialist is in progress and will accompany the camera-ready version.

### 2.3 Splits

Folds are patient-grouped (5-fold, stratified by class), and a fourth cohort from a hospital contributing to no fold is held out entirely: 1,324 images from 232 infants, divided into a development half (663) used during method selection and a **locked half (661 images, 116 infants) opened once, on 18 August 2026, under a protocol committed to version control beforehand** [`LOCKED_EVAL_PROTOCOL.md`]. It has not been re-scored since.

### 2.4 Shortcut audit

Five gates run before any model is trained [`staging_shortcut_audit.json`]: (G1) patient leakage across folds — zero; (G2) exact and perceptual duplicate leakage, MD5 plus a dHash calibrated against known byte-duplicates — zero, closest distance 15; (G3) site decodability from trivial metadata — **balanced accuracy 0.997 against a 0.333 chance rate**, which names the confound rather than removing it; (G4) a metadata-only classifier using no pixels, whose CV macro-F1 of **0.187** is the floor every image model must clear; (G5) a disease-controlled embedding probe (§3.3). On an earlier binary split, image dimensions alone reached AUC 0.911 — the shortcut is not hypothetical.

### 2.5 Models

All arms share an ImageNet-initialised EfficientNetV2-S backbone and identical preprocessing (circle-crop, CLAHE, letterbox, 384²). The **structured head** has three parts: a CORN ordinal branch [C4] over stages with AP-ROP images masked from the ordinal loss, since ICROP-3 gives that form no stage; an independent AP-ROP sigmoid branch; and a DANN site adversary behind a gradient-reversal layer [C5]. The **flat baseline** is a 6-way softmax. Both are trained at natural class prevalence — an earlier version of this comparison varied the sampler as well as the head, and the sampling effect was five times the apparent architecture effect. Checkpoints are last-epoch by protocol, never selected by validation argmax.

### 2.6 Evaluation

The primary metric is quadratic-weighted kappa (QWK). Because the human ROP literature reports *unweighted* kappa, we report both: quadratic weighting inflates ordinal agreement when most errors are adjacent, and reporting only QWK would make our numbers incomparable to the only benchmark that matters. For context, inter-expert agreement on ROP stage among seven graders is Fleiss κ = 0.24, intra-expert κ = 0.56, with a 40% stage disagreement rate [C9, C10]. Metrics are computed per image (primary) and per eye (co-primary), the latter by taking the most severe prediction across an eye's frames — the same operation by which the labels were assigned. All locked-set figures are means over five checkpoints with their standard deviation.

---

## 3. Results

### 3.1 The evaluation unit decides the result

On the locked half, scored once:

| Arm | QWK per **image** | QWK per **eye** | Δ |
|---|---|---|---|
| Structured, seed 42 | 0.523 ± 0.018 | 0.801 ± 0.028 | +0.278 |
| Structured, seed 1337 | 0.509 ± 0.016 | 0.791 ± 0.013 | +0.282 |
| Flat, seed 42 | 0.561 ± 0.023 | 0.795 ± 0.007 | +0.234 |
| Flat, seed 1337 | 0.553 ± 0.018 | 0.793 ± 0.021 | +0.240 |

Nothing about any model changes between the two columns. Only the unit the score is computed over changes, and it moves QWK by 0.23–0.28 in every cell — larger than any difference between the architectures, the seeds, or the two combined. **This is a property of the labels, not of the network.**

The mechanism is anatomical rather than statistical. The ROP ridge grows in the retinal periphery, so whether a frame contains the finding depends on where the camera was pointed. Frames of confirmed-diseased eyes captured at the posterior pole are called normal **71.7%** of the time; temporal frames of the same eyes, which look at where the ridge sits, are called normal **12.7%** of the time [Fig. 3]. The pooled confusion matrix says the same thing from the other side: 73% of stage-1 frames, 44% of stage-2 and 37% of stage-3 fall into the Normal column, while the Normal row is 99% correct. Errors are not scattered around the diagonal the way genuine ordinal confusion would scatter them; they are concentrated in one column, which is the signature of frames that inherited a label their pixels do not carry.

The same caution cuts the other way on AP-ROP, where the unit effect is real but is mostly *not* what it appears to be. Pooled out-of-fold, recall rises from 0.345 per image to 0.840 per session (100/119 units, CI 0.76–0.90; 0.832 at seed 1337). Decomposed, +0.348 of that is population re-weighting — one infant supplies 76% of AP-ROP images — and only +0.147 is frame aggregation [`ablation_session_level.json`, pooled 5-fold]. Roughly 70% of a jump routinely attributed to aggregation belongs to corpus composition instead. A per-fold analysis reverses that split, which is precisely why we report the pooled one: it covers every AP-ROP unit exactly once.

The mechanism is directly observable. Among AP-ROP-labelled frames from the source that labels per *session*, 70.4% score below 0.1 (470 frames, 16 sessions); among sources that label 1–2 frames per infant, only 10.1% do (148 frames). Same model, same class, seven times the "label carried, appearance absent" rate exactly where the annotation is per-session.

### 3.2 The architectures are indistinguishable

Under 5-fold CV the structured head scores macro-F1 0.692 ± 0.086 against the flat baseline's 0.698 ± 0.099 at seed 42, and 0.708 ± 0.097 against 0.686 ± 0.109 at seed 1337. The difference is **−0.0055 at one seed and +0.0224 at the other** — opposite signs, an order of magnitude below the per-fold spread of ±0.09. On the locked set the flat head is in fact *ahead* on image-level QWK and behind on eye-level, by margins inside one standard deviation.

We report this because our own earlier draft did not. A single fold showed the structured head failing on AP-ROP, reproducibly, across five backbones; it dissolved under cross-validation, because that fold's validation split held the one heavily-imaged session-labelled infant. Five architectures "failed" identically because they shared an evaluation unit, not a defect. What survives for the taxonomy-faithful head is not accuracy but error structure, and no aggregate metric captures it.

### 3.3 A naive site probe understates residual site information

Domain-adversarial training is normally validated with a linear probe over all classes. Because disease prevalence varies by site, that probe can score well by reading disease. We ran both, on normal-class images only for the controlled version, with patient-clustered bootstrap CIs (2,000 replicates, paired across arms) [`dann_probe_ci.json`]:

| Adversary weight | Naive probe | Disease-controlled probe |
|---|---|---|
| w = 0.0 | 0.790 [0.725–0.882] | 0.866 [0.807–0.917] |
| w = 0.5 | 0.809 [0.751–0.889] | 0.852 [0.788–0.909] |
| w = 2.0 | 0.867 [0.821–0.933] | 0.917 [0.869–0.958] |

Two readings. The controlled probe sits **above** the naive one at every strength, so a paper reporting only the naive probe understates the site information its features retain. And neither probe falls as the adversary strengthens — site appearance is not removed at any setting we tested. We report the branch as a negative result.

### 3.4 External validation retires a model that looked excellent at home

A binary ResNet50 screening head reached sensitivity 0.990 and AUC 0.970 on its own hospitals. Scored at the held-out hospital with its deployed threshold unchanged, the lowest score it assigned any image (0.279) exceeded its own decision threshold (0.193): it flagged **663 of 663 images, including all 150 healthy eyes**, for specificity 0.000 and AUC 0.691. This is the documented failure mode of cross-camera transfer — Chen et al. [C2] report AUROC falling 0.99 → 0.62 in the same direction — and it is invisible to any internal metric.

We re-based screening onto the staging model's any-ROP probability using a rule fixed before the curves were drawn: the largest threshold holding sensitivity ≥ 0.90 at both sites. It yields sensitivity 0.958 / specificity 0.781 internally and 0.905 / 0.387 externally, against the retired head's 1.000 / 0.000 [`screening_rebase.json`]. The external half used for selection makes those two numbers in-sample and optimistic; we say so wherever they appear.

### 3.5 Attention is measurable, and the measurement needs its own audit

We measured Grad-CAM attention falling outside the retinal aperture over all 663 development images. Our first cross-model comparison was invalid: the structured model letterboxes, so a quarter of its frame is padding introduced by preprocessing, and swapping only the scalar being differentiated moved total off-retina attention by 26 percentage points on identical weights. Measured under matched objectives, with padding separated from the camera's own surround, the served model's attention peak lands outside the retina in 11–12% of images against 2–3% for the retired head — a real difference, roughly a quarter the size of the one the naive measurement reported.

---

## 4. Discussion

The unit an ROP model is scored over changes its headline agreement by more than architecture, initialisation, or both together. That is the paper's central claim, and the locked-set design is what licenses it: four cells, two architectures, two seeds, one opening, same direction, similar magnitude.

The practical consequence is that image-level ICROP metrics are not comparable across studies unless the label's native unit is stated. A system reporting 0.98 accuracy under an image-level split on session-labelled data [C1] and a system reporting 0.52 QWK per image on the same source are not necessarily describing different models; they may be describing different denominators. Our own numbers are unremarkable next to the published state of the art, and we think that is the honest position rather than a shortfall: against a human inter-expert Fleiss κ of 0.24 on this task [C9], reported model agreements above 0.95 warrant more scrutiny than celebration.

Two of our three headline results are negative. The adversarial branch does not remove site information; the taxonomy-faithful head does not beat a flat softmax. We report them at full weight because the alternative — reporting the seed that agrees with us — is available, cheap, and was in an earlier version of this manuscript.

**Limitations.** First, we did not re-grade the labels; the 43% non-ROP contamination in the positive class is documented from the source records rather than adjudicated, and expert re-grading is in progress. Second, stage 4/5 (n = 89) and AP-ROP are not externally validated — too few cases reach the held-out hospital — and we mark them so rather than quoting them, a decision taken before the locked set was opened. Third, the re-based screening threshold was selected using the development half of the held-out hospital, so its external figures are in-sample for that choice; confirmation requires a third site. Fourth, our corpus carries no zone or plus-disease annotation, so we cannot address the criteria on which treatment decisions are actually made. Fifth, the AP-ROP session analysis rests on genuine multi-frame structure from a single infant and is reported as a methodological caution, not a benchmark.

## 5. Conclusion

On a 6-class ICROP staging benchmark with patient-grouped folds and an external hospital scored once, the evaluation unit moves quadratic-weighted kappa from ~0.53 to ~0.80 across four independent model–seed cells, and the errors it explains are anatomically directed rather than random. Architecture choice, over the same corpus, moves nothing that survives a change of random seed. We suggest that ICROP staging results should state the unit their labels describe, report both image- and eye-level metrics, and validate externally before a threshold is deployed — and we provide an audited corpus, a metadata-only floor, and a pre-registered locked evaluation to make that comparison possible.

---

## 6. Compliance with Ethical Standards

This research used publicly available de-identified retinal image datasets and one de-identified institutional corpus released for research use; no new patient data were collected and no identifiable information was accessed. Ethical approval was not required, as confirmed by the licence terms of each constituent dataset. The authors declare no conflicts of interest.

---

## 7. References *(to be formatted in IEEE style at conversion)*

- **[C1]** M. Vahidmoghadam, P. Ghorbani, M. J. Ahmadi *et al.*, "Automated diagnosis of plus form and early stages of ROP using deep learning models," *Sci. Rep.*, 2026. — same Ostrava corpus; image-level split; no external validation.
- **[C2]** J. S. Chen, A. S. Coyner, S. Ostmo *et al.*, "Deep learning for the diagnosis of stage in retinopathy of prematurity: accuracy and generalizability across populations and cameras," *Ophthalmol. Retina*, vol. 5, no. 10, pp. 1027–1035, 2021.
- **[C3]** International Committee for the Classification of Retinopathy of Prematurity, "International Classification of Retinopathy of Prematurity, Third Edition (ICROP3)," *Ophthalmology*, vol. 128, no. 10, pp. e51–e68, 2021.
- **[C4]** X. Shi, W. Cao, S. Raschka, "Deep neural networks for rank-consistent ordinal regression based on conditional probabilities (CORN)," *Pattern Anal. Appl.*, 2023.
- **[C5]** Y. Ganin, E. Ustinova, H. Ajakan *et al.*, "Domain-adversarial training of neural networks," *J. Mach. Learn. Res.*, vol. 17, no. 59, pp. 1–35, 2016.
- **[C6]** A. J. DeGrave, J. D. Janizek, S.-I. Lee, "AI for radiographic COVID-19 detection selects shortcuts over signal," *Nat. Mach. Intell.*, vol. 3, pp. 610–619, 2021.
- **[C7]** L. Oakden-Rayner, J. Dunnmon, G. Carneiro, C. Ré, "Hidden stratification causes clinically meaningful failures in machine learning for medical imaging," in *Proc. ACM CHIL*, 2020, pp. 151–159.
- **[C8]** Y. Wang *et al.*, "Automated explainable multidimensional deep learning platform of retinal images for retinopathy of prematurity screening," *JAMA Netw. Open*, vol. 4, no. 5, e218758, 2021. — image/eye/patient F1 0.956/0.915/0.898.
- **[C9]** G. Gschliesser *et al.*, "Inter-expert and intra-expert agreement on the diagnosis and treatment of retinopathy of prematurity," *Am. J. Ophthalmol.*, vol. 160, no. 3, pp. 553–560, 2015.
- **[C10]** J. P. Campbell *et al.*, "Expert diagnosis of plus disease in retinopathy of prematurity from computer-based image analysis," *Ophthalmology*, vol. 123, no. 8, pp. 1795–1801, 2016.
- **[C11]** R. R. Selvaraju *et al.*, "Grad-CAM: visual explanations from deep networks via gradient-based localization," in *Proc. IEEE ICCV*, 2017, pp. 618–626.
- **[C12]** Timkovič J. *et al.*, "Retinal image dataset of infants and retinopathy of prematurity," *Sci. Data*, vol. 11, 814, 2024. — the Ostrava corpus.

---
---

# AUTHOR NOTES — strip before submission

## A. Figure plan for the 4-page limit

All technical content must fit in 4 pages. That buys roughly **three figures and three tables**. Recommended set, in priority order:

| # | Figure | Source file | Why it earns the space |
|---|---|---|---|
| **1** | Inclusion cascade | `graphs/paper/01_inclusion_flow.png` | Reviewers check this first. It shows the 43% non-ROP exclusion and the locked/dev split in one glance, and it is the visual proof that the audit exists. **Non-negotiable.** |
| **2** | Evaluation unit | `graphs/paper/10_evaluation_unit.png` | The paper's central claim, in one picture: identical predictions, two units, four cells. **Non-negotiable.** |
| **3** | Camera-view misses | `graphs/paper/11_camera_view_misses.png` | Turns the claim from a statistical curiosity into an anatomical mechanism (71.7% vs 12.7%). This is what makes a reviewer believe §3.1 rather than suspect an aggregation trick. **Strongly recommended.** |
| 4 | Confusion matrix, locked | `graphs/paper/12_confusion_locked.png` | Same mechanism from the other side (errors in the Normal column, not around the diagonal). **Use only if space allows — otherwise state the three percentages in text**, which §3.1 already does. |
| 5 | Head equivalence | `graphs/paper/14_head_equivalence.png` | Sign-flip between seeds. Compresses well into a sentence + the CV numbers already in §3.2. **Cut first.** |
| 6 | ROC transfer | `graphs/paper/05_roc_transfer.png` | 0.970 → 0.691. Compresses to text losslessly. **Cut.** |
| 7 | Site probe | `graphs/paper/15_site_probe.png` | Already fully carried by the §3.3 table. **Cut.** |

Figures 17/18 (Grad-CAM audit) do **not** go in the paper. §3.5 is three sentences, and the attention material is a paper of its own — it dilutes the unit finding if given space here.

**Composite option if space is tight:** merge Fig. 2 and Fig. 3 into a single two-panel figure ("the unit effect and its anatomical cause"). This is the single best space saving available and loses nothing.

## B. Length check

Body text is ~2,450 words against a two-column IEEE budget of roughly 2,200–2,600 with three figures and three tables. Expect to be **slightly over on first conversion**. Cut in this order:

1. §3.5 (attention) → reduce to two sentences, or drop entirely and keep it for the journal version.
2. §2.4 → compress the five gates to a single sentence with the two numbers that matter (0.997 decodability, 0.187 floor).
3. §3.1 AP-ROP decomposition paragraph → one sentence.
4. §1 paragraph 4 (the [C1] control-condition paragraph) → fold into §4.

Do **not** cut: the four-cell table, the 71.7/12.7 contrast, the seed sign-flip, or the limitations list. Those are the paper.

## C. What still blocks submission

1. **Clinical co-author.** The author list has a placeholder. ISBI is single-blind, so a reviewer will see the affiliation list; an ROP staging paper with no ophthalmologist on it invites a desk-level credibility problem, independent of the science.
2. **Label adjudication.** §2.2 promises "in progress… will accompany the camera-ready version." That promise must either be kept or removed before submission.
3. **IEEE template conversion + PDF eXpress check.** Papers that do not comply with the official template are returned before review.
4. **Pin the false-alarm denominator.** `session_level_by_seed.json` warns that normal-session FP is 1.16% using all normal units and 0.25% using Ostrava-keyed sessions only. Both are honest; they are different denominators. State which one the paper uses, once, in §2.6.
5. **Decide the fifth page.** $200 buys references + ethics + acknowledgements only. With 12 references, the 4 pages will be tight; budget for it.

## D. Positioning summary — why this gets accepted

The reviewer's question is "what is new?" The answer is not the model. It is:

- A finding that is **model-independent and pre-registered**: four cells, one opening of a locked set, same direction and magnitude.
- A **mechanism**, not just a correlation: peripheral disease, posterior frames, 71.7% vs 12.7%.
- **Negative results reported at full weight**, including the withdrawal of our own earlier claim — which is rare enough to be a differentiator in itself.
- A **direct, citable contrast** with a 2026 paper on the same corpus that reports 0.98–1.00 accuracy under an image-level split and admits the leakage risk in its own limitations.

The risk to manage: a reviewer reading §3.1 as "you just aggregated predictions and the number went up, which is trivially expected." §3.3's anatomical mechanism and the four-cell consistency are the defence, which is why Fig. 3 is close to non-negotiable.
