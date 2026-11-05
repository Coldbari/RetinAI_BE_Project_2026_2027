# Related work: what the closest ROP deep-learning papers did, and how

Consolidates the literature that was previously scattered across `isbi2027_draft.md` §1,
`isbi2027_paper.md` §7, `human_agreement_ceiling.md`, and `docs/ROP_PROJECT_STATE.md` §8
into one place. Method-level detail below was re-verified against the published sources in
September 2026; where a number in our older notes disagrees with the source, both are shown
and the disagreement is flagged rather than silently resolved.

Purpose: this is the evidence base for the Introduction and Discussion. Every claim we make
of the form "the field does X" needs a row in this file.

---

## 1. The roster at a glance

| # | Paper | Venue / year | Data | Split unit | Eval unit | Headline | Overlaps our corpus? |
|---|---|---|---|---|---|---|---|
| P1 | **Vahidmoghadam et al.** — Automated diagnosis of plus form and early stages of ROP | *Sci Rep* 16:7234, 2026 | **Ostrava/RIDIRP only** (6,004 img, 188 infants) | **image-level** | image | Plus acc 0.996; stage acc 0.98 | **Yes — same corpus. Primary competitor.** |
| P2 | **VEA-Net** (Wang et al.) — vascular enhancement attention, dual-backbone multi-task | *Front Cell Dev Biol* 2026 | FARFUM-ROP + **Ostrava** + A-Fundus (8,636 img, 737 subjects) | **subject-level GroupKFold (10)** | image | Stage acc 74.81%; Plus acc 85.44% | **Yes — same corpus, honest split.** |
| P3 | **ROPDeepX** (Kumar et al.) — ResNet50 + EfficientNet-B4 hybrid | *Discov Artif Intell* 2026 | FARFUM-RoP + **RIDIRP** + HVDROPDB | **image-level on RIDIRP** | image | 92.60% combined; ~99.7% RIDIRP* | **Yes — leakage exemplar.** |
| P4 | **Peng et al.** — deep feature fusion + ordinal classification | *IEEE TMI* 40(7):1750–62, 2021 | 635 img / 196 exams (per-image); 1,173 exams (per-exam) | not stated | **image AND examination** | κ 0.9786, ACC1 0.9827 | No — single centre, private |
| P5 | **Kiran Yenice, Kara & Erdaş** — type 1 / type 2 / A-ROP detection | *Eye* 2024 | 634 img, 317 infants | stratified 10-fold (**image-level**) | image | high acc/spec/AUC, small n | No — single centre |
| P6 | **i-ROP DL / Campbell, Brown, Redd et al.** — vascular severity score | *JAMA Ophthalmol* 2018; *Ophthalmology* 2020–22 | multi-centre US + Nepal/Mongolia | patient | **examination-level** | exam-level plus AUC 0.968 | No — but **the unit precedent** |

\* The 99.7% RIDIRP figure is from our earlier full-text read and could not be re-verified in
September 2026 (Springer paywall). **Verify against the PDF before it goes in the paper** —
it is doing real rhetorical work as our leakage exemplar and must not be wrong.

Supporting, non-rival citations already in the reference list: ICROP3 (2021), CORN ordinal
regression (Shi 2023), DANN (Ganin 2016), Grad-CAM (Selvaraju 2017), DeGrave shortcut
learning (2021), Oakden-Rayner hidden stratification (2020), Gschließer inter-expert
agreement (2015), Campbell diagnostic discrepancies (2016), Timkovič RIDIRP data descriptor
(2024).

---

## 2. Paper by paper — what they did and how

### P1 · Vahidmoghadam et al., *Sci Rep* 2026 — the direct competitor

**Why it matters most:** same 6,004 images, same 188 infants, same three RetCam/Phoenix
devices. Any reviewer who knows the corpus will ask why our numbers are lower than theirs.
The answer is this paper's methodology, so we need it exactly right.

**What they did:**
- Two tasks. Plus: binary, all 6,004 images (5,375 Normal / 629 Plus). Stage: **4 classes
  only — Stage 0,1,2,3** on a subset of **880 images** (45 / 252 / 458 / 125).
- A vessel-segmentation front end (U-Net++ won: acc 0.9766, Dice 0.8584), trained on FIVES +
  HRF + HVDROPDR (681 images), feeding a classifier.
- Preprocessing: CLAHE, Gaussian blur, contrast enhancement, noise reduction.
- Backbone bake-off: ResNext50, ResNet152, RegNetX3, VGG16, GoogleNet, MobileNetV2,
  **EfficientNetB4** (selected).
- Split: **80% train / 16% val / 4% test, image-level.** No patient grouping stated. For
  stage that means a **test set of 37 images** and a validation set of 140 (~35 per class).

**What they report:** Plus val acc 0.996, F1 0.98, sens 0.99, spec 0.997, ROC-AUC **1.000**.
Stage val acc 0.98, F1 0.97. Test: Plus accuracy **1.000 across the board**; stage acc
0.9730.

**What they did not do:** no external validation (single centre, stated), no patient-level
split, no device/shortcut audit, no confidence intervals, nothing on Stage 4/5, nothing on
AP-ROP. Their own limitations list covers single-centre generalisability, sample size
(they compute statistical power 0.85 against a 0.9 target), Stage 0 underrepresentation, and
the Stage-3 ceiling — but **never names data leakage**, despite Ostrava photographing each
infant ~29 times per sitting.

**How we use it:** the control condition. Same images, same labels; the difference in
reported performance is the split and the unit, not the model. Cite in §1 and §4.

---

### P2 · VEA-Net, *Front Cell Dev Biol* 2026 — the honest-split rival

**Why it matters:** this is the paper that did the splitting right on our corpus, so we
cannot claim patient-grouping as our novelty. Our differentiator against it is the taxonomy
and the instruments, not the hygiene.

**What they did:**
- Three centres pooled: FARFUM-ROP (1,533 img / 68 infants, Iran, RetCam III), **Ostrava**
  (6,004 / 188, Czechia), A-Fundus (1,079 / 481, Shenzhen). Total 8,636 / 737.
- **Subject-level GroupKFold, 10-fold** — all images from one infant stay in one fold.
- Dual backbone: ResNet50 (local vessel detail) + EfficientNet-B4 (multi-scale context),
  fused by a soft attention mechanism that weights the two streams dynamically.
- **VEA module:** multi-scale conv (1×1, 3×3, 5×5) + directional selective depthwise conv +
  SE channel attention + CBAM spatial attention, with residual connections, inserted into
  the feature pathway.
- Three heads (Plus / stage / treatment), each a 2-layer MLP.
- **Dual domain adaptation: DANN with gradient reversal (3-class domain classifier) plus MMD
  with multi-scale Gaussian RBF kernels (K=5 bandwidths).**
- Loss: focal (plus, treatment) + weighted CE (stage) + plus↔stage consistency +
  stage↔treatment consistency + task auxiliary + domain adversarial + MMD.

**What they report:** Plus acc 85.44% ± 15.78 / AUROC 91.76%; **stage acc 74.81% ± 10.17 /
AUROC 88.26%**; treatment acc 56.80% ± 23.48. Per centre: Ostrava Plus 97.17% ± 3.74,
FARFUM Plus 59.83% ± 23.25, A-Fundus stage 66.15% ± 7.17.

> ⚠️ **Discrepancy to resolve.** `ROP_PROJECT_STATE.md` §8 records "their Ostrava stage acc
> 90.4% vs 89.5% majority-class prior". The figures retrieved in Sept 2026 are pooled stage
> 74.81% and A-Fundus stage 66.15%; no 90.4% appears in the summary we could pull. The 90.4%
> may be a per-centre Ostrava stage cell in a table we read in full earlier. **Do not cite
> 90.4% until it is located in the PDF.** The prior-comparison argument is strong and worth
> keeping — but it needs the real number.

**What they did not do:**
- **AP-ROP folded into Stage 4, "status post ROP" into Stage 5.** This contradicts ICROP3,
  which puts the aggressive form explicitly *outside* the stage ladder. This is our single
  cleanest taxonomy differentiator.
- Accuracy and AUROC only — no κ, no per-class F1, no CIs beyond fold SD, and no comparison
  against a majority-class prior (which, on these class distributions, is the whole game).
- DANN is used but **never quantified** — no probe, no dose–response, no disease-controlled
  control. We measure exactly what they assert.
- **No external validation and no held-out centre.** They state it as a limitation; their
  domain adaptation is validated only within the pooled folds.

---

### P3 · ROPDeepX, *Discov Artif Intell* 2026 — the live leakage example

**What they did:** hybrid ResNet50 + EfficientNet-B4. Three datasets: FARFUM-RoP (primary
training), **RIDIRP** and HVDROPDB (cross-dataset robustness). 10-fold CV run on FARFUM-RoP
alone, on LAN-RoP alone, and on the combined set. Combined multi-centre cohort: accuracy
92.60%, recall 92.58%, precision 92.69%, F1 92.60%.

**The problem:** the split is patient-safe on FARFUM only. On RIDIRP it is **image-level**,
at roughly 32 frames per infant — which is what produces the ~99.7% figure. That is the
canonical illustration of what patient leakage buys, and it is worth more to our Introduction
than any architectural criticism.

**Note the circularity worth mentioning in §4:** VEA-Net (P2) explicitly justifies its
ResNet50 + EfficientNet-B4 dual backbone by citing ROPDeepX's "state-of-the-art performance
on publicly available ROP benchmarks" — i.e. an architecture choice propagated through the
field on the strength of a leaked number. This is a concrete, citable instance of the harm,
not a hypothetical.

---

### P4 · Peng et al., *IEEE TMI* 2021 — the ordinal prior, and a partial unit precedent

**What they did:** deep feature fusion with an ordinal classification strategy for ROP
staging. Evaluated **per-image** on 635 fundus images from 196 examinations (303 Normal,
26 Stage 1, 127 Stage 2, 106 Stage 3, 61 Stage 4, 12 Stage 5) — weighted recall 0.9055,
weighted precision 0.9092, weighted F1 0.9043, **ACC1 0.9827, κ 0.9786**. Separately they ran
**per-examination** staging over 1,173 examinations with 4-fold CV.

**Why it matters both ways:**
- It is the ordinal-ROP prior, so we never claim "first ordinal ROP model."
- It already evaluates at the examination unit. We must therefore **not** claim to have
  invented examination-level ROP evaluation. Our claim is narrower and survives: we quantify
  *how far the unit alone moves the number on identical predictions* (AP-ROP 0.345 → 0.840),
  show it is model-independent, and give the anatomical mechanism (71.7% vs 12.7%
  camera-view misses). Peng reports both units; they do not treat the gap as the finding.

**What they did not do:** patient grouping is not stated, single centre, private data, no
site/device analysis, no CIs. Full methods are behind IEEE paywall — the numbers above come
from the abstract/results. **Get the PDF before writing the §4 comparison paragraph.**

---

### P5 · Kiran Yenice, Kara & Erdaş, *Eye* 2024 — A-ROP as a softmax class

**What they did:** 634 fundus images from 317 premature infants (GA 23–34 weeks), RegNetY002,
**stratified 10-fold CV at image level**. Classes: type 1 ROP / type 2 ROP / A-ROP, i.e.
A-ROP treated as one more mutually-exclusive softmax class. Reports high accuracy,
specificity and AUC; authors flag retrospective design and small sample.

**How we use it:** second example of AP-ROP forced into a flat categorical scheme, and an
example of image-level stratified CV on a corpus with ~2 images per infant. Supports the
taxonomy argument; weak as a performance rival given n.

---

### P6 · i-ROP DL (Campbell / Brown / Redd / Coyner et al.) — the unit precedent we must cite

**What they did:** the vascular severity score (VSS, 1–9) line of work — a per-image score
aggregated to the examination. Reported **examination-level plus disease AUC 0.968**; analysed
6,344 clinical examinations showing higher VSS associated with more posterior disease, higher
stage, and greater extent of stage 3. Later work (Nepal, Mongolia) is genuine prospective
external validation across populations and cameras; Chen et al. *Ophthalmol Retina* 2021
[C2] covers generalisability across populations and cameras specifically for stage.

**Why this is the most dangerous citation for us and the most necessary:** a reviewer who
knows this group will say examination-level ROP evaluation is a decade old. It is. We cite it
first, concede the precedent explicitly, and state the narrower claim (see P4 above). Not
citing it is the failure mode.

---

## 3. What the field does not do — our four open lanes

Derived from the six rows above; these are the Introduction's contribution claims, each with
the specific papers they are claimed against.

1. **Nobody quantifies the evaluation unit as a variable.** P6 and P4 *use* the examination
   unit; none of the six measures the gap between units on identical predictions, or shows it
   is model-independent, or gives the anatomical mechanism. Claimed against P1, P2, P3, P5.
2. **Nobody validates their domain-adaptation claim.** P2 runs DANN + MMD and reports no
   probe at all. We show the naive probe is confounded by disease prevalence and that a
   disease-controlled probe credits far less. Claimed against P2.
3. **Nobody audits shortcuts on these corpora.** No paper in the roster reports a
   metadata-only performance floor, a device/dimension probe, or per-hospital calibration.
   Ours: image dimensions alone score AUC 0.911 on the ROP test split, and the site signature
   is in colour rather than vignette. Claimed against all six.
4. **Nobody preserves the ICROP3 taxonomy for the aggressive form.** P2 folds AP-ROP into
   Stage 4; P5 makes it a softmax sibling. Claimed against P2 and P5.

**Claim wording that survives review** (unchanged from `ROP_PROJECT_STATE.md` §8 — this
wording was chosen to be defensible against exactly the papers above):

> never "first ordinal ROP" / "first AP-ROP task" / "first examination-level ROP" — always
> *"first joint ordinal + AP-ROP-branch model with probe-quantified site debiasing and
> held-out-hospital evaluation."*

---

## 4. The human ceiling, and why it is a separate argument

Full treatment in [`human_agreement_ceiling.md`](human_agreement_ceiling.md). In brief:
Gschließer 2015 (7 experts, 260 wide-field photographs, read twice 8 weeks apart) gives
**inter-expert κ 0.24 for ROP stage**, 0.35 for AP-ROP, and **intra-expert κ 0.56** — the same
expert disagreeing with themselves. Campbell 2016 (281 infants, 7 centres, 1,553 comparisons)
gives **40% stage disagreement** but **>95% agreement on type 1 / treatment-requiring disease**.

Two consequences for how we position against the roster:
- None of P1–P5 reports agreement against this benchmark, and several report accuracies
  (0.98–1.00) that exceed what two experts achieve with each other on the same task. That
  gap is itself evidence about their protocol, and it is a fairer criticism than attacking
  their architectures.
- It tells us which of our own numbers to lead with: humans agree poorly on stage and almost
  perfectly on treatment-requiring disease, so the clinically-decisive axis is the one with a
  trustworthy reference standard — which is the AP-ROP/session axis.

---

## 5. Open items before the related-work section is submission-ready

- [ ] **Verify ROPDeepX's RIDIRP 99.7%** against the PDF (paywalled; try institutional access
      or contact authors). It is our leakage exemplar and must be exact.
- [ ] **Locate VEA-Net's Ostrava stage accuracy** and its majority-class prior; resolve the
      90.4% / 74.81% discrepancy in §2 above before citing either.
- [ ] **Get the Peng TMI 2021 PDF** — specifically whether patient grouping was used, and how
      per-examination staging was aggregated from per-image predictions. This determines how
      much of our unit claim they already occupy.
- [ ] **Add P6 (i-ROP DL) to the formal reference list.** It is discussed in our positioning
      but [C2] Chen 2021 is currently the only entry from that group. Add the examination-level
      plus AUC 0.968 source explicitly.
- [ ] Decide whether P5 (*Eye* 2024) earns a citation slot in a 4-page ISBI paper, or whether
      the taxonomy point rests on P2 alone. With 12+ references and a 4-page limit, this is a
      real budget question — see `isbi2027_paper.md` author note C.5.

## Sources

P1 [Sci Rep 2026](https://www.nature.com/articles/s41598-026-37064-2) ·
[PMC12923887](https://pmc.ncbi.nlm.nih.gov/articles/PMC12923887/) ·
P2 [Front Cell Dev Biol 2026](https://www.frontiersin.org/journals/cell-and-developmental-biology/articles/10.3389/fcell.2026.1865503/full) ·
[PMC13447418](https://pmc.ncbi.nlm.nih.gov/articles/PMC13447418/) ·
P3 [Discov Artif Intell 2026](https://link.springer.com/article/10.1007/s44163-026-00834-y) ·
P4 [IEEE TMI 2021](https://ieeexplore.ieee.org/document/9376907/) ·
P5 [Eye 2024](https://www.nature.com/articles/s41433-024-03184-0) ·
P6 [Ophthalmol Sci 2022 (Nepal/Mongolia)](https://www.ophthalmologyscience.org/article/S2666-9145(22)00054-9/fulltext) ·
[Ophthalmology 2021 VSS scale](https://www.sciencedirect.com/science/article/abs/pii/S0161642020310277)

---

## 6. Prior-art check for plan v3 (2026-10-19) — the works a reviewer will say we missed

Verified by a web sweep on 2026-10-19 (FT = full text read; AB = abstract/card only).

| Work | Venue/yr | What it did | Overlap | Ver. | URL |
|---|---|---|---|---|---|
| **ROP-Bench** (Luo, Guo, Chen, Liang, Lin) | Zenodo+GitHub Jun 2026 | Ostrava + Shenzhen + FARFUM + HVDROPDB (8,821 img), patient-level splits, DINOv2 ViT-B/14 vs RETFound ViT-L/16, frozen and full FT; DINOv2 wins everywhere; binary presence / plus only, image level | Same cohorts, same retina-vs-natural question, NOT architecture-matched, not staging | FT (repo notes/results.md) | github.com/foye80/rop-bench · zenodo.org/records/20721027 |
| RETFound-Green paper (Engelmann) | Nat Commun 2025 | Its ROP benchmark is the Shenzhen Sci-Data set, patient-level 80/20, linear probe only; no fine-tune recipe | Same cohort + model, frozen | FT | pmc.ncbi.nlm.nih.gov/articles/PMC12297332 |
| **SABPI-Net** (Chen … Sun; Shenzhen/Macao) | MICCAI 2025 | 174,540 CFP / 13,314 pts, 5 classes incl. A-ROP as its own class, patient-level; RETFound ViT-L baseline (92.3 vs 95.3 acc); image level | RETFound on ROP; A-ROP class | FT | papers.miccai.org/miccai-2025/0798-Paper2355.html |
| ROP-VL descriptor | Sci Data Aug 2026 | 8 classes incl. A-ROP, laser, stage 1–5; VLM baseline best | our data; A-ROP separate, image level | AB | doi.org/10.1038/s41597-026-07901-0 |
| **DeepAdapter** (Zhao, Li) | BJO Aug 2026 | MoCo-v2 SSL on ~500k infant CFP (ResNet50) + UDA; RETFound ViT-L fine-tuned baseline; held-out hospital; binary | SSL on infant images + held-out hospital | FT | doi.org/10.1136/bjo-2025-328559 |
| ProtoEvi-ROP | MICCAI 2026 | Shenzhen 4-class, patient-level 80/20 + 5-fold, ResNet-50, image level | protocol on Shenzhen | FT | papers.miccai.org/miccai-2026/0891-Paper2826.html |
| Kian Ara … Timkovič | Front Med Sep 2026 | Ostrava Phoenix ICON, patient-disjoint plus; shows exam-derived label ≠ isolated image | our label-unit motivation, on our corpus | FT | fmed.2026.1960317 |
| Derks … Campbell | J AAPOS 2025 | i-ROP DL external validation, 419 eye-exams; eye = mean VSS, infant = max of eyes; binary | eye-level rule + held-out hospital | FT | pmc.ncbi.nlm.nih.gov/articles/PMC13345673 |

Not found: VisionFM/RetiZero/UrFound/DINOv3 on ROP; any RETFound-Green fine-tune recipe (all citing works use it frozen);
any multi-class ICROP staging evaluated at eye level with a pre-registered threshold on an external hospital; any use
of Multi-View ROP beyond its card; LSE/bag-loss training for stage from session labels.

**Novelty statement that survives:** "First architecture-matched test (ViT-S/14-reg, 392 px) of retina-specific vs
natural-image self-supervised pretraining for multi-class ICROP-3 staging on three public cohorts with pre-registered
patient-grouped CV, and the first eye-level, worst-view ICROP staging benchmark with a pre-registered operating point and
calibration on a held-out hospital." Never claim: first FM on ROP, first patient-level split, first A-ROP class, first
eye-level ROP. **Position against ROP-Bench explicitly** — it concludes retinal FMs do not transfer (DINOv2 > RETFound),
on binary tasks with mismatched architectures; our matched design tests exactly the claim it leaves open.

---

## 7. Transformer / ViT sweep for ROP staging (2026-10-29)

Searched OpenAlex (8 queries, 2021 onward), PubMed (transformer / ViT / Swin / DINOv2 / RETFound /
foundation / self-supervised / MIL AND "retinopathy of prematurity"), arXiv and the open web. Litmaps itself needs a
signed-in account; `rop_transformer_sweep_2026-10-29.bib` (23 entries, from doi.org) is ready to import into a
Litmaps map. Ver.: AB = abstract only, FT = full text read. Nothing below was read past its abstract unless marked.

**7a. Close to us: cite and position against these**

| Work | Venue/yr | What it did | Why it matters to us | Ver. | DOI |
|---|---|---|---|---|---|
| **Mahboob, Ahsan, Wilms** | SPIE Med Imaging 2025 | Fine-tunes RETFound (adult retinal FM) vs ImageNet classifiers for ROP **staging** on a large public ROP dataset; RETFound does **not** beat the generic models | The same retinal-vs-natural question as arms A vs B, same null answer. **Missing from paper 2's references; add it.** | AB | 10.1117/12.3047303 |
| **Shri Vatssan, Devishree, Sathishkumar** | IConSCEPT 2025 | Swin-Tiny, **5 classes (Normal–Stage 4)**, 224 px, weighted CE + label smoothing, held-out test acc 0.950, macro-F1 0.916 | Nearest task + loss recipe; dataset, split level and external test not stated in the abstract | AB | 10.1109/iconscept66142.2025.11436196 |
| **ROPGCViT** (Yurdakul, Uyar, Taşdemir …) | IEEE Access 2025 | 25 CNNs + 25 ViTs compared, Global-Context ViT + SE + residual MLP proposed; 1,099 images; acc 94.7%, κ 0.931 | Large backbone comparison like ours, but one small dataset; classes and split unverified | AB | 10.1109/access.2025.3564213 |
| **Lei, Zhao, Xie …** | IEEE BIBM 2023 | ViT branch (cascaded group attention) + CNN branch with feature interaction, predicting ROP **stages**; internal data | Transformer staging, Shenzhen group | AB | 10.1109/bibm58861.2023.10385575 |
| **Zhao, Lei, Xie …** | IEEE ISBI 2023 | ResNet-50 + MaxViT dual branch, Swin spatial-pyramid pooling, CBAM; ROP staging on an internal dataset | Transformer staging at our target venue | AB | 10.1109/isbi53787.2023.10230827 |
| **Hassan & Hasan, CAA Ensemble** | Comput Biol Med 2026 (arXiv 2602.05208) | **Ostrava** (188 infants, 6,004 img), **patient-wise** split (17-patient test + 5-fold CV), 4-class severity (normal / mild / severe / other), macro-F1 0.92; EfficientNet-B0 streams + gated MIL on vessel maps; single centre, no external test | Same cohort as our CV pool; not ICROP stages, no transformer, no external hospital | FT (arXiv HTML) | 10.1016/j.compbiomed.2026.111945 |
| **Lan, Tsai, Wu** | TVST 2026 | **Eye-level MIL** over 5 fields of view per eye (284 eyes, CGMH + Osaka), type-1 ROP; feature-level fusion beats outcome-level (late) fusion; temporal + nasal + central views best | Directly bears on our worst-view (outcome-level) rule. Our Oct spike found attention-MIL *lost* to max at the new hospital: discuss the difference (5 fixed views and one task vs our 3-view external eyes) | AB | 10.1167/tvst.15.8.13 |
| **DS-Swin-T** (Chen, Zhao, Liu …) | IEEE JBHI 2025 | Dual-scale Swin + style synthesis + adversarial alignment to resist cross-dataset style shift; 97.9% source, up to 93.7% on target styles | Transformer + domain shift (compare with arm F and the site head); classes unverified | AB | 10.1109/jbhi.2025.3573318 |
| **MS-ROPNet** (Luo, Chen, Ying …) | Ophthalmol Sci 2026 | VGG-Swin image features + demographics random forest; e-ROP, 1,257 infants, 12 centres, 7,741 **eye visits**; referral-warranted ROP (binary) | Transformer, eye-visit unit, multicentre; binary rather than staging | AB | 10.1016/j.xops.2026.101242 |

**7b. ViT / Swin / hybrid ROP papers with high single-dataset accuracy (cite as a group, carefully)**

| Work | Venue/yr | Claim (abstract) | DOI |
|---|---|---|---|
| Murugan R et al. | ICNSoC 2025 | ViT fine-tuned for ROP detection and stage identification, >2,000 images, acc 97% | 10.1109/icnsoc66817.2025.00080 |
| Kariakin, Stepanova, Kashkin … | USBEREIT 2025 | ViT vs ResNet vs EfficientNet, all >96% acc | 10.1109/usbereit65494.2025.11054213 |
| Malarkodi (DeepROP) | ICISC 2025 | EfficientNet-B4 + Swin, stage-wise, Kaggle "ROP Stages Reviewed", 96.5% | 10.1109/icisc65841.2025.11187674 |
| Kurian & Suresh (RetinoAid) | APCIT 2025 | CNN + ViT hybrid staging, Kaggle data | 10.1109/apcit65661.2025.11411044 |
| Cai, Li, Wang … | Int J Imaging Syst Technol 2025 | Swin + EfficientNet-B2, 96.7% acc | 10.1002/ima.70191 |
| Luo, Zhao, Chen … (VGG-ST) | IEEE BIBM 2024 | VGG19 + Swin V2, referral-warranted ROP, e-ROP | 10.1109/bibm62325.2024.10822551 |
| Raja Sankari, Umapathy, Chandrasekaran | Eng Appl Artif Intell 2025 | Quantum mobile ViT vs ViT/Swin/MobileViT; stages 95.5%, zones 96.9%, plus 96.7% | 10.1016/j.engappai.2025.111938 |
| Koçak, Bayramoğlu, Sayın … | Int J Imaging Syst Technol 2025 | ViT vs ResNet-50 for plus disease, 1,205 images, external validation on a public set | 10.1002/ima.70174 |
| Deepthi, Josephine, Jayabala Raja | Biomed Signal Process Control 2025 | Transformer unsupervised curriculum learning, plus disease | 10.1016/j.bspc.2025.107521 |
| Remya Bharathy & Anoop | IJASIS 2026 | ICROP3 hybrid (vessel GNN + EfficientNet-SE) on HVDROPDB, FundusROP and Ostrava; compared with RETFound | 10.29284/whrct502 |
| Acharla | IEEE COMPSAC 2026 | CNN + transformer ensemble for ROP screening | 10.1109/compsac69091.2026.00454 |
| Wang, Zhao, Wu … (CMIFNet) | IEEE EMBC 2025 | CNN + Mamba (state-space) fusion for ROP diagnosis | 10.1109/embc58623.2025.11252937 |
| Okuwobi, Liu, Wan (ROPNet) | Biomed Signal Process Control 2026 | Argues ViT patch fragmentation hurts stage II vs III (cites 19% lower sensitivity) | 10.1016/j.bspc.2026.110193 |
| Neima, Ghadban, Abdulhamed | J Appl Sci Eng 2025 | Lightweight ViT + clinical-text VLM, explainable | 10.6180/jase.202607_30.006 |

Most 7b papers report 95–97% image-level accuracy on one dataset, often a Kaggle re-upload; the abstracts do not
state patient-level splitting or an external hospital. Do not call any of them leaky without reading the methods.

**What the sweep changes**
- Still not found: an architecture-matched retinal-vs-natural ViT comparison for multi-class ICROP-3 staging, or any
  ViT staging model scored per eye at an unseen hospital. The §6 novelty statement survives.
- Add Mahboob 2025 to paper 2 (it reports the same null as A vs B) and Lan 2026 to the eye-level discussion (its
  feature-level MIL win is the counterpoint to our worst-view rule).
- The Ostrava cohort now has a third rival (Hassan & Hasan 2026, patient-wise, 4-class), alongside Vahidmoghadam 2026
  and Kian Ara 2026.

**7c. Why the rivals report 95–97% (2026-10-30)**

Same dataset, three numbers. On Shenzhen (Zhao et al., Sci Data 2024; 1,099 photos, 483 infants, ~2.3 photos per
infant, 5 classes incl. laser scars), the dataset's own authors report test accuracy **84.1% for ViT-B** and 87.6% for
ResNet50, with an 8:1:1 split that does not say whether it is grouped by patient (FT, PMC11130119). ROPGCViT reports
**94.7%** on 1,099 images; the count matches this dataset, but its split is unread (AB, PDF behind a 403). Our
DINOv2 (arm B, patient-grouped 5-fold CV, laser scars excluded, 740 photos) scores **0.90 accuracy**, macro-F1 0.865
on Shenzhen photos (seed 42 0.896, seed 1337 0.905). On Ostrava, with ~29 frames per sitting, the same model
scores 0.72; Vahidmoghadam's image-level split reports 97.3% on a 37-image test set (FT), and ROPDeepX's image-level
split 99.7% (FT).

The verified mechanisms: (1) image-level random splits on multi-frame cohorts (Vahidmoghadam, ROPDeepX); (2) small
test sets (37 images; 15% of 614); (3) synthetic images added before a 70/15/15 split (Remya Bharathy & Anoop 2026:
diffusion samples for stages 1 and 3; whether any reached the test set is not stated, FT); (4) easier label sets
(stage 4/5 and AP-ROP dropped or merged; Hassan & Hasan use 4 severity groups, FT); (5) no unseen-hospital test
(cross-style accuracy in DS-Swin-T drops from 97.9% to ≤93.7%, AB; VEA-Net 74.8% with honest pooled splits, FT;
ours 63% per eye, 43% per photo at Multi-View). HVDROPDB ships 785 files but only 329 unique photographs, so
splitting it by file puts copies on both sides.
