# Literature Survey

Reviewed work, grouped by the design question it answered. Each entry ends with **what we took
from it** — a survey that does not change the design is not a survey.

Full IEEE-format citations are in [../reference/paper.md](../reference/paper.md).

---

## 1. Deep learning for diabetic retinopathy

**Gulshan et al. (JAMA, 2016)** trained an Inception-v3 on 128,175 fundus photographs graded by
54 ophthalmologists and reached AUC 0.991 for referable DR, with sensitivity 0.90+ at high
specificity. It is the reference point for every DR screening result since.

Two things in it shaped this project more than the headline number:

- The endpoint is **referable DR** — a binary "does this patient need a specialist" decision —
  not 5-class grading. Screening is a referral decision, so that is what should be measured.
- Their labels came from multiple graders per image. Public datasets do not have that.

**What we took from it.** Our screening endpoint is referable DR (grade ≥ 2) reported as an AUC,
not 5-class accuracy. Our 5-class numbers are reported but explicitly framed as capped by label
noise.

**Grader agreement.** Published inter-grader agreement on EyePACS-style 5-class DR grading is
around QWK 0.83–0.85. This is a ceiling: a model cannot exceed the consistency of the labels it
learns from. Our 5-class QWK of 0.728 sits below but near that ceiling, which is the expected
place for it to sit — and is the reason the endpoint was reframed.

---

## 2. Retinopathy of prematurity

**ICROP — International Classification of Retinopathy of Prematurity, Revisited (2005)** defines
the clinical grammar of ROP: **zone** (I, II, III — how far the vascularisation has progressed
from the optic disc), **stage** (1–5 — the severity of the ridge/detachment), and **plus
disease** (venous dilation and arteriolar tortuosity, the strongest treatment indicator).

**What we took from it.** Our deployed model collapses all of this to binary ROP / No-ROP, which
is a real loss of clinical meaning and is listed as a limitation. The six-class ICROP staging
work now in progress exists because of this gap.

**Screening economics.** ROP screening requires repeated bedside examination of every pre-term
infant within a narrow window in which treatment still works. This is why the ROP operating point
was chosen to maximise sensitivity (0.990) at the cost of specificity (0.398) — a missed case is
irreversible blindness, a false alarm is one extra examination.

---

## 3. Glaucoma from fundus photographs

Classical methods estimate the **cup-to-disc ratio** by segmenting the optic disc and cup.
Learning-based methods classify the whole image or an optic-disc ROI crop.

**Benchmark datasets reviewed:** ORIGA (650 images), REFUGE (1,200), G1020 (1,020), Drishti-GS,
PAPILA, and the SMDG-19 multi-channel aggregation which pools many of these.

**What we took from it.** These datasets are heterogeneous in camera, field of view, population
and prevalence, and several contribute only one class. Pooling them and quoting one AUC hides
that. Our source-stratified evaluation — reporting per-source AUC and a source-controlled
aggregate — came directly from noticing this while assembling SMDG-19.

---

## 4. Architectures

- **ResNet (He et al., CVPR 2016)** — residual connections; still the strongest generic
  backbone on small datasets in our own sweep.
- **EfficientNetV2 (Tan & Le, ICML 2021)** — compound scaling with fused-MBConv blocks; better
  accuracy per parameter on large datasets.

**What we took from it.** We did not assume. A five-backbone sweep (EfficientNet-B0,
EfficientNetV2-S, MobileNetV3-Large, ResNet50, DenseNet121) on the ROP set found **ResNet50 well
ahead** (AUC 0.9635 vs 0.8700 for the next best) — the opposite of what the "newer is better"
assumption predicts, and consistent with the literature's observation that heavily-scaled
architectures underperform on small datasets. DR and Glaucoma, with far more data, use
EfficientNetV2-S.

---

## 5. Class imbalance and ordinal labels

- **Focal loss (Lin et al., ICCV 2017)** — down-weights easy examples so training attends to
  the hard, rare ones.
- **Class-balanced loss (Cui et al., CVPR 2019)** — weights classes by *effective number of
  samples* rather than raw frequency, which handles the diminishing return of duplicated data.

**What we took from it.** All three diseases train with focal loss (γ = 2.0) **and**
class-balanced weights **and** a weighted random sampler. DR is roughly 73% "No DR"; without
this the minority grades collapse entirely. They still do not fully recover — Mild DR F1 is
0.30 — which is reported rather than hidden.

---

## 6. Explainability

**Grad-CAM (Selvaraju et al., ICCV 2017)** produces a class-discriminative localisation map from
the gradients flowing into the last convolutional layer.

**What we took from it — including its limits.** Grad-CAM shows *where the network responded*,
not *why*, and it is coarse. A heatmap that lands on the macula is not evidence that the model
reasoned about the macula; it is consistent with that and also consistent with a spurious
correlation. We ship it as a sanity aid and say so explicitly.

---

## 7. Calibration

**Guo et al. (ICML 2017)** showed that modern deep networks are systematically over-confident,
and that a single scalar **temperature** fitted on a validation set fixes most of it without
changing any prediction's ranking.

**What we took from it.** Temperature scaling is fitted per model and reported with ECE before
and after (DR referable 0.141 → 0.111 at T = 0.70). A screening tool that displays "94%
confidence" when it is right 78% of the time is actively misleading.

---

## 8. Statistical comparison

**DeLong et al.**, with the fast implementation of **Sun & Xu (IEEE SPL, 2014)**, compares two
correlated ROC curves — the correct test when two models are evaluated on the *same* images.
**McNemar's test** compares paired binary decisions.

**What we took from it.** Both are implemented. Both are still pending for the final
architecture choices, so the ResNet50-for-ROP recommendation currently rests on point metrics —
stated as an open item rather than glossed over.

---

## 9. Shortcut learning — the survey entry that changed the project most

**DeGrave, Janizek & Lee (Nature Machine Intelligence, 2021)** showed that COVID-19 detectors
reported at high AUC were substantially reading **dataset provenance** — laterality markers,
patient positioning, which hospital the scan came from — rather than pathology. Performance
collapsed under external validation.

**What we took from it.** This is the origin of every audit in this project:

| Audit | What it asks | What it found |
|---|---|---|
| Device confound (ROP) | Does the camera predict the label? | **Yes.** Patient-level ROP prevalence varies 30.8% across three devices, and every ROP-positive test image comes from one device. Defensible AUC 0.881, not 0.927 |
| Source confound (Glaucoma) | Does the collection predict the label? | **Yes.** Pooled 0.967 → source-controlled 0.878; 340 of 506 positives come from single-class sources |
| Resolution sensitivity (DR) | Does input size alone move the score? | **Yes** — worth 0.08 AUC |
| Cross-disease admissibility | Do models abstain outside their world? | **No.** The ROP model flagged 59 of 59 adult eyes |
| Prevalence (all) | Does the metric survive a change of population? | Sensitivity and specificity do; **PPV does not** — ROP is ~13% precise in a real NICU |

---

## Where this survey left the design

1. Screen on **referable disease**, not fine-grained grading, because that is the clinical decision.
2. Pick architectures **by measurement**, not by recency.
3. Validate **externally**, with no retraining, and report the drop.
4. Report **confound-controlled** numbers next to pooled ones, always.
5. Treat explainability as a **sanity aid**, not evidence.
6. Calibrate before displaying a confidence.
7. Give the system a way to **refuse** — a gradability gate and a context router — because a
   model with no abstention path will confidently answer a question that has no answer.
