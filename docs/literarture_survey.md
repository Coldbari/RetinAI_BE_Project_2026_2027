# Literature Survey

The work we reviewed, grouped by the design question it answered. Each entry ends with what we
actually changed because of it, since a survey that does not change the design is not much of a
survey.

Full IEEE-format citations are in [../reference/paper.md](../reference/paper.md).

---

## 1. Deep learning for diabetic retinopathy

Gulshan et al. (JAMA, 2016) trained an Inception-v3 on 128,175 fundus photographs graded by 54
ophthalmologists and reached AUC 0.991 for referable DR, with sensitivity above 0.90 at high
specificity. It is the reference point for basically every DR screening result since.

Two things in that paper shaped our project more than the headline number did. First, the
endpoint is referable DR, a binary "does this patient need a specialist" decision, rather than
5-class grading. Screening is a referral decision, so that is what should be measured. Second,
their labels came from multiple graders per image, and public datasets do not have that.

So we made referable DR (grade ≥ 2) our screening endpoint, reported as an AUC. We still report
5-class numbers, but we frame them as capped by label noise.

On that point: published inter-grader agreement on EyePACS-style 5-class DR grading sits around
QWK 0.83 to 0.85. That is a ceiling, because a model cannot be more consistent than the labels it
learns from. Our 5-class QWK of 0.728 sits below but near that ceiling, which is roughly where we
would expect it to be, and it is why we reframed the endpoint.

## 2. Retinopathy of prematurity

ICROP, the International Classification of Retinopathy of Prematurity (2005 revision), defines
the clinical vocabulary: zone (I, II, III, how far vascularisation has progressed from the optic
disc), stage (1 to 5, the severity of the ridge or detachment), and plus disease (venous dilation
and arteriolar tortuosity, which is the strongest treatment indicator).

Our deployed model collapses all of that into binary ROP / No-ROP. That is a real loss of
clinical meaning and we list it as a limitation. The six-class ICROP staging work we are doing
now exists precisely because of this gap.

The screening economics also matter. ROP screening means repeated bedside examination of every
pre-term infant inside a narrow window where treatment still works. That is why we chose the ROP
operating point to maximise sensitivity (0.990) at the cost of specificity (0.398). A missed case
is irreversible blindness. A false alarm is one extra examination.

## 3. Glaucoma from fundus photographs

Classical methods estimate the cup-to-disc ratio by segmenting the optic disc and cup.
Learning-based methods either classify the whole image or crop an optic-disc region of interest
first.

We looked at ORIGA (650 images), REFUGE (1,200), G1020 (1,020), Drishti-GS, PAPILA, and the
SMDG-19 aggregation that pools many of these.

What we noticed while assembling SMDG-19 is that these datasets are very heterogeneous in camera,
field of view, population and prevalence, and several of them contribute only one class. Pooling
them and quoting a single AUC hides that completely. Our source-stratified evaluation, where we
report per-source AUC and a source-controlled aggregate, came directly out of this.

## 4. Architectures

ResNet (He et al., CVPR 2016) introduced residual connections and is still the strongest generic
backbone on small datasets in our own sweep. EfficientNetV2 (Tan & Le, ICML 2021) uses compound
scaling with fused-MBConv blocks and gets better accuracy per parameter on large datasets.

We did not assume which would win. A five-backbone sweep on the ROP set (EfficientNet-B0,
EfficientNetV2-S, MobileNetV3-Large, ResNet50, DenseNet121) put ResNet50 well ahead at AUC 0.9635
against 0.8700 for the next best. That is the opposite of what "newer is better" would predict,
though it is consistent with the observation that heavily-scaled architectures underperform on
small datasets. DR and Glaucoma, which have far more data, use EfficientNetV2-S.

## 5. Class imbalance and ordinal labels

Focal loss (Lin et al., ICCV 2017) down-weights easy examples so training pays attention to the
hard, rare ones. Class-balanced loss (Cui et al., CVPR 2019) weights classes by effective number
of samples rather than raw frequency, which handles the diminishing returns of duplicated data.

We use all of it: focal loss at gamma 2.0, class-balanced weights, and a weighted random sampler,
on all three diseases. DR is roughly 73% "No DR", and without this the minority grades collapse
entirely. They still do not fully recover, with Mild DR F1 at 0.30, and we report that rather
than hide it.

## 6. Explainability

Grad-CAM (Selvaraju et al., ICCV 2017) produces a class-discriminative localisation map from the
gradients flowing into the last convolutional layer.

We took the limits along with the method. Grad-CAM shows where the network responded, not why,
and it is coarse. A heatmap landing on the macula is not evidence the model reasoned about the
macula. It is consistent with that, and equally consistent with a spurious correlation. We ship
it as a sanity aid and say so.

## 7. Calibration

Guo et al. (ICML 2017) showed that modern deep networks are systematically over-confident, and
that a single scalar temperature fitted on a validation set fixes most of it without changing any
prediction's ranking.

We fit a temperature per model and report ECE before and after, for example DR referable going
from 0.141 to 0.111 at T = 0.70. A screening tool that displays "94% confidence" while being right
78% of the time is actively misleading.

## 8. Statistical comparison

DeLong's test, with the fast implementation from Sun & Xu (IEEE SPL, 2014), compares two
correlated ROC curves, which is the right test when two models are evaluated on the same images.
McNemar's test compares paired binary decisions.

Both are implemented in our codebase. Both are still pending for our final architecture choices,
so the ResNet50-for-ROP recommendation currently rests on point metrics. We state that as an open
item rather than glossing over it.

## 9. Shortcut learning, the paper that changed this project most

DeGrave, Janizek & Lee (Nature Machine Intelligence, 2021) showed that COVID-19 detectors
reported at high AUC were substantially reading dataset provenance, things like laterality
markers, patient positioning, and which hospital the scan came from, rather than pathology.
Performance collapsed under external validation.

Every audit in this project traces back to that paper:

| Audit | What it asks | What we found |
|---|---|---|
| Device confound (ROP) | Does the camera predict the label? | Yes. Patient-level ROP prevalence varies by 30.8% across three devices, and every ROP-positive test image comes from one device. Defensible AUC 0.881, not 0.927 |
| Source confound (Glaucoma) | Does the collection predict the label? | Yes. Pooled 0.967 falls to 0.878 source-controlled, and 340 of 506 positives come from single-class sources |
| Resolution sensitivity (DR) | Does input size alone move the score? | Yes, it is worth 0.08 AUC |
| Cross-disease admissibility | Do models abstain outside their world? | No. The ROP model flagged 59 of 59 adult eyes |
| Prevalence (all) | Does the metric survive a change of population? | Sensitivity and specificity do. PPV does not, and ROP is only about 13% precise in a real NICU |

## 10. Ordinal regression and adversarial debiasing, for the staging work

CORN (Shi, Cao & Raschka, 2023) reformulates a K-class ordinal problem as K−1 conditional binary
tasks, which guarantees rank consistency in a way a flat softmax cannot.

ICROP stages are ordered. Stage 1 to 2 to 3 to 4/5 is a progression, and confusing Stage 1 with
Stage 4/5 is a much worse error than confusing Stage 1 with Stage 2, but a flat 6-way softmax
treats every confusion as equally wrong. Swapping to a CORN head, plus a separate AP-ROP branch
since AP-ROP is not a point on that ladder but an aggressive form that co-occurs with stages,
took fold-0 macro-F1 from 0.520 to 0.554.

DANN (Ganin et al., JMLR 2016) attaches a domain classifier through a gradient-reversal layer so
the encoder gets pushed toward features the domain cannot be predicted from.

Site is 99.7% decodable from our staging corpus, so an adversary is clearly warranted. But the
literature almost always evaluates the adversary with a probe trained over all classes, and that
is not a sound instrument when disease prevalence varies by site, because the probe can score
well by reading disease instead of provenance.

Running both probes gave us different answers. The naive probe falls from 0.859 to 0.820 and
reads as success. The disease-controlled probe, run on normal-class images only so disease is
held constant, does not move at all: 0.882 to 0.885. The adversary removed the disease-site
correlation, not site appearance.

This is the methodological point our paper draft is built around. A naive site probe
systematically overstates adversarial debiasing, and reporting one without the other makes a
domain-adaptation result look stronger than it really is.

---

## Where this left the design

1. Screen on referable disease rather than fine-grained grading, because that is the clinical
   decision being made.
2. Pick architectures by measurement, not by recency.
3. Validate externally with no retraining, and report the drop.
4. Report confound-controlled numbers next to pooled ones, every time.
5. Treat explainability as a sanity aid, not as evidence.
6. Calibrate before displaying a confidence.
7. Give the system a way to refuse. A gradability gate and a context router, because a model with
   no abstention path will confidently answer a question that has no answer.
