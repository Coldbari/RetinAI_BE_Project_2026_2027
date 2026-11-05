# 5 · Results — Accuracy, Decisions and What They Mean

**Graphs:** [`08_overall_accuracy.png`](graphs/08_overall_accuracy.png) ·
[`09_confusion_matrix.png`](graphs/09_confusion_matrix.png) ·
[`10_per_class_performance.png`](graphs/10_per_class_performance.png) ·
[`11_confidence_and_calibration.png`](graphs/11_confidence_and_calibration.png) ·
[`12_new_hospital_and_why.png`](graphs/12_new_hospital_and_why.png)

*Two words you need. **Recall** = of the photos that really are Stage X, how many did it find.
**Precision** = when it says Stage X, how often it is right. **QWK** = agreement with the human
grader on an ordered scale (0 = guessing, 1 = perfect), where being one stage off costs less
than being four stages off.*

---

## 1 · Headline accuracy

| Measure | 3 training hospitals<br>*(6,700 photos, babies never trained on)* | **A NEW hospital**<br>*(661 photos, sealed, opened once)* |
|---|---|---|
| Exact stage correct (accuracy) | **0.786** | **0.546** |
| Balanced accuracy (fair to rare classes) | 0.652 | 0.520 |
| **Agreement with the grader (QWK)** | **0.897**  *[0.869 – 0.923]* | **0.493** |
| Spotting **any** disease (AUC) | **0.970** | **0.835** |
| Macro-F1 (6 classes) | 0.652  *[0.572 – 0.760]* | 0.476 |

*Square brackets are 95% confidence intervals, computed by resampling **babies** (not
photographs) 2,000 times — because one baby can contribute up to 470 photographs.*

> **The one-line summary:** *"Is anything wrong with this eye?"* transfers to a new hospital.
> *"Which exact stage is it?"* largely does not.

---

## 2 · Performance for every stage (3 training hospitals)

| Stage | Precision | Recall | Photos | Comment |
|---|---|---|---|---|
| **Normal** | 0.929 | 0.951 | 3,837 | reliable |
| **Stage 1** | **0.376** | 0.499 | 577 | **the weak point** — when it says "Stage 1" it is right about a third of the time |
| **Stage 2** | 0.559 | 0.631 | 959 | |
| **Stage 3** | 0.758 | 0.713 | 620 | the best of the disease stages |
| **Stage 4/5** | 0.663 | 0.775 | 89 | ⚠️ all 89 images come from one dataset |
| **AP-ROP** | 0.899 | **0.345** | 618 | says it rarely, but is usually right when it does |

**The single biggest confusion is Stage 1 ↔ Stage 2** — 373 mistakes out of 1,433, in both
directions. That is also the boundary clinicians find hardest (a flat line vs a raised ridge),
so part of this may be label noise rather than model error. Resolving that needs an independent
eye specialist to re-grade a sample; that has not been done.

---

## 3 · What its mistakes look like

1,433 mistakes out of 6,700 predictions (error rate 21.4%).

| Kind of mistake | Count | Share | Average certainty |
|---|---|---|---|
| One stage off (adjacent) | 603 | 42% | 0.73 |
| AP-ROP ↔ a stage | 338 | 24% | 0.66 |
| Healthy eye called diseased | 187 | 13% | 0.73 |
| **⚠️ Diseased eye called NORMAL** | **186** | **13%** | **0.93** |
| **⚠️ AP-ROP called NORMAL** | **89** | 6% | 0.82 |
| Several stages off | 27 | 2% | 0.66 |
| ⚠️ Stage 4/5 called NORMAL | 3 | 0.2% | 0.91 |

**Good news:** only 2% of mistakes are wildly wrong. The ordinal design works — when it is wrong
it is usually *nearly* right, and the severity ordering is preserved.

**Bad news, and the most important safety finding in this project:**

> **The model's average certainty is HIGHEST exactly when it makes the most dangerous mistake.**
> Calling a diseased eye "Normal" carries an average certainty of **0.93** — higher than any
> other kind of error, and higher than the model's own overall accuracy. **141 of those 186
> mistakes were made at over 90% stated certainty.**
>
> This means you **cannot** make the system safe with a rule like "only act on high-confidence
> predictions" — such a rule would keep almost all of the missed-disease cases.

---

## 4 · Can the percentages be trusted? **No.**

| | 3 training hospitals | **NEW hospital** |
|---|---|---|
| Accuracy | 0.786 | 0.546 |
| Average stated certainty | 0.895 | 0.880 |
| **Over-confidence (certainty − accuracy)** | **+0.109** | **+0.334** |
| Calibration error (ECE) | 0.109 | **0.334** |
| Certainty when right **minus** when wrong | 0.193 | **0.041** |

At the new hospital the model is **almost equally confident whether it is right or wrong**
(0.898 vs 0.858). The number carries essentially no information there.

> ### Label the output: **UNCALIBRATED RANKING SCORE — not a probability of disease.**

A simpler baseline model (plain softmax, same backbone, same data) is **3.2× better calibrated
at exactly the same accuracy** (calibration error 0.035 vs 0.109). That is a genuine, measured
cost of the fancy ordinal design. **Fixing it does not need retraining** — temperature scaling on
the existing saved outputs would do it, and it is the number-one item on the to-do list.

---

## 5 · Why the new-hospital score looks so bad — the most important finding

At the new hospital, per photograph, the model finds only about **1 in 4** diseased eyes
(Stage 1: 0.276 · Stage 2: 0.305 · Stage 3: 0.292) while correctly calling **99.6%** of healthy
eyes healthy. That looks like a failure. Part of it is — but a large part of it is **not**.

**ROP grows at the OUTER (temporal) edge of the retina.** A screening exam takes several
photographs of one eye pointing in different directions, and the eye's single diagnosis is
written onto **all** of them — including photographs that physically cannot show the ridge.

Scoring by the direction the camera was pointing, with **no combining at all**:

| Which photographs | Agreement with the grader (QWK) | Photos |
|---|---|---|
| Back of the eye (posterior pole) only | **0.229** | 213 |
| Nose side (nasal) only | **0.352** | 221 |
| **Outer side (temporal) only** | **0.841** | 227 |
| All photographs pooled | 0.473 | 661 |
| *(for reference)* combining one eye's photographs | *0.813* | 230 eyes |

**Photographs of the outer edge, scored one at a time, do as well as combining all of an eye's
photographs.** So a large share of the apparent "failure" is the model correctly calling a
normal-looking photograph normal, and being marked wrong because the label belonged to the eye,
not to that photograph.

**What this does and does not mean:**

- ✅ ROP staging systems should be scored **per eye**, not per photograph, and papers must state
  which unit their labels describe. Results across studies are not comparable otherwise.
- ❌ It does **not** rescue the model. Even scored per eye it still calls **17.7%** of diseased
  eyes normal *(95% CI: 11% – 25%)*.
- ❌ It does **not** describe the current app, which scores **one photograph at a time** — i.e.
  it runs at the worst-performing unit measured here.

---

## 6 · Does the complicated model beat a simple one? **No.**

The ordinal + AP-ROP design was compared against a plain 6-way softmax on the identical backbone,
folds, preprocessing and metrics:

| | Ordinal model | Plain softmax | Verdict |
|---|---|---|---|
| Macro-F1, seed 42 | 0.6923 | 0.6977 | softmax ahead by 0.0055 |
| Macro-F1, seed 1337 | 0.7079 | 0.6855 | **ordinal ahead — the sign reverses** |
| **QWK (agreement)** | **0.8964** | **0.8962** | identical (p = 0.977) |
| **Calibration error** | 0.109 | **0.035** | **softmax clearly better** |
| **New hospital, per photo** | 0.473 | **0.492** | **softmax ahead at both seeds** |
| AP-ROP kept off the stage ladder | ✅ | ❌ | **only the ordinal model can do this** |

Three separate sources of noise are each **larger** than the difference being measured: changing
only the random seed (0.018–0.043), the wobble over a run's last five epochs (0.007–0.056), and
the spread between cross-validation folds (0.086–0.099).

**Honest conclusion:** the two designs are equivalent in accuracy, the simple one is better
calibrated, and the ordinal one's justification is that it represents the disease correctly —
not that it scores higher. That is reported as a finding, not hidden.

---

## 7 · The heat-map explanation is not trustworthy

The app shows a Grad-CAM heat map beside each verdict. It was audited on the live model:

| Measure | Value |
|---|---|
| Attention falling inside the retina | 56.0% |
| Retina's share of the picture area | 53.8% |
| **Attention "lift" (56.0 ÷ 53.8)** | **1.04 — i.e. no better than chance** |
| Peak activation **outside** the retina | **30.7%** |
| Peak activation on the **invented black padding** | 19.3% |
| On **healthy** images: attention on retina / on padding | 39.5% / **43.9%** |

> **The heat map must not be presented as evidence that the model understands the disease.**
> It should be labelled as an audited-and-limited feature, or improved (explain one class
> instead of a sum, use a higher-resolution layer, and refuse to display maps that peak off the
> retina).

---

## 8 · Every decision that was made, and why

| Decision | Choice | Reason |
|---|---|---|
| Binary screening or 6-class staging? | **6-class staging** | 43% of the binary "ROP-positive" class turned out not to be ROP |
| Split by photograph or by baby? | **by baby** | one baby contributes up to 470 photos; splitting by photo inflates scores to 98–100% |
| Balanced sampling? | **no** | it collapsed the ordinal head (QWK 0.121) |
| Best epoch or last epoch? | **last** | picking the best of 20 noisy scores flatters the result |
| Tune the AP-ROP threshold? | **no — fixed at 0.5 in advance** | tuning it on validation data is cheating |
| Merge Stage 4 and Stage 5? | **yes** | only 89 images between them |
| Include treated / regressed eyes? | **no** (1,063 excluded) | post-treatment appearance is not an ICROP stage |
| Include stage 0? | **no** (45 excluded) | immature retina without ROP, and too few |
| Glaucoma and diabetic retinopathy? | **dropped from scope** | the glaucoma evaluation was contaminated by duplicate eyes; rather than publish a broken 0.973 AUC, the disease was removed |
| Open the final test set more than once? | **no — opened exactly once** | the script refuses a second run |
| Report the negative results? | **yes** | they are the finding |
