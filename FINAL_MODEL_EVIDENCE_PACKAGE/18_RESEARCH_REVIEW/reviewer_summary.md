# Reviewer Summary

*Written for a reviewer, examiner or collaborator who has 15 minutes.*

## What this work is

A 6-class ICROP staging model for retinopathy of prematurity, trained on 3,112 images from
1,528 infants across three hospitals, evaluated under infant-grouped 5-fold cross-validation
plus a fourth hospital whose second half was scored **exactly once** under a protocol
committed beforehand.

The architectural contribution is an ordinal (CORN) stage head with a **separate** AP-ROP
branch — encoding ICROP-3's position that the aggressive form is not a point on the stage
scale — plus a hospital adversary. That contribution is **6,405 parameters, 0.031% of the
model**.

## The three results that survive review

1. **The evaluation unit dominates the architecture.** On the locked external set the same
   predictions score corrected QWK **0.473 per photograph** and **0.813 per eye**. This
   package adds the control that makes the claim robust: **temporal-periphery frames, scored
   at image level with no aggregation at all, reach 0.841**, against 0.352 nasal and 0.229
   posterior pole. ROP grows temporally; frames that cannot contain the lesion inherit the
   eye's label. The pattern holds for both architectures.

2. **The architecture does not matter on this corpus.** Structured vs flat softmax: macro-F1
   −0.0055 at seed 42, **+0.0224 at seed 1337 — the sign reverses**; corrected QWK differs by
   **0.0002** (p = 0.977). Three independent noise floors — seed change (0.018–0.043),
   last-5-epoch spread (0.007–0.056) and fold spread (0.086–0.099) — are each larger than the
   effect.

3. **A screening head with 0.970 internal AUC falls to 0.835 externally, and per-stage recall
   falls to ≈0.28.** The negative results are reported as results.

## What this package changes relative to the project's own reporting

| # | change | effect |
|---|---|---|
| 1 | **QWK recomputed under its own declared definition** (5 ordinal classes; `eval.qwk_classes` was read by no code) | fold-0 QWK 0.569 → **0.908**; fold SD 0.145 → **0.029**. The "fold 0 is anomalous on QWK" claim is **withdrawn**; its macro-F1 and calibration anomalies survive |
| 2 | **Aggregation-operator sensitivity measured** | only `max` produces the eye-level gap; five other rules land at the image-level value |
| 3 | **Anatomical control run** | temporal-only image-level QWK 0.841 ≈ eye-level max 0.813 — answers the obvious objection to (2) |
| 4 | **Calibration compared against the baseline for the first time** | ECE **0.109 vs 0.035** at identical accuracy; 359 vs 181 confident errors. A real cost of the contributed head |
| 5 | **Served checkpoint identified** | it is the **fold-0** checkpoint (100% agreement over 663 images) — so the advertised internal operating point does not describe it |
| 6 | **Training history for the current model located and analysed** | it exists in `kaggle_cache/struct42/`; an earlier audit had recorded it as missing |
| 7 | **Error confidence stratified** | "disease called Normal" has the **highest** mean confidence (0.934) of any error class |

## Claims that can safely be made

- "Under patient-grouped cross-validation the model reaches corrected QWK 0.897 [0.869, 0.923] and macro-F1 0.652 [0.572, 0.760] at the image level."
- "On a held-out hospital scored once, image-level corrected QWK is 0.49 and eye-level (max over views) is 0.84 [0.77, 0.89]; temporal-periphery frames alone reach 0.84 at image level."
- "The evaluation unit and the acquisition view move quadratic-weighted kappa by more than 0.35 on identical predictions, across two architectures and two seeds."
- "A taxonomy-faithful ordinal head and a flat softmax are statistically indistinguishable here, and the sign of their difference reverses with the random seed."
- "The ordinal head buys error *structure*: 42% of errors are adjacent-stage and only 1.9% are non-adjacent."
- "The contributed head is measurably worse calibrated than the flat baseline at equal accuracy."
- "ICROP staging results are not comparable across studies unless the label's native unit and the view composition are stated."

## Claims that should NOT be made

- ❌ "The model achieves QWK 0.84." — only per eye under a max rule, or on temporal frames. Per photograph, as the product runs, 0.47.
- ❌ "The structured head outperforms flat softmax." — it does not, on any measured aggregate; it is behind on calibration and on external image-level QWK at both seeds.
- ❌ Anything about stage 4/5 (n = 4) or AP-ROP (n = 12) externally.
- ❌ "The Grad-CAM shows the model has learned ROP pathology." — retina attention lift is 1.04, i.e. chance.
- ❌ "The model's confidence can be used to decide which predictions to trust." — the most dangerous error class is the most confident.
- ❌ "The model generalises to new hospitals." — one site, and per-stage recall ≈0.28 there.
- ❌ "First ordinal ROP model" / "first AP-ROP task." — the project's own literature review rules these out. The defensible framing remains *"first joint ordinal + AP-ROP-branch model with probe-quantified site debiasing and held-out-hospital evaluation."*

## Anticipated reviewer questions, answered

**Q: Your image→eye gap is just the max operator.**
Partly — and we now show the alternatives (majority 0.398, mean-prob 0.396, max-prob 0.351, random view 0.482, image 0.473). But the gap reproduces *without any aggregation* on temporal frames alone (0.841 vs nasal 0.352, posterior 0.229), which no operator argument can explain. See [`../11_AGGREGATION_AND_VIEW_ANALYSIS/`](../11_AGGREGATION_AND_VIEW_ANALYSIS/).

**Q: Five folds is too few for a paired test.**
Agreed, and stated. The p-values are reported as "nothing detectable", not as equivalence at power β. The stronger argument is the **sign reversal across seeds** plus three noise floors each exceeding the effect.

**Q: Isn't the QWK correction self-serving?**
It moves the *external* number **down** (0.523 → 0.473) and withdraws one of the project's own claims. Both values are published side by side in [`../07_TEST_RESULTS/qwk_analysis/`](../07_TEST_RESULTS/qwk_analysis/) with the pair counts.

**Q: Why is AP-ROP recall 0.345 when you claim the branch helps?**
Because 476 of 618 AP-ROP validation frames sit in fold 0 and 470 of those come from **one infant** with a per-session label. The patient-clustered CI is **[0.254, 0.836]**. We do not defend the point estimate; the interval *is* the label-unit finding in statistical form.

**Q: Is there leakage?**
Recomputed from scratch: zero patient overlaps and zero image overlaps in every comparison; the five validation folds exactly partition 6,700 images. The residual risk is class–source confounding, which is quantified and declared rather than denied.

**Q: Why report a model you say is equivalent to a baseline and worse calibrated?**
Because the paper's contribution is measurement, not architecture — and reporting the architecture as a null result, with its calibration cost, is what makes the measurement claims credible.
