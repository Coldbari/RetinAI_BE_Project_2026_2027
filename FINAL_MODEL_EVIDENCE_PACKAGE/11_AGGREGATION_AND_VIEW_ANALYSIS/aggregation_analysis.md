# Aggregation and Anatomical-View Analysis

*External locked hospital · 661 photographs · 230 eyes · 116 infants · 1–5 acquisition views per eye.
Region labels come from the dataset's own Acquisition View Table (temporal 227 · nasal 221 · posterior pole 213).
Tables: [`aggregation_results.csv`](aggregation_results.csv), [`region_results.csv`](region_results.csv).*

---

## The question

Scored per photograph, the model's agreement with the human grade is **QWK 0.473**. Scored per eye with the protocol's max-over-views rule, it is **0.813**. That is a gap of +0.34 from nothing but a change of denominator.

Two explanations compete:

- **(A) Label-unit / anatomical.** The diagnosis belongs to the *eye*. Views that cannot physically contain the ridge inherit it anyway. The model correctly calls those views normal and is marked wrong. Aggregating repairs the mismatch.
- **(B) Operator artifact.** `max` over ~3 draws from a low-sensitivity classifier mechanically recovers a skewed label distribution. The gap is arithmetic, not anatomy.

Both predict the observed gap. They are separated below.

---

## Test 1 — does *any* eye-level aggregation produce the gap, or only `max`?

Figure: [`aggregation_comparison.png`](aggregation_comparison.png)

| rule | units | QWK (corrected) | accuracy | diseased units called Normal |
|---|---|---|---|---|
| no aggregation (image level) | 661 | 0.473 ± 0.014 | 0.520 | 48.9% |
| one random view per eye | 230 | 0.482 ± 0.012 | 0.525 | 49.8% |
| max probability per class | 230 | 0.351 ± 0.048 | 0.471 | 69.5% |
| mean probability over views | 230 | 0.396 ± 0.027 | 0.482 | 61.8% |
| majority vote | 230 | 0.398 ± 0.022 | 0.499 | 57.6% |
| **max over views (protocol rule)** | 230 | **0.813 ± 0.017** | 0.646 | 17.8% |

**Result: only `max` produces the gap.** Every other rule lands at or below the image-level value. Reducing to one unit per eye (the random-view rule) buys nothing — so the gap is not "fewer, cleaner units".

Taken alone this favours explanation **(B)**, and it is the objection any reviewer will raise. It is also the reason the eye-level figure must never be quoted without naming its operator.

---

## Test 2 — the decisive control: score only the views that can contain the finding

ROP's defining lesion — the demarcation line, ridge, and extraretinal proliferation — arises at the **temporal periphery** of the developing retina. If explanation (A) is right, then restricting to temporal frames and scoring them **one at a time, with no aggregation whatsoever**, should recover the eye-level agreement.

Figure: [`region_qwk_comparison.png`](region_qwk_comparison.png)

| unit | n images | QWK (corrected) | accuracy | stage 1–3 frames called Normal |
|---|---|---|---|---|
| all frames pooled | 661 | 0.473 ± 0.014 | 0.520 | 48.9% |
| **posterior pole only** | 213 | **0.229 ± 0.023** | 0.451 | **71.7%** |
| **nasal only** | 221 | **0.352 ± 0.018** | 0.425 | **63.4%** |
| **temporal only** | 227 | **0.841 ± 0.015** | 0.678 | **12.7%** |
| *(reference)* eye level, max rule | 230 eyes | 0.813 ± 0.017 | 0.646 | 17.8% |

**Temporal frames, scored individually with no aggregation at all, reach QWK 0.841 — matching and slightly exceeding the eye-level max value of 0.813.**

This cannot be produced by explanation (B): there is no aggregation operator involved. It is a pure image-level score on a subset selected by *anatomy*, not by outcome. And the direction is exactly what ROP anatomy predicts — the model calls 71.7% of stage 1–3 posterior-pole frames normal, against 12.7% of temporal frames.

The same pattern holds for the flat baseline (temporal 0.843, nasal 0.369, posterior 0.257), so it is a property of the data, not of this architecture.

## Verdict

> **Explanation (A) is supported by evidence.** The image→eye gap is driven by the label's
> native unit being the eye combined with the anatomical distribution of the lesion.
>
> **Explanation (B) is not refuted as a contribution to the specific number 0.813** — `max`
> *is* the most optimistic aggregator, and it was chosen after seeing dev-half results.
>
> Therefore: report the temporal-view control as the primary instrument, and the eye-level
> max figure as a secondary with its operator named. The control requires no aggregation
> choice, so it cannot be argued with on those grounds.

Per-region confusion matrices: [`temporal_analysis.png`](temporal_analysis.png), [`nasal_analysis.png`](nasal_analysis.png), [`posterior_analysis.png`](posterior_analysis.png).

---

## What this does and does not license

**Licensed:**
- "On this corpus, per-image ICROP metrics understate agreement, and the size of the understatement is a function of which anatomical view was photographed."
- "ROP staging results are not comparable across studies unless the label's native unit and the view composition are stated."
- "A screening system built on these labels should aggregate to the eye."

**Not licensed:**
- "The model achieves QWK 0.84." It does so *on temporal frames*, or *per eye under a max rule*. Per photograph, as the product actually runs, it achieves 0.47.
- "The model stages eyes well." Even at the eye level it calls **17.7% [11.1–25.5%]** of diseased eyes Normal.
- Any extension of this to the internal corpus. Ostrava and Shenzhen carry no acquisition-view metadata, so the decomposition cannot be repeated there. The internal error counts in [`../09_ERROR_ANALYSIS/`](../09_ERROR_ANALYSIS/) are therefore **not** view-corrected.

## Caveats

- `±` throughout is **spread across the 5 fold-checkpoints**, not a confidence interval.
- Region counts are 213–227 images each; the per-region QWK intervals are correspondingly wide and are not bootstrapped here (the checkpoint spread is given instead).
- 11 of 230 eyes carry more than one distinct label across their views; the max rule resolves those to the highest stage.
- 8 eyes have only a single view, for which every aggregation rule is identical.
