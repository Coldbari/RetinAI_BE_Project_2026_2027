# Eye-Level Analysis — external locked hospital

*230 eyes from 116 infants, 661 photographs, 1–5 acquisition views per eye (median 3).
Eye identity comes from the dataset's own Acquisition View Table. 219 of 230 eyes carry a
single diagnosis across all of their views — **the label's native unit is the eye, not the photograph**.*

Tables: [`eye_metrics.csv`](eye_metrics.csv), [`eye_qwk.csv`](eye_qwk.csv), [`eye_confidence_intervals.csv`](eye_confidence_intervals.csv)
Figures: [`eye_confusion_matrix.png`](eye_confusion_matrix.png), [`eye_confusion_matrix_normalized.png`](eye_confusion_matrix_normalized.png), [`eye_per_class_metrics.png`](eye_per_class_metrics.png)

## Headline, protocol rule (label = max over views, prediction = max over views)

| metric | value | 95% CI (patient-clustered, 2,000 reps) |
|---|---|---|
| QWK (corrected, 5 ordinal) | **0.8394** | [0.7683, 0.8903] |
| accuracy | 0.6783 | [0.6043, 0.7468] |
| macro-F1 (present classes) | 0.6160 | [0.5277, 0.7181] |
| **diseased eyes called Normal** | **17.7%** | [11.1%, 25.5%] |

The interval resamples **infants**, not eyes — the two eyes of one baby are correlated and must move together.

## The same predictions under six aggregation rules

| rule | units | QWK (corrected) | accuracy | diseased eyes called Normal |
|---|---|---|---|---|
| **no aggregation (image level)** | 661 images | 0.473 ± 0.014 | 0.520 | 48.9% |
| one random view per eye | 230 | 0.482 ± 0.012 | 0.525 | 49.8% |
| max probability per class | 230 | 0.351 ± 0.048 | 0.471 | 69.5% |
| mean probability over views | 230 | 0.396 ± 0.027 | 0.482 | 61.8% |
| majority vote | 230 | 0.398 ± 0.022 | 0.499 | 57.6% |
| **max over views (protocol)** | 230 | **0.813 ± 0.017** | 0.646 | 17.8% |
| **temporal view only, image level** | 216 | **0.851 ± 0.015** | 0.690 | 11.5% |

*± is spread across the 5 fold-checkpoints, not a confidence interval.*

**Two things follow, and both must be stated together:**

1. **Moving to the eye as a unit buys nothing by itself.** Mean-probability, majority-vote and one-random-view aggregations all land at the image-level value (0.40–0.48). Only `max` produces 0.81.
2. **But the `max` result is not an artifact** — because the temporal-only control reproduces it *without any aggregation at all*. See below.

## Why the max rule is nonetheless the right rule here

Taking the maximum over an eye's views is the most optimistic ordinal aggregator available, and it was adopted after inspecting the dev half. That is a legitimate reason for a reviewer to be suspicious.

The suspicion is answered by the anatomical control in [`../11_AGGREGATION_AND_VIEW_ANALYSIS/`](../11_AGGREGATION_AND_VIEW_ANALYSIS/): scoring **only temporal frames, one at a time, with no aggregation**, gives QWK 0.841–0.851 — matching the eye-level max value. Nasal frames give 0.352 and posterior-pole frames 0.229.

The ROP ridge grows in the temporal periphery. A posterior-pole photograph of a Stage 2 eye does not contain Stage 2; it contains a normal-looking posterior pole with the eye's diagnosis stamped on it. The model calls it normal and is marked wrong. `max` recovers the eye's diagnosis because at least one view usually does contain the finding — which is exactly what a clinician does when reading a set of views.

**Recommended reporting:** quote the eye-level number *with the rule named*, and quote the temporal-view control beside it as the mechanism. The temporal control is the stronger instrument because it requires no aggregation choice.

## Per-class at the eye level

See [`eye_metrics.csv`](eye_metrics.csv). Supports are small (Stage 4/5 and AP-ROP are in single digits at the eye level) and are printed on the figure; **any cell with fewer than 10 eyes is marked red and should not be interpreted.**

## What this does NOT show

- It does not show that the model stages eyes well enough to be useful: **~1 in 6 diseased eyes is still called Normal**, with a CI reaching 1 in 4.
- It does not transfer to the deployed product. **The web app scores one photograph at a time** and therefore operates at the worst-performing unit measured here (QWK 0.47, 49% of diseased frames called Normal). Any clinical framing of the 0.84 figure would be describing a workflow the software does not implement.
