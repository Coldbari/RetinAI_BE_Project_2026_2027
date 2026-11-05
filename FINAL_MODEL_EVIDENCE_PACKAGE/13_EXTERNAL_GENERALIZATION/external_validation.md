# External Generalization

*One held-out hospital (Multi-View cohort), never used in training. Split by infant into a
**dev** half (663 images — used freely for audits and threshold selection) and a **locked**
half (661 images — scored exactly once, under a protocol committed before it was opened).
All figures below are the locked half unless stated.*

Tables: [`external_metrics.csv`](external_metrics.csv), [`external_metrics_per_checkpoint.csv`](external_metrics_per_checkpoint.csv), [`site_comparison.csv`](site_comparison.csv), [`external_image_level.json`](external_image_level.json)

---

## Internal ≠ external: the headline

Figure: [`internal_vs_external.png`](internal_vs_external.png)

| metric, **per photograph** | internal (5-fold OOF, 3 training hospitals) | **external locked (new hospital)** |
|---|---|---|
| accuracy | 0.786 | **0.546** |
| balanced accuracy | 0.652 | 0.520 |
| macro-F1 (6 classes) | 0.652 | **0.476** |
| **QWK (corrected)** | **0.897** | **0.493** |
| unweighted κ | 0.655 | 0.355 |
| "any ROP" AUC | **0.970** | **0.835** |
| ECE | 0.109 | **0.334** |

## Per-class recall — the shape of the failure

Figure: [`external_confusion_matrix.png`](external_confusion_matrix.png)

| class | internal recall | **external recall** | external n |
|---|---|---|---|
| Normal | 0.951 | **0.996** | 232 |
| Stage 1 | 0.499 | **0.276** | 105 |
| Stage 2 | 0.631 | **0.305** | 164 |
| Stage 3 | 0.713 | **0.292** | 144 |
| Stage 4/5 | 0.775 | *0.750* | **4 — NOT VALIDATED** |
| AP-ROP | 0.345 | *0.500* | **12 — NOT VALIDATED** |

At the new hospital the model becomes a **near-perfect normal detector and a poor stager**. External Normal *precision* is only 0.530 — it says "Normal" far too often. Stage 4/5 and AP-ROP were pre-registered as not externally validated at these counts; their cells are shown for completeness and **must not be quoted**.

## Checkpoint spread vs uncertainty — kept separate

Five fold-checkpoints scoring the same locked set give image-level QWK **0.4732 ± 0.0140** and accuracy **0.5204 ± 0.0150**. That `±` is **variation between checkpoints on a fixed test set**. It is *not* a confidence interval about the population and is never labelled as one in this package. Population uncertainty is given only where a patient-clustered bootstrap was run (see [`../08_EYE_LEVEL_ANALYSIS/eye_confidence_intervals.csv`](../08_EYE_LEVEL_ANALYSIS/eye_confidence_intervals.csv)).

The 5-checkpoint ensemble (mean of prob6, argmax) gives accuracy 0.527, QWK 0.464 — no better than the served single checkpoint.

## Per-site comparison

Figure: [`site_comparison.png`](site_comparison.png) · [`site_comparison.csv`](site_comparison.csv)

Performance is broken down across the three training hospitals and the held-out one. The drop is **not** a gradual degradation across sites — the three training hospitals are broadly comparable and the held-out one is qualitatively different. Device variation could not be analysed at the external site: its device field is `unknown` for every image.

## How much of the external drop is real?

This is the most important nuance in the package, and it is answered in [`../11_AGGREGATION_AND_VIEW_ANALYSIS/`](../11_AGGREGATION_AND_VIEW_ANALYSIS/).

A large part of the per-photograph "failure" is the model correctly calling a photograph normal when that photograph does not show the lesion — because the label belongs to the **eye**, and ROP grows **temporally**:

| external locked, image level | QWK |
|---|---|
| all frames | 0.473 |
| posterior-pole frames only | 0.229 |
| nasal frames only | 0.352 |
| **temporal frames only** | **0.841** |
| *(eye level, max over views)* | *0.813* |

So the correct statement is: **staging at the photograph level does not transfer; staging at the eye level partially does (QWK 0.839 [0.768–0.890]), but even there ~18% of diseased eyes are called Normal.** The deployed app scores photographs, i.e. it operates at the worse unit.

## What may be driving the residual drop

| candidate | evidence available |
|---|---|
| view composition / label unit | **strong** — the region table above |
| colour and resolution domain shift | post-CLAHE mean blue 53.6 / 73.3 / 74.5 at training sites vs **87.6** externally; four different native resolutions. The site-randomisation augmentation and the DANN branch exist to attack exactly this |
| class-mix shift | external prevalence differs from internal; balanced accuracy drops less (0.652 → 0.520) than raw accuracy, so mix alone does not explain it |
| grader differences | **untestable here** — no image in this corpus has been read by two graders |
| device differences | **untestable here** — the external device field is `unknown` |

## Limits of this evidence

- **One** external hospital, one country pairing, retrospective.
- The external **dev** half was used to select the served binary threshold, so any operating-point number quoted on the dev half is optimistic. The locked half is clean for the image-level metrics but was *not* used to select anything.
- The eye-level aggregation rule was adopted after inspecting dev-half results (disclosed in the protocol's Amendment 1). The image-level locked number is the only fully pre-registered external result.
- Nothing here establishes clinical generalisation. **One external site is a sanity check, not a generalisation claim.**
