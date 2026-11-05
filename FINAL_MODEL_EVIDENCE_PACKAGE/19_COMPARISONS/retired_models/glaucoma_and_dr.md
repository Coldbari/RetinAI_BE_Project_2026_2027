STATUS:
RETIRED / HISTORICAL — OUT OF PROJECT SCOPE

MODEL:
(a) Glaucoma classifier — WITHDRAWN.
(b) Diabetic retinopathy referable-grade classifier — PARKED.

DATE:
Glaucoma withdrawn post-presentation; DR parked 2026-08-18 into junk/.

NOT CURRENT MODEL RESULT
================================================================================

# (a) Glaucoma — withdrawn, not merely deprioritised

A reported AUC of 0.973 was found to be contaminated: the G1020 arm **double-mounts per eye**,
so the same eye appeared more than once across the evaluation. Rather than publish a broken
number, the disease was removed from scope entirely.

**Nothing about glaucoma appears anywhere in the current model or this package.**

# (b) Diabetic retinopathy — parked

Last reported figures (EfficientNetV2-S, referable = grade ≥ 2): AUC 0.906 (95% CI 0.89–0.92),
sensitivity 0.84 / specificity 0.89, Messidor-2 external 0.837, 5-class QWK 0.728.

Parked because the project narrowed to ROP. The code, configs and results were **moved, not
deleted**, to `junk/`, with paths mirroring their origin.

> ⚠️ The repository README still headlines the DR AUC of 0.906 in its results table even
> though the supporting evidence now lives in `junk/`. That is a documentation inconsistency,
> not a claim about the current model.

# Why this is in the package at all

To make explicit that **the current model does one thing**: 6-class ICROP staging of ROP from
a single fundus photograph of a preterm infant. Any figure, metric or claim about glaucoma or
diabetic retinopathy found elsewhere in the repository is out of scope here and must not be
attributed to the current model.
