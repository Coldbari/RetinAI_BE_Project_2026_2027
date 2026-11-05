STATUS:
RETIRED / HISTORICAL

MODEL:
ResNet50 binary ROP screening head ("No ROP" / "ROP Detected"), configs/rop.yaml,
served threshold 0.1933, trained on the device-stratified binary split.

DATE:
Retired 2026-08-19 and replaced by the current structured staging model's P(any ROP).

NOT CURRENT MODEL RESULT
================================================================================

# Why it was retired

An external audit (`scripts/rop_screening_external_spec.py`) ran it over 663 images from a
hospital it had never trained on:

| | value |
|---|---|
| images flagged positive | **663 of 663** — including every healthy eye |
| specificity at the served threshold | **0.000** |
| external AUC | **0.691** |
| minimum score among external healthy eyes | 0.2939 |
| served threshold | 0.1933 |

The site's **entire** score distribution sat above the cut-off. The threshold had been
derived on a different hospital and did not transfer.

# The one number worth carrying forward

| | retired ResNet50 | CURRENT structured model |
|---|---|---|
| external dev AUC ("any ROP") | 0.691 | **0.821** |
| external locked AUC ("any ROP") | not measured | **0.835** |
| behaviour at its served threshold externally | flags 100% of images | sens 0.905 / spec 0.387 (dev, in-sample for the threshold choice) |

# The awkward result that must not be hidden

On Grad-CAM quality the **retired** model is *better*:

| | retired ResNet50 | CURRENT structured model |
|---|---|---|
| attention inside the retina | **75.7%** | 55.5% |
| CAM peak outside the retina | **2.9%** | 35.8% |
| attention on synthetic padding | 0.0% (it does not letterbox) | 27.8% |

The project's own artifact `results/rop/cam_leakage.json` records the prediction that the
disqualified head would read the frame *less* than its replacement, and states
`"prediction_held": false`. The comparison is not like-for-like — the current model
letterboxes and the retired one does not — but it is not flattering and it is reported.

# Lesson carried into the current model

1. A threshold is a property of a **site**, not of a model. Any new deployment site needs a
   locally re-derived operating point.
2. External specificity must be checked before a screening head is served, not after.
3. Acting on one's own negative audit — retiring a served model — is the behaviour that makes
   the rest of the project's claims credible.

Source artifacts: `results/rop/screening_external_spec.json`, `results/rop/screening_rebase.json`,
`results/rop/cam_leakage.json`.
