STATUS:
RETIRED / HISTORICAL

MODEL:
Structured head v1 — identical architecture to the current model, but trained with a
class-balanced WeightedRandomSampler instead of natural-prevalence sampling.

DATE:
Superseded before the current model; v2 (natural sampling) is the current model.

NOT CURRENT MODEL RESULT
================================================================================

# What happened

| metric, fold 0 | v1 (balanced sampler) | CURRENT model (natural sampling) | flat softmax baseline |
|---|---|---|---|
| QWK (legacy 6-class, as reported at the time) | **0.121** | 0.569 | 0.594 |
| stage_4_5 precision | **0.043** | — | — |
| Normal recall | **0.456** | 0.939 | — |
| macro-F1 | 0.403 | 0.553 | 0.563 |

Roughly 420 false predictions of a class with 19 validation images.

# Why

CORN is a **cumulative cascade**: rank *k* is predicted when the running product of
conditional sigmoids exceeds 0.5. Those conditional thresholds calibrate to the prevalence
the model was **trained** on. A balanced sampler shows the model a world where stage_4_5 is
about 9× its true share, so on natural-prevalence validation data every high-rank conditional
fires far too often.

A flat softmax tolerates the same sampler because `argmax` is rank-free — which is exactly
why the two heads must be compared **at matched sampling**. An earlier comparison in this
project was confounded on this axis (flat trained weighted, structured trained natural); the
current comparison in [`../../12_MODEL_COMPARISON/`](../../12_MODEL_COMPARISON/) is matched.

# Why this matters for the CURRENT model today

> ⚠️ **`configs/rop_staging_structured.yaml` still says `weighted_sampler: true`.**
> Every headline run overrode it to `false` from the Kaggle kernel. Running the repository's
> own documented command therefore reproduces **this retired arm**, not the current model.
> See finding H-6 in [`../../18_RESEARCH_REVIEW/critical_findings.md`](../../18_RESEARCH_REVIEW/critical_findings.md).

Source: `kaggle_kernels/rop_structured_benchmark.py` (the `WEIGHTED_SAMPLER = "false"` constant
and its comment block), `docs/ROP_PROJECT_STATE.md` §5.
