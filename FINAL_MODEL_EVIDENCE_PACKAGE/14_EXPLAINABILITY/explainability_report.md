# Explainability Report — Grad-CAM audit of the CURRENT model

*Generated fresh for this package by `20_AUDIT/scripts/s08_gradcam.py`: the **served**
checkpoint, the **served** objective (`∂ log Σ prob6[1:] / ∂ features`), the last convolutional
block of the EfficientNetV2-S backbone, 88 stratified images from the external dev half.
Attention is measured on the **raw** map before any display masking.*

Table: [`gradcam_quality_per_image.csv`](gradcam_quality_per_image.csv) · Summary: [`gradcam_quality_summary.json`](gradcam_quality_summary.json) · Figure: [`gradcam_quality_analysis.png`](gradcam_quality_analysis.png)

## Method

The model's 384×384 input frame is decomposed into three disjoint regions:

1. **retina** — the fitted retinal field of view (`models/common/data_quality.py::retina_mask`)
2. **synthetic letterbox padding** — black pixels the preprocessing *invented* to square the image
3. **real camera surround** — genuinely captured non-retinal frame

Separating 2 from 3 matters: without it, attention on invented pixels would be counted as "the model reading the frame".

## Results

| measure | value |
|---|---|
| mean attention inside the retina | **56.0%** |
| mean attention on **synthetic letterbox padding** | **28.7%** |
| mean attention on real camera surround | 15.3% |
| retina's share of the frame area | 53.8% |
| **attention "lift" (attention ÷ area)** | **1.04** |
| CAM peaks falling **outside** the retina | **30.7%** |
| CAM peaks landing on **invented padding** | **19.3%** |

### The central finding

> **Lift = 1.04.** The model's attention on the retina is statistically indistinguishable
> from what you would get by spreading attention uniformly over the whole frame. The
> Grad-CAM carries essentially **no evidence** that the model localises to retinal tissue,
> let alone to pathology.

### Per-class breakdown — worse where it matters most

| true class | attention on retina | attention on invented padding |
|---|---|---|
| **normal** | **39.5%** | **43.9%** |
| stage_1 | 53.2% | 34.4% |
| stage_2 | 62.6% | 24.2% |
| stage_3 | 60.0% | 24.0% |
| stage_4_5 | 58.8% | 16.3% |
| arop | 78.9% | 12.5% |

On **healthy** eyes the model puts more attention on black bars the preprocessing created than on the retina. There is a plausible mechanical reading — with no lesion to attend to, the map defaults to frame geometry — but it means the heat map shown beside a "Normal" verdict is not an explanation of that verdict.

Attention on the retina is also **uncorrelated with confidence** (right-hand panel of the quality figure): the model is not more certain when it has looked in the right place.

## Corroboration with the parent project's independently-run audit

| measure | this package (n=88, fresh run) | parent `results/rop/cam_leakage.json` (n=663) |
|---|---|---|
| attention on retina | 56.0% | 55.5% |
| attention on synthetic padding | 28.7% | 27.8% |
| peak outside retina | 30.7% | 35.8% |

Two independent implementations agree. The finding is not an artifact of this package's code.

The parent audit also records that the **retired, disqualified ResNet50 screening head attends the retina MORE** than the model that replaced it (75.7% vs 55.5%; peak outside retina 2.9% vs 35.8%), and explicitly notes that its own prediction to the contrary "did not hold". That comparison is quarantined in [`../19_COMPARISONS/retired_models/`](../19_COMPARISONS/retired_models/).

## Examples

- [`gradcam_examples/`](gradcam_examples/) — cases where the peak **is** on the retina (contact sheet + individual PNGs)
- [`gradcam_failure_examples/`](gradcam_failure_examples/) — cases where the peak is **outside** the retina, including on invented padding

Both galleries draw the fitted retinal boundary in white so the reader can judge the overlay themselves. Case IDs are salted hashes; no filenames or patient identifiers appear.

**These galleries are shown as evidence *about* the explanation method, not as evidence that the model works.** Selecting only the attractive overlays would invert the finding.

## Verdict

**The Grad-CAM output of this model must not be presented as evidence that it understands or localises ROP pathology.** It is currently usable only as a debugging surface for the frame-geometry problem it exposes.

See [`explainability_limitations.md`](explainability_limitations.md) for the consequences and the fixes.
