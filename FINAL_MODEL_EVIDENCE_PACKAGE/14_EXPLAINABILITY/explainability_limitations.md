# Explainability — Limitations and What Would Fix Them

## What is wrong

1. **Attention on the retina is at chance** (lift 1.04 against the retina's 53.8% area share).
2. **30.7% of CAM peaks fall outside the retina**; 19.3% land on letterbox padding that the preprocessing invented — pixels that contain no information about any eye.
3. **The problem is worst on healthy images** (39.5% retina / 43.9% padding), i.e. exactly the cases a screening tool reports most often.
4. **Attention quality is uncorrelated with confidence**, so a "trust the heat map when the model is sure" heuristic does not work.

## Why it happens

- **Letterboxing creates large, perfectly uniform black regions.** A final-block activation map has a coarse spatial grid; gradients pooled over channels can put mass on these regions without that mass meaning anything. The letterbox is nonetheless *load-bearing* for accuracy (it is the anti-shortcut step that removes an aspect-ratio site cue), so it cannot simply be removed.
- **Grad-CAM at the last convolutional block of EfficientNetV2-S at 384×384 gives a ~12×12 map**, upsampled 32×. Fine peripheral structures — which is where ROP lives — are below that resolution.
- **The explained objective is `log Σ prob6[1:]`**, a sum over five classes. A gradient with respect to a sum of heterogeneous classes is not a gradient with respect to any one finding.

## What this does and does not invalidate

**Does not invalidate:** the classification metrics. Nothing here suggests the model is classifying *on* the padding — the letterbox is identical across sites by construction, so it cannot carry site or class information. The measured performance stands.

**Does invalidate:** any use of the heat map as clinical evidence, as a reason to trust an individual prediction, or as a claim that the model has learned retinal pathology. If the product displays a heat map beside a verdict, users will read it as the reason for the verdict, and on this evidence it is not.

## Recommended fixes, in order of cost

| # | fix | cost | expected effect |
|---|---|---|---|
| 1 | **Mask the CAM to the fitted retina before display, and print the raw off-retina fraction beside it** | trivial | stops the invented-padding overlay from being shown as evidence; makes the defect visible instead of hidden. The app already masks the display — it should also *report* the unmasked statistic. |
| 2 | **Explain a single class, not a sum** — take the gradient of the predicted class's logit, or of the AP-ROP branch alone | trivial | removes the heterogeneous-objective problem |
| 3 | **Use an earlier block, or Grad-CAM++/HiResCAM** | small | recovers spatial resolution in the periphery where ROP lives |
| 4 | **Add a retina-localisation sanity gate**: refuse to display a CAM whose peak is off-retina | small | turns a silent failure into an explicit abstention |
| 5 | **Train with a vessel/ridge segmentation auxiliary task** (HVDROPDB masks are already in the corpus) | large — requires retraining | the only route to attention that is *anatomically* grounded rather than merely masked |

## Current recommended wording for any report or demo

> "The heat map indicates which image regions most influenced the screening score. It has
> been audited: on this model roughly 31% of peak activations fall outside the retinal field
> of view, and attention on retinal tissue is no greater than chance given its area. **The
> heat map should not be used to judge whether an individual prediction is trustworthy.**"
