# Comparisons — HISTORICAL AND RETIRED RESULTS ONLY

> ## ⚠️ NOTHING IN THIS FOLDER DESCRIBES THE CURRENT MODEL.
>
> Every file here is kept because it is scientifically useful as a **contrast**. None of it
> appears in any current-model chart elsewhere in this package. Each file carries its own
> `STATUS: RETIRED / HISTORICAL` header.

## Why keep them at all

Three of the project's most instructive results are negative results about models that were
subsequently withdrawn. Deleting them would hide the reasoning that produced the current
model.

| folder / file | what it is | why it is useful |
|---|---|---|
| [`retired_models/resnet50_binary_screener.md`](retired_models/resnet50_binary_screener.md) | the previous binary ROP screening head, retired 2026-08-19 | it flagged **663 of 663** images at a hospital it had never trained on. The canonical demonstration that thresholds do not transfer |
| [`retired_models/weighted_sampler_v1_arm.md`](retired_models/weighted_sampler_v1_arm.md) | the first version of the current architecture, trained with a class-balanced sampler | CORN + balanced sampler = prior-shift collapse (QWK 0.121). Explains why the current model trains at natural prevalence — and why the committed config, which still says `weighted_sampler: true`, is wrong |
| [`retired_models/legacy_qwk_values.csv`](retired_models/legacy_qwk_values.csv) | every QWK value as previously published, beside its corrected value | the audit trail for the metric correction |
| [`retired_models/five_backbone_benchmark.md`](retired_models/five_backbone_benchmark.md) | the single-fold, single-seed architecture sweep that selected EfficientNetV2-S | provisional; its in-fold spread is inside the noise of one training run |
| [`retired_models/glaucoma_and_dr.md`](retired_models/glaucoma_and_dr.md) | two diseases removed from project scope | shows what was dropped and why |

## Rules observed in this package

1. No historical number appears in a current-model figure.
2. No historical file is referenced from `00_START_HERE/`, `17_USER_FACING/` or any results folder as though it were current.
3. Every historical file opens with an explicit STATUS block naming the model and the date.
4. Where a historical result is *contrasted* with the current model (e.g. the Grad-CAM comparison), the contrast is stated in prose in the current-model report and the numbers are sourced from here, labelled.
