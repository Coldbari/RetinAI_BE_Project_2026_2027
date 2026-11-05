# The Model in Ten Pictures

*Read these in order. No machine-learning background needed — every term is explained in
[`USER_GUIDE.md`](USER_GUIDE.md). Every figure names the model, the evaluation unit and the
data split in its footer, so none of them can be quoted out of context.*

| # | figure | the question it answers |
|---|---|---|
| 01 | [`01_dataset.png`](01_dataset.png) | **What data was used?** How many photographs, how many babies, how unbalanced |
| 02 | [`02_model_pipeline.png`](02_model_pipeline.png) | **How does the model work?** Every step from photograph to prediction |
| 03 | [`03_training_behavior.png`](03_training_behavior.png) | **Did it learn properly?** Yes — but its score swings wildly between training passes |
| 04 | [`04_overall_performance.png`](04_overall_performance.png) | **How good is it overall?** And how much it drops at a new hospital |
| 05 | [`05_confusion_matrix.png`](05_confusion_matrix.png) | **What does it confuse with what?** |
| 06 | [`06_per_class_performance.png`](06_per_class_performance.png) | **How good is it at each category?** With uncertainty bars |
| 07 | [`07_error_examples/`](07_error_examples/) | **What do its mistakes look like?** Real, de-identified examples |
| 08 | [`08_confidence.png`](08_confidence.png) | **Is it sure when it's right?** Not reliably — and it is most sure on its worst mistakes |
| 09 | [`09_calibration.png`](09_calibration.png) | **Can the percentages be trusted?** No |
| 10 | [`10_external_validation.png`](10_external_validation.png) | **Does it work at a hospital it has never seen?** Partly, and there is an anatomical reason why not |

## 11 — The important limitations (no figure; read this)

1. **It has only been tested at one outside hospital**, retrospectively, on stored photographs.
2. **Per photograph, at a new hospital, it gets the stage right about 1 time in 4.** It is only reliable at telling healthy from not-healthy.
3. **The percentages it shows are not probabilities.** Treat them as a ranking.
4. **It is most confident exactly when it makes the most dangerous mistake** — calling a diseased eye normal.
5. **The heat-map picture is not a real explanation.** About a third of the time it points outside the retina entirely.
6. **The rarest and most severe categories rest on a handful of images** — 89 training photographs for stage 4/5, and only 4 at the external hospital.
7. **No independent eye specialist has re-read the labels** used to train or test this model.
8. **It is not a medical device** and must not be used for any clinical decision.

## Reading the figures honestly

- Wherever a bar has very few photographs behind it, the count is printed on the figure and coloured red when it is too small to interpret. **A bar built on 4 observations is not evidence, however tall it is.**
- "Error bars" mean different things in different figures and each figure says which: a spread between five training runs, a spread between five model checkpoints, or a true 95% confidence interval. They are never mixed.
- Percentages always have their denominator shown.
