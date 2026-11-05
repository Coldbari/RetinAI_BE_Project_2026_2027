# Deployment Limitations

> ⚠️ **This model is a research prototype. It is not a medical device, has no regulatory
> clearance, has never been evaluated prospectively, and must not be used to make or
> influence any clinical decision.**

## Hard limits (no configuration can work around these)

| # | limit | number |
|---|---|---|
| 1 | Per-photograph staging at an unseen hospital is close to unusable | stage recall 0.28–0.31 |
| 2 | Probabilities are not probabilities | external ECE 0.334; right-vs-wrong confidence separation 0.041 |
| 3 | Missed disease is the most confident error class | mean confidence 0.934; 76% above 0.90 |
| 4 | Heat-map explanations are unreliable | 30.7% of peaks off-retina; attention lift 1.04 |
| 5 | Two of six classes are externally unvalidated | stage 4/5 n=4, AP-ROP n=12 — pre-registered as such |
| 6 | Stage 1 output carries little information | precision 0.376 internally |
| 7 | Tested at exactly one external site | — |
| 8 | Labels were never re-read by an independent grader | no inter-grader κ exists |

## Operational limits

| # | limit | note |
|---|---|---|
| 9 | **Preterm infants only.** | On adult fundus images the model flagged 41/45 eyes as ROP-positive. The routing table enforces this and must never be bypassed. |
| 10 | **The served threshold does not transfer between sites.** | The previous screening head flagged 663 of 663 images at a hospital it had not trained on, because that site's entire score distribution sat above its cut-off. Any new site needs a locally re-derived threshold. |
| 11 | **The served operating point is not estimated for the served checkpoint.** | Threshold selected on pooled out-of-fold scores; the served model is fold 0's checkpoint, which saw 4/5 of that data. |
| 12 | **The app scores single photographs.** | All the favourable evidence in this package is at the eye level. The deployed unit is the worst-performing one measured. |
| 13 | **No test-time augmentation at serving**, despite the repository README claiming 5-view TTA. | Harmless — the evaluation artifacts are also single-pass — but the documentation is wrong. |
| 14 | **No input quality gate is applied by this model.** | A defocused, off-centre or non-fundus image will still receive a confident six-class verdict. |
| 15 | **CC BY-NC** on the external test hospital's dataset. | Blocks commercial use of any result validated on it. |

## Failure modes an operator would actually encounter

| symptom | underlying cause |
|---|---|
| "Everything looks Normal at our site" | external Normal recall 0.996 with Normal precision 0.530 — the model over-calls Normal off-distribution |
| "It was 99% sure and completely wrong" | 359 of 6,700 internal predictions are wrong above 0.90 confidence; externally 178 of 661 |
| "The heat map is pointing at the black border" | 19.3% of CAM peaks land on synthetic letterbox padding |
| "Stage 1 alerts are mostly noise" | Stage 1 precision 0.376 |
| "It disagreed with itself between two photos of the same eye" | the model has no eye-level state; per-photograph QWK is 0.47 externally |

## Correct disclaimer for any demo, report or screenshot

> Research prototype for ROP staging. Evaluated on 6,700 internal and 661 external retrospective
> photographs. **Per-photograph staging accuracy at an unseen hospital is 55%, and the displayed
> percentages are an uncalibrated ranking score, not probabilities of disease.** The heat map has
> been audited and does not reliably localise to retinal tissue. Not a medical device; not
> validated prospectively; every finding must be confirmed by a qualified clinician.
