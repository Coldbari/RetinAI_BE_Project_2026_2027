# Showing RetinAI — 5-minute runbook

```bash
./demo.sh
```
Boot takes ~15–40s (five DINOv2 checkpoints load and warm up, plus the gate), then the
browser opens on the one page. Everything runs on this laptop — no internet needed once the
weights are in place.

Fetch the weights first if this is a fresh clone; they are not in the repository:

```bash
.venv/bin/python scripts/get_weights.py
```

> **On a fresh clone this is not the served model.** `get_weights.py` fetches the two older
> checkpoints. The five DINOv2 checkpoints are not on the model repository yet, and without
> them the verdict comes from the retired ResNet50, which flagged 663 of 663 images at a new
> hospital, and no stage is shown. Do not quote a run like that.

## The model

One model answers both questions on the page: **DINOv2 ViT-S/14, 5-fold ensemble, seed 42**
(arm B of the v3 protocol). Five networks, one per cross-validation fold, each grade the
photo into six ICROP classes (Normal, Stage 1, 2, 3, Stage 4/5, AP-ROP); their probabilities
are averaged. The stage is the top class; the ROP verdict is 1 − P(Normal) against a fixed
line. The configuration was fixed by the protocol before the new hospital was scored, and
the v4 rule (cleaner data, newer recipe) did not beat it, so B stayed.

## Pick the patient context first

`Preterm infant (NICU screening)` is mandatory. It is not a formality: the model is
trained only on preterm infants, and without routing an earlier model flagged **41 of 45
adult eyes** as ROP-positive. If someone asks "what stops it running on the wrong
patient?" — this is the answer, and the API returns 400 if the context is missing.

## Images for the demo

**The demo images are not in this repository, and this is deliberate.**

The 34 images the demo was scored on are private infant fundus photographs from the
Ostrava cohort, and their filenames encode gestational age, birth weight, sex and
diagnosis. A filename is patient data. This log book is a public repository, so the
images and the file list live only in the private working repository, and this page
reports what they produced rather than shipping them.

The same reasoning retired the Gallery page, and since 7 Nov 2026 the whole multi-page design
is gone; `webapp/static/` stays free of patient photographs. See the PHI remediation note in the main README.

To run the demo on your own data, upload any preterm infant fundus photograph on the
page with the context set to `Preterm infant (NICU screening)`.

**Whole-eye mode:** upload 1–16 photos of one eye with OD/OS. The eye gets the most severe
stage across its photos; the heatmap is drawn for the photo that drove it.

## Measured on the 34 evaluation images — read the caveat first

**These images are not evidence.** All 20 normal images and 6 of the 14 ROP images are
byte-identical to images in the ensemble's cross-validation pool, so each was in the
training data of four of the five networks. They demonstrate behaviour, not accuracy.

| | screening verdict correct | stage correct (Normal vs any stage) |
|---|---|---|
| ROP present | 1 / 14 | 7 / 14 |
| No ROP | 20 / 20 | 19 / 20 |

The ROP misses are the more mature, heavier infants (GA 33–39, BW 1700–3600 g) with milder
disease; the hits are GA 24–26. Even so, the screening line flags only 1 of the 14 ROP
photos on its own. That is the single-photo weakness measured at the new hospital below,
and the reason to use whole-eye mode. Say this before you're asked.

## Numbers to quote

All from the new hospital's development half (a hospital the models never trained on),
eye level, this exact ensemble. The locked half has not been scored for this model.

- **230 eyes: QWK 0.752, accuracy 63.9%, macro-F1 0.568.** The five single checkpoints
  range QWK 0.745–0.774, so the ensemble is about as good as one network. It is served for
  stability, not as an improvement.
- Screening line **0.9894** on 1 − P(Normal): the pre-registered cross-validation operating
  point (largest threshold keeping out-of-fold any-ROP sensitivity ≥ 0.90; specificity
  0.996 there).
- At the new hospital that line caught **66% of eyes with any ROP** and **92% of eyes with
  stage 3 or worse**, with **no false alarms** on the 50 normal eyes. A single photograph
  caught only **31%** of ROP images. It under-calls mild ROP: a negative means "not
  flagged", not "healthy".
- Scores are not probabilities: calibration error (ECE) **0.144** at the new hospital.
- Two models were **retired** on the way here. The ResNet50 screening head flagged **all
  663** images at the new hospital, every healthy eye included. The EfficientNet model's
  threshold could not be carried over: on this ensemble it would flag every one of the 150
  normal images. Retiring both is the strongest thing in the project — it shows the audits
  were real.
- The contribution is measurement, not architecture: the ViTs tie the EfficientNet on the
  protocol's cross-validation rule.

## The one page

Since 7 Nov 2026 the app is a single page at `/`: upload, verdict, stage, Grad-CAM, PDF
download, whole-eye mode, and both model cards at the foot. The dashboard, history, about,
gallery and diseases pages are gone with the old design. `webapp/static/` is web-servable, so a
repo guard still forbids putting patient photographs there.

**Comparison panel.** Beside our model's stage the page shows the stage from a teammate's
RETFound-DINOv2 + LoRA model (three seeds averaged) with an agree / disagree mark. It never
decides anything: the verdict, the Grad-CAM and the PDF stay with the served ensemble. Say
plainly if asked: it was trained on a photo-level split of our photos, so its own 84.4% carries
infant leakage and it has seen 390 development-half and 331 sealed-half photos. **Demo it on
new photos only.** The page warns when an uploaded photo is one that either model trained or
tested on (the seen-photo guard, `results/comparison/seen_photos.json`, hashes of image bytes).
The panel's weights (630 MB) come from the teammate's `rop_foundation` package through
`scripts/deploy_comparison_weights.py` and are not on the model repository; a clone without
them boots without the panel, and everything else on the page is unchanged.

## If it will not start

```bash
.venv/bin/python scripts/get_weights.py   # re-pull the two older checkpoints
tail -20 /tmp/retinai-demo.log
```
