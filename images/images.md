# Images

Every figure in this directory is **generated from recorded results**, not drawn by hand.
Re-running an experiment and re-running the generator keeps the picture and the number in step.

- Figures 01–12 come from `scripts/make_graphs.py` in the development repository, which reads
  `results/*.json`.
- `system_architecture.png` and `flowchart.png` come from
  [../software/make_diagrams.py](../software/make_diagrams.py) in this repository.

---

## Diagrams

| File | What it shows |
|---|---|
| [`system_architecture.png`](system_architecture.png) | Training on Kaggle GPU, serving on CPU, and the single shared preprocessing module that both use |
| [`flowchart.png`](flowchart.png) | The screening algorithm: gate → route → preprocess → infer → explain → report, including the reject path |

---

## Evidence figures

| # | File | The question it answers |
|---|---|---|
| 01 | [`01_confusion_matrices.png`](01_confusion_matrices.png) | Of N patients, how many did each model get right? TP/FN/FP/TN per model |
| 02 | [`02_ppv_vs_prevalence.png`](02_ppv_vs_prevalence.png) | **Of 100 patients flagged, how many really have it?** And why that answer changes between a test set and a clinic |
| 03 | [`03_roc_curves.png`](03_roc_curves.png) | Ranking ability on each model's own held-out set |
| 04 | [`04_rop_operating_curve.png`](04_rop_operating_curve.png) | ROP's sensitivity/specificity trade — the deployed point is one choice among many |
| 05 | [`05_rop_device_confound.png`](05_rop_device_confound.png) | How much of ROP's 0.927 was the model recognising the camera |
| 06 | [`06_glaucoma_source_confound.png`](06_glaucoma_source_confound.png) | The same question for glaucoma: pooled 0.967 vs within-source 0.878 |
| 07 | [`07_glaucoma_heldout.png`](07_glaucoma_heldout.png) | Zero-shot on two collections withheld from training — 0.735 vs 0.936 |
| 08 | [`08_dr_external_and_resolution.png`](08_dr_external_and_resolution.png) | DR on IDRiD, and what happens when the same images are downscaled |
| 09 | [`09_cross_disease_matrix.png`](09_cross_disease_matrix.png) | Every image through every model — what patient-context routing prevents |
| 10 | [`10_calibration.png`](10_calibration.png) | Is the displayed confidence honest? |
| 11 | [`11_safety_before_after.png`](11_safety_before_after.png) | What the gradability gate and routing removed |
| 12 | [`12_rop_subgroups.png`](12_rop_subgroups.png) | Can fairness be demonstrated? No — and why that is the finding |

---

## How to read them

**Start with 02.** It carries the result that most changes how this system should be described:
sensitivity and specificity survive a change of population, **PPV does not**. ROP catches
essentially every case and is right about **13%** of the time it raises a flag in a real NICU.
That is still useful — it is a triage filter — but it is not what "AUC 0.93" sounds like.

**05, 06 and 07 are the same lesson three times.** A pooled number across cameras or collections
is not purely a measure of the disease; part of it is the model recognising where the image came
from. Every headline in this project is shown with and without that confound.

**09 is the argument for routing.** The ROP model fires on 59 of 59 adult eyes. ROP is a disease
of prematurity and cannot occur in an adult, so all 59 are wrong — the model has no way to
abstain from a question outside its world. Routing, not retraining, is the fix.

**12 is a negative result and is kept as one.** The available metadata cannot support a subgroup
fairness claim. Reporting that honestly is better than a breakdown built on inadequate data.

---

## Caveats that travel with these figures

- **Small samples.** The confusion matrices rest on 33–40 images. DR's flawless 20/20 has a 95%
  interval of [83.2%, 100%] — read it as "at least 83%", not "flawless".
- **Flat 100% lines in figure 02** are an artefact of zero false positives on 20 and 18 negatives,
  not a guarantee. ROP's curve is the shape to trust.
- **Intervals cluster by patient.** The ROP figures resample infants, not images — 2,411 images
  come from 29 infants, so image-level intervals would be far too narrow.

---

## On the colours

The original deck palette (cyan / purple / gold / green / red) **fails** a colourblind check: red
against green sits at deuteranopia ΔE 6.5, inside the 6–8 warn band, and all five hues fall
outside the lightness band for a dark surface.

These figures use validated replacements — **DR blue `#3987e5`, ROP orange `#d95926`, Glaucoma
aqua `#199e70`** — which pass all six checks with worst adjacent CVD ΔE 9.4.

Colour is never the only cue: every series is direct-labelled, correct cells in the confusion
matrices carry an outline rather than a different hue, and the impossible cells in figure 09 are
ringed as well as coloured.

---

## Not in this repository

**No patient images.** No fundus photograph of any real patient — adult or infant — is committed
here. During development, six infant patient photographs were found to be publicly served; they
were removed, the history was swept, and an egress guard was added. Nothing of that kind is in
this repository, and nothing of that kind should be added to it.
