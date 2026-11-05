# START HERE — RetinAI ROP Staging, Current Model Evidence Package

> **Public copy.** The per-image CSV files, the case photographs and Grad-CAM examples, the
> privacy audits and the prediction arrays that carry patient IDs are kept out of this
> repository, because they are patient-level. Links to them in this package will not resolve.
> The reports, summary figures and scripts are all here.

*Built 2026-09-21 from repository commit `f0d4508`. Every number in this folder was
recomputed from raw per-sample prediction artifacts by the scripts in
[`20_AUDIT/scripts/`](../20_AUDIT/scripts/). Nothing was copied from previous reports.*

---

> **A second, independent verification pass has been run over this package.**
> Read [`VERIFICATION_PASS_2.md`](VERIFICATION_PASS_2.md) first. It confirms every conclusion
> below that it re-measured, corrects three of them, and adds the one analysis that was
> missing everywhere: the screening endpoint evaluated on the locked test at the **infant**
> unit, where specificity is **0.000**. It also carries an open PHI release blocker
> ([`../20_AUDIT/PHI_PRIVACY_AUDIT.md`](../20_AUDIT/PHI_PRIVACY_AUDIT.md)).



## What is this?

A single, self-contained evidence folder for **one** model. You should be able to read this
file and the ten figures in [`17_USER_FACING/`](../17_USER_FACING/) and understand what the
model does, how well, and where it fails — without opening the original project.

It is an **audit package, not a marketing package**. It contains the results that make the
model look good and the results that do not, side by side, with the same prominence.

> ⚠️ **Not a medical device.** Research prototype. No regulatory clearance, no prospective
> validation. Nothing here supports clinical use. See [`18_RESEARCH_REVIEW/limitations.md`](../18_RESEARCH_REVIEW/limitations.md).

---

## Which model is being evaluated?

An **EfficientNetV2-S** network with three output heads: an ordinal (CORN) stage head, a
separate AP-ROP yes/no branch, and a hospital-adversary branch used only during training.
Seed 42, natural-prevalence sampling, 20 epochs, last-epoch weights.

The exact checkpoint that the web app serves was **identified empirically** for this package:
it is the **fold-0 checkpoint** of that run (100% prediction agreement over 663 images).
Full identity card: [`CURRENT_MODEL_ID.txt`](CURRENT_MODEL_ID.txt).

Retired and superseded models (the ResNet50 binary screener, the weighted-sampler arm)
appear **only** in [`19_COMPARISONS/retired_models/`](../19_COMPARISONS/retired_models/) and are
labelled as historical. They are never mixed into a current-model chart.

---

## What data was used?

| | images | infants | hospitals |
|---|---|---|---|
| Training pool | 3,112 | 1,528 | 3 (Ostrava · Shenzhen · ROP-VL) |
| Internal testing (5-fold out-of-fold) | 6,700 | 1,528 | same 3 |
| **External testing (locked, opened once)** | **661** | **116** | **1 — never seen in training** |

Photographs of the retinas of premature babies. Splits are grouped **by infant**: no baby
appears in both a training set and a test set. This was re-verified from scratch for this
package — **zero** patient overlaps and **zero** image overlaps across every comparison
([`03_DATA_LEAKAGE/`](../03_DATA_LEAKAGE/)).

The data is very unbalanced: 1,109 normal images against 89 of the rarest disease class.

---

## What does the model predict?

For one photograph it outputs one of six labels — **Normal, Stage 1, Stage 2, Stage 3,
Stage 4/5, AP-ROP** — plus a probability for each. The web app also derives a binary
screening score, *"any ROP" = 1 − P(Normal)*.

---

## How was it tested?

1. **5-fold cross-validation grouped by infant** on the three training hospitals (6,700 images).
2. **A held-out hospital**, split in two halves by infant. The second half (661 images) was
   scored **exactly once**, under a protocol written and committed before it was opened.
3. Everything is reported at three units: **per photograph**, **per eye**, and **per anatomical view**.

---

## What does it do well?

- **It is genuinely learning.** Training error falls smoothly; performance is far above a
  metadata-only baseline (macro-F1 0.19) and above the majority-class prior.
- **Finding healthy eyes.** Normal recall 0.951 internally, **0.996** at the new hospital.
- **Ranking disease severity in order.** Corrected agreement with the human grade (QWK)
  **0.896 ± 0.029** internally — errors are mostly "one stage off", not random.
- **Screening for *any* disease.** AUC **0.970** internally, **0.835** at the new hospital.
- **Late-stage and advanced disease internally.** Stage 3 recall 0.713, Stage 4/5 recall 0.775.

## Where does it fail?

- **Stage 1 is unreliable.** When the model says "Stage 1" it is right **38%** of the time.
- **AP-ROP recall is 0.345** per photograph — it misses two thirds of aggressive-form frames.
- **Staging does not survive a change of hospital.** Per-photograph recall at the new site:
  Stage 1 **0.276**, Stage 2 **0.305**, Stage 3 **0.292**. Overall accuracy falls 0.786 → 0.546.
- **It is most confident when it makes the most dangerous mistake.** Calling a diseased eye
  "Normal" carries the *highest* average confidence of any error type (0.934); 141 of those
  186 errors were made at >90% confidence.
- **The heat-map explanation is not trustworthy** — 31% of Grad-CAM peaks land outside the
  retina, and on healthy images 44% of the attention sits on black padding that the
  preprocessing invented.

---

## What do the graphs mean?

Read [`17_USER_FACING/`](../17_USER_FACING/) in numbered order; each figure has a plain-English
title and [`17_USER_FACING/USER_GUIDE.md`](../17_USER_FACING/USER_GUIDE.md) explains every term.
Every figure in this package carries a footer naming the **model, evaluation unit and split**,
so no chart can be read out of context.

Two terms you will keep meeting:

- **Recall (sensitivity)** — of the photographs that really are class X, what share did the model find?
- **QWK (quadratic-weighted kappa)** — agreement with the human grade on an *ordered* scale,
  where being one stage off costs less than being four stages off. 0 = chance, 1 = perfect.

---

## Can the probabilities be trusted?

**No. Treat the percentages as a ranking score, not as a probability of disease.**

- Internally the model's stated confidence exceeds its accuracy by **0.109** (ECE 0.109).
- At the new hospital calibration error is **0.334**, and the gap between its confidence
  when right (0.898) and when wrong (0.858) shrinks to **0.041** — effectively no signal.
- The *simpler* baseline model is **3.2× better calibrated at the same accuracy** (ECE 0.035).

Details: [`10_CONFIDENCE_CALIBRATION/calibration_report.md`](../10_CONFIDENCE_CALIBRATION/calibration_report.md).

---

## Does it generalize to another hospital?

**Partly, and in a specific way that matters.**

| per photograph | training hospitals | new hospital |
|---|---|---|
| accuracy | 0.786 | **0.546** |
| QWK (corrected) | 0.896 | **0.493** |
| Normal recall | 0.951 | 0.996 |
| Stage 1/2/3 recall | 0.499 / 0.631 / 0.713 | **0.276 / 0.305 / 0.292** |

Scored **per eye** (combining an eye's several photographs with a max rule) the agreement
rises to **0.839 [0.768–0.890]**. That figure is real but it is *operator-dependent*: five
other reasonable aggregation rules all land at 0.35–0.48.

The reason is anatomical, and this package demonstrates it with a control the original
project had not run: **photographs of the temporal (outer) retina, scored one at a time with
no aggregation at all, reach QWK 0.841** — while nasal frames reach 0.352 and posterior-pole
frames 0.229. The ROP ridge grows temporally. Frames that cannot contain the finding inherit
the eye's diagnosis as their label, and the model correctly calls them normal.
See [`11_AGGREGATION_AND_VIEW_ANALYSIS/`](../11_AGGREGATION_AND_VIEW_ANALYSIS/).

---

## What has NOT been demonstrated?

- Any clinical validity, utility or safety.
- Per-stage performance at a new hospital (it is poor: ~0.28 recall).
- Stage 4/5 and AP-ROP externally — only **4** and **12** locked images exist. Pre-registered
  as *not externally validated*; do not quote those cells.
- That the structured head beats a plain softmax on anything measurable except taxonomy fidelity.
- That the Grad-CAM shows the model "understanding" the pathology.
- Generalisation beyond **one** external hospital.

---

## What are the main limitations?

1. Staging collapses off-site; only "is anything wrong" transfers.
2. Probabilities are uncalibrated and become almost uninformative externally.
3. The rarest classes are tiny (89 stage 4/5 training images, all from a single source).
4. Grad-CAM cannot be used as evidence of what the model looked at.
5. The served checkpoint is a single CV fold, so internal out-of-fold operating points
   do not describe it.
6. One external hospital, retrospective data, no clinician adjudication of the labels.

Full list with evidence: [`18_RESEARCH_REVIEW/limitations.md`](../18_RESEARCH_REVIEW/limitations.md).

---

## What should happen next?

In priority order (full specifications in
[`18_RESEARCH_REVIEW/recommended_future_experiments.md`](../18_RESEARCH_REVIEW/recommended_future_experiments.md)):

1. **Temperature-scale the CORN cascade** on out-of-fold logits — the calibration defect is
   the cheapest real problem to fix and the most consequential for the product.
2. **Fix `configs/rop_staging_structured.yaml`** (`weighted_sampler: false`) so the repository
   reproduces its own model, and stamp the served checkpoint with its fold/seed/hash.
3. **Adopt the temporal-view control** as the paper's primary instrument for the label-unit
   finding, replacing max-over-views aggregation.
4. **Clinician adjudication** of ~300 stratified labels. Nothing here substitutes for it.

---

## Where everything lives

| folder | what's in it |
|---|---|
| `00_START_HERE` | this file, the model card, the executive summary, the evidence-status ledger |
| `01_MODEL` … `05_TRAINING` | model, data, leakage, preprocessing, training behaviour |
| `06_VALIDATION`, `07_TEST_RESULTS` | internal cross-validated performance, corrected QWK |
| `08`–`11` | eye-level, error, calibration, and aggregation/view analyses |
| `12_MODEL_COMPARISON` | current model vs its own flat-softmax baseline |
| `13_EXTERNAL_GENERALIZATION` | the held-out hospital |
| `14_EXPLAINABILITY`, `15_DEPLOYMENT`, `16_STATISTICAL_ANALYSIS` | Grad-CAM audit, cost, uncertainty |
| `17_USER_FACING` | **the ten-figure story — start here if you are not an ML specialist** |
| `18_RESEARCH_REVIEW` | findings, limitations, gaps, reviewer Q&A, next experiments |
| `19_COMPARISONS` | retired / historical models, clearly quarantined |
| `20_AUDIT` | provenance, corrections ledger, reproducibility, privacy audit, build scripts |

---

## ⚠️ Note on package contents (read before sharing)

While this package was being built, a **second, independent process** (a peer Claude Code
session working on the same repository) wrote into the same folder. Its files are:

- `_build/` — its own scripts and intermediate CSVs
- `20_AUDIT/verification_pass_2/` — a parallel results tree
- ~60 additional CSV files dropped into `02_DATASET/`, `03_DATA_LEAKAGE/`, `06_VALIDATION/`,
  `07_TEST_RESULTS/`, `08_EYE_LEVEL_ANALYSIS/`, `09_ERROR_ANALYSIS/`,
  `10_CONFIDENCE_CALIBRATION/`, `11_AGGREGATION_AND_VIEW_ANALYSIS/`, `16_STATISTICAL_ANALYSIS/`

**Nothing produced by this build was deleted or altered** — every headline number was
re-verified after the collision and reproduces exactly. Those files were **left in place rather
than removed**, because deleting another process's work is the repository owner's decision.

**Two consequences you should act on:**

1. **Some of those foreign CSVs contain raw image paths**, including filenames that carry an
   infant index. This build's own files contain none. Review or remove them before distributing:
   ```bash
   find . -name '*.csv' -exec grep -lIE 'data/rop/(ostrava|shenzhen|rop_vl|multiview)/' {} \;
   ```
2. **Where a foreign file and one of this build's files cover the same topic, they may not
   agree** — they were produced by different code with different conventions. Everything
   described in this README and in [`../20_AUDIT/source_artifacts.md`](../20_AUDIT/source_artifacts.md)
   was produced by `20_AUDIT/scripts/s01`–`s12` and verified by
   [`../20_AUDIT/verification_report.md`](../20_AUDIT/verification_report.md) (23 of 25 checks
   pass; both failures are documented, external to this build, and reported rather than hidden).

Full account: [`../20_AUDIT/privacy_audit.md`](../20_AUDIT/privacy_audit.md) §6.2.
