# Per-hospital calibration — a committed conclusion is withdrawn

*Verification pass 2. Figure: [`site_calibration_corrected.png`](site_calibration_corrected.png).
Tables: [`site_calibration_corrected.csv`](../20_AUDIT/verification_pass_2/10_CONFIDENCE_CALIBRATION/site_calibration_corrected.csv),
[`per_class_gap_by_site.csv`](../20_AUDIT/verification_pass_2/10_CONFIDENCE_CALIBRATION/per_class_gap_by_site.csv).
Source data: pooled 5-fold out-of-fold predictions, n=6,700, every image scored once by a model that did not train on it.*

## The claim being checked

`results/rop/per_source_calibration.json` states:

> "the disease-controlled panel shows most of that spread is CASE-MIX, not a site-specific
> calibration defect: holding the class constant collapses the range to 0.012-0.027 (2.3×)
> and every site is well calibrated on the class they share. **Do not report the 8x.**"

## What was verified

Every raw number reproduces exactly: pooled accuracy 0.7861, mean confidence 0.8954, ECE
0.1092, Brier(top-1) 0.1388; per-site ECE Ostrava 0.1534 / ROP-VL 0.0493 / Shenzhen 0.0182;
Normal-only ECE 0.0272 / 0.0218 / 0.0117.

## What does not hold

The "disease control" restricts to **Normal only** — the single class every site is nearly
perfect on (accuracy 0.943 / 0.974 / 1.000). Restricting to the easiest shared class is not a
case-mix adjustment; it removes the classes on which the sites differ.

Per-class over-confidence (confidence − accuracy), which the control excludes by construction:

| | Normal | Stage 1 | Stage 2 | Stage 3 | AP-ROP |
|---|---:|---:|---:|---:|---:|
| ostrava | +0.027 (n=2980) | **+0.533** (252) | +0.268 (458) | **+0.564** (125) | +0.530 (470) |
| ropvl | −0.008 (623) | +0.059 (236) | +0.060 (338) | +0.112 (241) | +0.124 (148) |
| shenzhen | −0.012 (234) | +0.052 (89) | −0.016 (163) | −0.029 (254) | — |

Under **direct standardisation** — reweighting each site's per-class gaps to the pooled class
mix over the four classes with n ≥ 30 at all three sites (Normal, Stage 1, Stage 2, Stage 3) —
the spread does not collapse. It widens:

| Site | Raw gap | Normal-only "control" | **Standardised gap** | 95% CI (patient-clustered) |
|---|---:|---:|---:|---|
| ostrava | +0.1534 | 0.0272 | **+0.1699** | [+0.135, +0.202] |
| ropvl | +0.0493 | 0.0218 | **+0.0216** | [+0.009, +0.034] |
| shenzhen | −0.0110 | 0.0117 | **−0.0080** | [−0.021, +0.007] |

**Ostrava − Shenzhen after adjustment: +0.178, 95% CI [+0.140, +0.215], P(>0) = 1.000**
(2,000 replicates resampling infants; 247 Ostrava replicates dropped as degenerate when a
rare class was absent from a draw, so the interval is approximate).

## Corrected statement

> A residual site effect on calibration survives case-mix adjustment and is **≈0.18 between
> the best- and worst-calibrated site**, roughly six times the +0.027 the committed artifact
> calls the defensible residual. The model's over-confidence is concentrated in the diseased
> classes at the site that supplies 64% of the corpus.

## Caveats that travel with the correction

- Ostrava's diseased classes are dominated by very few infants (148 IDs, 29 images each;
  AP-ROP is essentially one infant). The standardised gap is therefore driven by a small
  number of clusters.
- The clustering key is `patient_id`, which
  [is not a person identifier for two of the four sources](../03_DATA_LEAKAGE/DUPLICATE_AND_PATIENT_ID_AUDIT.md);
  for Ostrava it is, so this particular interval is the better-behaved case.
- The out-of-fold pool averages five checkpoints' behaviour. It does not describe the served
  fold-0 checkpoint, which saw 4/5 of this data in training.

**Classification: analysis correction. No retraining implied. The cheap intervention is
temperature scaling fitted per site on out-of-fold CORN logits, evaluated on the locked half.**
