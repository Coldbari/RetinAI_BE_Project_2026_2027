# The screening endpoint on the locked test — image, eye and infant

*Verification pass 2. This analysis did not exist in any prior artifact of this project.
Figure: [`screening_performance.png`](screening_performance.png). Tables:
[`screening_image_eye_infant.csv`](../20_AUDIT/verification_pass_2/07_TEST_RESULTS/screening_image_eye_infant.csv),
[`screening_confidence_intervals.csv`](../20_AUDIT/verification_pass_2/07_TEST_RESULTS/screening_confidence_intervals.csv).*

## What was evaluated

| | |
|---|---|
| Model | served checkpoint, `results/rop_staging/weights.pth`, SHA-256 `11b4396086…` = fold-0 checkpoint |
| Score | `P(any ROP) = 1 − prob6[:,0]`, one forward pass (the app does **not** use TTA for staging) |
| Threshold | **0.0155**, read from `results/rop/screening_rebase.json` exactly as `webapp/inference.py` reads it |
| Data | locked half of Multi-View: 661 images / 230 eyes / 116 infants, patient-disjoint from CV pool and from the dev half |
| Aggregation | *any frame flags the unit* — the rule a screening service uses, and the rule the project itself used for the retired model in `results/rop_statistics.md` §S4 |
| Uncertainty | patient-clustered bootstrap, 2,000 replicates, percentile CI |

## Result

| Unit | n | positives | negatives | Sensitivity [95% CI] | Specificity [95% CI] | PPV | NPV |
|---|---:|---:|---:|---|---|---:|---:|
| image | 661 | 429 | 232 | **0.911** [0.879, 0.942] | **0.323** [0.252, 0.392] | 0.714 | 0.664 |
| eye | 230 | 153 | 77 | **0.994** [0.979, 1.000] | **0.052** [0.011, 0.110] | 0.676 | 0.800 |
| infant | 116 | 77 | 39 | **1.000** [1.000, 1.000] | **0.000** [0.000, 0.000] | 0.664 | — |

**Every one of the 39 healthy infants in the locked test is flagged.** 73 of 77 healthy eyes
are flagged. The same holds on the dev half (infant specificity 0.000, 0/39) and for the
5-checkpoint mean score, so it is not a locked-half artifact and not a checkpoint artifact.

## Why this happens, and why it was not visible before

The image-level operating point is genuine: 0.911 / 0.323 out of sample, close to the
in-sample dev estimate (0.905 / 0.387) that the re-basing reported. Nothing about the
re-basing was wrong at the image level.

But an infant contributes ~6 frames across 2 eyes. At a per-image false-positive rate of
0.677, the probability that *no* frame of a healthy infant trips is 0.323^6 ≈ 0.001. The
per-image specificity does not survive the aggregation the clinic performs.

This is the identical failure mode `results/rop_statistics.md` §S4 recorded for the **retired**
ResNet50 (per-infant sensitivity 1.000 (7/7), specificity 0.000 (0/22)), and which figure 04
of the paper presents as fixed by re-basing. **The re-basing fixed the image-level
distribution shift. It did not fix the aggregation behaviour, which was never measured.**

## What this does and does not mean

- It **does not** mean the model has no signal: image-level AUC on the locked half is 0.835
  (served checkpoint), and the staging evidence is unaffected.
- It **does** mean that any claim of the form "specificity 0.39 at a hospital it never saw"
  must name the unit. At the unit a NICU screens — the infant — measured specificity is zero.
- It **does** mean a deployment that scores a whole examination and reports one verdict per
  infant, using this threshold and this any-frame rule, would refer every infant.

## Fix path (in order of cost)

1. **Change the decision rule, not the model.** Require *k* of *n* frames above threshold, or
   threshold the per-eye mean/max probability rather than the per-frame flag. Both are free to
   evaluate on these artifacts — no retraining, no new data.
2. **Re-derive the threshold at the unit of decision.** The current 0.0155 was selected on
   *image-level* out-of-fold scores plus the external dev half. A threshold chosen for an
   eye-level or infant-level endpoint would be a different number.
3. Only if 1 and 2 fail does the model itself become the suspect.

**Classification: fix the decision rule / threshold formulation. Retraining is not indicated
by this finding.**
