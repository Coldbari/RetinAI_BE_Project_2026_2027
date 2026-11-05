# ROP — patient-level statistical analysis

```
==============================================================================
S4 — PATIENT-LEVEL ROP METRICS (ResNet50, deployed checkpoint)
==============================================================================
test split: 1502 images from 29 infants (7 with ROP, 22 without)
images per infant: median 30, max 470

AUC (image-level bootstrap)   0.9271  95% CI [0.913, 0.940]  width 0.027
AUC (patient-level bootstrap) 0.9271  95% CI [0.865, 0.967]  width 0.102   <- honest (29 clusters)
the image-level interval is 3.8x too narrow

at the deployed threshold 0.1933:
  sensitivity 0.982  95% CI [0.970, 0.990]   (809/824 images)
  specificity 0.456  95% CI [0.418, 0.494]   (309/678 images)

PER-INFANT (an infant counts as ROP-detected if ANY of their images is flagged):
  sensitivity 1.000  95% CI [0.590, 1.000]   (7/7 infants)
  specificity 0.000  95% CI [0.000, 0.154]   (0/22 infants)
  -> with only 7 ROP-positive infants, the sensitivity interval spans 0.41. This is the binding constraint on the ROP claim.

S4b — HOW CONCENTRATED IS THE TEST SET? (upgrade)
  largest infant: 470/1502 images = 31.3% of the test set
  top 3 infants : 45.4% of the test set
  image-level AUC    0.9271   (each IMAGE counts once - biased to the most-photographed infants)
  patient-weighted   0.8989   (each INFANT counts once)
  leave-one-infant-out AUC ranges 0.9193 - 0.9457 (swing 0.0264 from removing ONE infant)

==============================================================================
S5 — OPERATING THRESHOLD PROVENANCE
==============================================================================
threshold chosen on VAL  for sens>=0.9: 0.182 (val sens 0.901 / spec 0.363)
threshold chosen on TEST for sens>=0.9: 0.371 (test sens 0.900 / spec 0.827)
deployed threshold in configs/rop.yaml: 0.1933

HONEST estimate — apply the VAL-selected threshold once to TEST:
  thr=0.182 -> test sensitivity 0.985, specificity 0.414
OPTIMISTIC (currently reported) — threshold picked on TEST itself:
  thr=0.371 -> test sensitivity 0.900, specificity 0.827
  specificity is overstated by +0.413 by selecting on the test set.

==============================================================================
S6 — ARCHITECTURE COMPARISON (paired, same test images)
==============================================================================
ResNet50 vs weights_v2s_reg (efficientnet_v2_s):
  DeLong : AUC 0.9271 vs 0.9195  z=1.527  p=0.1268  NOT significant
  McNemar: b=94 c=99  p=0.7734 (chi2-cc)  NOT significant
ResNet50 vs weights_overfit_v2s (efficientnet_v2_s):
  DeLong : AUC 0.9271 vs 0.9146  z=2.641  p=0.0083  SIGNIFICANT
  McNemar: b=88 c=202  p=0.0000 (chi2-cc)  SIGNIFICANT

PATIENT-LEVEL AUC DIFFERENCE (cluster bootstrap over infants — upgrade):
DeLong assumes independent samples; 1,502 images from 29 infants violates that, so
its p-values are anti-conservative. This resamples INFANTS instead.
  ResNet50 - weights_v2s_reg: dAUC +0.0126 95% CI [-0.0177, +0.0567]  NOT significant (CI spans 0)
  ResNet50 - weights_overfit_v2s: dAUC +0.0155 95% CI [-0.0192, +0.0658]  NOT significant (CI spans 0)

==============================================================================
S7 — HEADLINE METRICS WITH INTERVALS
==============================================================================
metric                       point      95% CI (patient-level)
------------------------------------------------------------------------------
AUC                         0.9271              [0.865, 0.967]
accuracy                    0.7443              [0.516, 0.866]
macro F1                    0.7125              [0.486, 0.797]
```
