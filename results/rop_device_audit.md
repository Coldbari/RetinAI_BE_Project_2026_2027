# ROP device-confound audit

```
==============================================================================
T8 — ROP DEVICE-CONFOUND AUDIT
==============================================================================
parsed 6004/6004 filenames

1) ROP PREVALENCE BY DEVICE (images, and patients)
device     images  ROP+ img    prev  patients  ROP+ pat    prev
------------------------------------------------------------------------------
D1           2578      1925   74.7%        81        39   48.1%
D2           2452       741   30.2%        61        13   21.3%
D3            974       358   36.8%        46         8   17.4%

patient-level prevalence spread across devices: 30.8%
  -> STRONG confound: device carries substantial information about the label.

2) DEVICE DISTRIBUTION ACROSS SPLITS (images)
dev        1      2      3
split                     
test   0.629  0.321  0.050
train  0.363  0.402  0.235
val    0.363  0.573  0.064

If a device is concentrated in one split, test performance partly measures
that device rather than generalisation.

3) WITHIN-DEVICE AUC ON THE TEST SPLIT (deployed ResNet50)
  overall test AUC 0.9271
device     images   ROP+      AUC   note
------------------------------------------------------------------------------
D1            945    824   0.8808   -0.0462 vs overall
D2            482      0      n/a   single-class (all ROP-)
D3             75      0      n/a   single-class (all ROP-)

  devices in test: 3   measurable (both classes present): 1   ROP+ present on: ['D1']

  -> CRITICAL STRUCTURAL FINDING: every ROP-positive test image comes from a
     SINGLE device. Discrimination can only be measured inside that device;
     for the others there is nothing to discriminate. The overall AUC is
     therefore inflated by an easy device-vs-device separation that has
     nothing to do with disease.

  overall AUC (all devices pooled) 0.9271
  within-device AUC               0.8808   (-0.0462)

  -> The defensible device-controlled ROP AUC is 0.8808, not 0.9271.
```
