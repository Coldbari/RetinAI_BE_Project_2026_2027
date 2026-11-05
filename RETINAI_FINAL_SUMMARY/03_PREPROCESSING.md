# 3 · Preprocessing — What We Do to Every Photograph Before the Model Sees It

**Graphs:** [`graphs/05_preprocessing_pipeline.png`](graphs/05_preprocessing_pipeline.png) ·
`graphs/06_preprocessing_examples.png` (infant photographs, not in this public copy)

---

## The five steps, in order

Every photograph — during training, during testing, and in the live app — goes through exactly
the same five steps:

| # | Step | Settings | Why it is there |
|---|---|---|---|
| **1** | **`circle_crop`** | keep pixels brighter than 7/255 | The retina is a bright circle on a black background. This crops away the black border so the retina fills the frame. |
| **2** | **`letterbox_square`** | pad with black bars to a square | **The most important step.** Simply resizing to 384×384 would *stretch* each photo by a different amount, because each hospital's camera has a different shape (aspect ratio 1.16–1.21 at the three training hospitals vs **1.333** at the test hospital). The model could then guess the hospital from how squashed the vessels look. Padding to a square instead means every hospital gets the **same** geometric treatment. |
| **3** | **`CLAHE`** | clip limit 2.0, 8×8 tiles, on the brightness channel of LAB | Contrast-Limited Adaptive Histogram Equalisation. Boosts *local* contrast so the faint ridge at the edge of the retina is visible even in badly-exposed photographs. |
| **4** | **`Resize`** | to **384 × 384** pixels, bilinear | One single resize, applied to an already-square image, so nothing is distorted. |
| **5** | **Normalise** | ImageNet mean (0.485, 0.456, 0.406), std (0.229, 0.224, 0.225) | Puts the pixel values on the scale the pretrained network expects. |

**Deliberately NOT used:** *Ben-Graham* local-colour subtraction. It is a diabetic-retinopathy
technique for spotting microaneurysms; the ROP signal is vascular and peripheral, so it was
switched off after testing.

---

## Why step 2 exists — a mistake we had already made once

Earlier in this project, a classifier trained on **nothing but the image width and height** —
no pixels at all — scored **AUC 0.911**. The cameras differ between hospitals, the hospitals
have different disease mixes, so image *shape* alone predicted the label.

Note that step 1 on its own makes this **worse**: cropping brings ROP-VL and Shenzhen to an
aspect ratio of ~1.21 while leaving Ostrava at 1.00 and the test hospital at 1.333. Steps 1 and
2 must always be used together.

---

## Extra steps used ONLY during training (never at test time)

These are *augmentations* — random distortions that make the model robust instead of brittle.

| Group | What it does |
|---|---|
| **Geometric** | random horizontal flip · random vertical flip · random rotation ±15° |
| **Brightness / colour** | colour jitter (brightness 0.3, contrast 0.3, saturation 0.2) |
| **Hospital randomisation** *(targeted, not generic)* | hue/saturation jitter (hue ±0.05) · random grayscale 10% of the time · **resolution jitter** (shrink to a random 33–100% and blow back up) · **RandomResizedCrop** (random 60–100% zoom) |

The hospital-randomisation block was added for a measured reason. Ordinary brightness jitter
only changes overall lightness and leaves the **red/blue colour balance** untouched — and that
balance is the single strongest remaining hospital fingerprint in this data (after CLAHE, mean
blue is **53.6 / 73.3 / 74.5** at the three training hospitals versus **87.6** at the test
hospital). Resolution jitter attacks the other fingerprint: the four hospitals use cameras at
512², ~1450², 640×480/1440×1080 and 1600×1200, and even after a common resize the *sharpness*
gives them away.

---

## How many images before and after preprocessing

**Preprocessing does not remove any images.** It transforms every photograph and keeps it. The
counts change only because of the *inclusion rules and frame caps* described in
[`02_DATASET_AND_CLASSES.md`](02_DATASET_AND_CLASSES.md):

| Stage | Images |
|---|---|
| Acquired from 11 datasets | 13,037 |
| Have an ICROP stage label | 10,427 |
| Eligible for the 6-class task | 8,024 |
| → training pool / external hospital | 6,700 / 1,324 |
| **Fed through preprocessing during training** (after per-baby frame caps) | **3,112** |
| Fed through preprocessing for validation | 6,700 (every eligible image, uncapped) |
| Fed through preprocessing for the final test | 661 |

Every one of those images comes out of preprocessing as a **384 × 384 × 3** tensor.
**0 images were lost to corruption or failed reads** — all paths were verified present.

---

## Is preprocessing the same during training and in the live app?

**Yes — and this was verified, not assumed.**

The web app calls the *same function*, `build_transforms(config, "val")`, on the *same*
configuration file used for training. To prove there is no drift, the live model's weights were
re-run over all 663 test-hospital photographs and compared against the predictions stored months
earlier by a different script on different hardware:

- prediction agreement: **663 / 663 = 100%**
- largest difference in any probability: **0.000003** (floating-point rounding)

If preprocessing differed anywhere along that chain, that comparison could not close.

---

## One honest caveat about preprocessing

The black bars added in step 2 are **invented pixels** — they are not part of any retina. They
are harmless for accuracy (they are identical for every hospital, so they cannot carry any
information), **but they corrupt the heat-map explanation**: about **29%** of the model's
Grad-CAM attention and **19%** of its peak activations land on those invented bars. On healthy
images it is worse — 44% of attention sits on padding. This is why the heat map cannot be used
to judge an individual prediction. See [`05_RESULTS_AND_ACCURACY.md`](05_RESULTS_AND_ACCURACY.md) §7.
