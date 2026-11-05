# Preprocessing — what happens to a photograph before the model sees it

Diagram: [`preprocessing_pipeline.png`](preprocessing_pipeline.png) ·
Worked examples: [`before_after_examples/preprocessing_examples.png`](before_after_examples/preprocessing_examples.png)

| # | step | parameters | why it exists |
|---|---|---|---|
| 1 | `circle_crop` | tolerance 7/255 on greyscale | removes the black surround so the retinal disc fills the frame |
| 2 | `letterbox_square` | pad with black to a square | **the load-bearing anti-shortcut step.** `Resize((384,384))` squashes a non-square image by a factor that depends on the camera — 1.16–1.21 across the three training sites vs 1.333 at the held-out hospital. Without letterboxing, "how distorted the vessels look" becomes a hospital cue. This project has already been burned once: a classifier on image **dimensions alone** scored AUC 0.911. |
| 3 | `clahe` | clipLimit 2.0, 8×8 grid, on the L channel of LAB | local contrast equalisation; makes the peripheral ridge visible across very different exposures |
| 4 | `Resize((384, 384))` | bilinear | a single isotropic resize, applied to an already-square image |
| 5 | `ToTensor` + `Normalize` | ImageNet mean/std | matches the pretrained backbone's input statistics |

`ben_graham` (local-average colour subtraction) is **off** for ROP: it is a diabetic-retinopathy tool for microaneurysms, and the ROP signal is vascular and peripheral.

## Training-only augmentation

Applied inside the DataLoader, never at inference:

| group | ops |
|---|---|
| geometric (always) | RandomHorizontalFlip · RandomVerticalFlip · RandomRotation(15) |
| photometric (always) | ColorJitter(brightness 0.3, contrast 0.3, saturation 0.2) |
| **site randomisation** (`site_randomization: true`) | ColorJitter(hue 0.05, saturation 0.3) · RandomGrayscale(p=0.10) · ResolutionJitter(p=0.5, min_frac=0.33) · **RandomResizedCrop(384, scale 0.6–1.0, ratio 0.95–1.05)** replaces the fixed resize |

The site-randomisation block is targeted, not generic. The default jitter is luminance-only and leaves the per-channel R/B ratio intact — and that ratio is the strongest surviving hospital signature in this corpus (post-CLAHE mean blue 53.6 / 73.3 / 74.5 across training sites vs **87.6** at the held-out hospital). `ResolutionJitter` attacks the sharpness residue left by four different native resolutions.

`strong_aug` (affine / perspective / blur / random-erasing) is **not** enabled for this model.

## Could any of this introduce bias or leakage?

| step | assessment |
|---|---|
| circle_crop | **can worsen a site cue on its own** — it crops ROP-VL and Shenzhen to ≈1.21 aspect while leaving Ostrava at 1.00 and Multi-View at 1.333. This is precisely why step 2 exists; the two must stay paired. |
| letterbox | introduces **synthetic black pixels** that are not part of any retina. Harmless for classification, but it corrupts explainability: **28.7% of the model's Grad-CAM attention and 19.3% of its CAM peaks land on this invented padding** (see `../14_EXPLAINABILITY/`). |
| CLAHE | stateless and per-image, so no cross-split information flows; it is also the pipeline's compute bottleneck (~27 ms of the 112 ms per image). |
| normalisation | fixed ImageNet constants, **not** fitted on this dataset — no leakage channel. |
| RandomResizedCrop | train-only, so training and evaluation see different field-of-view scales. Standard practice; it is the intended regularisation. |
