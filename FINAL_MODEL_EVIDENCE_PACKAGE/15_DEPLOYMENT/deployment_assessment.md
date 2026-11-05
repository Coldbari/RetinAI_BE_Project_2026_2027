# Deployment Assessment

## Can it run? Yes, easily.

| | |
|---|---|
| checkpoint | 82.96 MB (`results/rop_staging/weights.pth`) |
| parameters | 20,512,600 fp32 ≈ 82 MB resident |
| **CPU latency, end-to-end** | **112 ms/image** (27 ms preprocessing + 85 ms forward) — Apple Silicon, single image, no batching |
| measured over 663 real images | 166 ms/image including disk I/O and JPEG decode |
| throughput | ~6–9 images/second single-threaded CPU; a full 6-view examination in ~1 second |
| GPU | not required |
| dependencies | torch, torchvision, opencv, PIL, numpy, Flask, ReportLab |

Tables: [`inference_latency.csv`](inference_latency.csv), [`memory_usage.csv`](memory_usage.csv), [`model_size.txt`](model_size.txt)

Computationally this is comfortably deployable on a laptop or a modest clinic workstation, which was the design target. **Compute is not the constraint.**

## Should it be deployed? No.

| blocker | evidence | severity |
|---|---|---|
| **Staging does not transfer to a new hospital** | per-photograph recall 0.276 / 0.305 / 0.292 for stages 1/2/3 on the locked external set | **blocking** |
| **The probability output is uncalibrated, and worst externally** | ECE 0.109 internal, **0.334 external**; confidence-when-right minus confidence-when-wrong falls to 0.041 | **blocking** |
| **The most dangerous error is the most confident one** | "disease called Normal" carries mean confidence 0.934; 141 of 186 above 0.90. No confidence threshold can filter these out. | **blocking** |
| **The explanation shown to users is unreliable** | 30.7% of Grad-CAM peaks outside the retina; attention lift 1.04 (chance) | high |
| **The app scores photographs, but the evidence supports eyes** | per-image external QWK 0.47 vs per-eye 0.84. The deployed unit is the worst-performing one measured. | high |
| **Served operating point is not estimated for the served checkpoint** | the 0.0155 threshold was chosen on pooled out-of-fold scores; the served model is the fold-0 checkpoint that saw 4/5 of that data. Its advertised internal sensitivity/specificity (0.958/0.781) do not describe it. | high |
| **Rare and severe classes rest on tiny samples** | 89 stage 4/5 training images all from one source; 4 stage 4/5 and 12 AP-ROP images externally | high |
| **No clinician has re-read the labels** | all labels are the corpora's own; no inter-grader κ exists for this task in this project | high |
| **One external hospital, retrospective** | — | high |
| **Licence** | the external test hospital's dataset is **CC BY-NC** | commercial blocker |

## What the product already does right

These are real safety engineering and should be preserved:

- **Context routing.** The model runs only on a declared *preterm infant* context. Measured reason: on adult fundus photographs it flagged 41 of 45 eyes as ROP-positive. The operator declares the context, never the image.
- **Honest calibration badge.** The binary screening card ships `"calibration": {"status": "unverified", "applied": false}` with the text *"Read it as a ranking, not a probability of disease."* That is exactly right — and it should be extended to the six staging percentages, which come from the *worse*-calibrated head and currently carry no such qualifier.
- **The threshold is read from a committed artifact**, never hardcoded, and the artifact records its own selection rule and caveats.
- **The previous screening head was retired** after an external audit found it flagged 663 of 663 images. Acting on one's own negative audit is the behaviour that matters most here.
- **Graceful degradation.** Missing weights produce "model not loaded" rather than a silent fallback.

## What would have to be true before any clinical pilot

1. Eye-level (or examination-level) operation, not per-photograph.
2. A calibrated probability, with the calibration measured **at the deployment site**.
3. A site-specific threshold re-derived on local data — the retired head's failure proves thresholds do not transfer here.
4. Prospective evaluation with clinician adjudication.
5. An explanation surface that has passed a localisation audit, or none at all.
6. Stage 4/5 and AP-ROP validated on real sample sizes, or explicitly excluded from the product's claims.

Full list: [`deployment_limitations.md`](deployment_limitations.md).
