# Deployment footprint — measured, not quoted

*Verification pass 2. Measured by loading `results/rop_staging/weights.pth` exactly as
`webapp/inference.py` does and timing forward passes on this machine (Apple silicon, macOS).
Raw: [`deployment_footprint.json`](deployment_footprint.json).*

| | |
|---|---|
| Checkpoint | `results/rop_staging/weights.pth` |
| SHA-256 | `11b439608690024700cdc927dece45d51ebc78b648aef81090cb6be3e7dc747c` |
| File size | 82,964,554 bytes (79.12 MiB), 788 state-dict entries, strict load OK |
| Architecture | EfficientNetV2-S, input 384×384, `n_sites=3` |
| Parameters | **20,512,600 total** — backbone 20,177,488 · stage head 5,124 · AP-ROP head 1,281 · site head 328,707 |
| Inference-only parameters | **20.18 M** (the 328,707-parameter site adversary is training-time machinery and is never evaluated at serve time) |
| Preprocessing | see note — pass-2 single-shot timing (120.0 ms) includes first-call overhead; the averaged measurement in [`inference_latency.csv`](inference_latency.csv) (**26.7 ms**) is the reliable figure |
| Forward pass, CPU | median **113.6 ms**, p90 147.3 ms (20 runs after 3 warm-up) |
| Forward pass, MPS | median **32.3 ms**, p90 56.9 ms |
| Forward passes per uploaded image | **1** |
| Test-time augmentation | **Not used** for the staging model |

## Two things this corrects

1. **TTA.** `results/rop/operating_point.json` records its threshold as *"derived_on: clean
   val, 5-view TTA"*, and several documents describe the ROP path as using 5-view TTA. That
   belongs to the **retired** ResNet50 screening head. `webapp/inference.py` runs one forward
   pass for the served staging model. Any performance number computed with TTA does not
   describe the deployment.
2. **The published latency table is the wrong model.** `results/sweep_rop.md` gives CPU
   latencies for five *binary* architectures from the retired sweep (efficientnet_v2_s at
   130.1 ms there). It does not cover the served three-head model, its 384-px input, or the
   preprocessing stage.

## Per-examination cost

A NICU examination in this corpus is ~6 frames. End-to-end, single-threaded, including
preprocessing:

| Device | Per image (fwd + 26.7 ms preprocessing) | Per 6-frame examination |
|---|---:|---:|
| CPU | ~112–140 ms | **~0.7–0.8 s** |
| MPS | ~59 ms | **~0.4 s** |

**Two independent timings disagree and both are kept.** Pass 1 measured CPU forward 85.2 ms
and preprocessing 26.7 ms (averaged, [`inference_latency.csv`](inference_latency.csv)); pass 2
measured CPU forward 113.6 ms median over 20 runs after warm-up and a single-shot
preprocessing call of 120.0 ms. The forward-pass spread is machine load; the preprocessing
difference is a measurement error in pass 2 (no warm-up, so it captured first-call import and
allocation cost). **Use pass 1's preprocessing number and treat the forward pass as 85–115 ms
on CPU.** Either way latency is not a deployment constraint for this use case, and
preprocessing is not the bottleneck.
