# Input / Output Definition — CURRENT MODEL

## Input

| property | value |
|---|---|
| Modality | RGB digital fundus photograph of a premature infant's retina |
| Acquisition | wide-field contact paediatric retinal cameras (RetCam-class); 4 cohorts, 4 native resolutions (512², ~1450², 640×480 / 1440×1080, 1600×1200) |
| Accepted at serving | any JPEG/PNG ≤ 10 MB |
| Tensor fed to the network | `float32 [1, 3, 384, 384]`, ImageNet-normalised (mean .485/.456/.406, std .229/.224/.225) |
| Required patient context | **preterm infant.** The serving registry refuses to run this model on any other context — measured reason: on adult fundus photographs it flagged 41 of 45 eyes as ROP-positive. |

## Output

The model returns a dict of three raw tensors, which are then decoded:

| key | shape | meaning |
|---|---|---|
| `stage` | `[B, 4]` | CORN conditional logits. `σ(·)` cumulatively multiplied gives P(rank > k) for k = 0..3 |
| `arop` | `[B]` | logit for "is this the aggressive form?" |
| `site` | `[B, 3]` | hospital-of-origin logits. **Training-time machinery only — ignored at inference.** |
| `embedding` | `[B, 1280]` | penultimate features, used by the site probe |

### Decode to the reported label and probability vector

```
p_arop   = sigmoid(arop)
rank_p   = corn_rank_probs(stage)                    # [B,5], sums to 1, monotone by construction
rank     = corn_predict(stage)                       # int 0..4
prob6    = concat[ (1 - p_arop) * rank_p , p_arop ]  # [B,6], sums to 1
pred6    = 5 if p_arop > 0.5 else rank               # int 0..5
```

| index | class | ordinal? |
|---|---|---|
| 0 | Normal | yes (rank 0) |
| 1 | Stage 1 | yes |
| 2 | Stage 2 | yes |
| 3 | Stage 3 | yes |
| 4 | Stage 4/5 (merged) | yes (rank 4) |
| 5 | **AP-ROP** | **no — an aggressive *form*, deliberately off the stage scale** |

### Derived product outputs

| output | formula | status |
|---|---|---|
| binary screening verdict | `prob6[1:].sum() > 0.0155` | threshold from `results/rop/screening_rebase.json`; selected partly on the external dev half, so its external numbers are optimistic |
| displayed per-class percentages | `prob6 * 100` | **UNCALIBRATED — a ranking score, not a probability of disease** |
| Grad-CAM overlay | ∂ log Σ prob6[1:] / ∂ features | **unreliable — 31% of peaks fall outside the retina** |

## Two consistency caveats a reader should know

1. **`argmax(prob6)` disagrees with `pred6` in 1.19% of predictions** (2.4% on fold 0), because the label uses a hard 0.5 threshold on the AP-ROP branch while the probability vector composes multiplicatively. Accuracy/F1 and AUC therefore rest on slightly different decision rules. The flat baseline has 0.00% disagreement.
2. **No test-time augmentation is applied at serving.** The repository README describes 5-view TTA; that is stale for this model. Serving matches the evaluation artifacts (also single-pass), so the numbers in this package *do* describe the served behaviour.
