# 4 · How the Model Was Built and Trained — every parameter

**Graph:** [`graphs/07_training_behaviour.png`](graphs/07_training_behaviour.png)

---

## The model

An **EfficientNetV2-S** network, pretrained on ImageNet, with its original classifier removed
and **three custom heads** attached to its 1,280-number description of the image.

```
 384×384 photo  →  EfficientNetV2-S backbone  →  1,280 numbers  →  Dropout(0.3)
                                                                       │
                          ┌────────────────────────────┬───────────────┴──────────────┐
                          ▼                            ▼                              ▼
              1. ORDINAL STAGE HEAD          2. AP-ROP BRANCH            3. HOSPITAL ADVERSARY
                 Linear(1280 → 4)               Linear(1280 → 1)            1280 → 256 → 3
                 5,124 parameters               1,281 parameters            328,707 parameters
              "how far has it progressed?"   "is the aggressive       "which hospital is this?"
                                              form present?"           — TRAINING ONLY, switched
                                                                         off when the model is used
```

**Why three heads instead of one:**

1. **Ordinal stage head (CORN).** A normal classifier treats "Stage 2 called Normal" and
   "Stage 2 called Stage 3" as equally wrong. They are not. This head is built so the stages
   stay in order, and being one stage off costs less than being four stages off.
2. **Separate AP-ROP branch.** The international standard says AP-ROP is an aggressive *form*,
   not a rung on the 1-2-3 ladder. Forcing it to compete with the stages for probability is a
   modelling error, so it gets its own yes/no output. AP-ROP images are also **excluded** from
   the stage head's training signal, rather than being given a made-up stage.
3. **Hospital adversary.** During training the network is *penalised* for encoding which
   hospital a photo came from (gradient reversal / DANN). This is machinery, not a product — it
   is inert when the model is used.

| | |
|---|---|
| **Total parameters** | **20,512,600** — all trainable, none frozen |
| backbone | 20,177,488 (98.4%) |
| the three heads | 335,112 (1.6%) |
| **The actual architectural contribution** (stage head + AP-ROP branch) | **6,405 parameters = 0.031% of the model** |
| Checkpoint file size | 82.96 MB |

---

## Every training parameter

| Parameter | Value | Note |
|---|---|---|
| **Optimiser** | AdamW | |
| **Learning rate** | 3 × 10⁻⁴ (peak) | ⚠️ never swept — reasonable, not proven optimal |
| **Learning-rate schedule** | OneCycleLR, 10% warm-up, cosine decay | hardcoded in the trainer |
| **Weight decay** | 0.05 | |
| **Batch size** | 16 (last incomplete batch dropped) | 154 steps per epoch |
| **Epochs** | **20** | |
| **Mixed precision (AMP)** | on | |
| **Gradient clipping** | 1.0 | |
| **Dropout** | 0.3 | on the 1,280-number embedding |
| **Class sampling** | **natural** — *no* balanced sampler | see the warning below |
| **Stage loss** | CORN ordinal loss, **AP-ROP images masked out** | |
| **AP-ROP loss** | binary cross-entropy, positive weight = √(neg/pos) ≈ 4× | |
| **AP-ROP decision threshold** | **0.5, fixed in advance** | deliberately never tuned on validation data — that would be cheating |
| **Hospital-adversary loss** | cross-entropy through a reversed gradient, strength ramped 0 → 1.0, weighted 0.5 | |
| **Early stopping** | **OFF** | |
| **Which weights are kept** | **the LAST epoch, never the best epoch** | see below |
| **Random seed** | **42** (replicated at 1337) | |
| **Frozen layers** | none | |
| **Hardware** | Kaggle GPU (P100 / T4), ~3.5 min per epoch, ~70 min per fold | |

### Two deliberate choices that are worth explaining

**Why natural sampling and not class balancing?**
The first version of this model *did* use a class-balanced sampler, and it collapsed — quadratic
kappa fell to **0.121** and Stage 4/5 precision to **0.043**. The reason is specific to the
ordinal head: it makes decisions by multiplying a chain of probabilities calibrated to the
disease rates it was *trained* on. A balanced sampler shows it a world where Stage 4/5 is ~9×
more common than reality, so on real data every high-stage decision fires far too often. An
ordinary softmax tolerates a balanced sampler; this cascade cannot.

> ⚠️ **Known repository issue:** the committed config file `configs/rop_staging_structured.yaml`
> still says `weighted_sampler: true`. Every real run overrode it to `false` from the training
> script. **Running the repository's documented command as-is reproduces the broken version.**
> This is a one-line fix and it is the top item on the to-do list.

**Why keep the last epoch instead of the best?**
Picking the best of 20 validation scores would flatter the result — you would be reporting the
maximum of 20 noisy measurements. Keeping the last epoch is honest. The cost was measured: the
best epoch beats the final epoch by 0.000 – 0.120 macro-F1 depending on the run.

---

## What the training looked like

**The model is genuinely learning:** total training error falls smoothly from **1.83 to about
0.45** across all five runs, with no divergence. The component losses behave as designed — the
stage loss and AP-ROP loss both converge, while the hospital-adversary loss stays pinned near
chance level, which is exactly what it should do when the reversed gradient is working.

**But the validation score is very unstable.** Between two consecutive epochs the validation
macro-F1 moves by **0.04–0.065 on average and up to 0.31** at worst, while training error barely
moves. Over the **last five epochs** of a run the score still varies by **0.007–0.056**.

> **Practical consequence:** on this dataset, any difference between two models smaller than
> about 0.02 macro-F1 is meaningless noise. This matters, because the difference between the
> fancy ordinal model and a plain softmax turned out to be **0.0055**.

**What is missing:** validation *loss* and the realised learning rate were never logged, for any
run. So the classic "training loss falling while validation loss rises" overfitting check
**cannot be performed from the available evidence**. It is not reported as if it were. Fixing
this costs one line of code and one 70-minute run.
