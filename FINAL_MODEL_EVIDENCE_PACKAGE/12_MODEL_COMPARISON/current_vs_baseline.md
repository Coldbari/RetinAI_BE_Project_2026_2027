# Current Model vs Its Baseline — matched comparison

*Same corpus, same folds, same preprocessing, same backbone, same seeds, same metric
definitions, same uncertainty methodology. The only difference is the output head.*

| | CURRENT model | Baseline |
|---|---|---|
| head | CORN ordinal (4 conditional logits) + AP-ROP binary branch + DANN site adversary | plain 6-way softmax |
| backbone | EfficientNetV2-S, identical | identical |
| sampling | natural prevalence | natural prevalence (**matched** — an earlier comparison was confounded here) |
| trainer | `train_structured.py` | `train.py` |

Table: [`model_comparison.csv`](model_comparison.csv) · Figure: [`structured_vs_flat.png`](structured_vs_flat.png)

## Where they are indistinguishable

| dimension | current | baseline | test |
|---|---|---|---|
| macro-F1, CV, seed 42 | 0.6923 ± 0.0861 | 0.6977 ± 0.0986 | paired *t* **p = 0.589** |
| macro-F1, CV, seed 1337 | 0.7079 ± 0.0965 | 0.6855 ± 0.1084 | **p = 0.494** — and the **sign has flipped** |
| **QWK (corrected), CV, seed 42** | **0.8964 ± 0.0288** | **0.8962 ± 0.0285** | **p = 0.977** — a difference of 0.0002 |
| accuracy, CV | 0.7956 | 0.7973 | p = 0.880 |
| eye-level QWK, external locked | 0.8127 | 0.8082 | 0.005 apart |
| Brier score | 0.3226 | 0.3304 | marginal, favours current |

Context for these nulls: **seed-to-seed spread within a single arm is 0.018 (structured) to 0.043 (flat)**, and the **SD of a single run's last five epochs is 0.007–0.056**. Both are larger than any difference in the table.

## Where the BASELINE is clearly better

| dimension | current | baseline | margin |
|---|---|---|---|
| **Expected Calibration Error** | **0.1093** | **0.0346** | baseline **3.2× better at identical accuracy** |
| over-confidence gap | +0.109 | +0.022 | baseline |
| predictions above 0.90 confidence | 68.6% | 13.1% | baseline |
| high-confidence **wrong** predictions | **359** | **181** | baseline, ~2× fewer |
| **external locked QWK (image level), seed 42** | 0.4732 ± 0.0140 | **0.4924 ± 0.0209** | baseline |
| **external locked QWK (image level), seed 1337** | 0.4556 ± 0.0232 | **0.4856 ± 0.0218** | baseline — **the sign does NOT flip externally** |
| decode / argmax consistency | 1.19% disagreement | 0.00% | baseline |
| AP-ROP recall, pooled CV | 0.345 | 0.364 | baseline, marginal |

## Where the CURRENT model is better

| dimension | detail |
|---|---|
| **Taxonomy fidelity** | AP-ROP is emitted as an *independent* probability rather than competing with the stages for softmax mass; the stage output is a monotone ordinal rank by construction. This is the design's stated purpose and it is delivered. |
| Stage 2 recall, pooled CV | 0.631 vs 0.615 — marginal, inside noise |
| Error structure | both heads produce ordinally-structured errors; the current model's non-adjacent error count is marginally lower. The benefit is real but does not appear in any aggregate metric. |

## Computational cost

Essentially identical: same backbone, +6,405 parameters for the contributed heads and +328,707 for the site adversary (inert at inference). 82.96 MB checkpoint, 112 ms/image CPU end-to-end.

## Honest verdict

**Neither model is globally better.** Stated precisely:

- On **accuracy-family metrics** the two are indistinguishable, and the sign of the difference is not stable across random seeds. This is an equivalence result, and the corrected QWK (0.8964 vs 0.8962) makes it sharper than the uncorrected metric did.
- On **calibration**, the baseline is clearly and substantially better, at the same accuracy. This is a real cost of the CORN cascade's probability geometry and had not previously been measured.
- On the **external hospital at the pre-registered image-level unit**, the baseline is ahead at **both** seeds. Unlike cross-validation, the external sign does not flip. This is the only dimension where one model beats the other consistently, and it favours the baseline.
- On **taxonomy fidelity**, the current model is the only option — a softmax cannot represent "an aggressive form that is not a stage".

The scientifically defensible claim is therefore: *a taxonomy-faithful ordinal head performs equivalently to a flat softmax on this corpus while producing worse-calibrated probabilities; its justification is representational, not performance.* The parent project's paper states the equivalence; **the calibration cost and the consistent external direction are new here and should be added.**

## Why equivalence is not surprising

The architectural contribution is **6,405 parameters — 0.031%** of a 20.5M-parameter ImageNet-pretrained network. Finding that swapping a 6.4k-parameter head does not move macro-F1 on 3,112 training images is close to the prior expectation. Framing the result this way is more honest, and more useful to a reader, than presenting it as a surprise.
