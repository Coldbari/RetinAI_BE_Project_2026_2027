# Training Behaviour Analysis

*Source: the CURRENT MODEL's own Kaggle kernel log — 120 epoch records across 6 runs
(5 CV folds + the leave-site fold), seed 42, 20 epochs each.
Data: [`training_history.csv`](training_history.csv) ·
Figures: [`training_curves/`](training_curves/)*

## Cross-verification first

The final validation macro-F1 in the training log matches the value recomputed from the
committed prediction `.npz` files **exactly**, for all five folds:

| fold | from the training log | recomputed from raw predictions |
|---|---|---|
| 0 | 0.5527 | 0.5527 |
| 1 | 0.7573 | 0.7573 |
| 2 | 0.6712 | 0.6712 |
| 3 | 0.7184 | 0.7184 |
| 4 | 0.7616 | 0.7616 |

Two independent artifacts, produced months apart, agree. The training history and the
prediction files describe the same run.

## Is the model learning?

**Yes, unambiguously.** Total training loss falls from 1.83 to ≈0.45 across every run, smoothly and without divergence. The component losses behave as designed:

- **CORN stage loss** 0.66 → ≈0.15: the ordinal head is fitting.
- **AP-ROP branch loss** 0.65 → ≈0.08: the binary branch converges fastest.
- **Site-adversary loss** stays near ≈0.99 (≈ ln 3 = 1.099 is chance for 3 sites): the adversary is being actively fought by the reversed gradient, which is the intended behaviour — a *falling* site loss would mean the adversary was winning.

Final validation macro-F1 (0.55–0.76) sits far above the metadata-only floor of 0.187.

## Is learning stable?

**The training signal is. The validation signal is not.**

| run | mean \|Δ\| in val macro-F1 between consecutive epochs | max \|Δ\| |
|---|---|---|
| fold 0 | 0.058 | 0.258 |
| fold 1 | 0.051 | 0.201 |
| fold 2 | 0.065 | 0.255 |
| fold 3 | 0.065 | 0.308 |
| fold 4 | 0.050 | 0.276 |
| fold_site | 0.040 | 0.306 |

Validation macro-F1 can move **0.3** between two consecutive epochs while training loss changes by a few hundredths. Causes, in order of likely contribution: small rare-class support in each val fold (16–19 stage_4/5 images), a decision boundary that moves with the CORN cascade's thresholds, and an LR that is still high for the first ~14 epochs. Metrics settle as OneCycle anneals.

**Practical consequence:** any comparison between models differing by less than ≈0.02 macro-F1 on this corpus is uninterpretable without multiple seeds. The structured-vs-flat difference is 0.0055.

## Overfitting assessment

**Cannot be assessed properly — validation loss was never recorded.**

This is stated rather than substituted for. What *can* be said from the available signals:

- Training loss is still falling at epoch 20 in every run, so the model has not saturated the training set.
- Validation macro-F1 trends **upward** to the end of training in 4 of 6 runs (best epoch 14, 15, 19, 20, 15, 15) — it is not degrading.
- Internal out-of-fold accuracy is 0.786 against a training loss of ≈0.45, which is not the signature of severe memorisation.

A classic train-loss-vs-val-loss divergence diagnosis is **not available in the provided evidence**. Obtaining it requires a new run.

> **FUTURE EXPERIMENT — NOT PART OF CURRENT RESULTS:** add per-epoch validation loss and the realised learning rate to the training loop and re-run one fold. This would cost ~70 GPU-minutes and would close the only genuinely unanswerable question in this section.

## Learning-rate curve

[`learning_rate_curves/learning_rate_schedule.png`](learning_rate_curves/learning_rate_schedule.png) is **reconstructed** from the trainer's hardcoded `OneCycleLR(max_lr=3e-4, pct_start=0.1, anneal_strategy="cos")` over 154 steps/epoch. **It was not logged during training** and is labelled as a reconstruction on the figure itself. The DANN λ ramp in [`training_curves/dann_lambda_ramp.png`](training_curves/dann_lambda_ramp.png), by contrast, **was** logged and is measured.

## Which epoch should be used?

The project's rule — keep the **last** epoch, never the best — is the methodologically correct one: selecting the argmax over 20 noisy validation evaluations would bias every reported number upward by the amount in the table in [`training_configuration.md`](training_configuration.md) (up to 0.12 macro-F1 on fold_site).

The rule is correctly applied: the served checkpoint's predictions match fold 0's *final* metrics, not its best.

**What is missing is the error bar that comes with it.** Reporting "macro-F1 0.5527" for fold 0 without noting that the last five epochs ranged over ±0.017 presents a point estimate as if it were stable. Every headline in this package that derives from a single checkpoint should be read with that band attached.
