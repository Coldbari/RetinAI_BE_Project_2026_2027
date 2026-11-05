# Training Configuration — every parameter, its value, and whether the evidence supports it

Config: [`../01_MODEL/model_configuration.yaml`](../01_MODEL/model_configuration.yaml) ·
Log: `kaggle_cache/struct42/rop-structured-effnetv2s-s42.log` ·
History: [`training_history.csv`](training_history.csv)

> ⚠️ **The committed config does not reproduce this model.** It sets `weighted_sampler: true`. Every headline run overrode it to `false` from the Kaggle kernel. `weighted_sampler: true` + CORN is the **documented v1 failure** (QWK 0.121, stage_4_5 precision 0.043). Running the repository's own documented command reproduces the broken arm.

| parameter | what it does | value used | reasonable? | evidence | further experiment? |
|---|---|---|---|---|---|
| **optimizer** | parameter update rule | AdamW | yes | convention for transformer-era CNNs with decoupled weight decay; not compared against SGD here | low value |
| **learning rate** | step size | 3e-4 (max, OneCycle) | plausible, **untested** | no LR sweep exists for this model | **yes — never swept** |
| **scheduler** | LR over time | OneCycleLR, pct_start 0.1, cosine | yes | hardcoded in the trainer, not configurable; warmup ≈ 2 epochs at 20 epochs. Val metric stabilises exactly as the LR anneals, which is the expected signature | no |
| **batch size** | images per step | 16, `drop_last=True` | plausible, **untested** | chosen for Kaggle memory; 154 steps/epoch | low value |
| **epochs** | passes over the data | 20 | plausible, **untested** | best-epoch is 14–20 across the 6 runs, so 20 is not obviously too few or too many | **yes — jointly with LR** |
| **weight decay** | L2-style regularisation | 0.05 | plausible | AdamW convention; not swept | low value |
| **grad clipping** | caps gradient norm | 1.0 | yes | standard; multi-task loss with an adversary benefits from it | no |
| **AMP** | mixed precision | on (CUDA only) | yes | speed; no measured accuracy effect | no |
| **loss (stage)** | ordinal objective | CORN over 4 conditional logits, **AP-ROP samples masked out** | **yes, and principled** | ICROP-3 says the aggressive form has no stage; a fabricated stage would be a label error. Ablation: removing CORN is the single largest head effect measured (macro-F1 0.553 → 0.479 on fold 0) | **yes — that ablation is fold-0-only** |
| **loss (AP-ROP)** | binary branch | BCE-with-logits, `pos_weight = sqrt(n_neg/n_pos)` ≈ 4× | plausible, **untested** | raw inverse frequency would be ~16×; sqrt-damping is reasoned, not measured | **yes — compare 1.0 / sqrt / raw** |
| **AP-ROP threshold** | decode cut-off | 0.5, **fixed a priori** | **yes — methodologically correct** | tuning it on validation would be selection. Cost: `argmax(prob6)` disagrees with the decoded label in 1.19% of cases | measure the cost, don't tune |
| **loss (site)** | adversary | CE with gradient reversal, λ ramped 0→1.0, weighted by `w_site = 0.5` | see note | ⚠ **deviation**: `w_site` multiplies the *whole* site loss, so it damps the adversary's own learning too. Canonical DANN scales only the reversed gradient into the backbone. Consistent across arms, so ablations stay internally valid | **yes — separate the two scalars** |
| **sampling** | class balance | **natural prevalence** | **yes, and measured** | v1 used a weighted sampler and CORN collapsed (QWK 0.121). CORN's conditional thresholds calibrate to training prevalence; a cumulative cascade cannot tolerate prior shift, while argmax-softmax can | settled |
| **label smoothing** | softens targets | declared `0.0` — **but the key is never read** by `train_structured.py` | harmless (value is 0) | third silent no-op key in this file | remove the key |
| **dropout** | regularisation | 0.3 on the embedding | plausible, **untested** | — | low value |
| **early stopping** | val-based selection | **off** — last-epoch weights kept | **yes, and deliberate** | avoids taking a maximum over 20 noisy evaluations. Cost measured below | quantify, don't change |
| **checkpointing** | which weights are saved | every epoch overwrites, so the file holds the last epoch | yes | matches the "report the last epoch" rule | no |
| **seed** | reproducibility | 42 (and 1337 as a replicate) | partial | `set_seed` omits cuDNN determinism, `PYTHONHASHSEED`, and DataLoader generators — runs are approximately, not bit-, reproducible | **yes — and it matters, since seed noise is a headline finding** |
| **frozen layers** | transfer strategy | **none** — all 20.5M parameters trainable | reasonable | full fine-tuning from ImageNet on 3,112 images is aggressive but the loss curves show no divergence | compare against a frozen-stem arm |

## Config keys that are declared but never read

Violating the project's own house rule ("config keys must be READ by code"):

| key | declared in | read by |
|---|---|---|
| `eval.qwk_classes` | both staging configs | **nothing** — this caused the QWK defect corrected in this package |
| `eval.unvalidated_classes` | both staging configs | **nothing** — a human reporting convention only |
| `train.label_smoothing` | the structured config | **nothing** in `train_structured.py` (only `losses.py::build_loss`, which this trainer does not call) |

## The cost of the last-epoch rule, measured

| run | final val macro-F1 | best val macro-F1 | best − final | SD over last 5 epochs |
|---|---|---|---|---|
| fold 0 | 0.5527 | 0.5964 (ep 14) | 0.0437 | 0.0175 |
| fold 1 | 0.7573 | 0.7660 (ep 15) | 0.0087 | 0.0213 |
| fold 2 | 0.6712 | 0.6835 (ep 19) | 0.0123 | 0.0149 |
| fold 3 | 0.7184 | 0.7463 (ep 15) | 0.0279 | 0.0171 |
| fold 4 | 0.7616 | 0.7616 (ep 20) | 0.0000 | 0.0070 |
| fold_site | 0.5454 | 0.6652 (ep 15) | **0.1198** | 0.0558 |

**The end-of-schedule noise (0.007–0.056) is larger than the entire structured-vs-flat difference (0.0055).** That is the strongest available argument that the architecture comparison is a null result, and it is not currently used as one.
