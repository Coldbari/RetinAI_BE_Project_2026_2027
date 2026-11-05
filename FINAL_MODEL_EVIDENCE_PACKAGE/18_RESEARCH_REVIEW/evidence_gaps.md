# Evidence Gaps — what would be needed for a complete evaluation

*Nothing in this list is filled with an assumption. Where evidence does not exist, this
package says so.*

## Gaps that a new training run would close

| # | missing | consequence | cost |
|---|---|---|---|
| 1 | **Validation loss per epoch** | no classical train-vs-val-loss overfitting diagnosis is possible for any run in this project | one config line + ~70 GPU-min for one fold |
| 2 | **Realised learning rate per step** | the LR curve here is reconstructed from hardcoded settings, not measured | one log line |
| 3 | **Per-epoch history committed for every arm** | it exists for the current model (found in `kaggle_cache/struct42/`) but the Kaggle summary discards it; other arms have none | change the kernel's summary writer |
| 4 | **A third seed** | two seeds replicate but cannot distinguish "equivalent" from "noisy" with confidence | ~7 GPU-h |
| 5 | **Ablations on folds 1–4** | the seven-arm table and the DANN dose–response rest on the corpus's most atypical fold | ~10 GPU-h for the three highest-value arms |
| 6 | **A hyper-parameter sweep of any kind** | LR, batch size, weight decay, dropout, epochs, image size and `arop_pos_weight` have **never** been swept for this model. None is claimed optimal, correctly — but none is known reasonable either | moderate |
| 7 | **A frozen-backbone / partial-fine-tune arm** | full fine-tuning of 20.5M parameters on 3,112 images is aggressive and untested against alternatives | ~2 GPU-h |

## Gaps that require no GPU (measurement only)

| # | missing | consequence |
|---|---|---|
| 8 | **Temperature-scaled calibration** | the calibration defect is unaddressed although the fix needs only existing out-of-fold logits |
| 9 | **The served checkpoint's own score distribution** | the served operating point is derived from a distribution this checkpoint does not produce |
| 10 | **Corrected QWK for the ablation arms and the five-backbone benchmark** | raw `.npz` files exist for only some arms; those tables remain on the legacy metric |
| 11 | **Region analysis on the dev half and internally** | the anatomical control is demonstrated only on the locked half; dev would replicate it for free, but the internal corpora have no view metadata at all |
| 12 | **Per-class ROC/PR internally with clustered CIs** | only pooled any-ROP curves are given |

## Gaps that require data or people, not compute

| # | missing | consequence | blocking for |
|---|---|---|---|
| 13 | **Independent paediatric-ophthalmologist adjudication of ~300 stratified labels** | the human ceiling is unknown, so it is impossible to say whether Stage 1 precision 0.376 is a model failure or a label-noise floor. No κ-vs-human exists | Tier-2 venues; any clinical claim |
| 14 | **A second external hospital** | one site cannot distinguish "this model does not generalise" from "this particular site is different" | any generalisation claim |
| 15 | **Published inter-grader κ for ICROP staging** | without it there is no benchmark for the 0.897 internal QWK | paper framing |
| 16 | **Device metadata at the external site** | its device field is `unknown` for every image, so device-stratified external analysis is impossible | device-shift claims |
| 17 | **Prospective data** | everything here is retrospective and stored | any clinical pilot |
| 18 | **Subgroup data (GA, birth weight, sex) at the external site** | no subgroup fairness analysis can be run externally | equity claims |
| 19 | **More stage 4/5 and AP-ROP images from any site other than ROP-VL** | the class–source confound cannot be broken with the current corpus | per-class claims for those two classes |

## Gaps that are structural and may never close with this corpus

| # | gap |
|---|---|
| 20 | **Cross-cohort patient identity.** No key exists to determine whether an infant appears in both Shenzhen and ROP-VL under different IDs. The 256-bit dHash sweep is the practical proxy and is clean (closest pair at distance 15 vs threshold 4). |
| 21 | **Internal view metadata.** Ostrava and Shenzhen record no acquisition view, so the anatomical decomposition that explains the external results cannot be applied to the internal ones. |
| 22 | **Ground truth independent of the corpora's own graders.** Every label in every split comes from the dataset it shipped with. |
