# Recommended Experiments, in priority order

*Each specifies objective, what changes, what is held constant, the metric, the information
gained, and a decision criterion agreed in advance.*

---

## E1 · Temperature-scale the CORN cascade  ·  **no GPU, highest value**

- **Objective** — remove the calibration defect (ECE 0.109 internal / 0.334 external) without touching accuracy.
- **Change** — fit a single scalar temperature *T* on the CORN conditional logits, per fold, using **out-of-fold predictions only**. Apply `σ(logits/T)` before the cumulative product.
- **Constant** — weights, backbone, decode rule, AP-ROP threshold 0.5, all splits.
- **Metric** — pooled ECE (15 bins), Brier, macro-F1, corrected QWK, count of >0.90-confidence errors; then the same on the locked set **without refitting**.
- **Information gained** — whether the over-confidence is a monotone squashing artifact (fixable) or a structural mis-ranking (not).
- **Decision** — ship the calibrated head if pooled ECE < 0.035 (the flat baseline's) **and** macro-F1 moves by < 0.01. If external ECE stays above 0.15, conclude that calibration does not transfer and remove percentages from the UI entirely.

## E2 · Repair the reference config and stamp the checkpoint  ·  **no GPU, 30 minutes**

- **Objective** — make the repository reproduce its own headline model, and make the served artifact self-identifying.
- **Change** — `weighted_sampler: false` in `configs/rop_staging_structured.yaml`; delete or implement `qwk_classes`, `unvalidated_classes`, `label_smoothing`; write `results/rop_staging/WEIGHTS.json` = {fold: 0, seed: 42, sampler: natural, epochs: 20, sha256: `11b4396086…`, git: …}; assert the hash at app boot.
- **Constant** — everything else.
- **Metric** — a single fold re-run lands within the last-5-epoch band of 0.5527 ± 0.017.
- **Decision** — if it lands outside that band, the incomplete `set_seed` (M-4) is the next suspect and must be fixed before any further seed claim.

## E3 · Promote the anatomical control  ·  **no GPU**

- **Objective** — replace an operator-dependent headline with an operator-free one.
- **Change** — score the external **dev** half by acquisition region, exactly as the locked half was scored here; add the six-rule aggregation table as a standing artifact.
- **Constant** — the locked half stays closed. No re-evaluation of locked.
- **Metric** — per-region QWK (corrected), accuracy, stage-1–3-called-normal rate.
- **Information gained** — whether temporal ≈ eye-level replicates outside the locked half.
- **Decision** — if dev reproduces the pattern (temporal ≫ nasal ≈ posterior), make the region result the paper's primary instrument and demote max-aggregation to a secondary with its rule named.

## E4 · Re-derive the served operating point on the served checkpoint  ·  **no GPU**

- **Objective** — give the product an operating point that actually describes the model it serves.
- **Change** — score the fold-0 checkpoint on **fold 0's validation split only** (the 20% it never saw) and on the external dev half; re-derive the any-ROP threshold under the existing stated rule.
- **Constant** — the selection rule, the locked half.
- **Metric** — sensitivity, specificity, PPV, NPV at the re-derived threshold, with patient-clustered CIs.
- **Decision** — if the re-derived threshold differs from 0.0155 by more than the width of its own CI, replace the served value and update the UI's quoted operating point.

## E5 · Ablations on folds 1–4, both seeds  ·  **~10 GPU-h**

- **Objective** — test whether the one surviving "head effect" survives off fold 0.
- **Change** — run *CORN-removed*, *branch-removed* and *w_site = 0* on folds 1–4 at seeds 42 and 1337.
- **Constant** — backbone, preprocessing, sampling, epochs, all other hyper-parameters.
- **Metric** — per-fold macro-F1 and corrected QWK, paired across folds, with the seed spread reported alongside.
- **Information gained** — whether the fold-0 CORN-removal effect (0.553 → 0.479) is real.
- **Decision** — if it collapses off fold 0, withdraw the component-contribution table and state total equivalence. That is a cleaner and more defensible paper than the current mixed message.

## E6 · Instrument the training loop, re-run one fold  ·  **~70 GPU-min**

- **Objective** — close the only genuinely unanswerable question in this package.
- **Change** — log per-epoch validation loss and the realised LR; have the Kaggle summary copy the `history` block it already builds.
- **Constant** — model, data, hyper-parameters, seed 42, fold 0.
- **Metric** — train loss vs validation loss vs epoch; validation macro-F1 vs epoch.
- **Decision** — if validation loss rises while training loss falls after epoch ~14, the fixed 20-epoch budget needs revisiting and E7 becomes worthwhile.

## E7 · A minimal hyper-parameter sweep  ·  **~15 GPU-h**

- **Objective** — establish that the untested hyper-parameters are at least reasonable.
- **Change** — LR ∈ {1e-4, 3e-4, 1e-3} × epochs ∈ {20, 35}, fold 0 and fold 3, seed 42.
- **Constant** — architecture, preprocessing, sampling.
- **Metric** — corrected QWK and macro-F1, each reported with its last-5-epoch band.
- **Decision** — adopt a new setting only if its improvement exceeds the last-5-epoch band **and** replicates on both folds. Otherwise record that the current settings are inside the noise and stop.

## E8 · Clinician adjudication of ~300 stratified labels  ·  **human, blocking**

- **Objective** — establish the human ceiling and separate model error from label noise.
- **Change** — an independent paediatric ophthalmologist re-grades ~300 images stratified by class and source, blind to the model.
- **Constant** — the images.
- **Metric** — κ vs the recovered labels; per-class agreement, especially Stage 1 vs Stage 2.
- **Information gained** — whether Stage 1 precision 0.376 is the model or the labels. Whether an internal QWK of 0.897 is near or far from the human ceiling.
- **Decision** — if human–human κ on Stage 1/2 is itself low, reframe the paper around label reliability and stop treating Stage 1 recall as a model deficiency.

## E9 · A second external hospital  ·  **data acquisition, blocking for generalisation**

- **Objective** — distinguish "does not generalise" from "that one site is different".
- **Metric** — the full external battery at both image and eye level, plus ECE.
- **Decision** — if per-stage recall at site 3 is again ≈0.28, the conclusion is general and should be stated as such. If it is materially better, the Multi-View drop is site-specific and the cause becomes the research question.

---

### What to do first

**E1 → E2 → E3 → E4** are all CPU-only and could be completed in a day. They fix the
calibration defect, make the repository reproducible, replace a contestable headline with a
robust one, and give the product an honest operating point. **E5–E7** need GPU quota.
**E8** is the long pole and every week of delay is risk.
