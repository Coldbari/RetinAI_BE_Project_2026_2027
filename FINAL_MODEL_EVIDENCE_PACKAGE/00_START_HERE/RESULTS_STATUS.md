# What We Know / What We Don't Know

*Evidence ledger for the CURRENT MODEL only. A claim is placed by the strength of the
evidence in this package, not by how much we would like it to be true.*

---

## ✅ Proven by evidence

| claim | evidence |
|---|---|
| **No patient or image leakage between training and any test set.** | Recomputed from the split CSVs: 0 overlapping infants and 0 overlapping images across every train/val, corpus/dev, corpus/locked and dev/locked comparison; the 5 validation folds exactly partition 6,700 images. `03_DATA_LEAKAGE/` |
| **The model learns from pixels, not from metadata.** | macro-F1 0.692 against a metadata-only floor of 0.187 (shortcut-audit gate G4) and above the majority prior. `06_VALIDATION/` |
| **Training is stable and does not overfit in the loss sense.** | Training loss falls 1.83 → ~0.45 monotonically across all 6 runs, no divergence. `05_TRAINING/training_curves/` |
| **Ordered severity agreement is strong internally.** | Corrected QWK 0.8964 ± 0.0288 (fold SD); pooled 0.8969 [0.869 – 0.923] patient-clustered. `07_TEST_RESULTS/qwk_analysis/` |
| **Detecting *any* disease works internally.** | AUC 0.9695, AP 0.9662 at prevalence 0.427. `07_TEST_RESULTS/roc_pr_analysis/` |
| **Detecting healthy eyes transfers to a new hospital.** | Normal recall 0.996 on the locked set (n=232). `13_EXTERNAL_GENERALIZATION/` |
| **Per-stage performance does *not* transfer.** | Locked recall: Stage 1 0.276, Stage 2 0.305, Stage 3 0.292; accuracy 0.786 → 0.546. `13_EXTERNAL_GENERALIZATION/` |
| **The model's probabilities are not calibrated.** | ECE 0.109 internal, **0.334 external**; confidence−accuracy gap +0.109 / +0.334. `10_CONFIDENCE_CALIBRATION/` |
| **The most dangerous error is the most confident one.** | "disease stage called Normal": mean confidence 0.934, 141 of 186 above 0.90 — the highest of any error category. `09_ERROR_ANALYSIS/error_categories.csv` |
| **The structured head is worse calibrated than the plain baseline at equal accuracy.** | ECE 0.1093 vs 0.0346; 359 vs 181 confident errors; accuracy 0.786 vs 0.789. `12_MODEL_COMPARISON/` |
| **The image→eye QWK gap is anatomical.** | Temporal frames scored at **image** level with **no aggregation**: QWK 0.841. Nasal 0.352, posterior pole 0.229. The eye-level max rule gives 0.813. `11_AGGREGATION_AND_VIEW_ANALYSIS/` |
| **The eye-level headline is operator-dependent.** | Same predictions: max 0.813, majority 0.398, mean-prob 0.396, max-prob-per-class 0.351, one-random-view 0.482, image level 0.473. `11_AGGREGATION_AND_VIEW_ANALYSIS/aggregation_results.csv` |
| **Grad-CAM attention on the retina is at chance.** | 56.0% of attention on a region occupying 53.8% of the frame → lift 1.04; 30.7% of peaks outside the retina; 28.7% of attention on invented padding. `14_EXPLAINABILITY/` |
| **The served checkpoint is the fold-0 checkpoint of the seed-42 run.** | Re-ran the served weights over 663 dev images: 100% agreement, max prob difference 3e-06. `20_AUDIT/served_checkpoint_identification.json` |
| **The reported QWK did not match its own declared definition.** | `eval.qwk_classes` is read by no code; `compute_metrics` scored AP-ROP as ordinal rank 5. Correcting it moves fold-0 QWK 0.569 → 0.908. `07_TEST_RESULTS/qwk_analysis/` |
| **Preprocessing is identical in training, evaluation and serving.** | Same config, same `build_transforms(cfg,"val")` call path. `04_PREPROCESSING/preprocessing_consistency.md` |

---

## 🟢 Strongly supported (consistent evidence, but from a single corpus / limited replication)

| claim | evidence | what would settle it |
|---|---|---|
| **The structured head and a flat softmax are equivalent on accuracy-family metrics.** | macro-F1 −0.0055 (seed 42) / +0.0224 (seed 1337); paired t p = 0.589 / 0.494; corrected QWK differs by 0.0002 (p = 0.977). Sign reverses across seeds. | a third seed, or a second corpus |
| **Architecture is not the binding constraint.** | The five-backbone in-fold spread (0.028) equals the SD of a single run's last five epochs (0.007–0.056). | multi-seed backbone sweep |
| **The flat baseline is better than the current model at the external site.** | Image-level QWK 0.492 / 0.486 vs 0.473 / 0.456 — ahead at **both** seeds, and unlike CV the sign does not flip. | more seeds; a second external site |
| **Errors are ordinally structured, not random.** | 603 of 1,433 errors are adjacent-stage; only 27 are non-adjacent. | — |
| **Stage 1 is the model's weakest class.** | precision 0.376, the lowest cell; Stage 1 ↔ Stage 2 accounts for 373 errors. | clinician re-read of Stage 1 labels |

---

## 🟡 Uncertain

| question | why it is uncertain |
|---|---|
| Whether the internal AP-ROP recall of 0.345 reflects the model or the labels. | Fold 0 contributes 476 of 618 AP-ROP validation frames, 470 from a **single infant** whose diagnosis is a per-session code stamped on every frame. The patient-clustered CI on AP-ROP recall is correspondingly wide. |
| Whether Stage 4/5 performance means anything. | 89 training images, **all from one source** (ROP-VL). Internal recall 0.775 may be partly source recognition. Pre-registered as unvalidated. |
| Whether the site adversary helps. | Its dose–response was measured on fold 0 only, one seed; the implementation also damps the adversary's own learning (`w_site` multiplies the whole site loss, not just the reversed gradient), which deviates from canonical DANN. |
| The true operating point of the *served* checkpoint. | The served binary threshold (0.0155) was selected on **pooled out-of-fold** scores, but the served model is one fold's checkpoint that saw 4/5 of that data. Its internal sensitivity/specificity are therefore not the advertised 0.958 / 0.781. |
| Whether the last-epoch rule cost or bought accuracy here. | Best-minus-final val macro-F1 ranges 0.000 – 0.120 across runs. The rule is methodologically correct (no selection on val) but its noise is unquantified in any reported figure. |

---

## 🔴 Not demonstrated

- **Any clinical validity, clinical utility, or safety.** Dataset accuracy establishes none of these.
- **Per-stage usability at a new hospital.** ~0.28 recall is not a usable staging system.
- **Stage 4/5 and AP-ROP externally.** n = 4 and n = 12 locked images. Pre-registered as not externally validated; the numbers in those cells (0.750, 0.500) must not be quoted.
- **That the model "looks at" the pathology.** Grad-CAM retina attention is at chance; on Normal images 44% of attention is on padding the preprocessing invented.
- **That the probability outputs can be read as probabilities.** They cannot — see the calibration section.
- **That the structured head is preferable to a plain softmax** on anything except taxonomy fidelity. It is measurably worse on calibration and on external image-level QWK.
- **Generalisation beyond one external hospital**, one imaging vendor set, or beyond retrospective data.
- **Agreement against an independent grader.** All labels are the corpora's own; none have been re-read for this project.

---

## 🔬 Requires additional experimentation

| # | experiment | answers |
|---|---|---|
| 1 | Temperature-scale the CORN cascade on out-of-fold logits | can the calibration defect be removed without touching accuracy? |
| 2 | Re-run the reference config with `weighted_sampler: false` and stamp the checkpoint | does the repository reproduce its own headline model? |
| 3 | Score the temporal-view control on the dev half and on the CV folds | does the anatomical explanation replicate outside the locked set? |
| 4 | Run the CORN-removed / branch-removed / w_site=0 arms on folds 1–4, both seeds | does the one surviving "head effect" survive off fold 0? |
| 5 | Log validation loss and LR; re-run one fold | is there late-epoch overfitting the current logs cannot show? |
| 6 | Independent paediatric-ophthalmologist adjudication of ~300 stratified labels | what is the human ceiling, and is Stage 1 precision a model or a label problem? |
| 7 | A second external hospital | is the ~0.28 external stage recall site-specific or general? |

Full protocols: [`../18_RESEARCH_REVIEW/recommended_future_experiments.md`](../18_RESEARCH_REVIEW/recommended_future_experiments.md).
