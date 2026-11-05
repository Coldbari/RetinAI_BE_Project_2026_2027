# Critical Findings

*Ordered by severity. Each carries evidence, impact, a recommended action, and whether it
must be fixed before the next experiment. Findings prefixed **C** are outside this package
(repository-level); **H** = high; **M** = medium; **L** = minor.*

---

## C-1 · CRITICAL — patient-identifying data in a public repository (OUTSIDE this package)

**Evidence.** `DEMO.md` lines 25–26 contain two original Ostrava filenames of the form
`rop_<id>_<sex>_GA<weeks>_BW<grams>_PA<days>_DG1_PF0_D1_S<nn>_<n>.jpg` — the real values are
deliberately not reproduced in this package. They encode sex, gestational age, birth weight and
two postnatal ages for a single infant. The repository's **own** guard test
`tests/test_repo_guards.py::test_no_clinical_record_identifiers_in_tracked_text_files` **fails on
HEAD** (1 failed, 151 passed, 4 skipped). Introduced in `d8a20b7`, an ancestor of `main`;
`origin/main == HEAD`, so it is **pushed** to `github.com/Coldbari/archive`.

**Why it matters.** Sex + GA + birth weight is a re-identifying quasi-identifier set within a named NICU cohort. This is the fourth occurrence of this defect class in the project, and the first the guard caught but that was committed anyway.

**Impact.** Ethical and potentially legal; also a disclosure item for any paper's ethics section.

**Action.** Redact, rewrite history, force-push, log as a disclosed incident, and make the guard a pre-commit hook / CI gate so it blocks rather than reports.

**Must fix before next experiment?** **Yes — before anything else.** *(This package contains no such identifiers; see [`../20_AUDIT/privacy_audit.md`](../20_AUDIT/privacy_audit.md).)*

---

## H-1 · The reported QWK never matched its own declared definition — **CORRECTED IN THIS PACKAGE**

**Evidence.** Both staging configs declare `eval.qwk_classes: [Normal … Stage 4/5]`. A repository-wide grep finds **zero** Python references to the key. `models/common/metrics.py` computed quadratic-weighted κ over all six labels, placing AP-ROP at ordinal rank 5 — the exact modelling error the architecture exists to avoid.

**Impact.** Fold-0 QWK 0.5685 → **0.9077**. CV mean 0.8255 ± 0.1454 → **0.8964 ± 0.0288** (fold SD cut 5×). Every QWK in the project's docs, deck, ablation table and locked table is affected. **The "fold 0 is anomalous on QWK" claim is withdrawn**; the macro-F1 and calibration signatures of fold 0 survive.

**Action.** Done here: both values reported side by side in [`../07_TEST_RESULTS/qwk_analysis/`](../07_TEST_RESULTS/qwk_analysis/). Upstream, implement `qwk_classes` in `compute_metrics` and re-issue affected numbers.

**Must fix before next experiment?** **Yes** — every future QWK is otherwise mis-specified.

---

## H-2 · The eye-level headline is operator-dependent — **and a better instrument now exists**

**Evidence.** Same predictions, external locked set, corrected QWK: max-over-views **0.813**, majority 0.398, mean-probability 0.396, max-probability-per-class 0.351, one-random-view 0.482, no aggregation 0.473. Only `max` produces the gap, and `max` was adopted after inspecting dev-half results.

**But:** temporal-only frames, scored at **image** level with **no aggregation**, give **0.841** (nasal 0.352, posterior pole 0.229; the same pattern holds for the flat baseline).

**Impact.** The scientific claim — that the image→eye gap is a property of the labels and the anatomy — is **supported**, and by a control that cannot be attacked on aggregation grounds. But any report quoting "0.81 per eye" without naming the operator is quoting a selected number.

**Action.** Promote the region analysis to the primary instrument; report the eye-level figure as secondary with its rule named and the alternatives tabulated.

**Must fix before next experiment?** No — but before the paper is submitted.

---

## H-3 · The served checkpoint's operating point is not the advertised one — **IDENTITY NOW RESOLVED**

**Evidence.** The served weights were re-run over 663 dev images: **100% prediction agreement with the fold-0 checkpoint**, max probability difference 3e-06. So the served model is fold 0's checkpoint, trained on 4/5 of the internal corpus. The served binary threshold **0.0155** was selected on *pooled 5-fold out-of-fold* internal scores plus the external *dev* half.

**Impact.** The internal operating point shown in the app (sensitivity 0.958 / specificity 0.781) describes an out-of-fold distribution that this checkpoint does not produce — 4/5 of that internal data is in-sample for it. The external half of the selection was also used to choose the threshold. Neither advertised number is an unbiased estimate for the served model.

**Action.** Write `results/rop_staging/WEIGHTS.json` with fold, seed, run tag and SHA-256 (`11b4396086…`); assert the hash at app boot; re-derive the operating point **on this checkpoint's own score distribution**, holding out data it never saw.

**Must fix before next experiment?** Yes for the provenance stamp; the re-derivation can follow.

---

## H-4 · The contributed head is materially worse calibrated than its own baseline

**Evidence.** At essentially identical accuracy (0.786 vs 0.789): ECE **0.1093 vs 0.0346** (3.2×); predictions above 0.90 confidence **68.6% vs 13.1%**; high-confidence wrong predictions **359 vs 181**. Externally the current model's ECE is **0.334** and its confidence separation between right and wrong collapses to **0.041**.

**Impact.** A measured disadvantage of the CORN cascade's probability geometry that no prior artifact in the project compared. The product displays these percentages.

**Action.** Fit temperature scaling on out-of-fold CORN logits (experiment 1). Until then, label the staging percentages in the UI as an uncalibrated ranking, as the binary card already does.

**Must fix before next experiment?** No, but it is the highest-value cheap fix.

---

## H-5 · The most dangerous error class is the most confident one

**Evidence.** "Disease stage called Normal": n=186, **mean confidence 0.934**, 141 (76%) above 0.90 — the highest mean confidence of any error category, and above the model's overall accuracy.

**Impact.** Rules out the obvious safety mitigation ("only auto-report high-confidence cases") — such a rule would *retain* 141 of the 186 missed-disease cases while discarding harmless adjacent-stage errors.

**Action.** Report this explicitly in any deployment discussion. Re-check after temperature scaling: if calibration does not separate these cases, no confidence-based gate is viable.

**Must fix before next experiment?** No — but it must be stated wherever the model is proposed for use.

---

## H-6 · The committed config reproduces the documented failure arm, not this model

**Evidence.** `configs/rop_staging_structured.yaml` sets `weighted_sampler: true`. Every headline run overrode it to `false` from `kaggle_kernels/rop_structured_benchmark.py`. `weighted_sampler: true` + CORN is the recorded v1 collapse (QWK 0.121, stage_4/5 precision 0.043). The seed-42 run summary does not record its own sampler setting; only the seed-1337 summary does.

**Impact.** Running the repository's own documented command reproduces the broken model. Reproducibility is broken at the most visible entry point.

**Action.** Set `weighted_sampler: false`; record sampler, `w_site`, seed and git hash in every run summary.

**Must fix before next experiment?** **Yes** — trivial, and it protects every future run.

---

## H-7 · Grad-CAM cannot support any claim about what the model looks at

**Evidence.** Fresh audit: attention on retina **56.0%** against a retina area share of **53.8%** → **lift 1.04 (chance)**. **30.7%** of peaks fall outside the retina; **19.3%** on invented letterbox padding. On Normal images, 39.5% retina vs **43.9% padding**. Attention quality is uncorrelated with confidence. Independently corroborated by the project's own `cam_leakage.json` (55.5% / 27.8%).

**Impact.** The product ships a heat map as evidence. Users will read it as the reason for the verdict. It is not.

**Action.** Explain a single class rather than a sum; use an earlier block or Grad-CAM++; add an off-retina abstention gate; print the raw off-retina fraction beside the overlay.

**Must fix before next experiment?** No — but the honest wording in [`../14_EXPLAINABILITY/explainability_limitations.md`](../14_EXPLAINABILITY/explainability_limitations.md) should be adopted immediately.

---

## M-1 · Class–source confounding is under-declared

Only `stage_4_5` is listed in `unvalidated_classes`, but **AP-ROP has zero Shenzhen images** and **Stage 3 is only 7% Ostrava**, while site identity is decodable at 0.997 balanced accuracy. Add both to the declaration. *(Fix before publishing per-class claims.)*

## M-2 · All ablations sit on fold 0, the atypical fold

The seven-arm component table, the DANN dose–response and the five-backbone benchmark are single-fold, single-seed — and fold 0 holds 476 of 618 AP-ROP validation frames, 470 from one infant. *(Fix before the ablation table is published.)*

## M-3 · Validation loss and learning rate were never logged

Train-vs-validation-loss overfitting diagnosis is **not available in the provided evidence** for any run. The LR curve in this package is reconstructed and labelled as such. *(Fix in the next training run.)*

## M-4 · `set_seed` is incomplete

No cuDNN determinism, no `PYTHONHASHSEED`, no DataLoader generator. Runs are approximately, not bit-, reproducible — while seed noise is itself a headline finding. *(Fix before the next seed experiment.)*

## M-5 · `FundusDataset` substitutes a black image for an unreadable file, silently

`__getitem__` returns a zero tensor **with the true label**, uncounted and unlogged. No failure occurred in these runs (all paths verified present), but the failure mode is silent by construction. *(Fix: raise, or count and log.)*

## M-6 · `pred6` and `argmax(prob6)` disagree in 1.19% of predictions

The label uses a hard 0.5 threshold on the AP-ROP branch; the probability vector composes multiplicatively. F1/accuracy and AUC therefore rest on different decision rules. The flat baseline's disagreement is 0.00%. *(Report the rate; do not "fix" by tuning the threshold.)*

## M-7 · Stage 1 precision 0.376 is undiscussed upstream

The weakest cell in the package, CI [0.273, 0.487]. 169 Normal and 193 Stage 2 images are called Stage 1. *(Needs the clinician-adjudication experiment to separate model error from label noise.)*

## M-8 · dev and locked halves are not distribution-matched

dev: 190 Stage 1 / 150 Normal. locked: 105 Stage 1 / 232 Normal. Since dev drives threshold selection and locked produces the final number, the mismatch propagates.

## M-9 · DANN deviates from its own specification

`w_site` multiplies the entire site loss, damping the adversary's own learning, rather than scaling only the reversed gradient into the backbone. Consistent across arms, so ablations remain internally valid, but the `w_site` axis is not the λ axis the DANN literature reports.

## L-1 · Three silent no-op config keys

`qwk_classes`, `unvalidated_classes`, `label_smoothing` are declared and read by nothing — in a file whose header says every key is read.

## L-2 · README claims 5-view TTA; the served path does one forward pass

Harmless numerically (the evaluation artifacts are also single-pass) but the documentation is wrong.

## L-3 · `requirements.txt` is unpinned

All `>=`; no per-run environment freeze beyond the torch version string in `meta.json`.
