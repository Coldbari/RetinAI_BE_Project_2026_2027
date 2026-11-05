# RetinAI — Remediation Plan

**Status:** design complete, nothing implemented.
**Provenance:** five parallel design agents drafted 56 tasks (~165 h raw). The adversarial
critique pass was cut short by a session limit, so the de-duplication, cutting and sequencing
below were done by hand. Where two sections designed the same feature, this document picks one.

---

## Executive summary

Direct testing against an external dataset (HRF, FAU Erlangen) and the project's own infant
cohort found that **all three deployed models return a positive finding on 100% of non-retinal
input** — random noise, a solid grey square, and two of the project's own matplotlib charts. The
ROP model fires on 91% of adult eyes (anatomically impossible); the Glaucoma model calls 55% of
preterm infants glaucomatous. Separately, the reported ROP AUC of 0.927 rests on only **29 test
patients, 7 of them positive**, and the Glaucoma external result is **contaminated** — HRF is
dataset #9 inside SMDG-19.

None of this is visible in the reported metrics, because every model was only ever evaluated on
the population it was trained for. The models are not broken; the **system around them** has no
input validation, no domain boundary, and no honest uncertainty.

The plan below is ordered by risk-reduction per hour, not by topic. Phases are independently
shippable — stopping after any phase leaves the project in a coherent, defensible state.

| Phase | Focus | Effort | Kaggle GPU? |
|---|---|---|---|
| **0A** | Domain routing — stop models running on the wrong patient | ~1.5 days | no |
| **0B** | Gradability gate — stop non-fundus input | ~2 days | 1 short session |
| **1** | Honest numbers (patient-level ROP, CIs, significance tests) | ~2 days | no |
| **2** | Calibration, abstention, honest confidence | ~3 days | 1–2 sessions |
| **3** | Contamination audit + clean external validation | ~3 days | 2–3 sessions |
| **4** | Standing test suite + CI | ~2.5 days | no |
| — | Deferred (see final section) | — | — |

**If the presentation is under one week:** do **0A + Phase 1** only (~3.5 days). That removes the
two most embarrassing live failures and makes every quoted number defensible. Everything else can
be presented as measured, understood, and scheduled — which is a stronger position than silence.

---

## Phase 0A — Domain routing

**Problem.** `Registry.analyze()` (`webapp/inference.py:131-140`) runs all three models on every
upload with no notion of who the patient is. ROP is a disease of preterm infants only; the DR and
Glaucoma models were trained exclusively on adults. Nothing prevents any model from judging any
patient.

**Evidence.**

| Cohort | n | ROP flags | Glaucoma flags |
|---|---|---|---|
| Adult healthy (HRF) | 15 | **93%** | 0% |
| Adult DR (HRF) | 15 | **93%** | 13% |
| Adult glaucoma (HRF) | 15 | **87%** | 87%⚠ |
| Infant, no ROP | 22 | 32% | **59%** |
| Infant, ROP+ | 7 | 100% ✓ | **43%** |

The ROP model is *correct in its own domain* (7/7 true positives) — it simply has no boundary.

**Design decision.** Routing is driven by an **operator-declared patient context**, not an
image-derived classifier. An image-derived signal may *warn*, never *route*. Rationale: a learned
infant/adult classifier is a new model with its own unvalidated failure modes, layered underneath
three models whose failure modes we are currently trying to contain. The upload form already
collects patient fields; adding one required selector is two hours and cannot silently misfire.

| ID | Task | Effort | Acceptance criterion |
|---|---|---|---|
| R1 | Add a declarative `contexts:` routing table to `webapp/registry.yaml` (`adult`→[DR, Glaucoma], `infant`→[ROP], `demo`→all three marked advisory). `Registry.__init__` hard-fails at boot if `contexts` is absent or names an unknown disease. | 1.5 h | App refuses to start with a malformed routing table; unit test asserts the boot failure. |
| R2 | Rewrite `Registry.analyze(image, context)` to be context-aware; routed-out models return an explicit `not_applicable` state rather than being silently omitted. Generate one Grad-CAM **per positive disease**, not one overall. | 3 h | Re-running the adult cohort with `context=adult` yields **0** ROP findings; infant cohort with `context=infant` yields 0 Glaucoma findings. |
| R3 | Make patient context a required field on `/predict` and `/report`; reject with 400 if missing. | 2.5 h | `POST /predict` without context returns 400; no default fallback exists in code. |
| R4 | Screen page: required context selector, "Not assessed — outside this model's population" cards for routed-out diseases, per-disease heatmaps. | 3 h | Cannot submit the form without choosing a context. |
| R5 | Fix `reports/report_generator.py`, which currently prints **"Healthy retina"** for models that never ran. | 2 h | PDF for a routed-out disease reads "Not assessed", never "Healthy". |
| R6 | Routing regression test; re-run all six cohorts through the routed stack and commit the result table. | 3 h | Committed baseline shows 0 cross-domain firings. |
| R7 | Correct product copy that routing makes false (dashboard/About claims of "three diseases from one image"). | 1 h | No page claims all three run on every image. |

**Exit criteria.** Zero cross-domain firings on all six cohorts. No route defaults silently. PDF
never reports a non-run model as healthy.

> **R5 is the sleeper here.** A downloadable PDF that says "Healthy retina" for a model that never
> executed is the single most dangerous artifact in the system, because it outlives the session.

---

## Phase 0B — Gradability gate

**Problem.** There is no check that an uploaded file is a retinal photograph, or that it is
gradable. The literature is blunt: poor-quality images cause AI failures where a clinician would
simply reject the image and re-image.

**Evidence.** Non-fundus probes — random RGB noise, a solid grey square, `evaluation_dashboard.png`,
`training_curves_unified.png` — scored **positive on all three diseases, 4/4, at high confidence.**

**Design decision.** Classical signals first. `models/common/preprocessing.py` already has
`circle_crop`, and `models/common/data_quality.py` already has variance-of-Laplacian blur
detection. A learned fundus-vs-not gate is **deferred**, not adopted, until the classical gate is
measured and shown insufficient.

| ID | Task | Effort | Acceptance criterion |
|---|---|---|---|
| G1 | Implement `gradability()` in `data_quality.py`: circular-FOV presence, retinal colour-distribution check, blur threshold, over/under-exposure. Returns verdict + reason. | 3 h | Returns `not_fundus` for all four probe images. |
| G2 | Calibrate thresholds **on training splits only**, never on test or on the HRF cohort. Publish the calibration report. | 4 h | Written report states each threshold and the split it came from. |
| G3 | Wire into `Registry.analyze`, **fail-closed**, on both the predict and PDF paths. | 3 h | An ungradable image produces no disease prediction and no downloadable PDF. |
| G4 | Front-end verdict banner; block the report download when ungradable. | 3 h | Banner states the specific reason (blur / not fundus / exposure). |
| G6 | Harden the upload path (MIME/type/size validation, decompression-bomb guard). | 1 h | Non-image uploads rejected before reaching any model. |
| G7 | Gate validation suite — the project's first automated tests. | 4 h | **False-rejection rate < 2%** on the 6,004 infant images and the 45 HRF images. |
| G8 | Measure false-rejection on EyePACS + APTOS (Kaggle — the only cohort not testable locally). | 3 h | FRR reported per source; any source above 5% triggers threshold review. |
| G9 | Mirror to `retinal-ai/`, redeploy, re-run the four original probes against the live Space. | 2 h | All four probes rejected in production. |

**Exit criteria.** 4/4 probes rejected in production; FRR < 2% on legitimate cohorts. **G7's FRR
budget is the real gate on this phase** — a gate that rejects real patients is worse than no gate.

*Note: G5 from the original draft duplicated R3/R4 and has been removed. G10 (learned gate) is deferred.*

---

## Phase 1 — Honest numbers

**Problem.** Headline metrics overstate their own precision and, in one case, may measure the
wrong thing entirely.

**Evidence.**
- ROP test split: **1,502 images, 29 patients, 7 ROP-positive.** Image-level metrics treat repeat
  photographs of the same infant as independent samples. AUC 0.927 has a far wider real interval
  than quoted.
- McNemar and DeLong are **implemented** in `models/evaluation/statistical_tests.py` and have
  **never been run**, yet ResNet50 was selected over EfficientNetV2-S on an unweighted comparison.
- Only DR referable AUC has a bootstrap CI on disk.
- `results/dr/config.yaml` records `circle_crop/ben_graham/clahe = false`; `configs/dr.yaml` (what
  the app serves) has all three `true`. **Unresolved:** the checkpoint's own config produced *worse*
  predictions in spot-testing, so the recorded config may be stale rather than authoritative.

**This entire phase runs on the laptop.** No GPU, no downloads.

| ID | Task | Effort | Acceptance criterion |
|---|---|---|---|
| S1 | Make `evaluate.py --dump-preds` also save `image_path` (required to group by patient). | 45 m | Dumps contain a path column. |
| S2 | Dump ROP predictions for all three on-disk checkpoints, test and val. | 2 h | Three prediction files on disk. |
| S3 | Add **cluster bootstrap** (resampling patients, not images) and Clopper-Pearson exact CIs to `statistical_tests.py`. | 1.5 h | Self-test on synthetic clustered data recovers known coverage. |
| S4 | Patient-level ROP metrics with cluster-bootstrap CIs. | 4 h | Reported ROP AUC carries a patient-level CI; the 7-positive limitation is stated in the output. |
| S5 | Re-select the ROP operating threshold on **val**, then measure it once on test. | 2 h | Threshold provenance documented; test used exactly once. |
| S6 | Run McNemar + DeLong on the ROP architecture choice. | 3 h | A p-value exists for ResNet50 vs EfficientNetV2-S. |
| S7 | Bootstrap CIs for every headline metric computable from on-disk predictions. | 3 h | No headline number is quoted without an interval. |
| S8 | **Resolve the DR config ambiguity empirically** — run both preprocessing configs over the val set; whichever reproduces the recorded QWK 0.728 is the true one. (Kaggle.) | 5 h | A written verdict naming which config trained the checkpoint, with the QWK each produced. |
| S9 | Act on the S8 verdict across serving, external validation and docs. | 3 h | `webapp/registry.yaml` points DR at the verified config. |

**Exit criteria.** Every published number has a unit, a split, an interval, and a contamination
status. The DR config question is answered with evidence rather than assumption.

> **Do not apply the "obvious" DR fix before S8.** Pointing the registry at
> `results/dr/config.yaml` looked correct until testing showed that config performs *worse*. S8
> exists precisely to prevent a confident wrong fix.

---

## Phase 2 — Calibration, abstention and honest confidence

**Problem.** The app displays raw TTA-averaged softmax as "confidence", and the model must always
answer.

**Evidence.** Temperature scaling is computed offline (DR 5-class T=0.78, referable T=0.70, ROP)
but **never loaded by `webapp/inference.py`**. Worse: those constants were fitted on single-view
logits while serving averages **five TTA views** — so even wiring them in as-is would be invalid.
Observed pathology: a healthy eye scored `No DR 58.5%`, yet `Severe 17.6% + Proliferative 15.9%` —
33% of mass on the two worst grades, a non-monotonic differential for an ordinal disease.

| ID | Task | Effort | Acceptance criterion |
|---|---|---|---|
| C1 | `fit_serving_calibration.py` producing per-disease calibration artifacts. | 4 h | One script regenerates every constant. |
| C2 | Dump val predictions for ROP and Glaucoma (neither is currently calibratable at all). | 3 h + 1 session | Val dumps exist for all three diseases. |
| C3 | **Refit all constants under the real 5-view TTA pipeline.** | 1 day | ECE measured on TTA-averaged outputs, not single-view. |
| C4 | Load and apply calibration in `webapp/inference.py`, with visible degradation if a constant is missing. | 4 h | Missing constant shows "uncalibrated", never silently falls back. |
| C5 | Replace argmax with an explicit **referable** decision for the ordinal DR head. | 3 h | DR decision matches the reported referable endpoint. |
| C6 | Abstention band + "Uncertain — needs human review" state, threshold chosen on val. | 6 h | Abstention rate reported per cohort. |
| C8 | Retire the word "confidence" for the argmax probability across UI, API and PDF. | 4 h | No surface labels a raw probability "confidence". |

**Exit criteria.** ECE reported under the true serving pipeline; an abstention path exists; no UI
surface overstates certainty.

*Cut from the original draft: C7 (ordinal ladder UI redesign), C9 (doc rewrite — folded into S13/Phase 1), C10 (TTA-disagreement signal). All are polish relative to C3–C6.*

---

## Phase 3 — Contamination audit and clean external validation

**Problem.** The Glaucoma model's external results cannot be trusted, and no glaucoma benchmark
has been verified clean.

**Evidence.** **HRF is dataset #9 inside SMDG-19.** The Glaucoma model scored 87% on HRF glaucoma
images it may have trained on. Meanwhile SMDG-19 contains REFUGE1-**TRAIN** and
REFUGE1-**VALIDATION** but *not* REFUGE1-**TEST** — that split is the one candidate clean benchmark.

| ID | Task | Effort | Acceptance criterion |
|---|---|---|---|
| S11 | SMDG-19 source-provenance audit; per-source contamination verdict. | 6 h | Every glaucoma number labelled in-distribution or held-out. |
| S12 | Cross-split perceptual-hash duplicate audit for ROP (patient-level splits still leak if duplicates exist). | 3 h | Duplicate count across splits reported; zero or explained. |
| T11 | **Prove REFUGE1-TEST is uncontaminated before using it** — phash it against the SMDG-19 training pool. | 3 h | Written proof of zero overlap, or the benchmark is abandoned. |
| T12 | Run the REFUGE1-TEST external glaucoma validation. | 2 h + session | A zero-shot glaucoma AUC on genuinely unseen data. |
| T8 | **ROP device-confound audit.** Check whether camera/device identity correlates with diagnosis in the private database. | 4 h | Per-device ROP prevalence table; confound quantified. |
| T10 | Fix `external_validation.py` output collisions and its binary-reporting gap. | 1.5 h | Two external runs no longer overwrite each other. |
| T13 | DR external panel: DDR (has an ungradable class — useful for G1) and IDRiD. | 4 h + sessions | DR referable AUC on two further sources. |

**Exit criteria.** Every external claim is labelled contaminated or clean, with evidence.

> **T8 is the highest-value item nobody asked for.** If camera identity correlates with ROP
> diagnosis in the private database, the model may be reading the *device* rather than the disease
> — and no amount of patient-level splitting would catch it. This is the most likely hidden reason
> ROP scores 0.927 internally.

---

## Phase 4 — Standing test suite and CI

**Problem.** No automated tests, no CI. Every finding in this document was discovered by hand and
nothing prevents regression.

| ID | Task | Effort | Acceptance criterion |
|---|---|---|---|
| T1 | `tests/` scaffold + pytest config. | 1.5 h | `pytest` runs green. |
| T2 | Deterministic cohort builders (public fetch / private extract / synthetic probes). | 4 h | Any cohort rebuilds from one command. |
| T3 | **PHI / private-data leak guard.** | 45 m | CI fails if any infant image or patient identifier is staged for commit. |
| T4 | Weight-free unit tests, including a **train/serve config-parity guard**. | 3 h | CI fails if a served config diverges from its checkpoint's recorded config. |
| T5 | Cohort behaviour regression with a committed baseline. | 5 h | Cross-domain firing rate is asserted, not just reported. |
| T6 | Minimal CI (Tier-0 + baseline-staleness gate). | 2 h | Runs on push. |
| T7 | Degradation harness → per-disease fragility curves. | 5 h | Fragility curve per disease. |

**T3 is not optional and should be done first in this phase.** The infant database is private
patient data; `rop_dataset.zip` (2.6 GB) currently sits in the repo root. Verify it is
gitignored before anything else here.

**Exit criteria.** CI green on push; the config-parity guard would have caught the DR skew.

---

## Contingency — what to do if a validation comes back bad

Several planned tasks can produce unflattering results. Decide the response **now**, before
running them, so the decision isn't made under pressure.

| If… | Then |
|---|---|
| **T12** — REFUGE1-TEST glaucoma AUC collapses (as G1020 did, 0.957→0.589) | **Report it.** This is the same class of finding as the existing domain-shift story, which is already the strongest material in the deck. A second independent confirmation strengthens the thesis; hiding it and being asked is fatal. Reframe glaucoma as "internally strong, externally unproven". |
| **S6** — DeLong says ResNet50 vs EfficientNetV2-S is **not** significant | Report the p-value and soften the claim from "ResNet50 is better" to "ResNet50 scored higher; with 29 test patients the comparison is underpowered". This is a *more* sophisticated statement than the current one. |
| **S8** — neither DR config reproduces QWK 0.728 | The checkpoint's provenance is unknown. Say so, quote the reproducible number you *can* measure, and stop quoting 0.906 until a re-run exists. |
| **T8** — device correlates with ROP diagnosis | Major finding. Re-split by device, retrain, report both numbers. Costly but this is exactly the kind of self-caught confound that distinguishes real work. |
| **S4** — patient-level ROP CI is very wide (likely, with 7 positives) | Expected, not a failure. Report the interval and state plainly that more ROP patients are the binding constraint. |
| **G7/G8** — gate false-rejection exceeds budget | Do **not** ship the gate loose to hit a deadline. Ship it in warn-only mode (banner, no block) and state that blocking awaits threshold work. |

**Governing principle:** every one of these outcomes is more defensible when self-reported than
when discovered by an examiner. The project's strongest existing asset is the glaucoma
domain-shift story — a failure that was found, diagnosed, and fixed. These are the same kind.

---

## What each phase lets you defend

| Phase | Question you can then answer |
|---|---|
| 0A | *"What if I upload an adult eye to your ROP model?"* → "It isn't offered — routing is by declared patient context, and here's the cohort table proving zero cross-domain firings." |
| 0B | *"What if the image is blurry or isn't a retina?"* → "Rejected with a reason. Here's the false-rejection rate on 6,000 real images." |
| 1 | *"Is 0.927 real?"* → "It's 29 patients and 7 positives. Here's the patient-level interval, and here's why the image count overstates it." |
| 2 | *"Does 87% confidence mean 87%?"* → "Now it does — measured under the actual TTA pipeline. And below this band the system declines to answer." |
| 3 | *"How do you know the glaucoma result isn't memorised?"* → "I found HRF inside SMDG-19 myself. Here's the contamination audit and a verified-clean benchmark." |
| 4 | *"How do you know this doesn't regress?"* → "CI asserts it, including a config-parity guard that would have caught the DR skew." |

Phase 1 has the highest defensibility-per-hour and needs no GPU. Phase 0A has the highest
demo-risk reduction. Those two are the short-timeline package.

---

## Explicitly not doing, and why

| Deferred | Why |
|---|---|
| **Learned fundus-vs-not gate** (G10) | A new unvalidated model beneath three models whose failure modes we're containing. Revisit only if the classical gate misses G7's budget. |
| **Image-derived infant/adult classifier** (R9) | Same reasoning. Operator-declared context cannot silently misroute. |
| **ROP device-held-out retrain** (T9) | Gated on T8's audit. Retraining before knowing whether a confound exists is speculative. |
| **RETFound pretraining** | Highest-value long-term change (retina-pretrained encoder directly addresses ResNet50 overfitting 6k images) — but it invalidates every current number and needs re-validation. Post-presentation. |
| **ROP zone/stage/plus staging** | The right thesis contribution if ROP becomes the flagship. Needs new labels. |
| **Patient persistence / auth / DICOM** | Fully designed in `docs/future/`. Irrelevant to the failures found here. |
| **Regulatory work** | Out of scope. This is a decision-support prototype and should keep saying so. |
| **Animal-retina testing** | Already answered *a fortiori*: a solid grey square scored positive on all three diseases, so an animal retina certainly would. Becomes a meaningful adversarial check only **after** Phase 0B, at which point it's a good test of the gate. |
