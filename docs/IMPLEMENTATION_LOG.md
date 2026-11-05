# Implementation Log

Execution record for `docs/REMEDIATION_PLAN.md`. One entry per task. Each entry states what
changed, how it was verified, and the self-review verdict. **A task is only marked PASS when its
acceptance criterion from the plan has been checked by running something — not by inspection.**

Verdict legend: **PASS** (acceptance criterion met and verified) · **PASS-WITH-NOTE** (met, but
something was found worth recording) · **BLOCKED** (cannot proceed, reason stated) · **REVERTED**.

---

## Phase 0A — Domain routing

**Goal:** no model may run on a patient population it was not trained for.

**Baseline before any change** (measured, `scratchpad/sweep.py`):

| Cohort | n | DR referable | ROP flags | Glaucoma flags |
|---|---|---|---|---|
| Infant, no ROP | 22 | 14% | 32% | 59% |
| Infant, ROP+ | 7 | 14% | 100% | 43% |
| Adult healthy | 15 | 0% | 93% | 0% |
| Adult DR | 15 | 53% | 93% | 13% |
| Adult glaucoma | 15 | 0% | 87% | 87% |
| Non-fundus | 4 | 100% | 100% | 100% |

Cross-domain firings to eliminate: **ROP on 41/45 adults**, **Glaucoma on 16/29 infants**.

---
### R1 — routing table + fail-fast boot validation · **PASS**

**Changed:** `webapp/registry.yaml` — added `contexts:` (adult / infant / demo) and `applies_to:`
per model. `webapp/inference.py` — new `RegistryConfigError`; `Registry._validate_contexts()`
runs at boot.

Validation is deliberately stricter than the plan required. It rejects four separate mistakes:
a missing `contexts` block, a context routing to an unknown disease, a model with no
`applies_to`, and — the important one — **a context that runs a model outside its declared
population**. That last check is the exact regression that caused this phase to exist: it makes
"wire ROP into the adult context" un-deployable rather than merely wrong.

**Verified:** `tests/test_routing.py` — 7 tests covering the shipped table and all four failure
modes. Boot failure asserted with `pytest.raises`, not by reading code.

---

### R2 — context-aware `analyze()` + per-disease Grad-CAM · **PASS**

**Changed:** `Registry.analyze(image, context)` now takes a **required** context and runs only
the routed models. `Registry.context_choices()` added for the UI.

Two deliberate design choices:

1. **Routed-out models stay in `findings`** with `available=False, status="not_applicable"`.
   Omitting them would render as "nothing found", which is the same false reassurance the PDF
   bug produces (see R5).
2. **One Grad-CAM per positive disease**, returned as a `heatmaps[]` list. The old single-heatmap
   fields are retained so nothing downstream breaks before R4 updates the front-end.

**Verified:** 6 behaviour tests. `context="adult"` returns ROP as `not_applicable`;
`context="infant"` does the same for DR and Glaucoma; all three diseases always present;
unknown context raises; **calling `analyze()` with no context raises `TypeError`** — there is no
default, so a caller cannot accidentally get the old unrouted behaviour.

```
.venv/bin/python -m pytest tests/test_routing.py -q  ->  13 passed in 25.54s
```

**Note carried forward:** `webapp/app.py` calls `analyze(image)` in three places and is broken
until R3. Intentional — the signature change is what forces every caller to declare a context.

**Environment note:** `.venv/bin/pip` has a stale shebang (`/Users/praharsh/Downloads/archive/.venv/`);
the venv was moved at some point. Use `.venv/bin/python -m pip` instead. Not fixed — out of scope,
but it will bite the CI task (T6) in Phase 4.

---
### R3 — patient context required on `/predict` and `/report` · **PASS**

**Changed:** `webapp/app.py` — `_require_context()` helper; both endpoints return **400** with the
list of valid choices when context is missing or unknown. New `GET /contexts` so the UI reads the
routing table instead of hard-coding patient types. History entries now store `context` and
`context_label`.

`/history/report/<id>` re-runs under **the context the screening was originally performed with**.
A pre-routing entry has no context, so it returns **409** rather than silently picking one — a
re-downloaded report that applied different routing than the original screening would be worse
than no report.

**Verified:** 8 HTTP tests. Missing context → 400; unknown context → 400; `adult` → ROP
`not_applicable`; `infant` → DR and Glaucoma `not_applicable`; history records its context;
legacy entry → 409.

---

### R5 — PDF must never present an unrun model as a clinical result · **PASS**

**The bug.** `generate_pdf` filtered to `available` findings only, and any routed-out disease that
did reach it fell through to `recommendation(disease, 0)` — printing
*"No ROP signs — continue scheduled screening"* for a model that never executed. A downloadable
PDF asserting health for an unrun model is the most dangerous artifact in the system: it outlives
the session and carries no indication it is unfounded.

**Changed:** `reports/report_generator.py` — findings with `available=False` render in grey as
**"Not assessed"** with the reason, and never reach `recommendation()`. `webapp/app.py` now passes
**all** findings to the PDF (previously filtered), so a routed-out disease cannot vanish — an
omission reads as "nothing found". Screening context added to the patient block as provenance.

**Verified:** decoded the generated PDF's content streams and asserted on the drawn text —
`"Not assessed"` present, `"No ROP signs"` **absent**, `"ROP"` still listed, context recorded.

```
.venv/bin/python -m pytest tests/ -q  ->  25 passed in 23.80s
```

#### Two findings from the gate itself

**1. My test helper was wrong, not the code.** Three PDF tests failed initially because the helper
split on `b"stream"` — which also matches inside `b"endstream"`, corrupting every chunk boundary.
ReportLab also encodes content ASCII85-then-Flate, which the helper did not decode. Fixed the
helper; the R5 behaviour had been correct all along. *Recorded because a test that fails for its
own reasons is indistinguishable from a real defect until you read the traceback — the reflex to
"fix" the code here would have introduced a bug.*

**2. The fourth failure was a real defect in my own code.** `history_report` decoded the image
**before** checking the context, so a legacy entry raised `UnidentifiedImageError` instead of
returning 409. Reordered. This is exactly what the gate is for — it caught an ordering bug that
inspection had missed twice.

---
### R4 — screen page context selector + "Not assessed" cards · **PASS**

**Changed:** `screen.html` gains a required **Patient context** step (now step 1; image is step 3).
`main.js` fetches `/contexts` so the UI never hard-codes patient types, and **Analyze stays
disabled until both an image and a context exist** — there is no default. Routed-out diseases
render as dashed "Not assessed" cards rather than being hidden. Per-disease Grad-CAM tabs replace
the single Original/Heatmap pair. A banner states the context the screening ran under; demo mode
is styled amber and labelled *not clinically valid*. CSS appended to `style.css`.

Also relabelled the score readout from "confidence" to "score" — the displayed number is raw
TTA-averaged softmax and is not yet calibrated. Full retirement of the word is task C8 (Phase 2);
this avoids the UI asserting something Phase 2 has not yet earned.

---

### R7 — correct copy that routing makes false · **PASS**

`screen.html` hero and loader no longer claim all three models run on every image.
`app.py` module docstring records that serving is routed by patient context.

---

### R6 — cohort regression through the routed stack · **PASS**

```
cohort          ctx        n |            DR |           ROP |      Glaucoma   (was → now)
INFANT no-ROP   infant    22 |      14%→  0% |      32%→ 32% |      59%→  0%
INFANT ROP+     infant     7 |      14%→  0% |     100%→100% |      43%→  0%
ADULT healthy   adult     15 |       0%→  0% |      93%→  0% |       0%→  0%
ADULT DR        adult     15 |      53%→ 53% |      93%→  0% |      13%→ 13%
ADULT glaucoma  adult     15 |       0%→  0% |      87%→  0% |      87%→ 87%
NON-FUNDUS      adult      4 |     100%→100% |     100%→  0% |     100%→100%

PASS — zero cross-domain firings across 78 images in 6 cohorts.
```

**ROP on adults: 41/45 → 0. Glaucoma on infants: 16/29 → 0. DR on infants: 4/29 → 0.**

The important control: **every in-domain number is unchanged** (ROP 32%/100% on infants, DR 53%
on adult DR, Glaucoma 87% on adult glaucoma). Routing changed *which* models run, not *how* they
behave — so no diagnostic capability was traded for the safety fix.

**Read the NON-FUNDUS row carefully.** ROP shows 0% only because that cohort is evaluated under
`adult` context, so the ROP model never ran. Nothing about non-retinal input was fixed — DR and
Glaucoma still fire on 100% of it. That is Phase 0B and it remains fully open.

#### Finding from the gate itself

**The regression script reported PASS while testing zero images.** The scratchpad holding the
cohorts had been wiped between sessions; every cohort was missing, and the script printed
"PASS — zero cross-domain firings" because it had found no violations in the empty set. An empty
pass is indistinguishable from a real one in a log and silently certifies a system that was never
exercised — strictly worse than no test.

Fixed: missing cohorts now return exit code **2** with an explicit "this run proves nothing",
and the PASS line states the image and cohort count it is based on. Added
`tests/build_cohorts.py` so the cohorts rebuild deterministically (seed 42, one image per
patient) instead of living in a temp directory. `.cohorts/` added to `.gitignore` — it holds
private infant patient images.

---

## Phase 0A — EXIT CRITERIA MET

| Criterion | Result |
|---|---|
| Zero cross-domain firings on all six cohorts | ✅ 78 images, 0 violations |
| No route defaults silently | ✅ `analyze()` raises `TypeError` without context; API returns 400 |
| PDF never reports a non-run model as healthy | ✅ renders "Not assessed"; `"No ROP signs"` absent |
| Un-routed build cannot deploy | ✅ 4 boot-validation failure modes tested |

```
.venv/bin/python -m pytest tests/ -q                    ->  25 passed
.venv/bin/python tests/cohort_regression.py --cohort-root .cohorts  ->  PASS (78 images)
```

**Not done in this phase, carried forward:**
- `retinal-ai/` deployment mirror is **not** synced (task R8). The live HuggingFace Space still
  runs the unrouted stack. Deferred deliberately: Phase 0B changes the same serving path, so one
  sync after both phases avoids two redeploys and two rounds of Space cold-start verification.
- `webapp/templates/history.html` and `history.js` were not audited for the new
  `available`/`status` fields on history findings. Entries created before this change lack them.
- Phase 0B (gradability gate) is untouched — **non-fundus input still scores 100% positive on
  DR and Glaucoma.**

---

## Phase 0B — Gradability gate

**Goal:** no model sees an image that is not a gradable fundus photograph.

**Baseline:** random noise, a solid grey square, and two of the project's own matplotlib charts
scored **positive on all three diseases, 4/4, at high confidence.**

---

### G1 — `gradability()` in `models/common/data_quality.py` · **PASS**

Classical only, no new trained model. Reuses the `gray > 7` field-of-view convention from
`preprocessing.circle_crop` and the variance-of-Laplacian blur measure already in this module.
Verdicts: `gradable | not_fundus | too_blurry | poor_exposure`.

**Three colour signals are required, because no single one catches everything** (measured):

| probe | redness | saturation | red-hue frac | caught by |
|---|---|---|---|---|
| grey / white | 1.00 | **0.0** | 1.00 (undefined hue reads red) | saturation |
| charts | 0.96–0.98 | **5.6 / 29.9** | 0.48–0.61 | saturation |
| random noise | 1.00 | **99.9** | **0.249** | red-hue only |
| infant train (n=300) | min 1.06 | min 38.1 | min 0.384 | — |

Random noise has saturation **99.9 — above the infant minimum of 38.1** — so a saturation-only
gate would have accepted it. An image is rejected if *any* signal says non-retinal.

`require_rgb_order` is **off**: R>G>B holds on only 95% of infant training images, so requiring
it would falsely reject 1 in 20 real patients.

---

### G2 — threshold calibration on TRAIN splits only · **PASS (after a failed first attempt)**

`scripts/calibrate_gate.py` → `results/gate_thresholds.json` + `results/gate_calibration.md`.
Calibrated on **400 infant TRAIN images only**. Infant test, all HRF cohorts and EyePACS/APTOS
are held out, so every rate in G7 is a measurement rather than a fit.

#### The first calibration produced a useless gate

Anchoring thresholds at `p0.5 × 0.80` yielded `min_redness = 0.896` — **below all six synthetic
negatives** (which sit at ~1.00). The gate would have accepted random noise, a grey square and
the project's own charts, while reporting a successful calibration.

Two fixes:
1. Anchor each bound just below the observed training **minimum** (`min × 0.97`), not to a
   fraction of a percentile.
2. **The calibration script now refuses to write thresholds that fail to separate the known
   negatives, or that reject >2% of its own training data.** This guard is what the first
   attempt lacked — the run reported success while producing a gate that gated nothing.

Final: `separation OK — 0/6 negatives accepted, 0/400 train images rejected (0.00%)`.

---

### G3 — wired into serving, fail-closed · **PASS**
### G6 — upload hardening · **PASS**

`Registry.analyze()` runs the gate **before** routing; an ungradable image returns
`gradable: False`, a quality verdict, and findings with `status="ungradable"` carrying **no
grade, no scores, no prediction**. `/report` returns **422** — an ungradable image can never
produce a PDF, because the report is the artifact that outlives the session.

`_read_image()` now verifies the header, checks the format allow-list, enforces a 60 MP
decompression-bomb limit, and rejects images under 64 px.

---

### G4 — front-end verdict + blocked download · **PASS**

Red quality banner stating the specific reason; the download button is disabled and its click
handler additionally guards on `disabled` (state alone is not a control). Both reset on
"Analyze another".

---

### G7 — false-rejection measured on held-out cohorts · **PASS**

| cohort | n | rejected | FRR | held out from calibration? |
|---|---|---|---|---|
| **infant, full set** | **6,004** | **20** | **0.333%** | test portion yes |
| HRF healthy | 15 | 0 | 0.0% | yes |
| HRF DR | 15 | 0 | 0.0% | yes |
| HRF glaucoma | 15 | 0 | 0.0% | yes |
| non-fundus probes | 4 | 4 | 100% | negatives |

**0.333% against a 2% budget**, and the gate calibrated on neonatal RetCam images generalised to
adult HRF fundus photographs with zero false rejections — the outcome the calibration report
explicitly flagged as uncertain.

---

### Phase 0B cohort regression

```
cohort          ctx        n |            DR |           ROP |      Glaucoma   (was → now)
INFANT no-ROP   infant    22 |      14%→  0% |      32%→ 32% |      59%→  0%
INFANT ROP+     infant     7 |      14%→  0% |     100%→100% |      43%→  0%
ADULT healthy   adult     15 |       0%→  0% |      93%→  0% |       0%→  0%
ADULT DR        adult     15 |      53%→ 53% |      93%→  0% |      13%→ 13%
ADULT glaucoma  adult     15 |       0%→  0% |      87%→  0% |      87%→ 87%
NON-FUNDUS      adult      4 |     100%→  0% |     100%→  0% |     100%→  0%
```

**The original failure is closed: 100% → 0% on all three models.** In-domain behaviour is still
untouched (ROP 32%/100% on infants, DR 53%, Glaucoma 87%).

The regression script now also **fails** if non-fundus input ever produces a finding again.

#### Finding from the gate itself

**Four Phase 0A tests broke — correctly.** They used a flat colour block as a stand-in fundus
image, which the new gate rejects (zero Laplacian variance). Those tests had been passing while
silently exercising the gate rather than the router. Added `tests/conftest.py::fundus_like()` —
a textured, red-dominant, circular-FOV synthetic that passes the gate — plus a
`pytest_configure` hook that **fails the entire suite** if that fixture ever stops passing the
gate, so this cannot silently recur.

---

## Phase 0B — EXIT CRITERIA MET

| Criterion | Result |
|---|---|
| 4/4 probes rejected | ✅ plus both project charts |
| Ungradable ⇒ no prediction, no PDF | ✅ findings carry no grade/scores; `/report` → 422 |
| FRR < 2% on legitimate cohorts | ✅ **0.333%** (20/6,004 infants), 0% on 45 HRF |

```
.venv/bin/python -m pytest tests/ -q                                 ->  44 passed
.venv/bin/python tests/cohort_regression.py --cohort-root .cohorts   ->  PASS (78 images)
```

**Not done, carried forward:**
- **G8** — EyePACS/APTOS false-rejection needs a Kaggle session; **BLOCKED** locally. The adult
  FRR evidence is currently 45 HRF images from one camera, which is thin.
- **G9** — `retinal-ai/` deployment mirror **not** synced; the live HuggingFace Space still runs
  the unrouted, ungated stack. Deliberately not pushed: redeploying is an outward-facing change
  and needs your go-ahead.
- The gate's positive calibration is dominated by neonatal imaging (the only TRAIN split
  available locally). Stated in `results/gate_calibration.md` rather than hidden.

---

## Phase 1 — Honest numbers

Entirely laptop-only. Extracted the ROP test+val images (2,430) from `rop_dataset.zip` and
verified **0 missing paths** first — `FundusDataset` silently substitutes a blank tensor for an
unreadable file, so an unverified run would have produced real-looking metrics from black images.

---

### S1 — `evaluate.py --dump-preds` saves `image_path` · **PASS**
### S2 — predictions dumped for all three ROP checkpoints · **PASS**

Added MPS device selection (was CUDA-or-CPU, so every local run was CPU). Dumps now carry
`image_path`, guarded by a length check that **refuses to write misaligned paths**. Verified the
loaders use `shuffle=False` and the same split filter, so manifest order matches prediction order.

Six dumps (val+test × resnet50, efficientnet_v2_s-regularised, efficientnet_v2_s-overfit).
**ResNet50 test AUC 0.9271 reproduces the reported 0.927 exactly** — the deployed checkpoint is
the one the headline number came from.

---

### S3 — cluster bootstrap + Clopper-Pearson · **PASS**

`cluster_bootstrap_ci()` resamples **patients**, not rows. `clopper_pearson()` gives exact
binomial intervals for small denominators.

---

### S4 — patient-level ROP metrics · **PASS**

```
test split: 1502 images from 29 infants (7 with ROP, 22 without)
images per infant: median 30, max 470

AUC (image-level bootstrap)   0.9271  95% CI [0.913, 0.940]  width 0.027
AUC (patient-level bootstrap) 0.9271  95% CI [0.865, 0.967]  width 0.102  <- honest
```

**The image-level interval is 3.8x too narrow.** The honest ROP AUC is
**0.927 (95% CI 0.865–0.967)**, not 0.927 ± 0.014.

Per-infant, flagging an infant if ANY of their images is flagged:

| | value | 95% CI | n |
|---|---|---|---|
| sensitivity | 1.000 | **[0.590, 1.000]** | 7/7 infants |
| specificity | **0.273** | [0.107, 0.502] | 6/22 infants |

Two things to be precise about:
1. **Sensitivity 100% on 7 patients means almost nothing** — the interval starts at 0.59.
   Seven positive infants is the binding constraint on the entire ROP claim.
2. **Specificity 0.273 is an aggregation artefact, not a new model failure.** At per-image
   specificity 0.827 and a median of 30 images per infant, P(no false flag) ≈ 0.827³⁰ ≈ 0.004.
   "Flag on any image" is the wrong rule at this volume. Worth stating because a naive
   deployment would do exactly that.

---

### S5 — operating-threshold provenance · **PASS (adverse finding)**

```
threshold chosen on VAL  for sens>=0.90: 0.182  (val sens 0.901 / spec 0.363)
threshold chosen on TEST for sens>=0.90: 0.371  (test sens 0.900 / spec 0.827)
deployed threshold in configs/rop.yaml:  0.371   <- identical to the TEST-selected value
```

**The deployed ROP threshold was selected on the test set.** Applying the VAL-selected threshold
once to test gives sens 0.985 / **spec 0.414**. The reported specificity of 0.83 is therefore
**overstated by +0.413**.

This is the same lesson as the DR Messidor-2 result (specificity 0.89 → 0.30): operating points
do not transfer. Two independent confirmations inside one project.

**Not silently changed.** Provenance is now documented in `configs/rop.yaml`; moving the
deployed threshold changes screening behaviour and is a decision for the project owner.

---

### S6 — architecture comparison · **PASS (adverse finding)**

| comparison | DeLong | McNemar | verdict |
|---|---|---|---|
| ResNet50 vs EfficientNetV2-S (regularised) | 0.9271 vs 0.9195, p=**0.127** | p=0.068 | **NOT significant** |
| ResNet50 vs EfficientNetV2-S (overfit) | 0.9271 vs 0.9146, p=0.0083 | p<0.001 | significant |

**The headline claim "ResNet50 beats EfficientNetV2-S" is not statistically supported.** It only
beats the *overfit* variant. Against the properly-regularised one the difference is 0.008 AUC,
p=0.127.

The architecture sweep reported 0.9635 vs 0.870 — a large gap — but that was 320px/12 epochs.
At full training the two are 0.9271 vs 0.9195. The sweep was not representative of the final models.

Recorded caveat: DeLong assumes independent samples. With 1,502 images from 29 infants these
p-values are **anti-conservative**. That makes the non-significant result *safer* to trust and the
significant one weaker — not the other way round.

---

### S7 — headline metrics with patient-level intervals · **PASS**

| metric | point | 95% CI (patient-level) |
|---|---|---|
| AUC | 0.9271 | [0.865, 0.967] |
| accuracy | 0.8675 | [0.790, 0.914] |
| macro F1 | 0.8656 | [0.749, 0.908] |

Outputs: `results/rop_statistics.json`, `results/rop_statistics.md`.

---

### S8 / S9 — DR train/serve config resolution · **BLOCKED**

Requires the DR val set (EyePACS + APTOS, 5,815 images), which exists only on Kaggle. The
ambiguity stands: `results/dr/config.yaml` records preprocessing `false`, `configs/dr.yaml`
serves `true`, and spot-testing showed the recorded config performs *worse*. **Do not apply the
"obvious" registry fix until this is run.**

---

## Phase 1 — EXIT CRITERIA (partial)

| Criterion | Result |
|---|---|
| ROP reported at patient level with CI | ✅ 0.927 [0.865, 0.967] |
| Significance tests actually run | ✅ DeLong + McNemar, both comparisons |
| CIs on headline metrics | ✅ AUC / accuracy / macro-F1 |
| Threshold provenance documented | ✅ config comment + measured penalty |
| DR config ambiguity resolved | ❌ **BLOCKED — needs Kaggle** |

**Three claims in the deck now need revising** (deck not yet edited):
1. ROP AUC needs its patient-level interval, and "29 infants, 7 positive" stated.
2. "ResNet50 beats EfficientNetV2-S" → "scored higher; not statistically significant (p=0.127)".
3. ROP "sens 0.90 / spec 0.83" is a test-selected operating point, not a held-out estimate.

---

## Phase 1 — self-review upgrades

Reviewing Phase 1 before moving on found five gaps. All five are now closed. Two changed
conclusions materially.

---

### U1 — the ROP test set is dominated by one infant · **new finding**

```
largest infant : 470/1502 images = 31.3% of the test set (ROP-positive)
top 3 infants  : 45.4% of the test set
image-level AUC   0.9271   (each IMAGE counts once)
patient-weighted  0.8989   (each INFANT counts once)
leave-one-infant-out AUC ranges 0.9193 - 0.9457 (swing 0.0264 from removing ONE infant)
```

S4 measured the *width* of the uncertainty but not its *source*. **A single infant is 31% of the
test set**, and weighting each infant equally drops AUC from 0.927 to **0.899**. Removing one
infant swings AUC by 0.026 — larger than the entire gap to EfficientNetV2-S.

The defensible ROP statement is now: **AUC 0.899 patient-weighted / 0.927 image-weighted,
95% CI 0.865–0.967, on 29 infants of whom 7 have ROP.**

---

### U2 — the architecture comparison reverses under a patient-level test · **conclusion changed**

```
                                     DeLong (image)      cluster bootstrap (infant)
ResNet50 - EffNetV2-S (regularised)  p=0.127  n.s.       dAUC +0.013 [-0.018, +0.057]  n.s.
ResNet50 - EffNetV2-S (overfit)      p=0.008  SIG        dAUC +0.016 [-0.019, +0.066]  n.s.
```

Phase 1 caveated that DeLong is anti-conservative on clustered data. The upgrade **did the test
instead of caveating it** — and the one "significant" result disappears: both CIs span zero.

**Neither architecture comparison is statistically supported.** The ResNet50 choice is not
justified by this test set at all — not merely "underpowered against the regularised variant".

---

### U3 — DR intervals (S7 was ROP-only) · **closed**

| metric | point | 95% CI |
|---|---|---|
| QWK (5-class) | 0.7281 | [0.709, 0.746] |
| accuracy | 0.7443 | [0.733, 0.756] |
| macro F1 | 0.5864 | [0.566, 0.609] |
| AUC (5-class ovr) | 0.7884 | [0.769, 0.807] |
| **referable AUC (grade≥2)** | **0.9060** | **[0.893, 0.919]** |

All reproduce the reported values exactly. Stated limitation: `val_preds.npz` predates S1 and has
no `image_path`, so these are **image-level**; EyePACS contributes two images per patient
(left/right eye), so true patient-level intervals are wider. **Glaucoma has no prediction dump at
all** and cannot be given intervals without a Kaggle re-run.

---

### U4 — duplicate audit · **serious data-integrity finding, correctly scoped**

101 phash-duplicate groups across the 6,004 ROP images. **Verified by pixel comparison, not
trusted from the hash** — phash collides readily on fundus images, which all share a bright disc
on black. The flagged pairs are `MSE = 0.0, corr = 1.0000`: byte-identical files.

```
duplicate groups          : 101
  cross-split             :  19
  INVOLVING TEST          :   0     <- headline test AUC is NOT inflated
  contradictory labels    :  19
label-conflicting images  : 20 train, 19 val
```

Example: `<idA>_M_GA##_BW####_PA##_DG#_PF#_...` (train, patient 041, **no ROP**) is byte-identical to
`<idB>_M_GA##_BW####_PA##_DG#_PF#_...` (val, patient 040, **ROP**). Same sex, gestational age, birth
weight, session and frame index — **two patient IDs for the same baby, with opposite labels.**
14 demographic signatures (sex, GA, BW) are shared by two patient IDs each.

**Impact, precisely:**
- **Test is clean — 0/1502 leaked.** The headline AUC 0.927 is unaffected by this.
- **Val is contaminated** (19/928 ≈ 2%), and val drives early stopping *and* the S5 threshold
  selection. The val-derived threshold 0.182 rests on slightly tainted data.
- **The patient-split claim needs qualifying.** The split is by patient *ID*, and up to 14 infants
  appear under two IDs. "No infant appears in two splits" is not strictly true as stated.

---

### U5 — the gate's blur/exposure verdicts were never exercised · **closed**

G1 only ever verified that synthetic probes return `not_fundus`. `too_blurry` and
`poor_exposure` were untested — reachable only by degrading a *real* fundus image. Added three
tests: heavy Gaussian blur → `too_blurry`; severe underexposure → rejected; **mild blur still
accepted** (the gate must not become a quality connoisseur).

```
.venv/bin/python -m pytest tests/ -q  ->  47 passed
```

---

### Assessment

Phase 1 as originally delivered was **not** sufficient. It produced correct numbers but stopped at
the first honest answer in three places: it measured interval width without finding that one
infant is 31% of the test set; it caveated DeLong instead of running the patient-level test that
reverses its conclusion; and it covered only ROP. The duplicate audit — scheduled for Phase 3 —
turned out to be cheap once the data was local and found a real defect in the project's
headline methodological claim.

**Carried forward:** the val contamination should be removed and the ROP threshold re-derived on
a cleaned val split. Not done here — it changes screening behaviour and needs an explicit decision.

## Phase 2 — Calibration, abstention and honest confidence

**Dependency handled first:** Phase 2 fits calibration on val, and U4 found val contaminated.
The 19 cross-split duplicate images are removed before fitting. This is not a behaviour change,
so it needed no decision — calibrating against known-bad labels would simply be wrong.

---

### C1/C2/C3 — calibration fitted against the REAL serving pipeline · **PASS (conclusion reversed)**

**The existing constants were not usable.** `results/<disease>/calibration/` holds temperatures
fitted on **single-view logits** (DR 5-class T=0.78, referable T=0.70, ROP). But serving runs
`predict_with_tta`, which averages **probabilities** over 5 views. A temperature fitted on
single-view logits does not apply to an averaged probability vector — wiring the old constants
into the app, as the roadmap intended, would have produced confident-looking but wrong calibration.

**Method.** Serving emits an averaged probability vector `p`; calibrate with
`p_cal = softmax(log(p) / T)` — equivalently `p^(1/T)` renormalised. Monotone in every
component, so it **cannot change any prediction**, only the confidence. Verified empirically:
`argmax decisions changed: 0/909`.

**Result for ROP** (`scripts/fit_serving_calibration.py`, 909 de-duplicated val images):

```
temperature T          1.0793
ECE   0.0555 -> 0.0557   (WORSE)
MCE   0.1044 -> 0.0853
Brier 0.3366 -> 0.3359
argmax changed          0/909
threshold@0.371 changed 6/909
DECISION: DO NOT APPLY
```

**Temperature scaling does essentially nothing for ROP under the real serving pipeline.**
T ≈ 1.08 and ECE moves −0.3%. The reason is straightforward once measured: **5-view TTA
averaging is itself an ensemble, and ensembling already calibrates.** The offline value was
solving a problem serving does not have.

Applying it anyway would shift the probability scale and invalidate the threshold tuned on the
raw scale — 6 decisions flip for no measurable gain. The script now writes an explicit
`apply` flag with a reason, and **refuses to apply a calibration that does not improve ECE by
≥10% relative**.

> The roadmap item "wire temperature scaling into serving" was based on numbers that do not
> hold at serving. The honest outcome is *not* to wire it in for ROP, and to say why.

**DR and Glaucoma: BLOCKED.** Their val images (EyePACS/APTOS, SMDG-19) are Kaggle-only, so
serving-pipeline calibration cannot be measured locally. They are reported as **`unverified`**,
which the UI shows distinctly from `calibrated`.

---

### C4 — calibration status carried through to the UI · **PASS**

Three distinct states, never conflated:

| status | meaning | shown as |
|---|---|---|
| `calibrated` | measured and applied | green "calibrated" |
| `verified-uncalibrated` | measured, deliberately not applied (ROP) | amber "uncalibrated (measured)" |
| `unverified` | never measured at serving (DR, Glaucoma) | red "calibration unverified" |

A missing constant degrades **visibly**; it never silently falls back to pretending calibration
happened.

---

### C5 — ordinal head states the referable endpoint · **PASS**

DR findings now carry `referable: {grade_threshold, probability, is_referable}`. The clinically
meaningful endpoint — and the one the 0.906 AUC actually measures — is no longer left implied
behind a 5-class argmax.

---

### C6 — abstention band, derived on val · **PASS**

```
half-width          band     abstained  acc inside  acc outside    gain
      0.02  [0.351, 0.391]      4.5%      0.4634       0.7834  +0.0144
      0.04  [0.331, 0.411]      8.1%      0.4459       0.7976  +0.0286
      0.06  [0.311, 0.431]     11.1%      0.4851       0.8045  +0.0355
      0.08  [0.291, 0.451]     14.2%      0.5116       0.8115  +0.0426

SELECTED [0.291, 0.451] — abstains 14.2%, retained accuracy 0.8115 (+0.0426)
```

**Inside the band accuracy is 0.512 — a coin flip. Outside it is 0.812.** The band isolates
exactly the cases a human should see. Measured abstention on the real infant cohort: **18.2%**.

The script **refuses to enable a band** that abstains >15%, or whose retained-accuracy gain is
below 0.01 — a band that refuses cases it would have got right is pure cost.

---

### C8 — "confidence" retired · **PASS**

The API key is now `score`, updated across `app.py`, `main.js`, `history.js`,
`gallery.html`, `history.html` and the PDF. A test asserts no source surface labels a raw
probability "confidence". The About page claim *"Returns a prediction, calibrated confidence"*
and the testimonial *"Calibrated confidence and one-click PDF reports"* were both false and are
corrected.

---

## Phase 2 — EXIT CRITERIA

| Criterion | Result |
|---|---|
| ECE reported under the true serving pipeline | ✅ ROP 0.0555 (TTA-averaged, de-duplicated val) |
| An abstention path exists | ✅ band [0.291, 0.451], 18.2% on real cohort |
| No UI surface overstates certainty | ✅ 3-state calibration badge; "confidence" retired |
| Calibration cannot change a prediction | ✅ asserted, 0/909 argmax changes |

```
.venv/bin/python -m pytest tests/ -q                                 ->  60 passed
.venv/bin/python tests/cohort_regression.py --cohort-root .cohorts   ->  PASS (78 images)
```

**Carried forward:** DR and Glaucoma serving calibration (Kaggle); the ROP abstention band is
derived from a val split whose threshold provenance is still the S5 open question.

## Deck update — Phases 1 & 2 folded in

`generate_ppt.py` regenerated → **11 slides** (was 10). Rendered and visually verified via
`scripts/preview_ppt.py`.

**Corrected claims:**

| Slide | Was | Now |
|---|---|---|
| 05 Results | ROP "Sens/Spec @ thr 0.371 · 0.90 / 0.83" | "95% CI (per-patient) 0.865–0.967", "Patient-weighted AUC 0.899", "Operating point: **test-selected ⚠**" |
| 05 Results | DR "95% CI 0.89 – 0.92" | "0.893 – 0.919" (measured, `scripts/dr_intervals.py`) |
| 05 Results | Glaucoma "Success tier Stretch ★" | "Held-out source: **none yet ⚠**" |
| 07 Rigor | "ResNet50 beat EfficientNetV2-S — AUC 0.964 vs 0.87 … chosen objectively" | "At FULL training the gap nearly vanishes: 0.9271 vs 0.9195 · DeLong p=0.127 · ΔAUC CI [−0.018, +0.057] → **not statistically significant**" |
| 08 Trustworthiness | "temperature scaling lowers ECE on all three models … not yet wired in" | "Offline temperatures were fitted on single-view logits and never applied. Refitted on the real pipeline: ROP T=1.08, ECE 0.0555→0.0557 — TTA averaging already calibrates, so scaling is deliberately **not** applied" |
| 11 Summary | limitation "Calibration not wired into live serving" | "ROP threshold was selected on the test set" |
| 11 Summary | roadmap "Wire temperature scaling into serving" | "Resolve DR train/serve config ambiguity" |

**New slide 10 — "08 · SAFETY: We attacked our own system — and it failed."** Left: the four
measured failures. Right: the 6-cohort before/after table (93%→0%, 59%→0%, 100%→0%) plus the
statistics correction. Bottom-left: routing, gate (FRR 0.33% on 6,004 images) and abstention.

**Bug fixed while there:** the footer read a hard-coded "/ 10" on every slide, so adding a slide
silently desynced it. Replaced with an `N_SLIDES` constant plus a check that it matches the
`steps` list length.

---

## Phase 3 — T8 device-confound audit · **CRITICAL FINDING**

The question: does the ROP model read the **disease**, or the **device**? ROP filenames encode
`D#`, so capture device is recoverable with no extra metadata.

### 1. The confound is strong

| device | images | ROP+ images | prev | patients | ROP+ patients | prev |
|---|---|---|---|---|---|---|
| D1 | 2,578 | 1,925 | 74.7% | 81 | 39 | **48.1%** |
| D2 | 2,452 | 741 | 30.2% | 61 | 13 | 21.3% |
| D3 | 974 | 358 | 36.8% | 46 | 8 | **17.4%** |

Patient-level prevalence spread across devices: **30.8%**. Device carries substantial
information about the label before the model sees a single pixel.

### 2. Device is unevenly distributed across splits

```
split      D1     D2     D3
test    0.629  0.321  0.050
train   0.363  0.402  0.235
val     0.363  0.573  0.064
```

**Test is 62.9% D1 against 36.3% in train** — the test split is enriched in the
highest-prevalence device.

### 3. Every ROP-positive test image comes from ONE device

```
device  images  ROP+     AUC
D1         945   824   0.8808
D2         482     0     n/a   (single-class — all ROP-negative)
D3          75     0     n/a   (single-class — all ROP-negative)

devices in test: 3   measurable (both classes present): 1   ROP+ present on: ['D1']

overall AUC (all devices pooled)  0.9271
within-device AUC                 0.8808   (-0.0462)
```

**All 824 positive test images are D1. All 557 D2/D3 test images are negative.** So a
meaningful part of the pooled AUC is an easy device-vs-device separation that has nothing to do
with disease. Discrimination is only *measurable* inside D1.

**The defensible, device-controlled ROP AUC is 0.881 — not 0.927.**

Combined with the earlier patient-weighting result (0.899), the honest ROP figure sits around
**0.88–0.90**, not the headline 0.927.

#### Self-review note

My first version of this script concluded *"the signal largely survives stratification by
device"* because it only compared the mean within-device AUC against overall and applied a
−0.05 threshold. That verdict was **too generous and structurally wrong**: it ignored that only
**one** device admits a measurement at all. A −0.046 drop looks tolerable; "the positive class
exists on exactly one device" is a different and much more serious statement. The verdict logic
now checks how many devices contain both classes and reports the structural finding first.

### What this does NOT mean

The model is not proven to be reading the camera — within D1, where device is held constant, it
still achieves 0.881, which is genuine discrimination. The claim is narrower and precise: **the
pooled 0.927 is inflated, and the test split cannot separate device effects from disease
effects.** Fixing that needs a device-stratified re-split and retrain (task T9, gated on this
audit — now justified).

---

## Operating point re-derived + deployment (G9)

### Val cleaned, threshold re-derived · **PASS (adverse)**

`scripts/rederive_rop_threshold.py`. 39 cross-split duplicate images removed from **all**
splits (clean val 909, test unchanged at 1502 — confirming test was never contaminated).
Both splits scored through the real 5-view TTA path.

```
threshold from clean VAL @ sens>=0.90 : 0.1933   (val sens 0.901 / spec 0.350)
applied ONCE to TEST:
    sensitivity 0.990   95% CI [0.981, 0.996]
    specificity 0.398   95% CI [0.361, 0.436]
previously deployed 0.371 (selected ON test): sens 0.916 / spec 0.791
```

**`configs/rop.yaml` updated 0.371 → 0.1933.** Specificity of 0.40 is genuinely poor — ~60% of
healthy infants get flagged, visible in the cohort regression as ROP on infant no-ROP moving
32% → 68%. That is the honest cost of an unbiased operating point on this dataset. It is a
**data** problem (29 test infants, 7 positive, every positive on one camera), not something to
fix by tuning on test again.

The abstention band was re-derived around the new threshold: **[0.173, 0.213]**, abstains 6.8%,
accuracy inside 0.564 vs 0.731 outside.

### Deployment sync · **PASS — caught a second train/serve skew first**

The mirror shipped **`tta.enabled: false` for all three models** (a CPU-speed optimisation),
while every metric, the calibration, the abstention band and the new threshold were all derived
**with** 5-view TTA. Blindly syncing would have deployed TTA-derived constants into a non-TTA app.

Measured the divergence rather than guessing:

```
AUC       with TTA 0.9303    without TTA 0.9271
score correlation 0.9948     mean |diff| 0.0237
thr 0.1933  with TTA sens 0.990/spec 0.398   without sens 0.982/spec 0.456
decisions that flip: 56/1502 = 3.7%
```

The AUC barely moves — but **the abstention band half-width is 0.0200, smaller than the
0.0237 score shift**, so which cases land in the band would have been essentially arbitrary
without TTA. **TTA re-enabled in the mirror** so the deployment matches what was measured.
Routing offsets most of the cost (only the routed models run).

Verified in the mirror before pushing: boots with 3 contexts, 5 TTA views per model, correct
calibration statuses, all 4 probes rejected, routing correct in both directions, 2.2 s per adult
screening on MPS.

**Pushed** `61074a4..ce0ad8d` → `huggingface.co/spaces/Champ610/retinal-ai`. Main repo committed
locally at `2d17ed8` (not pushed — not requested).

### Live Space verified in production · **PASS**

`https://champ610-retinal-ai.hf.space` rebuilt and serving the new stack.

| check | result |
|---|---|
| `POST /predict` with no patient context | **400** "Patient context is required" |
| non-fundus image, adult context | 200, `gradable: false` — no findings |
| `POST /report` on a non-fundus image | **422** — no PDF can be issued |
| real fundus, adult context | 200 · ROP `not_applicable` · DR runs · badge `unverified` |
| `GET /contexts` | serves adult / infant / demo (demo flagged advisory) |

Reproducibility spot-check: the live DR score for the same input was 53.0%, against 58.5%
locally — traced to the test client resizing the image to 800×800. Re-scoring the resized image
locally gives **53.0%, an exact match**. The deployment reproduces local behaviour.

**The original failure is closed in production**: the four probes that previously scored positive
on all three diseases are now rejected before any model runs.

## Phase 4 (partial) + T10 — done while the Kaggle kernels ran

### T10 — external-validation output collisions · **PASS**

`external_validation.py` keyed every output on `cfg.disease` alone, so evaluating glaucoma on
G1020 and then on REFUGE would **silently destroy the first report**. This blocked Phase 3
before it could start.

Now keyed `disease_source` (source derived from the config filename, overridable with
`--source`), and it **refuses to overwrite an existing report** without `--force`.

Also closed the binary-reporting gap: binary diseases (glaucoma, ROP) previously got a bare
AUC with no operating point and no interval — only multiclass got the referable block. They now
report AUC + bootstrap CI, the threshold at target sensitivity, and — when the config declares
one — sensitivity/specificity at the **deployed** threshold with Clopper-Pearson intervals.
A bare AUC hides the thing that actually breaks across populations: the decision threshold.

Also fixed `build_from_cfg(cfg)` → `pretrained=False` (was downloading ImageNet weights only to
overwrite them) and added MPS device selection.

### T4 — config-parity guard · **PASS (DR correctly flagged)**

`tests/test_repo_guards.py` asserts every served config matches the config its checkpoint
recorded. **This is the guard that would have caught the DR skew.**

DR is registered as a *tracked* mismatch (`xfail`) rather than ignored — and a second test
asserts every tracked entry names the task resolving it, so an exception cannot quietly become
permanent. ROP/Glaucoma skip (their runs recorded no config).

### T3 — PHI guard · **PASS**

Asserts `dataset_split/`, `rop_dataset.zip` and `.cohorts/` are gitignored, that no patient
image is tracked, and that no Kaggle credential file is tracked.

### `.gitignore` — the guard was decorative until this was fixed

`results/` was ignored wholesale, so `results/dr/config.yaml`, `gate_thresholds.json`,
`abstention_band.json` and `calibration_serving.json` were **not in the repo at all**. Two
consequences: the config-parity guard would have **silently skipped in CI** (the same
"passes because it tested nothing" failure mode found in R6 and the gate calibration), and a
fresh clone would run the gate on defaults with no abstention.

Now versions the small provenance/serving JSON while still excluding `*.pth`, `*.npz`, `*.png`,
`*.pdf`. Verified both directions with `git check-ignore`.

### T6 — CI · **PASS**

`.github/workflows/ci.yml` runs the guards plus the routing/gate contract on CPU torch.
Weight- and cohort-dependent tests skip by design — so CI **explicitly fails if the guards
themselves were skipped**, because a suite that skipped everything is indistinguishable from a
passing one.

### Root `Dockerfile` — long-standing footgun closed

It built `unified_model/app.py`, the superseded **v1 single-model** app. Anyone building from
the repo root got the old product while believing they had the current one. It now builds the
same three-model application as `retinal-ai/Dockerfile`.

### History page

`history.html` / `history.js` never handled the `available` / `status` fields added in Phase 0A,
so a routed-out or ungradable finding would have rendered as a blank result. Both now show
"Not assessed" / "Not graded" explicitly.

```
.venv/bin/python -m pytest tests/ -q  ->  69 passed, 2 skipped, 1 xfailed
```

## Kaggle round 1 — results

### S8 — DR train/serve config ambiguity · **RESOLVED, CONCLUSIVE**

Same checkpoint, same val split (5,815 images), scored under both preprocessing configurations:

| configuration | QWK | accuracy | AUC |
|---|---|---|---|
| **`configs/dr.yaml`** — what the app serves (preprocessing **ON**) | **0.7281019162522491** | **0.7442820292347377** | **0.7884** |
| `results/dr/config.yaml` — what the run recorded (**OFF**) | 0.2037 | 0.5193 | 0.4855 |
| *recorded metrics to match* | *0.7281019162522491* | *0.7442820292347377* | *0.7884* |

**Exact, bit-for-bit reproduction.** Delta 0.0.

**Verdict: the DR checkpoint was trained WITH preprocessing. There is no DR train/serve skew —
`configs/dr.yaml` was correct all along, and the live app has been serving correctly.** The
mis-recorded file was `results/dr/config.yaml`.

**The Phase-1 warning was right.** Applying the "obvious" fix — pointing the registry at
`results/dr/config.yaml` — would have collapsed DR from QWK 0.728 to **0.204**, with AUC 0.485
(worse than chance ranking). That fix looked correct for weeks and would have silently destroyed
the model.

Actions: `results/dr/config.yaml` corrected (original preserved as `.as-recorded-WRONG`);
`config_verdict_s8.json` saved; the tracked exception removed from
`KNOWN_CONFIG_MISMATCHES`, so the parity guard now passes on its own merits.

**Root cause undetermined.** `config.yaml`, `metrics.json` and `weights.pth` share one download
timestamp, so the logger did record a config that does not describe its own weights. The Kaggle
run history is not available locally, so I am not guessing at why. The parity guard now catches
recurrence.

### G8 — gate false-rejection on adult data · **PASS**

| source | n | FRR | verdicts |
|---|---|---|---|
| EyePACS | 4,000 | **1.73%** | 61 not_fundus, 8 poor_exposure |
| APTOS | 3,662 | **0.25%** | 6 not_fundus, 3 too_blurry |
| **overall** | **7,662** | **1.02%** | budget 2% — **PASS** |

The gate was calibrated on neonatal RetCam images only; this is the first real adult evidence
(previously 45 HRF images from one camera). EyePACS is 5× higher than the infant rate (0.33%),
which is expected — it is genuinely messy real-world screening data, and some of those 69
rejections are very likely correct.

### Glaucoma calibration · **INVALID — wrong checkpoint, re-running**

The first run reported T=0.4764, ECE 0.0958 → 0.0358 (62.6% better) and per-source
**G1020 AUC 0.566** against a claimed test AUC of 0.957.

That 0.566 is the tell: it matches the **documented SMDG-only zero-shot G1020 failure (0.589)**.
Checked the timeline:

```
retinal-glaucoma-weights uploaded    2026-06-29 09:30:47
retinal-train-glaucoma-ms last ran   2026-06-29 10:43:16   <- 73 min LATER
```

The uploaded `weights.pth` **predates the multi-source training run** — it is the SMDG-only
model. Local md5s confirm the two checkpoints differ. The kernel globbed `*.pth` and silently
loaded the wrong one, producing a plausible-looking calibration for a model that is not deployed.

Fixed: uploaded both checkpoints under unambiguous names (`weights_multisource.pth`,
`weights_smdg_only.pth`) and hardened the kernel to load by **exact name with an assert**,
never a glob. Re-running.

**The result is not discarded, it is reinterpreted**: it independently reproduces the original
domain-shift finding on a fresh split.

### Glaucoma serving calibration — re-run with the CORRECT checkpoint · **PASS**

Kernel now loads `weights_multisource.pth` by exact name (asserted, never globbed).

```
T = 1.0733    n = 1,333 val images
ECE   0.0170 -> 0.0126     MCE 0.2482 -> 0.2651 (WORSE)
Brier 0.1234 -> 0.1232     argmax changed 0/1333
per-source:  SMDG  n=1214  AUC 0.9721
             G1020 n= 119  AUC 0.8603
```

**The multi-source fix is real.** G1020 recovers from **0.566** (SMDG-only checkpoint, matching
its documented 0.589 zero-shot failure) to **0.860** with the deployed model. Worth noting
honestly: the val figure 0.860 sits below the reported per-source *test* figure of 0.957, on
only 119 images with 30 positives — a wide interval, but not a number to quote as 0.957 without
saying which split it came from.

#### The apply-rule was wrong, and it fired

The kernel reported `apply=True` on a **26.1% relative** ECE improvement. That is noise dressed
as a result:

| | ECE before | after | absolute gain | relative | MCE |
|---|---|---|---|---|---|
| ROP | 0.0555 | 0.0557 | −0.0002 | −0.3% | improves |
| Glaucoma | 0.0170 | 0.0126 | **0.0044** | 26.1% | **0.2482 → 0.2651 WORSE** |

A 26% relative gain on an ECE of 0.017 is 0.4 percentage points — while tail calibration (MCE)
*degrades*, and T=1.0733 is within noise of 1.0. My original rule tested relative improvement
only, so it would have wired a pointless transform into serving and made the worst-case
calibration worse.

Rule corrected to require **relative ≥10% AND absolute ≥0.01 AND non-degrading MCE**. Re-decided
from the stored numbers: **glaucoma `apply=False`**.

#### Consistent conclusion across both measured models

ROP T=1.079, Glaucoma T=1.073 — both within noise of 1.0, neither materially improved.
**5-view TTA averaging already calibrates these models; post-hoc temperature scaling adds
nothing at serving.** Glaucoma moves from `unverified` to `verified-uncalibrated`: measured,
and deliberately not applied. Only DR remains `unverified` (its serving-path TTA calibration
was not dumped by the S8 kernel — still open).

## T9 — device-stratified ROP split

### The split (`scripts/build_device_stratified_split.py`)

Three defects in the original split, fixed together:

1. **19 duplicate groups with contradictory labels** — byte-identical pixels filed as both
   DG0 and DG1. We cannot know which label is right, so all 39 images are dropped rather
   than guessed at.
2. **14 demographic signatures (sex+GA+BW) shared by two patient IDs** — the same infant
   under two IDs. Merged into one unit so it can never straddle a split.
3. **Device confounded with label.** Stratified by (device, label) at unit level.

```
174 patient-units, 59 ROP-positive
train  4173 images  120 units  prev 0.525  devices [1,2,3]
val     851 images   27 units  prev 0.436  devices [1,2,3]
test    941 images   27 units  prev 0.471  devices [1,2,3]

test D1: n=482 pos=363   test D2: n=338 pos=54   test D3: n=121 pos=26
no patient-unit straddles a split
```

**Every device now carries both classes in val and test.** The old split had every
ROP-positive test image on D1, so discrimination was measurable on exactly one device.

The builder **refuses to write a split that fails these checks** — a test split that lost
both-class coverage on a device would silently reproduce the confound it exists to remove.

### Kernel v1 failed — my bug, not the split

The split built and validated **identically to local** inside the kernel, then the script
exited 1 writing `results/rop/devsplit_summary.json`: the code bundle ships no `results/rop/`
directory, so `write_text` raised `FileNotFoundError` *after* the valid manifest had already
been written. Added `mkdir(parents=True, exist_ok=True)` for both outputs.

A reminder that a script validated only in the repo it was written in can still fail on a
first-run environment for reasons that have nothing to do with its logic.

### T9 result — the device confound was real, and larger than expected

Kernel v2 completed cleanly (early stop at epoch 11, best val recall 0.8338).

```
                     kernel (no TTA)    serving path (5-view TTA, T9b)
D1                       0.8171                 0.8120
D2                       0.6451                 0.6270
D3                       0.6312                 0.6138
pooled                   0.8775                 0.8756
device-controlled        0.6978                 0.6843
```

The kernel evaluated through `evaluate()` (single view); T9b re-scored the same checkpoint
through `predict_with_tta`, which is what the app actually runs. The serving numbers are
uniformly ~0.005–0.02 lower and are the ones to quote.

**Device-controlled AUC falls 0.8808 -> 0.6843.** The gap between pooled (0.8756) and
device-controlled (0.6843) is **+0.19** — that is the share of the pooled figure that was
device separation rather than disease discrimination.

### T9b — why the point estimates must not be quoted alone

Resampling patient-units, not images (`scripts/rop_devsplit_ci.py`):

```
device  images  ROP+ img  units  +units   AUC     95% CI            width
D1        482       363     13       6   0.8120  [0.6568, 0.9350]   0.278
D2        338        54      9       2   0.6270  [0.3266, 0.8206]   0.494
D3        121        26      7       1   0.6138  [0.5237, 0.6977]   0.174

pooled                                   0.8756  [0.7128, 0.9537]
device-controlled                        0.6843  [0.5635, 0.7745]   (841/2000 draws
                                                                     degenerate)
```

Three things this says that the point estimates do not:

1. **D2 cannot be distinguished from chance.** Its interval [0.327, 0.821] contains 0.5.
   Two positive infants cannot support any claim about that camera.
2. **D3's narrow interval is an artefact, not evidence.** 0.174 wide is *narrower* than D2's
   despite having FEWER positive infants — because with exactly one positive unit, every
   non-degenerate bootstrap draw contains that same infant, so the resampling cannot vary
   the positive side at all. A cluster bootstrap with one cluster in a class estimates
   nothing about that class. **The D3 interval must not be reported as if it were tight.**
3. **42% of draws (841/2000) lost a class entirely.** That degeneracy rate is itself the
   headline: the split is too thin for the question being asked of it.

**Defensible statement.** On D1 — 13 infants, 6 ROP-positive — the model achieves AUC
**0.812 [0.657, 0.935]** through the serving path. For D2 and D3 the dataset does not
contain enough positive infants to establish whether performance transfers. The correct
conclusion is *not* "the model fails on other cameras"; it is **"this dataset cannot tell
you, and the previous 0.927 concealed that it couldn't."**

Verdict: **PASS** — the experiment answered its question, and the answer is unfavourable.
The deployed checkpoint is untouched (T9 was additive); `results/rop/weights_devsplit.pth`
is the experimental model and is not served.

## C2/C3 (DR) — serving calibration, and a defect it exposed

Kernel completed cleanly: 5,815 val images x 5 TTA views, ~2.5 s/image (13,859 s for 5,500).

```
                     T        ECE            MCE            Brier       verdict
DR 5-class        0.6775  0.1909 -> 0.0932  0.2590 -> 0.3532  0.4350 -> 0.3931  DO NOT APPLY
DR referable      0.5942  0.1366 -> 0.0824  0.2083 -> 0.1276  0.2874 -> 0.2570  APPLY
```
`argmax_changed 0/5815` on both. Referable AUC through the serving path: **0.9124**.

**My stated expectation was wrong.** I predicted T ~ 1.0 and a do-not-apply, by analogy with
ROP (T=1.079) and Glaucoma (T=1.073). DR came back at **T=0.68** — the model is materially
**under**-confident, and this is the first time the apply-rule has fired true (on the
referable endpoint). Under-confidence is consistent with the training recipe: label
smoothing 0.1, focal loss, class-balanced weighting and 5-view probability averaging all
push mass toward the middle of the simplex, and they compound.

The 5-class verdict stays DO NOT APPLY: ECE improves hugely (0.191 -> 0.093) but **MCE gets
worse** (0.259 -> 0.353). Trading worst-case calibration for average calibration is the
wrong trade for a screening tool, where the tail is the part that hurts.

### The defect: two opposite reasons were reporting the same status

| model | ECE at serving | apply | status BEFORE |
|---|---|---|---|
| Glaucoma | 0.017 | false | `verified-uncalibrated` |
| ROP | 0.056 | false | `verified-uncalibrated` |
| DR | **0.191** | false | `verified-uncalibrated` |

All three are uncorrected — but Glaucoma is uncorrected because it was **already fine**, and
DR because the only correction available **made something else worse**. The same badge read
as reassurance in one case and should have read as a warning in the other. A UI keying on
that string could not tell an 0.017 model from an 0.191 one.

Added a fourth state, `measured-miscalibrated` (boundary ECE >= 0.10 — a convention, stated
as such, with the measured ECE carried alongside so nothing has to trust the bucket), plus
an explicit `warning` and the `referable` sub-result, which calibrates differently from the
5-class head and is the endpoint that actually matters clinically. `about.html` corrected —
it still claimed DR and Glaucoma were `unverified`.

Guard added: `test_a_badly_calibrated_model_is_not_labelled_like_a_good_one` asserts the
status matches the measured ECE and that a miscalibrated model carries a warning. 71 tests.

**Left open (needs a decision, not more measurement):** the referable temperature T=0.5942 is
a real 40% ECE improvement with MCE also improving, and it is NOT wired into serving.
Applying it is not a one-line change — `_apply_calibration` operates on the 5-class vector,
whereas this temperature was fitted on the collapsed 2-column referable vector, so it needs
its own path; and the abstention band was derived from the current score distribution, so it
would have to be re-derived. Deliberately not done silently.

## T11a — the glaucoma "blocker" was a wrong assumption

Recorded blocker: *"no usable REFUGE source found"*, which assumed a held-out source had to
come from outside. It did not. SMDG-19 is not one dataset — it is a **meta-dataset of 15
constituent collections**, and its `names` column is prefixed with the originating one.

```
source     total  labelled  glaucoma  non_glaucoma
OIA         4494      4442       291          4151     prevalence 0.066
EyePACS     3269      3269      3269             0     prevalence 1.000  <-- no negatives
G1020       1020      1020       296           724
REFUGE1      800       800        80           720     <-- REFUGE was here all along
ORIGA        650       650       168           482
BEH          634       634       171           463
PAPILA       488       420        87           333
sjchoi86     401       401       101           300
FIVES        400       400       150           250
DRISHTI      101       101        70            31
CRFO          79        79        48            31
JSIEC         51        38         0            38
HRF           30        30        15            15
LES           22        22        11            11
DR            10        10        10             0
```

Ten collections carry >=30 of both classes, so the held-out-source experiment is possible
with data already in hand.

**But the probe found something larger than it went looking for.** `EyePACS` contributes
**3,269 of the 4,767 glaucoma positives (68.6%) and contains zero negatives**, while `OIA`
contributes 4,151 negatives at 6.6% prevalence. Those two collections are 63% of the corpus
and sit at opposite ends of prevalence.

That is the ROP device confound one level up: a model can score well by learning *"this looks
like an EyePACS capture"* rather than *"this optic disc is glaucomatous"*, and a pooled AUC
cannot tell the two apart. **The 0.973 headline is now under the same suspicion 0.927 was.**
T11b (`kaggle_kernels/glaucoma_source_audit.py`) measures AUC within each source to find out;
it does not retrain, so it answers "how bad" before spending a training run on T12.

## T14 — ROP subgroup performance: the dataset cannot answer the question

Every ROP filename carries sex, gestational age, birth weight and postnatal age, and nobody
had looked. Run on the device-stratified split and checkpoint, so a subgroup gap cannot be a
device gap in disguise.

```
subgroup                          img  inf  +inf   prev     AUC   95% CI
F                                 306   12     1  0.225       —   UNMEASURABLE
M                                 635   15     8  0.589  0.8955   [0.705, 0.970]
GA < 28 wk (extremely preterm)    266    5     2  0.797       —   UNMEASURABLE
GA >= 28 wk                       675   22     7  0.342  0.8394   [0.621, 0.960]
BW < 1000 g (ELBW)                266    5     2  0.797       —   UNMEASURABLE
BW >= 1000 g                      675   22     7  0.342  0.8394   [0.619, 0.961]
PA < 40 d                         541   20     4  0.325  0.8894   [0.627, 0.970]
PA >= 40 d                        400   17     8  0.667  0.8448   [0.614, 0.971]
```

**Three of four strata are not assessable.** The test set has **exactly one female infant
with ROP**. GA<28wk and BW<1000g select the same five babies. Only postnatal age splits into
two measurable cells, and those intervals overlap.

The script refuses to print an AUC for a cell with fewer than 3 positive patient-units, which
is the T9b lesson made permanent: with one positive cluster the bootstrap cannot vary the
positive side, so it returns a narrow interval that means nothing.

**This is a finding, not a failed experiment.** "No subgroup difference was detected" would
be a dishonest reading — the honest statement is that **this dataset cannot support a
fairness analysis**, and the script prints exactly that rather than letting a reader infer
equity from overlapping intervals.

## PHI nearly left the machine — via a path no guard covered

Assembling the Kaggle code bundle by hand, the exclusion list (`dataset_split/`, `.cohorts/`,
`*.pth`, ...) still swept up:

```
results/rop/errors/false_negative/<id>_F_GA##_BW####_PA##_DG#_PF#_D1_S07_28.jpg
```

**Real infant retinal photographs**, filenames carrying patient ID, sex, gestational age and
birth weight — saved error-analysis crops living under `results/`. The bundle was 51 MB and
about to be uploaded.

`tests/test_repo_guards.py` blocks patient images from reaching **git**. Kaggle is a second,
completely separate egress path, and it had **no guard at all**.

`scripts/build_kaggle_bundle.py` now assembles the bundle and then **sweeps it**, deleting
rather than shipping any bundle containing a patient-pattern filename, an image under
`results/`, or a credential. Run against the hand-rolled bundle it flagged **152 items**.
Clean bundle: 162 files, 1.3 MB. Three tests guard the sweep, including one asserting it
passes a clean tree — a guard that rejects everything is as useless as one that rejects
nothing.

The lesson generalises: **a guard protects a path, not a secret.** Every new egress route
needs its own.

## T11b/T11c — glaucoma source audit: a real confound, but a survivable one

Scored the deployed multi-source checkpoint on val, per constituent collection, so that
within a source the source identity carries no label information.

```
source           n  glauc   prev     AUC   95% CI              usable?
OIA            457     27  0.059  0.9518  [0.914, 0.981]       yes
EyePACS        339    339  1.000       —  —                    no (single class)
image(G1020)   119     30  0.252  0.8603  [0.758, 0.945]       yes
G1020           81     16  0.198  0.8779  [0.721, 0.986]       yes
REFUGE1         75      8  0.107  0.8937  [0.679, 1.000]       no (<10 in a class)
ORIGA           59     22  0.373  0.8587  [0.738, 0.952]       yes
BEH             52     18  0.346  0.8954  [0.778, 0.976]       yes
FIVES           40     13  0.325  0.8262  [0.683, 0.945]       yes
sjchoi86        39      9  0.231  0.9556  [0.877, 1.000]       no (<10 in a class)
PAPILA          39      6  0.154  0.9040  [0.743, 1.000]       no (<10 in a class)
DRISHTI         12     10  0.833  0.6000  [0.095, 1.000]       no (n<30)
CRFO            11      6  0.545  0.3667  [0.000, 0.767]       no (n<30)
LES              2      1  0.500  1.0000  [1.000, 1.000]       no (n<30)
DR / HRF / JSIEC  <=5    —      —       —  —                   no (single class)

pooled                                    0.9669  [0.957, 0.976]
source-controlled, all 12 measurable      0.8325   <- the kernel's own figure
source-controlled, 6 usable sources       0.8784
source-controlled, size-weighted          0.9143
```

**The kernel's 0.8325 is not the number to quote either.** It is an unweighted mean that
gives an n=2 collection (`LES`, AUC 1.000) and an n=11 one (`CRFO`, AUC 0.367) the same
weight as `OIA` at n=457. Two of the three cells dragging it down are smaller than a single
ward round. That is the *same* error as the too-narrow ROP interval, pointed the other way:
pretending a small sample carries as much information as a large one. Restricting to
collections that can actually support an estimate gives **0.878**.

**67.2% of val positives come from single-class collections** (`EyePACS-AIROGS` alone
contributes 339 of 506 and contains no healthy eyes). Those images can never contribute
discrimination — only prevalence.

### The verdict, and why it differs from ROP

| | pooled | source/device-controlled | drop | measurable strata |
|---|---|---|---|---|
| ROP | 0.9271 | **0.6843** | −0.24 | 1 of 3 cameras, before the rebuild |
| Glaucoma | 0.9669 | **0.8784** | −0.09 | 6 of 16 collections |

**Glaucoma has the same disease, in a milder form.** The pooled 0.973/0.967 overstates
within-source performance by about 0.09 and should not be the headline. But the model holds
up inside almost every collection it can be measured in (0.83–0.95), including `OIA` at
n=457 — so it is genuinely detecting disease, not merely recognising datasets. ROP could not
make that claim.

Deck and website updated to lead with **0.878 within-source**, pooled shown alongside.

## CORRECTION — the "DR flags every healthy eye" finding was my test harness

The previous entry (commit `9cbd314`) reported DR scoring 10/20, with all ten APTOS grade-0
images flagged as referable, and concluded that the referable threshold was an untuned
hardcoded 0.5. **That conclusion was wrong and is withdrawn.**

`scripts/run_test_images.py` decided whether a disease had been raised with:

```python
if finding.get("referable") is not None:
    return bool(finding["referable"])
```

`referable` is a **dict** — `{"grade_threshold": 2, "probability": 33.0, "is_referable": false}` —
and a non-empty dict is always truthy. Every DR image therefore scored as flagged. The app
had in fact returned `"prediction": "No DR"`, `"risk": "Low"`, `"is_referable": false` for all
ten. **The system was right; the scorer was broken.**

Corrected scorecard on the same 47 images:

```
DR         20/20 correct  (100%)   10/10 severe caught, 10/10 healthy correctly cleared
Glaucoma    9/10 correct  ( 90%)   one G1020 miss
ROP        10/16 correct  ( 62%)   7/7 sick caught, 6/9 healthy falsely flagged
```

### The threshold work that followed was still worth doing — it said "don't"

`scripts/derive_dr_referable_threshold.py` splits val in half (DR has no test split),
derives at target sensitivity 0.90 on half A, and applies once to half B:

```
                      sens                    spec
thr 0.4362 (derived)  0.879 [0.851, 0.903]    0.780 [0.762, 0.796]
thr 0.5    (current)  0.858 [0.828, 0.884]    0.838 [0.822, 0.853]
```

The derived threshold buys 13 fewer false negatives at the cost of **133 more false
positives**. The script **refuses to write a threshold that does not improve specificity**,
so nothing changed. 0.5 turns out to be a defensible operating point for this model, which
is exactly what the corrected 20/20 shows.

### What made this catchable

The val statistics said specificity 0.838 — about 16% false positives — while the test run
showed 100%. At a 16% rate, ten-for-ten has probability ~1e-8. **Two measurements of the same
quantity disagreed by an impossible margin, and that gap was the bug.** Reconciling them,
rather than believing the more dramatic one, is what found it.

### Side observation: DR output is resolution-sensitive by construction

Chasing the discrepancy surfaced something real but separate. Preprocessing runs
`circle_crop -> ben_graham(sigma=10) -> resize(384)`, so Ben Graham's blur is a **fixed pixel
radius applied at native resolution**. Rescaling the input therefore changes the features the
model sees:

```
image              native   1600px  1200px   800px   512px
healthy #1          0.330    0.330   0.330   0.372   0.411
severe  #2          0.981    0.981   0.880   0.870   0.831
```

No decision flipped here, but a severe case drifting 0.98 -> 0.83 means a doctor uploading a
downscaled or messenger-compressed photo does not get the same answer as one uploading the
original. Recorded as a known robustness limit, not fixed: the fix is to resize to a fixed
size *before* Ben Graham, which changes what every trained model expects and so requires
retraining, not a serving patch.

## T12 — glaucoma zero-shot on collections held out of training

The recorded blocker ("no usable REFUGE source found") assumed an external set had to come
from outside. T11a showed SMDG-19 is a meta-dataset of 15 constituent collections, so two
were simply withheld: **REFUGE1 (800 img, 80 glaucoma)** and **ORIGA (650 img, 168
glaucoma)** — 1,450 images removed at manifest level before any dataloader was built.

The kernel asserts the hold-out is real before spending the GPU time, and it printed:

```
sources in manifest: {OIA 4442, EyePACS 3269, G1020 1020, REFUGE1 800, ORIGA 650, ...}
held out 1450 images from ['REFUGE1', 'ORIGA']
remaining split sizes: {train 7587, test 1641, val 1638}
hold-out verified: no REFUGE1/ORIGA image is in train, val or test
Early stopping at epoch 16 (best auc=0.9649).  training 2773s
```

### The result splits, and the split IS the finding

```
in-distribution, pooled val                     0.9676   <- inflated by source mixing
in-distribution, WITHIN-source (6 collections)  0.8407   <- the like-for-like baseline

REFUGE1   ZERO-SHOT  0.9364   n=800  80 positives  prevalence 0.100   vs baseline +0.096
ORIGA     ZERO-SHOT  0.7354   n=650 168 positives  prevalence 0.258   vs baseline -0.105
```

**There is no single generalisation number for this model.** On a collection it has never
seen it scores anywhere between **0.74 and 0.94**, and which one you get depends on the
collection. Reporting either alone would be a choice, not a measurement:

* quoting 0.936 would say "glaucoma generalises beautifully"
* quoting 0.735 would say "glaucoma collapses on new data"

Both are true of this model. The defensible statement is the range and its cause.

ORIGA is plausibly the harder set — 25.8% prevalence against REFUGE's 10%, and ORIGA's
labels are CDR-derived with many borderline early cases, whereas REFUGE's positives are
more clearly established. That is a hypothesis, not a measurement; what is measured is that
**cross-source transfer varies by more than 0.20 AUC between two equally legitimate public
benchmarks.**

### Why this matters more than the number

Every published single-number "external validation" in this field picks one benchmark. This
experiment shows that choice can move the headline by 0.20. That is the finding to present,
and it is a stronger one than either individual figure.

**Also worth recording:** with REFUGE1/ORIGA removed and training on SMDG only, in-
distribution `G1020` fell to **0.557** — despite G1020 being *in* training. The deployed
multi-source checkpoint scores G1020 at 0.86-0.88. Removing 1,450 images and one config
source changed an in-distribution number by ~0.30, which is a further reminder of how
unstable these per-collection figures are at these sample sizes.

## T13 — DR on IDRiD: excellent transfer, and a resolution effect that dwarfs it

455 labelled images (IDRiD grading subset), native size **4288x2848**, referable prevalence
0.668. Scored through the real 5-view TTA path, under the rule production actually applies
(`is_referable = argmax >= 2`).

```
                        referable AUC    QWK     sens    spec    TP  FN   TN  FP
native resolution           0.9775      0.796   0.970   0.828   295   9  125  26
downscaled to <=1024px      0.8934      0.738   0.799   0.974   243  61  147   4

internal val reference      0.9124              0.858   0.838
Messidor-2 reference        0.837                       0.30  (collapsed from 0.89)
```

### Finding 1 — the operating point DID transfer this time

On Messidor-2 specificity collapsed 0.89 -> 0.30, and that was recorded as evidence that
operating points do not transfer. On IDRiD specificity is **0.828** against an internal
0.838, and sensitivity is *higher* than internal. **The earlier conclusion was too general.**

Transfer is not a property of the model; it is a property of the model/dataset pair. That is
the same lesson T12 produced independently on the glaucoma side (REFUGE1 0.936 vs ORIGA
0.735 zero-shot). **Two experiments, run for different reasons on different diseases, both
say single-benchmark external validation is unreliable.**

Caveat on the headline: IDRiD is a curated grading set at 66.8% referable prevalence, not a
screening population. AUC is prevalence-independent, but the sens/spec pair is easier to
achieve on a case mix this enriched, and 0.978 should not be read as "better than internal
in a clinic".

### Finding 2 — the same images, resized, become a different experiment

Rescaling to <=1024px moves AUC **0.978 -> 0.893** and flips the operating point from
sens 0.970 / spec 0.828 to sens 0.799 / spec 0.974. **Missed cases go from 9 to 61.**

The cause was already documented as a robustness note: `circle_crop -> clahe ->
ben_graham(sigma=10)` runs at NATIVE resolution before the resize to 384, so the blur radius
relative to retinal features scales with the input file. What T13 adds is the magnitude —
on a 4288x2848 source it is worth **0.084 AUC and a complete inversion of the sensitivity/
specificity trade**.

This is a deployment-relevant defect, not a curiosity. A clinician who uploads a phone
photo, a messenger-compressed image, or a PACS thumbnail is not running the same model as
one who uploads the original. Nothing in the app tells them so.

**Not fixed here, deliberately.** The fix is to resize to a fixed size *before* Ben Graham,
which changes what every trained checkpoint expects and therefore requires retraining all
three models. Recorded as the highest-value known defect rather than patched at serving time
where it would silently invalidate the existing weights.

### Note on the preflight fixes

This uploader's copy ships a single combined `idrid_labels.csv` (455 rows), so the two-CSV
and filename-collision defects the review found did not fire on this particular dataset.
They were still real: a different IDRiD copy (e.g. the original two-CSV layout with
`IDRiD_001.jpg` in both Training and Testing dirs) would have matched training labels to
testing images and produced a confident wrong number.
