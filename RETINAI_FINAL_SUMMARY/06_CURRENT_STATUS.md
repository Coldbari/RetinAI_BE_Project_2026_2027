# 6 · Current Status — where the project stands today

*Last updated 2026-09-21, repository commit `f0d4508`.*

---

## What exists right now

| Item | Status |
|---|---|
| **Trained model** | ✅ EfficientNetV2-S with ordinal + AP-ROP + hospital-adversary heads, seed 42 |
| **Live checkpoint** | ✅ `results/rop_staging/weights.pth`, 82.96 MB, SHA-256 `11b4396086…` — **identified** as the fold-0 checkpoint of the seed-42 run |
| **Web app** | ✅ runs locally (Flask); upload a photo → stage + probabilities + heat map + PDF report |
| **Cross-validation results** | ✅ 5 folds, grouped by baby, raw predictions committed |
| **External hospital test** | ✅ done, opened **once**, under a protocol written beforehand |
| **Full evidence package** | ✅ `FINAL_MODEL_EVIDENCE_PACKAGE/` — 275 files, 54 charts, 49 tables, 60 documents |
| **Paper draft** | 🟡 in progress — target ISBI 2027, deadline **26 Oct 2026** |
| **Clinician review of the labels** | ❌ **not done** — the biggest outstanding gap |
| **Prospective / real-world testing** | ❌ not done |
| **Regulatory status** | ❌ none. **Not a medical device.** |

---

## What has been proven

- **No data leakage.** Re-verified from scratch: zero babies and zero photographs shared between
  any training and any test set; the five validation folds exactly partition all 6,700 images.
- **The model genuinely learns from pixels** — macro-F1 0.692 against a metadata-only floor of 0.187.
- **Strong ordered agreement internally** — QWK 0.897 [0.869 – 0.923].
- **"Any disease" screening works** — AUC 0.970 internally, 0.835 at a new hospital.
- **Healthy-eye detection transfers** — 99.6% recall at the new hospital.
- **Mistakes are ordinally structured** — only 2% are wildly wrong.
- **The evaluation-unit finding** — per-photograph scoring understates agreement by more than
  0.35 QWK, for an anatomical reason that holds across two architectures and two random seeds.

## What has *not* been proven

- Any clinical validity, utility or safety.
- Usable per-stage performance at a new hospital (≈1 in 4 correct).
- That the probabilities mean anything (they do not — see below).
- That Stage 4/5 or AP-ROP work outside the training hospitals (only 4 and 12 test images).
- That the heat map reflects real pathology.
- Generalisation beyond **one** external hospital.

---

## The five open problems, in priority order

### 1. ⚠️ Patient-identifying data is in the public code repository — **fix first**
`DEMO.md` contains two original Ostrava filenames encoding a baby's sex, gestational age, birth
weight and two postnatal ages. The project's **own** automated guard test detects this and is
**currently failing** on the committed code, and the commit is already pushed to GitHub.
**Action:** redact, rewrite the history, force-push, and make the guard block commits instead of
just reporting.

### 2. The probabilities are uncalibrated
Calibration error 0.109 internally, **0.334** at a new hospital, where certainty-when-right and
certainty-when-wrong differ by only 0.041. **Fix:** temperature scaling on the already-saved
outputs. No retraining, no GPU, a few hours.

### 3. The committed config does not reproduce the model
`configs/rop_staging_structured.yaml` says `weighted_sampler: true` — the setting that produced
the known-broken first version. Every real run overrode it. **Fix:** one line, plus recording the
checkpoint's fold/seed/hash in a `WEIGHTS.json`.

### 4. The live model's advertised operating point is not its own
The alarm threshold (0.0155) was chosen on pooled cross-validation scores, but the served model
is a single fold's checkpoint that saw 4/5 of that data. **Fix:** re-derive the threshold on data
that checkpoint never saw. No GPU.

### 5. No independent grader has ever re-read the labels
Every label comes from the dataset it shipped with. Without a second opinion it is impossible to
say whether Stage 1's 0.376 precision is the model failing or the labels disagreeing — and there
is no human ceiling to compare 0.897 against. **This is the long pole and it needs a person, not
compute:** a paediatric ophthalmologist re-grading ~300 stratified images.

---

## What to do next, in order

| # | Action | Needs GPU? | Time |
|---|---|---|---|
| 1 | Redact the patient identifiers and gate the guard test in CI | no | 1 hour |
| 2 | Temperature-scale the model's probabilities | no | half a day |
| 3 | Fix the config, stamp the checkpoint's identity | no | 1 hour |
| 4 | Re-derive the live threshold on the served checkpoint | no | half a day |
| 5 | Make the outer-edge (temporal-view) result the paper's headline | no | 1 day |
| 6 | Re-run the component ablations on folds 1–4, both seeds | yes, ~10 h | 1 week |
| 7 | Log validation loss + learning rate, re-run one fold | yes, ~70 min | 1 day |
| 8 | **Get a paediatric ophthalmologist to re-grade ~300 images** | no | weeks — **start now** |
| 9 | Find a second external hospital | no | months |

Items 1–5 are all CPU-only and could realistically be finished in a couple of days.

---

## Can it be used today?

**Computationally, yes:** 112 ms per photograph on an ordinary laptop CPU, 83 MB model, no GPU
required. Compute is not the constraint.

**Clinically, no.** The blockers are:

1. Per-photograph staging at a new hospital is ≈1 correct call in 4.
2. The probabilities are not probabilities, and are worst where it matters most.
3. The most dangerous error is the most confident one, so no confidence-based safety gate works.
4. The heat-map explanation does not reliably point at the retina.
5. The severe classes rest on a handful of images.
6. One external hospital, retrospective data, no clinician adjudication.

**What it is genuinely good for today:** methods research. The finding about how ROP datasets
are *labelled* and how ROP models should be *scored* is more solid than the model itself, and
that is what the paper should lead with.

---

## Things the project already does right, and should keep

- The final test set was **opened exactly once**, under rules written and committed beforehand.
- A previously-served model was **retired** when an audit found it flagged 663 of 663 healthy
  images at a new hospital.
- **Glaucoma was dropped** rather than publishing a contaminated AUC of 0.973.
- The app **refuses to run on non-infant photographs** (it had flagged 41 of 45 adult eyes as
  ROP-positive).
- The app already labels the binary score *"read it as a ranking, not a probability of disease"*.
- Raw per-photograph predictions are **committed**, which is why every number in this folder
  could be independently recomputed rather than trusted.
