# Pre-registered protocol: opening `test_locked`

*Committed BEFORE any locked-half inference is run. The dev half (`test_dev`, 663 images)
may be used freely to debug the pipeline; the locked half (`test_locked`, 661 images,
116 infants) is scored exactly once, by `scripts/rop_locked_eval.py --set locked`, and the
script refuses to run twice. Anything not specified here may not be decided after seeing
locked results.*

## What is evaluated

The four matched arm-seed cells — the same 2×2 the equivalence verdict rests on:

| arm | seed | checkpoints |
|---|---|---|
| structured (natural sampling) | 42 | 5 CV fold checkpoints |
| structured (natural sampling) | 1337 | 5 CV fold checkpoints |
| flat softmax (natural sampling) | 42 | 5 CV fold checkpoints |
| flat softmax (natural sampling) | 1337 | 5 CV fold checkpoints |

All checkpoints are last-epoch, recovered from their Kaggle kernels of origin. No model is
retrained, fine-tuned, or selected for this evaluation.

## Aggregations (both computed from the same forward passes)

1. **PRIMARY — per-checkpoint mean ± sd**: each of the 5 fold checkpoints scores every
   image; each metric is reported as mean ± sd over the 5 checkpoints. This matches how CV
   was reported and invents no new model.
2. **SECONDARY — 5-checkpoint ensemble**: the mean of the 5 checkpoints' prob6 vectors,
   argmax-decoded. Deployment-style; reported but not the headline.

## Metrics, fixed in advance

- Image-level only (Multi-View labels are per-image; no session structure exists there).
- **QWK** (primary ordinal metric), accuracy, macro-F1 over PRESENT classes, 6-label
  macro-F1 (stated together, per cross_site_macro_f1.json's lesson), per-class recall.
- AP-ROP decoded at the a-priori threshold 0.5. **Whatever the numbers say, stage 4/5
  (4 images) and AP-ROP (12 images) remain flagged "not externally validated"** — those
  counts cannot validate anything; their cells are reported in italics with n.
- Additionally macro-F1 over classes with ≥30 locked images, so the headline is not
  hostage to a 4-image cell.
- 95% CI: patient-clustered bootstrap (2000 replicates, numpy seed 42) on QWK and
  present-class macro-F1, for the PRIMARY aggregation of each arm-seed.

## Outputs (committed in the same run)

- `results/rop/locked_eval/<arm>_<seed>_ckpt<k>.npz` — raw prob6 + preds per checkpoint
  (and the same for the dev dry-run under `dev_eval/`).
- `results/rop/locked_eval/locked_table.json` — every number above.
- The script writes a `OPENED_AT` timestamp into the JSON; a second invocation with
  `--set locked` exits with an error if `results/rop/locked_eval/` exists.

## What would count as a surprise, stated in advance

CV says the arms are equivalent (the difference flips sign across seeds) and the
site-fold QWK is ~0.95. If locked QWK lands far below the dev half's, that is a finding
about the split, to be reported as such — not a license to re-evaluate with different
settings. There is no second shot.

---

## Amendment 1 — after the dev dry-run, BEFORE opening the locked half (2026-08-18)

The dev half falsified this protocol's premise that "Multi-View labels are per-image."
Measured on `test_dev` (freely inspectable by design):

- **217 of 230 eyes carry a single label across all of their acquisition views** — the
  labels are per-EYE diagnoses stamped onto every view of that eye.
- Stage 1–3 images are predicted normal at **77% (posterior pole) / 69% (nasal) / 17%
  (temporal)** — and the temporal periphery is where the ROP ridge anatomically lives.
  The model calls views normal that do not show the finding; the label came from the eye.
- Image-level, 53.5% of stage 1–3 dev images are "missed"; **eye-level (max over the
  eye's views), 18.5% of eyes are.**

This is the paper's central label-unit mechanism, in per-eye form, in the held-out
hospital. Accordingly, and stated before any locked image is scored:

1. **Eye-level aggregation is added as CO-PRIMARY**: unit = (patient, eye) from the
   dataset's own Acquisition View Table; eye label = max over its views (they are
   constant for 94% of eyes); eye prediction = max over the same views; QWK, accuracy and
   present-class macro-F1 at this unit, with patient-clustered CIs.
2. Image-level metrics are still reported in full, per the original protocol — the
   CONTRAST between the two units is the result, exactly as it was for Ostrava sessions.
3. The per-view miss table (stage 1–3 predicted-normal rate by acquisition region) is
   reported for the locked half as the mechanism check.
4. Everything else in the original protocol stands unchanged.
