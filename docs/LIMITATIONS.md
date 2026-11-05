# Limitations

An honest, complete account of what RetinAI **cannot** do and where it is weak. Written
plainly on purpose: a screening prototype that overstates itself is dangerous. Grouped into
clinical, data, model, explainability, and engineering limits — the last group includes
**known implementation gaps** between what the app claims and what it currently does.

> **Bottom line:** RetinAI is a research/education prototype. It is **not** a medical device,
> has **no** regulatory clearance, and was **not** prospectively validated. Do not use it for
> diagnosis or treatment decisions.

---

## 1. Clinical & regulatory

- **Not a diagnostic device.** No FDA/CE/CDSCO clearance; no IEC 62304 process; no clinical
  trial. It is decision *support* only — a clinician confirms every result.
- **No prospective validation.** All numbers are retrospective, on curated public datasets.
  Real-world screening performance (varied cameras, lighting, operators, populations) is
  unknown and would very likely be lower.
- **Single image, no context.** One fundus photo per inference. No patient history, no
  laterality (OD/OS) handling, no fusion of both eyes, no longitudinal comparison, no
  integration of IOP/OCT/visual-field data that real glaucoma or DR staging uses.
- **Operating points are population-specific.** Thresholds (e.g. ROP 0.1933) were tuned on one
  population's held-out set. External validation showed the operating point does **not**
  transfer (Messidor-2: at target sens 0.90, specificity fell to 0.30). Any new site needs
  its own threshold re-tuning.

## 2. Data

- **Label noise caps DR.** EyePACS 5-class grades have inter-grader agreement of only
  QWK ≈ 0.83–0.85, so 5-class accuracy > 90% is not achievable — the model's 74.4% / QWK 0.728
  sits near that ceiling. This is why the screening endpoint was reframed to **referable-DR
  AUC** (0.906), not 5-class accuracy.
- **ROP has no external test set.** No public ROP dataset exists, so ROP was only ever
  validated on a held-out split of the **same** private database — its true generalization is
  unmeasured. It is also small (6,004 images / 188 patients), so metrics have wide uncertainty.
- **Glaucoma generalization is only partly proven.** The multi-source model reaches G1020 AUC
  0.957, but G1020 is now *in the training distribution*. Zero-shot performance on a **further**
  unseen source (e.g. REFUGE/AIROGS) is untested — the honest zero-shot number remains the
  0.589 that motivated the fix.
- **Demographic & device bias unknown.** No breakdown by age, sex, ethnicity, camera model, or
  image quality. Public retinal datasets skew toward certain populations and devices; fairness
  across groups was not audited.
- **Residual class imbalance.** Despite focal loss + class-balanced weights + a weighted
  sampler, rare grades (Mild DR, Severe DR) remain weak (Mild-DR recall 0.41).

## 3. Model

- **Weak early-DR detection.** The No-DR ↔ Mild boundary is the largest error source
  (443 No-DR → Mild confusions). Early disease — the most valuable to catch — is the hardest.
- **DR external drop on 5-class.** On Messidor-2 the 5-class AUC drops from 0.788 → 0.600
  (−0.19); only the *referable* view transfers acceptably (−6.9%).
- **Binary collapse loses ROP detail.** ROP is reduced to No-ROP / ROP; it does not stage
  (zone/stage/plus disease) the way clinical ROP grading requires.
- **No uncertainty/abstention.** The model always answers; it has no "I don't know" or
  out-of-distribution rejection for a non-fundus or ungradable image.

## 4. Explainability

- **Grad-CAM is coarse and not causal.** The heatmap shows *where* the last conv layer
  activated, not clinical reasoning. It can look plausible while the model relies on a
  spurious cue; it is a sanity aid, not evidence.
- **One heatmap per screening.** `Registry.analyze` generates Grad-CAM only for the first
  abnormal finding, so a multi-disease-positive image shows explanation for just one disease.

## 5. Engineering & implementation gaps

These are gaps between the app's stated behaviour and its current code — worth fixing (see
[ROADMAP.md](ROADMAP.md)) and worth stating plainly.

- **Calibration is not wired into serving.** The About page/README mention "calibrated
  confidence," and temperature scaling *is* computed offline
  (`models/evaluation/calibration.py`, ECE improvements in [RESULTS.md](RESULTS.md)). But
  `webapp/inference.py` displays the **raw TTA-averaged softmax** — no temperature is applied
  at inference. Shown confidence is therefore uncalibrated.
- **Dashboard DR metric ≠ cited DR metric.** `results/dr/metrics.json` has only a `val` block,
  so the dashboard card shows **5-class val AUC 0.788**, while the About page cites the
  **referable-DR AUC 0.906**. Same model, two different measurements shown in two places.
- **DR preprocessing discrepancy.** The persisted DR run's `results/dr/config.yaml` recorded
  `circle_crop/ben_graham/clahe = false`, although the curated `configs/dr.yaml` enables them.
  The deployed DR checkpoint may have been trained on minimally-preprocessed images, which
  weakens claims that train and serve preprocessing are identical *for DR specifically*.
- **History is in-memory only.** Recent screenings live in a `deque(maxlen=24)` and are **lost
  on restart**. There is no database, no patient records, no persistence (a full design exists
  but is unbuilt — [future/patient_management.md](future/patient_management.md)).
- **`/history/report/<hid>` re-runs inference** instead of caching the earlier result, so a
  re-downloaded report can differ if anything changed.
- **Two Dockerfiles — the root one is a footgun.** The **root** `Dockerfile` still deploys the
  *legacy* `unified_model/` single-model app; only `retinal-ai/Dockerfile` deploys the real
  three-model product. Building the wrong one ships the old model.
- **CPU-only latency.** The live Space is CPU-only; three models × 5-view TTA + Grad-CAM take a
  few seconds per image. Not real-time, and no batching.
- **No authentication / multi-user / audit.** The app is single-tenant and unauthenticated —
  fine for a demo, not for anything with real patient data.
- **No automated tests / CI.** Correctness rests on manual verification.

## 6. Statistical rigor not yet closed

- The architecture-sweep recommendation (ResNet50 for ROP) is by point metrics; the planned
  **McNemar / DeLong** significance tests to confirm it are implemented
  (`models/evaluation/statistical_tests.py`) but not yet run/reported for the final choices.
- Confidence intervals exist for DR referable AUC; other headline AUCs are reported as point
  estimates without CIs on disk.

---

**In one line:** the results are genuine and honestly measured, but this is a prototype with a
small/biased data footprint, uncalibrated live confidence, no persistence, and no prospective
or regulatory validation. Treat every output as a hint for a clinician, never a decision.

See [CHALLENGES.md](CHALLENGES.md) for how many of these were discovered, and
[ROADMAP.md](ROADMAP.md) for what would close them.
