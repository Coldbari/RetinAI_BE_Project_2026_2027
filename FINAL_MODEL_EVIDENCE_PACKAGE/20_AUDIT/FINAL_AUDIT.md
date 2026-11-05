# Final audit — verified status, claims, and the retraining decision

*Verification pass 2, 2026-09-21. Every number below was recomputed from raw per-sample
prediction artifacts, split CSVs and dataset tables. Nothing is carried over from a report.*

---

## 1. Claims audit

| Claim | Evidence | Unit | Dataset | Verified? | Safe to state? |
|---|---|---|---|---|---|
| Staging QWK 0.80 per eye | `locked_eval/*.npz` + view table | eye, max over views | locked (230 eyes / 116 infants) | **Yes**, 0.8012 [0.730, 0.857] | **Only with the operator named** and the alternatives tabulated |
| Staging QWK 0.52 per image | same | image | locked (661) | **Yes**, 0.5232 [0.433, 0.594] | Yes |
| "The unit was the problem, not the model" | aggregation + view tables | — | both halves | **Partly** | **No, as worded.** Say: view/label mismatch; temporal-only image scoring reproduces the eye number |
| Stage-1 frames called Normal 73% | pooled confusion, 5 ckpts | image | locked (525 preds) | **Yes**, 73.14% | Yes |
| ~18% of diseased eyes called Normal | eye aggregation | eye, stage 1–3 only | locked (147 eyes) | **Yes**, 0.1782 | Yes, **with the stage-1–3 denominator stated** |
| Temporal 12.7% vs posterior pole 71.7% | view join | image, stage 1–3 | locked | **Yes** | Yes — **and report nasal 63.4%** |
| Screening sens 0.958 / spec 0.781 (internal, in app) | `screening_rebase.json` | image | pooled OOF | Value yes | **No** — describes a 5-model ensemble, not the served fold-0 checkpoint |
| Screening spec 0.387 externally | `screening_rebase.json` | image | dev half | Value yes | **No** — in-sample; dev selected the threshold |
| **Screening, served checkpoint, locked half** | this pass | image / eye / infant | locked | **Yes** | **Yes — 0.911/0.323, 0.994/0.052, 1.000/0.000** |
| Model is over-confident | pooled OOF | image | CV pool (6,700) | **Yes**, +0.1092 | Yes, **stating unit + that it is an ensemble average** |
| Site calibration spread is case mix | `per_source_calibration.json` | image | CV pool | **No** | **No — withdrawn**, see A-2 |
| Structured ≈ flat (equivalence) | `matched_equivalence.json` | image | CV, 2 seeds | **Yes**, −0.0055 / +0.0224 | Yes |
| Structured better calibrated than flat | — | — | — | **No — the reverse** | ECE 0.109 vs 0.055 at equal accuracy |
| No data leakage | full-corpus hashing | — | all splits | **Split boundaries yes**; 4 cross-fold duplicate pairs | State the scope |
| 1,528 / 1,760 infants | split CSVs | — | corpus | Counts yes | **As upper bounds** — ROP-VL/Shenzhen IDs are record-like |
| Corpus 13,037 → 8,024 → 3,112 | PROVENANCE + CSVs | image | corpus | **Yes**, exact | Yes |
| Grad-CAM shows the model reads the retina | `cam_*` artifacts | image | dev | Not re-derived here | **No** — a visualisation, not causal evidence; the project's own fig 17 shows the metric is objective-dependent |
| Model comparable to human graders | published κ | differs on 8 axes | — | Context only | **No** — never as parity |

---

## 2. The ten scientific questions

**Q1 · Does the model learn meaningful ROP staging information?**
Yes, with a bounded claim. On the locked half it has never seen: image-level QWK 0.5232
[0.433, 0.594], accuracy 0.5204, macro-F1 0.4116; on temporal frames alone QWK 0.7993. Normal
recall is 0.990 at precision 0.529. A metadata-only learner reaches macro-F1 0.187, so the
image carries the signal. The ordinal structure is real: errors concentrate in the Normal
column, not scattered.

**Q2 · Does it generalize to the locked test?**
Yes, with degradation. Pooled OOF accuracy 0.786 → locked 0.520; QWK (5-class map) 0.807 →
0.517. Note the comparison is unfair to the locked half in one direction (OOF is an ensemble
of 5 models on held-out data) and unfair in the other (the locked half is a different
hospital). Both halves of Multi-View behave consistently, which is the relevant replication.

**Q3 · Does it generalize to another hospital/source?**
Partially. External discrimination for any-ROP is 0.835 (served checkpoint, locked) against
0.9695 internal OOF. Calibration does not transfer: pooled internal ECE 0.109 vs external
0.334. The retired head scored 0.691 externally, so the current model is a clear improvement
on the axis that motivated the change.

**Q4 · Is the model calibrated?**
No. Pooled out-of-fold, image level: accuracy 0.7861, mean confidence 0.8954, ECE 0.1092,
Brier 0.1388 — systematic over-confidence, monotone across bins. The flat baseline reaches
the same accuracy (0.7888) at ECE 0.0547. A residual per-site effect of ≈0.18 survives
case-mix standardisation. **Confidence is not usable as a gate**: in-distribution, predictions
above 0.90 confidence are wrong 7.8% of the time (359/4,597); on the locked half the same
bucket is wrong **40.9%** of the time (904/2,210).

**Q5 · Is the screening endpoint clinically meaningful at image, eye and infant level?**
At image level, yes: 0.911 sensitivity / 0.323 specificity out of sample. At eye level,
marginal: 0.994 / 0.052. **At infant level, no: 1.000 / 0.000 — every healthy infant in the
locked test is flagged.** A screening tool that refers everyone carries no information at the
unit the clinic acts on. This is a decision-rule and threshold problem, not necessarily a
model problem.

**Q6 · Why does eye-level QWK differ so much from image-level QWK?**
Because the label describes the eye (436/460 eyes single-label) and the finding is visible in
one view. Majority-vote aggregation gives 0.49 — below image level. Max gives 0.80. Temporal
frames scored individually with no aggregation give 0.7993. The mechanism is view/label
mismatch; `max` approximates "pick the temporal frame".

**Q7 · Does the structured/CORN architecture outperform the flat baseline?**
No. Matched CV: −0.0055 at seed 42, +0.0224 at seed 1337 — opposite signs. Locked half, image
level, the flat baseline is *ahead* (QWK 0.5606 vs 0.5232; macro-F1 0.4502 vs 0.4116), and at
eye level they are level (0.7948 vs 0.8012). The structured head is materially **worse
calibrated**. The defensible statement is equivalence on accuracy with a calibration
disadvantage; the contributed heads are 6,405 of 20.5 M parameters.

**Q8 · Are the errors caused by model, label, view, or dataset limitations?**
All four, separably: **view/label** (73% of stage-1 frames called Normal, but 12.7% of
temporal ones); **dataset** (Stage 4/5 n=4 and AP-ROP n=12 in the locked test — unmeasurable;
one infant supplies 470 AP-ROP frames); **label** (5 byte-identical images carry contradictory
labels; the published inter-expert κ for this task is 0.24); **model** (temporal frames are
still wrong 32% of the time, which no label argument explains).

**Q9 · Is Grad-CAM trustworthy enough to support claims?**
No, and the project already proved this itself: swapping the differentiated scalar moves
off-retina attention by ~26 points on identical weights and images. Grad-CAM may illustrate an
individual case; it cannot support a claim about what the model uses.

**Q10 · Is the current model ready for clinical deployment?**
No. Infant-level specificity 0.000 on the locked test; confidence unreliable out of
distribution (40.9% error above 0.90); calibration not transferable; Stage 4/5 and AP-ROP
unmeasurable at locked-test sample sizes; no prospective validation; no grader adjudication;
and an open PHI remediation. It is a defensible research prototype.

---

## 3. Retraining decision

### **NOT YET.**

Not a single defect found in this pass is evidence that the weights are the problem. The
evaluation was mis-specified in ways that are free to fix, and the decision rule was never
measured at the unit that matters.

| Issue | Classification |
|---|---|
| QWK computed over 6 classes against a declared 5-class endpoint | **Fix evaluation** |
| Eye-level headline operator-dependent | **Fix reporting** (promote the view analysis) |
| No eye-level clustered CI | **Fix evaluation** (done in this pass) |
| Infant-level screening specificity 0.000 | **Fix code/configuration** — decision rule and threshold, re-derived at the unit of decision |
| Over-confidence, ECE 0.109; worse than baseline | **Calibration method first** (temperature scaling on OOF CORN logits) |
| Site calibration residual +0.178 | **Fix analysis**, then per-site temperature scaling |
| Stale error galleries, wrong thresholds in docs | **Fix reporting** |
| 4 cross-fold duplicates, 5 label contradictions | **Fix dataset** (dedup + adjudicate) — 0.12% of the pool, no re-run needed |
| `patient_id` not a person for 2 sources | **Fix dataset metadata**; widen all intervals |
| Ridge visibility unknown | **Collect human labels** |
| Stage 4/5 n=4, AP-ROP n=12 on locked | **Collect data** — no model change can fix a denominator |
| Clinical readiness | **Requires prospective validation** |
| 34 tracked PHI-named images | **Repository remediation — blocking** |

Retraining becomes the right answer only if, after the evaluation is corrected and the
decision rule is re-derived at the eye/infant unit, performance is still inadequate. Retraining
now would simply re-measure a corrected artifact with an uncorrected yardstick.

---

## 4. Highest-value next experiment — pick one

**Re-derive the screening decision rule at the eye and infant unit, on the artifacts already
on disk.** Sweep (threshold × aggregation rule: any-frame, k-of-n, mean-probability,
max-probability) on the *dev* half, then evaluate the single selected rule once on the locked
half at all three units with patient-clustered intervals.

Why this one: it costs no GPU time, no new data and no labelling; it targets the only finding
that makes the system unusable as described (infant specificity 0.000); it is the one result
that would change what the product may claim; and it is a prerequisite for any sensible
retraining decision, because it tells you whether the problem is the weights or the rule.

Second: temperature scaling on out-of-fold CORN logits, evaluated on the locked half — the
cheapest fix for Q4 and for the confident-miss problem in Q5.
