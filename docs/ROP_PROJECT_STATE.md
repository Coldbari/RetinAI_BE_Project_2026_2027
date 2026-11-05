# ROP Staging — Complete Project State

*The single document to read before touching this project. Everything here was done between
2026-08-05 and 2026-08-12 (thesis: DR + ROP retinal screening; glaucoma dropped post-presentation).
If a claim below matters to you, every number is reproducible from a committed script.*

**Mission:** a publishable paper (target: ISBI 2027, deadline **Oct 26, 2026**; fallbacks MIDL
→ MELBA/TVST) on 6-class ICROP staging of retinopathy of prematurity across four cohorts from three hospitals,
built on public data with evaluation honest enough to survive review. The same head design
extends to DR (grades = ordinal, DME = orthogonal branch) later.

---

## 1. The corpus — `data/rop/`

| source | images | infants | role |
|---|---|---|---|
| Ostrava (RIDIRP, Czech) | 6,004 | 188 | largest; stage labels **recovered** from filename DG codes |
| Shenzhen (China) | 1,099 | 483 | training site |
| ROP-VL (same Shenzhen hospital, different cohort) | 2,020 | 1,116 | only public source of stage 4/5 |
| Multi-View (held-out hospital) | 1,324 | 232 | **never trained on** |

Plus non-staging assets: FARFUM-RoP (plus disease), HVDROPDB (vessel/ridge masks — future
vessel-attention work), ORVS/Shantou segmentation sets. `data/rop/README.md` + `PROVENANCE.json`
trace every source to its original registry — **never cite Kaggle mirrors** (2025-26 retraction
wave for untraceable Kaggle data; see §8).

Data engineering that any reader must know:
- **Label recovery:** Ostrava filenames encode a per-session diagnosis code (DG0–13). Join on
  `(ID, device, serie)` — infants change diagnosis between sessions, so ID-only joins are wrong.
  6,004/6,004 resolve (`scripts/rop_recover_staging.py`). 43% of the old binary "ROP-positive"
  class is actually hemorrhage/toxoplasma/hamartoma/hypoplasia — why the binary label was unusable.
- **De-identification:** original Ostrava names carry sex/GA/birth-weight/postnatal-age.
  `scripts/anonymise_ostrava_tree.py` rebuilds the tree as `<id>_D<dev>_S<serie>_<frame>.jpg`
  (hardlinks, zero disk). The PHI sweep in `scripts/build_kaggle_bundle.py` fails closed on the
  original pattern; the name-map lives outside `data/rop/` and is gitignored.
- **Dedup:** ROP-VL and Shenzhen overlap (same hospital) — 20 re-encoded duplicates removed via
  256-bit dHash calibrated on known byte-dupes (64-bit dHash is UNSAFE on fundus: it matched a
  Stage 2 to a Normal at d=0).
- **Identity fix:** Ostrava IDs 40 and 41 are ONE infant (19 byte-identical JPEGs, same
  sex/GA/BW, contradictory DG codes on identical pixels). Merged in the manifest builder.
  The other 13 same-(sex,GA,BW) ID pairs have no shared images → NICU twins, kept separate.
- **Class vocabulary:** normal, stage_1..3, stage_4_5 (merged; 89 images), arop. stage_0,
  treated, other_pathology are **out of scope** for the 6-class task.

## 2. Splits — `results/prepared/rop_staging/` (committed; the folds ARE the experiment)

Built by `build_rop_staging_manifest.py → build_rop_splits.py → make_staging_folds.py`:
- 5 patient-grouped CV folds over Ostrava+Shenzhen+ROP-VL (3,112 train images after per-infant
  caps: normal capped at 2/infant — 18.6:1 → 12.5:1 while keeping all 646 normal infants).
  Caps apply to TRAIN only; val is natural prevalence or the metric measures the cap.
- `fold_site.csv`: leave-Shenzhen-out cross-site check (Shenzhen has no stage 4/5 or AP-ROP —
  it scores 4 classes; not directly comparable to 6-class numbers).
- Multi-View split by patient into `test_dev` (663, for audits) and **`test_locked` (661,
  opened ONCE at the very end — never load it in any loop; this is the project's one
  irreversible rule)**.
- Manifest paths are repo-relative and PHI-free.

## 3. The audit — `scripts/rop_shortcut_audit.py` (run before trusting ANY number)

| gate | result |
|---|---|
| G1 patient leakage | PASS — zero overlap, all folds and both test halves |
| G2 duplicate leakage | PASS — closest train↔held-out pair at dHash d=15 vs threshold 4 |
| G3 site decodability from file metadata | 0.997 balanced acc (the shortcut exists; the model must not use it) |
| G4 metadata-only baseline | CV macro-F1 **0.187** — the bar any image model must beat |
| G5 linear site probe on embeddings | see §6 — runs with disease-held-constant + untrained-backbone controls |

G3 also records **class–source confounding: all 89 stage_4_5 images are ROP-VL-only** →
performance on that class may be site recognition; it stays in `unvalidated_classes` forever.

## 4. Five-backbone benchmark (flat softmax, fold 0, seed 42, 20 epochs)

`kaggle_kernels/rop_staging_benchmark.py`, pushed via `scripts/push_staging_kernels.py`.

| backbone | macro-F1 | cross-site F1 | AP-ROP recall |
|---|---|---|---|
| EfficientNetV2-S 20.2M | **0.520** | **0.641** | 0.183 |
| DeiT3-S 21.8M | 0.517 | 0.507 | 0.174 |
| ConvNeXt-T in12k 27.8M | 0.509 | 0.530 | 0.162 |
| CAFormer-S18 24.3M | 0.498 | 0.507 | 0.221 |
| EfficientNet-B0 4.0M | 0.492 | 0.536 | (pre-metric-fix) |

Two findings: (a) spread 0.028 across 4–28M params, CNN/ViT/hybrid → **architecture is not the
constraint**; (b) **every backbone fails identically on AP-ROP** (recall 0.16–0.22 at precision
0.89–0.97) and Stage 2 (0.45–0.50) → the failure is structural (softmax + taxonomy), not
capacity. These two facts motivate the model in §5. EfficientNetV2-S won cross-site → chosen.
Figures: `graphs/rop_staging/` (per-model + comparison; all stamped PROVISIONAL).

## 5. The contribution — clinically-structured head (`models/common/structured.py`)

`EfficientNetV2-S (stripped) → z∈ℝ¹²⁸⁰ (+Dropout 0.3) →` three branches:
1. **Ordinal CORN stage head** `Linear(1280→4)` — Normal<S1<S2<S3<S4/5; **AP-ROP images
   masked** from this loss (ICROP-3: the aggressive form has no stage).
2. **AP-ROP binary branch** `Linear(1280→1)` — √-damped pos_weight (~4×), threshold 0.5
   **fixed a priori** (tuning it on val = selection).
3. **Site adversary** GRL(−λ·∇) → `1280→256→ReLU→Dropout→3`, λ ramped 0→w_site (DANN).

Decode: AP-ROP if pₐ>0.5 else CORN rank; prob = `[(1−pₐ)·P(rank=k)…, pₐ]` → scored by the
SAME `compute_metrics` as the baselines. Trainer: `train_structured.py` (separate from
`train.py` on purpose — shared loop also trains shipped DR/binary models). Config:
`configs/rop_staging_structured.yaml`. Unit tests: `tests/test_structured.py` (9).
Diagram: `graphs/rop_staging/architecture.png` (`scripts/make_architecture_diagram.py`).

**v1 failed and taught:** balanced sampler + CORN = prior-shift catastrophe (QWK 0.121,
stage_4_5 precision 0.043). CORN's conditional thresholds calibrate to training prevalence;
argmax-softmax tolerates a sampler, a cumulative cascade does not. **v2 = natural sampling.**

**v2, fold 0 vs flat (same backbone/fold/metrics):** macro-F1 0.554 vs 0.520 · QWK 0.632 vs
0.594 · AP-ROP recall 0.311 vs 0.183 at HIGHER precision (0.980) · Stage 2 recall 0.607 vs
0.448. Regressions, reported as-is: Stage 1 recall 0.657→0.394; Stage 3 precision 0.866→0.519
(missed AP-ROP lands on Stage 3 — masked from ordinal training, it slots high at inference).

**Full 5-fold CV (seed 42), flat vs structured — the honest headline:**

**MATCHED at natural sampling (2026-08-16, `matched_equivalence.json`) — this supersedes
the confounded table below:**

| CV (5 folds) | flat **natural** | structured | verdict |
|---|---|---|---|
| macro-F1, seed 42 | **0.698 ± 0.099** | **0.692 ± 0.086** | diff **−0.0055** (flat ahead) |
| macro-F1, seed 1337 | **0.686 ± 0.109** | **0.708 ± 0.097** | diff **+0.0224** (structured ahead) |
| ⇒ across seeds | — | — | **THE SIGN FLIPS.** Equivalence is not just "small difference" — the two seeds disagree about which head is better. Any single-seed report would have claimed an architecture effect in whichever direction it landed. |
| AP-ROP recall | 0.723 | 0.671 | flat marginally ahead |
| Stage 2 recall | 0.615 | 0.632 | structured ahead on 3/5 folds — **the "every fold" claim is withdrawn** |
| cross-site F1(present) / QWK, s42 | 0.8182 / 0.9470 | 0.8180 / 0.9462 | **identical — the "structured transfers better" claim is withdrawn** |
| cross-site F1(present) / QWK, s1337 | 0.8242 / 0.9540 | 0.8201 / 0.9600 | flat ahead on F1, structured on QWK — mixed again |

⚠️ Both of the structured head's apparent wins were the SAMPLER, not the head. What
survives is taxonomy-faithful outputs and error *structure* (§6: 284 of 506 missed severe
images land on the adjacent stage, not normal) — no aggregate metric favours it.

<details><summary>Superseded: the confounded table (flat trained WEIGHTED, structured natural)</summary>

| CV (5 folds) | flat softmax (weighted) | structured | verdict as then written |
|---|---|---|---|
| macro-F1 | 0.683 ± 0.118 | 0.692 ± 0.086 | "equivalent" |
| QWK | 0.836 ± 0.134 | 0.826 ± 0.145 | equivalent |
| AP-ROP F1 | 0.733 ± 0.238 | 0.725 ± 0.216 | equivalent |
| Stage 2 recall | 0.570 ± 0.105 | 0.632 ± 0.083 | "structured ≥ flat on every fold" |
| cross-site F1 / QWK | 0.792 / 0.930 | 0.818 / 0.946 | "structured slightly better" |
</details>

Cross-site macro-F1 is averaged over the **4 classes the held-out hospital has** (no stage
4/5 or AP-ROP exists there; a perfect classifier scores 0.667 on the 6-label average).
Earlier revisions quoted the 6-label figures 0.528/0.545 — an artifact of two structural
zeros in the denominator, not of the models; corrected 2026-08-15, recomputation in
`results/rop/cross_site_macro_f1.json` (s1337 structured: 0.820). QWK never had the
problem, which is why the two metrics appeared to disagree.

⚠️ **THE FOLD-0 STORY DID NOT GENERALIZE.** The five-backbone "structural AP-ROP failure"
(recall 0.16–0.22) was a property of FOLD 0 — the fold whose val holds the corpus's ONE
heavily-imaged, session-label-noisy Ostrava AP-ROP infant (`ostrava:43`, 470 of fold 0's
476 AP-ROP frames; the other five AP-ROP infants there are ROP-VL with 1–2 frames each).
⚠️ Earlier revisions of this doc, the deck and the log book said "six heavily-imaged Ostrava
infants" — WRONG on both counts, corrected 2026-08-14. On folds 1–4 flat softmax
reaches AP-ROP recall 0.83–0.97 at image level. The session-label finding (§6) explains
BOTH models' fold-0 floor, and is the real discovery; the branch's contribution is
taxonomy-faithful outputs (independent AP-ROP probability + ordinal ranks) at equivalent
performance — not superiority.
Raw per-fold predictions committed at `results/rop/staging_runs/structured_5fold/*.npz`
and, for every ablation arm, `results/rop/staging_runs/ablations/*.npz` (incl. a
flat-natural arm with npz — the first flat run with raw predictions; `train.py` now
always saves them).

**Seven-arm component table (fold 0, seed 42 — `staging_runs/ablations_fold0.json`):**

| arm | macro-F1 | QWK | AP-ROP r (image) | AP-ROP r (session, 22) |
|---|---|---|---|---|
| flat softmax, weighted sampler | 0.515 | 0.603 | 0.191 | — (run kept no npz) |
| flat softmax, natural | 0.563 | 0.580 | 0.237 | **0.952** |
| structured, full | 0.553 | 0.569 | 0.212 | 0.857 |
| — adversary off (w=0) | 0.546 | **0.648** | 0.265 | 0.857 |
| — branch removed | 0.549 | 0.575 | 0.195 | 0.810 |
| — CORN removed | 0.479 | 0.394 | 0.181 | 0.714 |
| — weighted sampler | 0.403 | **0.129** | 0.141 | 0.905 |

Readings, all measured (`ablation_session_level.json` for the session column): (1) the sampling
confound closes as a NON-issue — flat holds either way, while CORN only trains natural;
(2) CORN removal is the one big head effect (0.394, with end-of-schedule instability);
(3) **no head design separates at session level either** — every arm lifts to 0.73–0.95
with overlapping CIs, flat-natural highest, even the sampler-collapsed arm recovers to
0.909. The branch's justification is taxonomy fidelity alone; the evaluation-unit finding
is model-independent, which is what makes it a finding about labels, not architectures.

**Patient-clustered bootstrap + calibration** (`structured_bootstrap_calibration.json`):
pooled macro-F1 0.652 [0.574, 0.762] · QWK 0.761 [0.627, 0.912] · AP-ROP image recall
0.345 **[0.25, 0.84]** — an interval that wide is the label-unit finding as statistics.
ECE: fold 0 **0.196** vs 0.057–0.092 on all other folds (third independent signature of
the fold-0 label problem); pooled 0.107.

## 6. The two findings bigger than the model

**Session-level AP-ROP** (`scripts/rop_session_level_analysis.py`): Ostrava AP-ROP labels are
per-session codes stamped on every frame; **70.4%** of Ostrava AP-ROP frames score
p<0.1 (median session 74.6%) vs only 10.1% in the frame-labelled sources — label carried,
appearance absent, and the contrast IS the control. (⚠️ the old "47%" came from the
superseded `session_level_arop.json`.) Pooled out-of-fold across 5 folds:
image-level recall **0.345** → session-level **0.840** (100/119, CI 0.76–0.90) at **0.25%**
false-alarm rate on 789 normal sessions. The image-level ceiling belonged to the evaluation
unit, not the model.

**DANN dose–response** (`results/rop/dann_probe_ci.json`, patient-clustered bootstrap CIs
added 2026-08-15): disease-controlled site decodability — untrained 0.894 · flat 0.882 ·
w=0 0.866 · w=0.5 0.852 · w=2.0 **0.917 [.87,.96] (above untrained!)**, while w=2.0
destroyed AP-ROP recall (0.311→0.168) and QWK (0.632→0.525). Paired CIs: the w=0→0.5 step
is noise (CI spans 0), the jump to w=2.0 is real (+0.066 over w=0.5, CI [+0.022,+0.115]).
The naive probe meanwhile IMPROVED at w=0.5 (0.859→0.809): **the standard probe overstates
adversarial debiasing; only a disease-controlled probe sees the truth.** No prior ROP work runs
any probe. This is a finding, not a failed component. ⚠️ Earlier revisions said "rises
monotonically 0.867→0.885→0.921" — the 0.885 was measured on a sibling training run whose
checkpoint was later overwritten on Kaggle; the canonical (publicly served, hash-verified)
checkpoint measures 0.852, and the sibling-run spread (0.033) is itself the size of the
step that had been read as signal. Corrected 2026-08-15.

## 7. Kaggle infrastructure

Datasets (all private, praharshnagpure): `rop-staging-corpus` (5.2 GB de-identified images),
`rop-staging-backbones` (weight caches), `retinal-framework-code` (bundle built ONLY by
`scripts/build_kaggle_bundle.py` — PHI sweep fails closed; never rsync by hand).
Hard-won gotchas, each of which cost a failed run:
- Kaggle strips the archive top level: `data/rop/…` mounts as `…/rop/…` → anchor on source dirs.
- `/kaggle/input` is read-only; HF_HOME needs `.locks/` → copy caches to `/tmp/_weights`.
- P100s (sm_60) cannot run preinstalled torch → kernel self-detects and installs cu121.
- Kaggle **silently discards** kernel pushes beyond 2 concurrent — always re-list to verify.
- Runs are CPU-bound (CLAHE), not GPU-bound: num_workers=4; model size barely affects wall-clock
  (~3.5 min/epoch, ~70 min/fold, ~7h per 5-fold+site session; quota ≈30 GPU-h/week).
- Push-script substitution is line-anchored with exactly-one-match assertions (a docstring
  collision once made all five "different" backbones train ConvNeXt).

## 8. Literature position (all four key papers read in full; agents' sweeps in git history)

| rival | their gap we fill |
|---|---|
| **VEA-Net** (Frontiers 2026; same Ostrava data, subject-level 10-fold — honest) | AP-ROP **folded into Stage 4**, treated→Stage 5; accuracy-only (their Ostrava stage acc 90.4% vs 89.5% majority-class prior!); DANN unquantified; no held-out hospital |
| **ROPDeepX** (Discover AI 2026) | patient-safe split on FARFUM only; **RIDIRP split at image level → 99.7%** with ~32 img/infant. Cite as the live leakage example |
| **Peng** (IEEE TMI 2021, kappa 0.979) | ordinal-ROP prior; per-image, single-center, no patient grouping stated |
| **Eye 2024** | A-ROP as softmax class, 41 images, single center |

Claims wording (survives review): never "first ordinal ROP" / "first AP-ROP task" — always
*"first joint ordinal + AP-ROP-branch model with probe-quantified site debiasing and
held-out-hospital evaluation."* VEA-Net also used RIDIRP's DG codes — we are NOT first there;
our differentiator is the taxonomy (AP-ROP preserved) and the instruments.

## 9. What remains (in order) — updated 2026-08-15

DONE since the last revision: flat 5-fold table (§5), ALL seven ablation arms incl. w_site=0
and the sampler-confound closure (§5 table), bootstrap CIs + calibration, session-level
scoring of every arm, `train.py` npz saving.

**Analyses completed 2026-08-15, all CPU-only, all committed with their scripts:**

| analysis | artifact | what it changed |
|---|---|---|
| McNemar / DeLong / clustered dAUC on every fold-0 pair | `significance_fold0.json` | seed-pair McNemar p=0.0001 — image-level paired tests find "significance" from seed noise alone; under patient clustering NO pairwise AP-ROP difference survives |
| DANN probe patient-clustered CIs | `dann_probe_ci.json` | killed "monotonic": w=0→0.5 spans zero, only w=2 is real (+0.066 [.02,.12]). Exposed that the committed 0.885 cell came from an overwritten sibling run (canonical: 0.852) |
| cross-site macro-F1 denominator | `cross_site_macro_f1.json` | 0.528/0.545 was 2 structural zeros — the held-out site has 4 of 6 classes. Present-class: flat 0.792, structured 0.818/0.820 |
| inclusion/exclusion cascade | `inclusion_flow.json` | the undocumented 5,013-image gap between "13,037 acquired" and "8,024 eligible", reconciled at every layer (script refuses to emit a flow that doesn't sum) |
| per-hospital calibration | `per_source_calibration.json` | raw ECE spread 8.4× is mostly CASE-MIX: disease-controlled it collapses to 2.3×. Residual Ostrava over-confidence +0.027 (CI excludes 0) |
| treatment endpoint + miss anatomy | `treatment_endpoint.json` | session-level severe sensitivity 0.904 [.88,.93] vs image 0.619 [.47,.88]; of 506 missed severe images 284 were called stage 2 (adjacent) and 106 normal — the ordinal head buys error STRUCTURE, not accuracy |

Paper sections written from these: §2 inclusion cascade, §4.4 rewritten, §4.5 calibration,
§4.6 clinical endpoint, plus a full **Compliance with ethical standards** section (licences,
IRB position, re-identification incident, availability). ⚠️ Licence constraint surfaced
there: Multi-View — the **held-out test hospital** — is CC BY-**NC**.

Also shipped, outside the paper: the HF Space URL is live again as a free **static** project
page (Docker/Gradio now need PRO), weights live in the free model repo
`Champ610/retinai-rop-weights`, and the BE log-book repo
(`Coldbari/RetinAI_BE_Project_2026_2027`) carries the runnable app + updated README.

⚠️ **PHI, third miss (2026-08-15).** `scripts/scrub_phi_from_results.py` — the scrubber —
illustrated the filename format in its DOCSTRING with a real infant's row. Caught by the
content guard added the day before, on a commit that had never been pushed. Fixed with the
synthetic sentinel. Standing lesson, now also in the paper's ethics section: each of the
three misses had a different shape and each was found by WIDENING the check, never by
re-reading the code.

1. **GPU queue REVISED by the 2026-08-14 audit.** Structured s1337 completed. The old plan
   (flat-WEIGHTED s1337, then s2024) is DROPPED. Verified confound: the committed flat CV
   baseline is the weighted-sampler arm while structured trained natural; fold-0 sampling
   effect on macro-F1 (+0.047) is 5× the claimed architecture effect (+0.009). Next quota:
   (a) flat-NATURAL 5-fold+site at s42, split into TWO kernels (3 folds each — six folds in
   one session trips the kernel's own 8h budget guard and silently drops fold_site);
   (b) flat-NATURAL s1337 the same way → a matched head×seed 2×2 at one sampling regime.
   Seed 2024 dropped: no finding asks for a third seed; two seeds already replicate.
2. **`test_locked` — DONE, opened once, 2026-08-18** (`results/rop/locked_eval/`;
   protocol pre-registered in `docs/paper/LOCKED_EVAL_PROTOCOL.md` + dev-evidenced
   amendment; the script refuses a second run). Headlines: eye-level QWK **0.79–0.80
   across all four matched arm-seed cells** (equivalence replicates externally); the
   label-unit mechanism replicates in per-EYE form with anatomy as witness (stage 1–3
   frames called normal: posterior 71.7% / nasal 63.4% / **temporal 12.7%** — the ridge
   lives temporally; normal recall 0.996); and fold_site's 0.95 was never
   leave-HOSPITAL-out (ROP-VL = same hospital as Shenzhen stayed in training) — the first
   genuine new-hospital number is 0.79–0.80. Stage 4/5 (n=4) + AP-ROP (n=12) stay
   "not externally validated" as pre-registered.
3. Paper: ISBI 4-page (deadline 2026-10-26) + CLAIM checklist; white-background variants
   of the dark figures. Skeleton: `docs/paper/isbi2027_draft.md`.
4. **HUMAN ACTION, blocking Tier-2 venues: a pediatric ophthalmologist co-author** to
   adjudicate ~300 stratified labels (κ vs our recovered labels). Lead: KIDROP / Narayana
   Nethralaya, Bangalore (user's own contact-hunting note). Every week of delay is risk.
5. Open literature item: published human inter-grader kappa for STAGING (the ceiling for our QWK).

## 10. House rules (learned, not decreed)

- No number without its evaluation regime attached. No mean without an SD. No single-fold SD.
- Last-epoch weights, never best-epoch (`eval.early_stopping: false` disables BOTH halves).
- Thresholds fixed a priori. Synthetic test fixtures deleted immediately after use.
- Config keys must be READ by code — this repo shipped five silent no-op keys once.
- `git log` is the lab notebook: every commit message carries the why and the numbers.
