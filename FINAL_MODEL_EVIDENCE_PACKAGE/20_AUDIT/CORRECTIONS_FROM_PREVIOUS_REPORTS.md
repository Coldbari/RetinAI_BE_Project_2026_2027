# Corrections register

*Every correction this verification pass makes, with the superseded value preserved. Nothing
was deleted; where a file name collided, the pass-1 version was regenerated from its own
script and the pass-2 version sits beside it with a `_v2` suffix or in
[`verification_pass_2/`](verification_pass_2/).*

## A. Corrections to the project's committed artifacts

| # | Artifact | Said | Verified value / correction | Class |
|---|---|---|---|---|
| A-1 | `configs/rop_staging_structured.yaml` `eval.qwk_classes` | declares a 5-level ordinal scale | **key is never read** — all QWK computed over 6 labels with AP-ROP at rank 5. CV pooled OOF 0.7610 → **0.8067**; fold 0 0.5685 → **0.6437**; locked eye 0.8012 → **0.8225** | evaluation fix |
| A-2 | `results/rop/per_source_calibration.json` | "most of that spread is CASE-MIX… do not report the 8x" | Normal-only control is not a case-mix adjustment. Directly standardised: Ostrava **+0.170**, Shenzhen −0.008, difference **+0.178 [+0.140, +0.215]** | analysis correction |
| A-3 | `results/rop/staging_shortcut_audit.json` `G2 {pass:true, exact_md5:0}` | reads as "no duplicate leakage" | scope is train-pool vs held-out-hospital only. Full-corpus MD5 finds **4 cross-fold duplicate pairs, 5 label contradictions on byte-identical images** | scope correction |
| A-4 | paper figure 10 caption | "only the unit the label describes changes" | the gap is produced by the `max` operator and is reproduced by **temporal-view image-level scoring with no aggregation** (0.7993 vs 0.8012 locked; 0.7182 vs 0.7206 dev) | interpretation correction |
| A-5 | paper figure 10 error bars | checkpoint SD (≈0.015–0.028) | statistical uncertainty is the patient-clustered CI: image **[0.433, 0.594]**, eye **[0.730, 0.857]** | uncertainty correction |
| A-6 | paper figure 11 caption | "the ridge grows in the PERIPHERY" | nasal retina is peripheral and misses **63.4%**; the evidence supports *temporal* specifically. The figure is right, the caption is not | caption fix |
| A-7 | `results/rop/errors/error_summary.json` | acc 0.8336, FN 177, FP 73 | not reproducible from any committed artifact; current `preds_weights_test.npz` gives **0.8435 / 166 / 69**. Also scores an argmax rule the product never used | stale artifact |
| A-8 | `configs/rop.yaml` `screen_threshold: 0.1933`, `results/rop_statistics.md` §S5 "deployed threshold" | presented as deployed | the app serves the **structured model at 0.0155**; 0.1933 is the retired ResNet50 path | documentation fix |
| A-9 | `results/rop/operating_point.json` "derived_on: 5-view TTA" and docs describing TTA | implies TTA at serve time | the served staging model runs **1 forward pass**; TTA belongs to the retired head | documentation fix |
| A-10 | `data/rop/PROVENANCE.json` ostrava record | `licence: CC0` **and** "PRIVATE infant photographs" in the same record | mutually contradictory; must be resolved before any ethics statement | contradiction |
| A-11 | `results/sweep_rop.md` latency table | reads as the model's deployment cost | benchmarks the **retired binary sweep**; served model measured in [`15_DEPLOYMENT/`](../15_DEPLOYMENT/DEPLOYMENT_FOOTPRINT.md) | scope correction |

## B. Corrections to this audit's own earlier passes

| # | Earlier statement | Corrected | Why |
|---|---|---|---|
| B-1 | "446/460 eyes are single-label" | **436/460** (and 210/232 patients, not 216) | `pd.to_numeric(errors="coerce")` turned the string labels `Normal` and `AROP` into NaN, and `nunique()` ignores NaN |
| B-2 | "17.8% of **diseased** eyes called Normal" | 17.8% of **stage-1–3** eyes (n=147); all-diseased is **17.1%** (n=153) | denominator excludes Stage 4/5 and AP-ROP by construction |
| B-3 | "the headline lacks an interval, so it is under-evidenced" | the interval is absent from the artifacts but computable and reasonably tight, **[0.730, 0.857]** | severity overstated; the operator dependence (A-4) is the real defect |
| B-4 | "the product is DR + ROP" | **ROP only** | `webapp/registry.yaml` states DR and glaucoma were both removed |
| B-5 | "error galleries are one generation stale" | **two** generations | even the retired model's own current prediction file disagrees with its summary (A-7) |
| B-6 | R2 "qualitative error atlas" recommended as evidence | **withdrawn as evidence** | no ridge annotation exists anywhere in the corpus; lesion visibility needs a human grader |
| B-7 | pass-2 preprocessing latency 120.0 ms | use pass-1's averaged **26.7 ms** | pass-2 timed a single cold call |
| B-8 | pass-2 initially overwrote 33 pass-1 files | pass-1 outputs regenerated from `20_AUDIT/scripts/`; pass-2 outputs relocated to `verification_pass_2/` | no pass-1 content lost; both versions retained |

## C. Where pass 1 and pass 2 agree (independent replication)

Two independently written pipelines, same raw artifacts, same conclusions: the QWK
class-definition defect; the `max`-operator dependence of the eye-level headline; the
temporal-view control reproducing it; structured-vs-flat calibration (ECE 0.109 vs 0.055 at
equal accuracy); confident misses being the dangerous class; the served checkpoint being
fold-0; zero patient/path leakage across split boundaries; and all dataset counts.

The numeric differences between the two are entirely explained by convention, not by data:
pass 1 reports QWK under the **5-class drop** rule and quotes the **served checkpoint alone**;
pass 2 reports **6-class and 5-class-map** and quotes the **5-checkpoint mean**. Both are
stated; the recommendation is in
[`06_VALIDATION/QWK_DEFINITION_CORRECTION.md`](../06_VALIDATION/QWK_DEFINITION_CORRECTION.md).

## D. Corrections applied to the repository itself (outside this package)

| File | Change |
|---|---|
| `DEMO.md` | five clinical-record filenames replaced with infant/session/frame references + a note; the repo's own failing guard now passes (24 passed, 1 skipped) |
| `tests/test_repo_guards.py` | new `test_no_clinical_record_identifiers_in_tracked_filenames`; **fails by design** on the 34 tracked images until they are renamed |

Not done, and requiring the owner's decision: renaming the 34 tracked images, and purging
them and the `DEMO.md` content from git history on a shared remote. See
[`PHI_PRIVACY_AUDIT.md`](PHI_PRIVACY_AUDIT.md).
