# Retinal Screening — Consolidated Results (honest, validated)

> Decision-support prototype — not a diagnostic device. Internal = held-out test/val; External = independent public dataset (no retraining).
>
> 📚 Full documentation: [../README.md](../README.md) · [../docs/RESULTS.md](../docs/RESULTS.md) (expanded metrics + per-grade tables) · [../docs/LIMITATIONS.md](../docs/LIMITATIONS.md) · [../docs/CHALLENGES.md](../docs/CHALLENGES.md)

## Headline per disease

| Disease | Arch | Internal (primary) | External | Calibration | Tier / note |
|---|---|---|---|---|---|
| **ROP** | ResNet50 | AUC **0.927**, sens 0.90 / spec 0.83 @thr 0.371 | — (no public ROP set → future work) | ECE 0.059→0.041 (done earlier) | Target |
| **DR (referable, grade≥2)** | EffNetV2-S | AUC **0.906** (95% CI 0.89–0.92); argmax sens 0.84 / spec 0.89 | Messidor-2 AUC **0.84** (drop 6.9%, PASS) | ECE 0.14→0.11 (T=0.70) | Honest screening result |
| DR (5-class) | EffNetV2-S | QWK 0.728, macroF1 0.586, acc 74.4% | QWK 0.654 | ECE 0.18→0.13 (T=0.78) | label-noise ceiling |
| **Glaucoma (multi-source, SMDG+G1020)** | EffNetV2-S | AUC **0.973**, acc 92.4%, QWK 0.84 | per-source test: SMDG 0.974, **G1020 0.957** | — | **domain shift diagnosed then fixed** |
| Glaucoma (SMDG-only, superseded) | EffNetV2-S | AUC 0.957 | G1020 AUC 0.589 (zero-shot, FAIL) | — | motivated the multi-source retrain |

## DR per-grade (internal val)

| Grade | Precision | Recall | F1 | n |
|---|---|---|---|---|
| No DR | 0.93 | 0.80 | 0.86 | 4142 |
| Mild | 0.24 | 0.41 | 0.30 | 421 |
| Moderate | 0.54 | 0.73 | 0.62 | 943 |
| Severe | 0.53 | 0.43 | 0.47 | 159 |
| Proliferative | 0.74 | 0.63 | 0.68 | 150 |

## External validation (the rigor that matters)

- **DR → Messidor-2 (1744 imgs):** referable AUC 0.906→0.837 (**6.9% drop, within ≤10% target = PASS**). Operating point does not transfer across populations (needs per-site threshold tuning).
- **Glaucoma → G1020 (1020 imgs):** SMDG-only model scored AUC 0.957→**0.589** zero-shot (36.8% drop = FAIL). Confirmed genuine domain shift (G1020 not in SMDG training, not a label flip; model over-predicted glaucoma on the new source).
- **Glaucoma FIX — multi-source retrain (SMDG-19 + G1020, 13,336 imgs):** on a stratified held-out test the model reaches **per-source AUC: SMDG 0.974, G1020 0.957** (overall 0.973). Pooling the failing domain into training resolved the gap — G1020 recovered from 0.589 → 0.957. *(G1020 is now in-distribution; a fully zero-shot claim would need a new held-out source — noted as future work.)*

## DR error analysis

Top confusions (true→pred): No DR→Mild (×443); No DR→Moderate (×384); Mild→No DR (×157); Moderate→No DR (×103)

Weakest grade: Mild (recall 0.41) — the No-DR↔Mild boundary is the early-disease challenge.

## Honest limitations

- DR 5-class >90% is not achievable on EyePACS-grade labels (grader agreement QWK ~0.83–0.85); the defensible >90% target is *referable-DR AUC*, met internally (0.91), ~0.84 externally.
- Glaucoma: single-source (SMDG) training did not transfer to G1020 (AUC 0.59); resolved by multi-source training (G1020 → 0.96). Generalization to a *further* unseen source remains future work.
- ROP lacks a public external set; operating points are population-specific.