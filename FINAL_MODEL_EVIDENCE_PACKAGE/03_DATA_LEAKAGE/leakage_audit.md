# Data Leakage Audit — recomputed from scratch

*Script: `20_AUDIT/scripts/s02_dataset_leakage.py`. Tables: [`patient_overlap_results.csv`](patient_overlap_results.csv),
[`image_overlap_results.csv`](image_overlap_results.csv). Figure: [`leakage_summary.png`](leakage_summary.png).*

**Verdict: no leakage detected. Every comparison returns zero.**

## Patient-level (the one that matters most — one infant contributes up to 470 frames)

| comparison | group A | group B | overlapping infants |
|---|---|---|---|
| fold_0 train vs val | 1,231 | 297 | **0** |
| fold_1 train vs val | 1,229 | 299 | **0** |
| fold_2 train vs val | 1,256 | 272 | **0** |
| fold_3 train vs val | 1,201 | 327 | **0** |
| fold_4 train vs val | 1,195 | 333 | **0** |
| fold_site train vs val | 1,131 | 397 | **0** |
| CV corpus vs external dev | 1,528 | 116 | **0** |
| CV corpus vs external locked | 1,528 | 116 | **0** |
| external dev vs external locked | 116 | 116 | **0** |

## Image-level

| comparison | overlapping images |
|---|---|
| every fold, train vs val | **0** |
| CV corpus vs external dev | **0** |
| CV corpus vs external locked | **0** |
| external dev vs external locked | **0** |
| all 10 pairs of validation folds | **0** (folds must be disjoint — they are) |

**The five validation folds exactly partition 6,700 images** (union size = sum of sizes = 6,700). This is a stronger property than "no overlap": it proves the cross-validation is a true partition with no image counted twice and none dropped.

## Other leakage channels

| channel | status | evidence |
|---|---|---|
| Near-duplicate images across splits | **not detected** | 256-bit dHash, closest train↔held-out pair at distance 15 vs threshold 4 (parent gate G2) |
| Augmentation leakage | **not possible** | augmentation is applied inside the DataLoader per split; no augmented copies are written to disk or shared between splits |
| Preprocessing fitted before splitting | **not present** | all preprocessing is per-image and stateless (crop, pad, CLAHE, resize, fixed ImageNet normalisation). No dataset-level statistic is fitted. |
| Target leakage from filenames | **not present in the model** | file paths never reach the network; the loader reads pixels only. (Metadata *is* highly informative — see below.) |
| **Site / device shortcut** | **PRESENT but bounded** | hospital identity is linearly decodable at 0.997 balanced accuracy from file metadata. A metadata-only classifier reaches macro-F1 0.187; the image model reaches 0.692, so the model is not merely reading the shortcut — but the shortcut exists and the DANN branch exists because of it. |
| **Class–source confounding** | **PRESENT, partly under-declared** | stage_4_5 is 100% ROP-VL; arop has zero Shenzhen images; Stage 3 is only 7% Ostrava. Performance on these classes may partly be source recognition. Only stage_4_5 is formally declared unvalidated upstream. |
| Test-set information influencing training | **not detected** | the locked half was opened once, after training, under a protocol committed beforehand; the script refuses a second run |
| **Selection on evaluation data** | **PRESENT and disclosed** | (a) the binary screening threshold 0.0155 was chosen partly on the external **dev** half, so its external operating point is optimistic; (b) the eye-level co-primary unit was adopted after inspecting dev-half results. Both are documented upstream; neither contaminates the locked *image-level* number, which remains the only fully pre-registered external result. |

## Classification of the leakage finding

- **Confirmed leakage:** none.
- **Suspected leakage:** none.
- **Not detected from available evidence:** patient, image, near-duplicate, augmentation, preprocessing, target.
- **Impossible to determine from the artifacts:** whether any two infants across *different* cohorts are the same person (no cross-cohort identity key exists). The parent project merged one such pair inside Ostrava (IDs 40/41) and kept 13 same-(sex,GA,BW) pairs separate as NICU twins.
