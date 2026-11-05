# Limitations

## Clinical

1. **Not a medical device.** No regulatory clearance, no prospective evaluation, no clinical validation. Dataset accuracy establishes neither clinical validity nor clinical utility.
2. **Per-photograph staging at an unseen hospital is not usable** — recall 0.276 / 0.305 / 0.292 for stages 1 / 2 / 3.
3. **Even at the eye level, ~18% of diseased eyes are called Normal** (CI 11–25%).
4. **The probability output is not a probability.** External ECE 0.334; confidence separation between right and wrong 0.041.
5. **The most dangerous error is the most confident**, so confidence cannot be used as a safety gate.
6. **Stage 4/5 and AP-ROP are not externally validated** — n = 4 and n = 12 locked images, pre-registered as such.
7. **No independent grader.** All labels are the corpora's own; no inter-grader κ exists for this task in this project, so the human ceiling is unknown.
8. **Preterm infants only.** On adult fundus photographs the model flagged 41 of 45 eyes as ROP-positive.

## Data

9. **Severe imbalance**: 12.46 : 1, with 89 stage 4/5 training images.
10. **Class–source confounding**: stage_4_5 100% ROP-VL; AP-ROP zero Shenzhen; Stage 3 only 7% Ostrava. Site identity is decodable at 0.997 balanced accuracy. Performance on these classes may partly be source recognition.
11. **One infant dominates AP-ROP evaluation**: 470 of fold 0's 476 AP-ROP validation frames. The patient-clustered CI on AP-ROP recall is [0.254, 0.836].
12. **Labels are not image-level facts.** Ostrava diagnoses are per **session**; Multi-View diagnoses are per **eye** (219/230 eyes carry one label across all views). Many labelled frames cannot contain the finding.
13. **dev and locked halves are not distribution-matched** (Stage 1: 190 vs 105; Normal: 150 vs 232).
14. **Retrospective, stored images only.** No prospective acquisition, no real-world image-quality distribution.
15. **The external test set is CC BY-NC**, which constrains commercial use of any result validated on it.

## Model

16. **The architectural contribution is 6,405 parameters (0.031%)**, and it is measurably equivalent to a plain softmax on every aggregate metric.
17. **It is worse calibrated than that baseline** at the same accuracy, and behind it on external image-level QWK at **both** seeds.
18. **Stage 1 output carries little information** — precision 0.376 [0.273, 0.487].
19. **The decode rule and the probability argmax disagree** in 1.19% of predictions.
20. **The site adversary is 1.6% dead weight at inference** and its specification deviates from canonical DANN.
21. **Grad-CAM does not localise to the retina** better than chance (lift 1.04); 19.3% of peaks land on invented letterbox padding.

## Evaluation

22. **One external hospital.** A sanity check, not a generalisation claim.
23. **The eye-level aggregation rule was selected after seeing dev-half results.** Disclosed upstream, but it means only the image-level locked number is fully pre-registered.
24. **The binary screening threshold was selected partly on the external dev half**, so its external operating point is optimistic.
25. **All ablations are single-fold, single-seed, on the atypical fold.**
26. **Five folds is a small n for paired tests**, and folds share training data, so they are not independent. The reported p-values mean "nothing detectable", not "equivalence demonstrated at power β".
27. **No validation loss exists**, so a classical overfitting diagnosis cannot be made from the available evidence.

## Engineering / reproducibility

28. **The committed config reproduces the documented failure arm**, not this model.
29. **The seed-42 run summary does not record its own sampler setting.**
30. **`set_seed` is incomplete** — runs are approximately, not bit-, reproducible.
31. **`FundusDataset` silently substitutes a black image** for an unreadable file, uncounted.
32. **`requirements.txt` is unpinned**; no per-run environment freeze.
33. **Three declared config keys are read by no code.**
34. **The served path uses no TTA** despite the README stating otherwise.

## Limitations of *this package*

35. Internal errors are **not** view-corrected: Ostrava and Shenzhen carry no acquisition-view metadata, so the anatomical decomposition that explains the external results cannot be repeated internally.
36. The Grad-CAM audit samples **88** images, not the full 663; it corroborates the project's larger run rather than replacing it.
37. The seven-arm ablation table and five-backbone benchmark were **not** re-scored with the corrected QWK — raw predictions do not exist for every arm. Those remain quarantined in [`../19_COMPARISONS/`](../19_COMPARISONS/) with legacy values and a warning.
38. The learning-rate curve is **reconstructed**, not measured.
39. No new training was performed. Everything here is measurement of existing artifacts, by design.
