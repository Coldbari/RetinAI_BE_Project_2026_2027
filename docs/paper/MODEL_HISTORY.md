# RetinAI model history (June–October 2026)

For the report's methodology chapter and any paper's "development history" paragraph. Every number is
sourced; where a later re-scoring changed a definition, both are shown. Written 2026-10-31.

## Did the project use classical machine learning?

Not as a diagnostic model. The project began with a deep CNN (EfficientNet-B3), the standard for
photographic classification since ~2016. Classical learners appear only as tools:
- **Audit probes (Aug 2026):** logistic regression and a random forest on image width/height alone
  predicted ROP with AUC 0.911 (`scripts/rop_shortcut_audit.py`), proving a device shortcut.
- **Frozen-feature heads (spike, 2026-10-18):** logistic / MLP / attention heads on frozen encoder
  features; best frozen CV QWK 0.859 vs 0.917 fine-tuned (`experiments/eye_spike/RESULTS.md`).

Path: CNN (v1) → CNN comparison, ResNet50 then EfficientNetV2-S (v2) → vision transformers (v3).

## Chronology

| When | Model(s) | Task / data | Result | Fate | Source |
|---|---|---|---|---|---|
| 2026-06-08, v1 | EfficientNet-B3, one shared model for ROP + DR | ROP yes/no (5,004 Ostrava images); DR 35,126 | ROP test acc 73.3%, F1 0.71; DR macro-F1 0.50 | Retired: image-level split, two diseases in one model | `junk/unified_model/evaluation_unified.txt` |
| 2026-06-28, W6 sweep | EfficientNet-B0, EfficientNetV2-S, MobileNetV3-L, ResNet50, DenseNet121 | ROP yes/no, Ostrava | ResNet50 best, AUC 0.964 (val), 0.927 test | ResNet50 deployed as screener | `results/sweep_rop.md`, `results/rop/metrics.json` |
| 2026-06/07 | EfficientNetV2-S | Glaucoma; referable DR | Glaucoma AUC 0.973; DR AUC 0.906, QWK 0.728 | Glaucoma withdrawn (G1020 double-mount); DR parked | `19_COMPARISONS/retired_models/glaucoma_and_dr.md` |
| 2026-08-02, T9 | ResNet50, device-stratified retrain | ROP yes/no | Device confound confirmed; all ROP+ test images from one camera | Retired 2026-08-19: flagged 663/663 images at a new hospital, external AUC 0.691 | `results/rop_device_audit.md`, `retired_models/resnet50_binary_screener.md` |
| 2026-08-07–09, v2 sweep | EfficientNetV2-S, DeiT3-S, ConvNeXt-T (in12k), CAFormer-S18, EfficientNet-B0 | 6-class staging, 6,700 photos, fold 0 + site fold, seed 42 | fold-0 macro-F1 0.492–0.520 (inside run noise); cross-site macro-F1 0.641 (EffNetV2-S) vs 0.507–0.536 | EfficientNetV2-S selected | `retired_models/five_backbone_benchmark.md` |
| 2026-08, v2 | EffNetV2-S structured (CORN ordinal + AP-ROP branch + site head) vs flat softmax, seeds 42/1337; v1 arm with balanced sampler | 6-class, 5-fold patient-grouped CV | CV QWK 0.896 [0.869, 0.923], acc 0.786, macro-F1 0.652, any-ROP AUC 0.970. Sealed half (opened once, 2026-08-18): per photo QWK 0.493, acc 0.546, AUC 0.835; per eye (max rule) QWK 0.839, acc 0.678. Balanced-sampler arm QWK 0.121 | Structured ≈ flat; app model | `FINAL_MODEL_EVIDENCE_PACKAGE/00_START_HERE/MODEL_CARD.md` |
| 2026-10-18 spike | Frozen RETFound-Green / DINOv2 / EffNet features + logistic, MLP, gated-attention MIL | eye/exam-level | best frozen CV QWK 0.859 < fine-tuned 0.917; attention-MIL −0.080 QWK vs worst-view rule at the new hospital | Fine-tune a retina/transformer encoder; keep the worst-view rule | `experiments/eye_spike/RESULTS.md` |
| 2026-10, v3 | A RETFound-Green ViT-S/14 · B DINOv2 ViT-S/14 · C EffNetV2-S (re-scored) · D DINOv2 + exam bag loss · E ensemble A+B+C · F DINOv2 + self-ensembling · G DINOv2 at 518 px; seeds 42/1337 | 6-class, 6,700 CV photos; 663 new-hospital photos scored per eye | CV QWK5 0.794 / 0.825 / 0.816 / 0.797 / 0.821 / 0.812 / 0.814. New hospital per eye QWK5 0.774 / 0.756 / 0.760 / 0.746 / 0.770 / 0.744 / 0.767; B: acc 0.630, macro-F1 0.566, any-ROP AUC 0.937 | B selected by the pre-registered CV rule | `results/rop/v3/v3_scores.json`, `report_data.json` |

Totals: 11 distinct architectures, ~20 trained arms (heads, losses, training variants), 4 frozen-feature heads.

## Comparability cautions

- v2's EfficientNet CV QWK 0.896 vs v3's 0.816 for the same architecture is not a drop: v2 used a different
  checkpoint rule and QWK definition; v3 re-scored it under the stricter pre-registered rules (last epoch,
  5-class map, AP-ROP folded, unit-level pooling).
- v2's eye-level 0.839 was on the sealed half with max-of-hard-labels; v3's 0.756 is on the development half
  with max-of-predicted-stage. Different data and rule.
- The June binary numbers (0.96, 0.93) were inflated by the device shortcut; the v2 protocol's shortcut audits
  exist because of them.
