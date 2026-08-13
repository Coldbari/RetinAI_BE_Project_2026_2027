# ICROP staging research preview — checkpoint provenance

- Model: StructuredROPModel (EfficientNetV2-S backbone, ordinal CORN stage head,
  AP-ROP branch, site-adversary head inert at inference)
- Checkpoint: fold 0 of the 5-fold CV, seed 42, 20 epochs, natural sampling,
  last-epoch checkpoint (no argmax-over-epochs), trained on Kaggle
  (kernel rop-structured-effnetv2s-s42).
- 5-fold CV for this configuration: macro-F1 0.692 +/- 0.086, QWK 0.826 —
  statistically equivalent to a flat softmax; served for taxonomy-faithful outputs.
- The held-out hospital's locked test set has never been opened.
- NOT a clinical grade. Research preview only.
