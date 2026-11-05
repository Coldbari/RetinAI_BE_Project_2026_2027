# Data Quality Report

## What was checked, and by whom

| check | result | source |
|---|---|---|
| Patient-level leakage, all splits | **PASS — 0 overlaps** | recomputed here, `03_DATA_LEAKAGE/` |
| Exact image duplication across splits | **PASS — 0 overlaps** | recomputed here |
| Validation folds partition the corpus exactly | **PASS** — 5 disjoint folds covering 6,700 images | recomputed here |
| Near-duplicate (perceptual) leakage | **PASS** — closest train↔held-out pair at 256-bit dHash distance 15 vs threshold 4 | parent project gate G2 (`results/rop/staging_shortcut_audit.json`) |
| Exact MD5 duplicates | 0 | parent project gate G2 |
| Hospital decodable from file metadata | **0.997 balanced accuracy** — the shortcut exists | parent project gate G3 |
| Metadata-only classifier floor | CV macro-F1 **0.187**, dev macro-F1 0.062 | parent project gate G4 |
| Missing or unreadable image files | **0** — all 4,165 fold-0, 663 dev and 661 locked paths resolve on disk | recomputed here |
| Label-order integrity between manifests and prediction artifacts | **PASS** — asserted row-by-row for all 5 folds | `s03_cv_metrics.py` assertion |

## Known data defects that survive into the current model

### 1. Label unit — session codes stamped onto frames
Ostrava diagnoses are recorded **per examination session** and inherited by every frame captured in it. Multi-View diagnoses are recorded **per eye** (219 of 230 locked eyes carry one label across all their views) and inherited by every acquisition view. Many of those frames physically cannot contain the finding. This is the single largest source of apparent "error" in the per-image metrics and is quantified in [`../11_AGGREGATION_AND_VIEW_ANALYSIS/`](../11_AGGREGATION_AND_VIEW_ANALYSIS/).

### 2. One infant dominates a fold
`ostrava:43` contributes 470 of fold 0's 476 AP-ROP validation frames. Fold 0's macro-F1 (0.553 vs 0.67–0.76 elsewhere) and its calibration error (0.201 vs 0.058–0.093) are both driven by this. **The served checkpoint is fold 0's** — i.e. the checkpoint that never saw this infant.

### 3. Class–source confounding
See [`dataset_summary.md`](dataset_summary.md). stage_4_5 = 100% one source; arop = 0% Shenzhen.

### 4. Tiny external cells
The locked set has **4** stage_4/5 images and **12** AP-ROP images. Those cells are pre-registered as *not externally validated* and are excluded from every headline in this package.

### 5. Silent corrupt-image substitution
`models/common/datasets.py::FundusDataset.__getitem__` returns a **zero tensor carrying the true label** when a file cannot be read, with no log line and no counter. No failure occurred in the runs behind this package (all paths verified present), but the failure mode is silent by construction and should be made loud.

## Data engineering carried out upstream (inherited, not re-verified here)

- **Label recovery**: Ostrava stage labels recovered from filename DG codes, joined on `(ID, device, serie)`; 6,004/6,004 resolve. 43% of the old binary "ROP-positive" class turned out not to be ROP.
- **De-identification**: original Ostrava filenames encoded sex / gestational age / birth weight / postnatal age; the tree was rebuilt as `<id>_D<dev>_S<serie>_<frame>.jpg`.
- **Deduplication**: 20 re-encoded duplicates between ROP-VL and Shenzhen removed using a 256-bit dHash (64-bit was measured unsafe on fundus images).
- **Identity merge**: Ostrava IDs 40 and 41 were the same infant (19 byte-identical JPEGs); merged in the manifest builder.

> ⚠️ **Privacy note, outside this package.** The parent repository's own PHI guard test currently **fails** on `DEMO.md`, which contains two original Ostrava filenames encoding sex, gestational age and birth weight (values redacted here) for one infant. This package contains no such identifiers — see [`../20_AUDIT/privacy_audit.md`](../20_AUDIT/privacy_audit.md).
