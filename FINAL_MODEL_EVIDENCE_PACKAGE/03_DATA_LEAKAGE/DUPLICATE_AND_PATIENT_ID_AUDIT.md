# Duplicates, label conflicts, and what `patient_id` actually identifies

*Verification pass 2. Every image in the eligible corpus (8,024 files, 0 missing) was hashed
with MD5 and with a 256-bit dHash. Tables:
[`exact_duplicate_audit.csv`](../20_AUDIT/verification_pass_2/03_DATA_LEAKAGE/exact_duplicate_audit.csv),
[`exact_duplicate_members.csv`](../20_AUDIT/verification_pass_2/03_DATA_LEAKAGE/exact_duplicate_members.csv),
[`label_conflict_audit.csv`](../20_AUDIT/verification_pass_2/03_DATA_LEAKAGE/label_conflict_audit.csv),
[`perceptual_duplicate_audit.csv`](../20_AUDIT/verification_pass_2/03_DATA_LEAKAGE/perceptual_duplicate_audit.csv),
[`split_leakage_matrix.csv`](../20_AUDIT/verification_pass_2/03_DATA_LEAKAGE/split_leakage_matrix.csv),
[`patient_id_semantics.csv`](../20_AUDIT/verification_pass_2/03_DATA_LEAKAGE/patient_id_semantics.csv).*

## Scope correction to the existing verdict

`03_DATA_LEAKAGE/leakage_recomputation.json` records `"verdict": "PASS"` on patient and image
**path** overlap. That result is correct and is reproduced here. It is also narrower than it
reads: no prior audit in this project hashed image *content* across folds.
`results/rop/staging_shortcut_audit.json` reports `G2_duplicate_leakage {pass: true, exact_md5: 0}`,
and [scripts/rop_shortcut_audit.py:110](../../scripts/rop_shortcut_audit.py) shows its scope is
**"train pool vs held-out hospital"** only. Neither statement covers fold-to-fold duplication.

## 1. No leakage across the boundaries that matter — confirmed

| Comparison | Shared patient IDs | Shared paths | Shared MD5 |
|---|---:|---:|---:|
| train_cv vs multiview_dev | 0 | 0 | **0** |
| train_cv vs multiview_locked | 0 | 0 | **0** |
| multiview_dev vs multiview_locked | 0 | 0 | **0** |
| every CV fold, train vs val | 0 | 0 | — |

The five validation folds exactly partition the 6,700-image pool (1,691 + 1,252 + 1,252 +
1,252 + 1,253), so pooled out-of-fold predictions score each image once. **The locked test is
clean.** No result in this package is contaminated by test leakage.

## 2. Exact duplicates inside cross-validation — confirmed, small, real

10 duplicate groups (20 images). None cross a split boundary. **Four cross a fold boundary:**

| MD5 | Copy A | Copy B | Conflict |
|---|---|---|---|
| `4907c9d8…` | shenzhen:8758, fold 0, **stage_3** | shenzhen:29094, fold 1, **stage_2** | fold + label + ID |
| `543e153b…` | shenzhen:2143, fold 3, **stage_2** | shenzhen:2185, fold 1, **stage_1** | fold + label + ID |
| `df9556c7…` | shenzhen:32004, fold 2, **stage_2** | shenzhen:18405, fold 3, **stage_3** | fold + label + ID |
| `65ec6bbd…` | shenzhen:10057, fold 0 | shenzhen:31918, fold 1 | fold + ID |

A fifth conflict sits inside one fold: `3b13db66…` is `ropvl:525` labelled **arop** and
**stage_3** on a byte-identical file.

**Magnitude: 8 images of 6,700 (0.12%).** Too small to move any reported metric. What it
falsifies is the blanket reading of `G2 pass: true`, and what it demonstrates is item 4 below.

## 3. Near-duplicates — dHash is not discriminative on this corpus

Full-corpus dHash-256, all 8,024 images, every pair:

| Hamming ≤ | Pairs | Cross-split | Cross-fold | Different patient ID | Label conflict | Same source |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 30 | 0 | 4 | 9 | 7 | 30 |
| 2 | 131 | 0 | 37 | 51 | 39 | 131 |
| 4 | 505 | **1** | 243 | 302 | 169 | 503 |
| 6 | 1,453 | 9 | 849 | 1,041 | 584 | 1,429 |
| 10 | 7,680 | 96 | 5,296 | 6,618 | 3,505 | 7,358 |

**Do not read the h≤10 row as 7,680 leaks.** 7,075 of those pairs are Ostrava↔Ostrava, where
one infant contributes up to 470 near-identical frames of the same eye in one session — that
is the corpus, not a defect. Letterboxed fundus frames share so much low-frequency structure
that dHash saturates. The rows worth acting on:

- **h = 0 but not byte-identical: 20 additional pairs** — re-encoded or resized copies of the
  same photograph, invisible to an MD5 check.
- **h ≤ 4, cross-fold, label-conflicting: 169 pairs** — candidates for genuine within-CV
  contamination, requiring visual adjudication.
- **9 cross-split pairs at h ≤ 6, all multiview dev ↔ locked.** Dev and locked are
  patient-disjoint, so these are either coincidental frame similarity between infants or an
  identity error in the source table. **They must be adjudicated visually before the locked
  test is described as fully clean.** They are 9 of 1,324 images.
- **255 Ostrava ↔ ROP-VL pairs** — cross-source near-duplicates, consistent with the
  re-upload problem the inclusion flow already excludes elsewhere.

## 4. `patient_id` is not a person for two of four sources

| Source | Images | Distinct IDs | Images/ID | IDs with 1 image | Max images/ID | Identical images under **different** IDs |
|---|---:|---:|---:|---:|---:|---:|
| ostrava | 4,285 | 148 | 28.95 | 0 | 470 | 0 |
| multiview | 1,324 | 232 | 5.71 | 1 | 9 | 0 |
| ropvl | 1,675 | 983 | 1.70 | 450 | 6 | 0 |
| **shenzhen** | 740 | 397 | 1.86 | 150 | 12 | **6** |

Six byte-identical photographs are registered under two different Shenzhen `patient_id`
values. A person identifier cannot do that; a per-record identifier can.

### Consequences that must be carried with every number in this package

1. **"1,528 infants" and "1,760 infants" are upper bounds on distinct people**, not counts.
   Ostrava (148) and Multi-View (232) behave like people; ROP-VL (983) and Shenzhen (397)
   behave like records.
2. **Every patient-clustered bootstrap in this project under-corrects.** Resampling a key
   that splits one person into several units leaves residual within-person correlation, so
   the intervals are narrower than the true clustering warrants. This applies to the
   calibration CIs, the CV bootstrap, and the eye-level CIs computed in this pass. The
   direction of the bias is known (too narrow); its size is not.
3. Person-level independence must not be claimed for ROP-VL or Shenzhen anywhere.
4. The locked and dev halves are Multi-View only, where the ID is patient-like — so the
   **test-set** intervals are the least affected.

## Verdict by category

**Confirmed leakage** — 4 exact-duplicate pairs across CV folds (8 images, 0.12% of the pool),
3 of them with contradictory labels; 1 further label contradiction within a fold.

**Potential leakage** — 169 cross-fold near-duplicate pairs with conflicting labels at h ≤ 4;
9 cross-split near-duplicate pairs in Multi-View; 255 cross-source near-duplicates.
All require visual adjudication before classification.

**No leakage detected** — patient, path and byte identity across train_cv / dev / locked, and
across train/val inside every fold. Verified independently by hashing, not by documentation.

**Cannot determine** — whether ROP-VL and Shenzhen IDs correspond to distinct people;
whether the same examination appears under different IDs; acquisition dates are not recorded
in any manifest, so temporal duplication cannot be tested at all.
