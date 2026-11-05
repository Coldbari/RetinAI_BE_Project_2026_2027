# Duplicate and Near-Duplicate Analysis

## Exact duplicates

Recomputed here at the path level: **0** shared image paths between any train and any evaluation split (see [`image_overlap_results.csv`](image_overlap_results.csv)).

Byte-level MD5 duplication across splits was measured upstream at **0** (`results/rop/staging_shortcut_audit.json`, gate G2).

## Near-duplicates

Upstream gate G2 used a **256-bit difference hash** over the corpus. Result: the closest train ↔ held-out pair sits at Hamming distance **15**, against a duplicate threshold of **4**. No near-duplicate leakage.

**Why 256-bit and not the usual 64-bit:** the parent project measured that a 64-bit dHash is *unsafe on fundus images* — it matched a Stage 2 image to a Normal image at distance 0. This is a genuine methodological contribution and worth preserving: perceptual hashing calibrated on natural photographs does not transfer to low-texture, radially symmetric retinal images.

## Duplicates removed before the splits were built

- **20 re-encoded duplicates** between ROP-VL and Shenzhen (the same hospital contributed both cohorts) were removed by the manifest builder.
- **Ostrava IDs 40 and 41 are one infant** — 19 byte-identical JPEGs, identical sex/GA/birth-weight, contradictory diagnosis codes on identical pixels. Merged. Thirteen other same-(sex, GA, BW) ID pairs share no images and were kept separate as presumed NICU twins.

## Residual risk

The only channel that cannot be closed from the artifacts is **cross-cohort patient identity**: there is no key that would reveal whether an infant photographed at Shenzhen also appears in ROP-VL under a different ID. The dHash sweep is the practical proxy, and it is clean at distance 15.

This matters least for the external hospital (a different country and a different vendor from all three training cohorts) and most for the Shenzhen / ROP-VL pair, which *is* the same hospital. Those two cohorts are always on the same side of the site fold, so a shared infant would inflate CV but not the external result.
