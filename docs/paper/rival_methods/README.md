# What the rival ROP papers did, and what RetinAI should adopt (2026-10-30)

Four method-level reviews sit next to this file. Every fact in them is tagged FT (full text read),
AB (abstract only), LOCAL (checked against our own data) or INF (inference).

| File | Papers |
|---|---|
| `01_ostrava_rivals.md` | Vahidmoghadam 2026 (Sci Rep), ROPDeepX 2026, Hassan & Hasan 2026 (CAA Ensemble), Kian Ara 2026 (Front Med); all FT |
| `02_multicentre_and_foundation_models.md` | VEA-Net 2026, SABPI-Net (MICCAI 2025, with code), ROP-Bench (code), DeepAdapter 2026, Mahboob SPIE 2025 (AB), Lan TVST 2026 (preprint FT) |
| `03_vit_papers_and_datasets.md` | ICROP3 hybrid (IJASIS 2026), ROPGCViT, IConSCEPT Swin-T, DS-Swin-T, DeepROP + its Kaggle set (AB), Shenzhen/Zhao 2024 and HVDROPDB dataset papers (FT), ROP-VL (AB) |
| `04_data_generation_and_training_recipes.md` | Evidence on GAN/diffusion data, augmentation, ViT fine-tuning recipes, new real data; 20-row ranked table |

## How the 95–97% numbers arise

- **Photo-level splits on multi-frame cohorts**: Vahidmoghadam (Ostrava, 37-image stage test, 97.3%) and
  ROPDeepX (99.7%), FT.
- **Tiny test sets, many models tried**: ROPGCViT's 94.69% is 107/113, the Shenzhen published test split; 50 models
  were compared on it (95% CI 88.9–97.5).
- **Duplicates and augmented copies**: DeepROP's Kaggle set has 91 files but 85 unique images; 96.5% is not a
  correct-count ratio for any test set under 57 images (LOCAL + INF).
- **Not credible**: the ICROP3 hybrid paper (IJASIS 2026) cites dataset URLs that do not resolve and an Ostrava DOI
  (10.1016/j.biomedimag.2023.105982) that returns 404; its class counts match no real dataset. Do not cite as
  evidence.
- **In-distribution only**: DS-Swin-T 97.9% in-style, at best 93.7% on other styles.
- On Shenzhen with patient-grouped CV our DINOv2 scores 0.90, above the dataset's own baselines (ResNet50 87.6%,
  ViT-B 84.1%).

## Can we generate data?

- **Geometric and photometric augmentation (rotate, flip, colour, blur, crop):** yes, and we already do most of it
  (`models/common/preprocessing.py`). What is missing is listed in the plan below.
- **GAN or diffusion synthetic images: no.** No study shows synthetic images improving ICROP stage classification.
  The best-controlled fundus study with an external test (Nderitu 2025) gained +0.024 AUROC internally and lost
  0.039 externally. Generators drop fine lesions (mild-DR images came out as no-DR), the very failure that would hit
  the Stage 1 demarcation line, and memorise small classes (up to 54.7% of a 5k training set). Our scarce classes
  are 89 (Stage 4/5) and 577 (Stage 1) photos.

## Already done, so not new levers

Eight backbones within ±0.02 QWK (incl. RETFound-Green, EfficientNetV2-S), 518 px input, ensembles, ordinal
decoder, mean-teacher adaptation, exam-level bag loss (helped on Ostrava exams only), a CV-fixed 90%-sensitivity
operating point (Kian Ara's rule; it did not transfer: 0.65–0.68 at the new hospital), class-balanced CE, label
smoothing, layer decay, drop path, site randomisation.

## Plan, ranked by evidence and cost

**Rule for every item:** choose by the pre-registered CV rule and by leave-one-hospital-out, never by the Multi-View
development half again; the locked half stays for one final scoring.

Tier 1: cheap, evidence-backed (one Kaggle run per variant per seed)

1. **Leave-one-hospital-out evaluation**: train on two of Ostrava / ROP-VL / Shenzhen, test on the third. Three
   more unseen-hospital tests without touching Multi-View; ROP-Bench shows in-domain 0.93–1.0 falling to 0.74–0.87
   cross-site, so effects will show.
2. **Weight averaging**: EMA of weights (the code exists from arm F) or SWA over the last epochs. DomainBed SWAD
   +1.6 pp; fundus DRGen 79.3 vs 75.9 AUC. ~0 cost.
3. **LP-FT**: 1–2 head-only epochs before full fine-tuning (Kumar 2022, ~+10% OOD on natural images; untested on
   fundus).
4. **Augmentation fixes**: full 360° rotation; gamma and Gaussian noise; raise RandomResizedCrop minimum scale from
   0.6 to ~0.85, because a 0.6 crop can cut off the peripheral Stage 1 line while keeping the Stage 1 label
   (DeepAdapter uses resize-and-pad for this reason); ablate colour-jitter strength (it helped in GDRNet, +4.5 AUC,
   and hurt in RADR, −5 QWK).
5. **Fourier low-frequency style swap + embedding-consistency loss** (SABPI-Net DAESSM, FDA family): swap a 2×2–3×3
   centred FFT amplitude block between training images, add a cosine-consistency term (λ 0.01–0.1). Their ablation:
   accuracy 93.14 → 94.24. Source-only, so no target data is used.
6. **Label smoothing 0.1** and **horizontal-flip TTA** at inference (no retraining).

Tier 2: worth one experiment each

7. **Auxiliary ridge supervision** from HVDROPDB's 100 ridge/demarcation masks (light segmentation head on training
   folds); aimed at Stage 1, weak evidence.
8. **Vessel map as a 4th input channel**, never replacing RGB (Hassan & Hasan: plus AUC 0.965 → 0.995; untested for
   stage).
9. **Self-supervised continued pre-training** (DINOv2 objective) on all unlabelled infant photos from the training
   cohorts (DeepAdapter: external 0.739 → 0.828, binary only). Expensive on a T4; never include locked images.

Not recommended: GAN/diffusion data, plain mixup/CutMix (GDRBench 75.0 vs 75.9 without), RandAugment, DANN, FixMatch,
SAM, distillation, higher resolution, more backbones.

## The biggest levers are not training tricks

- **Label quality**: Stage is where ROP experts disagree most (Campbell 2016). Two or three ophthalmologists
  re-grading the Stage 1 and borderline photos, and selecting models on that adjudicated set, is the best-supported
  route to higher Stage 1 numbers (DR analogue: Krause 2018).
- **More real hospitals**: going from one camera to three raised DR QWK from 68.6 to 74.2 (FT). Candidates:
  Macretina (India, Dec 2025; normal, pooled ridge, 248 A-ROP; download not confirmed live). Note that Hugging Face
  `lijuanliao/ROP-DATASET` *is* Multi-View ROP, our held-out hospital: never train on it. FARFUM-RoP's "stage"
  column behaves like zone (checked 2026-10-18), so it is not stage data.
- Our Ostrava DG mapping was re-checked against the dataset paper's Table 10 (DG2 = Stage 1, 252 images); the
  "DG 2 – hemorrhage" example sentence in that paper is a typo.
