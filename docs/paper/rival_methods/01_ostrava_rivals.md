# Method-level reading: four 2026 papers on the Ostrava ROP data (agent 1)

Compiled 2026-10-30. Retrieval used WebFetch/WebSearch only. Three publisher PDFs and the arXiv PDF came back through WebFetch's normal redirect handling (cookie hand-off) and were read page by page. No paywall, bot check or user-agent workaround was used.

## Tag legend

| Tag | Meaning |
|---|---|
| **[FT]** | Read in the paper's full text, including its tables and figures |
| **[FT-derived]** | Arithmetic I did on FT numbers (e.g., per-class recall from a printed confusion matrix) |
| **[FT-fig]** | Read off a figure, legend, or axis; less precise than a printed table |
| **[FT: Timkovič 2024]** | Read in the full text of the Ostrava dataset paper (Sci Data 11:814, 2024), not in the rival paper |
| **[REPO]** | Read in a code-repository README, not in the paper |
| **[AB]** | Abstract only (none of the four papers needed this) |

"Not stated" means the paper does not give the value. Nothing in these tables is guessed.

## Access level

| # | Paper | Access | What I read |
|---|---|---|---|
| 1 | Vahidmoghadam et al., Sci Rep 2026 | **FT** | PMC HTML (PMC12923887) and the Nature PDF (18 pp) |
| 2 | Mohiy et al. (ROPDeepX), Discov Artif Intell 2026 | **FT** | Springer HTML (tables did not render) and the Springer PDF (25 pp, all tables read from the PDF) |
| 3 | Hassan & Hasan (CAA Ensemble) | **FT of arXiv v2** (2602.05208v2, 19 Feb 2026, 16 pp) | I could not find or verify the Comput Biol Med version of record; a web search found only the arXiv preprint. Numbers below are from arXiv v2 and may differ from the journal version. |
| 4 | Kian Ara … Timkovič, Front Med 2026 | **FT** | Frontiers HTML and PDF (28 pp). The Supplementary Tables S1–S7 were **not** read. |

---

## 0. Cross-paper synthesis (read this first)

### 0.1 Comparison at a glance

| | P1 Vahidmoghadam (Sci Rep) | P2 ROPDeepX (Discov AI) | P3 CAA Ensemble (arXiv / CBM) | P4 Kian Ara (Front Med) |
|---|---|---|---|---|
| Data | Public Ostrava: 6,004 images / 188 infants. Stage subset is 880 images. [FT] | FARFUM-RoP (Iran) for training; Ostrava ("RIDIRP") and HVDROPDB v3 as "cross-dataset" tests [FT] | Public Ostrava: 6,004 images / 188 infants, plus GA/BW/PA metadata [FT] | **Non-public** Ostrava Phoenix ICON archive: 732 images / 38 infants, **with eye laterality** [FT] |
| Task | Plus vs normal; Stage 0–3 [FT] | Normal / Pre-plus / Plus, and Normal vs ROP [FT] | 4 classes (Normal / Mild / Severe / Other), plus Plus [FT] | Plus vs non-plus [FT] |
| Split | **Image-level** 80/16/4. No patient grouping is stated. [FT] | FARFUM: "3-fold CV, 70% of patients"; no grouping is stated for the other datasets [FT] | **Patient-level**: 17-infant test set, StratifiedGroupKFold on 171 infants [FT] | **Infant-disjoint**: 30 development infants and an 8-infant hold-out [FT] |
| Prediction unit | Image [FT] | Image [FT] | Image (implied by the 668-image test set) [FT] | Image, with infant-weighted analyses on top [FT] |
| External / unseen hospital | No [FT] | Claimed (Ostrava, HVDROPDB) but inconsistent (§2.11) [FT] | No (single centre) [FT] | No (same hospital, unseen infants) [FT] |
| Headline | Plus test 1.000 on all metrics; Stage test accuracy 0.973 [FT] | FARFUM 3-class accuracy 96.9%; RIDIRP 3-class 99.7% [FT] | Macro-F1 0.922, κ 0.942; Plus AUC 0.999 [FT] | DenseNet121 AUC 0.929, sensitivity 0.647; infant-equal sensitivity 0.394 [FT] |
| Credibility | Low: leaky split and perfect scores | Very low: internally impossible numbers | Medium: honest split, but 17 test infants and a metadata shortcut risk | **High**: careful and self-critical, but small |

### 0.2 Signals that recur across papers and matter to RetinAI

1. **Exam-derived labels are weak labels for single images.**
   - P4 [FT]: two ophthalmologists rated **14/18 and 17/18** plus-labelled images as *Healthy* when they saw each image alone.
   - Ostrava labels attach to a series, i.e., an examination [FT: Timkovič 2024].
   - This is independent evidence for RetinAI's v3 finding that the exam-level bag loss is the arm that wins. It is also a reason to expect a ceiling on image-level Stage 1 recall.
2. **Image-only models struggle with Mild / Stage 1 even on Ostrava with a patient-disjoint split.**
   - P3 [FT], Mild/Treated sensitivity: baseline EfficientNet-B0 **0.316**, MS-AQNet (image + query) **0.214**.
   - The gain to 0.971 appeared only after late fusion with metadata and the plus stream.
   - RetinAI's Stage 1 recall of about 0.5 at an unseen hospital is therefore not out of line with the field.
3. **Use vessel maps alongside RGB, never instead of it.**

   | Paper | Comparison | Result | Tag |
   |---|---|---|---|
   | P1 | Original + mask vs original only | 0.98 vs 0.94 accuracy (leaky split) | [FT] |
   | P1 | Mask only | 0.86 accuracy | [FT] |
   | P3 | RGB + Frangi channel vs RGB only (plus MIL) | AUC 0.995 vs 0.965 | [FT] |
   | P4 | Green vessel overlay on a dimmed background vs raw fundus | AUC **0.906** vs **0.959** | [FT] |

4. **Smaller models win on about 188 infants.** P3 Table 5 [FT]:

   | Backbone | Macro-F1 |
   |---|---|
   | EfficientNet-B0 (7.44 M params) | 0.92 |
   | ConvNeXt-T | 0.87 |
   | ResNet-50 | 0.79 |
   | Inception-v3 | 0.77 |

   This is consistent with RetinAI getting no gain from EfficientNetV2-S, 518 px input, or ensembles.
5. **Every rival omits stage 4/5.** The Ostrava dataset has **no images of ROP 4A, 4B or 5** [FT: Timkovič 2024]. Stage 3 has only **125** images in P1's stage subset [FT].

---

## 1. Paper 1: Vahidmoghadam et al., Sci Rep 2026

### 1.1 Citation and access
- Mahdi Vahidmoghadam, Parisa Ghorbani, Mohammad Javad Ahmadi (equal contribution), Esmaeil Asadi Khameneh, Babak Tavassoli, Hamid D. Taghirad (corresponding). [FT]
- "Automated diagnosis of plus form and early stages of ROP using deep learning models." *Scientific Reports* 16:7234 (2026), published 4 Feb 2026. DOI 10.1038/s41598-026-37064-2. CC BY 4.0. [FT]
- Affiliations: K.N. Toosi University of Technology (ARAS) and Farabi Eye Hospital, Tehran. [FT]
- Access: **FT**.

### 1.2 Datasets
- **Classification data**: the public "Retinal Image Dataset of Infants and ROP" from University Hospital Ostrava, downloaded from Kaggle (`jananowakova/retinal-image-dataset-of-infants-and-rop`). [FT]
  - **188 infants, 6,004 images.** [FT]
  - Cameras: Clarity RetCam 3, Phoenix ICON, Natus RetCam Envision. Per-camera counts are not stated. [FT]
  - Images per infant: not stated. The average is about 32 [FT-derived: 6,004 / 188].
  - **Plus task**: all 6,004 images; **5,375 Normal and 629 Plus**. [FT]
  - **Stage task**: **880 images**: Stage 0 = 45, Stage 1 = 252, Stage 2 = 458, Stage 3 = 125. [FT]
  - How the 880 images were selected, which diagnosis codes map to "Stage 0", and how many infants or cameras the subset covers: **not stated**.
  - Labels "adhered strictly to ICROP guidelines" (demarcation line = stage 1, ridge = stage 2, extraretinal neovascular proliferation = stage 3). [FT]
- **Vessel-segmentation training data** (681 images in total): FIVES (601), HRF (30), "HVDROPDR" (50; this is HVDROPDB, ref 21). [FT]
  - FIVES and HRF are **adult** fundus datasets. Only the 50 HVDROPDB images are infant images.
  - How vessel ground truth was obtained for Ostrava images: not stated. The segmenter was applied without ROP-specific masks.

**Plus split by class** [FT; validation from Fig. 7a, test from Fig. 15a]

| Split | Normal | Plus | Total |
|---|---|---|---|
| Train | 4,300 | 503 | 4,803 |
| Validation | 860 | 100 | 960 |
| Test | 215 | 26 | 241 |

**Stage split by class** [validation and test FT from Figs. 7b and 15b; train FT-derived by subtraction]

| Split | S0 | S1 | S2 | S3 | Total |
|---|---|---|---|---|---|
| Train | 36 | 201 | 366 | 100 | 703 |
| Validation | 7 | 40 | 73 | 20 | 140 |
| Test | 2 | 11 | 19 | 5 | 37 |

### 1.3 Split
- **80% train / 16% validation / 4% test**, "entirely independent, with no overlap in images". [FT]
- **Image-level.** Patient-level separation is not stated anywhere. With about 32 near-sequential frames per infant, train/test leakage is almost certain.
- Test sets: 241 images (plus) and 37 images (stage). [FT]
- A 5-fold CV for stage is also reported; whether folds are grouped by patient is not stated (presumably image-level). [FT]
- External or unseen-hospital test: **no**. The authors call it single-centre. [FT]

### 1.4 Preprocessing
- **Contrast enhancement before segmentation** (Fig. 3A) [FT]:
  - Sequence: **CLAHE → Gaussian blur → CLAHE → Gaussian blur** (the filter pair is applied twice). Clip limit, tile grid, kernel and sigma: **not stated**.
  - "The enhanced image is then reapplied to the original image." The blend weights and operator: **not stated**.
  - The enhanced image goes into the U-Net++ segmenter. A denoising step then cleans the predicted vessel map; the method is **not stated**.
- **Classifier pipeline** (Figs. 4–6): Resize → Normalize → Segmentation → (merge with original) → backbone → Dropout → Linear. [FT]
  - Resize value, normalisation statistics and dropout rate: **not stated**.
- **Input variants for stage** (Table 4) [FT]:
  - (a) original RGB;
  - (b) vessel mask only;
  - (c) four-channel RGB + vessel map;
  - (d) "Original image + Mask", i.e., vessels "merged with the original images". The merge operator is **not stated**.
- **Plus input**: the segmented vessel map **only**. [FT]
- No green-channel extraction, circle crop or ROI masking is described.

### 1.5 Augmentation
- **None described.** No online or offline augmentation and no synthetic, GAN or diffusion images. [FT, absence]

### 1.6 Model
- **Segmenters compared** (validation set, Table 2) [FT]. **U-Net++ was chosen.** Its encoder, input size, loss, epochs and LR are **not stated**.

  | Model | Accuracy | Recall | IoU | Dice |
  |---|---|---|---|---|
  | DeepLabV3 | 0.9036 | 0.6228 | 0.4254 | 0.5967 |
  | DeepLabV3+ | 0.9451 | 0.7172 | 0.5315 | 0.6932 |
  | U-Net | 0.91 | 0.6621 | 0.4564 | 0.6381 |
  | MAnet | 0.9749 | 0.8358 | 0.7453 | 0.8534 |
  | **U-Net++** | **0.9766** | **0.8385** | **0.7562** | **0.8584** |

- **Classification backbones compared**: ResNeXt50, ResNet152, RegNetX3, VGG16, GoogleNet, MobileNetV2, **EfficientNetB4 (chosen)**. [FT]
  - Pretraining: the tables call them "pre-trained models"; the weight source (ImageNet?) is **not stated**. [FT]
  - Input size: **not stated**. Custom modules: none (backbone, dropout, linear classifier). [FT]

### 1.7 Training
- Loss: weighted (binary) cross-entropy for plus, citing Mao et al. ICML 2023. [FT]
- Class weights: inverse to training frequency, w_Plus = N/(2×503) and w_Normal = N/(2×4300), normalised to sum to 1. [FT]
  - After normalisation, w_Plus ≈ 0.895 and w_Normal ≈ 0.105 [FT-derived].
  - Stage-task loss weights are not described.
- Optimiser **Adam, lr = 0.001**, **10 epochs**. [FT]
- No schedule, batch size or early stopping are stated.
- Hardware: **NVIDIA T4, 16 GB**. Inference is about **0.6 s per image**. A TensorFlow Lite deployment is mentioned. [FT]

### 1.8 Evaluation
- Metrics: accuracy, F1 (macro for stage), sensitivity, specificity, PR-AUC, ROC-AUC (one-vs-rest for stage), confusion matrices, t-SNE and Grad-CAM. [FT]
- Unit: **image**. [FT]
- CIs: **bootstrap, 1,000 iterations**, 2.5th/97.5th percentiles, on the **validation** set. [FT]
- Power analysis: one-sided z-test, α = 0.05, power 0.9, reference sensitivity p0 = 0.9 (Bujang & Adnan). [FT]
  - Plus: n ≈ 77 positives needed, about 735 images in total; **achieved power ≈ 0.95**.
  - Stage: n ≈ 165 positives, about 1,154 images; **achieved power ≈ 0.85**.
- Primary reporting is on the **validation set**, which was also used for model selection. The test set is called "supplementary". [FT]

### 1.9 Results

**Plus, validation set (Table 3)** [FT]

| Model | Acc | F1 | Sens | Spec |
|---|---|---|---|---|
| ResNeXt50 | 0.97 | 0.86 | 0.79 | 0.9942 |
| ResNet152 | 0.98 | 0.89 | 0.90 | 0.986 |
| RegNetX3 | 0.95 | 0.68 | 0.52 | 0.9977 |
| VGG16 | 0.95 | 0.71 | 0.59 | 0.9907 |
| GoogleNet | 0.97 | 0.80 | 0.68 | 0.9977 |
| MobileNetV2 | 0.98 | 0.88 | 0.89 | 0.9849 |
| **EfficientNetB4** | **0.996** | **0.98** | **0.99** | **0.997** |

- Validation confusion: TN 857, FP 3, FN 1, TP 99. [FT]
- PR-AUC 0.999 and ROC-AUC 1.000. [FT]
- 95% CIs: accuracy 0.9942–1.0, F1 0.9786–1.0, sensitivity 0.9846–1.0, specificity 0.9958–1.0. [FT]

**Stage input type, EfficientNetB4, validation (Table 4)** [FT]

| Input | Acc | F1 | Sens | Spec |
|---|---|---|---|---|
| Original | 0.94 | 0.91 | 0.93 | 0.98 |
| Mask only | 0.86 | 0.82 | 0.86 | 0.95 |
| 4-channel | 0.91 | 0.91 | 0.91 | 0.96 |
| **Original + Mask** | **0.98** | **0.97** | **0.96** | **0.9925** |

**Stage backbones, Original + Mask input (Table 5)** [FT]

| Model | Acc | F1 | Sens | Spec |
|---|---|---|---|---|
| ResNeXt50 | 0.89 | 0.85 | 0.88 | 0.96 |
| ResNet152 | 0.74 | 0.74 | 0.82 | 0.91 |
| RegNetX3 | 0.81 | 0.80 | 0.77 | 0.92 |
| VGG16 | 0.78 | 0.71 | 0.69 | 0.91 |
| GoogleNet | 0.85 | 0.82 | 0.84 | 0.94 |
| MobileNetV2 | 0.87 | 0.86 | 0.87 | 0.95 |
| **EfficientNetB4** | **0.98** | **0.97** | **0.96** | **0.9925** |

**Stage, per class, validation** (Fig. 7b) [FT]
- Confusion matrix: S0 row 6/1/0/0; S1 40/40 correct; S2 row 0/2/71/0; S3 20/20 correct.
- Per-class recall: **S0 0.857, S1 1.000, S2 0.973, S3 1.000**. S1 precision 40/43 = 0.930. [FT-derived]
- The errors are S0→S1 and S2→S1, the same boundary RetinAI struggles with.
- Per-class AP (Fig. 10 legend): S0 0.89, S1 0.95, S2 1.00, S3 1.00. Per-class ROC-AUC (Fig. 11 legend): S0 0.98, S1 0.98, S2 1.00, S3 1.00. [FT-fig]
- The text instead says "average AUC 0.975, range 0.96–0.98" (PR) and "0.980, range 0.97–0.99" (ROC); these do not match the legends.
- 95% CIs: accuracy 0.9643–1.0, F1 0.9637–1.0, sensitivity 0.9562–1.0, specificity 0.9584–1.0. [FT]

**Stage 5-fold CV (Table 8)** [FT]

| Metric | F1 | F2 | F3 | F4 | F5 | Mean ± SD |
|---|---|---|---|---|---|---|
| Accuracy | 0.9630 | 0.9886 | 0.9716 | 0.9536 | 0.9866 | 0.9727 ± 0.0147 |
| F1 | 0.9552 | 0.9763 | 0.9555 | 0.9485 | 0.9722 | 0.9615 ± 0.0119 |
| Sensitivity | 0.9624 | 0.9702 | 0.9545 | 0.9496 | 0.9696 | 0.9613 ± 0.0087 |
| Specificity | 0.9734 | 0.9921 | 0.9814 | 0.9621 | 0.9861 | 0.9790 ± 0.0117 |

**Test set (Table 9)** [FT]
- Plus: 1.000 on every metric (215 TN, 26 TP).
- Stage: accuracy 0.9730, macro-F1 0.9058, sensitivity 0.8750, specificity 0.9904. The only error is one of the 2 Stage-0 images predicted as Stage 1.

**Interpretability** [FT]
- Grad-CAM for plus highlights vessel dilation and tortuosity.
- For stage, attention spreads to the optic disc, macula and "overall retinal texture". The authors present this as a strength; I read it as a possible shortcut.

### 1.10 Key references for methods [FT]
1. [18] Timkovič J. et al., Retinal image dataset of infants and ROP, *Sci Data* 2024 (dataset).
2. [19] Jin K. et al., FIVES vessel-segmentation dataset, *Sci Data* 2022.
3. [20] Odstrcilik J. et al., HRF / matched-filter vessel segmentation, *IET Image Process* 2013.
4. [21] Agrawal R. et al., HVDROPDB datasets for ROP research, *Data Brief* 2024.
5. [26] Zhou Z. et al., UNet++, DLMIA/ML-CDS (MICCAI workshops) 2018. [25] Ronneberger, U-Net, MICCAI 2015. [27] Fan, MA-Net, *IEEE Access* 2020. [23/24] Chen, DeepLabV3 (arXiv 2017) / V3+ (ECCV 2018).
6. [28] Reza A.M., CLAHE for real-time image enhancement, *J VLSI Signal Process* 2004.
7. [29] Mao, Mohri & Zhong, cross-entropy loss functions, ICML 2023.
8. [30] Bujang & Adnan, sample size for sensitivity/specificity, *JCDR* 2016.
9. [34] Rahim S. et al., novel fundus preprocessing for RetCam images to improve DL classification of ROP, arXiv 2302.02524 (2023). This is RetCam-specific preprocessing and worth reading.
10. [32] Selvaraju, Grad-CAM, ICCV 2017. [31] van der Maaten & Hinton, t-SNE, *JMLR* 2008.

No reference is given for EfficientNet, the other backbones, or Adam. [FT]

### 1.11 Weaknesses and leakage risks
- **Image-level 80/16/4 split** of about 32 frames per infant: train/val/test almost certainly share infants and near-duplicate frames. This alone explains plus test = 1.000 and ROC-AUC 1.000.
- **Device and label confounding is not examined.** RetinAI's own audit found image dimensions alone give AUC 0.911 on the Ostrava test split. The stage subset's device mix is unreported.
- **The segmenter is trained mostly on adult fundus** (632 of 681 images). Plus uses vessel maps **only**, so errors in infant vessel extraction propagate. No ROP segmentation ground truth was used.
- **The validation set is used for model selection and as the headline result.** The test set (37 stage images, only 2 of them Stage 0) is too small.
- **Internal inconsistencies** [FT]:
  - "≈35 images per class" vs the true 7/40/73/20.
  - "45 images out of 140 validation" for Stage 0; 45 is the dataset total.
  - The Discussion's plus numbers (F1 0.99, sens 0.98, spec 0.99) differ from Table 3 (F1 0.98, sens 0.99, spec 0.997).
  - The Discussion cites "Kavitha [33]", "Zhou [34]" and "Brown [35]", but refs 33–35 are Varysova, Rahim and Agrawal.
  - Legend AUCs differ from text AUCs.
- **No augmentation, 10 epochs, no batch size**: not reproducible.
- **"Stage 0"** is undefined. ICROP has no stage 0; the Ostrava dataset paper mentions "Stage 0" as resolving spontaneously [FT: Timkovič 2024].

### 1.12 What RetinAI could adopt
- **"Original + vessel map" input for staging, as an ablation.** Their only within-paper comparison (same leaky split for all four arms, so the relative ordering is more trustworthy than the absolute numbers) puts original+mask (0.98) above original (0.94), above 4-channel (0.91), above mask-only (0.86).
  - For a DINOv2 ViT, the cheapest version is an alpha-blend of the vessel map into RGB, which keeps 3 channels and the pretrained patch embedding. P4's negative result came from *dimming the background*, so keep the full RGB intensity.
- **A vessel segmenter from public data**: U-Net++ (segmentation_models_pytorch) trained on FIVES + HRF + HVDROPDB. This supplies the vessel channel without manual annotation.
  - Better still for the infant domain: add the HVDROPDB vessel masks and the Ostrava ReLeSeT tooling, or reuse P4's approach of training on ophthalmologist-annotated Phoenix ICON vessels.
- **The power-analysis framing** (Bujang & Adnan) for the paper's limitations section: compute the number of Stage 4/5 and AP-ROP eyes needed to show recall above a reference, and report achieved power. This answers "is your rare-class result meaningful?".
- **Error structure to expect.** Even in a leaky in-distribution setting, the only stage errors are S0↔S1 and S2→S1. Stage 1 is the confusion hub, so treat it as a boundary problem (threshold or metadata), not a capacity problem.

---

## 2. Paper 2: ROPDeepX (Mohiy et al.), Discover Artificial Intelligence 2026

### 2.1 Citation and access
- Esraa Mohiy, Hamada I. AbdulWakel, Marwa Khairy, Essam H. Houssein (corresponding). [FT]
- "An efficient deep learning model for reliable detection and classification of retinopathy of prematurity." *Discover Artificial Intelligence* 6:305 (2026). Received 15 Aug 2025, accepted 1 Jan 2026, published 26 Mar 2026. DOI 10.1007/s44163-026-00834-y. CC BY-NC-ND 4.0. [FT]
- Affiliations: Deraya University, Minia University and Minia National University, Egypt. [FT]
- Access: **FT** (all tables read from the PDF).

### 2.2 Datasets
- **FARFUM-RoP** (Akbari et al., *Sci Data* 2024; public on Figshare). [FT]
  - Farabi Eye Hospital and Ferdowsi University of Mashhad, Iran.
  - **1,533 images from 68 patients**, RetCam Shuttle wide-field.
  - Labelled Normal / Pre-Plus / Plus by **5 pediatric ophthalmologists**.
  - Per-class counts: **not stated**. Images per patient: about 22.5 [FT-derived].
  - **Training source.**
- **RIDIRP**, the Ostrava dataset (Timkovič 2024). [FT]
  - 6,004 images; Clarity RetCam 3, Natus RetCam Envision, Phoenix ICON; 3 experts.
  - Used for "generalization testing (3-class)" and a 2-class version.
  - Per-class counts are **not stated**, and the mapping to Normal/Pre-Plus/Plus is **not stated**.
- **HVDROPDB v3** (Agrawal *Data Brief* 2024): 100 images, RetCam Shuttle / RetCam 3 and 3nethra Neo; H.V. Desai Eye Hospital, Pune, India. Normal vs ROP. Labelling experts: "not specified". [FT]
- **Binary variants**: Pre-Plus and Plus merged into "ROP" for FARFUM-RoP (2-class) and RIDIRP (2-class). [FT]
- Table 2 also surveys public and private ROP datasets (USA-iROP 5,561; China–Chengdu 20,795; SUNDROP 79,807; Shenzhen 20,822; etc.). [FT]

### 2.3 Split
- "The images were split with a three-fold cross-validation (70% training, 15% validation and 15% testing)." [FT]
- §3.4: on FARFUM-RoP "a three-fold cross-validation strategy was adopted to prevent patient overlap. For each fold, 70% of patients were used for training, 15% for validation, and 15% for testing", reported as "mean ± 95% bootstrap CIs across folds." [FT]
  - So FARFUM is **claimed to be patient-level**. No grouping procedure is given for the Ostrava or HVDROPDB experiments.
  - Fig. 2 shows an 85% train+val / 15% test split. [FT-fig]
- Test-set size: not stated as a number.
- **Anomaly**: Table 4's per-fold test support for FARFUM is 261/260/256 per class, i.e., **777 test images per fold** and **near-perfect class balance**. That is inconsistent with 15% of 1,533 (about 230) or 1/3 of 1,533 (511). [FT; FT-derived] It suggests the class-balanced expansion was done **before** splitting; this is not explained.
- External test: the paper claims the model was "trained exclusively on FARFUM-RoP and tested on completely unseen datasets – RIDIRP and HVDROPDB" (§4.6). But §3.4 and §4.4 describe experiments run "on" RIDIRP with baselines "trained and evaluated on the corresponding datasets". It is not determinable whether RIDIRP was a training set or a pure external test. [FT]

### 2.4 Preprocessing
- **Training (and "training–validation") images**: RandomResizedCrop to **380×380**, flips, rotation, colour jitter, tensor conversion, per-channel normalisation with mean/std. Normalisation values are not stated, though ImageNet is implied by "pre-trained". [FT]
- **Validation and test images**: resize to **400 px**, centre crop **380×380**, normalise. [FT]
- No CLAHE, circle crop, green channel or vessel processing. [FT, absence]

### 2.5 Augmentation (online, torchvision-style transforms)
- RandomResizedCrop to 380×380; scale range **not stated**. [FT]
- Random horizontal or vertical flip; probability **not stated**. [FT]
- Random rotation **±15°**. [FT]
- Colour jitter on brightness, contrast and saturation; magnitudes **not stated**. [FT]
- Timing relative to the split: per the pipeline figure, after splitting. The FARFUM support anomaly above (777 balanced test images per fold) is unexplained. [FT]
- No GAN or diffusion synthetic images. [FT, absence]

### 2.6 Model ("ROPDeepX")
- Two parallel backbones: **ResNet-50** (2048-d GAP feature) and **EfficientNet-B4** (1792-d). Pretrained weights are shown in Fig. 3; ImageNet is implied by the limitations text. [FT]
- **Projection heads**: FC 2048→1024 and FC 1792→1024. [FT]
- **Fusion block**: concatenate to 2048 → FC(2048→1024) + BatchNorm + ReLU + Dropout. The dropout rate is not stated. [FT]
- **Soft attention**: FC(1024→2) + softmax gives weights a1 and a2; the attended feature f_att = a1·z_r + a2·z_e (1024-d). [FT]
- **Classifier**: FC(1024→C). Softmax and categorical CE for multiclass; sigmoid and BCE for binary (Fig. 3). [FT]
- Input **380×380**. [FT]
- The trainable parameters are "projection heads, fusion block, attention mechanism, and **selected layers** of the backbone networks"; which layers is not stated. [FT]

### 2.7 Training
- Loss: cross-entropy **with label smoothing**; the smoothing value is not stated. [FT]
- Imbalance handling: not described beyond merging classes for the binary task. [FT]
- Optimiser **AdamW**; scheduler **OneCycleLR**; **mixed precision**. [FT]
  - Learning rate, max LR, weight decay and epochs: **not stated**. Fig. 8's x-axis spans about 25 epochs [FT-fig].
- Batch size **8**. [FT]
- Early stopping: "evaluated on a validation set after each epoch … repeated for a fixed number of epochs". [FT]
- Hardware: "GPU or CPU"; not stated. [FT]

### 2.8 Evaluation
- Metrics: per-class precision, recall and F1 with support; weighted accuracy; macro averages; AUC; specificity; normalised confusion matrices. [FT]
- **Calibration**: ECE, Brier score and reliability diagrams. Results are labelled "post-calibration" and "(Calibrated)", but **the calibration method is not stated**. [FT]
- Unit: image. [FT]
- CIs: "mean ± 95% bootstrap CIs across folds" on FARFUM. No statistical tests. [FT]

### 2.9 Results

**FARFUM-RoP 3-class, per class, mean over 3 folds with 95% CI (Table 4)** [FT]

| Class | Accuracy | Precision | Recall | F1 | Support F1/F2/F3 |
|---|---|---|---|---|---|
| Normal | 0.951 [0.935–0.965] | 0.957 [0.940–0.973] | 0.949 [0.930–0.967] | 0.952 [0.935–0.969] | 261/261/260 |
| Pre-Plus | 0.939 [0.919–0.960] | 0.926 [0.910–0.942] | 0.938 [0.922–0.954] | 0.932 [0.920–0.945] | 260/259/259 |
| Plus | 0.959 [0.945–0.974] | 0.960 [0.945–0.974] | 0.975 [0.960–0.988] | 0.967 [0.954–0.980] | 256/257/257 |
| Mean | 0.950±0.010 | 0.948±0.011 | 0.954±0.009 | 0.950±0.010 | |

- Normalised confusion matrices, calibrated (Fig. 6) [FT-fig]:
  - Fold 1: Normal 0.96/0.01/0.03; Plus 0.00/0.99/0.00; Pre-plus 0.09/0.01/0.90.
  - Fold 2: 0.94/0.01/0.05; 0.01/0.97/0.02; 0.07/0.01/0.92.
  - Fold 3: 0.94/0.00/0.06; 0.00/1.00/0.00; 0.02/0.02/0.96.
- **Calibration (Table 5)** [FT]: ECE 0.0176 / 0.0316 / 0.0257 (mean **0.0256 ± 0.0071**); Brier 0.0318 / 0.0335 / 0.0381 (mean **0.0345 ± 0.0033**).
- Train / val / test [FT]: recall 99.69 / 96.00 / 97.36%; precision 99.45 / 95.11 / 97.04%; AUC 99.69 / 96.00 / 99.32%.

**Multi-class (Table 6, %)** [FT]

| Dataset | Metric | VGG19 | ResNet50 | EffNet-B4 | ROPDeepX |
|---|---|---|---|---|---|
| FARFUM 3-class | Acc | 79.1 | 92.9 | 92.8 | **96.9** |
| | Recall | 79.1 | 93.1 | 93.1 | **97.4** |
| | Precision | 80.3 | 93.1 | 92.8 | **97.0** |
| | Specificity | 79.1 | 93.7 | 92.8 | **97.0** |
| | F1 | 79.0 | 92.8 | 92.8 | **96.9** |
| | AUC | 93.8 | 98.4 | 98.9 | **99.2** |
| RIDIRP 3-class | Acc | 89.8 | 85.8 | 89.9 | **99.7** |
| | Recall | 69.0 | 78.2 | 89.4 | **99.7** |
| | Precision | 86.2 | 88.6 | 91.7 | **99.7** |
| | Specificity | 78.0 | 75.3 | 85.0 | **99.8** |
| | F1 | 86.5 | 87.0 | 90.4 | **99.7** |
| | AUC | 94.8 | 95.4 | 95.6 | **100.0** |

**Binary (Table 7, %)** [FT]

| Dataset | Metric | VGG19 | ResNet50 | EffNet-B4 | ROPDeepX |
|---|---|---|---|---|---|
| FARFUM 2-class | Acc | 93.4 | 94.6 | 95.1 | 98.0 |
| | Recall | 96.5 | 97.1 | 95.9 | 98.0 |
| | Precision | 85.4 | 85.7 | 90.7 | 98.0 |
| | Specificity | 91.9 | 92.5 | 94.7 | 97.3 |
| | F1 | 90.6 | 91.9 | 93.2 | 98.0 |
| | AUC | 97.6 | 98.6 | 98.6 | 99.8 |
| RIDIRP 2-class | Acc | 83.3 | 87.6 | 90.4 | 92.4 |
| | Recall | 83.9 | 88.9 | 90.2 | 92.4 |
| | Precision | 87.8 | 89.7 | 89.6 | 92.5 |
| | Specificity | 82.4 | 85.9 | 90.6 | 92.4 |
| | F1 | 85.8 | 89.3 | 89.9 | 92.4 |
| | AUC | 86.5 | 89.6 | 94.2 | 94.8 |
| HVDROPDB v3 2-class | Acc | 80.0 | 80.2 | 80.0 | 93.3 |
| | Recall | 77.8 | 87.5 | 75.0 | 93.8 |
| | Precision | 87.5 | 77.8 | 85.7 | 94.2 |
| | Specificity | 83.3 | 71.4 | 85.7 | 93.8 |
| | F1 | 82.4 | 82.4 | 80.0 | 93.3 |
| | AUC | 94.4 | 87.5 | 80.4 | 94.6 |

- **Fig. 10 confusion matrices** (dataset not named; columns normal/plus/preplus) [FT-fig]:
  - ROPDeepX: normal 76/1/2, plus 1/68/0, preplus 5/0/72 (n = 225).
  - VGG19 and ResNet50 also total 225; EfficientNet-B4 totals **349**.
- Fig. 9 bar labels give ROPDeepX FARFUM: Acc 96.89, Recall 97.36, Precision 97.04, Specificity 97.02, F1 96.90, AUC 99.23. [FT-fig]

### 2.10 Key references for methods [FT]
1. [10] Akbari M. et al., FARFUM-RoP: a dataset for computer-aided detection of ROP, *Sci Data* 11:1176 (2024).
2. [11] Timkovič J. et al., Retinal image dataset of infants and ROP, *Sci Data* 11:814 (2024).
3. [12] Agrawal R. et al., HVDROPDB datasets for research in ROP, *Data Brief* 52:109839 (2024).
4. [13] Agrawal R. et al., Classification of stages 1, 2, 3 and pre-plus/plus using MultiCNN_LSTM, *MethodsX* 14:103182 (2025).
5. [44] He K. et al., ResNet, CVPR 2016. [45] Tan & Le, EfficientNet, ICML 2019.
6. [46] Sokolova & Lapalme, performance measures for classification, *Inf Process Manag* 2009.
7. [47] Deepthi K. et al., plus disease via transformer-based unsupervised curriculum learning (FARFUM subset), *BSPC* 104:107521 (2025).
8. [48] Almeida J. et al., automatic vessel segmentation for plus-form diagnosis on Ostrava/RIDIRP, *Res Eng* 24:103054 (2024).
9. [21] Chen J.S. et al., DL for stage of ROP: accuracy and generalizability across populations and cameras, *Ophthalmol Retina* 2021.
10. [7] Huang K. et al., GFF-Net graph feature fusion for plus disease, *Appl Intell* 2023.

### 2.11 Weaknesses and leakage risks (several are disqualifying)
- **Logically impossible numbers.** RIDIRP 3-class accuracy is 99.7%, but RIDIRP 2-class (Pre-Plus+Plus merged) is **92.4%**. Merging two classes can never lower the accuracy of the same predictions, so these cannot come from the same model and test set. [FT-derived]
- **Pre-plus does not exist in the Ostrava labels.**
  - The dataset paper states "the pre-plus form is not observed in the dataset" [FT: Timkovič 2024].
  - P1 and P3 both report a binary 629-image plus label [FT].
  - The provenance of ROPDeepX's RIDIRP "Pre-Plus" class is therefore unexplained.
- **Test support anomaly**: 777 near-balanced FARFUM test images per fold from a 1,533-image dataset. Likely an expanded or oversampled set formed before splitting, which would put augmented copies of test images in training.
- **Training on RIDIRP is ambiguous**: "trained exclusively on FARFUM" vs baselines "trained and evaluated on the corresponding datasets".
- **The relative-improvement claims cannot be reproduced** from the tables. "+4.36% AUC over EfficientNet-B4" vs Table 6's 98.9 → 99.2. [FT-derived]
- **Data-availability statement**: "No datasets were generated or analysed during the current study", which contradicts the use of three datasets. [FT]
- Calibration method unnamed; no LR or epoch count; label-smoothing value unstated.

### 2.12 What RetinAI could adopt
- **FARFUM-RoP as an extra external test set for plus and pre-plus.** RetCam Shuttle, Iran, 68 patients, 5-grader Normal/Pre-plus/Plus labels, public on Figshare. RetinAI has no pre-plus labels and lacks Middle-East/RetCam-Shuttle coverage. The paper's own results are unreliable, but the dataset pointer is valuable.
- **HVDROPDB v3** (100 images, India, 3nethra Neo): a tiny, device-diverse sanity check. Check for overlap with any Indian data RetinAI already uses.
- **Per-fold calibration reporting** (ECE + Brier + reliability diagram per fold). RetinAI v3 already has reliability plots. Adding per-fold ECE/Brier with SD, and stating the calibration method (e.g., temperature scaling fitted on out-of-fold logits), is cheap.
- **Not worth adopting**: the dual-CNN soft-attention fusion. RetinAI already found no gain from ensembles, and nothing here isolates the fusion's contribution (no ablation).

---

## 3. Paper 3: Hassan & Hasan, "Context-Aware Asymmetric Ensembling …" (CAA Ensemble)

### 3.1 Citation and access
- Md. Mehedi Hassan; Taufiq Hasan (corresponding). [FT]
- Affiliations: mHealth Lab, Dept. of Biomedical Engineering, BUET, Dhaka; Center for Bioengineering Innovation and Design, Johns Hopkins University. [FT]
- "Context-Aware Asymmetric Ensembling for Interpretable Retinopathy of Prematurity Screening via Active Query and Vascular Attention."
- Read as **arXiv:2602.05208v2** [eess.IV], 19 Feb 2026. [FT]
- The user lists it as *Comput Biol Med* 2026. **I could not find or verify the journal version, its DOI or its volume** (the web search returned only arXiv). Numbers may differ in the version of record.
- Code: github.com/mubid-01/MS-AQNet-VascuMIL-for-ROP_pre. [FT] As of today the repo holds only a README; the README says the code will be "released soon". [REPO]

### 3.2 Dataset
- The public Ostrava dataset (Timkovič 2024): **6,004 longitudinal images of 188 preterm infants**. [FT]
  - Cameras: Clarity RetCam 3, Natus RetCam Envision, Phoenix ICON.
  - Resolutions: **640×480, 1440×1080, 1240×1240**.
- Cohort: **GA 31.0 ± 5.5 wk**, **BW 1674 ± 1032 g**, **PA 40.1 ± 9.1 wk**. [FT]
- **Target taxonomy** (Table 3): the 13 granular codes are aggregated into 4 classes. [FT]

  | Class | DG codes | Rationale given |
  |---|---|---|
  | 0 Normal | 0 | Healthy |
  | 1 Mild / Treated | 1, 2, 9 | Observe: low immediate risk |
  | 2 Severe | 3, 4–8 | Refer: pre-threshold / threshold ROP |
  | 3 Other | 10–13 | Exclude: non-ROP (e.g., haemorrhage) |

  - Stage 2 (DG3) is deliberately put in Severe, citing ETROP. [FT]
  - ⚠ **Label-mapping conflict**: the dataset paper's worked example defines "**DG 2 – hemorrhage**" [FT: Timkovič 2024]. P3 places DG2 in **Mild** while saying haemorrhage belongs in Other. If the dataset paper is right, P3's Mild class contains haemorrhage images.
- **Class counts by split (Table 2)** [FT]

  | | Total | Train (fold 0) | Val (fold 0) | Test (held-out) |
  |---|---|---|---|---|
  | Patients | 188 | 137 | 34 | 17 |
  | Images | 6,004 | 4,261 | 1,075 | 668 |
  | Normal | 2,980 (49.6%) | 2,149 | 550 | 281 |
  | Mild/Treated | 676 (11.3%) | 342 | 223 | 111 |
  | Severe ROP | 1,053 (17.5%) | 832 | 143 | 78 |
  | Other pathology | 1,295 (21.6%) | 938 | 159 | 198 |
  | Plus disease | 629 (10.5%) | 470 | 81 | 78 |

  - Train vs val GA, BW and PA differ with p < 0.001 (t-test). Means: train GA 31.1, val 30.5, test 31.4; PA 40.8 / 37.6 / 39.2. [FT]
  - "Other" is 1,295 of the 3,024 non-normal images = **42.8%** [FT-derived]. This independently matches RetinAI's internal finding that about 43% of "ROP-positive" images are not ROP.

### 3.3 Split
- **Patient-wise.** One-tenth of patients (**N = 17**) form a held-out test set, "stratified to maintain a balanced representation of samples across all classes". [FT]
- The remaining **171** patients go into **stratified group 5-fold CV**, with all images of a patient in one fold ("Stratified Group K-Fold"). [FT]
- Ablations and the backbone comparison are run on **fold 0** only. [FT]
- Test: 668 images from 17 infants, about 39 images per infant [FT-derived].
- External or unseen hospital: **no**. The authors acknowledge it is single-centre. [FT]

### 3.4 Preprocessing ("Intelligent Data Engineering", Fig. 1)
- **Morphological artefact removal** [FT]:
  - Binary threshold to separate the foreground (threshold value not stated).
  - **Morphological erosion with a 3×3 kernel** to shrink the mask and remove the glowing aperture edge.
- **Enhancement** [FT]:
  - **Gamma correction γ = 1.5**, then **CLAHE** (clip limit and tile grid not stated) on the masked image.
- **Structure stream** [FT]: rectangular image → **zero-padded to square** → resize to **384×384** (described as geometry-preserving, i.e., letterbox).
- **Texture stream** (Vascular Topology Map, VMAP) [FT]:
  - From a **768×768** image, take the **green channel**.
  - Compute the **multi-scale Frangi vesselness** (Hessian eigenvalues |λ1| ≤ |λ2|):
    - V = 0 if λ2 > 0;
    - otherwise V = exp(−R_B²/2β²)·(1 − exp(−S²/2c²)), with R_B = |λ1|/|λ2| and S = √(λ1² + λ2²).
  - Scales, β and c: **not stated**.
  - The sparse map is stacked depth-wise onto RGB to give a **4-channel tensor**.
  - **Patches of 224×224** are extracted from the 768×768 image.
- Tooling: OpenCV (morphology), scikit-image (Frangi). [FT]
- Normalisation: **frozen BatchNorm with ImageNet statistics** in the backbone. Input normalisation values: not stated. [FT]

### 3.5 Augmentation
- **Not described in the paper** (no augmentation section). [FT, absence]
- The README says the loader has "aggressive augmentation" but lists no operations. [REPO]
- **No synthetic images.** The authors argue against GAN augmentation because of "intra-class mode collapse" (citing Ding et al. 2024). [FT]

### 3.6 Model

**Stream A: MS-AQNet ("Structure Specialist", staging)** [FT]
- **EfficientNet-B0** (ImageNet) with **frozen BatchNorm statistics**.
- Three scales V_s at **48×48, 24×24 and 12×12** from the 384 input, each reduced to **64 channels** by a 1×1 conv.
- **Metadata projector**: x_tab = (GA, BW, PA), normalised → 2-layer ReLU MLP → query q_s ∈ R^64.
- **Active Query Unit**:
  - e_ij = q_sᵀk_ij / √64 → spatial softmax A_s.
  - Gated residual: V̂_s = (1−g)·V_s + g·(A_s ⊙ V_s), with g = σ(α) and **α initialised to −2.0 (g ≈ 0.12)**.
- GAP per scale → concat **3×64 = 192**.
- **FiLM**: γ, β = MLP(x_tab) ∈ R^192; z_final = γ ⊙ z + β.
- Linear projection → **GroupNorm** → ReLU → 4-class logits.
- **Deep supervision**: auxiliary classifiers on the first two Active Query Units, weight **λ = 0.2**.

**Stream B: VascuMIL ("Texture Specialist", plus)** [FT]
- **EfficientNet-B0** with a **4-channel stem**: the first conv keeps the pretrained RGB weights, and the extra channel is **Kaiming-initialised**.
- Each patch → embedding h_k ∈ R^**256** (FC → LayerNorm → GELU, per Fig. 3).
- **Gated attention MIL** (Ilse 2018): a_k ∝ exp(wᵀ[tanh(Vh_k) ⊙ σ(Uh_k)]), attention dimension **M = 128**; z = Σ a_k h_k.
- Classifier MLP: d_hid = **64**, **dropout 0.2**, sigmoid.
- **Bag = N = 24 patches.** "Patient examinations are represented as a bag", and the stream claims "patient-level decision integrity". Whether a bag spans one image or a whole exam is **ambiguous**. [FT] The README also leaves it open. [REPO]

**Fusion meta-learner** [FT]
- z_fusion = concat(structure logits ∈ R^4, texture logit ∈ R^1, **re-injected x_tab ∈ R^3**) ∈ R^8.
- MLP (8→32; FC + BatchNorm + ReLU + Dropout per Fig. 1) → softmax 4-class head and sigmoid plus head.

### 3.7 Training [FT unless marked]
- PyTorch; single NVIDIA GPU (model not stated); **AMP**. **AdamW + cosine annealing** for all stages.
- **MS-AQNet**:
  - **30 epochs, batch 16**.
  - Backbone **frozen for the first 5 epochs**, then fully trained with **differential LR: 1e-3 heads, 1e-6 backbone**.
  - Loss: **class-weighted focal loss, γ = 2.0, α weights w_severe = 5.0 and w_normal = 0.5** (Mild and Other weights not stated), plus deep supervision λ = 0.2.
  - The README adds `--use_sampler`. [REPO]
- **VascuMIL**:
  - **30 epochs** (the README says 20 [REPO]), **bag batch 4**.
  - **Weighted BCE**, w_pos = inverse class frequency.
  - **Bag-level weighted random sampler with p(positive) = 0.5.**
  - The text says positive bags occur with "probability ≈ 3%", which conflicts with the 10.5% plus prevalence.
- **Fusion MLP**:
  - **Stacking on frozen logits from the validation folds** (out-of-fold).
  - **80 epochs, LR 1e-2**.
  - Loss L_total = CE(diag) + BCE(plus).
- **Checkpoint selection**: **Clinical Score CS = ½·(macro-F1 + Sensitivity_target)**, where the target is Severe ROP or Plus.
- Early stopping: only via CS checkpointing.

### 3.8 Evaluation
- Metrics: sensitivity, specificity, precision, F1, **Cohen's κ**, ROC-AUC (one-vs-rest per class), confusion matrices, t-SNE, **SHAP** on fusion inputs. [FT]
- Unit: **image** (test = 668 images; not stated explicitly). [FT]
- **No CIs and no statistical tests on performance.** The only t-tests are on demographics. [FT]

### 3.9 Results

**Test set (Table 4)** [FT]

| Task / class | Model | Sens | Spec | Prec | F1 | AUC | κ |
|---|---|---|---|---|---|---|---|
| Normal | Baseline CNN | 0.648 | 0.914 | 0.861 | 0.740 | 0.894 | |
| | MS-AQNet | 0.573 | 0.936 | 0.915 | 0.705 | 0.833 | |
| | **CAA Ensemble** | 0.954 | 0.963 | 0.969 | 0.962 | 0.994 | |
| Mild/Treated | Baseline | **0.316** | 0.927 | 0.389 | 0.349 | 0.811 | |
| | MS-AQNet | **0.214** | 0.878 | 0.267 | 0.237 | 0.784 | |
| | **CAA Ensemble** | 0.971 | 0.968 | 0.862 | 0.913 | 0.996 | |
| Severe ROP | Baseline | 0.935 | 0.866 | 0.509 | 0.660 | 0.947 | |
| | MS-AQNet | 0.985 | 0.872 | 0.310 | 0.471 | 0.951 | |
| | **CAA Ensemble** | 0.803 | 0.997 | 0.930 | 0.862 | 0.995 | |
| Other | Baseline | 0.809 | 0.884 | 0.619 | 0.702 | 0.933 | |
| | MS-AQNet | 0.908 | 0.820 | 0.596 | 0.719 | 0.897 | |
| | **CAA Ensemble** | 0.937 | 0.989 | 0.962 | 0.950 | 0.997 | |
| Macro (4-class) | Baseline | 0.677 | 0.898 | 0.594 | 0.612 | 0.896 | 0.553 |
| | MS-AQNet | 0.670 | 0.876 | 0.522 | 0.533 | 0.866 | 0.443 |
| | **CAA Ensemble** | **0.916** | **0.979** | **0.931** | **0.922** | **0.996** | **0.942** |
| Plus (binary) | Baseline | 0.778 | 0.962 | 0.594 | 0.674 | 0.977 | 0.647 |
| | VascuMIL | 0.975 | 0.971 | 0.712 | 0.823 | 0.995 | 0.808 |
| | **CAA Ensemble** | 0.901 | 0.996 | 0.936 | 0.918 | **0.999** | 0.897 |

- The baseline CNN is EfficientNet-B0 with late concatenation of metadata. [FT]

**Fig. 4, ensemble on the test set** (row-normalised confusion %, columns Normal/Mild/Severe/Other) [FT-fig]

| True class | Normal | Mild | Severe | Other |
|---|---|---|---|---|
| Normal | 94.7 | 4.4 | 0.0 | 0.9 |
| Mild | 1.9 | 93.7 | 4.4 | 0.0 |
| Severe | 1.5 | **15.2** | 83.3 | 0.0 |
| Other | 7.0 | 0.0 | 0.0 | 93.0 |

- Plus: no-plus 99.3 / 0.7; plus 8.6 / 91.4.
- Per-class AUC: Normal 0.994, Mild 0.996, Severe 0.995, Other 0.997.
- Plus ROC: structure 0.967, texture 0.995, ensemble 0.999.
- These diagonals (94.7 / 93.7 / 83.3 / 93.0; plus 91.4) **do not match** Table 4's sensitivities (0.954 / 0.971 / 0.803 / 0.937; plus 0.901).

**Backbone comparison, fold 0, broad diagnosis (Table 5)** [FT]

| Backbone | Params (M) | Sens | Spec | Prec | F1 | AUC |
|---|---|---|---|---|---|---|
| **EfficientNet-B0** | 7.44 | 0.91 | 0.98 | 0.93 | **0.92** | 0.99 |
| MobileNetV3-L | 8.52 | 0.89 | 0.97 | 0.92 | 0.91 | 0.98 |
| DenseNet-121 | 14.02 | 0.88 | 0.96 | 0.90 | 0.89 | 0.98 |
| ConvNeXt-Tiny | 55.74 | 0.86 | 0.96 | 0.88 | 0.87 | 0.97 |
| ResNet-50 | 47.14 | 0.78 | 0.93 | 0.81 | 0.79 | 0.94 |
| Inception-v3 | 43.68 | 0.76 | 0.92 | 0.79 | 0.77 | 0.93 |

**Ablation, fold 0 (Table 6)** [FT]

| Config | Meta | VMAP | Attention | Sens | Spec | Prec | F1 | κ | AUC |
|---|---|---|---|---|---|---|---|---|---|
| 1 Baseline CNN | | | | 0.645 | 0.890 | 0.680 | 0.662 | 0.385 | 0.892 |
| 2 MS-AQNet (image) | | | Active | 0.720 | 0.915 | 0.745 | **0.730** | 0.412 | 0.935 |
| 3 MS-AQNet (full) | ✓ | | Active | 0.670 | 0.876 | 0.522 | **0.533** | 0.443 | 0.866 |
| 4 VascuMIL (RGB), plus | | | Gated | 0.910 | 0.935 | 0.620 | 0.735 | 0.710 | 0.965 |
| 5 VascuMIL (+VMAP), plus | | ✓ | Gated | 0.975 | 0.971 | 0.712 | 0.823 | 0.808 | **0.995** |
| 6 CAA Ensemble | ✓ | ✓ | Hybrid | 0.919 | 0.973 | 0.927 | 0.926 | 0.945 | 0.997 |

- Note: **injecting metadata into MS-AQNet lowered macro-F1 from 0.730 to 0.533.** All the metadata gain arrives through late fusion. [FT]
- SHAP: the texture "Plus" logit is the strongest predictor; BW, PA and GA modulate it. [FT-fig]
- Inconsistencies between abstract and tables: abstract macro-F1 "0.93" vs Table 4 0.922; abstract plus AUC "0.996" vs Table 4 0.999. The README claims "severe-ROP sensitivity 96.3%" [REPO] vs Table 4 0.803.

### 3.10 Key references for methods [FT]
1. [10] Timkovič J. et al., *Sci Data* 2024 (dataset).
2. [32] Frangi A.F. et al., Multiscale vessel enhancement filtering, MICCAI 1998.
3. [39] Ilse M., Tomczak J., Welling M., Attention-based deep multiple instance learning, ICML 2018 (arXiv 1802.04712).
4. [35] Perez E. et al., FiLM: visual reasoning with a general conditioning layer, AAAI 2018 (arXiv 1709.07871).
5. [40] Lin T.-Y. et al., Focal loss for dense object detection, ICCV 2017.
6. [36] Wu Y., He K., Group normalization, ECCV 2018. [38] He K. et al., Kaiming init, ICCV 2015. [33] Tan & Le, EfficientNet, 2019.
7. [37] Quellec G. et al., MIL for medical image and video analysis, *IEEE Rev Biomed Eng* 2017. [16] Matten P. et al., MIL for DR in widefield OCTA, *Sci Rep* 2023.
8. [12] Li Y. et al., multimodal net combining metadata and images for glaucoma, *Sci Rep* 2023 (precedent for active metadata fusion).
9. [20] Subramaniam A. et al., **image harmonization** and DL classification of plus disease, *J Med Imaging* 10(6):061107 (2023). Relevant to cross-device shift.
10. [41] ETROP Cooperative Group (Good W.V.), *Trans Am Ophthalmol Soc* 2004 (referral taxonomy). [28] Huang Y.-P. et al., tortuosity measurement vs ROP severity, *Eye* 2024. [7] Cole E.D. et al., single vs serial images, *Ophthalmol Retina* 2022.

### 3.11 Weaknesses and leakage risks
- **17-infant test set.** About 39 correlated images per infant are scored as independent observations, and no CIs are given. The effective sample size is 17.
- **Metadata shortcut.** GA and BW are constant per infant, and PA is constant per series.
  - The fusion MLP sees (GA, BW, PA) directly, so with few test infants it can assign whole infants or series to a class from metadata alone.
  - Evidence: Mild sensitivity jumps from 0.214 (image MS-AQNet) to 0.971 only after fusion.
  - The metadata-injected MS-AQNet alone **got worse** (Table 6, row 3).
  - The gains are unlikely to transfer to sites without metadata, or with different GA/PMA distributions.
- **Labels inherit series diagnoses** (weak labels; see P4). The DG2 = haemorrhage conflict may contaminate the Mild class.
- **Internal inconsistencies**: Table 4 vs Fig. 4; abstract vs tables vs README; "≈3% positive bags" vs 10.5% plus; VascuMIL 30 vs 20 epochs.
- **Unreported settings**: augmentation, CLAHE and Frangi parameters, and the bag definition.
- **Selection bias**: the Clinical Score checkpointing on validation folds is fine, but the backbone choice and ablations are on fold 0, and the test is then a single 17-infant draw.
- **Device confounding unaddressed.** Three cameras with different native resolutions; the 384-pixel letterbox standardises size but not the camera signature.

### 3.12 What RetinAI could adopt
1. **A high-resolution patch-MIL branch for thin peripheral structures**, which may help Stage 1.
   - VascuMIL design: 768 px image → 224 px tiles (bag of 24) → shared backbone → gated attention (Ilse; M = 128, L = 256) → MLP (64, dropout 0.2).
   - This differs from the 518-px global input RetinAI already tried: each tile is seen at a higher effective resolution with the backbone's native receptive field, and attention pooling can pick out the few tiles containing the demarcation line.
   - A DINOv2 ViT-S/14 fits naturally: encode 224 tiles, pool CLS tokens with gated attention.
   - Their evidence is for plus (κ 0.71 → 0.81 with VMAP), not for Stage 1, so treat Stage 1 benefit as a hypothesis to test.
2. **Frangi vesselness on the green channel as a 4th input channel**, with the extra channel's patch-embed weights Kaiming- or zero-initialised so pretrained RGB behaviour is preserved.
   - Same-split ablation: AUC 0.965 → 0.995 for plus.
   - Most relevant to AP-ROP and plus. Stage 1's demarcation line is not tubular, so its benefit there is doubtful.
3. **Training mechanics for rare classes**:
   - A **class-balanced sampler with p = 0.5** for the rare positive (Stage 4/5, AP-ROP), on top of RetinAI's class-balanced CE.
   - **Focal loss with clinically asymmetric α** (P3 used w_severe = 5.0, w_normal = 0.5; γ = 2).
   - **Deep supervision** (λ = 0.2) on intermediate ViT blocks.
   - Frozen-backbone warm-up for 5 epochs, then differential LR. RetinAI's layer decay of 0.8 already approximates this.
4. **Checkpoint selection on a clinical composite** such as CS = ½(macro-F1 + Stage-1 recall), or the worst-class recall, instead of accuracy or macro-F1. This is cheap and targets the weakest metric directly.
5. **Late-fusion stacking on out-of-fold logits**: the right way to combine streams (global ViT, patch-MIL, optional metadata) without leakage.
   - Use metadata (Ostrava filenames carry GA, BW and PA [FT: Timkovič 2024]) **only with modality dropout**, and report with and without it.
   - Do not deploy it for the unseen hospital unless that site supplies GA/PMA.
   - PMA is a legitimate clinical prior for Stage 1 timing, but P3 shows it can also act as a shortcut.
6. **Audit RetinAI's own DG → stage mapping** against the Timkovič code table (DG2 = haemorrhage).

---

## 4. Paper 4: Kian Ara et al., Frontiers in Medicine 2026

### 4.1 Citation and access
- Rouhollah Kian Ara (corresponding), Jan Kubíček, Sumit Kaushik, Kristýna Maršolková, Juraj Timkovič. [FT]
- "Explainability-informed patient-disjoint evaluation of deep learning for plus disease in retinopathy of prematurity with an ophthalmologist reader study." *Frontiers in Medicine* 13:1960317 (2026), Ophthalmology section, Original Research. [FT]
- Received 6 Aug 2026; revised 11 Sep 2026; accepted 15 Sep 2026; published 30 Sep 2026. DOI 10.3389/fmed.2026.1960317. CC BY. [FT]
- Affiliations: VSB–Technical University of Ostrava; University Hospital Ostrava (Clinic of Ophthalmology); University of Ostrava (Faculty of Medicine). [FT]
- Ethics ref. 544/2018. [FT]
- Code: github.com/kaushikGeom/ROP_Classification. [FT]
- Reporting: CLAIM 2024 and STARD-AI checklists. [FT]
- Access: **FT** (main text). Supplementary Tables S1–S7, Fig. S1–S4 and the Supplementary Methods were not read.

### 4.2 Datasets
- Source: University Hospital Ostrava, **Phoenix ICON** wide-field camera, native **1240×1240**. [FT] This is **not** the public Kaggle release; the data are not public.
- **Primary dataset: 732 images from 38 infants.** [FT]
  - Development pool: **634 images / 30 infants** (331 plus, 303 non-plus).
  - Hold-out: **98 images / 8 infants** (51 plus, 47 non-plus).
  - Hold-out infants: 3 plus-reference and 5 non-plus-reference.
  - Images per hold-out infant: **1–30**. Laterality: **43 left-eye and 55 right-eye images**.
  - Infant ID **and eye laterality** are available for each image. [FT]
- **Historical secondary dataset: 834 images** (471 plus / 363 non-plus). [FT]
  - Development 751 images (424 / 327) in 39 labelled image collections (15 plus collections with 424 images; 24 non-plus collections with 327 images).
  - Random test set: 83 images (47 / 36).
  - Infant IDs are incomplete, so infant overlap **cannot be excluded**.
- **Labels**: binary plus / non-plus, assigned per **clinical case** by one of two ROP ophthalmologists (J.T., K.M.) after reviewing the full image set. [FT]
  - Each image **inherits** the examination label.
  - No adjudication or multi-grader consensus.
  - **Pre-plus is not encoded.**
- Metadata (GA, BW, PMA, stage, zone, treatment, exam date) exist in the archive but **could not be reliably linked**, so they were not used. [FT]
- Vessel segmentation: expert-annotated Phoenix ICON fundus images from **two ophthalmologists**. Their count is in the supplement, not the main text. [FT]

### 4.3 Split
- **Primary: patient-disjoint (infant-level)**, seed 42. [FT]
  - Inside the development pool: **5-fold subject-wise stratified group CV** grouped by infant.
  - The **5 fold models are averaged** on the hold-out set.
  - The hold-out set is excluded from fitting, checkpointing and threshold selection.
- **Secondary: random image-level**, label-stratified (about 10% test), seed 42. Explicitly treated as exploratory and potentially optimistic. [FT]
- External or unseen hospital: **no** (same hospital, unseen infants). [FT]
- Prediction unit: **image**. Infant-level results are "descriptive". [FT]

### 4.4 Preprocessing [FT]
- **Automatic crop**: foreground mask = pixels with mean RGB intensity > **10** (≤10 counts as background). Take the bounding box of non-black pixels, centre a **square crop** on it, and add a **10-pixel margin**.
- Resize to **224×224**.
- Normalise with **ImageNet mean (0.485, 0.456, 0.406) and SD (0.229, 0.224, 0.225)**.
- No quality or gradability exclusion; no CLAHE.
- The authors note that shortcut learning from illumination, field of view or camera features was **not** controlled.

### 4.5 Augmentation (online, training only) [FT]
- Random horizontal flip, p = 0.5.
- Random rotation **±10°**.
- Colour jitter: brightness, contrast and saturation **0.1**; hue **0.02**.
- None at validation or test.
- No synthetic images.
- The vessel-overlay VGG16 model **omits colour jitter**.

### 4.6 Models [FT]
- **ImageNet-pretrained** EfficientNet-B0, DenseNet121, ResNet50, ResNet-18 and MobileNetV3-Large.
- The head is replaced by a linear layer with 2 logits; softmax gives the plus probability. No custom modules.
- **Vessel segmenter "UNet2"**:
  - 5-level U-Net, encoder channels 32/64/128/256/512, 1024 bottleneck, symmetric decoder, skip connections, bilinear upsampling.
  - Trained on randomly sampled **640×640 patches** with vessel-aware sampling.
  - Two models, one per ophthalmologist's annotations; **pixel-probability maps are averaged** at inference.
  - Inference: sliding **640×640 tiles with 50% overlap**, threshold 0.5.
- **Vessel overlay**: predicted vessels painted **green** on the raw image, non-vessel regions **dimmed**. Fed to **VGG16** and compared with VGG16 on raw images.

### 4.7 Training [FT]
- **Primary CNNs**:
  - **AdamW, fixed LR 1×10⁻⁴**, **200 epochs per fold**, **batch 16**, **cross-entropy** with no class weighting (classes are near-balanced at 331/303).
  - Checkpoint: **highest validation accuracy** per fold. No early stopping.
  - **AMP disabled**. Seed 42 for Python, NumPy and PyTorch.
- **U-Net**:
  - Adam, initial LR **1×10⁻⁴**, batch **4**, up to **1,000 epochs**.
  - Loss **0.3·BCE + 0.7·Dice**, computed only inside the retinal ROI mask.
  - **ReduceLROnPlateau** (factor 0.1, patience 40).
  - Checkpoint: best validation Dice.
- **VGG16 (raw vs overlay)**: **50 epochs**, **class-weighted CE**, AdamW, **cosine annealing**, batch 16, **mixed precision**.
- Hardware: not reported.
- A training-fraction analysis (ResNet-18, single subsampling) is reported in Supplementary S3/S4 (not read).

### 4.8 Evaluation [FT]
- Metrics: accuracy, sensitivity, specificity, PPV, NPV, F1, balanced accuracy, MCC, ROC-AUC, PR-AUC, Brier, Youden's J (descriptive only).
- **Cluster-aware analyses**:
  1. **Infant-equal weighting**: each image weighted 1/n_i.
  2. **Leave-one-infant-out influence analysis**: no retraining; recompute metrics after dropping each infant.
  3. **Stratified infant-cluster bootstrap**: 10,000 resamples, sampling plus-reference and non-plus-reference infants **separately**, keeping all images of a sampled infant; 2.5th/97.5th percentiles reported as exploratory intervals.
- **Threshold rule (development-only)**: on out-of-fold development predictions, take the thresholds with sensitivity ≥ 90%, choose the one with the highest development specificity, and break ties toward the higher threshold. Freeze it before the hold-out.
- **Within-infant variability**: group by infant × laterality and order by sequence index. Compute the mean absolute change between adjacent images, Δ_adj = (1/M)·Σ|p_{g,j+1} − p_{g,j}|, the maximum change, and threshold crossings.
- **Aggregation rules compared post hoc**: mean, median, max, top-3 mean.
- **CIs**: Wilson 95% for secondary proportions; Clopper–Pearson for reader sensitivity and specificity.
- **Reader agreement**: Cohen's κ (unweighted, linear, quadratic) and repeat agreement.
- **No hypothesis tests or paired comparisons** between models.

### 4.9 Results [FT]

**Primary hold-out, threshold 0.5 (Table 2; 98 images / 8 infants)**

| Model | Acc | Sens | Spec | PPV | NPV | F1 | Bal acc | ROC-AUC | PR-AUC | Brier |
|---|---|---|---|---|---|---|---|---|---|---|
| DenseNet121 | 0.806 | 0.647 | 0.979 | 0.971 | 0.719 | 0.776 | 0.813 | **0.929** | 0.942 | 0.148 |
| EfficientNet-B0 | 0.735 | 0.549 | 0.936 | 0.903 | 0.657 | 0.683 | 0.743 | 0.861 | 0.884 | 0.191 |
| MobileNetV3-L | 0.806 | 0.627 | **1.000** | **1.000** | 0.712 | 0.771 | **0.814** | 0.927 | **0.944** | **0.140** |
| ResNet-18 | 0.786 | **0.667** | 0.915 | 0.895 | 0.717 | 0.764 | 0.791 | 0.869 | 0.889 | 0.164 |
| ResNet50 | 0.806 | 0.647 | 0.979 | 0.971 | 0.719 | 0.776 | 0.813 | 0.904 | 0.921 | 0.156 |

- DenseNet121 confusion: TP 33, TN 46, FP 1, FN 18.

**Cluster-aware analyses** (EffNet-B0 / DenseNet121 / ResNet50 / MobileNetV3 / ResNet-18)

| Measure | EffNet-B0 | DenseNet121 | ResNet50 | MobileNetV3 | ResNet-18 |
|---|---|---|---|---|---|
| Image-weighted ROC-AUC | 0.861 | 0.929 | 0.904 | 0.927 | 0.869 |
| **Infant-equal ROC-AUC** | 0.800 | 0.905 | 0.899 | 0.920 | 0.787 |
| Image-weighted sensitivity | 0.549 | 0.647 | 0.647 | 0.627 | 0.667 |
| **Infant-equal sensitivity** | 0.356 | 0.394 | 0.406 | 0.389 | 0.411 |
| Leave-one-infant-out AUC range | 0.804–0.914 | 0.838–0.989 | 0.797–0.978 | 0.830–0.994 | 0.721–0.962 |
| Infant-cluster bootstrap 95% AUC | 0.668–0.944 | 0.752–0.999 | 0.701–0.997 | 0.769–0.999 | 0.588–0.991 |

**Development-defined high-sensitivity threshold**
- DenseNet121 threshold **0.0184**: sensitivity **98.0%** (1 FN), specificity **46.8%** (25 FP). The flagged fraction rises from 34.7% (34/98) to **76.5%** (75/98).
- Other models (sensitivity / specificity): EfficientNet-B0 100% / **0.0%**; MobileNetV3 100% / 34.0%; ResNet50 96.1% / 42.6%; ResNet-18 94.1% / 38.3%.

**Within-infant variability, DenseNet121 (Tables 3–4)**

| Infant | Ref | Images (L/R) | Mean ± SD | Range | ≥0.5 | Mean L / R | Mean adj. Δ | Crossings |
|---|---|---|---|---|---|---|---|---|
| P01 | Plus | 20 (10/10) | 0.312 ± 0.256 | 0.015–0.892 | 5/20 | 0.228 / 0.396 | 0.248 | 7 |
| P02 | Plus | 30 (15/15) | 0.802 ± 0.199 | 0.167–0.999 | 28/30 | 0.758 / 0.846 | 0.152 | 2 |
| P03 | Plus | 1 (1/0) | 0.294 | — | 0/1 | 0.294 / — | — | — |
| P04 | Non | 8 (4/4) | 0.030 ± 0.058 | 0.000–0.171 | 0/8 | 0.006 / 0.054 | 0.060 | 0 |
| P05 | Non | 10 (5/5) | 0.037 ± 0.045 | 0.001–0.124 | 0/10 | 0.047 / 0.027 | 0.049 | 0 |
| P06 | Non | 10 (4/6) | 0.029 ± 0.037 | 0.000–0.115 | 0/10 | 0.006 / 0.044 | 0.026 | 0 |
| P07 | Non | 9 (4/5) | 0.249 ± 0.163 | 0.020–0.523 | 1/9 | 0.340 / 0.175 | 0.192 | 2 |
| P08 | Non | 10 (0/10) | 0.060 ± 0.086 | 0.001–0.212 | 0/10 | — / 0.060 | 0.098 | 0 |

- 84 adjacent pairs in 13 infant–laterality groups: mean |Δ| **0.142**; **11/84** pairs cross 0.5; P02's maximum change is 0.707.
- Across architectures, mean |Δ|: EffNet-B0 0.171, DenseNet 0.142, ResNet50 0.148, MobileNetV3 0.163, ResNet-18 0.137.
  - Maximum changes: 0.783 / 0.708 / 0.883 / 0.736 / 0.668.
  - Threshold crossings: 18 / 11 / 18 / 12 / 11 of 84 (13.1–21.4%).
  - Within-infant SD: 0.164 / 0.121 / 0.142 / 0.139 / 0.168.
- **Aggregation (threshold 0.5)**: **mean and median each flagged 1 of 8 infants; max flagged 3; top-3 mean flagged 2.**
  - From Table 3: mean and median flag only P02 (P01's mean is 0.312). Max flags P01, P02 and **P07 (a false positive from one 0.523 image)**. P03, with its single image at 0.294, is missed by every rule. [FT-derived]

**Secondary random image-level analysis (83 images; exploratory, likely optimistic; Table 5)**

| Model | Params (M) | Acc | PPV | Sens | Spec | F1 |
|---|---|---|---|---|---|---|
| EfficientNet-B0 | 5.3 | 0.939 | 0.956 | 0.935 | 0.944 | 0.945 |
| DenseNet121 | 8.0 | 0.939 | 0.936 | 0.957 | 0.917 | 0.946 |
| ResNet50 | 25.6 | 0.915 | 0.915 | 0.935 | 0.889 | 0.925 |
| MobileNetV3-L | 5.4 | 0.927 | 0.955 | 0.913 | 0.944 | 0.933 |
| ResNet-18 | 11.7 | 0.940 | 0.938 | 0.957 | 0.917 | 0.947 |

- ResNet-18 details: TP 45, TN 33, FP 3, FN 2. Accuracy 94.0% (Wilson 86.7–97.4), sensitivity 95.7% (85.8–98.8), specificity 91.7% (78.2–97.1), PPV 93.8% (83.2–97.9), NPV 94.3% (81.4–98.4), MCC 0.877, AUC 0.974, Brier 0.0573.
- Compare with the patient-disjoint primary (AUC 0.869–0.929, sensitivity 0.55–0.67). This is an illustration of split inflation, though the datasets differ.

**Raw vs vessel overlay, VGG16 (Table 6; secondary split)**

| Input | ROC-AUC | Acc | Sens | Spec | TN | FP | FN | TP |
|---|---|---|---|---|---|---|---|---|
| Raw fundus | **0.959** | 0.892 | 0.851 | 0.944 | 34 | 2 | 7 | 40 |
| Vessel overlay | 0.906 | 0.843 | 0.787 | 0.917 | 33 | 3 | 10 | 37 |

**Reader study (Table 7; 30 unique isolated images, 18 plus-ref and 12 non-plus-ref, plus 10 repeats)**

| Measure | Ophthalmologist 1 | Ophthalmologist 2 |
|---|---|---|
| Strict sensitivity (definite plus) | 1/18 (5.6%, 0.1–27.3) | 0/18 (0.0%, 0.0–18.5) |
| Strict specificity | 12/12 (100%, 73.5–100) | 12/12 (100%, 73.5–100) |
| Abnormality-sensitive sensitivity (borderline/pre-plus or plus) | 4/18 (22.2%, 6.4–47.6) | 1/18 (5.6%) |
| Healthy / Borderline / Plus counts | 26 / 3 / 1 | 29 / 1 / 0 |
| **Plus-reference images rated Healthy** | **14/18** (mean confidence 4.14/5) | **17/18** (4.59/5) |
| Mean confidence, all images | 4.27 ± 1.08 | 4.70 ± 0.53 |
| Repeat-grade agreement | 10/10 | 10/10 |
| Repeat-confidence exact agreement | 8/10 | 7/10 |

- Inter-observer: exact 3-category agreement 26/30 (86.7%); κ 0.161 (unweighted), 0.302 (linear), 0.478 (quadratic); binary agreement 27/30, κ 0.366.
- Model vs readers (ResNet-18, abnormality-sensitive): 56.7% agreement (κ 0.211) with Ophthalmologist 1; 46.7% (κ 0.051) with Ophthalmologist 2.
- On the 26 images both readers rated Healthy, the model said plus on 13.
- Grad-CAM and LIME (about 30 SLIC superpixels; local surrogate R² of 0.416, 0.515 and 0.251 on 3 examples) are qualitative only, with no localisation ground truth.

### 4.10 Key references for methods [FT]
1. [7] Cole E.D. et al., Variability in plus disease diagnosis using single and serial images, *Ophthalmol Retina* 6(12):1122–9 (2022). The core rationale for multi-image labels.
2. [12] Hu J. et al., Automated analysis for ROP by deep neural networks (multi-image, exam-level aggregation), *IEEE TMI* 38(1):269–79 (2019).
3. [11] Brown J.M. et al., Automated diagnosis of plus disease using deep CNNs (i-ROP DL), *JAMA Ophthalmol* 136(7):803–10 (2018).
4. [14] Bai A. et al., Multicenter validation of ROP.AI for plus disease, *TVST* 12(8):13 (2023).
5. [13] Wang J. et al., Automated explainable multidimensional DL platform for ROP screening (quality, stage, haemorrhage, pre-plus/plus), *JAMA Netw Open* 4(5):e218758 (2021).
6. [17] Gojić G. et al., DL methods for retinal vessel segmentation evaluated on ROP images, IEEE SISY 2020.
7. [16] Tejani A.S. et al., CLAIM 2024 update, *Radiol AI* 2024. [26] Sounderajah V. et al., STARD-AI, *Nat Med* 2025.
8. [24] Brier G.W., *Mon Weather Rev* 1950. [25] Van Calster B. et al., Calibration: the Achilles heel of predictive analytics, *BMC Med* 2019.
9. [18] Selvaraju, Grad-CAM, ICCV 2017. [27] Ribeiro, LIME, KDD 2016. [28] Achanta, SLIC superpixels, *TPAMI* 2012.
10. [19–23] EfficientNet (ICML 2019), DenseNet (CVPR 2017), ResNet (CVPR 2016), MobileNetV3 (ICCV 2019), U-Net (MICCAI 2015).

### 4.11 Weaknesses and leakage risks
- The hold-out is **8 infants, only 3 plus**: wide cluster-bootstrap intervals (e.g., 0.752–0.999). The authors say so themselves.
- Prediction unit is the image; labels are inherited exam labels (weak labels); there is no image-quality or posterior-pole-adequacy scoring.
- **224 px input from 1240 px images.** Plus-disease vessel calibre and tortuosity cues are heavily downsampled, which may explain the low image-level sensitivity (0.55–0.67).
- **Checkpoint by validation accuracy over 200 fixed-LR epochs**: prone to noisy selection. No early stopping.
- The secondary analysis is not patient-disjoint. In the vessel-overlay comparison, colour jitter differs between arms, so the cause of the drop is not isolated.
- The architecture panel has CNNs only, with no ViT or foundation model; the authors acknowledge this.
- Device shortcuts are not excluded (single camera here, so less of an issue).

### 4.12 What RetinAI could adopt (the most transferable paper)
1. **The evaluation protocol, almost verbatim**, for the per-eye scoring at the unseen hospital:
   - **Infant- or eye-equal weighting** alongside image weighting.
   - **Leave-one-infant-out** influence ranges.
   - A **stratified infant-cluster bootstrap** with 10,000 resamples, sampling each reference class's clusters separately.
   - Report all three for Stage 1 recall and for Stage 4/5 / AP-ROP, where few infants exist.
2. **A development-only operating point.**
   - Choose the Stage-1 (or "any-ROP") threshold on **out-of-fold CV predictions** to hit a target recall (e.g., ≥ 0.80–0.90), take the maximum specificity among those, break ties toward the higher threshold, and freeze it before scoring the unseen hospital.
   - This is the cheapest lever on Stage 1 recall: argmax over class-balanced CE does not target recall. Their numbers also show the cost: DenseNet121 sensitivity 0.647 → 0.980 for specificity 0.979 → 0.468.
3. **Pre-specified eye-level aggregation.**
   - Mean and median can hide a positive eye (P01 had 5/20 positive frames). Max is fragile to one atypical frame (P07 was a false positive).
   - Top-k mean (k = 3) was the most balanced in their 8 infants.
   - Pre-register the rule and test it once on the unseen hospital. This complements RetinAI's v3 exam-level bag loss.
4. **Report within-eye variability** (mean adjacent |Δ|, threshold crossings) as a robustness metric. It explains Stage-1 misses that come from unrepresentative frames.
5. **A quality and posterior-pole-adequacy gate** as a future step. Their discussion argues for "assess quality → select informative views → aggregate". A cheap version is a frame-quality score (e.g., predicted entropy or a small gradability head) used to down-weight frames inside the eye-level bag.
6. **Two vessel facts.**
   - A green-overlay-on-dimmed-background input **hurt** (AUC 0.959 → 0.906), so any vessel input should leave the RGB untouched.
   - An infant-specific vessel U-Net (640-px patches, 0.3·BCE + 0.7·Dice inside ROI, two annotators' models averaged) is a sound recipe if RetinAI ever trains its own segmenter.
7. **Data pointer.** University Hospital Ostrava's Phoenix ICON archive has **infant IDs and eye laterality** (43 L / 55 R in the hold-out), unlike the public Kaggle release. It is not public, but the VSB group (Kubíček, Timkovič) might share it under agreement. It would give RetinAI true eye-level Ostrava labels.

---

## 5. Ranked adoption list for RetinAI (across all four papers)

The first four are cheapest and highest-yield for the weakest point (Stage 1 recall at the unseen hospital; rare Stage 4/5 and AP-ROP).

| # | Technique | Source | Why for RetinAI | Cost |
|---|---|---|---|---|
| 1 | Development-only, recall-targeted thresholds per class from out-of-fold CV predictions, frozen before the unseen hospital | P4 | Directly raises Stage 1 recall with no retraining, and makes the trade-off explicit | Very low |
| 2 | Infant/eye-equal weighting, leave-one-infant-out, stratified infant-cluster bootstrap (10k) | P4 | Honest CIs for rare classes; the paper rests on measurement findings, so this strengthens it | Low |
| 3 | Pre-specified eye-level aggregation (top-k mean; report mean/median/max) plus a frame-quality down-weighting | P4 (+P3 MIL) | Complements the v3 exam-level bag loss; avoids single-frame false positives and false negatives | Low |
| 4 | Checkpoint selection on ½(macro-F1 + Stage-1 recall) or worst-class recall; p = 0.5 class-balanced sampler for rare classes; focal α weighting | P3 | Aligns selection with the weakest metric; rare-class exposure | Low |
| 5 | High-resolution patch-MIL branch (≥768 px, 224 tiles, gated attention) stacked on out-of-fold logits with the global 392-px model | P3 | Thin demarcation lines and early vessel changes are lost at 392 px; distinct from the failed 518-px global attempt | Medium |
| 6 | Frangi green-channel vesselness as a 4th channel (zero/Kaiming init) or alpha-blend; never replace RGB | P3, P1 (+P4 negative) | Plus and AP-ROP signal; cheap to compute with skimage | Low–medium |
| 7 | Metadata (GA/BW/PMA) late fusion with modality dropout, reported with and without | P3 | PMA is a real Stage-1 prior, but a shortcut risk; usable only if the target site has it | Medium |
| 8 | FARFUM-RoP (and HVDROPDB) as extra external test sets | P2 | Adds a pre-plus label and RetCam Shuttle / Iran coverage | Low |

## 6. Numbers that contradict, or bear on, RetinAI's stated context

1. **Ostrava size.** All four sources give the public Ostrava set as **6,004 images / 188 infants** [FT, all; FT: Timkovič 2024].
   - RetinAI's pooled CV is "6,700 photos (Ostrava, ROP-VL, Shenzhen)". ROP-VL alone is about 2,020 images (internal note), so the 6,700 total only works if Ostrava was heavily subset.
   - P3 counts 1,295 "Other pathology" images and P1 a 880-image stage subset. Worth re-checking that the documented count matches the manifest.
2. **"Ostrava has no OD/OS."** True for the public release, which has a series (exam) number in the filename but no laterality [FT: Timkovič 2024]. But the **hospital's internal Phoenix ICON archive does carry eye laterality and infant IDs** (P4: 43 L / 55 R hold-out images) [FT].
3. **DG code mapping.** The dataset paper defines **DG2 = haemorrhage** [FT: Timkovič 2024]; P3 maps DG2 into "Mild/Treated" [FT]. RetinAI should confirm its own DG → stage table.
4. **Pre-plus.** The Ostrava dataset has **no pre-plus** [FT: Timkovič 2024]; P1 and P3 use binary plus (629 / 6,004). ROPDeepX's 3-class Normal/Pre-plus/Plus "RIDIRP" results (99.7%) are not credible.
5. **The 43% "non-ROP among ROP-positive" internal finding is confirmed independently**: P3's Table 2 gives 1,295 Other / 3,024 non-normal = 42.8% [FT-derived].
6. **Stage 4/5**: the dataset paper states ROP 4A, 4B and 5 have **no images** in Ostrava [FT: Timkovič 2024]. Any Stage 4/5 signal must come from ROP-VL or another source.
7. **Stage-1 benchmark.** On Ostrava with a patient-disjoint split, image-only models reach Mild/Treated sensitivity of only **0.21–0.32** (P3 Table 4) [FT]. RetinAI's Stage 1 recall of about 0.5 at an unseen hospital is not anomalous.

## 7. Source links
- P1: https://pmc.ncbi.nlm.nih.gov/articles/PMC12923887/ ; PDF https://www.nature.com/articles/s41598-026-37064-2.pdf
- P2: https://link.springer.com/article/10.1007/s44163-026-00834-y ; PDF https://link.springer.com/content/pdf/10.1007/s44163-026-00834-y.pdf
- P3: https://arxiv.org/abs/2602.05208 (v2 PDF read) ; repo https://github.com/mubid-01/MS-AQNet-VascuMIL-for-ROP_pre (README only)
- P4: https://www.frontiersin.org/articles/10.3389/fmed.2026.1960317/full ; PDF https://www.frontiersin.org/journals/medicine/articles/10.3389/fmed.2026.1960317/pdf ; code https://github.com/kaushikGeom/ROP_Classification
- Dataset paper used for cross-checks: Timkovič et al., *Sci Data* 11:814 (2024), https://www.nature.com/articles/s41597-024-03409-7 (its DG code table, Table 10, did not render in the HTML; only the worked filename example was read)
