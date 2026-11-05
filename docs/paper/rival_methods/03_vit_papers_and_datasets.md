# ROP transformer papers reporting 95–97% accuracy, and the datasets they used: what they actually did

Compiled 2026-10-30 for RetinAI. Scope: 5 method papers and 3 dataset papers (listed in the brief), read as fully as open access allows.

## Tag legend

| Tag | Meaning |
|---|---|
| **[FT]** | Read in the paper's full text |
| **[AB]** | Only the abstract or index metadata (OpenAlex, Semantic Scholar, PubMed, Europe PMC, Crossref) was available |
| **[EXT]** | From an external primary source that isn't the paper, such as the Kaggle API, Mendeley/figshare pages or a related open paper. The source is named each time. |
| **[LOCAL]** | Checked by me on RetinAI's own downloaded copy of the data, or taken from RetinAI's existing audit docs (`docs/DATASETS.md`, `data/rop/README.md`, `FINAL_MODEL_EVIDENCE_PACKAGE/03_DATA_LEAKAGE/*`) |
| **[INF]** | My inference or arithmetic. The paper doesn't state it, and it's marked so it can be checked. |

"not stated" means the accessible text doesn't give that detail. No numbers were guessed.

---

## 0. Summary

| # | Paper | Headline | Access | Data actually used | Split | Most likely reason the number is high |
|---|---|---|---|---|---|---|
| 1 | Remya Bharathy & Anoop, IJASIS 2026 | weighted F1 92.49, P 92.68, R 93.52 | FT | Claims HVDROPDB + Shenzhen + Ostrava, but its per-class counts match **none** of the public releases | "70/15/15", said to be "patient-wise" | **Not credible as reported** (see §1 weaknesses): non-resolving dataset URLs, a non-existent DOI, impossible counts, RETFound "fine-tuned" in MATLAB on a Core i5, reviewer-response text left in the paper |
| 2 | ROPGCViT, IEEE Access 2025 | Acc 94.69, κ 93.10 | AB (+ Crossref reference list) | 1,099 images = Shenzhen/Zhao 2024 (cited as ref 14) | not stated; **94.69% = 107/113 exactly, and 113 is the size of Zhao's published test split** [INF] | Probably Zhao's split, which isn't stated to be patient-level (789 eyes for 1,099 images). One 113-image test set (95% CI 88.9–97.5) was reused to choose among more than 50 models. Weighted metrics. The 5 classes include an easy Laser class. |
| 3 | Shri Vatssan et al., IConSCEPT 2025 (Swin-T) | Acc 95.03, macro-F1 0.916, macro-AUC 0.991 | AB | not named; 5 classes Normal–Stage 4 | "held-out test set", level not stated | Can't be judged. The 3.4-point gap between accuracy and macro-F1 points to weak minority classes. Horizontal-flip TTA. |
| 4 | Chen et al., IEEE JBHI 2025 (DS-Swin-T) | 97.91 average Acc on the source "style"; max 93.66 on target styles | AB | not stated (Shenzhen Eye Hospital group; probably in-house) | not stated | 97.91 is **in-distribution**. The honest cross-domain figure is ≤93.66. |
| 5 | Malarkodi, ICISC 2025 (DeepROP) | Acc 96.5 | AB + Kaggle API + LOCAL | Kaggle "ROP-stages_reviewed": **91 files, 85 unique** (6 byte-identical pairs, 5 of them in Stage 1), unknown provenance and licence | not stated | **96.5% can't be k/n for any n < 57** [INF]. A held-out part of 91 files is about 27 images, so the denominator must include augmented copies, the whole set, or a fold average. The abstract does describe "data augmentation … and stratified sampling to ensure class balance". |
| 6 | Zhao et al., Sci Data 2024 (Shenzhen) | Baseline ResNet50 Acc 87.61 | FT | 1,099 images / 789 eyes / 483 infants, 5 classes | 8:1:1 (877/109/113); patient- or image-level **not stated** | — (dataset paper; its own baselines are 80–88%) |
| 7 | HVDROPDB, Data in Brief 2023 | Segmentation baseline Dice 0.52–0.92 | FT + Mendeley v3 + LOCAL | Public release: segmentation sets of 3×(50 RetCam + 50 Neo) images + masks; v3 adds **binary** ROP/Normal (185 files) | recommended 70:10:20, random_state 42 (image-level; no patient IDs) | — (785 files, but 329 unique photographs) |
| 8 | ROP-VL, Sci Data 2026 | not given in abstract | AB + figshare + LOCAL | 2,020 images / 1,116 infants / 8 classes | not stated | — |

**My reading:** RetinAI's 0.90 accuracy on Shenzhen under patient-grouped CV is already higher than the Shenzhen dataset's own published baselines (ResNet50 87.61, ViT-B 84.07 [FT]), and those baselines used an easier protocol whose split level isn't stated. The 95–97% papers differ from RetinAI mainly in evaluation protocol (split level, test-set size, model selection on the test set, in-distribution testing, weighted metrics), not in architecture.

---

## 1. Remya Bharathy K & Anoop V, "Automated Screening of ROP Based on ICROP3 Classification Protocol Using Hybrid Deep Neural Networks", IJASIS 2026

### Citation and access level
- Int. J. Adv. Sig. Img. Sci. (IJASIS) Vol. 12, No. 4s, 2026, pp. 242–265. Jyothi Engineering College, Thrissur (APJ Abdul Kalam Technological University) [FT].
- **Full text read** (24-page open PDF from xlescience.org, extracted with pdftotext) [FT].
- Note: the brief describes this paper as using "SimCLR, RETFound, diffusion, 70/15/15". It does mention all four, but gives almost no parameters for any of them.

### Dataset(s)
- Three "public" sets, per Table 1 [FT]:

| Dataset (as named in paper) | Total | Stage 0 | Stage 1 | Stage 2 | Stage 3 | Plus | Synthetic | Split |
|---|---|---|---|---|---|---|---|---|
| HVDROPDB [Public] | 614 | 65 | 138 | 146 | 139 | 126 | 60 (Stages 1–3) | 70/15/15 |
| Fundus Image Dataset for ROP Diagnosis [Public] | 1,099 | 392 | 207 | 232 | 169 | 0 | 40 (Stages 1–3) | 70/15/15 |
| Retinal Image Dataset of Infants [Public] | 6,004 | 4,273 | 20 | 15 | 15 | 630 | 50 (Stages 1–3) | 70/15/15 |

- Text: the Ostrava set "contains around 6004 retinal images from 188 patients across 484 examinations", and the second set has "1,099 images from 483 premature infants" [FT].
- Dataset links given: `https://hvdropdb.org`, `https://fundusrop-dataset.org`, and DOI `10.1016/j.biomedimag.2023.105982` for the Ostrava set [FT].
- Cameras: not stated. Images per infant: not stated.
- **Cross-check against the real releases:**
  - `hvdropdb.org` and `fundusrop-dataset.org` **don't resolve (DNS ENOTFOUND)** [EXT, fetched 2026-10-30].
  - The Ostrava DOI `10.1016/j.biomedimag.2023.105982` returns **404 at Crossref**. The real DOI is 10.1038/s41597-024-03409-7 [EXT].
  - **Shenzhen (Zhao 2024)** really is Normal 236 / S1 94 / S2 165 / S3 261 / Laser 343 [FT of paper 6]. The paper's 392/207/232/169 don't match, the class sum is 1,000 rather than 1,099, and there is no Laser class.
  - **Public HVDROPDB** has **binary** ROP/Normal labels only (Neo_Normal 50, Neo_ROP 35, RetCam_Normal 50, RetCam_ROP 50) plus segmentation sets. It has no stage or plus labels [LOCAL, Mendeley v3]. So the paper's 65/138/146/139/126 can't come from the public release.
  - **Ostrava (Timkovič 2024)** stage counts are S1 252 / S2 458 / S3 125 [EXT, PMC11266588], not 20/15/15. The paper's row sums to 4,953, not 6,004.

### Split
- "Both datasets were divided into 70 per cent training, 15 per cent validation and 15 per cent testing" [FT]. Fig. 4 text says "approximately 70–75%" for training [FT].
- Claimed: "patient-wise splitting was imposed" and "patient-wise cross-validation" [FT]. Public HVDROPDB has no patient IDs [LOCAL], so a patient-wise split is impossible on that set.
- Test-set size: not stated. External test: none. "Cross-dataset evaluation" (F1 90.8–92.0) is reported, but the train/test pairing isn't described [FT].

### Preprocessing
- In order [FT]:
  1. Grayscale conversion.
  2. Adaptive Histogram Equalization. Equations are given; tile size and clip limit aren't, although a sensitivity analysis says results moved less than ±0.9 "across tile sizes and clip limits".
  3. Median filter (window size not stated).
  4. "Normalization and resizing to suit the input needs of foundation models such as RETFound". The input is 224×224 per the complexity statement.
- Optic disc located with a Hough circle transform; Hessian (Frangi-type) vesselness with multi-scale **σ ∈ 2–4**. β and C are not stated [FT].
- Images that were low quality, incomplete or corrupted were removed by "domain specific filtering" (criteria not stated) [FT].

### Augmentation
- Ops: elastic deformations, rotations, "localized zoom-in patches". Table 3 text says rotation, flipping and scaling [FT].
- Parameters: none stated.
- Online vs offline: not stated. Before or after split: not stated for classical augmentation.
- **Synthetic images:**
  - Method: "diffusion-based generative model" (model, conditioning, steps and filtering not stated).
  - Counts: 60 / 40 / 50 for Stages 1–3 per dataset (Table 1), with the per-class breakdown not stated [FT].
  - "Synthetic samples were only present in the training set" [FT].
- SimCLR self-supervised pre-training "on unlabelled images", source images not stated. The paper claims it "improves cross-dataset generalization by 1.6% in macro-F1" [FT].

### Model
- **Segmentation path** [FT]:
  - LadderNet. It was compared with U-Net and Attention U-Net and gave +1.8–2.3% Dice.
  - A GNN over a vessel graph built from skeletonised masks (nodes at bifurcations and endpoints), running on features from a fine-tuned RETFound encoder.
  - Shape-prior level-set refinement (Chan–Vese-type equation).
  - Radon transform for vessel width and orientation.
- **Classification path** [FT]:
  - RETFound encoder combined with EfficientNet (variant not stated) and SE blocks.
  - Transformer "Vascular Attention Module".
  - Quantitative biomarkers (density, curvature, branching angle).
  - "Domain adaptation layer" (method not stated).
  - SHAP "explainability layer".
  - Auxiliary plus-disease branch. Output is stages 0–3 plus plus disease.
- Size: about 23 M trainable parameters, 92 MB FP32, 7.8 GFLOPs at 224×224 [FT].
- [INF] RETFound is ViT-Large (about 300 M parameters), so "RETFound encoder + EfficientNet ≈ 23 M parameters" is internally inconsistent.

### Training
- Table 4 [FT]:

| Setting | Value |
|---|---|
| Optimizer | Adam |
| LR | 1e-4 |
| Batch | 16 |
| Epochs | 50 |
| Weight decay | 1e-5 |
| RETFound | fine-tuned |
| Multi-task loss weights (λ, α, β) | 0.6, 0.3, 0.1, grid-searched on validation macro-F1 |

- Loss types: not stated.
- Imbalance handling: diffusion synthesis only (no weighting or sampler stated).
- Early stopping: "early stopping using weight decay regularization" [FT].
- Fig. 12 shows **10** epochs, with validation loss rising after epoch 7 [FT].
- Hardware: **MATLAB R2024a, Windows 10 Home, 16 GB DDR4, Intel Core i5 @3.5 GHz. No GPU is listed** [FT].

### Evaluation
- Metrics [FT]:
  - Precision, recall, F1 (weighted and macro).
  - Per-stage sensitivity and specificity.
  - Dice.
  - AUC is mentioned but never reported.
  - SHAP–clinician agreement.
- Unit: image (implied). CIs: none. 5-fold CV is reported only for segmentation Dice (90.8 ± 1.7) [FT].

### Results
- **Headline** [FT]: weighted F1 92.49, macro F1 91.8, recall 93.52. The abstract calls 92.68 "precision", the body calls it "accuracy".
- **Per stage** [FT]:
  - Sensitivity 92.0 (S3) to 94.0 (S0).
  - Specificity 90.5 (S1) to 92.5 (S3).
  - Plus-branch sensitivity 93.4.
  - Stage 3 false negatives: "7 cases (5.8%)".
  - Confusions: S2→S3 is 34% of errors, S3→S2 is 26%.
- **Comparisons and ablation** [FT]:
  - Cross-dataset F1 90.8–92.0.
  - Against "CNN, ResNet18, RETFound, MedSAM, RETFound 2.0": F1 88.2–92.7.
  - Ablation: removing the GNN costs 1.9–2.4 F1; other modules cost 0.5–1.5.
- **Segmentation** [FT]: Dice LadderNet 84–88 → full 88–93. Radon width vs experts (n = 50): r = 0.89, MAE 0.42 px.
- [INF] 7 = 5.8% implies about 121 Stage-3 test cases, but 15% of the paper's own Stage-3 total (139+169+15 = 323) is about 48.

### Key method references cited
- The paper cites **no primary reference** for SimCLR, RETFound, LadderNet, EfficientNet, SE, SHAP, diffusion models or level sets [FT].
- ROP-ML references it does cite [FT]:
  - [7] Rahim et al. 2023, "Novel fundus image preprocessing for RetCam images…" (arXiv 2302.02524)
  - [8] Absulhussein et al., SWIN-Transformer ROP staging
  - [21] Agrawal et al. 2025, MultiCNN_LSTM (MethodsX 14:103182)
  - [10] Burt et al. 2025, UWF-OCT stage
  - [22] Sutter et al. 2025, OCT zone
  - [3] Belenje et al. 2025, AI vs clinicians
- Citation mismatch: the text says "Bandi et al. [27] … SWIN Transformer", but [27] is a screening-programme audit in Limpopo [FT].

### Weaknesses / leakage risks
1. The dataset links don't resolve and the Ostrava DOI doesn't exist [EXT].
2. The class counts match none of the public releases (see the table above). The HVDROPDB stage and plus labels don't exist publicly.
3. A claimed "patient-wise split" is impossible on HVDROPDB, which has no IDs.
4. RETFound fine-tuning plus a GNN is claimed on a CPU-only MATLAB setup, and the parameter count is inconsistent with RETFound.
5. Text written for reviewers was left in the paper. Examples: "overcomes the objections of reviewers", "responds to reviewer issues", "This table responds to the issues of the reviewers" [FT].
6. Metric naming is inconsistent (precision vs accuracy). The epoch count is inconsistent (50 in the table, 10 in the figure).
7. Diffusion synthesis is described without a model, sampler, or a check that synthetic images don't copy test images.
8. There are no CIs, and no per-dataset test sizes.
- **Verdict:** don't cite this paper as evidence for any technique.

### What RetinAI could adopt
- Almost nothing at face value.
- The one testable idea is a **multi-scale Frangi vesselness map (σ ≈ 2–4 px at native scale) as an auxiliary input channel or auxiliary target**, aimed at ridge and vessel detail for Stage 1. This paper gives no trustworthy evidence that it helps, so treat it as a small ablation, not a recommendation.
- Its grayscale conversion would discard colour cues (white or grey ridge versus orange retina). Don't copy it.

---

## 2. ROPGCViT (Yurdakul, Uyar, Taşdemir, Atabaş), IEEE Access 2025

### Citation and access level
- M. Yurdakul, K. Uyar, Ş. Taşdemir, İ. Atabaş, "ROPGCViT: A Novel Explainable Vision Transformer for Retinopathy of Prematurity Diagnosis," IEEE Access 13:77064–77079, 2025, DOI 10.1109/ACCESS.2025.3564213, CC BY 4.0 (gold OA) [AB].
- **Access: abstract plus Crossref reference list only.** Three routes failed, and none was worked around:
  - IEEE Xplore: the HTML is JavaScript-only (empty to the fetcher), and `stamp.jsp` returned **HTTP 418**.
  - Alanya Alaaddin Keykubat University repository (handle 20.500.12868/5065): **HTTP 403** on both the item page and the bitstream.
  - DOAJ: **HTTP 403**.

### Dataset(s)
- "1,099 fundus images" [AB].
- Reference 14 in the Crossref reference list is **10.1038/s41597-024-03362-5 = Zhao et al. Sci Data 2024 (Shenzhen)** [AB/metadata].
- [INF] The dataset is therefore almost certainly the Shenzhen set described in §6:
  - Normal 236 / S1 94 / S2 165 / S3 261 / Laser 343.
  - 789 eyes, 483 infants.
  - Devices: RetCam, SW-8000, Nautilus RS-B002.
  - 512×512 images.
- Whether ROPGCViT kept the Laser class: not stated in the abstract. 1,099 is the full set including Laser, so [INF] it most likely did.

### Split
- Not stated [AB].
- [INF] **94.69% = 107/113 exactly.** 113 is the *smallest* test-set size for which any k/n rounds to 94.69%, and 113 is exactly the size of Zhao's published test split (Table 1: 877/109/113). The authors therefore very probably used Zhao's 8:1:1 split, which Zhao doesn't describe as patient-level.
- External test: none mentioned.

### Preprocessing
- Not stated [AB].

### Augmentation
- Not stated [AB].
- [EXT, indicative only] The same first author's MaxGlaViT paper (arXiv 2502.17154, Feb 2025, same group, same "test 80 models then modify the best" design) used:
  - Keras/TensorFlow, ImageNet transfer.
  - Categorical CE, Adam LR 1e-3, LR "weight reduction value of 0.8".
  - Batch 16, 50 epochs.
  - *Online* augmentation (scaling, rotation, vertical/horizontal shift).
  - The dataset's own predefined train/val/test split.
  - 2× RTX 3090.
- Don't attribute this recipe to ROPGCViT without the full text.

### Model
- Global Context ViT (GCViT, Hatamizadeh et al., ICML 2023) "enhanced with Squeeze-and-Excitation (SE) block and Residual Multilayer Perceptron (RMLP) structures" [AB].
- Benchmarked against **50 models (25 CNN + 25 ViT)** [AB]. Grad-CAM for explanation [AB].
- Pretrained weights and input size: not stated.

### Training
- Loss, imbalance handling, optimizer, schedule, epochs, batch, early stopping, hardware: all not stated [AB].

### Evaluation
- Accuracy, precision, recall, F1, Cohen's κ [AB].
- [INF] Recall (94.69) equals accuracy (94.69). Weighted-average recall always equals accuracy, so the metrics are **support-weighted**, not macro, and they hide minority-class (Stage 1) performance.
- CIs: none in the abstract. [INF] The Wilson 95% CI for 107/113 is **88.9–97.5%**.

### Results
- Acc 94.69, Precision 94.84, Recall 94.69, F1 94.60, κ 93.10 [AB]. Per-class: not available.

### Key method references cited
From the 40 Crossref references [AB/metadata]:
- Hatamizadeh et al. "Global Context Vision Transformers" (ICML)
- Dosovitskiy et al. ViT (arXiv 2010.11929)
- DenseNet, GoogLeNet, ResNet, Xception, MobileNets, CSPNet, GhostNet, ResNeSt
- **Zhao et al. 2024 Sci Data** (the dataset)
- Huang et al. 2020, Eye Vis (10.1186/s40662-020-00206-2)
- GabROP (Diagnostics 2023)
- DIAROP (Diagnostics 2021)
- Semi-supervised ROP staging (iScience 2023)
- "Assessing spectral effectiveness in colour fundus photography for DL classification of ROP" (J Biomed Opt 2024)
- Self-supervised ROP detection (Electronics Letters 2024)
- Quantum-ML ROP (IEEE Access 2023)
- MaxGlaViT (own prior work)

### Weaknesses / leakage risks
1. **Single dataset, single test set of probably 113 images.** One error is worth 0.88 points [INF].
2. **Model selection on that same split.** More than 50 candidates plus architecture variants were compared on one test set, which inflates the winner. Whether the validation set alone drove selection: not stated.
3. **Split level.** The Shenzhen set has 1,099 images from 789 eyes, so ≥310 images (28%) are second or later images of an eye already in the set [INF from FT §6]. A random image-level 8:1:1 split places many same-eye or same-infant pairs across train and test.
4. **Known defects in the Shenzhen files that leak under image-level splitting** [LOCAL]:
   - 6 byte-identical photographs filed under two different patient IDs.
   - 3 images filed under two different stage folders (contradictory labels).
   - About 20 photographs also appear in ROP-VL.
5. **Weighted metrics, and an easy Laser class.** The Laser class (31% of images) inflates 5-class accuracy. The same group's SABPI-Net reports that almost all models get high sensitivity on Laser-treated and Normal [EXT, SABPI-Net FT].
6. No external test and no CIs.

### What RetinAI could adopt
- **Not the architecture.** Adding SE and residual MLP heads to a DINOv2 backbone is a minor change with no evidence it transfers.
- **The protocol, deliberately, as a measurement.** Re-run the current model on Shenzhen with a *random image-level 8:1:1 split* (Zhao's per-class counts: 877/109/113) and report it next to the patient-grouped 0.90. If it lands around 94–95%, the paper can show that ROPGCViT's margin comes from protocol, not architecture. This is cheap and turns a competitor number into evidence.

---

## 3. Shri Vatssan, Devishree, Sathishkumar, "Multistage Retinopathy of Prematurity Classification using Swin Transformer with Hybrid Optimization Strategy", IConSCEPT 2025

### Citation and access level
- S. Shri Vatssan (NIT Puducherry), T. Devishree and R. Sathishkumar (Perunthalaivar Kamarajar Institute of Engineering and Technology, Karaikal).
- 2025 Int. Conf. on Signal Processing, Computation, Electronics, Power and Telecommunication (IConSCEPT), pp. 1–6, published 6 Dec 2025. DOI 10.1109/IConSCEPT66142.2025.11436196 [AB].
- **Abstract only.** OpenAlex and Semantic Scholar list it as closed, with no repository copy. No GitHub or Kaggle code was found.

### Dataset(s)
- Not named [AB]. Classes: five, "Normal to Stage 4" [AB].
- [INF] Among public ROP sets, only ROP-VL (Stage 4: 59 images) and the HF Multi-View set (Stage 4: 18) contain Stage 4 [LOCAL counts]. Shenzhen, Ostrava and HVDROPDB don't. So the data is one of those two, a re-upload of one, or private. Not determinable.

### Split
- "Held-out test set" [AB]. Ratios, level and size: not stated.
- [INF] If it is a single test set, 95.03% needs n ≥ 161 test images.

### Preprocessing
- Input 224×224 ("Swin-Tiny@224") [AB]. Anything else: not stated.

### Augmentation
- **Test-time augmentation with horizontal flip** "to improve prediction stability" [AB]. Training augmentation: not stated. Synthetic images: none mentioned.

### Model
- Swin-Tiny Transformer [AB]. Pretrained weights: not stated.
- "Hybrid Optimization Strategy" is not defined in the abstract. [INF] It probably refers to the hybrid loss below.

### Training
- Loss: **"hybrid Weighted Cross-Entropy and Label Smoothing (W-CE + LS)"** for class imbalance and label uncertainty [AB]. Class weights and smoothing ε: not stated.
- Optimizer, LR, schedule, epochs, batch, early stopping, hardware: not stated.

### Evaluation
- Test loss, accuracy, macro-F1, macro-AUC, Grad-CAM [AB]. Unit: image (implied). CIs: none.

### Results
- Test loss 0.1661, **accuracy 0.9503, macro-F1 0.916, macro-AUC 0.9907** [AB].
- Grad-CAM attends to ridge and vascular regions, with "minor overlap between Stage 1 and Stage 2" [AB]. Per-class: not available.

### Key method references cited
Nine references, via OpenAlex [AB/metadata]:
- Huang et al. 2020 (Eye Vis)
- Wang et al. 2021, "Automated explainable multidimensional DL platform…" (JAMA Netw Open)
- Quantum-ML ROP (IEEE Access 2023)
- "Automated detection of nine infantile fundus diseases…" (EPMA J 2024)
- "Development of an automated screening system for ROP using a DNN for wide-angle retinal images" (IEEE Access 2018)
- Plus disease CNN + feature extraction (TVST 2020)
- Agrawal et al. ROP segmentation (MTAP 2022)
- Plus-disease detection in preterm fundus images (BBE 2021)
- Dual-Branch Attention Network + Swin-SPP for ROP (ISBI 2023)

### Weaknesses / leakage risks
- Dataset unknown, split level unknown, no external test, no CIs.
- The accuracy-to-macro-F1 gap (95.0 vs 91.6) shows that at least one minority class, probably Stage 4 or Stage 1, is much weaker than the headline [INF].

### What RetinAI could adopt
- **Horizontal-flip TTA.** Average softmax over the identity and h-flipped image (RetinAI already trains with H/V flips, so both views are in-distribution). It costs 2× inference and usually gives a small, stable gain plus slightly better calibration. Verify on CV folds, not on the locked test.
- **Weighted CE with label smoothing.** RetinAI already does this (smoothing 0.05), so there's nothing new here.

---

## 4. Chen, Zhao, Liu et al., "Dual-Scale Swin Transformer via Feature Alignment and Adversarial Discrimination for ROP Diagnosis", IEEE JBHI 2025

### Citation and access level
- Shaobin Chen, Xinyu Zhao, Yiyao Liu, Hai Xie, Zhenquan Wu, Yingpeng Xie, Yongtao Zhang, Yafeng Li, Cheng Zhao, Tianfu Wang, Guoming Zhang, Baiying Lei.
- IEEE J Biomed Health Inform 2025 Dec; 29(12):9013–9026. DOI 10.1109/JBHI.2025.3573318, PMID 41289110 [AB].
- Affiliations: Shenzhen University and Shenzhen Eye Hospital (Southern Medical University) [AB/metadata].
- **Abstract only.** There's no PMC copy. Europe PMC says "subscription required", OpenAlex says closed, and no arXiv, author copy or GitHub was found.

### Dataset(s)
- Not stated. The abstract refers to a "source style dataset" and "other target style datasets", without defining "style" (camera, hospital or other) [AB].
- [EXT, context only, not this paper] The same hospital's public set (Zhao 2024) uses three devices (RetCam, SW-8000, Nautilus RS-B002) [FT §6]. The same first author's MICCAI 2025 SABPI-Net uses an in-house set of **174,540 images from 13,314 patients** at Shenzhen Eye Hospital:

| SABPI-Net class | Images |
|---|---|
| Normal | 112,770 |
| Laser-treated | 29,442 |
| Any-stage ROP | 19,513 |
| A-ROP | 2,363 |
| Other | 10,452 |

  SABPI-Net splits it 7:2:1 **at patient level**. Whether DS-Swin-T uses these data: not stated.

### Split
- Not stated [AB].

### Preprocessing
- Not stated [AB].

### Augmentation
- An **image synthesis (IS) module** generates "intermediate-style" images to bridge the gap between styles [AB]. Method (GAN or other) and counts: not stated [AB].
- [INF] The reference list (CycleGAN, pix2pix, StyleGAN, conditional/context-aware GAN synthesis) suggests a GAN-based translator.

### Model
- **DS-Swin-T**: dual-scale Swin Transformer [AB].
- **Feature alignment** in shallow feature space, using a *detail encoder* and a *style encoder* with detail and style losses [AB].
- **Adversarial learning** in deep feature space to learn "style-invariant features within the same class" [AB].
- **Feature fusion units** combine the two scales [AB].
- Pretrained weights and input size: not stated.

### Training
- Not stated [AB].

### Evaluation
- "Average accuracy", source style vs target styles [AB]. What is averaged (folds, runs or classes): not stated. CIs: not stated.

### Results
- **97.91% average accuracy on the source-style dataset. Maximum 93.66% average accuracy when transferred to target-style datasets** [AB].

### Key method references cited
From the 55 OpenAlex references [AB/metadata]:
- Swin Transformer, ResNet, ConvNeXt, CrossViT, DS-TransUNet
- **CycleGAN, pix2pix, StyleGAN**
- **DANN (domain-adversarial training), ADDA**, AdaptSegNet, DRCN, CAN
- "Unsupervised DA based image synthesis and feature alignment for joint OD/OC segmentation" (2021). This is probably the template for the IS + alignment design.
- REFUGE
- FIT-Net
- ROP: Brown 2018 plus disease; Wang 2018 automated ROP screening DNN (the original "DeepROP"); "Automatic staging for ROP with deep feature fusion and ordinal classification strategy" (2021); A-ROP attentive CNN (2021); "Multinational external validation of autonomous ROP screening" (2024)

### Weaknesses / leakage risks
- The 97.91% is **in-distribution**. It drops by ≥4.25 points even for a method built for style shift, and the best target result is 93.66.
- The data is probably private, the split level isn't stated, and "average" is undefined.
- [EXT] Cautionary context from the sibling SABPI-Net (FT):
  - Its 95.32% accuracy comes with only **61.5% A-ROP sensitivity**, because Normal is 64.6% of the data.
  - In the same table, the baselines range from ResNet50 90.96 to Swin-S 93.07, with RETFound at 92.30 and ViT-B at 91.19.
  - So "≥95% accuracy" in this group's work means a large, Normal-heavy, patient-split in-house set, not good minority-class staging.

### What RetinAI could adopt
- **The idea, in its cheap form.** The same group's open-access SABPI-Net implements style-invariance with a *Fourier low-frequency amplitude swap plus feature-consistency loss* instead of GAN synthesis plus adversaries (details in §9, item 1).
- **Class-conditional adversarial alignment** (cohort or device as the domain, per class) is the heavier fallback. It must be *intra-class*, as the abstract says. Stage mix differs strongly by cohort, so an unconditional domain adversary would also strip stage information.

---

## 5. Malarkodi, "DeepROP", ICISC 2025, and the Kaggle "ROP Stages Reviewed" dataset

### Citation and access level
- K. Malarkodi (Saveetha School of Engineering, SIMATS, Saveetha University, Chennai), "DeepROP: Automated Stage-Wise Classification of Retinopathy of Prematurity Using Enhanced Retinal Fundus Imaging."
- 2025 9th Int. Conf. on Inventive Systems and Control (ICISC), pp. 1009–1015, published 12 Aug 2025. DOI 10.1109/ICISC65841.2025.11187674 [AB].
- **Abstract only** (closed; no repository copy).
- Not to be confused with Wang et al. 2018 "DeepROP" (EBioMedicine), which is a different system.

### Dataset(s)
- The abstract says the model is "trained on the clinically annotated 'ROP Stages Reviewed' dataset" [AB].
- **Kaggle dataset identified** [EXT, Kaggle public API, 2026-10-30]:

| Field | Value |
|---|---|
| Ref | `solennollivier/ropstages-reviewed` |
| Title | "ROP-stages_reviewed" |
| Owner | Solenn Ollivier |
| Version | One version, created **2021-04-03**, note "Initial release" |
| Licence | **Unknown** |
| Description | **None** (empty subtitle and description) |
| Tag | "earth and nature" |
| Size | 136 MB |
| Usability | 0.31 |
| Downloads | 732 |
| Kernels | 3 |

- **Contents:** one folder, **`NewROPDataset_Sample_justtotry/`**, with **91 JPEGs**. The class is encoded only in the filename prefix [EXT]:

| Prefix | Files | Unique (md5) [LOCAL] |
|---|---|---|
| `NoROP_` | 39 | 38 |
| `Stage1_` | 19 | **14** |
| `Stage2_` | 22 | 22 |
| `Stage3_` | 11 | 11 |
| **Total** | **91** | **85** |

- **Byte-identical duplicate pairs** [LOCAL, md5 on RetinAI's copy]:
  - `Stage1_Patient13_2 = Stage1_Patient3`
  - `Stage1_Patient13 = Stage1_Patient3_2`
  - `Stage1_Patient12 = Stage1_Patient4`
  - `Stage1_Patient1 = Stage1_Patient15_2`
  - `Stage1_Patient14 = Stage1_Patient2`
  - `NoROP_Patient9 = NoROP_Patient9_3`
- **Duplicates span different "Patient" numbers**, so the patient numbers can't be trusted for grouping. RetinAI's docs record 85 images / 65 "infants" [LOCAL].
- Augmented copies: none detected [LOCAL]. A 256-bit dHash comparison of every image against every other image's h-flip, v-flip and 180° rotation found no flipped or rotated copies. The closest non-identical cross-class pair (NoROP_Patient22 vs Stage1_Patient10, dHash distance 5) was inspected visually and is a different eye or photograph.
- All 91 images are **1600×1200** with no EXIF camera data [LOCAL]. Camera, hospital, country, grading protocol and consent: **all unknown**.
- Most images have a strong blue cast (B/R > 1 in 90%), which does not separate the classes [LOCAL].
- No Stage 4/5 class, no plus labels.

### Split
- Not stated [AB]. The abstract says "stratified sampling techniques to ensure class balance" [AB].

### Preprocessing
- "Rigorous preprocessing" and "enhanced retinal fundus imaging" (no detail) [AB].

### Augmentation
- "Data augmentation" (no detail) [AB]. Whether it happened before the split: not stated. Synthetic images: none mentioned.

### Model
- Hybrid of **EfficientNet-B4 + Swin Transformer**, for "local fine-grained" and "global structural" features [AB]. Fusion method, pretrained weights and input size: not stated. Grad-CAM [AB].

### Training
- Not stated [AB].

### Evaluation
- Accuracy, compared against ResNet-50, VGG-16 and InceptionV3 [AB]. CIs: none.

### Results
- **Accuracy 96.5%** [AB]. Per-class: not available.

### Key method references cited
Fifteen references, via OpenAlex [AB/metadata]:
- **Relevant:**
  - Zhao 2024 Sci Data (Shenzhen), which is cited but apparently not used
  - Huang 2021 "DL for diagnosis of stage in ROP" (Ophthalmol Retina)
  - DIAROP (2021)
  - Wang 2021 JAMA Netw Open
  - Hybrid DL for ROP (BSPC 2023)
  - Agrawal 2022 segmentation
  - Plus classification with transform features (2023)
  - Spectral effectiveness (JBO 2024)
  - P Score (Ophthalmology 2024)
  - Image-analysis ML for ROP (Ophthalmol Retina 2024)
  - AI-in-ROP review (2025)
- **Off-topic:** Inception-V3 for tuberculosis CXR, VGG19 for apple leaves, transfer learning for pneumonia.

### Weaknesses / leakage risks
1. **Tiny, unverified, partly duplicated source.** 91 files, 85 unique, with Stage 1 at only 14 unique images. The folder is literally named "Sample_justtotry". Licence and provenance are unknown.
2. **The arithmetic doesn't fit a clean held-out set** [INF]:
   - 96.5% (one decimal) is not k/n for any n < 57.
   - A 70/30 or 80/20 split of 91 files gives a test set of about 18–27 images. The closest attainable values are 96.0 (24/25), 96.2 (25/26), 96.3 (26/27) and 96.4 (27/28), never 96.5.
   - Testing on all 91 files gives 95.6 or 96.7.
   - So the reported figure implies (a) a test set containing augmented or oversampled copies, (b) evaluation on the whole dataset including training images, or (c) an average over folds or runs. None of these is stated.
   - Combined with "data augmentation … stratified sampling to ensure class balance", **augmentation or oversampling before the split** is the most likely explanation. It is not proven.
3. **Duplicate leakage is guaranteed** under any image-level split. Five Stage-1 photographs exist twice, so a random split very likely places a copy on each side.

### What RetinAI could adopt
- **Nothing from the model.**
- On the data: keep RetinAI's existing decision to exclude `ropstages_sample` (`results/rop/inclusion_flow.json`: "sample too small to fold (91 images) and provenance unverified").
- If the paper discusses this work, the duplicate count (6 pairs) and the k/n arithmetic are a concise, checkable rebuttal.

---

## 6. Zhao et al., "A fundus image dataset for intelligent retinopathy of prematurity system", Sci Data 2024 (Shenzhen)

### Citation and access level
- Zhao X, Chen S, Zhang S, Liu Y, Hu Y, Yuan D, Xie L, Luo X, Zheng M, Tian R, Chen Y, Tan T, Yu Z, Sun Y, Wu Z, Zhang G. Sci Data 2024;11:543. doi:10.1038/s41597-024-03362-5, PMC11130119 [FT].
- **Full text read.** It was read from PMC first, and the quotes were confirmed through Europe PMC fullTextXML after PMC began serving a browser check.

### Dataset(s)
- Source: Shenzhen Eye Hospital, collected 2004–2023 [FT].
- Devices: **RetCam (Clarity), ROP Screening System SW-8000, Nautilus RS-B002**, all wide-field contact cameras. Per-device counts: not stated [FT].
- Pupils dilated with tropicamide about 30 minutes before imaging. Six standard orientations were captured (disc-centred, macula-centred, temporal, nasal, superior, inferior). Mostly temporal views were selected, with the posterior pole used when temporal quality was poor [FT].
- **1,099 images / 789 eyes / 483 infants** [FT]. Images per infant: not stated ([INF] about 2.3 on average).
- Demographics: 36.44% female; birth weight 1,247 ± 432 g; GA 29.10 ± 2.62 weeks; all Asian [FT].
- "The same eye of an infant may include more than one fundus image" (follow-ups, laser) [FT].
- Classes (ICROP3) with the official split [FT]:

| Class | Train | Val | Test | Total | Eyes | Infants |
|---|---|---|---|---|---|---|
| Normal | 188 | 24 | 24 | 236 | 215 | 130 |
| Stage 1 | 75 | 9 | 10 | 94 | 87 | 71 |
| Stage 2 | 132 | 16 | 17 | 165 | 138 | 112 |
| Stage 3 | 208 | 26 | 27 | 261 | 227 | 162 |
| Laser scars | 274 | 34 | 35 | 343 | 280 | 163 |
| Total | 877 | 109 | 113 | 1,099 | 789 | 483 |

- Excluded: Stages 4–5 (rare), redundant images, poor-quality images (unclear lesions, too bright or too dark) [FT]. Zone and plus are not labelled [FT].
- Grading [FT]:
  - Two junior annotators graded independently; a senior annotator re-graded discordant cases.
  - Inter-annotator κ = 0.856. Intra-annotator κ = 0.929 and 0.919.
  - **Disagreement was most evident between Stage 1 and Stage 2.**
- Files [FT]: figshare 10.6084/m9.figshare.25514449, one zip, folders by class.
- [LOCAL] The zip includes `information.xlsx` (image → infant → eye). All images are 512×512.

### Split
- "According to the ratio of 8:1:1" [FT]. Random or patient-grouped: **not stated** [FT].
- The infant column doesn't sum (infants appear in several classes), and the per-class split counts are given without infant breakdown [FT].
- Test set: 113 images. External test: none.
- **[LOCAL] Known defects relevant to any split:**
  - 6 byte-identical photographs filed under two different `patient_id`s.
  - 3 images filed in two different stage folders (contradictory stage labels).
  - About 20–22 photographs also appear, at higher resolution, in ROP-VL.
  - RetinAI's audit concludes that Shenzhen IDs "behave like records, not people".

### Preprocessing
- "Uniformly cropping the images to a resolution of 512 × 512", preserving the aspect ratio of the retinal region [FT]. Nothing else stated.

### Augmentation
- "Random crop, random horizontal and vertical flipping, etc." [FT]. Parameters: not stated. Online/offline: not stated. Synthetic images: none.

### Model
- Baselines: ResNet50, ResNet101, ConvNeXt-T, ViT-B [FT]. Pretrained weights and input size: not stated.

### Training
- SGD, LR 1e-3, momentum 0.9, weight decay 1e-4, batch 64, one NVIDIA V100 [FT].
- Loss, schedule, epochs, early stopping and imbalance handling: not stated [FT].

### Evaluation
- Accuracy, AUC, precision, sensitivity, F1, specificity, κ on the 113-image test set [FT]. Image-level. No CIs.

### Results (Table 2) [FT]

| Model | Acc | AUC | Pre | Sen | F1 | Spec | κ |
|---|---|---|---|---|---|---|---|
| ResNet50 | **87.61** | 96.71 | 82.64 | 83.31 | 82.81 | 97.04 | 83.97 |
| ResNet101 | 79.65 | 95.58 | 74.51 | 75.73 | 74.55 | 95.08 | 73.79 |
| ConvNeXt-T | 83.19 | 96.17 | 79.09 | 79.34 | 79.13 | 95.87 | 78.24 |
| ViT-B | 84.07 | 96.12 | 77.88 | 77.66 | 77.62 | 96.18 | 79.37 |

- Per-class: shown only in a figure, not given in the text.

### Key method references cited
- ICROP3 (Chiang 2021)
- ResNet (He 2016)
- ConvNeXt (Liu 2022)
- ViT (Dosovitskiy 2020)
- t-SNE (van der Maaten 2008) [FT]

### Weaknesses / leakage risks
- The split level isn't stated, with 1.39 images per eye.
- Single hospital, all-Asian cohort; Stages 4–5 are excluded.
- Stage 1 vs Stage 2 labels are the least reliable.
- Byte duplicates under different IDs and cross-folder duplicates [LOCAL].
- Any image-level use of the "official" 8:1:1 counts inherits these problems.

### What RetinAI could adopt
- **A comparability anchor.** The dataset's own baselines (80–88% accuracy) are *below* RetinAI's patient-grouped 0.90. That's worth stating explicitly in the paper.
- **The Stage 1/2 disagreement finding**, as a cited explanation for the Stage 1 recall ceiling. Also report adjacent-class-tolerant metrics: quadratic-weighted κ, "Stage 1 or 2" recall, and the Stage 1→2 confusion share.
- The training recipe itself (SGD 1e-3, batch 64, crop and flips) is weaker than RetinAI's. Nothing to copy there.

---

## 7. HVDROPDB, "HVDROPDB datasets for research in retinopathy of prematurity", Data in Brief 2023

### Citation and access level
- Agrawal R, Walambe R, Kotecha K, Gaikwad A, Deshpande CM, Kulkarni S. Data in Brief 52:109839 (2023), doi:10.1016/j.dib.2023.109839, PMC10749257 [FT].
- **Full text read**, plus the Mendeley Data v3 page (10.17632/xw5xc7xrmp.3, published 17 Jun 2024, CC BY 4.0) [EXT] and RetinAI's local copy [LOCAL].

### Dataset(s)
- Source: PBMA's H. V. Desai Eye Hospital, Pune, India; 2009–2022; ethics approval HVD/IEC/BHR/07/2020 [FT].
- Cameras [FT]:
  - **RetCam** (Clarity MSI): 640×480, .png
  - **Neo** (Forus Health, Bangalore): 2040×2040, .jpeg
  - Both 120° field of view.
- Cohort and acquisition [FT]:
  - 1,100 infants, GA 26–36 weeks, BW 760–3,000 g.
  - 10,570 RetCam and 8,280 Neo images acquired, 2–12 per eye.
  - About 1,900 RetCam and 1,100 Neo images "annotated". This doesn't match the totals and isn't explained.
  - Laser-treated and detached-retina infants were excluded.
- **What's actually released**:

| Subset | Folders | Files | Source |
|---|---|---|---|
| Segmentation | OD, BV, RIDGE; each with RetCam images, RetCam masks, Neo images, Neo masks | 50 per folder → 600 files (300 images + 300 masks) | FT + LOCAL |
| Classification (v3) | `Neo_Normal` 50, `Neo_ROP` 35, `RetCam_Normal` 50, `RetCam_ROP` 50 | 185 files, **binary labels only** | EXT + LOCAL |

- Uniqueness [LOCAL]:
  - **785 files, but 329 unique photographs.**
  - 21 classification files are byte duplicates.
  - 97 images appear in both the classification and segmentation subsets.
  - Every file is named 1.png…50.png, so basenames collide across folders.
- The Data in Brief article defines an ICROP class list (mature, immature by zone, Zone I/II/III × Stage 1/2/3, Stage 4/5, A-ROP; "No-ROP" = mature + immature) but gives **no per-class counts and no classification folder**. Zone I stage 3 has no images [FT].
- Stage-labelled derivatives ("cropped STAGE dataset", "HVDROPDB-PLUS") exist only in the authors' later MethodsX 2025 paper [AB] and are not in the public release.

### Split
- Recommended **70:10:20 (train:val:test), random_state 42** [FT]. This is image-level, since the release has no patient or eye IDs [LOCAL]. Counts: not stated.

### Preprocessing
- Not stated for the baselines [FT].

### Augmentation
- Not stated [FT]. Augmented images in the release: none indicated. The duplicates are byte copies, not augmentations [LOCAL].

### Model
- Baseline: Attention-Gated U-Net (from the authors' MTAP 2022 paper) for segmentation only [FT]. There's no classification baseline.

### Training
- Not stated [FT].

### Evaluation and results
Segmentation metrics [FT]:

| Camera | Dice OD / BV / RIDGE | Accuracy OD / BV / RIDGE | Precision OD / BV / RIDGE | TPR OD / BV / RIDGE |
|---|---|---|---|---|
| **Neo** | **0.92 / 0.66 / 0.72** | 0.99 / 0.91 / 0.98 | 0.94 / 0.78 / 0.75 | 0.90 / 0.57 / 0.70 |
| **RetCam** | **0.52 / 0.52 / 0.53** | 0.86 / 0.86 / 0.98 | 0.61 / 0.60 / 0.71 | 0.50 / 0.51 / 0.53 |

- The near-1.0 pixel accuracies are background-dominated [INF].

### Key method references cited
- Agrawal et al., J Digit Imaging 2021 (zone I detection)
- Agrawal et al., MTAP 2022 (AG U-Net segmentation)
- ICROP3 (Chiang 2021) [FT]

### Weaknesses / leakage risks
- No patient IDs, so the only possible split is image-level.
- Heavy file duplication (785 → 329 unique) and cross-subset overlap.
- Table 1 copy errors in the folder names [FT].
- Camera is perfectly identified by image size (640×480 vs 2040×2040).
- Single hospital. "Poor-quality images were kept" [FT].
- Annotation by an internal team with κ = 0.92 measured *before* annotation [FT].

### What RetinAI could adopt
- **The 100 RIDGE masks** (50 RetCam + 50 Neo; after md5 dedupe) as *auxiliary supervision* for the demarcation line and ridge, the feature that defines Stage 1 and Stage 2. For example, a light segmentation head or attention-alignment loss on the training folds only.
- **Neo (Forus) images as an extra device "style"** for the style-swap augmentation in §9. Even unlabelled at stage level, they widen device coverage.
- **Never as a test set.** It has no IDs, is duplicated, and its labels are binary.

---

## 8. ROP-VL, Sci Data 2026, DOI 10.1038/s41597-026-07901-0

### Citation and access level
- Chen S, Tan T, Zhao X, Huang J, Wu Z, Zhang G, Sun Y. "ROP-VL: An Open Dataset Integrating Fundus Photograph and Clinical Knowledge for Retinopathy of Prematurity." Sci Data, received 8 Oct 2025, accepted 13 Jul 2026, published 13 Aug 2026 [AB].
- Affiliations: Macao Polytechnic University; Southwest Medical University (Luzhou); Shenzhen Eye Hospital [AB]. Funding: Science and Technology Development Fund of Macao [AB/Crossref].
- Article licence CC BY-NC-ND 4.0 (Crossref). **Dataset licence CC BY 4.0** (figshare 10.6084/m9.figshare.30143461, posted 2025-09-28, one file `ROP-VL.zip`, 688,608,891 bytes) [EXT].
- **Abstract only.** The nature.com page served front matter and abstract but no body text; it's an early-release version "subject to further edits". The PDF route needed a cookie-redirect flow I didn't follow. Not yet in Europe PMC.

### Dataset(s)
- **2,020 colour fundus photographs / 1,116 infants / 8 categories**: Normal, A-ROP, Laser-treated, ROP Stages 1–5 [AB].
- Each image is paired with "patient biological metadata and structured textual descriptions" [AB].
- Per-class counts [LOCAL, RetinAI's download, counted 2026-08-05]:

| Class | Images |
|---|---|
| Normal | 623 |
| Laser-treated | 345 |
| Stage 2 | 338 |
| Stage 3 | 241 |
| Stage 1 | 236 |
| A-ROP | 148 |
| Stage 4 | 59 |
| Stage 5 | 30 |

- **The only public source of Stage 4/5** [LOCAL].
- Metadata [LOCAL]: `img_info.xlsx` (patient_id, gender, gestational age, birth weight, plurality, delivery mode) and `disease_description.xlsx` (per-class clinical text).
- Resolution 616–1,720 px [LOCAL].
- Filenames embed exam dates [LOCAL].
- Patient IDs behave like records rather than people (1.70 images per ID) [LOCAL].
- One byte-identical file is labelled both A-ROP and Stage 3 [LOCAL].
- Cameras, collection years and grading protocol: not stated in the accessible text.

### Split
- Not stated [AB].

### Preprocessing / Augmentation / Model / Training
- Not stated [AB].
- Technical validation: "four image-only models and one vision-language model" trained for ROP occurrence and severity. The vision-language model using the structured clinical data "performs better" [AB]. Model names, numbers and recipe: not available.

### Evaluation and results
- Not available beyond that qualitative claim [AB].

### Key method references
- Not available.

### Weaknesses / leakage risks
- Same hospital and authors as Shenzhen, with about 20–22 overlapping photographs [LOCAL]. Patient identity across the two ID spaces can't be checked [LOCAL].
- Stage 4/5 are single-site and small.
- The VLM's advantage may partly come from metadata (GA, BW) that correlates with severity. Whether that's legitimate depends on whether the deployment site supplies it [INF].

### What RetinAI could adopt
- Already used as a training cohort.
- **Clinical metadata (GA, birth weight) as an optional late-fusion input** is the abstract's claimed win. Only worth testing if the external hospital (Multi-View) also provides GA and BW per eye. Otherwise it can't be evaluated externally and adds a shortcut.
- `disease_description.xlsx` could seed text-anchored class prototypes. This is speculative and low priority.

---

## 9. Synthesis: what's worth copying into RetinAI

Ranked by (evidence quality × direct relevance to the weak point, Stage 1 recall at the unseen hospital) ÷ cost.

### 1. Fourier low-frequency amplitude style swap across cohorts, with a feature-consistency loss
- **Source:** SABPI-Net, MICCAI 2025 (open-access full text) [EXT]. It comes from the same Shenzhen group as the JBHI DS-Swin-T and is the lightweight version of DS-Swin-T's "image synthesis + adversarial style invariance".
- **How it works in SABPI-Net:**
  1. Keep a first-in-first-out memory bank of the low-frequency, centred FFT *amplitude* blocks of recent training images.
  2. For each image, swap in the bank entry *least* similar (Euclidean) to its own low-frequency amplitude, keeping the phase. This produces a re-styled copy X*.
  3. Add L_match = 1 − mean cos(z, z*) between the encoder features of X and X*, with **λ = 0.01**, next to the CE loss.
- **Evidence:**
  - Best swap window was **3×3 at 224 px** ("S3"). Larger windows (6, 12) blurred structure and moved the optic disc.
  - At S3, the DAESSM module alone took the baseline from Acc 93.14 / F1 78.88 to **94.24 / 84.70**.
  - Tested on 174,540 images with a patient-level split.
- **For RetinAI:**
  - Fill the bank across Ostrava (RetCam 3, RetCam Envision, Phoenix ICON), Shenzhen, ROP-VL and, optionally, HVDROPDB-Neo, so swaps are cross-device.
  - At 392 px, scale the window to about 5×5.
  - Apply it *after* the circle crop and *before* CLAHE, or test both orders, since CLAHE partly normalises the same low-frequency style.
- **Success criterion:** Stage 1 recall at Multi-View in CV-by-site, not overall accuracy.

### 2. Test-time augmentation with horizontal flip
- **Source:** IConSCEPT Swin-T [AB].
- Nearly free, and the training distribution already includes flips.
- Average softmax over {identity, h-flip} (optionally + v-flip). Check calibration (ECE) as well as accuracy.

### 3. Class-conditional domain-adversarial alignment (the heavier fallback if item 1 isn't enough)
- **Source:** JBHI DS-Swin-T [AB]; references DANN and ADDA.
- Gradient-reversal domain head on the DINOv2 CLS token, with **cohort or device** as the domain label.
- Make it **per class** (one domain head per stage, or conditioned on the label, as in CDAN). The abstract explicitly targets "intra-class" style invariance, and cohort stage-mix differs a lot (for example, Ostrava has almost no Stage 3 infants). An unconditional adversary would remove stage signal.
- Expect fragile training. Keep its weight small and ramp it up.

### 4. Ridge-aware auxiliary supervision for Stage 1 (small experiment, weak evidence)
- Use the HVDROPDB RIDGE masks (50 RetCam + 50 Neo) [FT/LOCAL] as an auxiliary segmentation or attention target on training folds.
- Optionally add a Frangi vesselness channel at σ ≈ 2–4 px (IJASIS [FT], but that paper is unreliable).
- **Rationale:** Stage 1 is defined by a thin, flat demarcation line, and both Zhao [FT] and the IConSCEPT Grad-CAM [AB] point to Stage 1/2 confusion. There is no strong external evidence that this improves staging.

### 5. A protocol-inflation experiment for the paper (cheap, high explanatory value)
- On Shenzhen, train and test RetinAI's model with a **random image-level 8:1:1 split** (Zhao's per-class counts) and keep the **byte duplicates in**, as published.
- Report it next to the patient-grouped 0.90.
- If it reaches about 94–95%, that directly explains ROPGCViT's 94.69 (= 107/113) without any architectural claim.
- Also report Wilson CIs for 113-image test sets (±4–5 points) to show why single-split numbers can't rank models.

### Not worth copying
- **Diffusion-synthesised Stage 1–3 images.** The only paper here that does it (IJASIS) gives no method and isn't credible. If RetinAI ever tries it: train-only, generator trained on train folds only, plus a nearest-neighbour memorisation check.
- **SE/RMLP heads (ROPGCViT); EfficientNet-B4 + Swin hybrids (DeepROP).** No evidence that they transfer to a DINOv2 ViT-S/14 that's already fine-tuned.
- **Grayscale preprocessing (IJASIS).**
- **Any data from Kaggle "ROP-stages_reviewed".** 85 unique images, unknown licence and source.
- **Accuracy or weighted metrics as headlines.** All four ≥94.7% papers use them, and they hide Stage 1. SABPI-Net's 95.3% comes with 61.5% A-ROP sensitivity.

---

## 10. Access log (what was tried)

| Paper | Route | Outcome |
|---|---|---|
| 1 IJASIS | xlescience.org PDF | **FT** (pdftotext) |
| 2 ROPGCViT | IEEE Xplore HTML | JavaScript-only page (empty) |
| 2 ROPGCViT | IEEE `stamp.jsp` PDF | **HTTP 418**, not retried |
| 2 ROPGCViT | Alanya repository item and bitstream | **HTTP 403** |
| 2 ROPGCViT | DOAJ | **HTTP 403** |
| 2 ROPGCViT | OpenAlex, Semantic Scholar, Crossref | Abstract + 40 references |
| 2 ROPGCViT | arXiv | No preprint; author's MaxGlaViT read for context |
| 3 IConSCEPT | OpenAlex, Semantic Scholar | Abstract + 9 references; closed |
| 4 JBHI | PubMed, Europe PMC, OpenAlex | Abstract + 55 references; no PMC, preprint or code |
| 4 JBHI | MICCAI 2025 SABPI-Net (open) | Read as context |
| 5 DeepROP | OpenAlex | Abstract + 15 references; closed |
| 5 DeepROP | Kaggle public API | Dataset metadata + full file list |
| 5 DeepROP | RetinAI local copy | md5, dHash and size checks |
| 6 Shenzhen | PMC11130119 | **FT**; quotes re-verified via Europe PMC fullTextXML after PMC started a browser check |
| 7 HVDROPDB | PMC10749257 | **FT** |
| 7 HVDROPDB | Mendeley v3 page; local folder counts | Read |
| 8 ROP-VL | nature.com | Abstract and front matter only |
| 8 ROP-VL | Crossref, figshare API | Read |
| 8 ROP-VL | RetinAI local counts | Read |
| 8 ROP-VL | Cookie-redirect PDF route | **Not followed** |

## 11. Sources

- IJASIS paper: https://xlescience.org/index.php/IJASIS/article/download/1714/767
- ROPGCViT:
  - https://doi.org/10.1109/ACCESS.2025.3564213
  - https://ieeexplore.ieee.org/document/10975756
  - https://api.crossref.org/works/10.1109/ACCESS.2025.3564213
  - https://api.openalex.org/works/doi:10.1109/ACCESS.2025.3564213
  - https://acikerisim.alanya.edu.tr/items/55d55c6c-1264-4554-b650-df22e689b9d9 (403)
- MaxGlaViT (context): https://arxiv.org/abs/2502.17154
- IConSCEPT Swin:
  - https://doi.org/10.1109/iconscept66142.2025.11436196
  - https://api.openalex.org/works/doi:10.1109/iconscept66142.2025.11436196
- JBHI DS-Swin-T:
  - https://pubmed.ncbi.nlm.nih.gov/41289110/
  - https://doi.org/10.1109/JBHI.2025.3573318
  - https://api.openalex.org/works/doi:10.1109/JBHI.2025.3573318
- SABPI-Net (context): https://papers.miccai.org/miccai-2025/paper/2355_paper.pdf
- DeepROP: https://api.openalex.org/works/doi:10.1109/icisc65841.2025.11187674
- Kaggle dataset:
  - https://www.kaggle.com/datasets/solennollivier/ropstages-reviewed
  - API: https://www.kaggle.com/api/v1/datasets/view/solennollivier/ropstages-reviewed
  - Files: https://www.kaggle.com/api/v1/datasets/list/solennollivier/ropstages-reviewed
- Shenzhen (Zhao 2024):
  - https://pmc.ncbi.nlm.nih.gov/articles/PMC11130119/
  - https://www.ebi.ac.uk/europepmc/webservices/rest/PMC11130119/fullTextXML
- HVDROPDB:
  - https://pmc.ncbi.nlm.nih.gov/articles/PMC10749257/
  - https://data.mendeley.com/datasets/xw5xc7xrmp/3
- Ostrava (for checking paper 1): https://pmc.ncbi.nlm.nih.gov/articles/PMC11266588/
- ROP-VL:
  - https://www.nature.com/articles/s41597-026-07901-0
  - https://api.crossref.org/works/10.1038/s41597-026-07901-0
  - https://api.figshare.com/v2/articles/30143461
- HVDROPDB stage derivative (MethodsX 2025, AB): https://doaj.org/article/87cc80371937493295a1dd0d716f068d
