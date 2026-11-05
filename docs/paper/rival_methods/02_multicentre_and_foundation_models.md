# Multi-centre / foundation-model / MIL rivals: method-level reading (agent 2)

Compiled 2026-10-30. Tags: **FT** means the full text, code or supplement was read. **AB** means only the abstract, search snippets or metadata were read. **DER** means the number was derived by us from public files (ROP-Bench's public manifest), not stated by the authors. "Not stated" means the source does not give it. No numbers are guessed.

## Access summary

| # | Paper | Access | What was read |
|---|---|---|---|
| 1 | VEA-Net (Front Cell Dev Biol 2026) | **FT** | Frontiers HTML and PMC13447418. The Frontiers scrape lost some symbols, which PMC filled in. |
| 2 | SABPI-Net (MICCAI 2025) | **FT** + code | Open MICCAI PDF (`papers.miccai.org/miccai-2025/paper/2355_paper.pdf`), reviews page, and the GitHub repo `SB-Chen/SABPI-Net` (all .py files) |
| 3 | ROP-Bench (Luo, Guo, Chen, Liang, Lin) | **FT** + code | Repo `foye80/rop-bench`: `paper/main.tex` (full manuscript), README, `notes/results.md`, every script, and the public manifest/splits (used for DER counts). Zenodo record (v1.0.2) also read. |
| 4 | DeepAdapter (Zhao, Li, … Lin; BJO 2026) | **FT** (main text partly) + supplement | The BJO page renders Methods and Results, but Tables 1–2 were not readable. The open (CC BY-NC) supplement PDF has Appendices 1–4 and Tables S1–S4. Hyperparameters are not reported anywhere. |
| 5 | Mahboob, Ahsan, Wilms (SPIE MI 2025) | **AB only** | SPIE pages render empty and lux.spie.org has a TLS certificate error. Semantic Scholar says "CLOSED", no arXiv copy exists, and Crossref has metadata and the reference list only. The abstract content comes from search snippets. |
| 6 | Lan, Tsai, Wu (TVST 2026) | **FT of the preprint** (Research Square v1, CC BY 4.0, 3 Sep 2025) | The TVST version of record returned **403** (Cloudflare check). Per the rules this was not worked around, and the VoR is not in PMC. Crossref shows the VoR has 30 references against the preprint's 26: it adds MixStyle, Matsuura domain generalisation and Tseng 2026. The VoR may therefore add a domain-generalisation discussion or experiment that we could not see. |

---

## 0. Fine-tuning recipes side by side

| | ROP-Bench (fine-tune check) **FT-code** | SABPI-Net **FT+code** | DeepAdapter **FT** | VEA-Net **FT** | Lan et al. **FT-preprint** | RetinAI (for reference) |
|---|---|---|---|---|---|---|
| Backbone / init | timm `vit_base_patch14_dinov2.lvd142m` (ViT-B/14, img_size=224). Comparators: RETFound-CFP ViT-L/16 and random-init ViT-B/14 | Swin-T-like (embed 96, depths 2-2-6-2, heads 3-6-12-24, window 7) + agent-attention "PIAM". The released code loads **no pretrained checkpoint** (the `pre_trained` flag is unused); the paper does not say | ResNet-50 + "attention mechanism between last two residual blocks". MoCo-v2 SSL from **random init**. The supervised control is ImageNet-init | ResNet-50 + EfficientNet-B4, torchvision ImageNet weights | CBAM-ResNet18. Pretraining not stated | DINOv2 ViT-S/14-reg |
| Input | 224 (train: RandomResizedCrop; eval: Resize+CenterCrop) | 224 (train: shorter side→224 then random 224 crop; **test: squash to 224×224**) | "resize-and-pad" to keep the periphery. Size not stated | 224 | 224 | 392 |
| Optimiser | AdamW, wd 0.05 | SGD, Nesterov, momentum 0.9, wd 1e-4 | not stated | AdamW, wd 1e-4 | head only: Adam, lr 1e-3, L2 1e-3 | AdamW |
| LR | backbone **1e-5**, head **1e-3** | **1e-3** base | not stated | 1e-4 | 1e-3 | OneCycle |
| Schedule | CosineAnnealingLR(T_max=epochs), stepped per epoch, no warmup | 1-epoch linear warmup (from 1e-3×) then per-step cosine to 1e-6× | not stated | cosine; T_max/min not stated | not stated | OneCycle |
| Layer decay | **none** | **none** | not stated | none stated | — | 0.8 |
| Drop path | none (timm default 0) | 0.1 | — | — | dropout 0.5 in head | 0.1 |
| Epochs / batch | ≤15 / 32 (eval 64) | 100 / not stated in paper (CLI arg) | not stated. UDA reached peak at epoch 27 on 954 target images | ≤30, early stop patience 5 / 16 | not stated | 20 / 16 |
| Augmentation | RandomResizedCrop(scale 0.6–1.0) + HFlip only | resize+random crop, HFlip 0.5, VFlip 0.5, plus **Fourier low-freq style swap** (second view) | brightness/contrast, gamma, Gaussian noise, rotation, H/V flip (no parameters) | RandomResizedCrop, H/V flip, colour jitter brightness/contrast 0.2 | shifts ≤20%, brightness 0.5–1.5, shear, zoom; **no flips/rotation** (to keep view identity) | very rich |
| Mixup / CutMix | **no** | **no** | not stated | no | no | no |
| Label smoothing | no | no | not stated | no | no | 0.05 |
| Imbalance | class-weighted CE, pos weight = neg/pos | plain CE (code default `train_rule=None`) | not stated | focal γ=2 (plus, treatment) + class-weighted CE for stage | none stated | CB-CE β=0.999 |
| Checkpoint selection | best val AUC (Czech val: 3 ROP-positive infants, DER) | best val accuracy | best val accuracy | best val per fold | not stated | — |

**Bottom line:** none of these rivals used mixup, CutMix or layer-wise LR decay. RetinAI's recipe is already richer than every one of them. What they offer is ideas around the recipe: style or frequency augmentation, an extra data source, class-conditional UDA and evaluation design. They do not offer a better optimiser setup.

---

## 1. VEA-Net

### Citation and access
- Chen Y, Wan J, Zhang T, Wan Z, Hong J, Jin Y. "VEA-net: vascular enhancement attention with dual-backbone multi-task learning for comprehensive ROP management across multi-center datasets." *Front Cell Dev Biol* 14 (2026), published 24 Jul 2026. DOI 10.3389/fcell.2026.1865503. PMC13447418. **FT**.
- Code: none linked. Data statement: "included in the article/supplementary material" (**FT**).

### Datasets (FT unless noted)
| Dataset | Source | Camera | Images | Subjects |
|---|---|---|---|---|
| FARFUM-RoP | Farabi Eye Hospital / Ferdowsi Univ. Mashhad, Iran | RetCam III | 1,533 | 68 |
| Ostrava | University Hospital Ostrava, Czech Rep. | three systems (not named in paper) | 6,004 | 188 |
| A-Fundus (Shenzhen) | Shenzhen Eye Hospital, China | not stated | 1,099 (1,079 used) | 483 (481 used) |

- Total 8,636 images and 737 subjects. The images actually used sum to 8,616, an internal inconsistency.
- A-Fundus: 342 of 483 infants (70.8%) have more than one image. Two subjects were dropped for incomplete metadata.
- Ostrava stage mapping: 4A/4B → Stage 4; **AP-ROP → Stage 4**; "status post ROP" → Stage 5, which is the laser-scar class. Non-ROP diagnoses are said to be excluded from stage supervision.
- FARFUM stage and treatment labels were derived by expert voting, with unresolved ties excluded. Expert stage fields cover Stages 1–3 only.
- Class counts (Table 1, FT):

| Task / class | FARFUM | Ostrava | A-Fundus | Total |
|---|---|---|---|---|
| Plus: Normal / Pre-plus / Plus | 782 / 479 / 272 | 5,375 / – / 629 | – | 6,157 / 479 / 901 |
| Stage 0 | 952 | 4,086 | 236 | 5,274 |
| Stage 1 | **373** | 252 | 94 | 719 |
| Stage 2 | 132 | 458 | 147 | 737 |
| Stage 3 | 76 | 125 | 261 | 462 |
| Stage 4 | – | 470 | – | 470 |
| Laser scar ("Stage 5") | – | 379 | 341 | 720 |
| Treatment: none / revisit / treat | 190 / 593 / 453 | – | – | — |

- **DER: what these Ostrava counts actually contain.** We checked them against Ostrava's diagnosis codes (DG) in ROP-Bench's public manifest, using RetinAI's own code dictionary in `scripts/rop_recover_staging.py`:
  - **Stage 0 = 4,086** equals DG0 physiological (2,980) + DG1 "ROP stage 0" (45) + **DG11 haemorrhage (1,061 images, 35 infants)**. The arithmetic matches exactly, and the 234 excluded images are exactly DG10, DG12 and DG13. Despite the claim that non-ROP diagnoses were excluded, haemorrhage images were apparently kept as "Stage 0".
  - **Stage 4 = 470** is exactly DG8 = AP-ROP, and comes from **a single infant**. Ostrava has no DG5–7 images. In a subject-grouped 10-fold split, that infant is in the test set of only one fold, and in that fold the training set has zero Stage-4 images.
  - **Plus = 629** Ostrava images come from **3 infants** (PF2). FARFUM has 15 plus-positive infants.
  - Ostrava Stage 1 (DG2) = 252 images from **12 infants**. Shenzhen Stage 1 = 94 images from 71 infants.

### Split (FT)
- 10-fold **Group K-Fold, subject-level**: patient IDs for FARFUM and Ostrava, infant IDs from metadata for A-Fundus. Within each fold, training data is split again into train and validation; the ratio is not stated.
- No external or unseen-hospital test. All three centres are pooled into the CV. The authors say prospective validation is needed.

### Preprocessing (FT)
Resize to 224×224 and ImageNet mean/std normalisation. Nothing else: no crop, mask or CLAHE is stated.

### Augmentation (FT)
- Train: RandomResizedCrop (scale not stated) → 224, random H and V flips, ColorJitter brightness 0.2 and contrast 0.2. No augmentation at validation or test.
- Online vs offline: not stated (torchvision-style ops imply online). No synthetic images.

### Model (FT)
- **Dual backbone:** ResNet-50 (ImageNet torchvision, FC removed, 2048-d) and EfficientNet-B4 (ImageNet torchvision, 1792-d). Both are fine-tuned; no freezing is stated.
- **VEA module**, placed after ResNet layer3 and layer4 and after the final EfficientNet stage:
  - Multi-scale branch: channel projection, then three parallel paths (1×1 conv; 3×3 depthwise; 5×5 depthwise with padding 2), each with BN and ReLU, concatenated and conv-fused.
  - Directional branch: 3×3 depthwise conv with BN and activation.
  - The two branches are added, then passed through SE (r=16) and CBAM spatial attention (avg+max pooling, 7×7 conv, sigmoid), with a residual connection.
- **Fusion:** each backbone is projected to D=2048, an MLP produces softmax weights over the two backbones, and their outputs are summed with those weights.
- **Heads:** three two-layer FC heads (→512, dropout rate not stated): Plus (3 classes), Stage (6 classes, 0–5), Treatment (3 classes). Xavier initialisation.
- **Domain adaptation:**
  - DANN: domain classifier FC 256→128→3 with dropout 0.5, behind a gradient reversal layer. Domains are FARFUM=0, Ostrava=1, A-Fundus=2. GRL λ_da is not stated.
  - MMD: multi-kernel RBF (K=5, base bandwidth h=2.0). Which pairs of domains are aligned per batch is ambiguous.

### Training (FT)
- **Loss:** a weighted sum, with weights λ1–λ7 that are not stated numerically.
  - Plus and Treatment use focal loss with γ=2.
  - Stage uses class-weighted CE with weights [1.0, 1.5, 1.2, 2.0, 3.0, 2.5].
  - Task-correlation losses: a Plus↔Stage≥3 term (γ_PS=0.5) and a Stage-bin↔Treatment term (γ_ST=0.5).
  - Auxiliary loss linking abnormal Plus to abnormal Stage (γ_aux=0.3).
  - DANN CE and MMD terms.
  - "Dynamic weighting" is mentioned but the mechanism is not described.
- AdamW, lr 1e-4, wd 1e-4, cosine annealing (T_max not stated). Batch 16. Up to 30 epochs with early stopping at patience 5; the mean best epoch is about 28.
- PyTorch 1.12 on 2× RTX 4090 D (48 GB).

### Evaluation (FT)
- Image-level accuracy (labels of −1 are ignored) and AUROC (binary, or one-vs-rest macro for multi-class), reported as mean ± SD over 10 folds.
- No CIs. No statistical tests; comparisons are explicitly "descriptive".

### Results (FT)
- **Pooled** test results: Plus ACC 85.44±15.78, AUROC 91.76±4.16. Stage ACC 74.81±10.17, AUROC 88.26±4.61. Treatment ACC 56.80±23.48, AUROC 66.81±15.87.
- **Per centre** (ACC / AUROC):
  - FARFUM: Plus 59.83 / 79.31; Stage 50.19 / 80.63; Treatment 76.45 / 86.93.
  - Ostrava: Plus 97.17 / 98.32; Stage 90.42 / 89.07.
  - A-Fundus: Stage 66.15 / 86.52.
  - The per-centre FARFUM accuracies do not reconcile with the pooled numbers, and the paper does not explain this.
  - Ostrava's majority-class prior for Plus is 89.5%.
- **Ablation** (Stage AUROC):
  - Full model 88.26. Without VEA 87.12. Without dual backbone 87.96. Without task correlation 87.94. Without DA 87.87.
  - **DANN-only 81.76**, MMD-only 88.14, ResNet-50 baseline 85.20.
  - DANN alone *hurt* every task, for example Stage ACC 67.87 against 72.28 for the baseline.
  - Without DA, Treatment AUROC was *higher* (68.09 vs 66.81).
- **Comparators** (Stage AUROC): SPIE/Graziani 75.07, VGG-ST 72.30, Mohiy 2026 74.06, Wei 2023 73.35, **Swin-Tiny 87.43** (run by the authors; AUROCs described as "estimated"), VEA-Net 88.26. Every gap over Swin-T is well inside one SD.

### Key method references (FT)
Graziani 2019 (SPIE 10950); Luo 2024 VGG-ST (BIBM); Mohiy 2026 (Discov Artif Intell 6:305); Wei 2023 supervised DA for wide-field fundus (BIBM); Liu 2021 Swin; Guan & Liu 2021 DA survey; Akbari 2024 FARFUM-RoP (Sci Data 11:1176); Timkovič 2024 Ostrava (Sci Data 11:814); Zhao 2024 A-Fundus (Sci Data 11:543); Chiang 2021 ICROP3. The reference list contains no entries for DANN, MMD, CBAM, SE or focal-loss primary sources.

### Weaknesses and leakage risks
1. **No held-out centre.** "Multi-center" means pooled CV, so the DANN and MMD terms are never tested on an unseen domain.
2. **Image-level metrics** from a 10-fold pool dominated by Ostrava (≈70% of images). The pooled Plus accuracy (85%) is close to the majority prior. FARFUM Plus accuracy is 59.8%.
3. **Stage 4 is one AP-ROP infant (DER).** It cannot be learned and tested in the same fold. AP-ROP is also mis-mapped into the stage ladder, against ICROP3.
4. **Haemorrhage images (DG11, 1,061 images) counted as Stage 0 (DER)**, contradicting the text.
5. Ostrava Plus has **3 infants (DER)**, so Ostrava Plus AUROC can only be computed in at most 3 of the 10 folds.
6. FARFUM is the only source of Treatment labels and Pre-plus, so the treatment task is single-site.
7. Hyperparameters are missing (λ1–λ7, λ_da). Swin-T AUROCs are "estimated". There are no significance tests, and almost all differences fall inside SD.

### What RetinAI could adopt
- **FARFUM-RoP stage labels as a Stage-1 data source.** VEA-Net derives 373 Stage-1, 132 Stage-2 and 76 Stage-3 images from FARFUM's expert-vote fields (68 infants, RetCam III, Iran). Stage 1 is RetinAI's weak class, and its current Stage-1 training data comes from few infants (DER: Ostrava 12, Shenzhen 71, plus ROP-VL). FARFUM adds a new country and new graders. Caveats: labels are sparse multi-grader (1–3 of 5 graders per image; ROP-Bench) and ties are excluded. Fold FARFUM in with patient grouping, and also use it as a leave-one-source-out test.
- **Their DANN-only result is negative evidence**, consistent with DeepAdapter's Table S4 below. Do not spend effort on DANN.
- **Report a majority-class prior next to each per-centre metric.** Their pooled-vs-per-centre gap is exactly the trap a measurement paper can point out.
- Not worth adopting: the VEA attention module (it is CNN-specific, and its gain is inside SD) and the dual backbone.

---

## 2. SABPI-Net

### Citation and access
- Chen S, Zhao X, Fu H, Tan T, Huang J, Xiong X, Wu Z, Dashtbozorg B, Lei B, Zhang G, Sun Y. "SABPI-Net: A Novel Structure-Aware Network for Accurate and Domain-Invariant Retinopathy of Prematurity Diagnosis." MICCAI 2025, LNCS 15969, pp. 456–466. DOI 10.1007/978-3-032-05127-1_44. Affiliations: Macao Polytechnic Univ., Shenzhen Eye Hospital, A*STAR, NKI, Shenzhen Univ.
- **FT** from the open MICCAI PDF and the reviews page. **Code read**: https://github.com/SB-Chen/SABPI-Net (no licence, 0 stars, pushed Oct 2025).
- Reviews: R1 4 (weak accept), R2 5, R3 5. No rebuttal. All reviewers noted that there is no external hospital test.

### Dataset (FT)
- In-house dataset: **Shenzhen Eye Hospital**, 13,314 patients undergoing ROP screening, **174,540** colour fundus photos.
- Classes: Normal 112,770; laser-treated ROP 29,442; "any stage of ROP" 19,513 (Stages 1–5 *pooled*); **A-ROP 2,363**; "other" infant retinal disease 10,452.
- Camera: not stated. Images per infant: about 13 on average, which is our arithmetic (174,540 / 13,314), not stated.
- This is the same group and hospital as the public A-Fundus/Shenzhen set (Zhao et al. 2024 Sci Data), which RetinAI already trains on.
- Also used: public **OIA-ODIR** (an *adult* multi-disease dataset, Li et al. 2021). Whether they retrained on it or transferred is not stated.

### Split (FT)
Train/val/test ≈ **7:2:1 at patient level**. No external ROP test. The released code loops over `train_k` / `valid_k` Excel sheets, so it has k-fold support, but how the paper's split maps onto that is not stated.

### Preprocessing (FT + code)
- **Fundus mask required**: the code reads a `mask/` directory. Outside the mask, pixels are filled with the per-channel **median + α**, where α=0.2 (Eq. 1).
- Train transform: resize the shorter side to 224 (aspect preserved), pad if needed, random 224×224 crop.
- **Test transform: `Resize((224,224))`**, which squashes the image without preserving aspect ratio. This is a train/test geometry mismatch (code).
- ImageNet mean/std normalisation (code `DatasetStats`).

### Augmentation (FT + code)
- Paper: "random cropping, scaling, horizontal flipping and vertical flipping are sequentially applied". Code: resize+random crop, HFlip p=0.5, VFlip p=0.5. All online.
- **Memory-bank low-frequency replacement (DAESSM):**
  - Paper Eq. 5 describes swapping the *amplitude* low-frequency block with the bank entry of *lowest similarity* (Euclidean). The bank is FIFO over preceding training images, of size S.
  - Code (`LowFrequencyBufferRGB`):
    - Buffer size **32**, `low_freq_size=(3,3)`.
    - Works per RGB channel with `fft2` + `fftshift` on the **full-resolution original image** (before resize).
    - Picks the buffer entry with the **maximum** summed Frobenius distance across R, G and B.
    - **Overwrites the complex coefficients** (amplitude *and* phase) in the centre window, then takes `np.abs(ifft2(...))`.
    - Because of the slicing `c-h//2 : c+h//2`, a "3×3" request actually swaps a **2×2** block that includes DC. In practice this is a global colour/brightness and lowest-order illumination transfer from a maximally different training image.
  - The style-swapped image then goes through the same train transform and HFC.
  - Ablation of window size (Table 2): S3 Acc 94.24 > S6 93.60 > S12 93.45 > baseline 93.14. Larger windows blur detail and can move the optic disc (Fig. 4).
- No synthetic or generated images.

### Model (FT + code)
- **Trunk:** a hierarchical Swin-style transformer ("standard hierarchical transformer blocks", Fig. 1). Code `SBAINet`: patch 4, embed 96, depths (2,2,6,2), heads (3,6,12,24), window 7, mlp 4, drop_path_rate 0.1, avg-pool then linear head. This is a **Swin-T configuration**.
  - Pretrained weights: **not stated** in the paper. The released `getModel` ignores `pre_trained` and loads no checkpoint, so as released it trains from scratch.
- **HFM (high-frequency mapping) branch:**
  - Eq. 2: F(x_p) = (clamp(x_p − x_p * g(r,σ), −1, 1) + 1)·M − 1.
  - Code (`HFCFilter`): three Gaussian filters (width 9/19/29, σ 5/9/13) are built, but **only the last (width 29, σ 13) is kept** (`[:, 6:]`). The code multiplies by **ratio=4** before clamping, a factor missing from Eq. 2. The filter is applied to the *normalised* tensor.
  - This is essentially **Ben Graham's preprocessing** (4·(x − G_σ(x))), masked to the fundus.
- **Weight sharing:** in the code the HFM branch **shares the patch embedding and all Swin stages with the trunk**. It is the same network run on the HFC image, not a second backbone.
- **PIAM (proxy interaction attention):**
  - Agent-attention style. Proxy tokens P = AdaptiveAvgPool(Q_trunk). First softmax(P·K_hf^T + B1)·V_hf aggregates high-frequency features. Then softmax(Q_trunk·P^T + B2) spreads them back. A 3×3 depthwise conv on V_trunk is added (Eq. 4). Proxy positional biases are learned.
  - Code: `agent_num=[9,16,49,49]` with **`attn_type='1100'`, meaning PIAM only after stages 1 and 2**. The paper's Fig. 3b says applying it at all four blocks is best: a paper/code discrepancy.
- **DAESSM loss:** cosine similarity between pooled embeddings of the original x and its style-swapped x*, L_match = 1 − mean(cos).
  - Paper λ = **0.01**. Code: `loss = loss_ce + 0.1 * loss_cosine_sim  # 0.01`. **λ = 0.1 in the released code.**
  - Note that x* goes through the full network with its own HFC, so four forward streams are computed per training step.

### Training (FT + code)
- CE + λ·L_match. No class weighting (code default `train_rule='None'`, `loss_type='crossEntropyLoss'`). The code also contains reweighting, focal and class-balanced-focal options and a vestigial class-balanced "tail fine-tuning" sampler (cRT-style head retraining, `SBAINet_tfs`), none of which the paper mentions.
- **SGD, momentum 0.9 (Nesterov in code), wd 1e-4, initial lr 1e-3, cosine schedule, 100 epochs.** Code default scheduler: 1 warmup epoch from 1e-3×, then per-step cosine down to 1e-6×.
- Batch size: not stated. NVIDIA 4090 GPUs.
- Checkpoint selection: best validation accuracy (code).

### Evaluation (FT + code)
- Image-level Accuracy, AUC, Precision, Sensitivity, F1, Specificity and Cohen's κ. The code computes macro-averaged P/R/F1, one-vs-rest macro AUC and unweighted κ.
- No CIs and no statistical tests.

### Results (FT, Table 1, in-house; then OIA-ODIR Acc/AUC/Sen/Spec)

| Method | Acc | AUC | Pre | Sen | F1 | Spec | κ | ODIR Acc | ODIR AUC |
|---|---|---|---|---|---|---|---|---|---|
| ResNet50 | 90.96 | 96.85 | 67.92 | 64.21 | 65.48 | 96.50 | 83.58 | 58.52 | 81.89 |
| ViT-B | 91.19 | 97.34 | 85.80 | 68.09 | 71.48 | 96.61 | 84.05 | 58.76 | 85.90 |
| Conformer-S | 91.98 | 97.42 | 78.50 | 77.41 | 77.84 | 97.50 | 86.01 | 58.10 | 84.06 |
| **RETFound** | 92.30 | 97.74 | 81.03 | 70.57 | 73.25 | 97.23 | 86.23 | 61.40 | 86.03 |
| ConvNeXt-S | 92.81 | 98.05 | 84.24 | 70.06 | 72.31 | 97.42 | 87.16 | 55.28 | 81.70 |
| FIT-Net | 92.59 | 97.57 | 86.03 | 72.50 | 76.89 | 97.05 | 86.60 | 60.14 | 84.76 |
| Swin-S | 93.07 | 98.15 | 86.51 | 76.30 | 80.17 | 97.41 | 87.59 | 60.62 | 85.21 |
| **SABPI-Net** | **95.32** | **99.23** | 87.22 | 85.03 | 86.03 | 98.54 | 91.82 | 63.57 | 88.71 |

- Per class (Fig. 2 text): A-ROP sensitivity is 61.54% for SABPI-Net, against a best comparator of 41.99% and **ConvNeXt-S 10.26%**. "Other" sensitivity is 74.52% (comparators up to 68.65%).
- Ablation (Acc / AUC / F1 / Spec):
  - Baseline: 93.14 / 98.35 / 78.88 / 97.32.
  - + DAESSM: 94.24 / 99.11 / 84.70 / 97.78.
  - + HFM+PIAM: 94.37 / 98.84 / 85.15 / 98.01.
  - All three: 95.32 / 99.23 / 86.03 / 98.54.
- In-distribution, DAESSM alone gives +1.1 points Acc and +5.8 points macro-F1.

### Key method references (FT)
Retinex (Land 1977); GFE-Net frequency self-supervision (Li 2023 MedIA); frequency-mixed single-source DG (Li 2023 MICCAI); **FDA (Yang & Soatto 2020)**; **Fourier-based DG / amplitude mix (Xu 2021 CVPR)**; VPTTA low-frequency prompts (Chen 2024 CVPR); Flatten/focused linear attention (Han 2023 ICCV), the source of the DWC diversity term; Swin (Liu 2021); RETFound (Zhou 2023); Peng 2021 ordinal staging (TMI); Xie 2023 AP-ROP distillation (MedIA).

### Weaknesses and leakage risks
1. **Single hospital. The "domain-invariant" claim is never tested on another ROP site.** OIA-ODIR is an adult dataset and does not test ROP domain shift.
2. "Any stage of ROP" pools Stages 1–5, so the model **does not stage**.
3. Image-level metrics, no CIs. About 13 images per patient means the effective sample size is much smaller than 174k.
4. Paper and code disagree on λ (0.01 vs 0.1), PIAM placement (all blocks vs stages 1–2), the FFT swap (amplitude vs complex, 3×3 vs 2×2) and the HFC gain (missing ×4). Test-time squash versus train-time aspect-preserving crop is a geometry mismatch.
5. Pretraining is unspecified, and the released code trains from scratch. The comparators' training recipes are not described.
6. RETFound is beaten by Swin-S and ConvNeXt-S here too, consistent with ROP-Bench and Mahboob.

### What RetinAI could adopt
1. **Low-frequency Fourier style swap with an embedding-consistency loss (DAESSM), as a source-only domain-generalisation augmentation.**
   - Implementation:
     - For each training image, pick a donor from a **different source dataset or camera**. RetinAI has dataset and device tags (Ostrava has three device codes, plus ROP-VL and Shenzhen), which is better than SABPI's max-distance FIFO.
     - Swap a tiny centred FFT block (2×2 to 3×3 including DC, on the resized 392 image; amplitude-only is the cleaner FDA variant).
     - Add λ·(1 − cos(z, z*)) on the CLS embedding, with λ in {0.01, 0.1}.
   - Why: camera colour and illumination is the main thing that changes at a new hospital, and colour jitter samples unrealistic styles while this samples real ones.
   - Unlike mean-teacher DA, which already failed for RetinAI, this needs no target data.
   - Evidence is weak: an in-distribution gain only. Treat it as an experiment, judged on Stage-1 recall at Multi-View ROP and on leave-one-source-out.
2. **High-pass "detail" view with shared weights.**
   - Stage-1 demarcation lines are thin, low-contrast, high-frequency structures. A masked high-pass image (4·(x − G_σ(x)), clamp, median fill outside the circle) removes low-frequency camera style.
   - Cheapest version: (a) a random-choice augmentation between RGB/CLAHE and the HFC view, plus (b) a test-time average of the two views' probabilities.
   - **Scale σ with resolution.** SABPI used σ=13 at 224 px, which is σ≈23 at 392 px. This is the same resolution-dependence defect RetinAI already hit with Ben Graham preprocessing in DR.
3. **A cRT-style class-balanced head re-fit** (present in their code, not their paper) is a cheap Stage-1 recall lever: freeze the backbone and retrain the head with class-uniform sampling for a few epochs. It is untested by them.

---

## 3. ROP-Bench (Luo, Guo, Chen, Liang, Lin)

### Citation and access
- Luo J, Guo D, Chen X, Liang H, Lin F. "Dataset source and label reference alter performance estimates in public retinopathy of prematurity AI benchmarks." Manuscript formatted for *International Ophthalmology*. Not yet published; no journal DOI.
- Affiliations: First Affiliated Hospital of Xiamen Univ., Women and Children's Hospital Xiamen, Rice Univ.
- Code: https://github.com/foye80/rop-bench (MIT). Zenodo "ROP-Bench: a public retinopathy of prematurity AI benchmark under center and annotator shift", v1.0.2, 17 Jun 2026. The record DOI resolves to 10.5281/zenodo.20724915; the paper cites 10.5281/zenodo.20721027. Author: Jin Luo.
- **FT**: `paper/main.tex`, `notes/results.md`, all scripts, and the public manifest and split CSVs.
- The manuscript discloses use of AI tools for language editing.

### Datasets (FT; DER where marked)
| Dataset | Country | Device(s) | Images | Infants | Label axis used |
|---|---|---|---|---|---|
| FARFUM-RoP | Iran | RetCam | 1,533 | 68 | Plus (Normal 782 / Pre-plus 479 / Plus 272); 5 graders, sparse (1–3 per image) |
| Shenzhen (A-Fundus) | China | "3 mixed" (per-image device not tagged) | 1,099 | 483 | Normal 236 / S1 94 / S2 165 / S3 261 / laser 343 (DER infants: 130 / 71 / 112 / 162 / 163) |
| HVDROPDB | India | RetCam, Neo | 185 | no IDs | binary. DER: Neo 50 N / 35 ROP; RetCam 50 N / 50 ROP |
| Ostrava ("Czech") | Czech Rep. | Clarity RetCam3 2,578 / Phoenix ICON 2,452 / Natus Envision 974 images (DER; 81 / 61 / 46 infants) | 6,004 | 188 | DG codes and PF |

**Harmonisation rules (FT from code):**
- **ROP presence:**
  - Czech: DG0/1 → 0; **DG2–9 → 1**; DG10–13 → excluded.
  - Shenzhen: Normal → 0; Stage 1–3 → 1; **laser scars excluded** ("post-treatment").
  - HVDROPDB: folder label.
  - FARFUM: not used for presence.
- **Plus:** FARFUM consensus label 3 → 1, labels 1 and 2 → 0. Czech PF2 → 1, PF0 → 0.
- **DER:** the Czech presence positives include **DG9 "status post ROP (treated or regressed)", 379 images from 13 infants**, and **DG8 AP-ROP, 470 images from 1 infant**. Post-treatment eyes are therefore *positive* in Czech but *excluded* in Shenzhen. The harmonisation is inconsistent across sites.

**Cohorts (FT):**
- Presence: 5,650 images (Czech 4,709 with 1,684 positive; Shenzhen 756 with 520 positive; HVDROPDB 185 with 85 positive). About 305 ROP-positive infants, of which **only 20 are Czech** (DER and notes).
- Plus: 7,537 images, 901 plus, but **18 plus-positive infants** (Czech 3, FARFUM 15).

### Split (FT)
- Patient-level 70/15/15 within each dataset, seed 42. Stratified by a patient-level **any-positive** label rather than majority vote, so that rare positives are not hidden.
- Patient counts: Czech 131/27/30, FARFUM 47/9/12, Shenzhen 337/71/75.
- **DER:** the Czech presence test set has **24 infants, 5 of them ROP-positive** (471 positive images). Czech val has 20 infants, **3 positive**.
- HVDROPDB is test-only (no IDs).
- Cross-site: train on **all** source images, test on **all** target images. No target data is used for selection.
- In-device check (`cross_device.py`): train on device A, test on device B within Czech; the in-device diagonal uses a 70/30 patient split. Reported as underpowered (6–8 positive infants per device).

### Preprocessing (FT)
- RGB conversion, then the timm backbone-specific eval transform (resize, centre crop, normalise).
- Primary cross-site DINOv2 analysis at the **native 518 px**. Backbone comparison at a matched 224.
- No CLAHE, masking or circle crop.

### Augmentation (FT)
- Linear probes: none (frozen features cached per image path).
- Fine-tune: `RandomResizedCrop(224, scale=(0.6,1.0))` and `RandomHorizontalFlip()` only, online. Eval: Resize(224) + CenterCrop(224).
- No synthetic images.

### Model (FT)
- Frozen backbones:
  - DINOv2 ViT-B/14 (`vit_base_patch14_dinov2.lvd142m`, LVD-142M SSL).
  - CLIP ViT-B/16 (`vit_base_patch16_clip_224.laion2b`).
  - Supervised ViT-B/16 (`vit_base_patch16_224.augreg2_in21k_ft_in1k`).
  - **RETFound-CFP ViT-L/16**, loaded into timm `vit_large_patch16_224` with `global_pool="token"`: the CLS token after the encoder norm, with MAE decoder keys ignored.
- Probe: sklearn LogisticRegression, L2 at the default C, `class_weight="balanced"`, max_iter 2000.
- Fine-tune: full fine-tuning of DINOv2-init, RETFound-init and random-init ViT-B/14, all at 224.

### Training of the fine-tune check (FT-code)
- Class-weighted CE with the positive weight set to the negative/positive ratio.
- **AdamW, backbone lr 1e-5, head lr 1e-3, weight decay 0.05, CosineAnnealingLR(T_max=epochs)** stepped per epoch, no warmup.
- **≤15 epochs, batch 32** (eval batch 64), `drop_last=True`. No layer decay, drop path, mixup, CutMix or label smoothing.
- Best Czech-val-AUC checkpoint. Single seed. Label fractions 10/30/100% (patient-stratified subsampling).
- Hardware not stated (CUDA).

### Evaluation (FT)
- Image-level AUC (primary) and AP.
- **Paired cluster bootstrap**: 2,000 resamples, seed 0, resampling test infants. HVDROPDB is resampled per image. Percentile 95% CIs for each AUC and for the paired ΔAUC; a difference is "significant" if the ΔAUC CI excludes 0.
- Annotator analysis:
  - One DINOv2 probe trained on consensus plus labels, scored against each FARFUM grader's own labels; grade 0 counts as abstention.
  - Cohen κ, human–human (overlapping labels only) and model–grader. For the model, the threshold is set so that predicted prevalence matches consensus prevalence.

### Results (FT)
- **Cross-site, DINOv2@518 probe, presence:**

| train \ test | Czech | Shenzhen | HVDROPDB |
|---|---|---|---|
| Czech | (0.926 [.826–.981]) | 0.822 [.789–.854] | 0.739 [.664–.810] |
| Shenzhen | 0.866 [.755–.921] | (0.999 [.996–1.000]) | 0.953 [.925–.978] |
| HVDROPDB (notes only) | 0.830 | 0.977 | — |

- Plus task (notes): about 0.93 in-domain, 0.83–0.88 cross-country, AP 0.57–0.61.
- **Backbone spectrum** at 224 (frozen; DINOv2 / CLIP / ImageNet / RETFound):

| Setting | DINOv2 | CLIP | ImageNet | RETFound |
|---|---|---|---|---|
| Czech in-domain | 0.893 | 0.845 | 0.866 | 0.854 |
| Shenzhen in-domain | 0.991 | 0.992 | 0.988 | 0.947 |
| Czech→Shenzhen | 0.851 | 0.827 | 0.845 | 0.816 |
| Shenzhen→Czech | 0.786 | 0.803 | 0.828 | 0.707 |
| Czech→HVD | 0.861 | 0.845 | 0.760 | 0.760 |
| Shenzhen→HVD | 0.948 | 0.902 | 0.920 | 0.760 |
| Transfer mean (paper, 4 rows) | **0.862** | 0.844 | 0.838 | **0.761** |
| Cross-site mean (notes, 8 rows incl. HVD→X) | 0.873 | 0.856 | 0.860 | 0.729 |

- **Paired ΔAUC, DINOv2 − RETFound:**
  - Significant: Shenzhen in-domain +0.045 [.01, .09]; Czech→HVD +0.101 [.03, .18]; Shenzhen→HVD +0.188 [.13, .25]. The notes add HVD→Czech +0.159 and HVD→Shenzhen +0.295.
  - Not significant: every row with Czech as the test set (5 positive infants).
- **Full fine-tune, 100% Czech data** (Czech-in / Shenzhen / HVD):
  - DINOv2 0.903 / 0.926 / 0.856.
  - RETFound 0.773 / 0.837 / 0.783.
  - Random init 0.516 / 0.699 / 0.611.
  - Low-label runs are degenerate: the random-init 30% run has val AUC 0.881 but Czech test AUC 0.581, and the random-init 100% run has val AUC 0.19. Validation with 3 positive infants is not usable for selection.
- **Annotators (FARFUM plus):** AUC ranges from 0.930 to 0.996 across graders (Δ 0.066). Human–human κ has mean 0.806 (range 0.671–1.000). Model–grader κ has mean 0.784 (range 0.494–0.908).

### Key method references (FT)
Coyner 2022 (JAMA Ophthalmol, LMIC external validation); Coyner 2024 (multinational autonomous screening); Brown 2018 i-ROP plus CNN; Campbell 2021 AI for ROP; Chiang 2007 inter-expert plus agreement; RETFound (Zhou 2023); DINOv2 (Oquab 2024); CLIP (Radford 2021); the four dataset papers.

### Weaknesses and leakage risks
1. **Cross-site label inconsistency (DER):** Czech counts treated or regressed eyes (DG9) and AP-ROP (DG8) as "active ROP", while Shenzhen excludes laser scars. One AP-ROP infant contributes 470 positive images.
2. **Tiny effective N for Czech**: 5 positive test infants and 3 positive val infants (DER). Hence the wide CIs and the meaningless checkpoint selection in the fine-tune.
3. **The RETFound comparison is recipe-disadvantaged:**
   - The frozen probe takes the MAE CLS token. MAE pretraining gives the CLS token no direct objective, and MAE features are known to linear-probe poorly. RETFound's own fine-tuning uses global average pooling.
   - The fine-tune used lr 1e-5 with no layer decay, no warmup, 15 epochs and 224 px. RETFound's published recipe uses layer decay 0.65, base lr 5e-3, mixup/CutMix, label smoothing 0.1, drop path 0.2 and 50 epochs (our knowledge of the RETFound repo, not part of ROP-Bench).
   - The authors themselves say the comparison "does not isolate pretraining domain from architecture, preprocessing, input resolution…".
4. Single seed. Image-level AUC. Cross-site probes train on all source images, including that dataset's test split (fine for cross-site, but it differs from the in-domain protocol).
5. Shenzhen in-domain AUC of 0.999 points to a curated or separable source. The authors say this themselves.
6. HVDROPDB has no IDs, so its bootstrap is image-level and its CIs are optimistic.

### What RetinAI could adopt
1. **Leave-one-source-out (LOSO) evaluation.** Train on two of {Ostrava, ROP-VL, Shenzhen} and test on the third, in addition to Multi-View ROP. This gives three more "unseen hospital" tests at no data cost, so any DG idea (style swap, HFC view) can be judged on four held-out sites instead of one. ROP-Bench shows in-domain 0.93–1.0 collapsing to 0.74–0.87 cross-site, so the signal will be visible.
2. **Paired, infant-clustered bootstrap for every A-vs-B claim** (2,000 resamples, ΔAUC CI excluding 0, or ΔStage-1-recall CI). This is the right test behind RetinAI's "X ties Y" conclusions, and it is 50 lines of code (`scripts/bootstrap_compare.py`).
3. **HVDROPDB as a second external binary test** (185 images; RetCam + Neo, India), scored image-level. ROP-Bench shows source-dependent AUC from 0.739 to 0.953 on it, so it discriminates between models.
4. **Use the public manifest to cross-check RetinAI's own harmonisation.** It has Ostrava device tags per image, Shenzhen patient IDs and eye from `zip_information.xlsx`, and DG/PF parsing. Check in particular how RetinAI maps DG8 (AP-ROP) and DG9 (status post).
5. **Label-reference sensitivity.** If FARFUM is added, report Stage-1 recall against each grader's labels, as ROP-Bench did for plus. This is a cheap extra measurement finding for the paper.
6. ROP-Bench independently supports RetinAI's backbone choice (DINOv2 over RETFound), which is useful as a citation.

---

## 4. DeepAdapter

### Citation and access
- Zhao J, Li L, Lin Z, Luo M, Yun D, Wen J, Liu L, Li J, Xu F, Xie S, Wei M, Liu B, Liu Z, Lin D, Lin H. "DeepAdapter: a generalisable algorithm integrating self-supervised learning and unsupervised domain adaptation for robust retinopathy of prematurity screening." *Br J Ophthalmol*, online first 2026 (Crossref issued 12 Jun 2026; article URL dated 12 Aug 2026). DOI 10.1136/bjo-2025-328559. CC BY-NC 4.0.
- Affiliations: Zhongshan Ophthalmic Center (ZOC) and Sun Yat-sen Univ.; Qilu Hospital; Guangdong Women and Children Hospital; Liuzhou and Linyi Maternal and Child hospitals.
- **FT** of the main text (Tables 1 and 2 bodies not readable) and the supplement PDF (Appendices 1–4, Tables S1–S4). Demo: https://deepadapter.cn.

### Dataset (FT)
- **SSL pool:** 500,000 unlabelled infant colour fundus photos from four Chinese centres (Guangdong Women and Children Hospital, Liuzhou MCH, Linyi MCH, ZOC).
- Cameras: RetCam (Natus), PanoCam (Visunex), Orthocone (RS-B002). The camera-to-centre mapping is in Table 1 and not readable.
- **Labelled (Table S1):**

| Phase | Total | Normal | ROP |
|---|---|---|---|
| Fine-tune | 13,544 | 10,500 | 3,044 |
| Internal validation | 5,805 | 4,500 | 1,305 |
| External test (Guangdong W&C) | 19,153 | 5,692 | **13,461** |
| Public test (Ostrava + HVDROPDB) | 954 | 284 | 670 |

  - 13,544 + 5,805 + 19,153 = 38,502 (the "Results" figure). Adding the 954 public images gives 39,456 (the "Abstract" figure). This resolves the apparent inconsistency.
- Binary task: "ROP" means any stage of a basic lesion and/or plus or pre-plus. Severity grading is out of scope.
- Labels: three ophthalmologists, with a senior expert (>15 years) adjudicating.
- Infant counts: not stated in the text (Table 1 not readable).

### Split (FT)
- The three non-external centres were "randomly divided" **7:3** into train and validation. **Unit (image vs patient): not stated.**
- External centre: Guangdong Women and Children Hospital.
- **This hospital is also one of the four SSL-pool centres.** The text never says its test images were excluded from the 500k SSL pool, and the Abstract's "non-overlapping" is unqualified.
- Public test: 954 images "curated from public datasets according to the same criteria". Per-source counts are not given.

### Preprocessing (FT)
- **DeepQuality** quality model (ZOC; Li et al. 2023 NPJ Digit Med): images scoring <0.2 removed, leaving 89.8% retained. DeepQuality also does enhancement in the web system.
- **Resize-and-pad, not crop**, explicitly "to preserve peripheral lesions". Target size not stated.

### Augmentation (FT)
- Random brightness and contrast, **gamma correction**, **Gaussian noise**, random rotation, H/V flips. **No parameters.**
- MoCo-v2 SSL used "scale-and-padding" instead of centre crop. Other MoCo augmentations: not stated.
- Mixup/CutMix: not stated. No synthetic images.

### Model (FT)
- ResNet-50 with "an attention mechanism … inserted between the last two residual blocks" (type not stated). A bottleneck layer before the head is used only during UDA.
- Init: **random init, then MoCo-v2** SSL. Appendix 4: "only the supervised model uses ImageNet-pretrained weights, whereas all three DeepAdapter configurations start from random initialization".
- **UDA = DSAN (Zhu et al. 2021): LMMD**, a class-conditional MMD weighted by source labels and target soft pseudo-labels, applied on bottleneck features.
- Loss: L = L_c(D_s) + λ·L_lmmd(D_s, D_t). **λ not stated.**
- Deployment: a "one-time, offline unsupervised alignment … using a small batch of unlabelled target-domain data". Example: adapting to the **954-image public set took 27 epochs (1 h 6 min, 4× V100)**.

### Training (FT)
- MoCo-v2 settings (queue, momentum, temperature, epochs, batch, lr): **not stated**. Fine-tune and UDA lr, epochs, batch, optimiser, schedule: **not stated**.
- Selection: best validation accuracy.
- Python 3.8, PyTorch, Ubuntu 20.04.6, 4× Tesla V100.

### Evaluation (FT)
- Accuracy, sensitivity, specificity, F1, AUC, AUPRC. Fixed **threshold 0.5**. Non-parametric bootstrap for p-values; the number of resamples and the clustering unit are not stated. CIs of about ±0.005 on 19,153 images suggest image-level resampling.
- **Domain-shift metric:** CORAL distance = (1/4d²)·‖C_S − C_T‖²_F on features (Appendix 2).

### Results (FT)
- CORAL distance (supervised → DeepAdapter): internal 1.335e-5 → 2.643e-5; **external 0.297 → 3.550e-6**; public 0.145 → 3.336e-6.
- Internal accuracy about 0.99 for both, no significant difference.
- **Ablation (Table A3.1), External / Public:**

| | Baseline (ImageNet sup.) | DeepAdapter w/o DA (SSL only) | DeepAdapter w/o SSL (random init + DSAN) | DeepAdapter |
|---|---|---|---|---|
| ACC | 0.739 / 0.813 | 0.766 / 0.751 | 0.715 / **0.494** | **0.828 / 0.839** |
| AUC | 0.942 / 0.932 | 0.952 / 0.898 | 0.873 / 0.727 | 0.976 / 0.934 |
| Sens | 0.837 / 0.748 | 0.840 / 0.666 | 0.718 / 0.505 | 0.882 / 0.790 |
| Spec | 0.884 / 0.899 | 0.897 / 0.827 | 0.867 / 0.753 | 0.923 / 0.902 |
| F1 | 0.701 / 0.713 | 0.712 / 0.641 | 0.614 / 0.454 | 0.777 / 0.783 |

- **RETFound** (ViT-L, official weights, fine-tuned on the internal training set; Table S3), External / Public: ACC 0.808 / 0.741, AUC 0.885 / 0.894, Sens 0.720 / 0.610, Spec 0.901 / 0.777, AUPRC 0.877 / 0.854. DeepAdapter is better on every metric (p<0.0001).
- **UDA method comparison (Table S4), External ACC / Sens / Spec / F1 | Public ACC / Sens / Spec / F1:**
  - DAN (MMD): 0.732 / 0.802 / 0.877 / 0.711 | 0.768 / 0.728 / 0.878 / 0.693
  - DeepCORAL: 0.743 / 0.831 / 0.885 / 0.710 | 0.828 / 0.791 / 0.899 / 0.782
  - DANN: 0.735 / 0.837 / 0.879 / 0.727 | **0.673** / 0.703 / 0.838 / 0.634
  - **DSAN (LMMD): 0.828 / 0.882 / 0.923 / 0.777 | 0.839 / 0.790 / 0.902 / 0.783**
- Appendix KNN check of SSL features, 5 folds: mean accuracy 0.7075 after the first epoch and 0.7525 after the last. The fold accuracies come in steps of 0.0125, which implies about 80 samples per fold (our inference).

### Key method references (FT)
MoCo-v2 (Chen et al. 2020); DSAN/LMMD (Zhu et al. 2021 TNNLS); Deep CORAL (Sun & Saenko 2016) and "Return of frustratingly easy DA" (Sun et al. 2016); DAN (Long 2015); DANN (Ganin 2016); Ben-David 2007/2010 DA bound; DeepQuality (Li 2023); RETFound; Coyner 2019 ROP image-quality CNN; Wagner 2023 Lancet Digit Health plus models; Timkovič 2024 (Ostrava) and Agrawal 2024 (HVDROPDB).

### Weaknesses and leakage risks
1. **Transductive adaptation on the test set.** The 954 public images are both the unlabelled UDA target and the test set (Appendix 3), and the external centre was likely handled the same way. Gains are "adapt-on-the-test-distribution" gains, not unseen-hospital gains.
2. **The external centre is in the SSL pretraining pool** (one of the four centres), so its images were probably seen without labels during SSL.
3. Split unit (image vs patient) is not stated, while internal accuracy is 0.99. Image-level leakage is possible.
4. **The headline metrics are internally inconsistent** with Table S1 under a binary confusion matrix (our arithmetic):
   - External: sensitivity 0.882 on 13,461 ROP and specificity 0.923 on 5,692 Normal imply accuracy 0.894 and F1 about 0.92. The paper reports accuracy 0.828 and F1 0.777. Accuracy below *both* sensitivity and specificity is impossible for one binary threshold.
   - Public: 0.790 and 0.902 imply accuracy 0.823, against 0.839 reported.
   - The metric definitions must differ (macro averages, or another class) but are not explained.
5. The baseline differs in initialisation (ImageNet vs random+SSL) as well as in UDA. "w/o SSL" (random init + DSAN) collapses to 0.494 on the public set, showing that DSAN alone is unstable.
6. A fixed 0.5 threshold under very different prevalence: 22% ROP in training against 70% in the external set.
7. No hyperparameters anywhere, including λ, the MoCo settings and the image size. Not reproducible.

### What RetinAI could adopt
1. **If UDA is tried again, use class-conditional LMMD (DSAN), not marginal alignment.** In their head-to-head, DAN, DeepCORAL and DANN barely moved from the 0.739 baseline on the external centre, while DSAN gave 0.828. VEA-Net's DANN-only also hurt. RetinAI's mean-teacher result is consistent with "marginal or self-ensembling DA doesn't help".
   - Protocol to keep the unseen-hospital claim honest: adapt on unlabelled images from a **disjoint half of Multi-View ROP infants** and score on the other half, swapping halves. Report it as a separate "unlabelled-target-adapted" setting.
   - Use binary any-ROP LMMD first. Stage-1 pseudo-labels at about 0.5 recall would reinforce errors in a 4–5-class LMMD.
2. **CORAL distance (or a linear domain probe) as a diagnostic.** Compute (1/4d²)‖C_S − C_T‖²_F on penultimate features between training sources and Multi-View ROP for each model variant. This fits RetinAI's measurement-paper framing, and is cheap.
3. **Augmentation gaps:** add **gamma correction** and **Gaussian noise**. Revisit **RandomResizedCrop(0.6–1.0)**: Stage-1 demarcation lines lie at the peripheral vascular/avascular border, and a 0.6-scale crop can cut the lesion while keeping the Stage-1 label. DeepAdapter chose resize-and-pad specifically to keep the periphery. A cheap ablation is min-scale 0.6 vs 0.85, or zoom-out/pad only, measured on Stage-1 recall at Multi-View ROP.
4. **A quality gate.** DeepQuality removed 10.2% of images. A quality score as an exclusion or flag at the new hospital could separate Stage-1 misses on poor images from true misses. It is not a training change.
5. In-domain SSL at their scale (500k) is not available to RetinAI. The SSL-only arm gave mixed results (+0.027 external, −0.062 public against ImageNet). Low priority.

---

## 5. Mahboob, Ahsan, Wilms (SPIE MI 2025): AB only

- Mahboob Z, Ahsan AO, Wilms M. "Using an adult retinal image analysis foundation model for retinopathy of prematurity staging: are there benefits?" *Proc SPIE* 13407, Medical Imaging 2025: Computer-Aided Diagnosis, paper 13407-116 (SPIE ID 1340731), published 4 Apr 2025. DOI 10.1117/12.3047303.
- University of Calgary, per the SPIE programme listing.
- **AB.** Semantic Scholar marks it CLOSED. No arXiv copy and no thesis copy were found.
- **Abstract content (AB, from search snippets of the SPIE abstract):**
  - Asks whether fine-tuning **RETFound** (pretrained on adult colour fundus photos) for **ROP severity staging** on limited infant data beats fine-tuning generic ImageNet-pretrained classifiers.
  - Finding: RETFound, "despite having seen nearly one million adult fundus images during pre-training, does not outperform the generic models on the largest public ROP dataset".
  - Effect sizes, metrics and the dataset name are not available to us.
- **Reference list (Crossref, AB-level metadata):**
  - Includes Zhao et al. 2024 *Sci Data* 11 (the Shenzhen/A-Fundus dataset). The "largest public ROP dataset" may be A-Fundus *or* Ostrava, since many reference entries lack metadata. **Not determinable.**
  - Includes He 2016 (ResNet), Simonyan 2014 (VGG), Dosovitskiy 2020 (ViT), Liu 2022 (ConvNeXt), He 2022 (MAE), Deng 2009 (ImageNet), RETFound, Wagner 2023 (Lancet DH), Coyner 2024 (JAMA Ophthalmol), ICROP3.
  - Our inference, not stated: the generic comparators were probably ResNet, VGG, ViT and ConvNeXt.
- Dataset, split, preprocessing, augmentation, training and evaluation details are **not stated (AB)**.
- **Weaknesses:** cannot be judged from the abstract. A single public dataset and no external test are implied by the abstract wording (AB).
- **Adopt:** nothing methodological. It is a third independent citation, alongside ROP-Bench and SABPI-Net's Table 1, for RetinAI's own finding that RETFound-Green gives no gain. Cite as "adult-retina foundation pretraining confers no benefit for infant ROP" in related work.

---

## 6. Lan, Tsai, Wu: Type-1 ROP field-of-view MIL

### Citation and access
- Lan W-H, Tsai C-L, Wu W-C. "Optimizing Field-of-View for Type-1 Retinopathy of Prematurity via Multiple Instance Learning." *Transl Vis Sci Technol* 15(8):13, 2026 (published online 18 Aug 2026). DOI 10.1167/tvst.15.8.13. CC BY-NC-ND 4.0.
- Affiliations: Chang Gung Memorial Hospital Linkou / Chang Gung Univ.; Queens College CUNY.
- **The VoR returned 403** (Cloudflare) and is not in PMC.
- **FT of the preprint** (Research Square rs-7513640 v1, 3 Sep 2025, CC BY 4.0). All facts below are from the preprint. The VoR adds four references (MixStyle, Matsuura domain generalisation, Tseng 2026, and one renumbering), so it may include domain-generalisation content we could not read.

### Dataset (FT-preprint)
- **CGMH** (Taiwan), May 2013 – Jul 2022: **204 eyes**, of which 141 are non-type-1 and 63 are type-1.
- **Osaka University** (Japan), Apr 2013 – Nov 2018: **80 eyes**, 40 non-type-1 and 40 type-1. Non-public.
- RetCam systems (model not stated). **5 images per eye**: temporal, nasal, central, superior, inferior, giving about 1,420 images (our arithmetic).
- Infant count: not stated.

### Split (FT-preprint)
- **5-fold stratified CV at eye level**, with all images of an eye in the same fold.
- **The two centres are pooled; there is no cross-centre test**, even though two centres were available.
- The *fellow eye of the same infant may fall in another fold*. Grouping by infant is not stated.

### Preprocessing (FT-preprint)
Resize to 224×224. **CLAHE on the green channel only**, merged back with the original R and B channels. Pixels scaled to [0,1].

### Augmentation (FT-preprint)
- Keras-style ImageDataGenerator, online: H/V shift up to 20%, brightness 0.5–1.5, shear (amount not stated), zoom (amount not stated).
- **Rotation and H/V flips deliberately excluded** so that view identity (temporal vs nasal) is preserved.

### Model (FT-preprint)
- **CBAM-ResNet18** feature extractor (GAP output). Pretraining not stated. How the extractor is trained ("two-stage") is not stated: whether it is frozen ImageNet or trained on image labels.
- **Feature-level MIL:** per-view features → **mean pooling** over the chosen FOV subset → MLP (Dense 256 + ReLU, dropout 0.5, sigmoid output, L2 0.001 on both dense layers). A separate model is trained per FOV combination.
- **Outcome-level MIL:** the same extractor and head trained on all images regardless of view (each image presumably labelled with its eye's label); **per-image probabilities averaged** per eye.

### Training (FT-preprint)
Adam, lr 0.001, BCE. Epochs, batch size, early stopping and hardware: not stated. Class imbalance handling: none stated (CGMH has 31% type-1).

### Evaluation (FT-preprint)
Eye-level accuracy, precision, recall and F1. **No AUC.** Threshold not stated (presumably 0.5). t-tests across the 5 folds (feature vs outcome level, and between FOV combinations). No CIs.

### Results (FT-preprint, Table 1)

| FOV set | Feature-MIL Acc / Prec / Rec / F1 | Outcome-MIL Acc / Prec / Rec / F1 |
|---|---|---|
| T+N+C+S+I | 0.858 / 0.760 / 0.740 / 0.745 | 0.766 / 0.867 / **0.299** / 0.439 |
| **T+N+C** | **0.888 / 0.839 / 0.823 / 0.825** | 0.804 / 0.942 / 0.406 / 0.558 |
| S+I+C | 0.838 / 0.701 / 0.782 / 0.723 | 0.745 / 0.843 / 0.241 / 0.354 |
| T+N | 0.880 / 0.830 / 0.817 / 0.818 | 0.800 / 0.886 / 0.456 / 0.570 |
| S+I | 0.817 / 0.705 / 0.741 / 0.685 | 0.795 / 0.848 / 0.362 / 0.492 |
| T | 0.876 / 0.835 / 0.793 / 0.809 | 0.793 / 0.871 / 0.376 / 0.505 |
| N | 0.862 / 0.776 / 0.842 / 0.804 | 0.806 / 0.931 / 0.400 / 0.546 |
| C | 0.867 / 0.798 / 0.836 / 0.804 | 0.805 / 0.838 / 0.487 / 0.609 |
| S | 0.839 / 0.728 / 0.784 / 0.754 | 0.796 / 0.867 / 0.371 / 0.492 |
| I | 0.820 / 0.817 / 0.659 / 0.704 | 0.795 / 0.848 / 0.362 / 0.492 |

- Feature-level beats outcome-level significantly on recall and F1 for every combination (p<.05).
- Significant pairwise differences within feature-level MIL (Table 2):
  - S vs T, precision: p=0.009
  - S vs T+N+C, precision: p=0.025
  - I vs T+N+C, accuracy: p=0.043
  - I vs T+N+C, F1: p=0.023
  - S+I vs T, precision: p=0.048
  - S+I vs T+N+C, precision: p=0.048
- A type-1 case is shown in which the **demarcation line is visible only in the temporal view**. Mean-of-probabilities missed it, and feature pooling caught it.

### Key method references (FT-preprint)
He 2016 (ResNet); Woo 2018 (CBAM); Pizer 1987 (CLAHE); Wu 2022 JAMA Netw Open (majority vote / 1-vote veto / image-level type-1); Li 2022 *Retina* (DeepROP feature-binding over 5–13 images); Kıran Yenice 2024 *Eye* (type 1/2/A-ROP); Rosenblatt 2021 (central view has highest agreement with gestalt; superior and inferior κ<0.5); Morrison 2016 e-ROP (number and quality of images); Ji 2022 SUNDROP (effective FOV); Huang 2022 and Wilson 2006 (temporal vessel angle). The VoR adds Zhou 2021 **MixStyle** and Matsuura 2020 domain generalisation.

### Weaknesses and leakage risks
1. Pooled CV across two centres with no held-out centre. Possible fellow-eye leakage, since eyes rather than infants are the grouping unit.
2. The outcome-level baseline is a straw man:
   - Every view is trained with the eye-level label, so views without lesions are labelled positive.
   - Probabilities are **averaged**, which dilutes a single positive view.
   - The fixed threshold explains recall of 0.24–0.49.
   - **There is no max-of-probabilities or noisy-OR baseline**, which is the obvious fix and is RetinAI's current rule.
3. No AUC, so all comparisons depend on the threshold. t-tests on 5 folds have very low power. Small N (284 eyes).
4. Extractor training and pretraining details are missing.

### What RetinAI could adopt
1. **A per-view contribution analysis at Multi-View ROP**, if its images carry view labels. Compute Stage-1 recall when the max rule is restricted to T, N, C, T+N+C, and so on. Lan's clinical rationale is that ROP shows up first in the temporal periphery in more mature infants and the nasal periphery in the least mature. This tells RetinAI whether Stage-1 misses come from views where the ridge is not visible (a ceiling of the label unit) or from a model failure. It is a measurement finding for the paper, and needs no retraining.
2. **Use Lan's data to explain why RetinAI's gated-attention MIL lost to max.** Lan's feature-level *mean* won only against mean-of-probabilities. Neither was compared with max. RetinAI's evidence (max beats attention-MIL at the new hospital; the LSE bag loss helps on Ostrava only) and Lan's (mean-probability pooling collapses recall) point the same way. For sparse, single-view lesions, use pooling close to max (max, LSE with a temperature, or top-k mean). Do not use mean or learned attention trained on a few sources.
3. **Green-channel-only CLAHE** keeps colour fidelity while boosting vessel contrast. It is worth a one-line check against RetinAI's current CLAHE variant, but low priority.
4. Group folds by **infant**, not eye. RetinAI already does this; mention it as a contrast in the paper.

---

## 7. Synthesis: ranked adoptables for RetinAI (Stage-1 recall at a new hospital)

1. **More held-out sites for measurement (ROP-Bench):** leave-one-source-out over {Ostrava, ROP-VL, Shenzhen} plus HVDROPDB as a binary external set, with infant-clustered paired bootstrap ΔAUC and ΔStage-1-recall CIs. Without this, every DG idea is judged on one hospital and is likely to "tie".
2. **More Stage-1 infants (VEA-Net's derivation):** FARFUM-RoP expert-vote stage labels give 373 Stage-1 images from 68 infants on RetCam III, from a new country. Today RetinAI's Stage 1 comes from about 83 infants outside ROP-VL (DER: Ostrava 12, Shenzhen 71).
3. **Source-only frequency-style DG (SABPI-Net DAESSM):**
   - Low-frequency FFT swap, 2×2 or 3×3 window including DC, with donors drawn from another source or camera.
   - Cosine embedding-consistency loss with λ 0.01–0.1.
   - Optional masked high-pass (Ben-Graham-like) view with shared weights, σ scaled to resolution, aimed at thin demarcation lines.
4. **Lesion-preserving geometry (DeepAdapter):** test RandomResizedCrop min-scale 0.6 against 0.85, or pad/zoom-out only, because peripheral Stage-1 ridges can be cropped away while the label is kept. Add gamma and Gaussian-noise augmentation.
5. **If DA is revisited:** class-conditional LMMD (DSAN) on unlabelled target images from a disjoint half of target infants, reported separately from the unseen setting. Track CORAL distance as a diagnostic. Skip DANN, DAN and marginal MMD, which failed in DeepAdapter Table S4 and VEA-Net's ablation.

**Fine-tuning recipes actually used:**
- ROP-Bench: AdamW, backbone 1e-5 / head 1e-3, wd 0.05, cosine with no warmup, 15 epochs, batch 32, 224 px, RRC(0.6–1)+HFlip, class-weighted CE. No layer decay, mixup, CutMix, label smoothing or drop path.
- SABPI-Net: SGD Nesterov 0.9, lr 1e-3, wd 1e-4, 1-epoch warmup then per-step cosine, 100 epochs, 224 px, resize+crop and H/V flips plus the Fourier style swap, plain CE + 0.01 (paper) or 0.1 (code) cosine-consistency loss, drop path 0.1. No layer decay, mixup or CutMix. No pretrained checkpoint in the released code.
- DeepAdapter: MoCo-v2 SSL from random init on 500k infant images, then supervised fine-tune with DSAN-LMMD on unlabelled target images. **Every hyperparameter is unreported.** Augmentation: brightness/contrast, gamma, Gaussian noise, rotation, H/V flip, resize-and-pad. Mixup and layer decay: not stated.
