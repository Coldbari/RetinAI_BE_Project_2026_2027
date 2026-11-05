# Data generation, augmentation and training recipes for ROP staging: what the evidence says

Evidence review compiled 2026-10-30. Scope: synthetic data (A), augmentation (B), fine-tuning and training recipes (C), and more real data (D), judged against one criterion: **does it improve accuracy at an unseen hospital?**

**Tags.** **FT** means I read the full text (or the full relevant section/table). **AB** means abstract, publisher or record page, or a search snippet only. **PP** marks a preprint (not peer-reviewed). Numbers are quoted only from sources tagged next to them. Where I give my own estimate (for example T4 cost) it is labelled *(estimate)*.

**Evidence-strength scale used in the ranked table.**
- **Strong:** several peer-reviewed studies show gains on held-out sites or domains, at least one of them in fundus imaging.
- **Moderate:** one fundus or medical study shows gains on an unseen domain, or there is strong natural-image domain-generalisation (DG) evidence plus a fundus use.
- **Weak:** gains are internal-test only, the evidence is mixed, or it comes from a single abstract or preprint.
- **None:** no external evidence found, or the evidence shows harm.

---

## Key findings

1. **Synthetic images do not reliably help at an unseen site.** The cleanest fundus test has a negative external result: Nderitu et al. 2025, where synthetic data gave +0.024 internal AUROC and −0.039 external. For ROP, no published study shows synthetic images improving stage classification at all, internal or external.
2. **More source hospitals and cameras is the most consistent lever.** In DR, going from one camera to three raised average quadratic weighted kappa (QWK) from 68.6 to 74.2, and from 49.2 to 58.9 on an unseen camera (FT). An ROP stage model collapsed to AUROC 0.62 when moved across populations (AB).
3. **Label quality caps everything.** Stage is the most common source of expert disagreement in ROP (AB). In DR, a small adjudicated set "allows substantial improvements" (AB). This is consistent with the team's observation that eight backbones sit within ±0.02 QWK.
4. **Cheap, evidence-backed training changes exist.** Weight averaging (EMA/SWA/SWAD), linear-probe-then-fine-tune (LP-FT) and fundus-realistic photometric/degradation augmentation all add almost no T4 cost. Colour jitter specifically has mixed evidence: it helped in one fundus DG study and hurt in another.
5. **New real data.** The main new public ROP release is Macretina (India, 3nethra Neo, Dec 2025). Its classes are Normal, ridge (stages 1–3 pooled) and A-ROP, so it adds no stage-specific labels. One Hugging Face dataset may add Stage 4 and A-ROP, but it may be a duplicate of a set the team already has; check before use. No 2026 multi-centre stage-labelled public release was found.

---

## A. Synthetic data (GANs and diffusion models)

### A1. ROP-specific work

| Study | What was generated | Data | Result | External test? | Tag |
|---|---|---|---|---|---|
| **Coyner et al. 2022**, *Ophthalmology Science* 2(2):100126. [EPMC full text](https://www.ebi.ac.uk/europepmc/webservices/rest/PMC9560638/fullTextXML) | **Grayscale retinal vessel maps (RVMs), 256×256**, not RGB fundus images. Three progressive GANs (PGANs) were trained, one each for normal, pre-plus and plus. | 5,842 images from 963 infants (i-ROP). Train 3,477 images (2,908 normal, 464 pre-plus, **105 plus**). The synthetic set matched the training set in size and class mix. | Plus vs not-plus AUROC: **0.971 (trained on synthetic) vs 0.934 (trained on real)**, DeLong P=0.006. On a 100-image expert set, κ vs consensus of 8 experts was **0.922 vs 0.701**. | **No.** Both test sets come from the same i-ROP study. | FT |
| Coyner 2022, privacy and memorisation check | | | FID 6.53 (synthetic vs real training set) against a real-vs-real baseline of 4.64. The closest synthetic-to-real feature distance (6.15) was larger than real-to-real distances (5.54 for the same infant, 5.71 for different infants). The authors say this "is not a guarantee of de-identification". No differential privacy was used. | | FT |
| **Chen, Coyner et al. 2021**, "Deepfakes in Ophthalmology", *Ophthalmology Science* 1(4):100079. [EPMC](https://www.ebi.ac.uk/europepmc/webservices/rest/PMC9562356/fullTextXML) | Pix2PixHD, vessel map to RGB RetCam image, 640×480 | 6,058 images from 970 infants; 880 synthetic test images | 4 ROP experts reached 49–61% accuracy at telling real from fake; only one was significant (P=0.043). **5/880 (0.57%) were "obviously unrealistic"**. **Stage 4–5 were excluded because those images were too sparse.** No classifier experiment. | No | FT |
| **Hou et al. 2023, ROP-GAN**, *Phys Med Biol* 68(20). [PubMed 37619572](https://pubmed.ncbi.nlm.nih.gov/37619572/) | U2-Net vessel extraction, then an adversarial autoencoder to diversify vessel maps, then a paired GAN with attention | The abstract gives no dataset size | Claims realistic synthesis and that it "eases data imbalance". **The abstract reports no classification gain and no external test.** | Not reported | AB |
| **Demirci et al. 2025**, topology-aware conditional latent diffusion for multi-view infant fundus synthesis, *CHASE 2025* (ACM). [record](https://researchconnect.stonybrook.edu/en/publications/topology-aware-conditional-latent-diffusion-for-multi-view-fundus/) | Latent diffusion that generates other fields of view from one image, with a persistent-homology vessel loss | 2,862 images from 954 patients | FID 47.93, which the authors say is 19.91 better than a baseline latent diffusion model. **No downstream classifier result.** | No | AB (search snippet) |
| EyeDiff 2026 (below) | | | ROP appears only as a downstream rare-disease class. The paper reports **low text–image alignment for ROP (VQAScore 0.493)**. | | FT (partial) |

**Bottom line for ROP.** No published study shows that synthetic images improve **ICROP stage** classification, internal or external. The single positive ROP result (Coyner 2022) covers plus disease only, on vessel maps only, with an internal test only.

### A2. Strongest fundus and medical evidence on external (unseen-site) accuracy

| Study | Setting | Internal effect | External effect | Tag |
|---|---|---|---|---|
| **Nderitu et al. 2025**, *Communications Medicine* 6:57. [EPMC](https://www.ebi.ac.uk/europepmc/webservices/rest/PMC12830830/fullTextXML) | Conditional cascaded diffusion at 512 px, used to add synthetic positives to a referable-DR prognosis model. 72,559 development eyes; external test 2,842 eyes. | Baseline 0.827. Adding ×2 synthetic positives gave **0.851 (+0.024, p=0.044)**. | Baseline 0.756. ×1 gave 0.761 (+0.005, p=0.83), **×2 gave 0.717 (−0.039)**, ×4 gave 0.750. External AUPRC improved by +0.056. Authors: the benefit "did not generalise". | FT |
| Nderitu 2025, failure modes | | | Images conditioned on **mild DR were graded as no DR**: the model could not synthesise microaneurysms. A senior grader identified synthetic images with 100% accuracy (vessels too straight, illogical branching). Memorisation was not assessed. | FT |
| **EyeDiff** (Chen et al. 2026), *npj Digit Med* 9:371. [EPMC](https://www.ebi.ac.uk/europepmc/webservices/rest/PMC13172480/fullTextXML) | Stable Diffusion 1.5 fine-tuned on 42,048 images; minority classes filled up to the majority-class count; RETFound downstream | OCTDL 0.982→0.996; rare diseases 0.871→0.919 | **Mixed.** Gains: IDRiD 0.826→0.837, Retina 0.857→0.892, OphthalWeChat 0.613→0.663. **No gain or a loss:** APTOS 0.946→0.945, MESSIDOR-2 0.884→0.879, PAPILA 0.820→0.814. **Retina Image Bank and OCTDL were used both to train the generator and as test sets**, with no deduplication reported. No memorisation check. | FT (first 100k characters) |
| **Ktena et al. 2024**, *Nature Medicine* 30:1166. [arXiv version](https://arxiv.org/html/2304.09218) | Diffusion augmentation for histopathology, chest X-ray and dermatology | n/a | OOD gains: **+7.7% accuracy (histopathology), +5.2% (chest X-ray), +63.5% high-risk sensitivity (dermatology)**. Caveats: the histopathology generator used **1.8 M unlabelled images**. Gains were largest in the low-data regime. In dermatology, only 50.0±12.6% of generated images were of diagnostic quality, there was a "minor decline" in the less-skewed setting, and "real data is more essential". | FT (preprint version, first 100k characters); AB (journal) |
| **Pandey et al. 2025**, *Br J Ophthalmol* 109(7):791 | Diffusion; 238 generated images added to the smallest class | F-score +5.3%, accuracy +5.8%, on a **100-image internal test** | Not tested | AB |
| **Cao-Xue et al. 2025** (SynFundus-1M classifiers), [arXiv 2508.15986](https://arxiv.org/abs/2508.15986) **PP** | Trained only on synthetic data | Internal macro-AUC 0.9973 | Combined DR set **0.7972**; AIROGS 0.9126. There is no real-data model on the same sets for comparison. | AB |

### A3. Failure modes, with evidence

- **Memorisation.** Dar et al. 2024 ([arXiv 2402.01054](https://arxiv.org/html/2402.01054v1), PP) found latent diffusion models memorised **32.6% of a 10k chest X-ray training set** and 41.7% of a 242-volume CT set. Memorisation fell with more data: **54.7%, 31.8% and 16.2%** for 5k, 10k and 20k training images. Flips plus ±5° rotation during generator training cut it to **5.6%**. Over-training increased it (FT). Akbar et al. ([arXiv 2305.07644](https://arxiv.org/abs/2305.07644), PP) found diffusion models memorise more than StyleGAN, especially on small datasets (AB).
  - *Implication:* the team's scarce classes (Stage 4/5 = 89, Stage 1 = 577) are exactly the regime where generators copy training images.
- **Subtle lesions are lost.** Nderitu 2025 found mild-DR conditioning produced images graded as no DR (FT). Stage 1 is defined by a thin, flat demarcation line, so the same failure is plausible for Stage 1 *(inference)*.
- **Label noise is reproduced.** A class-conditional generator learns p(image | noisy label). Stage-label noise is therefore copied into the synthetic set, and it is copied in greater volume *(inference; no ROP study tests this)*.
- **Leakage into test.** EyeDiff evaluated on datasets that were also in the generator's training set (FT). For cross-validation, the generator must be retrained per fold on training-fold patients only, or synthetic images become near-copies of test-fold patients *(inference, supported by the memorisation numbers above)*.
- **Mode collapse and hallucination.** Chen 2021 found 0.57% of images obviously unrealistic and noted GANs "can only synthesize within the training distribution" (FT). EyeDiff reports unrealistic colours, enhanced edges and noise (FT).
- **Privacy is not guaranteed.** Coyner 2022 (FT); see also the [multi-modality systematic review, arXiv 2407.00116](https://arxiv.org/pdf/2407.00116), which says "privacy preservation [is] mostly unaddressed" (AB, PP).

---

## B. Augmentation evidence for fundus classification

### B1. Geometric augmentation

- **Full 0–360° rotation.** Kaggle DR 2015 winner (Graham): training used ±10% scale, **0–360° rotation** and ±0.2 skew; at test time, images were "just rotated randomly". [Competition report](https://kaggle-forum-message-attachments.storage.googleapis.com/88655/2795/competitionreport.pdf) (FT). The report contains no ablation, so this is winning practice rather than measured evidence. I found no fundus study that ablates 360° vs ±15° rotation on an external test.
- **Horizontal flip and left/right eyes.** A horizontal flip maps a right eye onto a left-eye layout, so for ROP stage it is label-preserving. However, a 2022 *Scientific Reports* study reportedly found CNNs still identified laterality after flipping, i.e. flipped images remain distinguishable and could introduce bias ([DOAJ record](https://doaj.org/article/66cbde4c002d4fee8d355f8303f07f3a); AB, search snippet only, title not verified). I found no evidence that this harms external accuracy.
- **Vertical flip.** I found no fundus study testing its plausibility. It swaps superior and inferior retina. For ROP stage, which is ridge morphology, this is probably harmless *(inference)*.

### B2. Photometric augmentation and colour normalisation

The two most relevant fundus studies disagree:

- **For: GDRNet / FundusAug** (Che et al., MICCAI 2023; [arXiv 2307.04378](https://arxiv.org/html/2307.04378v1), FT). This is a leave-one-domain-out DR benchmark (GDRBench, 8 datasets).
  - FundusAug = brightness, contrast, saturation, hue and sharpness (each p=0.5), plus degradations: **halo, holes, spots, blur**.
  - Ablation, average AUC on unseen domains: ERM 75.92; visual transforms only **79.30**; degradations only 77.08; both **80.46**, i.e. **+4.5 AUC**. The full GDRNet reaches 82.6.
- **Against: RADR** (Monedero et al., MIDL 2024; [PDF](https://www.inf.uni-hamburg.de/en/inst/ab/cv/media/monedero-westhaeusser-yaghoubi-etal-midl2024.pdf), FT). This tests EyePACS camera domains A–E, two of them unseen.
  - **ColorJitter hurt everywhere.** Average QWK fell 68.6→64.0 (single-camera), 74.2→68.4 (multi-camera) and 76.2→70.4 (adversarial). Unseen camera D dropped by up to 18.2 points.
  - **AugMix** helped only under multi-camera training (+1.8 QWK).
  - *Implication:* hue/colour jitter strength needs a leave-one-hospital-out ablation; it is not a free win.
- **Local-mean subtraction (Ben Graham preprocessing).** Graham: "I tried adding color-space transformation… this did not seem to help", likely because local-mean subtraction already normalises colour (FT).
- **Histogram matching or colour constancy.** No fundus cross-site study found. The nearest analogue is computational pathology (Tellez et al. 2019, *MedIA*, 9 centres; [arXiv 1902.06543](https://arxiv.org/abs/1902.06543)): colour **augmentation** (HSV/HED) mattered more than normalisation, and normalisation alone was insufficient (AB plus a secondary summary).

### B3. Automated augmentation policies (RandAugment, TrivialAugment, AugMix)

- **Fang, Kornblith & Schmidt 2023** ([arXiv 2301.04644](https://www.arxiv.org/pdf/2301.04644), PP, FT), on APTOS 2019:
  - Architectures after VGG gave no significant improvement.
  - "increasing color and affine augmentation by itself can account for a 0.03 QWK point improvement".
  - Fine-tuning with RandAugment usually helped ResNet-50, but **"for DeiT models… additional augmentation did not significantly increase performance"**.
  - This is a random split, not external.
- **Ktena 2024:** RandAugment "harms fairness in the OOD setting" in dermatology (FT preprint).
- **No fundus external evidence found for TrivialAugment.**

### B4. Mixup and CutMix

- **GDRBench, leave-one-domain-out average AUC** (FT): **Mixup 75.0 vs ERM 75.9**, i.e. no gain on unseen domains. Under single-source DG it was 68.3 vs 67.1.
- **Balanced-MixUp** (Galdran et al., MICCAI 2021; [arXiv 2109.09850](https://arxiv.org/pdf/2109.09850), FT), EyePACS internal test, ResNeXt-50, quadratic kappa:
  - Instance sampling 80.17; **class-balanced loss 79.19**; focal loss 79.73; Balanced-MixUp α=0.2 **80.78**.
  - The gain is small and internal-only. The paper also reports a Messidor-2 score (85.14) but no Messidor-2 baseline ablation.
  - Note that class-balanced loss, which the team uses, was slightly below plain instance sampling here.
- **RETFound's own code defaults to mixup 0 and CutMix 0** ([main_finetune.py](https://raw.githubusercontent.com/rmaphoh/RETFound_MAE/main/main_finetune.py), FT).

### B5. Lesion copy-paste

LesionPaste ([arXiv 2203.06354](https://arxiv.org/pdf/2203.06354v1), PP) pastes lesion patches into normal images for **early-DR anomaly detection** (AB). I found no copy-paste study for graded classification with an external test, and none for ROP.

ROP stage depends on ridge morphology (flat line vs ridge vs extraretinal fibrovascular proliferation), not just whether a ridge is present. Pasted ridges would therefore carry uncertain stage labels, and paste seams can become shortcuts *(inference)*. HVDROPDB and Macretina provide ridge masks that would make this technically possible.

### B6. Fourier-domain style transfer (FDA)

- **FedDG / ELCFS** (Liu et al., CVPR 2021; [paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Liu_FedDG_Federated_Domain_Generalization_on_Medical_Image_Segmentation_via_Episodic_CVPR_2021_paper.pdf)): amplitude-spectrum exchange across sites gave **+2.02% Dice** for fundus optic disc/cup **segmentation** at unseen sites vs FedAvg (AB). It was unclear from the source whether that gain is absolute or relative.
- **No fundus classification study with an external test found.** Fourier thresholding for fundus ([arXiv 2309.09866](https://arxiv.org/abs/2309.09866v3)) is also segmentation-only (AB).
- **Relevance:** the team has unlabelled target-hospital photos. Using their amplitude spectra as style sources is cheap. The evidence is weak and comes from segmentation.

### B7. Test-time augmentation (TTA)

- Graham used random-rotation TTA but reported no ablation (FT).
- I found no fundus study that isolates a TTA gain on an external set.
- FunOTTA ([arXiv 2407.04396](https://arxiv.org/pdf/2407.04396v3), PP) warns that some test-time adaptation methods degrade fundus performance under shift (AB). That warning concerns adaptation, not plain TTA averaging.
- TTA is inference-only, so it is cheap to test on existing checkpoints.

### B8. What improves cross-site generalisation (summary)

| Lever | Evidence |
|---|---|
| More source domains/cameras | **Consistent.** RADR, single camera→three cameras: average QWK 68.6→74.2; unseen camera D 49.2→58.9 (FT). |
| Fundus-realistic photometric + degradation augmentation | **Positive in one benchmark.** GDRNet +4.5 AUC (FT). |
| Generic ColorJitter | **Negative in one benchmark.** RADR (FT). |
| Mixup | **No gain.** GDRBench (FT). |
| FDA, histogram matching, TTA | **No fundus-classification external evidence found.** |

---

## C. Training recipes for ViTs on small medical datasets

### C1. Reference recipes

| Recipe | LR | Layer decay | Drop path | Epochs / warmup | Label smoothing | Mixup / CutMix | Other | Tag |
|---|---|---|---|---|---|---|---|---|
| **RETFound paper** (Zhou et al., *Nature* 2023; [EPMC](https://www.ebi.ac.uk/europepmc/webservices/rest/PMC10550819/fullTextXML)) | Warmup to 5×10⁻⁴, cosine to 1×10⁻⁶ | not stated in the text | not stated in the text | **50 / 10 warmup** | "used" | not stated | Batch 16; checkpoint chosen by best validation AUROC | FT |
| **RETFound code defaults** ([main_finetune.py](https://raw.githubusercontent.com/rmaphoh/RETFound_MAE/main/main_finetune.py)) | blr 5e-3 × batch/256 | **0.65** | **0.2** | 50 / 10 | **0.1** | **0 / 0** | RandAugment `rand-m9-mstd0.5-inc1`, random erasing 0.25, weight decay 0.05, AdamW | FT |
| **RETFound repo README** ([GitHub](https://github.com/rmaphoh/RETFound_MAE)) | — | — | — | 50 | — | — | Batch 24, 224 px; now also supports DINOv2/DINOv3 backbones (ViT-S/14 included) | FT |
| **MAE fine-tuning** (He et al., CVPR 2022, Table 9; [arXiv HTML](https://arxiv.org/html/2111.06377v3)) | base 1e-3, batch 1024 | 0.75 | 0.1 (B/L), 0.2 (H) | 100 (B), 50 (L/H) / 5 | 0.1 | **0.8 / 1.0** | RandAug (9, 0.5), weight decay 0.05 | FT |

**How the team's recipe compares.** Layer decay 0.8, drop path 0.1, label smoothing 0.05, 20 epochs, OneCycle: all within the published ranges. The differences from RETFound are shorter training, weaker label smoothing and no RandAugment. **I found no external-test evidence that switching to the RETFound recipe would help.**

RETFound's own external DR result: trained on APTOS-2019 and tested on IDRiD, AUROC 0.822; on MESSIDOR-2, 0.738 (FT). Absolute values only; the baselines sit in a supplementary table I did not read.

### C2. DINOv2 fine-tuning practice

- DINOv2 on radiology: >200 evaluations covering kNN, linear probing, few-shot, PEFT and full fine-tuning ([arXiv 2312.02366](https://arxiv.org/abs/2312.02366), PP). The abstract does not say which adaptation strategy wins (AB).
- Natural-image DINOv2 is competitive with retina-specific models on graded fundus tasks, while RETFound-DINOv2 wins on oculomics and systemic tasks (search snippets of [arXiv 2502.06289](https://arxiv.org/pdf/2502.06289) and [arXiv 2511.22001](https://arxiv.org/html/2511.22001v1); AB, PP). This is consistent with the team's finding that RETFound-Green gave no gain.
- **LoRA vs full fine-tuning under shift:** mixed. Several 2025–26 preprints find full fine-tuning more robust to far-OOD shift (AB, PP; search snippets).

### C3. Robust fine-tuning and weight averaging

| Method | Evidence | Tag |
|---|---|---|
| **LP-FT** (train the linear probe first, then fine-tune everything) | Kumar et al., ICLR 2022: across 10 shift datasets, full fine-tuning was ~2% better than linear probing in-distribution but **~7% worse OOD**; **LP-FT was +1% ID and +10% OOD vs full fine-tuning**. Natural images; no fundus test found. | AB |
| **SWAD** (dense stochastic weight averaging) | Cha et al., NeurIPS 2021: **+1.6pp** average over 5 DomainBed benchmarks. Used inside DRGen (Fishr+SWAD) for fundus: GDRBench DG average AUC **79.3 vs ERM 75.9** (FT table; DRGen paper AB). | AB / FT |
| **EMA of weights** (Ensemble of Averages) | Arpit et al., NeurIPS 2022: a parameter moving average improves DG and model selection. EoA averages 68.0% on DomainBed (ResNet-50), **~4pp over ERM without averaging**. "Hyperparameter-free", and cheaper than SWAD. | AB |
| **WiSE-FT** (interpolate fine-tuned and pretrained weights) | Wortsman et al., CVPR 2022: +4–6pp under shift for CLIP and other zero-shot models. It needs a zero-shot head, so it does not apply directly to DINOv2. The analogous move is interpolating the LP-only and fully fine-tuned weights *(inference)*. | AB |

### C4. Sharpness-aware minimisation (SAM)

- Hassan et al., *IEEE Access* 2025, breast ultrasound: plain SAM consistently improved generalisation, but adaptive SAM did not help ViTs. This was in-distribution (AB).
- Anand et al., ISBI 2022: reports SAM gains of roughly 5–10% on cardiac ultrasound and COVID X-ray under test-time shift (AB).
- **No fundus external study found.**
- Cost: two forward/backward passes per step, so ~2× training time *(estimate)*.

### C5. Knowledge distillation

- The only fundus external evidence found: a dual-teacher (CNN + ViT) distillation into MobileNetV3 raised zero-shot Messidor-2 QWK **0.507→0.728** (IJCAI 2026; [PDF](https://www.ijcai.org/proceedings/2026/0774.pdf); AB, search snippet).
- A student rarely beats its teacher. The team's ensembles already gave no gain, so distilling them is unlikely to help *(inference)*.

### C6. Semi-supervised learning (FixMatch, pseudo-labelling)

- **No fundus study found** testing FixMatch with an external test.
- A teacher–student method with SSL pretraining reached 0.94 AUC on EyePACS and 0.89 on Messidor-2 using 2% of labels (Arrieta et al., per a search snippet; AB). It had no supervised-only external baseline in the snippet.
- **Known failure mode:** confirmation bias from noisy pseudo-labels, which is worst exactly where Stage 1 recall is 0.5. The team's mean-teacher run on target photos already found no gain.

### C7. Self-supervised continued pretraining on in-domain unlabelled images

| Study | Evidence | Tag |
|---|---|---|
| **DeepAdapter** (Zhao et al., *Br J Ophthalmol* 2026, doi 10.1136/bjo-2025-328559; [EPMC record](https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=DeepAdapter%20AND%20prematurity&format=json&resultType=core)) | Stage 1: SSL on **500,000 unlabelled infant fundus photos** from 4 Chinese centres. Stage 2: 39,456 labelled images plus an unsupervised domain-adaptation module. **External accuracy 0.828 vs 0.739** for the supervised model (p<0.001); cross-ethnicity public set 0.839 vs 0.813 (p=0.014); internal both 0.989. The task is **binary Normal vs ROP, not staging**. The abstract does not name MoCo-v2 or separate the SSL contribution from the domain-adaptation contribution. Paywalled. | AB |
| **Azizi et al.**, ICCV 2021 (MICLe) | Further SSL on unlabelled medical images after ImageNet. **MICLe uses two images of the same patient case as a positive pair**. Reported +6.7% top-1 in dermatology and +1.1% AUC in chest X-ray. The authors say self-supervised models "generalize better to distribution shifts". Relevant because ROP exams contain several photos per eye. | AB |
| **REMEDIS** (Azizi et al., *Nat Biomed Eng* 2023; includes a fundus DME task) | Matched strong supervised baselines with "1% to 33%" of target-site retraining data; up to 11.5% relative in-distribution gain. I could not retrieve the DME-specific OOD numbers. | AB |
| **DINORET / Block-Expanded DINORET** ([arXiv 2409.17332](https://arxiv.org/abs/2409.17332), PP) | Continued DINOv2 training on public fundus photos. Competitive with RETFound and better at few-shot. Adult DR and glaucoma only; no external numbers in the abstract. | AB |

### C8. Fundus competition tricks and whether they transfer

| Trick | Source evidence | Transfer to this team |
|---|---|---|
| **Ben Graham preprocessing** (rescale to a fixed radius, subtract a Gaussian-blurred local mean, clip to 90% radius) | Graham 2015 (FT). | Not tested across sites in any paper found. The team's CLAHE pipeline already does local contrast normalisation. |
| **Regression + optimised thresholds** | Graham used a random forest on CNN outputs, thresholded at {0.57, 1.37, 2.30, 3.12} (FT). The APTOS 2019 winner used **L1 regression**, GeM pooling, 88,702 external 2015 images and a multi-input-size ensemble, scoring 0.936 private QWK (as described by Fang et al. 2023, FT). | The team's ordinal decoder already gave no gain. **AP-ROP is not on the stage axis**, so a single regression output cannot represent the 6-class ICROP-3 label. A 2026 ROP paper found ordinal and regression models track expert severity rankings better than multi-class (Hoebel et al., *PLOS Digit Health*; AB). That concerns ranking, not external accuracy. |
| **High resolution** | Graham: "Increasing the scale beyond 270 [radius] slowed down the processing without any apparent benefit" (FT). | The team's 518 px run gave no gain, consistent with this. |
| **Ensembling** | Graham: "Combining the models seemed to offer very limited benefit" (FT). | Consistent with the team's result. |
| **Image metadata as features** | Graham fed **original image size and variance**, plus the other eye's prediction, into the random forest (FT). | **Do not copy the size feature.** In the team's data, image dimensions alone score AUC 0.911, i.e. a device shortcut. The other-eye idea maps onto the team's exam-level bag model. |
| **External data** (2015 EyePACS for APTOS) | The biggest single lever in the APTOS solutions (FT, via Fang). | Maps to section D. |

### C9. Label-noise handling

The team suspects label noise is the ceiling. Evidence:

- **ROP stage disagreement.** Campbell et al. 2016, *Ophthalmology* ("Diagnostic discrepancies in ROP classification", PMID 27238376): "the most common area of discrepancy in ROP classification is stage", while agreement on type 1/type 2 disease is high (AB, search snippet).
- **ROP stage models do not travel.** Chen, Coyner et al. 2021, *Ophthalmology Retina* 5(10):1027: binary stage detectors reached AUROC 0.99 (North America, RetCam) and 0.97 (Nepal, 3nethra). Across populations this fell to **AUROC 0.62, AUPRC 0.36, sensitivity 44%**. Labels came from 1–3 graders (AB, search snippet).
- **Adjudication helps.** Krause et al. 2018, *Ophthalmology* 125:1264 ([arXiv 1710.01711](https://arxiv.org/abs/1710.01711)): "a small set of adjudicated DR grades allows substantial improvements". News coverage reports moderate-or-worse DR AUC improving **0.934→0.986** with adjudicated labels ([Healio](https://www.healio.com/news/ophthalmology/20180314/human-adjudication-of-dr-grading-enhances-machine-learning-algorithm); AB/news).
- **Label smoothing under noise.** Lukasik et al., ICML 2020: label smoothing is "competitive with loss-correction under label noise" (symmetric noise), and smoothing the teacher helps distillation from noisy data (AB). RETFound and MAE both default to 0.1; the team uses 0.05.
- **Review of noisy-label methods** for medical imaging: Karimi et al. 2020, *MedIA* 65:101759 (AB).

---

## D. Getting more real data

### D1. Public ROP releases the team does not already list

| Dataset | Origin / device | Size and labels | Licence / access | Usefulness | Tag |
|---|---|---|---|---|---|
| **Macretina** (Trivedi, Srivastava, Mahajan; *Sci Rep* 16:1993, Dec 2025; [EPMC](https://www.ebi.ac.uk/europepmc/webservices/rest/PMC12808750/fullTextXML)) | **Indore, India**; 3nethra Neo (Forus), 120° FOV; 2040×2040 px | 1,432 images from 112 infants. Labelled subset of 973: **ridge present (stages 1–3 pooled) 250, A-ROP 248, normal 215**, plus 260 optic-disc-only images. No Stage 4/5, no plus labels, no patient or eye IDs. Expert κ = 0.89. | CC BY-NC-ND 4.0. Figshare DOI 10.6084/m9.figshare.29605802. The data statement says "will be made publicly available… after publication" while the introduction says it already is; **I could not confirm the file is live**. | **A-ROP (+248)** and a new camera/country domain. Ridge images need re-staging before use as stage labels. Ridge masks would enable ridge-aware training. | FT |
| **Hugging Face `lijuanliao/ROP-DATASET`** ([card](https://huggingface.co/datasets/lijuanliao/ROP-DATASET)) | Chinese; device not stated | 1K–10K images (2.1 GB). Classes: **Normal, Stage 1–4, A-ROP**, with Excel tables for acquisition view, annotation, metadata and pathology. Paper "under review". | CC-BY-NC-4.0; appears ungated | Could add **Stage 4** (the team has 89 Stage 4/5). **Check for overlap with Multi-View ROP / ROP-VL / Shenzhen by image hashing before use**; the per-view table suggests it may be the Multi-View ROP set. | FT (card only) |
| Roboflow "retinopathy-of-prematurity" (sriyush; [link](https://universe.roboflow.com/sriyush/retinopathy-of-prematurity)) | — | Normal, Stage 1–3, laser scars (2024) | CC BY 4.0 | Same label set as the Shenzhen/Zhao 2024 set, so **probably a re-upload**. | AB |
| ROPDB (Univ. Sfax; [Mendeley](https://data.mendeley.com/datasets/29w6jd6tvs/2)) | — | **Metadata only; "Files not available"** | CC BY 4.0 | None | FT (page) |

**No 2026 public multi-centre ROP data descriptor with stage labels was found** in Scientific Data, Data in Brief, Mendeley, figshare or Hugging Face searches.

### D2. Non-public data that could be requested

- **LAN-RoP** (First Hospital of Lanzhou Univ.; *Front Pediatr* 2026, [link](https://www.frontiersin.org/journals/pediatrics/articles/10.3389/fped.2026.1765353/full)): 1,002 RetCam images from 52 infants. **Plus-disease labels only** (normal / pre-plus / plus; 3 graders). Data "available without undue reservation" on request (FT).
- **Kraków RetCam set:** 18,690 photos from 243 patients and 970 sessions. Private (search snippet of [arXiv 2607.09164](https://arxiv.org/pdf/2607.09164); AB, PP).

### D3. Indian sources

| Source | Status | Tag |
|---|---|---|
| **Macretina** (Indore) | New in Dec 2025; see D1 | FT |
| **HVDROPDB** (Desai Eye Hospital, Pune; RetCam + Neo) | Already used by the team | — |
| **KIDROP** (Karnataka tele-ROP, Narayana Nethralaya) | Used in a 2023 *Front Pediatr* study with per-camera (RetCam / NEO) stage counts ([link](https://frontiersin.org/articles/10.3389/fped.2023.1197237/pdf)). **No public release found.** Collaboration only. | AB |
| **Aravind Eye Care System** | Used as an external validation set for i-ROP (5,635 exams from 2,699 infants; *JAMA Ophthalmol* 2024, [link](https://jamanetwork.com/journals/jamaophthalmology/fullarticle/2816005)). Not public. | AB |

### D4. Why more real data is the right lever

- RADR: adding cameras beat every augmentation policy tested (FT).
- Chen 2021: ROP stage models collapse across populations (AB).
- APTOS 2019: the winning solutions relied on external 2015 data (FT, via Fang).
- **Run every candidate dataset through leave-one-hospital-out evaluation.** Pooled cross-validation would hide the device shortcut.

---

## Ranked table

Ordered by expected value for this team at an unseen hospital. T4 costs are *(estimates)* relative to ~33 min per fold.

| # | Technique | Evidence strength for external generalisation | Expected cost on a Kaggle T4 | Risk | Recommended for this team? |
|---|---|---|---|---|---|
| 1 | **Adjudicate and re-label Stage 1 and borderline stage images** (2–3 graders on a few hundred images, especially validation and target-hospital images); use the adjudicated set for model selection | Moderate. DR adjudication: Krause 2018 (AB). ROP stage is the main disagreement axis: Campbell 2016 (AB). | ~0 GPU; grader hours | Low. Clinician time is the bottleneck. | **Yes** |
| 2 | **Add real source domains** (Macretina A-ROP and normal; HF ROP-DATASET Stage 4/A-ROP after dedup) and **evaluate leave-one-hospital-out** | Moderate–strong. RADR (FT); APTOS external data (FT); Chen 2021 collapse (AB). | +~15–30% per fold per added 1–2k images, plus extra folds for leave-one-hospital-out | Medium: duplicates across sets, label-scheme mismatch (ridge is pooled stages 1–3), NC-ND licence | **Yes** |
| 3 | **Weight averaging: EMA, then SWA/SWAD over the last epochs** | Moderate. DomainBed: SWAD +1.6pp, EoA ~+4pp over no-averaging ERM (AB). Fundus: DRGen 79.3 vs 75.9 AUC on GDRBench (FT). | ~0 (EMA); <5% (SWA) | Low | **Yes** |
| 4 | **LP-FT** (head-only warm-up, then full fine-tune; optionally interpolate LP and FT weights) | Moderate (natural images; Kumar 2022, +10% OOD vs FT; AB). No fundus test found. | +~5–10% (a few head-only epochs) | Low | **Yes** |
| 5 | **Fundus-realistic photometric + degradation augmentation** (FundusAug-style halo, holes, spots, blur, sharpness) **and full 360° rotation; ablate the hue/colour-jitter strength** | Moderate but mixed. GDRNet +4.5 AUC on unseen domains (FT); RADR ColorJitter −4.6 to −5.8 QWK (FT). 360° rotation: practice only (Graham, FT). | ~0 GPU; possible CPU dataloader slowdown | Medium: wrong colour strength can hurt | **Yes, as a leave-one-hospital-out ablation** |
| 6 | **In-domain SSL continued pretraining** (DINOv2 objective, or MICLe-style same-exam positive pairs) on all unlabelled infant photos including target-hospital photos | Moderate. DeepAdapter external 0.739→0.828, binary task (AB); MICLe/REMEDIS OOD gains (AB). | High: multi-hour to multi-day GPU budget; no T4 figure in any source | Medium: catastrophic forgetting, budget overrun, gain may be binary-only | **Yes, as a stretch goal**, only after 1–5 |
| 7 | **Label smoothing 0.1** (from 0.05) | Weak–moderate. Robust under symmetric noise: Lukasik 2020 (AB). RETFound/MAE default (FT). | 0 | Low | **Yes** (trivial) |
| 8 | **Test-time augmentation** (flips and rotations at inference) | Weak. Practice only (Graham, FT); no fundus external ablation. | Inference ×k only | Low | **Optional** (no retraining needed) |
| 9 | **Balanced-MixUp** for Stage 4/5 | Weak. +0.6 quad-κ, internal only (FT). | ~0 | Low | Optional |
| 10 | **FDA / amplitude mixing using unlabelled target photos as style sources** | Weak. Segmentation only: FedDG +2.02% Dice (AB). | ~0 GPU | Low–medium | Optional |
| 11 | Domain-adversarial training (DANN) | Weak–moderate. RADR +2.0 QWK on internal cameras, comparable to augmentation on external sets (FT). | +~20% | Medium: unstable; conflicted with augmentation in RADR | No (for now) |
| 12 | Sharpness-aware minimisation (SAM) | Weak. No fundus external test (AB). | ~2× per fold | Low | No |
| 13 | Plain mixup / CutMix | None. GDRBench: Mixup 75.0 vs ERM 75.9 (FT). | 0 | Low | No |
| 14 | RandAugment / TrivialAugment | Weak or mixed. Fang: no gain for DeiT (FT); Ktena: harmed OOD fairness (FT). | 0 | Medium | No |
| 15 | FixMatch / pseudo-labelling on target photos | Weak. No fundus external test; mean-teacher already failed. | 2–3× | High: confirmation bias on Stage 1 | No |
| 16 | Knowledge distillation from ensembles | Weak. One IJCAI 2026 snippet (AB); team ensembles gave no gain. | +1 run | Low | No |
| 17 | Longer schedules, progressive resizing, higher resolution | None. Graham: no gain above radius 270 (FT); team's 518 px run: no gain. | 1.5–3× | Low | No |
| 18 | Kaggle DR tricks: Ben Graham preprocessing, regression + thresholds, big ensembles | None for cross-site. Equivalent ordinal decoder and ensembles already tried; AP-ROP is off-axis. | 0–3× | Low; **the image-size feature is a known shortcut** | No |
| 19 | Ridge copy-paste synthesis | Weak. DR anomaly detection only (AB). | Medium (engineering) | High: label validity, paste-seam shortcuts | No |
| 20 | **GAN / diffusion synthetic images** | Weak/none for external. Nderitu 2025: external −0.039 at ×2 (FT); EyeDiff: no gain on APTOS, MESSIDOR-2 or PAPILA (FT); no ROP stage study exists. | Very high: generator training per fold plus sampling plus human QC | **High**: memorisation (up to 54.7% at 5k images, FT), loss of subtle lesions, label-noise amplification, test leakage | **No** |

---

## Verdict on synthetic data generation

Do not generate synthetic training images for this project. No study shows synthetic images improving ROP stage accuracy. The best-controlled fundus study with an external test found that internal gains did not carry over to the unseen site (FT). The generators fail on fine lesions like the Stage 1 demarcation line (FT). And with only 89–600 images in the classes that need help, memorisation risk is at its highest (FT).
