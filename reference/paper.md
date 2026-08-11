# References

IEEE format. Grouped by topic; the numbering is continuous and matches the
[README's reference list](../README.md#references).

How each of these fed into the design is written up in
[../docs/literarture_survey.md](../docs/literarture_survey.md).

---

## Clinical background

```text
[1]  V. Gulshan, L. Peng, M. Coram, M. C. Stumpe, D. Wu, A. Narayanaswamy, S. Venugopalan,
     K. Widner, T. Madams, J. Cuadros, R. Kim, R. Raman, P. C. Nelson, J. L. Mega and
     D. R. Webster, "Development and Validation of a Deep Learning Algorithm for Detection of
     Diabetic Retinopathy in Retinal Fundus Photographs," JAMA, vol. 316, no. 22,
     pp. 2402-2410, 2016.

[2]  International Committee for the Classification of Retinopathy of Prematurity, "The
     International Classification of Retinopathy of Prematurity Revisited," Archives of
     Ophthalmology, vol. 123, no. 7, pp. 991-999, 2005.
```

## Architectures

```text
[3]  M. Tan and Q. V. Le, "EfficientNetV2: Smaller Models and Faster Training," in Proc. 38th
     International Conference on Machine Learning (ICML), pp. 10096-10106, 2021.

[4]  K. He, X. Zhang, S. Ren and J. Sun, "Deep Residual Learning for Image Recognition," in
     Proc. IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pp. 770-778, 2016.
```

## Explainability

```text
[5]  R. R. Selvaraju, M. Cogswell, A. Das, R. Vedantam, D. Parikh and D. Batra, "Grad-CAM:
     Visual Explanations from Deep Networks via Gradient-Based Localization," in Proc. IEEE
     International Conference on Computer Vision (ICCV), pp. 618-626, 2017.
```

## Class imbalance and loss design

```text
[6]  T.-Y. Lin, P. Goyal, R. Girshick, K. He and P. Dollar, "Focal Loss for Dense Object
     Detection," in Proc. IEEE International Conference on Computer Vision (ICCV),
     pp. 2980-2988, 2017.

[7]  Y. Cui, M. Jia, T.-Y. Lin, Y. Song and S. Belongie, "Class-Balanced Loss Based on Effective
     Number of Samples," in Proc. IEEE/CVF Conference on Computer Vision and Pattern Recognition
     (CVPR), pp. 9268-9277, 2019.
```

## Calibration and statistics

```text
[8]  C. Guo, G. Pleiss, Y. Sun and K. Q. Weinberger, "On Calibration of Modern Neural Networks,"
     in Proc. 34th International Conference on Machine Learning (ICML), pp. 1321-1330, 2017.

[9]  X. Sun and W. Xu, "Fast Implementation of DeLong's Algorithm for Comparing the Areas Under
     Correlated Receiver Operating Characteristic Curves," IEEE Signal Processing Letters,
     vol. 21, no. 11, pp. 1389-1393, 2014.
```

## Datasets

```text
[10] E. Decenciere, X. Zhang, G. Cazuguel, B. Lay, B. Cochener, C. Trone, P. Gain, R. Ordonez,
     P. Massin, A. Erginay, B. Charton and J.-C. Klein, "Feedback on a Publicly Distributed
     Image Database: The Messidor Database," Image Analysis & Stereology, vol. 33, no. 3,
     pp. 231-234, 2014.

[11] P. Porwal, S. Pachade, R. Kamble, M. Kokare, G. Deshmukh, V. Sahasrabuddhe and
     F. Meriaudeau, "Indian Diabetic Retinopathy Image Dataset (IDRiD): A Database for Diabetic
     Retinopathy Screening Research," Data, vol. 3, no. 3, art. 25, 2018.

[12] A. Bajwa, G. A. P. Singh, R. Singh, M. I. Malik, M. Z. Afzal, A. Dengel and S. Ahmed,
     "G1020: A Benchmark Retinal Fundus Image Dataset for Computer-Aided Glaucoma Detection," in
     Proc. International Joint Conference on Neural Networks (IJCNN), pp. 1-7, 2020.

[13] J. I. Orlando, H. Fu, J. Barbosa Breda, K. van Keer, D. R. Bathula, A. Diaz-Pinto, R. Fang,
     P.-A. Heng, J. Kim, J. Lee et al., "REFUGE Challenge: A Unified Framework for Evaluating
     Automated Methods for Glaucoma Assessment from Fundus Photographs," Medical Image Analysis,
     vol. 59, art. 101570, 2020.

[14] Z. Zhang, F. S. Yin, J. Liu, W. K. Wong, N. M. Tan, B. H. Lee, J. Cheng and T. Y. Wong,
     "ORIGA-light: An Online Retinal Fundus Image Database for Glaucoma Analysis and Research,"
     in Proc. Annual International Conference of the IEEE Engineering in Medicine and Biology
     Society (EMBC), pp. 3065-3068, 2010.

[16] Kaggle, "Diabetic Retinopathy Detection (EyePACS)," 2015. [Online].
     Available: https://www.kaggle.com/c/diabetic-retinopathy-detection

[17] Kaggle, "APTOS 2019 Blindness Detection," 2019. [Online].
     Available: https://www.kaggle.com/c/aptos2019-blindness-detection
```

## Confounds and shortcut learning

```text
[15] A. J. DeGrave, J. D. Janizek and S.-I. Lee, "AI for Radiographic COVID-19 Detection Selects
     Shortcuts Over Signal," Nature Machine Intelligence, vol. 3, pp. 610-619, 2021.
```

Reference [15] is the one we leaned on hardest. It is the reason every headline number
here is reported twice: once pooled, and once controlled for the confound.

## Ordinal regression and domain-adversarial training (ROP staging)

```text
[18] X. Shi, W. Cao and S. Raschka, "Deep Neural Networks for Rank-Consistent Ordinal Regression
     Based On Conditional Probabilities," Pattern Analysis and Applications, vol. 26,
     pp. 941-955, 2023.

[19] Y. Ganin, E. Ustinova, H. Ajakan, P. Germain, H. Larochelle, F. Laviolette, M. Marchand and
     V. Lempitsky, "Domain-Adversarial Training of Neural Networks," Journal of Machine Learning
     Research, vol. 17, no. 59, pp. 1-35, 2016.
```

Reference [18] is the CORN head we use for ICROP staging. ROP stages are ordered, and a flat
softmax treats Stage 1 and Stage 4/5 as equally distant from Stage 2, which throws that
structure away.

Reference [19] is the gradient-reversal site adversary. Our own ablation qualifies it. Measured
with a naive site probe the adversary looks like it works, going from 0.859 to 0.820, but a
disease-controlled probe does not move at all, 0.882 to 0.885. It removed the disease-site
correlation rather than site appearance.

---

## Datasets used, and where

| Dataset | Disease | Images | Role |
|---|---|---|---|
| EyePACS | DR | ~35,126 | Training (pooled) |
| APTOS 2019 | DR | ~3,662 | Training (pooled) |
| Messidor-2 | DR | 1,744 | External validation, no retraining |
| IDRiD | DR | n/a | External validation plus resolution sensitivity |
| SMDG-19 | Glaucoma | ~12,449 | Training (multi-source aggregation) |
| G1020 | Glaucoma | 1,020 | Zero-shot failure case, then pooled into training |
| REFUGE1 / ORIGA | Glaucoma | 800 / 650 | Withheld-source zero-shot, 0.936 and 0.735 |
| Infant retinal database | ROP (binary) | 6,004 (188 patients) | Training and held-out test, split by patient |
| ROP staging corpus (4 sources) | ROP (6-class ICROP) | 3,112 (1,528 infants) | Training pool; one site held out as dev + locked |

No external ROP dataset exists publicly for the binary task, which is why binary ROP has no
external validation row. The staging corpus partially addresses this by consolidating four
sources and holding one out entirely. But that held-out site is where the 99.7% site
decodability was measured, so it is a generalisation test and a confound at the same time,
rather than a clean external set.

---

## A note on citation discipline

Where a number in this repository comes from a paper, we cite it. Where it comes from our own
measurement, it traces back to a file in the development repository's `results/` directory.
Nothing in the README is quoted from memory or estimated.
