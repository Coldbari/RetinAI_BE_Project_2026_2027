# User Guide — every term used in this package, in plain English

## The disease

**Retinopathy of prematurity (ROP)** is abnormal blood-vessel growth in the retina of a baby born too early. It is a leading *preventable* cause of childhood blindness. Babies are screened repeatedly in the neonatal unit, and treatment works only inside a narrow window.

**ICROP** is the international staging system:

| label | what it means |
|---|---|
| **Normal** | no disease |
| **Stage 1** | a faint flat white line where normal vessels stop |
| **Stage 2** | that line has become a raised ridge |
| **Stage 3** | abnormal vessels are growing out of the ridge |
| **Stage 4/5** | the retina is partly (4) or completely (5) detached. Merged here because both are rare |
| **AP-ROP** | "aggressive posterior ROP" — a fast, severe form that does **not** sit on the 1-2-3 ladder at all. This is why the model has a separate yes/no output for it |

**Where it grows matters:** the lesion appears at the **temporal** (outer) edge of the retina. A photograph aimed at the **posterior pole** (the centre, back of the eye) or the **nasal** (nose-side) periphery may simply not contain it.

## The measurements

| term | plain meaning |
|---|---|
| **Accuracy** | share of photographs where the model's label exactly matches the human's |
| **Balanced accuracy** | the same, but each category counts equally, so a huge "Normal" class cannot hide poor performance on rare ones |
| **Recall** (= sensitivity) | *Of the photographs that really are class X, how many did the model find?* High recall = it doesn't miss things |
| **Precision** (= PPV) | *When the model says class X, how often is it right?* High precision = it doesn't cry wolf |
| **F1** | a single number balancing precision and recall |
| **Macro-F1** | the average F1 across all six categories, each counting equally |
| **Specificity** | of the genuinely healthy eyes, how many did it correctly call healthy |
| **NPV** | when it says "healthy", how often that is true |
| **QWK** (quadratic-weighted kappa) | agreement with the human grade on an **ordered** scale. Being one stage off is penalised lightly; being four stages off, heavily. 0 = no better than guessing, 1 = perfect. It also corrects for agreement that would happen by chance |
| **AUC** | for a yes/no question, the chance the model scores a random diseased eye higher than a random healthy one. 0.5 = coin flip, 1.0 = perfect |
| **Average precision (AP)** | like AUC but focused on how clean the positive predictions are; more informative when one class is rare |
| **Confusion matrix** | a grid: rows are what the photograph really was, columns are what the model said. The diagonal is "correct" |
| **Support** | how many photographs a number is based on. **Always check this** — a perfect score on 4 photographs means nothing |

## Confidence and calibration

| term | plain meaning |
|---|---|
| **Confidence** | the highest percentage the model assigns to any one category |
| **Calibration** | whether that percentage is honest. If a model says "90% sure" a thousand times, a *calibrated* model is right about 900 of them |
| **ECE** (Expected Calibration Error) | how far off that is, on average. 0 = perfectly honest. This model: **0.109** at home, **0.334** at a new hospital |
| **Brier score** | another accuracy-of-probabilities measure; lower is better |
| **Over-confidence** | stated confidence minus actual accuracy. This model: **+0.11** at home, **+0.33** at a new hospital |
| **Uncalibrated score** | the label this package applies to the model's output. It is useful for *ranking* cases but is not a probability of disease |

## How the model was tested

| term | plain meaning |
|---|---|
| **Cross-validation** | the data is split into 5 parts; the model is trained 5 times, each time holding one part out to test on. Every photograph gets tested by a model that never saw it |
| **Grouped by patient** | all photographs of one baby go into the same part. Without this, a model could "recognise the baby" instead of the disease |
| **Out-of-fold (OOF)** | predictions made on data the model was not trained on |
| **Held-out hospital** | a fourth hospital used for nothing during development |
| **Locked test set** | half of that hospital's data, sealed, scored **exactly once** under rules written down in advance, so nobody could tune until it looked good |
| **Checkpoint** | a saved copy of the model's weights at a point in training |
| **Confidence interval (95%)** | a range that would contain the true value 95% of the time if the study were repeated. Here it is computed by resampling **babies**, not photographs, because one baby can contribute hundreds of photographs |
| **Seed** | the random number that decides shuffling and initialisation. Changing only the seed changes results — by more than the difference between the two model designs tested here |

## How the model works

| term | plain meaning |
|---|---|
| **EfficientNetV2-S** | the neural network that looks at the picture. It was pre-trained on millions of everyday photographs, then re-trained on retinas |
| **Head** | the small final layer that converts the network's internal description into an answer |
| **Ordinal / CORN head** | a head built to know that the stages are in *order*, so "Stage 2 called Stage 3" is a smaller mistake than "Stage 2 called Normal" |
| **AP-ROP branch** | a separate yes/no output, because AP-ROP is not a point on that ladder |
| **Site adversary (DANN)** | a training-only trick that penalises the network for encoding *which hospital* a photo came from. Switched off when the model is used |
| **CLAHE** | a contrast-enhancement step that makes the faint peripheral ridge easier to see |
| **Letterboxing** | padding the image to a square with black bars instead of stretching it. Stretching would let the model guess the hospital from the shape of the distortion |
| **Grad-CAM** | the heat map showing which pixels most influenced the answer. **On this model it has been audited and is not reliable** |
| **TTA** | showing the model several flipped/rotated copies and averaging. **Not used** by this model when serving |
| **Aggregation** | combining the several photographs of one eye into a single answer. *Which rule you use changes the result a lot* — see figure 10 |
