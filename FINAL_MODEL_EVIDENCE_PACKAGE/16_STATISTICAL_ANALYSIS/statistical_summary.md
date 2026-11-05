# Statistical Summary

*Methods: [`statistical_methods.md`](statistical_methods.md) · Tables: [`confidence_intervals.csv`](confidence_intervals.csv), [`bootstrap_results.csv`](bootstrap_results.csv), [`paired_tests.csv`](paired_tests.csv)*

## Internal, pooled out-of-fold (n = 6,700 images, 1,528 infants)

| metric | point | 95% CI (patient-clustered) | width | fold SD | fold SEM |
|---|---|---|---|---|---|
| accuracy | 0.7861 | [0.6970, 0.8574] | 0.160 | 0.0831 | 0.0371 |
| balanced accuracy | 0.6524 | [0.6156, 0.7569] | 0.141 | 0.0440 | 0.0197 |
| macro-F1 (6 classes) | 0.6517 | [0.5718, 0.7602] | 0.188 | 0.0861 | 0.0385 |
| **QWK (corrected)** | **0.8969** | **[0.8692, 0.9233]** | **0.054** | 0.0288 | 0.0129 |

The corrected QWK is by some distance the **most stable** metric available on this corpus — CI width 0.054 against 0.19 for macro-F1. That is a direct consequence of the correction: under the legacy 6-class definition, QWK's fold SD was 0.145 (5× larger) because the metric was largely tracking each fold's AP-ROP prevalence.

## External locked, eye level (230 eyes, 116 infants)

| metric | point | 95% CI | width |
|---|---|---|---|
| QWK (corrected) | 0.8394 | [0.7683, 0.8903] | 0.122 |
| accuracy | 0.6783 | [0.6043, 0.7468] | 0.142 |
| macro-F1 (present) | 0.6160 | [0.5277, 0.7181] | 0.190 |
| diseased eyes called Normal | 0.1769 | [0.1111, 0.2548] | 0.144 |

## Per-class intervals — where the evidence runs out

| class | metric | point | 95% CI | width | verdict |
|---|---|---|---|---|---|
| Normal | recall | 0.951 | [0.936, 0.965] | 0.028 | solid |
| Normal | precision | 0.929 | [0.877, 0.967] | 0.090 | solid |
| Stage 1 | recall | 0.499 | [0.426, 0.593] | 0.167 | wide |
| Stage 1 | precision | 0.376 | [0.273, 0.487] | 0.214 | wide — **and the whole interval is poor** |
| Stage 2 | recall | 0.631 | [0.568, 0.704] | 0.136 | usable |
| Stage 2 | precision | 0.559 | [0.410, 0.712] | **0.302** | **very wide — do not headline** |
| Stage 3 | recall | 0.713 | [0.577, 0.843] | 0.267 | wide |
| Stage 3 | precision | 0.758 | [0.643, 0.865] | 0.222 | wide |
| Stage 4/5 | recall | 0.775 | [0.677, 0.861] | 0.184 | wide — 89 images, one source |
| Stage 4/5 | precision | 0.663 | [0.465, 0.925] | **0.460** | **very wide — do not headline** |
| **AP-ROP** | **recall** | **0.345** | **[0.254, 0.836]** | **0.582** | **very wide — this interval IS the finding** |
| AP-ROP | precision | 0.899 | [0.768, 0.960] | 0.192 | wide |

> **The AP-ROP recall interval spans 0.25 to 0.84.** That is not noise to be averaged away —
> it is the label-unit problem expressed as statistics. 476 of the 618 AP-ROP validation
> frames sit in fold 0, and 470 of those come from a **single infant** whose diagnosis is a
> per-session code stamped on every frame. Resampling infants therefore either includes that
> infant (recall ≈ 0.25) or excludes them (recall ≈ 0.84). A point estimate of "0.345" for
> AP-ROP recall should never be quoted without this interval.

## Architecture comparison

| metric | seed | structured | flat | paired difference | paired *t* | Wilcoxon |
|---|---|---|---|---|---|---|
| macro-F1 | 42 | 0.6923 | 0.6977 | **−0.0055** | p = 0.589 | p = 0.813 |
| macro-F1 | 1337 | 0.7079 | 0.6855 | **+0.0224** | p = 0.494 | p = 1.000 |
| QWK (corrected) | 42 | 0.8964 | 0.8962 | **+0.0002** | p = 0.977 | p = 1.000 |
| QWK (corrected) | 1337 | 0.9045 | 0.8967 | +0.0077 | p = 0.490 | p = 1.000 |
| accuracy | 42 | 0.7956 | 0.7973 | −0.0017 | p = 0.880 | p = 1.000 |
| accuracy | 1337 | 0.8122 | 0.7937 | +0.0185 | p = 0.348 | p = 0.625 |

**Three independent noise floors, all larger than the effect being tested:**

| noise source | magnitude | vs the 0.0055 architecture difference |
|---|---|---|
| seed change, same architecture | 0.018 (structured) / 0.043 (flat) mean \|Δ\| per fold | **3–8×** |
| last-5-epoch SD within one run | 0.007 – 0.056 | **1–10×** |
| fold-to-fold SD | 0.086 – 0.099 | **16–18×** |

## What the statistics support

- **Equivalence between the two heads on accuracy-family metrics** — supported, and stronger under the corrected QWK than it was before.
- **Ordered severity agreement internally** — QWK 0.897 [0.869, 0.923] is a solid, tightly-bounded result.
- **The external drop is real** — non-overlapping by a wide margin (internal QWK CI lower bound 0.869 vs external point 0.493).
- **The eye-level external result is real but bounded** — CI [0.768, 0.890] excludes the image-level value.

## What the statistics do NOT support

- Any claim about AP-ROP recall as a point estimate (CI width 0.58).
- Any claim about Stage 4/5 or Stage 2 precision (CI widths 0.46 and 0.30).
- Any claim that one architecture outperforms the other on this corpus.
- Any external claim about stage 4/5 (n=4) or AP-ROP (n=12), regardless of what those cells say.
