# What produces the image → eye gap

*Verification pass 2, both halves, structured seed-42, 5 checkpoints. Tables:
[`aggregation_sensitivity.csv`](../20_AUDIT/verification_pass_2/08_EYE_LEVEL_ANALYSIS/aggregation_sensitivity.csv),
[`view_stratified_performance.csv`](../20_AUDIT/verification_pass_2/11_AGGREGATION_AND_VIEW_ANALYSIS/view_stratified_performance.csv),
[`eye_level_clustered_ci.csv`](../20_AUDIT/verification_pass_2/08_EYE_LEVEL_ANALYSIS/eye_level_clustered_ci.csv).
Existing figures [`aggregation_comparison.png`](aggregation_comparison.png) and
[`region_qwk_comparison.png`](region_qwk_comparison.png) carry the same result under the
5-class-drop QWK convention; the tables below use `qwk_6class` (as the project reports it)
with `qwk_5class_map` beside it.*

## The claim under test

`docs/FIGURE_GUIDE.md` and paper figure 10: *"The same predictions score QWK 0.52 per image
and 0.80 per eye. The unit was the problem, not the model."*

## 1. The gap is produced by one operator

Locked half, identical predictions, only the eye-aggregation rule changed:

| Rule | n eyes | QWK (6-class) | QWK (5-class map) | Accuracy | Macro-F1 |
|---|---:|---:|---:|---:|---:|
| **max over views** (protocol) | 230 | **0.8012** | 0.8225 | 0.6461 | 0.5619 |
| mean class, rounded | 230 | 0.6045 | 0.6042 | 0.4504 | 0.2848 |
| median | 230 | 0.5246 | 0.5099 | 0.4678 | 0.3821 |
| majority vote | 230 | 0.4898 | 0.4676 | 0.4991 | 0.4181 |
| mean probability, argmax | 230 | 0.4825 | 0.4615 | 0.4817 | 0.3885 |
| *image level, no aggregation* | *661* | *0.5232* | *0.5165* | *0.5204* | *0.4116* |

Under majority vote the eye-level score (0.49) is **at or below** the image-level score
(0.52). Only `max` produces the headline. A report quoting "0.80 per eye" without naming the
operator is quoting a selected number.

## 2. But `max` is not doing the work either — the view is

Image level, **no aggregation at all**, stratified by camera view:

| | Locked, n | QWK (6-class) | Dev, n | QWK (6-class) |
|---|---:|---:|---:|---:|
| **temporal frames only** | 227 | **0.7993** | 229 | **0.7182** |
| nasal only | 221 | 0.3746 | 218 | 0.2727 |
| posterior pole only | 213 | 0.3977 | 216 | 0.2808 |
| all frames | 661 | 0.5232 | 663 | 0.4049 |
| *eye level, max over views* | *230* | *0.8012* | *230* | *0.7206* |

**Single temporal frames, scored one by one with no aggregation, reproduce the eye-level
number to within 0.002 on the locked half and 0.003 on the dev half.** Taking the max over
six views is, to that precision, an expensive way of selecting the temporal frame.

Both halves are patient-disjoint and the dev half replicates the locked half, so this is not
a single-split coincidence. The pattern also holds for the flat baseline, so it is not a
property of the contributed head.

## 3. Corrected mechanism

The gap is a **view/label mismatch**, not an evaluation-unit effect:

- The label is a property of the eye — [436 of 460 eyes carry one label across all
  views](../02_DATASET/LABEL_GRANULARITY_ANALYSIS.md).
- The finding that label describes is visible in some views and not others. Frames of
  diseased eyes called Normal: temporal 12.7% (n=140), **nasal 63.4% (n=141)**, posterior pole
  71.7% (n=132), over stage-1–3 frames on the locked half.
- Scoring a posterior-pole frame against an eye-level label asks the model a question that
  frame may not contain the answer to.

**Correction to the caption of paper figure 11.** It reads *"The ROP ridge grows in the
retinal PERIPHERY. Temporal frames look at it; posterior-pole frames look away."* The figure
itself plots a nasal bar at 63.4% — and nasal retina is peripheral. The evidence supports
**temporal specifically**, consistent with ROP's temporal predilection, not periphery versus
pole. The figure is correct; the caption over-simplifies it.

## 4. What to report instead

1. **Primary:** view-stratified image-level performance. It cannot be attacked on aggregation
   grounds, needs no operator choice, and replicates across two patient-disjoint halves.
2. **Secondary:** the eye-level figure, with the operator named and the alternatives tabulated
   as above.
3. Never the eye-level number alone.

## 5. Eye-level uncertainty — now computed

The headline never carried an interval. `locked_table.json` stores only checkpoint SD
(≈0.015–0.028), which is *not* statistical uncertainty. Patient-clustered bootstrap, 2,000
replicates, resampling the 116 infants (redrawn patients re-tagged so duplicate eyes do not
collapse):

| Dataset | Unit | Metric | Point | 95% CI | Width |
|---|---|---|---:|---|---:|
| locked | image | QWK (6-class) | 0.5232 | [0.4330, 0.5944] | 0.161 |
| locked | eye (max) | QWK (6-class) | 0.8012 | **[0.7304, 0.8568]** | 0.126 |
| locked | eye (max) | QWK (5-class map) | 0.8225 | [0.7561, 0.8726] | 0.117 |
| locked | eye (max) | accuracy | 0.6461 | [0.5775, 0.7117] | 0.134 |
| locked | eye (max) | macro-F1 | 0.5619 | [0.4814, 0.6511] | 0.170 |
| dev | eye (max) | QWK (6-class) | 0.7206 | [0.6383, 0.7928] | 0.155 |

Note the company the 0.80 keeps: at the same unit and rule, **accuracy is 0.646** and
**macro-F1 is 0.562**. QWK is generous to an ordinal predictor that is usually within one
step; it should never be quoted alone.

## 6. The number that does not improve

`stage123_eyes_called_normal = 0.1782` (131 of 735 eye-predictions over 5 checkpoints;
denominator = **stage-1–3 eyes only**, n=147 per checkpoint). Including Stage 4/5 and AP-ROP
eyes gives 0.1712. After the best aggregation rule this project has, roughly **18% of
diseased eyes are still called Normal**. That is the number a clinical reader should be given
alongside the 0.80.
