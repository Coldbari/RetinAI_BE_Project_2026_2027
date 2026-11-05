# What the label actually describes

*Verification pass 2, recomputed from `data/rop/multiview/ROP-DATASET/Dataset Annotation Table.xlsx`
(1,324 rows) joined to `Dataset Acquisition View Table.xlsx`. Raw output:
[`20_AUDIT/verification_pass_2/02_DATASET/label_granularity_raw.json`](../20_AUDIT/verification_pass_2/02_DATASET/label_granularity_raw.json).*

The Multi-View corpus ships one label per **image**. It is not an image-level label in practice.

| | Count |
|---|---:|
| Images | 1,324 |
| Patients | 232 |
| Eyes (patient × side) | 460 |
| **Eyes carrying one label across all their views** | **436 / 460 (94.8%)** |
| Eyes with more than one distinct label across views | 24 |
| Patients carrying one label across all images | 210 / 232 (90.5%) |

Label values map exactly onto the six analysis classes, and the totals reconcile with the
dev + locked split counts with no residual:

`Normal 382 · 1 → Stage 1 295 · 2 → Stage 2 299 · 3 → Stage 3 312 · 4 → Stage 4/5 18 · AROP 18` = 1,324.

## Why it matters

Because the label is a property of the eye, an image-level metric scores frames that do not
contain the finding against a label that describes it. That is not a defect of the model; it
is a property of the annotation. It is also why the eye is a legitimate co-primary unit — and
why *which* aggregation rule is used then decides the headline number
([see the aggregation analysis](../11_AGGREGATION_AND_VIEW_ANALYSIS/aggregation_mechanism_analysis.md)).

## Correction to an earlier statement

An earlier pass of this audit reported **446/460** single-label eyes and 216/232 single-label
patients. Those figures came from coercing the `Label` column to numeric, which silently
turned the string labels `"Normal"` and `"AROP"` into NaN, and `nunique()` ignores NaN — so
eyes whose views split between `Normal` and a numeric stage were counted as single-label.
The corrected figures are **436/460 eyes** and **210/232 patients**.

## What cannot be established from these tables

Whether a given frame contains a visible ridge. The Multi-View tables carry view, label,
demographics and per-patient zone/stage/plus — **no lesion location and no segmentation**.
The only ridge masks in the project belong to HVDROPDB, excluded at layer 1 for having no
stage label. Any statement that a missed frame "genuinely contains no disease" requires a
human grader and is not supported by any committed artifact.
