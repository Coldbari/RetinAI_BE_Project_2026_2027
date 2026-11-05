# The human ceiling for ROP staging — and what it does to our numbers

Closes the open literature item: *what is published human inter-grader agreement for ROP
staging, and how should our kappa be read against it?* Answer, in one line: **ROP stage is
the least reliable part of the ICROP classification among human experts**, and this
reframes both our headline metric and the label-noise finding.

## The two citations

**[C1] Gschließer A, Stifter E, Neumayer T, Moser E, Papp A, Pircher N, Dorner G, Egger S,
Vukojevic N, Oberacher-Velten I, Schmidt-Erfurth U. "Inter-expert and Intra-expert
Agreement on the Diagnosis and Treatment of Retinopathy of Prematurity." *American Journal
of Ophthalmology* 2015;160(3):553–560. PMID 26004406, doi:10.1016/j.ajo.2015.05.016**

7 recognised ROP experts read 260 wide-field digital photographs from 52 patients, twice,
8 weeks apart.

| Agreement | κ | Interpretation |
|---|---|---|
| Inter-expert, **ROP stage** | **0.24** | fair |
| Inter-expert, plus disease | 0.32 | fair |
| Inter-expert, **AP-ROP** | 0.35 | fair |
| Inter-expert, necessity for treatment | 0.41 | moderate |
| Intra-expert, ROP stage | 0.56 | moderate — *the same expert, 8 weeks later* |
| Intra-expert, plus disease | 0.51 | moderate |

Inter-expert values are Fleiss' κ (multi-rater); intra-expert are Cohen's κ. **All
unweighted.**

**[C2] Campbell JP et al. "Diagnostic Discrepancies in Retinopathy of Prematurity
Classification." *Ophthalmology* 2016;123(8):1795–1801. PMID 27238376,
doi:10.1016/j.ophtha.2016.04.035**

281 infants across 7 centres, 1,553 expert-vs-expert comparisons:

- disagreement on **stage: 620/1553 (40%)** — the single largest source of discrepancy
- plus disease (incl. pre-plus): 18% · zone: 8% · overall ROP category: 40%
- **but** presence vs absence of **type 1 (treatment-requiring) disease: >95% agreement**

## What this does to our reporting

**1. We must report unweighted κ alongside QWK.** The human literature is unweighted;
quadratic weighting inflates ordinal agreement substantially because most errors are
adjacent. Computed on our committed pooled out-of-fold predictions:

| Structured model, pooled 5-fold (n=6,700) | unweighted κ | linear | quadratic (QWK) |
|---|---|---|---|
| seed 42 | 0.655 | 0.731 | 0.761 |
| seed 1337 | 0.681 | 0.754 | 0.783 |
| stage-only subset, seed 42 (n=6,058) | 0.701 | — | 0.897 |

**2. We must NOT claim superhuman agreement**, even though 0.66 > 0.24. Three reasons, all
of which a reviewer will raise:
- Fleiss' κ among 7 mutually-disagreeing experts is a different quantity from Cohen's κ
  between one model and one reference standard. Agreeing with a single institution's
  coding practice is an easier task than seven experts agreeing with each other.
- Our labels come from one hospital's diagnosis codes; the model has partly learned *that
  institution's labelling behaviour*, which is precisely why the held-out-hospital and
  adjudication work exist.
- Different image sets, different capture protocols, different class granularity.

The defensible sentence is: *"Our agreement with the reference standard (κ 0.66–0.68
unweighted) should not be read against published inter-expert agreement (κ 0.24 [C1]) as
if the tasks were comparable; we report it to show the reference standard itself is a
moving target."*

**3. It corroborates the label-noise finding rather than excusing it.** Our AP-ROP
session-level result argues the labels do not describe the frames they are attached to.
[C1] and [C2] independently establish that ROP staging labels are unreliable *even between
expert humans reading the same photograph* — 40% stage disagreement, and κ 0.56 for one
expert against their own earlier reading. So label noise in ROP staging is the field's
baseline condition, not a defect peculiar to our corpus.

**4. It sets the expectation for the clinician adjudication.** When our pediatric
ophthalmologist co-author re-reads ~300 images, we should NOT expect κ near 1.0 against
our recovered labels — the literature says two experts would not reach that with each
other. Plan to report the adjudication κ *against the [C1] inter-expert benchmark*, so a
moderate value reads as normal rather than as a failure of our corpus.

**5. It tells us which of our numbers matters most.** Humans agree on stage poorly (κ 0.24)
but on treatment-requiring disease almost perfectly (>95% [C2]). Metrics on the
clinically-decisive axis are therefore the ones with a trustworthy reference standard —
which is exactly the axis our AP-ROP session-level analysis measures (0.840 recall, 0.25%
false alarms). This is a strong argument for leading the paper with the AP-ROP/session
result and treating the 6-class QWK as secondary.

## Action items generated

- [ ] Add an unweighted-κ column to the CV results table in the draft (numbers above).
- [ ] Add [C1] and [C2] to the reference list; cite [C1] in Discussion §5, [C2] in Intro.
- [ ] Reframe the Discussion: staging labels are field-wide noisy, not corpus-specific.
- [ ] When adjudication happens, report κ against the [C1] benchmark.

Sources: [C1] [ajo.com](https://www.ajo.com/article/S0002-9394(15)00301-3/abstract) ·
[PubMed 26004406](https://pubmed.ncbi.nlm.nih.gov/26004406/) ·
[C2] [PubMed 27238376](https://pubmed.ncbi.nlm.nih.gov/27238376/)
