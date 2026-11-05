# What this model gets wrong — in plain language

*Every number here is measured on the **locked test**: 661 photographs of 230 eyes from 116
infants at a hospital whose images the model never saw during training or development. It was
opened once, under a protocol written beforehand.*

---

## 1. Used on a whole examination, it refers every baby

A screening check looks at an infant, not at one photograph. A nurse photographs each eye from
several angles — about six pictures — and the tool raises a flag if any picture looks
abnormal.

Measured that way, on the locked test:

| What is being judged | Catches disease when present | Leaves healthy alone |
|---|---|---|
| a single photograph | 91% | 32% |
| one eye (≈3 photos) | 99% | 5% |
| **one infant (≈6 photos)** | **100%** | **0% — all 39 healthy infants were flagged** |

Nothing is wrong with the individual photograph score. The problem is arithmetic: if roughly
two out of three healthy pictures are flagged, then across six pictures almost every healthy
infant collects at least one flag.

**What this means for you.** As currently configured, a positive result carries very little
information at the level of a baby. It cannot be used to decide who needs an ophthalmologist,
because it would send everyone. This is a fixable setting — the rule for combining pictures
and the cut-off both need to be chosen for whole examinations rather than single frames — but
until that is done and re-measured, the tool's output should be treated as a per-photograph
observation only.

## 2. It misses early disease, and it does so confidently

Of photographs showing **Stage 1** disease, **73% are called Normal**. Stage 2: 44%. Stage 3:
37%. Photographs of genuinely normal eyes are called Normal 99% of the time — so the model
errs almost entirely in one direction: toward saying "nothing here".

Worse, it is *more* confident when it makes this mistake than when it is right about Stage 1.
So the obvious safety rule — "only trust the model when it is very sure" — does not work here.
It would keep most of the missed disease and discard the harmless errors.

## 3. Which way the camera points matters more than anything else

ROP tends to appear in the outer part of the retina toward the temple. When the camera looks
there, the model misses disease **12.7%** of the time. When it looks at the centre of the
retina, it misses **71.7%**; toward the nose, **63.4%**.

**What this means for you.** A "Normal" result on a photograph that does not include the
temporal periphery says very little. The photograph may simply not contain the thing being
looked for.

## 4. The confidence percentage is not a probability

On data resembling its training hospitals, the model says 90% when it is right about 92% of
the time — close enough to be useful. On the locked test — a hospital it had never seen —
predictions made with more than 90% confidence were **wrong 41% of the time**.

**What this means for you.** Treat the percentage as a ranking, never as a chance of disease,
and do not transfer confidence judgements from one hospital to another.

## 5. Two categories cannot be judged at all

The locked test contains **4** Stage 4/5 photographs and **12** AP-ROP photographs. Any
percentage computed from numbers that small is noise. The project marks both "not externally
validated" and declines to quote them. That decision was made before the results were seen.

---

## What none of this proves

- It is **not** a medical device and has no regulatory clearance.
- It has **never** been tested prospectively — on live patients, in real time, in a clinic.
- It has been tested at **one** hospital it did not train on, not many.
- It cannot replace, and has not been compared against, an ophthalmologist examining a baby.
