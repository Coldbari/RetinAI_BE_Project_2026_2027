# 7 · Why Four Hospitals? — the reason for every image we took

**Graph:** [`graphs/13_why_four_hospitals.png`](graphs/13_why_four_hospitals.png)

*In plain words: we used **three hospitals to teach** the model and kept a **fourth hospital
completely separate to test** it. This page explains why, with the numbers behind each decision.*

---

## The short answer

| Question | Answer in one line |
|---|---|
| Why not just one hospital? | Because a model trained on one hospital learns **that hospital**, not the disease — and we proved it on our own data. |
| Why these three for training? | Because **no single one of them contains all six stages.** Each one fills a hole the other two have. |
| Why a **fourth** hospital, separate? | Because the only honest way to know if a model works somewhere new is to test it somewhere it has **never** seen. If we had trained on all four, we would have had nothing left to test with. |
| Why only these four and not more? | Because these are the **only public datasets in the world** that carry a per-image ICROP stage label. We checked all eleven we could find. |

---

## 1 · Why not just one hospital?

This is not a theoretical worry. We measured it three separate times on this exact data.

**Evidence 1 — The picture shape alone gives away the answer.**
Each hospital uses a different camera, and each camera produces a different image size. Because
each hospital also has a different mix of sick and healthy babies, a model that learns
"which camera took this?" gets the diagnosis almost for free. We tested this by training a
classifier on **nothing but the image width and height — no pixels at all.** It scored
**AUC 0.911**. That is close to what a real model scores. The shortcut was sitting right there.

**Evidence 2 — The hospital is written all over the picture.**
A simple test showed the hospital can be identified from the file properties with **99.7%
accuracy**. Even after our contrast processing, the average blue level is 53.6 / 73.3 / 74.5 at
the three training hospitals but **87.6** at the test hospital. Colour alone is a fingerprint.

**Evidence 3 — the one that settles it.**
An earlier version of this project trained a screening model on essentially one hospital. It
looked excellent. We then ran it on **663 photographs from a hospital it had never seen**:

> **It flagged 663 out of 663 as diseased — including every single healthy eye.**

Not "somewhat worse". Completely useless. That hospital's whole range of scores sat above the
alarm line the model had learned elsewhere. **That model was retired**, and it is the reason
this project is built the way it is.

**One more reason:** a published 2026 paper used the *same* Ostrava data as us, split it by
photograph instead of by baby, and reported **98–100% accuracy**. Our own honest number on that
data is far lower. Their number is not a better model — it is what happens when the same baby
appears in both the training and test sets. Using several hospitals, and splitting by baby, is
what stops that from happening to us.

---

## 2 · Why these three hospitals for training?

**Because no single hospital has all six stages.** The graph makes this obvious. Out of the
8,024 usable photographs:

| | Normal | Stage 1 | Stage 2 | Stage 3 | **Stage 4/5** | **AP-ROP** | Total | Babies |
|---|---|---|---|---|---|---|---|---|
| **Ostrava** (Czechia) | 2,980 | 252 | 458 | **125** | **0** | 470 | 4,285 | 148 |
| **Shenzhen** (China) | 234 | 89 | 163 | **254** | **0** | **0** | 740 | 397 |
| **ROP-VL** (China) | 623 | 236 | 338 | 241 | **89** | 148 | 1,675 | 983 |
| *Multi-View (TEST ONLY)* | *382* | *295* | *299* | *312* | *18* | *18* | *1,324* | *232* |

Read the bold numbers:

- **Ostrava is huge but lopsided.** 70% of its photographs are normal eyes, it has almost no
  Stage 3 (125), and **zero** Stage 4/5. It gives us volume and most of our AP-ROP cases — but
  on its own it cannot teach the severe end of the scale at all.
- **Shenzhen is small but it is our Stage 3 supplier** — 254 Stage 3 photographs, twice what
  Ostrava has, from a dataset less than a fifth the size. But it has **zero** Stage 4/5 and
  **zero** AP-ROP.
- **ROP-VL is the only source of Stage 4 and Stage 5 anywhere in the world that is public.**
  All **89** images of a detached retina in this entire project come from it. Remove ROP-VL and
  that class simply does not exist.

So the three are not three copies of the same thing. **They are three pieces of one jigsaw:**
Ostrava brings scale and the mild stages, Shenzhen brings Stage 3, ROP-VL brings the severe end.

A second benefit came free: **variety is itself training.** Three cameras, three countries,
three sets of lighting and colour habits force the model to look at the retina instead of at the
picture's style. We also added deliberate colour, sharpness and zoom randomisation on top of
that (see [`03_PREPROCESSING.md`](03_PREPROCESSING.md)) to attack the same problem.

---

## 3 · Why take a **fourth** hospital, and keep it out of training?

**Because you cannot mark your own homework.**

If we had thrown Multi-View into the training pot as well, we would have had four hospitals of
training data and **nothing at all** to answer the only question that matters:

> *"Does this work anywhere other than where it learned?"*

Cross-validation inside the three training hospitals is useful — it tells us the model is not
just memorising individual babies, because we always test on babies it never saw. But those
babies come from the **same hospitals, the same cameras and the same doctors**. It cannot tell
us whether the model has learned ROP or has learned Ostrava.

So **Multi-View was locked away from day one and never used for training.** 1,324 photographs
from 232 babies, never touched.

**And the result justified the decision.** Inside the three training hospitals the model gets
**79%** right; at Multi-View it gets **55%** right, and its per-stage detection falls to about
1 in 4. If we had never held a hospital back, we would have shipped the 79% figure believing it.

### Why was the fourth hospital then split in half?

We divided Multi-View **by baby** into two halves:

| Half | Photos | What it was for |
|---|---|---|
| **Tuning half** | 663 | Allowed to be looked at. Used to debug the pipeline, run audits, and choose the alarm threshold. |
| **Sealed half** | 661 | **Opened exactly once, at the very end.** The rules for how it would be scored were written down and committed *before* it was opened, and the script refuses to run a second time. |

The reason is simple: every time you look at a test set and then change something, that test set
becomes a little bit of training data. We needed somewhere we were *allowed* to look — so we
sacrificed one half to keep the other half genuinely clean.

### The bonus nobody planned for

Multi-View turned out to be the **only dataset in the world that records which direction the
camera was pointing** for each photograph (left/right eye × outer side / centre / nose side).
That one extra column produced the single most important finding in the whole project:

> ROP grows at the **outer edge** of the retina. Photographs of the outer edge score **0.84**;
> photographs of the nose side score **0.35**; photographs of the centre score **0.23**. The
> eye's one diagnosis gets written onto *all* of its photographs — including the ones that
> physically cannot show the disease. So a big part of the model's apparent "failure" is it
> correctly calling a normal-looking photograph normal, and being marked wrong for it.

We could not have discovered that with any of the other three hospitals.

---

## 4 · Why only these four? Why not more hospitals?

We looked for every ROP dataset we could find and acquired **eleven** of them — 13,037
photographs in total. Seven were **not usable for this task**, and it was not a matter of choice:

| Dataset | Photos | Why it could not be used |
|---|---|---|
| FARFUM-RoP (Iran) | 1,533 | Records **plus disease**, not ICROP stage. Different question. |
| HVDROPDB (India) | 485 | Outline drawings of the ridge, **no stage label** |
| Shantou optic disc | 129 | Outline drawings only |
| Shantou vessels | 101 | Outline drawings only |
| ORVS | 251 | Outline drawings only |
| ROP stages sample | 91 | Too small, and we could not verify where it came from |
| COph100 | 0 | Extra notes on the Ostrava images — no new photographs |

**So the honest answer is: we used every public hospital dataset that had what we needed.**
There was no fifth hospital available to take.

### Why not go and collect our own data from a local hospital?

Because collecting photographs of babies requires ethics-board approval, a hospital partnership
and parental consent — none of which a final-year student project has. Using published,
de-identified, licensed research data is the only legitimate route available to us. It is also
why **getting a paediatric ophthalmologist to review a sample of these labels** is the biggest
outstanding item on the to-do list ([`06_CURRENT_STATUS.md`](06_CURRENT_STATUS.md)).

---

## 5 · What each hospital was actually used for — the summary

| Hospital | Country | Photos | Babies | Role | Why it was needed |
|---|---|---|---|---|---|
| **Ostrava** (RIDIRP) | Czechia | 4,285 | 148 | **Training** | The bulk of the data. Most of our normal eyes and most of our AP-ROP cases. |
| **Shenzhen** | China | 740 | 397 | **Training** | Our Stage 3 supplier, and a second camera/country for variety. |
| **ROP-VL** | China | 1,675 | 983 | **Training** | **The only source of Stage 4 and Stage 5 that exists publicly.** |
| **Multi-View** | China | 1,324 | 232 | **TEST ONLY — never trained on** | The honest exam. Also the only source with camera-direction labels. |

### One more thing the graph shows: why we cap the number of photos per baby

Look at the right-hand panel. **Ostrava photographs each baby about 29 times in one sitting**
(one baby appears **470** times), while ROP-VL averages under 2 photographs per baby.

Two consequences, both handled:

1. **We split by baby, never by photograph.** Otherwise the same baby appears in both training
   and testing, and the score becomes meaningless — this is exactly the mistake that produces
   those 98–100% accuracy papers.
2. **We cap how many photographs each baby can contribute to training** (2 normal, 20 per other
   class). Without the cap, a handful of heavily-photographed Ostrava babies would supply most
   of the learning and the model would essentially be trained on them.

---

## 6 · The honest limitation of all this

Using three hospitals for training and one for testing is **much** better than one hospital.
It is still **not** proof of general performance.

> **One external hospital is a sanity check, not a generalisation claim.**

We know the model degrades at one new hospital. We do **not** know whether that is because
Multi-View is unusual, or because the model degrades at *every* new hospital. Only a **second**
independent hospital can separate those two explanations — and finding one is on the list of
next steps.
