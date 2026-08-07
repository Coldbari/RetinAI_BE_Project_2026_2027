# Hardware

**RetinAI is a software and machine-learning project. No custom hardware was designed or built.**

There is no circuit, no PCB, no CAD model, no microcontroller and no sensor interfacing in this
project. Rather than leave this section blank or invent hardware to fill a template, this file
states plainly what the system runs on and what it consumes images from.

---

## 1. What the system runs on

| Role | Specification | Notes |
|---|---|---|
| **Training** | Kaggle cloud GPU — NVIDIA **P100 16 GB** or **T4 16 GB** | Free tier. 12-hour session cap, 2 concurrent GPU sessions. Both constraints shaped the code: per-epoch resume-safe checkpointing exists because of the session cap |
| **Serving** | HuggingFace Spaces free tier — 2 vCPU, 16 GB RAM, **no GPU** | Deliberate. A screening tool that needs a GPU to serve is a tool most clinics cannot deploy |
| **Development** | Apple Silicon Mac, 16 GB RAM | Local inference uses the MPS backend; evaluation, figures and reports run here |
| **Mobile client** | Any Android 8.0+ (API 26) device with an autofocus rear camera | Capture, on-device quality check, PDF export |

### Two hardware constraints that changed the software

**P100 architecture support.** Kaggle's default image ships a torch build that dropped the P100's
`sm_60` compute capability, so every CUDA operation failed with *"no kernel image is available"*.
Every training kernel now pins `torch==2.5.1+cu121`, which supports both P100 and T4.

**16 GB VRAM at 384×384.** Training at 384px with batch 32 does not fit a T4. The fix was
mixed-precision (AMP) plus gradient accumulation — batch 16 accumulated twice for an effective
batch of 32 — rather than dropping the input resolution. That mattered: the DR resolution audit
later showed input size alone is worth roughly **0.08 AUC**, so reducing resolution to fit memory
would have quietly cost real accuracy.

---

## 2. Image-acquisition hardware (third-party, not built by this team)

The system consumes photographs produced by existing clinical fundus cameras. These are listed
because the **device the image came from turned out to matter more than any hyperparameter** in
this project.

| Population | Camera class | Characteristics |
|---|---|---|
| Adults (DR, Glaucoma) | Standard mydriatic / non-mydriatic fundus cameras | ~30–50° field of view, posterior-pole centred, patient seated at a chin rest |
| Pre-term infants (ROP) | RetCam-class wide-field contact camera | ~130° field of view, contact lens against the eye, bedside in a NICU, infant supine |

### The device confound

Three distinct imaging devices appear in the ROP database. They do not sample the same
population:

| Device | Images | ROP+ images | Patients | ROP+ patients | Patient-level prevalence |
|---|---|---|---|---|---|
| D1 | 2,578 | 1,925 | 81 | 39 | **48.1%** |
| D2 | 2,452 | 741 | 61 | 13 | 21.3% |
| D3 | 974 | 358 | 46 | 8 | 17.4% |

Patient-level prevalence spreads **30.8 percentage points** across devices. A model can therefore
score well by learning *which camera took the picture* rather than whether the retina is diseased.

Worse, in the original test split **every ROP-positive image came from a single device (D1)**.
Discrimination could only be measured inside that one device; for D2 and D3 there was nothing to
discriminate.

- Pooled test AUC: **0.9271**
- Within-device (D1) AUC: **0.8808**
- **Difference: −0.0462**, which is the part of the headline number that was device recognition

The defensible ROP figure is **0.881**, and the dataset was re-split with device stratification in
response. This is documented in the README's [Testing and Results](../README.md#testing-and-results)
and shown in [figure 05](../images/05_rop_device_confound.png).

**The general lesson:** in medical imaging, the acquisition hardware is part of the label
distribution. Any accuracy quoted without controlling for it is partly a measurement of the
camera.

---

## 3. Why there is no circuit diagram

The template's [Circuit Diagram](../README.md#circuit-diagram), PCB, CAD and simulation sections
do not apply. The equivalent artefacts for this project are:

| Template artefact | This project's equivalent |
|---|---|
| Circuit diagram | [System architecture](../images/system_architecture.png) — how data moves between components |
| Flowchart | [Screening flow](../images/flowchart.png) — gate → route → preprocess → infer → explain → report |
| Component datasheets | Dataset provenance and the confound audits |
| Bench testing | The 18-row test log in the [README](../README.md#testing-and-results) |
