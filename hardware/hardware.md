# Hardware

RetinAI is a software and machine-learning project. We did not design or build any custom
hardware.

There is no circuit, no PCB, no CAD model, no microcontroller and no sensor interfacing here.
Rather than leave this section blank or invent hardware to fill the template, this file states
what the system runs on and what it takes images from.

---

## 1. What the system runs on

| Role | Specification | Notes |
|---|---|---|
| Training | Kaggle cloud GPU, NVIDIA P100 16 GB or T4 16 GB | Free tier. 12-hour session cap and 2 concurrent GPU sessions. Both constraints shaped our code, and the per-epoch resume-safe checkpointing exists because of the session cap |
| Serving | HuggingFace Spaces free tier, 2 vCPU, 16 GB RAM, no GPU | Deliberate. A screening tool that needs a GPU to serve is one most clinics cannot deploy |
| Development | Apple Silicon Mac, 16 GB RAM | Local inference uses the MPS backend. Evaluation, figures and reports run here |
| Mobile client | Any Android 8.0+ (API 26) device with an autofocus rear camera | Capture, on-device quality check, PDF export |

### Two hardware constraints that changed the software

**P100 architecture support.** Kaggle's default image ships a torch build that dropped the P100's
`sm_60` compute capability, so every CUDA operation failed with "no kernel image is available".
All our training kernels now pin `torch==2.5.1+cu121`, which supports both P100 and T4.

**16 GB VRAM at 384×384.** Training at 384px with batch 32 does not fit a T4. We fixed it with
mixed precision (AMP) plus gradient accumulation, batch 16 accumulated twice for an effective 32,
rather than dropping the input resolution. That turned out to matter, because the DR resolution
audit later showed input size alone is worth roughly 0.08 AUC. Shrinking the images to fit memory
would have quietly cost us real accuracy.

---

## 2. Image-acquisition hardware (third-party, not built by us)

The system consumes photographs from fundus cameras that already exist in clinics. We list them
because the device an image came from turned out to matter more than any hyperparameter in this
project.

| Population | Camera class | Characteristics |
|---|---|---|
| Adults (DR, Glaucoma) | Standard mydriatic or non-mydriatic fundus cameras | Roughly 30 to 50 degree field of view, posterior-pole centred, patient seated at a chin rest |
| Pre-term infants (ROP) | RetCam-class wide-field contact camera | Roughly 130 degree field of view, contact lens against the eye, taken at the bedside in a NICU with the infant supine |

### The device confound

Three distinct imaging devices appear in the ROP database, and they do not sample the same
population:

| Device | Images | ROP+ images | Patients | ROP+ patients | Patient-level prevalence |
|---|---|---|---|---|---|
| D1 | 2,578 | 1,925 | 81 | 39 | 48.1% |
| D2 | 2,452 | 741 | 61 | 13 | 21.3% |
| D3 | 974 | 358 | 46 | 8 | 17.4% |

Patient-level prevalence spreads 30.8 percentage points across devices. A model can therefore
score well by learning which camera took the picture instead of whether the retina is diseased.

It gets worse. In our original test split, every ROP-positive image came from a single device
(D1), so discrimination could only be measured inside that one device. For D2 and D3 there was
nothing to discriminate at all.

- Pooled test AUC: 0.9271
- Within-device (D1) AUC: 0.8808
- Difference: −0.0462, which is the part of our headline number that was device recognition

So the defensible ROP figure is 0.881, and we re-split the dataset with device stratification in
response. This is written up in the README's
[Testing and Results](../README.md#testing-and-results) and shown in
[figure 05](../images/05_rop_device_confound.png).

The general point is that in medical imaging the acquisition hardware is part of the label
distribution. Any accuracy quoted without controlling for it is partly a measurement of the
camera.

---

## 3. Why there is no circuit diagram

The template's [Circuit Diagram](../README.md#circuit-diagram), PCB, CAD and simulation sections
do not apply to us. The equivalent artefacts for this project are:

| Template artefact | Our equivalent |
|---|---|
| Circuit diagram | [System architecture](../images/system_architecture.png), showing how data moves between components |
| Flowchart | [Screening flow](../images/flowchart.png): gate, route, preprocess, infer, explain, report |
| Component datasheets | Dataset provenance and the confound audits |
| Bench testing | The test log in the [README](../README.md#testing-and-results) |
