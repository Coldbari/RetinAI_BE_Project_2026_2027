# RetinAI — Documentation

Full documentation for the RetinAI retinal-disease screening project. Start with whichever
question you have.

> ⚠️ **Clinical decision-support prototype — not a diagnostic device.** Research/education
> only; not prospectively validated; no regulatory clearance. See [LIMITATIONS.md](LIMITATIONS.md).

## Map

| I want to know… | Read |
|---|---|
| What is this project, at a glance? | [../README.md](../README.md) (root) |
| Everything we built, in technical detail | [PROJECT_OVERVIEW.md](PROJECT_OVERVIEW.md) |
| The actual numbers (metrics, validation, calibration) | [RESULTS.md](RESULTS.md) |
| What it **can't** do / known gaps | [LIMITATIONS.md](LIMITATIONS.md) |
| Problems we hit and how we solved them | [CHALLENGES.md](CHALLENGES.md) |
| What comes next | [ROADMAP.md](ROADMAP.md) |
| How training runs on Kaggle | [../kaggle_kernels/README.md](../kaggle_kernels/README.md) |
| The one-page honest results summary | [../thesis_assets/RESULTS_SUMMARY.md](../thesis_assets/RESULTS_SUMMARY.md) |

## Design docs (future work — not yet built)

Complete architecture proposals for the next features, each reusing the existing
preprocessing / inference `Registry` / PDF generator:

| Feature | Doc |
|---|---|
| Persist patients, visits, predictions, reports; longitudinal comparison | [future/patient_management.md](future/patient_management.md) |
| Auto-ingest from a fundus camera (watched folder → DICOM) | [future/camera_integration.md](future/camera_integration.md) |
| End-to-end automated queue/worker pipeline + doctor notification | [future/realtime_pipeline.md](future/realtime_pipeline.md) |

## Data note

`docs/infant_retinal_database_info.csv` / `.xlsx` — metadata for the ROP infant retinal
database (188 patients: gestational age, birth weight, diagnosis code, device, etc.).

## The 60-second summary

RetinAI screens one fundus photo for **DR, ROP, and Glaucoma** with **three dedicated
models** (EfficientNetV2-S / ResNet50), each returning a Grad-CAM heatmap, a graded risk, and
a PDF report. Held-out results: **ROP AUC 0.927**, **DR referable AUC 0.906** (Messidor-2
external 0.837), **Glaucoma AUC 0.973** (per-source G1020 0.957, recovered from a 0.589
domain-shift failure). Training is config-driven and runs on Kaggle GPU; serving is a Flask
app deployed CPU-only on HuggingFace Spaces. It is an honest prototype with real limitations —
read [LIMITATIONS.md](LIMITATIONS.md).
