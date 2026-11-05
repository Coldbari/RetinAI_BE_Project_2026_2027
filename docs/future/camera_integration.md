# Fundus Camera Integration — Design (Future Work)

> **Status: documented future work — NOT yet implemented.** This is an
> architecture/design proposal. None of the watcher, DICOM, or device-driver code
> described here exists in the repo yet; it extends the current upload-based prototype.

## Purpose

Today images enter the system only through manual upload in the Flask app
(`webapp/app.py`). This document proposes integrating the screening pipeline directly
with **retinal/fundus imaging devices** so that an image captured at the camera flows
automatically through AI analysis to a clinical report on the doctor's dashboard, with
minimal manual steps. It is delivered in two phases: a realistic **watched-folder**
integration achievable now, and a future **direct device (USB / network / DICOM)**
integration. The system remains a **decision-support screening assistant**, not a
regulated diagnostic device.

---

## Target Workflow

```
Patient ─► Fundus Camera ─► Image Acquisition ─► AI Analysis
                                                     │
                                                     ▼
   Doctor Dashboard ◄── Clinical Report ◄── Disease Detection
```

Both phases implement this same flow; they differ only in *how the image is acquired*.

---

## Phase 1 — Watched Folder (realistic now)

Most clinical fundus cameras and their capture software can **export/save images to a
local or network folder**. Phase 1 exploits this: a background watcher observes an
"inbox" directory and automatically processes any new image.

### How it works
- A long-running watcher process uses the **`watchdog`** library
  (`Observer` + a `FileSystemEventHandler`) to fire on `on_created` events in
  `incoming/`.
- On a new file: wait for the write to settle (debounce / size-stable check, since
  cameras may write incrementally), validate it's a readable fundus image
  (`PIL.Image.open(...).convert("RGB")`), then hand it to the **existing inference
  backend** — `Registry` from `webapp/inference.py` — via `REGISTRY.analyze(image)`.
  No new model code is needed; preprocessing stays the single source of truth in
  `models/common/preprocessing.py`.
- The returned findings + Grad-CAM are passed to `generate_pdf(...)` in
  `reports/report_generator.py` to produce the clinical PDF.
- Processed files move to `processed/`; failures move to `failed/` with a log entry.
- Results are surfaced to the doctor dashboard. If the patient-management layer
  (`docs/future/patient_management.md`) exists, the watcher also persists the
  `Image`/`Prediction`/`Report` rows; otherwise it just drops the PDF + a JSON result
  in an `outbox/` the dashboard polls.

### Folder layout

```
camera_inbox/
  incoming/      # camera writes new captures here  ──► watcher picks up
  processing/    # claimed by watcher (atomic move) to avoid double-processing
  processed/     # originals, archived after success
  failed/        # unreadable / errored captures
  outbox/        # generated PDFs + result JSON for the dashboard
```

### Filename / metadata convention
Patient + laterality are often unknown from a bare file. Options, in order of
preference: (1) camera software encodes them in the filename
(`<patientId>_<OD|OS>_<timestamp>.jpg`) — parsed by the watcher; (2) a sidecar
`.json`/`.txt`; (3) the dashboard prompts the clinician to associate an unmatched
capture with a patient before finalizing.

### Pseudocode (watcher)

```python
# scripts/camera_watch.py  (proposed)
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from PIL import Image
from webapp.inference import Registry              # EXISTING backend
from reports.report_generator import generate_pdf  # EXISTING report

REGISTRY = Registry("webapp/registry.yaml")

class CaptureHandler(FileSystemEventHandler):
    def on_created(self, event):
        path = wait_until_stable(event.src_path)    # debounce partial writes
        path = atomic_move(path, "processing/")
        try:
            image = Image.open(path).convert("RGB")
            result = REGISTRY.analyze(image)        # same as the web app
            meta   = parse_filename(path)           # patient id, OD/OS
            pdf    = outbox(meta) + ".pdf"
            generate_pdf(pdf, meta, result["findings"], image, decode(result["heatmap"]))
            archive(path, "processed/"); notify_dashboard(meta, result, pdf)
        except Exception as e:
            archive(path, "failed/"); log(e)

Observer().schedule(CaptureHandler(), "camera_inbox/incoming", recursive=False)
```

### Why this is the right first step
- **Zero device drivers / SDKs** — works with virtually any camera that can export files.
- Reuses 100% of the existing inference + reporting code; the only new component is the
  watcher glue.
- Runs on the same desktop that already does "inference + reporting" — consistent with
  the Kaggle-first design (no training here).

### Phase 1 components
- `watchdog` watcher process (new) — `scripts/camera_watch.py`.
- Existing `Registry` (`webapp/inference.py`) and `generate_pdf` (`reports/`).
- Shared preprocessing `models/common/preprocessing.py` (unchanged).
- A small config block (inbox path, archive paths, filename pattern, poll/debounce).
- Optional: dashboard route to list `outbox/` results / notify the doctor.

---

## Phase 2 — Direct Device Integration (future)

Direct integration removes the manual export step and supports clinical interoperability.

### Modes
- **USB / vendor SDK:** many cameras expose a USB connection driven by a
  vendor-specific SDK or a TWAIN/UVC interface. A vendor adapter pulls frames directly.
- **Network / DICOM:** clinical-grade fundus cameras commonly speak **DICOM**. The
  relevant pieces:
  - **DICOM C-STORE SCP:** our system runs a storage service-class provider; the camera
    (or a PACS) pushes the *Ophthalmic Photography* DICOM object to us.
  - **DICOM Worklist (MWL):** the camera pulls the patient/visit worklist from us so the
    capture is already tagged with patient ID + laterality (closes the metadata gap from
    Phase 1).
  - Tooling: **`pydicom`** (parse), **`pynetdicom`** (DICOM networking SCU/SCP). Pixel
    data is extracted to an RGB array and fed into the same `Registry.analyze`.
- **Vendor landscape (generic):** major fundus/retinal camera vendors and tabletop /
  handheld / smartphone-based devices vary in SDK, file format, and DICOM conformance.
  The design treats them generically behind an abstraction so no single vendor is
  hard-coded.

### Abstraction layer (the key design element)
A `CameraSource` interface decouples *acquisition* from the unchanged AI pipeline:

```
                       +------------------------------+
                       |   CameraSource (interface)   |
                       |  acquire() -> RGB image+meta |
                       +---------------+--------------+
                                       |
        +----------------+-------------+--------------+----------------+
        |                |                            |                |
+---------------+ +----------------+         +----------------+ +----------------+
| WatchedFolder | |  DicomScpSource|         |  UsbSdkSource  | |  MockSource    |
|  Source (P1)  | | (C-STORE/MWL)  |         | (vendor SDK)   | | (tests/demo)   |
+---------------+ +----------------+         +----------------+ +----------------+
        \________________________ all yield (RGB image, metadata) ____/
                                       |
                                       ▼
                          REGISTRY.analyze(image)   (UNCHANGED)
                                       │
                          generate_pdf(...) -> dashboard / DB
```

Each source normalizes its output to `(PIL.Image RGB, metadata{patient_id, laterality,
captured_at, source})`. Everything downstream — preprocessing, the three EfficientNetV2-S
models, Grad-CAM, PDF — is identical to today. Phase 1's watcher is simply the first
concrete `CameraSource`.

### Phase 2 components
- `CameraSource` abstract base + concrete adapters (`DicomScpSource`, `UsbSdkSource`,
  `WatchedFolderSource`, `MockSource`).
- DICOM stack: `pydicom`, `pynetdicom`; a C-STORE SCP service and optional MWL SCP.
- DICOM <-> RGB conversion + metadata mapping (DICOM tags → our `metadata`).
- A device-config registry (which source is active, host/port/AE-title, SDK params).
- Security: device authentication, network segmentation, TLS for DICOM where supported.

---

## End-to-End Data-Flow Diagram

```
            Phase 1 (now)                         Phase 2 (future)
   ┌─────────────────────────┐          ┌──────────────────────────────┐
   │ Camera exports to folder│          │ Camera --USB SDK / DICOM-->   │
   │      incoming/          │          │   DicomScpSource / UsbSdk     │
   └────────────┬────────────┘          └───────────────┬──────────────┘
                │  watchdog on_created                   │ acquire()
                ▼                                        ▼
        ┌───────────────────────  CameraSource normalize  ───────────────────┐
        │              (PIL RGB image  +  patient/laterality metadata)        │
        └─────────────────────────────────┬──────────────────────────────────┘
                                           ▼
                       models/common/preprocessing.py  (shared, unchanged)
                                           ▼
                       webapp/inference.py  Registry.analyze()
                       → DR / ROP / Glaucoma predictions + Grad-CAM
                                           ▼
                       reports/report_generator.py  generate_pdf()
                                           ▼
                   Patient DB (optional)  ──►  Doctor Dashboard / notification
```

---

## Safety & Scope Notes
- All findings carry the existing "AI-assisted screening prototype — not a diagnostic
  device" disclaimer; a clinician confirms every result.
- Automated capture must not auto-finalize a diagnosis; the dashboard always requires
  doctor review before a report is signed.
- DICOM conformance, device certification, and IEC 62304 / regulatory pathways are
  explicitly **out of scope** for this prototype and noted as prerequisites for any real
  clinical deployment.
