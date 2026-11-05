# Patient Management System — Design (Future Work)

> **Status: documented future work — NOT yet implemented.** This document is an
> architecture/design proposal for a final-year B.Tech project. No code described
> here exists in the repository yet; it extends the current screening prototype.

## Purpose

The current system (`webapp/app.py`) is stateless: a clinician uploads a fundus
image, gets per-disease prediction cards + Grad-CAM, and can download a one-off PDF
report (`reports/report_generator.py`). Nothing is persisted. This document proposes a
secure patient management layer that stores patient demographics, fundus images, model
predictions with confidence scores, generated reports, visit history, and follow-up
records — enabling search and **longitudinal comparison** of a patient's retina across
repeat visits. The framing remains a clinical **decision-support prototype**, not a
regulated diagnostic device.

---

## 1. Goals & Non-Goals

**Goals**
- Persist every screening event (image + predictions + confidence + report).
- Group screenings under a stable patient identity; show visit history.
- Longitudinal view: side-by-side prediction trends across visits (e.g. DR grade over
  time, ROP progression).
- Search by patient name / ID / date / disease flag.
- Doctor authentication with role-based access; auditability of who saw/edited what.

**Non-Goals (kept out of scope deliberately)**
- No regulatory/HIPAA certification — this is a prototype with privacy *good practices*.
- No multi-site EHR/HL7-FHIR interoperability (a later phase).
- No automatic clinical decision-making — clinician confirmation always required.

---

## 2. Storage Choice & Rationale

| Option | When to use | Trade-offs |
|---|---|---|
| **SQLite** (local file DB) | Default for the local single-clinic prototype | Zero-config, ships with Python (`sqlite3`/SQLAlchemy), perfect for the desktop "inference + reporting" deployment. Single-writer; not for concurrent multi-user scale. |
| **PostgreSQL** | Multi-user clinic / hospital deployment | Concurrent writes, strong typing, row-level security, `pgcrypto` for column encryption, mature backups. Needs a server to run. |
| **Firebase Firestore** | Cloud / multi-device, offline-first | Managed, real-time sync, built-in auth (Firebase Auth) & rules; document model fits nested visit data. Vendor lock-in; query model is restrictive; cost at scale. |

**Recommendation:** start on **SQLite via SQLAlchemy ORM** so the same model classes
can later point at **PostgreSQL** by changing only the connection string (`DATABASE_URL`).
Firestore is documented as an alternative for a cloud-hosted variant. Large binaries
(fundus images, PDFs) are **not** stored as DB blobs — they live on disk
(`results/patients/<patient_id>/<visit_id>/`) or object storage (S3 / Firebase Storage),
with only the path/URI stored in the DB.

---

## 3. Data Model / Schema

Relational schema (SQLite/PostgreSQL). A `Patient` has many `Visit`s; each `Visit` has
one or more `Image`s; each `Image` yields three `Prediction`s (one per disease) and one
`Report`; follow-ups are tracked per visit.

```
Patient
  id              TEXT PRIMARY KEY        -- clinic MRN or UUID
  name_enc        BLOB                    -- encrypted PII
  dob_enc         BLOB
  gender          TEXT
  contact_enc     BLOB
  consent_given   BOOLEAN
  consent_ts      DATETIME
  created_at      DATETIME
  created_by      TEXT  -> User.id

Visit
  id              TEXT PRIMARY KEY
  patient_id      TEXT  -> Patient.id
  visit_date      DATETIME
  clinician_id    TEXT  -> User.id
  notes           TEXT
  followup_due    DATE                    -- next recommended screening
  status          TEXT                    -- open | reviewed | closed

Image
  id              TEXT PRIMARY KEY
  visit_id        TEXT  -> Visit.id
  laterality      TEXT                    -- OD (right) | OS (left)
  file_path       TEXT                    -- on-disk / object-store URI
  sha256          TEXT                    -- integrity check
  captured_at     DATETIME
  source          TEXT                    -- upload | camera_watch_folder

Prediction
  id              TEXT PRIMARY KEY
  image_id        TEXT  -> Image.id
  disease         TEXT                    -- DR | ROP | Glaucoma
  grade           INTEGER                 -- predicted class index
  label           TEXT                    -- class name
  confidence      REAL                    -- max softmax / screen prob
  probs_json      TEXT                    -- full class-probability vector
  model_version   TEXT                    -- weights hash + config id
  gradcam_path    TEXT
  created_at      DATETIME

Report
  id              TEXT PRIMARY KEY
  visit_id        TEXT  -> Visit.id
  pdf_path        TEXT
  recommendation  TEXT                    -- Healthy | Follow-up | Refer | Urgent
  generated_at    DATETIME
  generated_by    TEXT  -> User.id

FollowUp
  id              TEXT PRIMARY KEY
  visit_id        TEXT  -> Visit.id
  due_date        DATE
  reason          TEXT
  completed       BOOLEAN
  completed_visit TEXT  -> Visit.id       -- links to the visit that closed it

User
  id              TEXT PRIMARY KEY
  email           TEXT UNIQUE
  pw_hash         TEXT                    -- bcrypt/argon2
  role            TEXT                    -- admin | doctor | technician | viewer
  active          BOOLEAN

AuditLog
  id              INTEGER PK AUTOINCREMENT
  user_id         TEXT  -> User.id
  action          TEXT                    -- view | create | edit | export | login
  entity          TEXT                    -- Patient | Visit | Report ...
  entity_id       TEXT
  ts              DATETIME
  ip              TEXT
```

### Firestore alternative (collection mapping)
`patients/{patientId}` → subcollection `visits/{visitId}` → subcollections
`images/{imageId}`, `predictions/{predId}`, `reports/{reportId}`, `followups/{fId}`.
`users/{uid}` and `auditLog/{logId}` are top-level. Access governed by Firestore
Security Rules keyed on the authenticated user's custom-claim `role`.

### ASCII Entity-Relationship Diagram

```
        +-----------+        1      N   +---------+
        |  Patient  |------------------>|  Visit  |
        +-----------+                   +---------+
              ^                          |  |   |  \
        created_by                       |  |   |   \ 1:N
              |                  1:N      |  |   |    +---------+
        +---------+    +-----------------+  |   |    | FollowUp|
        |  User   |    |                    |   |    +---------+
        +---------+    v                    |   v
              ^   +---------+   1:1    +---------+
              |   |  Image  |--------->| Report  |
        +----------+ +---------+        +---------+
        | AuditLog |      | 1:N (one per disease)
        +----------+      v
                     +------------+
                     | Prediction |  (DR / ROP / Glaucoma)
                     +------------+
```

---

## 4. Integration with the Existing Flask App

The persistence layer is added as a **Flask Blueprint** (`webapp/patients/`) registered
in `webapp/app.py`, plus a thin DB module (`webapp/db.py`, SQLAlchemy). Existing routes
stay backward-compatible; persistence is opt-in (a `patient_id` in the request).

**Where predictions get persisted.** Today `/predict` and `/report` call
`REGISTRY.analyze(image)` (`webapp/inference.py`) and return JSON / a PDF without saving.
The change: after a successful `REGISTRY.analyze(...)`, if the request is tied to a
patient/visit, write the rows.

```
POST /patients/<pid>/visits/<vid>/screen   (new, auth-required)
  1. read image  -> save to results/patients/<pid>/<vid>/<image_id>.png (sha256)
  2. result = REGISTRY.analyze(image)        # UNCHANGED existing backend
  3. INSERT Image row
  4. for finding in result["findings"]:      # one per disease
        INSERT Prediction (grade, confidence, probs, model_version, gradcam_path)
  5. generate_pdf(...) -> INSERT Report row  # reuses reports/report_generator.py
  6. INSERT AuditLog(action="create")
  7. return prediction cards + links
```

The existing stateless `/predict` is retained for quick anonymous demos. New routes:

- `GET  /patients` — list/search (auth)
- `POST /patients` — create patient (consent captured here)
- `GET  /patients/<pid>` — profile + visit history + **longitudinal trend chart**
- `POST /patients/<pid>/visits` — open a visit
- `GET  /patients/<pid>/compare?diseases=DR` — side-by-side images + grade-over-time
- `GET  /visits/<vid>/report` — re-download stored PDF

**Longitudinal comparison** is a query over `Prediction` joined to `Visit.visit_date`
for a patient+disease, rendered as a trend line (e.g. DR grade across visits) plus
thumbnail Grad-CAMs per visit, so a clinician can eyeball progression.

---

## 5. Authentication & Role-Based Access

- **Doctor login** via Flask-Login (session) or JWT; passwords hashed with argon2/bcrypt.
- **Roles** (`User.role`):
  - `admin` — manage users, full access.
  - `doctor` — create/view patients & visits, generate/sign reports.
  - `technician` — capture images, create visits; cannot finalize reports.
  - `viewer` — read-only (e.g. auditor/research).
- Enforced with a `@require_role(...)` decorator on each blueprint route. Firestore
  variant enforces the same via Security Rules on custom claims.

---

## 6. Privacy & Security Considerations

- **Encryption at rest:** PII columns (`name_enc`, `dob_enc`, `contact_enc`) encrypted
  with a Fernet/AES key from an env var or OS keyring; never committed. With PostgreSQL,
  `pgcrypto` or full-disk encryption; Firestore encrypts at rest by default.
- **Encryption in transit:** TLS for the Flask app (reverse proxy / HTTPS) in any
  non-localhost deployment.
- **PII handling:** images and reports stored under per-patient directories outside the
  web root; served only through authenticated routes (never static-served).
- **Consent:** `Patient.consent_given` / `consent_ts` captured at registration; export
  and research use gated on it.
- **Audit log:** every view/create/edit/export/login appended to `AuditLog`
  (append-only) — supports accountability and incident review.
- **Data minimization & retention:** configurable retention; a purge job can delete or
  anonymize records past retention while keeping de-identified predictions for metrics.
- **De-identification for research:** an export path that strips PII and keeps
  image + predictions for model evaluation (ties into `models/evaluation/`).
- **Disclaimer preserved:** stored reports keep the existing "not a diagnostic device"
  disclaimer from `reports/report_generator.py`.

---

## 7. Suggested File Layout (when implemented)

```
webapp/
  app.py              # register patients blueprint
  db.py               # SQLAlchemy engine/session, models
  patients/
    __init__.py       # Blueprint
    routes.py         # routes in §4
    auth.py           # login, @require_role
    crypto.py         # field encryption helpers
    audit.py          # AuditLog helper
  templates/patients/ # list, profile, compare views
migrations/           # Alembic schema migrations
```

## 8. Phasing

1. **MVP** — SQLite + SQLAlchemy models, login, persist screenings, visit history.
2. **Comparison** — longitudinal trend view + search.
3. **Hardening** — field encryption, audit log, consent, retention.
4. **Scale** — swap `DATABASE_URL` to PostgreSQL (or Firestore variant), object storage.
