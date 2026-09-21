# Showing RetinAI — 5-minute runbook

```bash
./demo.sh
```
Boot takes ~40s (two checkpoints load), then the browser opens on **Screen**.
Everything runs on this laptop — no internet needed once the weights are in place.

Fetch the weights first if this is a fresh clone; they are not in the repository:

```bash
.venv/bin/python scripts/get_weights.py
```

## Pick the patient context first

`Preterm infant (NICU screening)` is mandatory. It is not a formality: the model is
trained only on preterm infants, and without routing it flagged **41 of 45 adult eyes**
as ROP-positive. If someone asks "what stops it running on the wrong patient?" — this is
the answer, and the API returns 400 if the context is missing.

## Images for the demo

**The demo images are not in this repository, and this is deliberate.**

The 34 images the demo was scored on are private infant fundus photographs from the
Ostrava cohort, and their filenames encode gestational age, birth weight, sex and
diagnosis. A filename is patient data. This log book is a public repository, so the
images and the file list live only in the private working repository, and this page
reports what they produced rather than shipping them.

The same reasoning retired the Gallery page and keeps `webapp/static/` free of patient
photographs. See the PHI remediation note in the main README.

To run the demo on your own data, upload any preterm infant fundus photograph from the
Screen page with the context set to `Preterm infant (NICU screening)`.

## Measured on the 34 bundled images

| | correct |
|---|---|
| ROP present | 11 / 14 |
| No ROP | 18 / 20 |
| **total** | **29 / 34 (85%)** |

The three ROP misses are GA 35 / BW 2370–3050 — more mature, heavier infants with milder
disease. The hits are GA 24–26 / BW 550–910. The model catches severe ROP in the most
premature infants and misses mild disease in the more mature ones. Say this before you're
asked; it is a real and clinically coherent limitation, and it is the honest version of
the 85%.

## Numbers to quote

- Threshold **0.0155**, chosen by a rule written down before looking at the curves.
- Internal: sens **0.958** / spec **0.781**. Held-out hospital: AUC **0.821**,
  sens 0.905 / spec 0.387.
- The earlier ResNet50 screening head was **retired**: audited on 663 images from a
  hospital it never trained on, it flagged **all 663**, every healthy eye included.
  Retiring it is the strongest thing in the project — it shows the audit was real.
- Staging preview: 6-class ICROP, 5-fold CV macro-F1 **0.692 ± 0.086** — and it is
  *equivalent* to a flat softmax, not better. The contribution is measurement, not
  architecture.

## Pages

- **Screen** — the demo. Upload, verdict, Grad-CAM, staging preview, PDF download.
- **History** — the screenings from this session (local only; disabled on any hosted build).
- **About** — routing table and model provenance.
- **Gallery** — *skip it, it is empty by design.* `webapp/static/` is web-servable, so a
  repo guard forbids putting patient photographs there. If asked, that is a good answer.

## If it will not start

```bash
.venv/bin/python scripts/get_weights.py   # re-pull both checkpoints
tail -20 /tmp/retinai-demo.log
```
