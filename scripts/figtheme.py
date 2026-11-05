"""Figure theme — one palette definition shared by every figure script.

    FIG_THEME=light .venv/bin/python scripts/make_staging_graphs.py
    FIG_THEME=light .venv/bin/python scripts/make_architecture_diagram.py

Default is `dark` (the on-screen/deck figures, graphs/rop_staging/). `light` writes to
graphs/rop_staging/light/ for print: journals reproduce figures on white paper, and a
dark-surface PNG either wastes ink or gets rejected outright.

THE LIGHT PALETTE IS NOT A RECOLOUR OF THE DARK ONE. It was selected by an exhaustive
search over a CVD-safe candidate pool (Okabe-Ito, Paul Tol, plus darkened variants of our
own hues), scoring every 5-subset by its WORST pairwise CIEDE2000 distance across normal
vision and simulated protanopia, deuteranopia and tritanopia, subject to >=3:1 contrast on
white and >=25 degrees of hue separation so no two series read as "two blues".

  worst dE:  light palette 15.8   |   dark palette 3.6 (deuteran, blue vs purple)

That is not a typo. The dark palette's blue (#3987e5) and purple (#a06be0) are close to
indistinguishable for a deuteranope — a real defect in the figures we have been producing.
It is mitigated on screen because every series is also direct-labelled, so colour is
redundant encoding rather than the sole channel; the light palette removes the defect
instead of relying on that. effnetv2s keeps orange, convnext blue, caformer green and
deit3 an olive close to its gold, so a reader comparing an old figure to a new one still
tracks each model; effnetb0's purple becomes wine, because purple was the failing slot.

Reproduce the search: scripts/validate_palette.py
"""
from __future__ import annotations

import os

THEME = os.environ.get("FIG_THEME", "dark").lower()
if THEME not in ("dark", "light"):
    raise SystemExit(f"FIG_THEME={THEME!r} — expected 'dark' or 'light'")
IS_LIGHT = THEME == "light"

if IS_LIGHT:
    SURF, INK, INK2, GRID = "#ffffff", "#1a1a19", "#55524b", "#d8d4cb"
    COLOR = {
        "effnetv2s":      "#b8471c",   # orange  — identity kept
        "convnext_in12k": "#2f6fbd",   # blue    — identity kept
        "caformer":       "#009e73",   # green   — identity kept
        "deit3":          "#999933",   # olive   — was gold, same family
        "effnetb0":       "#882255",   # wine    — was purple (the CVD-failing slot)
    }
    GOOD, BAD = "#14805b", "#b03434"
    # Architecture-diagram roles. GOLD is a MAGENTA in the light theme, deliberately: any
    # olive/gold dark enough for 4.5:1 text contrast on white collapses onto the orange
    # under deuteranopia (measured dE 2.4-7.0 for every candidate) — deutan flattens the
    # red-green axis, so "orange" and "gold" become the same yellow-brown. Magenta clears
    # it at dE 9.7. The GRL red is darkened for the same reason (vs the orange, tritan).
    ORANGE, BLUE, GREEN, GOLD, RED = "#b8471c", "#2f6fbd", "#009e73", "#8a2f8a", "#7a1f1f"
    # panel fills behind boxes in the diagram (tinted, not the near-black of the dark theme)
    PANEL, PANEL_STAGE, PANEL_AROP, PANEL_SITE, PANEL_GRL, PANEL_BB = (
        "#f4f1ea", "#e8f0fa", "#e4f4ed", "#f7e9f5", "#f8e4e4", "#fbeee6")
    DIAGRAM_TEXT = "#1a1a19"
    FOOTNOTE = "#7a766c"
else:
    SURF, INK, INK2, GRID = "#1a1a19", "#ffffff", "#c3c2b7", "#33322f"
    COLOR = {
        "effnetv2s":      "#d95926",
        "convnext_in12k": "#3987e5",
        "caformer":       "#199e70",
        "deit3":          "#c98500",
        "effnetb0":       "#a06be0",
    }
    GOOD, BAD = "#199e70", "#d94f4f"
    ORANGE, BLUE, GREEN, GOLD, RED = "#d95926", "#3987e5", "#199e70", "#c98500", "#d94f4f"
    PANEL, PANEL_STAGE, PANEL_AROP, PANEL_SITE, PANEL_GRL, PANEL_BB = (
        "#262523", "#16283f", "#12301f", "#33270e", "#301c05", "#3a2418")
    DIAGRAM_TEXT = "#ffffff"
    FOOTNOTE = "#8a8a80"


def outdir(base):
    """graphs/rop_staging → itself (dark) or its light/ subdirectory."""
    from pathlib import Path
    p = Path(base) / "light" if IS_LIGHT else Path(base)
    p.mkdir(parents=True, exist_ok=True)
    return p
