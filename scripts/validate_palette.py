"""Prove the figure palettes are colourblind-safe — or fail loudly.

    .venv/bin/python scripts/validate_palette.py

Scores every co-occurring colour set in scripts/figtheme.py by its WORST pairwise
CIEDE2000 distance under normal vision and simulated protanopia, deuteranopia and
tritanopia. A palette whose worst pair sits below dE 8 is not distinguishable for some
readers, and in a paper figure that is a defect, not a preference.

This is also the script that found the defect it now guards against: the DARK palette's
blue (#3987e5) and purple (#a06be0) are dE 3.6 apart under deuteranopia. It is reported
rather than silently tolerated — on screen every series is direct-labelled so colour is
redundant, but the light (print) palette was rebuilt to remove the problem outright.

Exit code is non-zero if the LIGHT palette regresses below the threshold, so this can gate
a figure rebuild; the dark palette's known deficiency is reported as a warning.
"""
from __future__ import annotations

import sys
from itertools import combinations

import numpy as np

THRESHOLD = 8.0


def hex2rgb(h):
    return np.array([int(h[i:i + 2], 16) / 255 for i in (1, 3, 5)])


def _lin(c):
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def rgb2lab(rgb):
    m = np.array([[0.4124, 0.3576, 0.1805],
                  [0.2126, 0.7152, 0.0722],
                  [0.0193, 0.1192, 0.9505]])
    xyz = (m @ _lin(rgb)) / np.array([0.95047, 1.0, 1.08883])
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16 / 116)
    return np.array([116 * f[1] - 16, 500 * (f[0] - f[1]), 200 * (f[1] - f[2])])


def de2000(lab1, lab2):
    L1, a1, b1 = lab1
    L2, a2, b2 = lab2
    C1, C2 = np.hypot(a1, b1), np.hypot(a2, b2)
    Cb = (C1 + C2) / 2
    G = 0.5 * (1 - np.sqrt(Cb ** 7 / (Cb ** 7 + 25 ** 7))) if Cb > 0 else 0.0
    a1p, a2p = (1 + G) * a1, (1 + G) * a2
    C1p, C2p = np.hypot(a1p, b1), np.hypot(a2p, b2)
    h1p, h2p = np.degrees(np.arctan2(b1, a1p)) % 360, np.degrees(np.arctan2(b2, a2p)) % 360
    dLp, dCp = L2 - L1, C2p - C1p
    dhp = h2p - h1p
    if abs(dhp) > 180:
        dhp -= 360 * np.sign(dhp)
    dHp = 2 * np.sqrt(C1p * C2p) * np.sin(np.radians(dhp) / 2)
    Lbp, Cbp = (L1 + L2) / 2, (C1p + C2p) / 2
    hbp = (h1p + h2p) / 2 if abs(h1p - h2p) <= 180 else (h1p + h2p + 360) / 2
    T = (1 - 0.17 * np.cos(np.radians(hbp - 30)) + 0.24 * np.cos(np.radians(2 * hbp))
         + 0.32 * np.cos(np.radians(3 * hbp + 6)) - 0.20 * np.cos(np.radians(4 * hbp - 63)))
    SL = 1 + 0.015 * (Lbp - 50) ** 2 / np.sqrt(20 + (Lbp - 50) ** 2)
    SC, SH = 1 + 0.045 * Cbp, 1 + 0.015 * Cbp * T
    RT = (-2 * np.sqrt(Cbp ** 7 / (Cbp ** 7 + 25 ** 7))
          * np.sin(np.radians(60 * np.exp(-(((hbp - 275) / 25) ** 2)))))
    return float(np.sqrt((dLp / SL) ** 2 + (dCp / SC) ** 2 + (dHp / SH) ** 2
                         + RT * (dCp / SC) * (dHp / SH)))


CVD = {
    "protan": [[0.152, 1.053, -0.205], [0.115, 0.786, 0.099], [-0.004, -0.048, 1.052]],
    "deutan": [[0.367, 0.861, -0.228], [0.280, 0.673, 0.047], [-0.012, 0.043, 0.969]],
    "tritan": [[1.256, -0.077, -0.179], [-0.078, 0.931, 0.148], [0.005, 0.691, 0.304]],
}


def simulate(rgb, kind):
    return np.clip(np.array(CVD[kind]) @ rgb, 0, 1)


def contrast(fg, bg):
    def lum(h):
        c = _lin(hex2rgb(h))
        return 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]
    a, b = sorted([lum(fg), lum(bg)], reverse=True)
    return (a + 0.05) / (b + 0.05)


def audit(name, colours, surface):
    worst = (1e9, "", "", "")
    for kind in ("normal", "protan", "deutan", "tritan"):
        for x, y in combinations(colours, 2):
            cx, cy = hex2rgb(x), hex2rgb(y)
            if kind != "normal":
                cx, cy = simulate(cx, kind), simulate(cy, kind)
            d = de2000(rgb2lab(cx), rgb2lab(cy))
            if d < worst[0]:
                worst = (d, kind, x, y)
    low = [c for c in colours if contrast(c, surface) < 3.0]
    ok = worst[0] >= THRESHOLD and not low
    print(f"  {'PASS' if ok else 'FAIL'}  {name}: worst dE {worst[0]:.1f} "
          f"({worst[1]}, {worst[2]} vs {worst[3]})")
    if low:
        print(f"        low contrast on {surface}: {', '.join(low)}")
    return ok


def main():
    import os
    results = {}
    for theme in ("dark", "light"):
        os.environ["FIG_THEME"] = theme
        for mod in [m for m in list(sys.modules) if m.endswith("figtheme")]:
            del sys.modules[mod]
        sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
        import figtheme as T

        print(f"\n{theme.upper()} theme (surface {T.SURF})")
        ok = audit("5-model series", list(T.COLOR.values()), T.SURF)
        ok &= audit("good/bad pair", [T.GOOD, T.BAD], T.SURF)
        ok &= audit("diagram roles", [T.ORANGE, T.BLUE, T.GREEN, T.GOLD], T.SURF)
        results[theme] = ok

    print()
    if not results["light"]:
        print("LIGHT palette is below threshold — print figures would be unreadable for "
              "some readers. Fix scripts/figtheme.py before rebuilding.")
        return 1
    if not results["dark"]:
        print("NOTE: the dark palette carries a known deuteranopia collision (blue vs "
              "purple). Mitigated because every series is direct-labelled; the light "
              "palette used for print does not have it. Not gating on it.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
