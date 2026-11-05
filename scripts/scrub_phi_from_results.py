"""De-identify patient filenames inside committed result JSONs.

    .venv/bin/python scripts/scrub_phi_from_results.py --check      # report only
    .venv/bin/python scripts/scrub_phi_from_results.py --write      # rewrite in place

WHY THIS EXISTS. Two committed evidence files — results/cross_disease_matrix.json (the T16
cross-disease routing audit) and results/manual_test_results.json (the 93-image manual test)
— recorded the ORIGINAL Ostrava filenames as their row keys. Those filenames are clinical
records, not identifiers:

    rop_<id>_<sex>_GA<weeks>_BW<grams>_PA<days>_DG<code>_PF<flag>_D<dev>_S<serie>_<frame>.jpg
        id     patient number
        sex    M or F
        GA     gestational age, weeks
        BW     birth weight, grams
        PA     postnatal age at imaging
        DG     diagnosis group
        PF     plus-disease flag

    That template is deliberately written with PLACEHOLDERS rather than values. Two earlier
    revisions of this docstring did it with values instead: the first pasted a REAL row from
    the Ostrava clinical record — an actual infant's sex, gestational age and birth weight —
    into the one file whose whole purpose is removing such things, caught by the content
    guard in tests/test_repo_guards.py on 2026-08-15 before it was pushed. The second used
    the synthetic 999/GA99/BW9999 sentinel, which was safe but got silently collapsed by the
    history rewrite that cleaned this repo, because a sentinel that looks exactly like the
    real pattern is also matched by every tool built to destroy the real pattern. Placeholders
    survive both. Illustrate formats with placeholders; never with a live record, and not with
    a look-alike either.

Sex + gestational age + birth weight + imaging date for a named hospital's NICU cohort is
re-identifying. The image files themselves were de-identified in August 2026
(scripts/anonymise_ostrava_tree.py) and the repo guard was taught to reject the pattern in
FILE PATHS — but nothing checked file CONTENTS, so these two JSONs kept the original
strings and were committed and pushed.

WHAT THIS DOES. Rewrites each identifier to the same de-identified form the image tree
already uses — patient number + device + series + frame, which is what the analysis needs
to group and locate a row — and drops every clinical attribute:

    rop_<id>_<sex>_GA..._BW..._PA..._DG..._PF..._D<dev>_S<serie>_<frame>.jpg
        ->  rop_<id>_D<dev>_S<serie>_<frame>.jpg

The evidence value is preserved: rows stay distinguishable and groupable by patient and
session. The clinical record does not survive.

This script alone was NOT sufficient, and the record of why is worth keeping. It made the
working tree safe on 2026-08-14, but the scrub was only ever committed locally — the
identifiers stayed live at the pushed tip of the private dev remote for a further day,
retrievable from the host by object hash regardless of branch state. On 2026-08-16 the full
history was rewritten (67 identifiers in these two files, plus three truncated forms in
docs/IMPLEMENTATION_LOG.md and tests/test_repo_guards.py that a tail-anchored pattern had
missed) and the remote was deleted and recreated, because a force-push leaves the old blobs
fetchable by hash. Scrubbing the tree is the first step of a remediation, never the whole of
one.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TARGETS = [
    ROOT / "results" / "cross_disease_matrix.json",
    ROOT / "results" / "manual_test_results.json",
]

# rop_<id>_<sex>_GA<n>_BW<n>_PA<n>_DG<n>_PF<n>_D<dev>_S<serie>_<frame>.jpg
# Keep group 1 (patient id) and group 2 (device/series/frame); discard the clinical middle.
PHI_RE = re.compile(
    r"(?P<prefix>[A-Za-z_]*?)(?P<pid>\d+)_[MF]_GA\d+_BW\d+(?:_PA\d+)?(?:_DG\d+)?(?:_PF\d+)?"
    r"_(?P<tail>D\d_S\d+_\d+)"
)


def scrub(text: str) -> tuple[str, int]:
    n = 0

    def sub(m):
        nonlocal n
        n += 1
        return f"{m.group('prefix')}{m.group('pid')}_{m.group('tail')}"

    return PHI_RE.sub(sub, text), n


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--check", action="store_true", help="report matches, change nothing")
    g.add_argument("--write", action="store_true", help="rewrite the files in place")
    a = ap.parse_args()

    total = 0
    for p in TARGETS:
        if not p.exists():
            print(f"  (absent) {p.relative_to(ROOT)}")
            continue
        raw = p.read_text()
        out, n = scrub(raw)
        total += n
        print(f"  {p.relative_to(ROOT)}: {n} identifiers")
        if n and a.write:
            # Re-serialise through json so a malformed edit cannot pass silently.
            json.loads(out)
            p.write_text(out)

    # Fail loudly if anything clinical survives — this script is a guard as much as a fixer.
    if a.write:
        left = sum(len(re.findall(r"GA\d+_BW\d+", p.read_text()))
                   for p in TARGETS if p.exists())
        if left:
            raise SystemExit(f"REFUSING TO PASS: {left} clinical attributes still present")
        print(f"-> scrubbed {total} identifiers; 0 clinical attributes remain")
    else:
        print(f"-> {total} identifiers would be scrubbed (run with --write)")


if __name__ == "__main__":
    main()
