r"""Rebuild data/rop/ostrava/ with de-identified filenames.

    .venv/bin/python scripts/anonymise_ostrava_tree.py

Ostrava ships each image named

    <id>_F_GA##_BW####_PA##_DG#_PF#_D3_S01_17.jpg
     |  |    |      |     |
     |  |    |      |     +-- postnatal age, days
     |  |    |      +-------- birth weight, grams
     |  |    +--------------- gestational age, weeks
     |  +-------------------- sex
     +----------------------- infant id

so every path is a five-field record about a premature infant. scripts/build_kaggle_bundle.py
refuses to ship anything matching /\d+_[MF]_GA\d+_BW\d+/ — including, in as many words, "a
patient-encoded path in a manifest" — and the fold CSVs are manifests full of these paths.
That guard exists because saved error-analysis crops once leaked to Kaggle through an
exclusion list written from memory.

The staging pipeline needs exactly three fields from the name: infant id (to group patients),
device and serie (to join the ICROP diagnosis code, which varies per session). Sex, GA, BW and
PA are never read. So the consolidated tree keeps only what is used:

    train/DG0/183_D3_S01_17.jpg

This does NOT touch dataset_split/, which stays as the untouched original — scripts that read
it (rop_recover_staging.py, the binary screener's splits) keep working unchanged. The files
are hardlinks, so the rename costs no disk.

The original->anonymous mapping is written OUTSIDE data/rop/ and is gitignored, because the
mapping itself contains the identifiers the rename removes. Uploading data/rop/ never carries
it along.
"""
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "dataset_split"
DST = ROOT / "data" / "rop" / "ostrava"
MAP = ROOT / "results" / "prepared" / "rop_staging" / "ostrava_name_map.csv"

# <id>_F_GA##_BW####_PA##_DG#_PF#_D3_S01_17.jpg -> id, device 3, serie 01, frame 17
SRC_RE = re.compile(r"^(\d+)_[MF]_GA\d+_BW\d+_PA\d+_DG\d+_PF\d+_D(\d)_S(\d+)_(\d+)\.jpg$", re.I)
PHI = re.compile(r"\d+_[MF]_GA\d+_BW\d+", re.I)


def main():
    if not SRC.is_dir():
        print(f"missing {SRC}", file=sys.stderr)
        return 1
    if DST.is_symlink():
        # An earlier build pointed data/rop/ostrava at dataset_split/ as a symlink. That is
        # not uploadable (symlinks do not survive a Kaggle dataset) and leaves half the corpus
        # outside data/rop/, so replace it outright.
        DST.unlink()
        print("removed the old data/rop/ostrava symlink")
    elif DST.is_dir():
        for dp, _, files in os.walk(DST):
            for f in files:
                os.unlink(os.path.join(dp, f))
        print("cleared the previous data/rop/ostrava tree")

    rows, seen, skipped = [], {}, []
    for dp, _, files in os.walk(SRC):
        rel_dir = os.path.relpath(dp, SRC)
        for f in sorted(files):
            if not f.lower().endswith(".jpg"):
                continue
            m = SRC_RE.match(f)
            if not m:
                skipped.append(os.path.join(rel_dir, f))
                continue
            iid, dev, serie, frame = m.groups()
            new = f"{int(iid)}_D{dev}_S{serie}_{frame}.jpg"
            key = os.path.join(rel_dir, new)
            if key in seen:
                # (id, device, serie, frame) must be unique or the rename would silently
                # collapse two distinct photographs into one.
                print(f"COLLISION: {f} and {seen[key]} both map to {new}", file=sys.stderr)
                return 1
            seen[key] = f
            out = DST / rel_dir / new
            out.parent.mkdir(parents=True, exist_ok=True)
            os.link(os.path.join(dp, f), out)
            rows.append((os.path.join(rel_dir, f), key))

    if skipped:
        print(f"WARNING: {len(skipped)} file(s) did not match the Ostrava name pattern and "
              f"were NOT copied, e.g. {skipped[:3]}", file=sys.stderr)

    MAP.parent.mkdir(parents=True, exist_ok=True)
    with open(MAP, "w", encoding="utf-8") as fh:
        fh.write("original,anonymised\n")
        for a, b in rows:
            fh.write(f"{a},{b}\n")

    leaked = [k for k in seen if PHI.search(k)]
    print(f"\n  linked  {len(rows):,} images into {DST.relative_to(ROOT)}")
    print(f"  map     {MAP.relative_to(ROOT)}  (gitignored — it holds the identifiers)")
    print(f"  PHI-pattern names remaining in the tree: {len(leaked)}")
    if leaked:
        print(f"  FAILED: {leaked[:3]}", file=sys.stderr)
        return 1
    print("\n  dataset_split/ is untouched; scripts reading it are unaffected.")
    print("  Now re-run: build_rop_staging_manifest.py -> build_rop_splits.py -> "
          "make_staging_folds.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
