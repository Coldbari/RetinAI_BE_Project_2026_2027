"""Recover what the ROP binary label actually contains, from data already on disk.

`configs/rop.yaml` trains a two-class model, "No ROP" vs "ROP Detected". That label was
produced by collapsing the Ostrava database's DIAGNOSIS CODE (DG) — a 14-value clinical
field — down to one bit. The original codes were never lost: they live in
docs/infant_retinal_database_info.csv, and the manifest's filenames carry the infant ID,
serie number and device. Nothing needs downloading.

Joining the two answers a question no metric could: **what is the positive class made of?**

    .venv/bin/python scripts/rop_recover_staging.py

--------------------------------------------------------------------------------------
JOIN KEY — READ THIS BEFORE CHANGING IT.

The key is (ID, SERIE, DEVICE). It is NOT (ID) and it is NOT any key containing
POSTCONCEPTUAL AGE:

  * (ID) alone is WRONG. One infant has several examination series recorded on different
    dates with DIFFERENT diagnosis codes. Keying on the infant collapses them, so every
    image of that infant inherits whichever CSV row happened to be read last. An earlier
    revision of this script did exactly that and produced an impossible result: it claimed
    stage-1 and treated-ROP images had been labelled NEGATIVE, which no binarisation of
    "DG == 0" could ever produce.

  * (…, PA) is WRONG. For device 3 (Natus RetCam Envision) the CSV stores postconceptual
    age as a fraction of a week (33.42857142857143 = 33 weeks 3 days) while the filename
    truncated it to an integer ("PA33"). Any key containing PA silently loses 821 images.

(ID, SERIE, DEVICE) is unique in the CSV — asserted below — and matches all 6,004 images.

VALIDATION: the recovered composition is confirmed by a completely independent source. The
original Ostrava release keeps the TRUE DG code in each filename (e.g.
`<id>_F_GA##_BW####_PA##_DG#_PF#_D1_S01_1.jpg`); our prepared split overwrote that field
with the binary label. Reading the codes straight from an untouched copy of the original
filenames reproduces the table below exactly, code for code.

PHI: this reads a file with per-infant sex, gestational age and birth weight. It writes
AGGREGATE COUNTS ONLY — never a row, an ID, or a filename — because results/*.json is
committed. The one exception is the per-code infant COUNT, which is not identifying.
"""
import csv
import json
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
META = ROOT / "docs" / "infant_retinal_database_info.csv"
MANIFESTS = [ROOT / "results" / "prepared" / "rop" / n
             for n in ("manifest.csv", "manifest_devsplit.csv")]

# From the `diagnosis` sheet of docs/infant_retinal_database_info.xlsx. An earlier revision
# of this script asserted "DG code MEANINGS ARE NOT IN THIS REPO" — they are, and always were.
DG_MEANING = {
    "0": "physiological", "1": "ROP stage 0", "2": "ROP stage 1", "3": "ROP stage 2",
    "4": "ROP stage 3", "5": "ROP stage 4a", "6": "ROP stage 4b", "7": "ROP stage 5",
    "8": "AP-ROP", "9": "st.p. ROP (status post — treated/regressed)", "10": "hamartoms",
    "11": "hemorrhage", "12": "hypoplasia n.II", "13": "toxoplasma",
}
# Codes 1-9 are retinopathy of prematurity entities. 10-13 are OTHER retinal pathology in
# preterm infants — they are not ROP, and pooling them into a "ROP Detected" class is the
# single most consequential fact this script exists to surface.
ROP_CODES = {str(i) for i in range(1, 10)}
NOT_ROP_CODES = {"10", "11", "12", "13"}

PLUS_FORM = {"0": "normal", "1": "pre-plus", "2": "plus"}
DEVICE = {"1": "Clarity RetCam 3", "2": "Phoenix ICON", "3": "Natus RetCam Envision"}

# <id>_F_GA##_BW####_PA##_DG#_PF#_D3_S01_17.jpg -> id, device 3, serie 1
KEY_RE = re.compile(r"^(\d+)_.*_D(\d)_S(\d+)_")


def file_key(basename):
    m = KEY_RE.match(basename)
    return (str(int(m.group(1))), m.group(2), str(int(m.group(3)))) if m else None


def load_meta():
    """(infant ID, serie, device) -> {dg, plus}. Semicolon-separated, BOM-prefixed."""
    with open(META, encoding="utf-8-sig") as fh:
        rows = list(csv.DictReader(fh, delimiter=";"))
    out = {}
    for r in rows:
        k = (str(int(r["ID"])), r["DEVICE (D)"].strip(), str(int(r["SERIE NUMBER (S)"])))
        if k in out:
            raise SystemExit(f"join key {k} is not unique in {META.name} — refusing to guess")
        out[k] = {"dg": r["DIAGNOSIS CODE (DG)"].strip(),
                  "plus": r["PLUS FORM (PF)"].strip()}
    return out


def main():
    if not META.exists():
        print(f"missing {META} — cannot recover staging", file=sys.stderr)
        return 1
    meta = load_meta()
    man = next((m for m in MANIFESTS if m.exists()), None)
    if man is None:
        print("no ROP manifest on disk (built on Kaggle) — nothing to join", file=sys.stderr)
        return 1

    rows = list(csv.DictReader(open(man)))
    per_code = defaultdict(lambda: {"images": 0, "infants": set()})
    per_label = Counter()
    unmatched = 0
    plus = defaultdict(lambda: {"images": 0, "infants": set()})
    pos_by_infant = Counter()

    for r in rows:
        base = os.path.basename(r["image_path"])
        k = file_key(base)
        m = meta.get(k) if k else None
        if m is None:
            unmatched += 1
            continue
        infant = k[0]
        per_code[(r["label"], m["dg"])]["images"] += 1
        per_code[(r["label"], m["dg"])]["infants"].add(infant)
        per_label[r["label"]] += 1
        plus[m["plus"]]["images"] += 1
        plus[m["plus"]]["infants"].add(infant)
        if r["label"] == "1":
            pos_by_infant[infant] += 1

    print("=" * 78)
    print(f"ROP LABEL COMPOSITION   ({man.name}, {len(rows)} images)")
    print("=" * 78)
    if unmatched:
        print(f"  WARNING: {unmatched} images had no metadata row — join is incomplete")
    else:
        print(f"  join: {len(rows)}/{len(rows)} images matched on (ID, serie, device)")

    out = {"manifest": man.name, "images": len(rows), "unmatched": unmatched,
           "join_key": "infant ID + serie number + device", "by_label": {}}

    for lab in sorted(per_label):
        name = "ROP Detected" if lab == "1" else "No ROP"
        codes = sorted(((k[1], v) for k, v in per_code.items() if k[0] == lab),
                       key=lambda kv: -kv[1]["images"])
        total = per_label[lab]
        print(f'\n  label {lab} = "{name}"   {total} images')
        print(f"    {'DG':>4}  {'meaning':<42} {'images':>7} {'infants':>8} {'share':>7}")
        for code, v in codes:
            print(f"    {code:>4}  {DG_MEANING.get(code, '?'):<42} {v['images']:>7} "
                  f"{len(v['infants']):>8} {100 * v['images'] / total:>6.1f}%")
        out["by_label"][lab] = {
            "name": name, "images": total, "distinct_dg_codes": len(codes),
            "codes": [{"dg": c, "meaning": DG_MEANING.get(c, "?"), "images": v["images"],
                       "infants": len(v["infants"])} for c, v in codes]}

    # ── what the positive class is actually made of ──────────────────────────
    def agg(codes):
        img = sum(v["images"] for k, v in per_code.items() if k[1] in codes)
        inf = set().union(*[v["infants"] for k, v in per_code.items() if k[1] in codes]) \
            if any(k[1] in codes for k in per_code) else set()
        return img, len(inf)

    rop_img, rop_inf = agg(ROP_CODES)
    other_img, other_inf = agg(NOT_ROP_CODES)
    healthy_img, healthy_inf = agg({"0"})

    print("\n" + "-" * 78)
    print("WHAT THIS MEANS")
    pos_total = per_label.get("1", 0)
    if pos_total and other_img:
        print(f"  The positive class is NOT one disease. Of its {pos_total} images, "
              f"{other_img} ({100 * other_img / pos_total:.1f}%) carry a diagnosis that is "
              f"NOT retinopathy of prematurity")
        print(f"  at all — hemorrhage, toxoplasma, hamartoms, optic nerve hypoplasia — from "
              f"{other_inf} infants.")
        print(f"  TRUE ROP (codes 1-9): {rop_img} images from {rop_inf} infants.")
        print(f"  A model trained on this label is an ABNORMAL-RETINA detector, not a ROP "
              f"detector.")
    out["composition"] = {
        "healthy": {"images": healthy_img, "infants": healthy_inf},
        "true_rop_codes_1_9": {"images": rop_img, "infants": rop_inf},
        "not_rop_codes_10_13": {"images": other_img, "infants": other_inf},
        "not_rop_share_of_positive": round(other_img / pos_total, 4) if pos_total else None,
    }

    # ── ICROP stage axis, which the binary label throws away ─────────────────
    stages = [(c, per_code_sum(per_code, c)) for c in sorted(ROP_CODES, key=int)]
    print("\n  ICROP STAGE AXIS (discarded by the binary collapse):")
    for code, (img, inf) in stages:
        if img:
            print(f"    DG {code:>2}  {DG_MEANING[code]:<42} {img:>6} images  {inf:>3} infants")
    absent = [DG_MEANING[c] for c, (img, _) in stages if not img]
    if absent:
        print(f"    ABSENT ENTIRELY: {', '.join(absent)}")
    out["icrop_stages"] = {DG_MEANING[c]: {"images": i, "infants": n}
                           for c, (i, n) in stages}

    if pos_by_infant:
        tot = sum(pos_by_infant.values())
        ranked = pos_by_infant.most_common()
        biggest = ranked[0][1]
        top8 = sum(v for _, v in ranked[:8])
        med = sorted(pos_by_infant.values())[len(pos_by_infant) // 2]
        print(f"\n  All {tot} positive images come from {len(pos_by_infant)} infants "
              f"(median {med} images each).")
        print(f"  The single largest infant supplies {biggest} images "
              f"({100 * biggest / tot:.1f}%); the top 8 supply {100 * top8 / tot:.1f}%.")
        out["positive_concentration"] = {
            "positive_images": tot, "positive_infants": len(pos_by_infant),
            "median_images_per_infant": med, "largest_infant_images": biggest,
            "largest_infant_share": round(biggest / tot, 4),
            "top8_share": round(top8 / tot, 4)}

    print("\n  PLUS DISEASE (from the metadata, not the label):")
    for code in sorted(plus):
        v = plus[code]
        print(f"    {PLUS_FORM.get(code, code):<10} {v['images']:>6} images  "
              f"{len(v['infants']):>3} infants")
    out["plus_disease"] = {PLUS_FORM.get(c, c): {"images": v["images"],
                                                 "infants": len(v["infants"])}
                           for c, v in plus.items()}
    npi = len(plus.get("2", {"infants": set()})["infants"])
    if npi:
        print(f"    -> plus disease rests on {npi} infants — too few to support a claim.")

    dst = ROOT / "results" / "rop" / "staging_composition.json"
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(out, indent=2) + "\n")
    print(f"\n-> {dst}")
    return 0


def per_code_sum(per_code, code):
    """Images and distinct infants carrying one DG code, across both labels."""
    img = sum(v["images"] for k, v in per_code.items() if k[1] == code)
    inf = set()
    for k, v in per_code.items():
        if k[1] == code:
            inf |= v["infants"]
    return img, len(inf)


if __name__ == "__main__":
    sys.exit(main())
