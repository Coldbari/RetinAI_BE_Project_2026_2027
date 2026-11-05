"""The inclusion/exclusion cascade, reconstructed from the artifacts rather than remembered.

    .venv/bin/python scripts/rop_inclusion_flow.py

ISBI reviewers expect a CONSORT-style flow: how many images were acquired, how many were
excluded, for exactly what reason, and how many reached each analysis. This project had the
end points documented (13,037 acquired · 8,024 eligible) and nothing in between, which meant
the difference — 5,013 images, 38% of everything downloaded — was undocumented.

Every layer here is recomputed from committed artifacts (data/rop/PROVENANCE.json,
results/prepared/rop_staging/{manifest,splits}.csv), and every layer is required to close
arithmetically: if the exclusions do not sum to the drop, the script fails rather than
printing a flow that looks tidy and is wrong. That check is the point — an inclusion diagram
whose numbers do not reconcile is worse than none, because it launders a guess.

Output: results/rop/inclusion_flow.json
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
PREP = ROOT / "results" / "prepared" / "rop_staging"
OUT = ROOT / "results" / "rop" / "inclusion_flow.json"

# Sources carrying ICROP STAGE labels. The rest of the corpus is real data that answers a
# different question: segmentation masks, plus-disease-only grades, or unlabelled samples.
STAGE_SOURCES = {"ostrava", "shenzhen", "rop_vl", "multiview"}
NON_STAGE_REASON = {
    "farfum_rop": "plus/pre-plus grades only — no ICROP stage",
    "hvdropdb": "ridge segmentation masks — no stage label",
    "shantou_od": "optic-disc segmentation only",
    "shantou_bv": "vessel segmentation only",
    "orvs_seg": "vessel segmentation supplement",
    "ropstages_sample": "sample too small to fold (91 images) and provenance unverified",
    "coph100_annotations": "annotations only — adds no photographs",
}
# Stage-labelled images excluded from the 6-class task, with the clinical reason.
CLASS_EXCLUSIONS = {
    "other_pathology": "not ROP — hemorrhage, toxoplasma, hamartoma, optic-nerve hypoplasia; "
                       "these are the 43% of the binary corpus's 'ROP-positive' that is not ROP",
    "treated": "st.p. ROP (status post — treated or regressed). Post-treatment appearance is "
               "not an ICROP stage; including it would teach the model to call treated eyes "
               "diseased at whatever stage they once were",
    "stage_0": "ICROP stage 0 is immature retina WITHOUT ROP — it belongs with normal or "
               "nowhere, and at 45 images cannot support its own class",
}


def check(label, expected, parts):
    got = sum(parts.values())
    if got != expected:
        raise SystemExit(f"FLOW DOES NOT RECONCILE at '{label}': reasons sum to {got}, "
                         f"drop was {expected}. Difference {expected - got}. Fix the "
                         f"accounting before publishing a diagram of it.")
    return {"total": expected, "by_reason": parts}


def main():
    prov = json.loads((ROOT / "data" / "rop" / "PROVENANCE.json").read_text())
    by_folder = {d["folder"]: d["images"] for d in prov["datasets"] if d["folder"] != "_redundant"}
    acquired = prov["total_images_excluding_redundant"]

    manifest = pd.read_csv(PREP / "manifest.csv")
    splits = pd.read_csv(PREP / "splits.csv")

    # ── Layer 1: acquired -> stage-labelled sources ───────────────────────────
    non_stage = {k: by_folder[k] for k in NON_STAGE_REASON if k in by_folder}
    stage_raw = sum(v for k, v in by_folder.items() if k in STAGE_SOURCES)
    dedupe = stage_raw - len(manifest)
    layer1 = check("acquired -> manifest", acquired - len(manifest),
                   {**non_stage, "shenzhen_duplicates_of_ropvl": dedupe})

    # ── Layer 2: manifest -> stageable (the 6-class task) ─────────────────────
    excl2 = manifest[~manifest.image_path.isin(set(splits.image_path))]
    layer2 = check("manifest -> eligible", len(manifest) - len(splits),
                   excl2.icrop.value_counts().to_dict())

    # ── Layer 3: eligible -> analysis sets ────────────────────────────────────
    sets = {}
    for name, sub in splits.groupby("split"):
        sets[name] = {"images": int(len(sub)), "infants": int(sub.patient_id.nunique())}

    # ── Layer 4: frame caps inside the CV pool ────────────────────────────────
    cv = splits[splits.split == "train_cv"]
    capped = cv[~cv.kept]

    flow = {
        "what": "CONSORT-style inclusion/exclusion cascade for the ROP staging corpus. Every "
                "layer recomputed from committed artifacts; the script refuses to emit a "
                "flow whose exclusion reasons do not sum to the observed drop.",
        "layer_0_acquired": {
            "images": acquired,
            "sources": len(by_folder),
            "note": "public research datasets plus one private hospital corpus (Ostrava). "
                    "Excludes the _redundant quarantine (re-uploads and mirrors of sources "
                    "already counted).",
        },
        "layer_1_stage_labelled_sources": {
            "kept": {"images": int(len(manifest)),
                     "infants": int(manifest.patient_id.nunique()),
                     "sources": sorted(manifest.source.unique().tolist())},
            "excluded": layer1,
            "reasons": {**NON_STAGE_REASON,
                        "shenzhen_duplicates_of_ropvl":
                            "perceptual-hash duplicates: the same photographs re-encoded at "
                            "lower resolution; the ROP-VL copy is kept"},
        },
        "layer_2_six_class_eligible": {
            "kept": {"images": int(len(splits)),
                     "infants": int(splits.patient_id.nunique())},
            "excluded": layer2,
            "reasons": CLASS_EXCLUSIONS,
        },
        "layer_3_analysis_sets": {
            "sets": sets,
            "note": "multiview_locked has never been opened. multiview_dev is the only "
                    "held-out set inspected during development; the CV pool is split "
                    "patient-grouped into 5 folds plus a leave-Shenzhen-out site fold.",
        },
        "layer_4_frame_caps_in_cv_pool": {
            "pool_images": int(len(cv)),
            "trained_on": int(cv.kept.sum()),
            "capped_out": int(len(capped)),
            "rule": "at most 20 frames per infant per class, and at most 2 normal frames per "
                    "infant. Ostrava images every infant repeatedly in one session, so "
                    "without a cap a handful of infants would supply most of the gradient.",
            "capped_by_class": capped.cls6.value_counts().to_dict(),
            "note": "Capped frames are excluded from TRAINING only. Validation scores every "
                    "eligible image in the fold, which is why AP-ROP validation is dominated "
                    "by one heavily-imaged infant even though training is not.",
        },
        "headline": (f"{acquired} images acquired -> {len(manifest)} with ICROP stage labels "
                     f"-> {len(splits)} eligible for the 6-class task "
                     f"({splits.patient_id.nunique()} infants) -> {int(cv.kept.sum())} used "
                     f"for training after per-infant frame caps."),
    }

    OUT.write_text(json.dumps(flow, indent=2) + "\n")
    print(f"  acquired                      {acquired:>6}")
    print(f"  - non-stage-labelled sources  {sum(non_stage.values()):>6}  "
          f"({len(non_stage)} datasets: segmentation, plus-only, too small)")
    print(f"  - shenzhen/ROP-VL duplicates  {dedupe:>6}")
    print(f"  = stage-labelled manifest     {len(manifest):>6}  "
          f"({manifest.patient_id.nunique()} infants)")
    for k, v in layer2["by_reason"].items():
        print(f"  - {k:<27} {v:>6}")
    print(f"  = eligible, 6-class           {len(splits):>6}  "
          f"({splits.patient_id.nunique()} infants)")
    for k, v in sets.items():
        print(f"      {k:<24} {v['images']:>6}  ({v['infants']} infants)")
    print(f"  - frame-capped (training only) {len(capped):>5}")
    print(f"  = trained on                  {int(cv.kept.sum()):>6}")
    print(f"-> {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
