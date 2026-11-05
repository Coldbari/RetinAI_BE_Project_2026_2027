"""Patient-group-aware splitting — the guard for sibling-image leakage.

An image-level split leaks whenever several images share a patient: the model trains on one
eye and is scored on the fellow eye, which is nearly the same picture of the same disease at
the same visit. Measured on this repo's own data before the fix:

  * EyePACS: 35,126 images from 17,563 patients, and EVERY patient contributes BOTH eyes.
  * ROP:     6,004 images from 188 infants (median 31 images each); one infant supplies 470.

The ROP splits were already patient-grouped by data_prep/Splitting_data_with_patient_ID.py and
verify clean. DR was not. These tests pin the behaviour so it cannot regress to per-image.

Run:  .venv/bin/python -m pytest tests/test_data_prep_grouping.py -v
"""
import re
import sys
from collections import Counter
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from models.common.data_prep import (  # noqa: E402
    UNGROUPED,
    _group_of,
    _stratified_split,
    audit_group_leakage,
)


class Src(dict):
    """Stand-in for the config's attr-access source object."""
    __getattr__ = dict.get


def rows_for(pairs, source="s"):
    """pairs: [(filename_stem, label), ...] -> manifest-shaped rows, split unassigned."""
    src = Src(name=source, group_pattern=r"^(\d+)_")
    return [{"image_path": f"/d/{stem}.jpg", "label": lab, "split": None, "source": source,
             "group": _group_of(src, Path(f"/d/{stem}.jpg"))} for stem, lab in pairs]


# ── the key extractor ────────────────────────────────────────────────────────
def test_group_pattern_extracts_the_patient_id():
    src = Src(name="eyepacs", group_pattern=r"^(\d+)_")
    assert _group_of(src, Path("/d/10_left.jpeg")) == "eyepacs:10"
    assert _group_of(src, Path("/d/10_right.jpeg")) == "eyepacs:10"


def test_group_is_namespaced_by_source():
    """Two datasets both numbering patients from 1 must not be merged into one patient."""
    a = _group_of(Src(name="eyepacs", group_pattern=r"^(\d+)_"), Path("/d/1_left.jpg"))
    b = _group_of(Src(name="aptos", group_pattern=r"^(\d+)_"), Path("/d/1_left.jpg"))
    assert a != b


def test_no_pattern_means_ungrouped():
    assert _group_of(Src(name="smdg"), Path("/d/anything.png")) == UNGROUPED


def test_a_pattern_that_does_not_match_is_fatal():
    """Silently falling back to per-image is the exact bug this feature exists to prevent."""
    with pytest.raises(ValueError, match="did not match"):
        _group_of(Src(name="x", group_pattern=r"^(\d+)_"), Path("/d/no_leading_digits.jpg"))


# ── the splitter ─────────────────────────────────────────────────────────────
def test_fellow_eyes_never_land_in_different_splits():
    rows = rows_for([(f"{p}_{side}", p % 2) for p in range(60) for side in ("left", "right")])
    _stratified_split(rows, val_split=0.2, test_split=0.2, seed=0)
    df = pd.DataFrame(rows)
    assert audit_group_leakage(df)["leaked_images"] == 0
    for _, grp in df.groupby("group"):
        assert grp["split"].nunique() == 1, f"group {grp['group'].iloc[0]} spans splits"


def test_every_split_is_populated_and_every_row_assigned():
    rows = rows_for([(f"{p}_{side}", p % 2) for p in range(60) for side in ("left", "right")])
    _stratified_split(rows, val_split=0.2, test_split=0.2, seed=0)
    got = Counter(r["split"] for r in rows)
    assert set(got) == {"train", "val", "test"}
    assert sum(got.values()) == len(rows)


def test_quotas_are_filled_by_image_count_not_group_count():
    """One huge group must not swing the val fraction — the ROP case, where a single infant
    supplies 470 of 3,024 positives."""
    pairs = [(f"999_{i}", 1) for i in range(200)]                  # one 200-image patient
    pairs += [(f"{p}_0", 1) for p in range(100, 300)]              # 200 one-image patients
    rows = rows_for(pairs)
    _stratified_split(rows, val_split=0.25, test_split=0.0, seed=3)
    df = pd.DataFrame(rows)
    frac = (df["split"] == "val").mean()
    assert 0.2 <= frac <= 0.35, f"val fraction {frac:.2f} — a single group distorted the split"
    assert audit_group_leakage(df)["leaked_images"] == 0


def test_ungrouped_rows_still_split_per_image():
    """Sources with no patient key must behave exactly as before this feature existed."""
    rows = [{"image_path": f"/d/{i}.jpg", "label": i % 2, "split": None,
             "source": "smdg", "group": UNGROUPED} for i in range(100)]
    _stratified_split(rows, val_split=0.2, test_split=0.0, seed=1)
    got = Counter(r["split"] for r in rows)
    assert got["val"] > 0 and got["train"] > 0
    assert audit_group_leakage(pd.DataFrame(rows))["leaked_images"] == 0


def test_split_is_deterministic_for_a_seed():
    pairs = [(f"{p}_{s}", p % 3) for p in range(40) for s in ("l", "r")]
    a, b = rows_for(pairs), rows_for(pairs)
    _stratified_split(a, 0.2, 0.1, seed=7)
    _stratified_split(b, 0.2, 0.1, seed=7)
    assert [r["split"] for r in a] == [r["split"] for r in b]


# ── the auditor ──────────────────────────────────────────────────────────────
def test_auditor_detects_a_planted_leak():
    """If this passes with a leak present, every other test here is worthless."""
    rows = rows_for([(f"{p}_{s}", 0) for p in range(10) for s in ("l", "r")])
    _stratified_split(rows, 0.2, 0.0, seed=0)
    rows[0]["split"] = "train"
    rows[1]["split"] = "val"          # same patient, both sides — a leak by construction
    audit = audit_group_leakage(pd.DataFrame(rows))
    assert audit["leaked_groups"] >= 1 and audit["leaked_images"] >= 2


# ── the real committed data ──────────────────────────────────────────────────
def test_dr_config_keeps_the_shipped_source_set_by_default():
    if not (ROOT / "configs" / "dr.yaml").exists():
        pytest.skip("DR left the deployment (2026-08-18 ROP-only cleanup; "
                    "config parked in junk/configs/) - guard retained for any return")
    """configs/dr.yaml defines eyepacs_test (53,576 images) but must not USE it unless asked.

    build_manifest falls back to EVERY defined source when `use_sources` is absent, so merely
    adding the source block would have doubled the training pool and made results/dr/metrics.json
    unreproducible against the shipped checkpoint.
    """
    import yaml
    cfg = yaml.safe_load((ROOT / "configs" / "dr.yaml").read_text())
    defined = {s["name"] for s in cfg["data"]["sources"]}
    assert "eyepacs_test" in defined, "the EyePACS test split block went missing"
    assert cfg["data"].get("use_sources") == ["eyepacs", "aptos"], (
        "configs/dr.yaml must pin use_sources, or defining a new source silently retrains "
        "on a different corpus than the one results/dr/metrics.json describes")


def test_eyepacs_train_and_test_patients_are_not_merged():
    """Both splits number patients from 1, but they are DIFFERENT people. If the group key
    were not namespaced by source, patient 10 of train and patient 10 of test would be treated
    as one person and forced into the same split for no reason."""
    pat = r"^(\d+)_"
    assert (_group_of(Src(name="eyepacs", group_pattern=pat), Path("/d/10_left.jpeg"))
            != _group_of(Src(name="eyepacs_test", group_pattern=pat), Path("/d/10_left.jpeg")))


ROP_MANIFESTS = [ROOT / "results" / "prepared" / "rop" / n
                 for n in ("manifest.csv", "manifest_devsplit.csv")]


@pytest.mark.parametrize("man", ROP_MANIFESTS, ids=lambda p: p.name)
def test_committed_rop_split_keeps_every_infant_on_one_side(man):
    """configs/rop.yaml claims 'already split by patient ID'. Verify, do not trust."""
    if not man.exists():
        pytest.skip(f"{man.name} not present (built on Kaggle)")
    df = pd.read_csv(man)
    df["group"] = df["image_path"].map(
        lambda p: re.match(r"^(\d+)_", Path(p).name).group(1))
    audit = audit_group_leakage(df)
    assert audit["leaked_images"] == 0, (
        f"{man.name}: {audit['leaked_images']} images from {audit['leaked_groups']} infants "
        f"appear in more than one split")
