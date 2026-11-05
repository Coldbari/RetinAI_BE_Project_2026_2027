"""v4 training pool (PROTOCOL_V2 Amendment 5a items 3-5, with the 5a revision): per-device
Ostrava quality, caps re-ranked by blur_rank_in_device with the (infant, class) floor of 2,
cross-fold true-duplicate exclusion, LOSO folds. Selection rules run on inline frames; the
fold files are built from the real pool into tmp and compared with the committed v3 files."""
from pathlib import Path

import pandas as pd
import pytest

from scripts.build_rop_v4_pool import (NEAR, QUALITY, THRESHOLDS, select_training_rows,
                                      write_folds, write_v3_loso_folds)
from scripts.rop_v4_audit import SPLITS, load_pool

ROOT = Path(__file__).resolve().parent.parent
V3 = ROOT / "results" / "prepared" / "rop_staging"
THR = {"rule": "test", "by_device": {"cam": {"min_blur": 10.0, "lum_lo": 50.0, "lum_hi": 150.0}}}
NO_NEAR = pd.DataFrame(columns=["image_a", "image_b", "distance", "cross_fold", "true_duplicate",
                                "pixel_mad"])


def _row(path, pid, cls6, rank, source="ostrava", fold=0, blur=100.0, lum=100.0, empty=False):
    return dict(image_path=path, source=source, patient_id=pid, cls6=cls6, fold=fold, kept=True,
                site_val=source == "shenzhen", device="cam" if source == "ostrava" else "unknown",
                blur_var=blur, luminance=lum, empty=empty, blur_rank_in_device=rank)


def _frames(rows):
    df = pd.DataFrame(rows)
    pool = df[["image_path", "source", "patient_id", "cls6", "device", "fold", "site_val", "kept"]]
    quality = df[["image_path", "source", "device", "blur_var", "luminance", "empty",
                  "blur_rank_in_device"]]
    return pool, quality


def _kept(sel):
    return set(sel[sel.kept_v4].image_path)


def test_ostrava_rows_failing_quality_never_train():
    rows = [_row(f"o/ok{i}.jpg", "ostrava:1", "stage_2", 0.5 - i / 10) for i in range(3)]
    rows += [_row("o/dark.jpg", "ostrava:1", "stage_2", 0.99, lum=10.0),     # sharpest, too dark
             _row("o/bright.jpg", "ostrava:1", "stage_2", 0.98, lum=200.0),
             _row("o/empty.jpg", "ostrava:1", "stage_2", 0.97, empty=True),
             _row("o/blur.jpg", "ostrava:1", "stage_2", 0.01, blur=5.0)]
    # ROP-VL and Shenzhen are never quality-filtered, and their device has no thresholds
    rows += [_row("r/hazy.jpg", "ropvl:1", "stage_2", 0.1, source="ropvl", blur=1.0, lum=10.0),
             _row("s/dim.jpg", "shenzhen:1", "stage_1", 0.2, source="shenzhen", lum=5.0)]
    pool, quality = _frames(rows)
    sel = select_training_rows(pool, quality, THR, NO_NEAR)
    assert _kept(sel) == {"o/ok0.jpg", "o/ok1.jpg", "o/ok2.jpg", "r/hazy.jpg", "s/dim.jpg"}
    assert not sel[(sel.source == "ostrava") & ~sel.quality_pass].kept_v4.any()
    assert sel[sel.source != "ostrava"].quality_pass.all()
    assert list(sel.image_path) == list(pool.image_path)          # pool order preserved


def test_cap_keeps_sharpest_frames_and_never_empties_an_infant_class():
    rows = [_row(f"o/s2_{i:02d}.jpg", "ostrava:2", "stage_2", i / 100) for i in range(25)]
    # normal cap 2; a rank tie is broken by image_path
    rows += [_row(f"o/n{c}.jpg", "ostrava:3", "normal", r)
             for c, r in zip("abcde", [0.2, 0.9, 0.9, 0.5, 0.95])]
    # 0 passing: the 2 best-ranked frames
    rows += [_row(f"o/z{i}.jpg", "ostrava:4", "stage_3", r, lum=10.0)
             for i, r in enumerate([0.3, 0.8, 0.6])]
    # 1 passing (the least sharp): still the 2 best-ranked frames overall
    rows += [_row("o/one_pass.jpg", "ostrava:5", "arop", 0.1),
             _row("o/one_f1.jpg", "ostrava:5", "arop", 0.7, lum=300.0),
             _row("o/one_f2.jpg", "ostrava:5", "arop", 0.4, empty=True)]
    # a single failing frame: all of it
    rows += [_row("o/lone.jpg", "ostrava:6", "stage_1", 0.2, blur=1.0)]
    # 2 passing: only the passing frames, even though the failing one is sharper
    rows += [_row("o/p1.jpg", "ostrava:7", "stage_1", 0.3), _row("o/p2.jpg", "ostrava:7", "stage_1", 0.2),
             _row("o/pf.jpg", "ostrava:7", "stage_1", 0.9, lum=10.0)]
    # Shenzhen over cap: re-ranked by blur_rank_in_device as well, no quality filter
    rows += [_row(f"s/n{i}.jpg", "shenzhen:9", "normal", r, source="shenzhen", lum=1.0)
             for i, r in enumerate([0.1, 0.8, 0.5])]
    pool, quality = _frames(rows)
    sel = select_training_rows(pool, quality, THR, NO_NEAR)
    k = _kept(sel)

    assert {p for p in k if p.startswith("o/s2_")} == {f"o/s2_{i:02d}.jpg" for i in range(5, 25)}
    assert {p for p in k if p.startswith("o/n")} == {"o/ne.jpg", "o/nb.jpg"}
    assert {p for p in k if p.startswith("o/z")} == {"o/z1.jpg", "o/z2.jpg"}
    assert {p for p in k if p.startswith("o/one")} == {"o/one_f1.jpg", "o/one_f2.jpg"}
    assert "o/lone.jpg" in k
    assert {p for p in k if p.startswith("o/p")} == {"o/p1.jpg", "o/p2.jpg"}
    assert {p for p in k if p.startswith("s/")} == {"s/n1.jpg", "s/n2.jpg"}

    floor = set(sel[sel.floor_cell].patient_id)
    assert floor == {"ostrava:4", "ostrava:5", "ostrava:6"}
    cells = sel.groupby(["patient_id", "cls6"]).kept_v4.sum()
    assert (cells >= 1).all(), "an infant lost a class"


def test_cross_fold_near_duplicates_absent_from_train_present_in_val(built):
    rows = [_row("o/a.jpg", "ostrava:1", "stage_2", 0.9, fold=0),
            _row("o/b.jpg", "ostrava:2", "stage_2", 0.9, fold=1),
            _row("o/c.jpg", "ostrava:1", "stage_2", 0.5, fold=0),
            _row("o/d.jpg", "ostrava:2", "stage_2", 0.5, fold=1),
            _row("o/e.jpg", "ostrava:1", "stage_2", 0.4, fold=0),
            _row("o/f.jpg", "ostrava:1", "stage_2", 0.3, fold=0),
            # normal cap 2: the excluded sharpest frame does not use up a cap slot
            _row("o/x.jpg", "ostrava:3", "normal", 0.9, fold=2),
            _row("o/y.jpg", "ostrava:3", "normal", 0.5, fold=2),
            _row("o/z.jpg", "ostrava:3", "normal", 0.4, fold=2),
            _row("s/x2.jpg", "shenzhen:4", "normal", 0.9, source="shenzhen", fold=3)]
    near = pd.DataFrame({
        "image_a": ["o/a.jpg", "o/c.jpg", "o/e.jpg", "o/x.jpg"],
        "image_b": ["o/b.jpg", "o/d.jpg", "o/f.jpg", "s/x2.jpg"],
        "distance": [0, 3, 1, 2], "cross_fold": [True, True, False, True],
        "true_duplicate": [True, False, True, True], "pixel_mad": [0.0, 20.0, 1.0, 2.0]})
    pool, quality = _frames(rows)
    sel = select_training_rows(pool, quality, THR, near)
    assert _kept(sel) == {"o/c.jpg", "o/d.jpg", "o/e.jpg", "o/f.jpg", "o/y.jpg", "o/z.jpg"}
    assert set(sel[sel.near_dup].image_path) == {"o/a.jpg", "o/b.jpg", "o/x.jpg", "s/x2.jpg"}

    # real pool: every excluded image is in no train side and in exactly one CV val side
    ex = pd.read_csv(built / "exclude_near_duplicates.csv")
    assert list(ex.columns) == ["image_path", "pair_with", "distance", "pixel_mad"]
    audit = pd.read_csv(NEAR)
    x = audit[audit.cross_fold & audit.true_duplicate]
    assert set(ex.image_path) == set(x.image_a) | set(x.image_b)
    assert len(ex) > 0
    for name in [f"fold_{k}" for k in range(5)] + ["fold_site", "fold_loso_ostrava",
                                                     "fold_loso_ropvl"]:
        d = pd.read_csv(built / f"{name}.csv")
        assert not set(ex.image_path) & set(d[d.split == "train"].image_path), name
    in_val = pd.concat(pd.read_csv(built / f"fold_{k}.csv").query("split == 'val'")
                       for k in range(5)).image_path.value_counts()
    assert all(in_val.get(p, 0) == 1 for p in ex.image_path)


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    needed = [SPLITS, QUALITY, THRESHOLDS, NEAR]
    if not all(p.exists() for p in needed):
        pytest.skip("splits.csv or the v4 audit outputs are not present")
    out = tmp_path_factory.mktemp("v4")
    write_folds(out)
    return out


def _val_lines(path):
    lines = path.read_text().splitlines()
    return lines[0], [ln for ln in lines[1:] if ln.split(",")[2] == "val"]


@pytest.mark.parametrize("name", [f"fold_{k}" for k in range(5)] + ["fold_site"])
def test_val_rows_identical_to_v3_for_every_fold(built, name):
    head4, val4 = _val_lines(built / f"{name}.csv")
    head3, val3 = _val_lines(V3 / f"{name}.csv")
    assert head4 == head3 == "image_path,label,split,patient_id,source,cls6"
    assert val4 == val3                       # same rows, same order, byte for byte
    assert len(val3) > 0
    tr = pd.read_csv(built / f"{name}.csv").query("split == 'train'")
    assert not tr.image_path.str.startswith("/").any()
    assert not set(tr.patient_id) & set(pd.read_csv(V3 / f"{name}.csv")
                                        .query("split == 'val'").patient_id)


@pytest.mark.parametrize("name,src", [("fold_loso_ostrava", "ostrava"),
                                      ("fold_loso_ropvl", "ropvl")])
def test_loso_folds_partition_sources(built, name, src):
    pool = load_pool()
    d = pd.read_csv(built / f"{name}.csv")
    tr, va = d[d.split == "train"], d[d.split == "val"]
    assert list(va.image_path) == list(pool[pool.source == src].image_path)   # every row, in order
    assert set(va.source) == {src}
    assert src not in set(tr.source)
    assert not set(tr.patient_id) & set(va.patient_id)
    # train = every v4 training row of the other two sources (each trains in 4 of 5 CV folds)
    v4_train = pd.concat(pd.read_csv(built / f"fold_{k}.csv").query("split == 'train'")
                         for k in range(5)).drop_duplicates("image_path")
    assert set(tr.image_path) == set(v4_train[v4_train.source != src].image_path)
    assert len(tr) == tr.image_path.nunique()


# ── fix round 1 (Task 7): arm B's LOSO folds train on the v3 pool ─────────────────
@pytest.fixture(scope="module")
def built_v3_loso(built, tmp_path_factory):
    out = tmp_path_factory.mktemp("v3")
    write_v3_loso_folds(out, v4_dir=built)
    return out


@pytest.mark.parametrize("name,src", [("fold_loso_ostrava", "ostrava"),
                                      ("fold_loso_ropvl", "ropvl")])
def test_v3_loso_folds_train_on_the_v3_pool_and_share_the_v4_val_side(built, built_v3_loso,
                                                                       name, src):
    head3, val3 = _val_lines(built_v3_loso / f"{name}.csv")
    head4, val4 = _val_lines(built / f"{name}.csv")
    assert head3 == head4 == "image_path,label,split,patient_id,source,cls6"
    assert val3 == val4 and len(val3) > 0          # same rows, same order, byte for byte
    pool = load_pool()
    tr = pd.read_csv(built_v3_loso / f"{name}.csv").query("split == 'train'")
    # train = the v3 kept rows of the other two sources, in pool order ...
    assert list(tr.image_path) == list(pool[pool.kept & (pool.source != src)].image_path)
    assert not tr.image_path.str.startswith("/").any()
    # ... which is every row a committed v3 CV fold trains on, from those two sources
    v3_train = pd.concat(pd.read_csv(V3 / f"fold_{k}.csv").query("split == 'train'")
                         for k in range(5)).drop_duplicates("image_path")
    assert set(tr.image_path) == set(v3_train[v3_train.source != src].image_path)


@pytest.mark.parametrize("name", ["fold_loso_ostrava", "fold_loso_ropvl"])
def test_committed_v3_loso_files_match_a_fresh_build(built_v3_loso, name):
    assert (V3 / f"{name}.csv").read_text() == (built_v3_loso / f"{name}.csv").read_text()
