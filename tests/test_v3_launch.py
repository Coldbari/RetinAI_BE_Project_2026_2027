"""Pure helpers behind the v3 Kaggle kernel: inner-probe split and the launch checks."""
from pathlib import Path

import pandas as pd
import pytest

from models.common.v3_launch import (check_cache_flags, check_cached_files,
                                     inner_probe_manifest)


def _fold0():
    rows = []
    for p in range(50):
        for i in range(4):
            rows.append(dict(image_path=f"data/rop/ostrava/train/p{p}_{i}.jpg", label=p % 3,
                             split="train", patient_id=f"ostrava:{p}", source="ostrava",
                             cls6="x"))
    for p in range(50, 60):
        rows.append(dict(image_path=f"data/rop/ostrava/val/p{p}.jpg", label=0, split="val",
                         patient_id=f"ostrava:{p}", source="ostrava", cls6="x"))
    return pd.DataFrame(rows)


def test_inner_split_patient_disjoint_80_20_and_never_uses_fold0_val():
    df = _fold0()
    out = inner_probe_manifest(df, seed=0)
    fold0_val = set(df[df.split == "val"].image_path)
    assert not fold0_val & set(out.image_path)
    tr, va = out[out.split == "train"], out[out.split == "val"]
    assert not set(tr.patient_id) & set(va.patient_id)
    assert len(tr) + len(va) == 200
    assert 0.15 <= len(va) / len(out) <= 0.25


def test_inner_split_is_deterministic():
    a = inner_probe_manifest(_fold0(), seed=0)
    b = inner_probe_manifest(_fold0(), seed=0)
    assert a.equals(b)


def test_cache_flags_mismatch_raises():
    info = dict(circle_crop=True, letterbox=True, clahe=True, ben_graham=False)
    cfg = dict(circle_crop=True, letterbox=True, clahe=True, ben_graham=False)
    check_cache_flags(info, cfg)
    with pytest.raises(SystemExit, match="clahe"):
        check_cache_flags({**info, "clahe": False}, cfg)


def test_missing_cached_file_raises_with_count(tmp_path):
    f = tmp_path / "rop" / "ostrava" / "train" / "a.jpg"
    f.parent.mkdir(parents=True)
    f.write_bytes(b"x")
    df = pd.DataFrame({"image_path": ["data/rop/ostrava/train/a.jpg",
                                      "data/rop/ostrava/train/b.jpg",
                                      "data/rop/ostrava/train/c.jpg"]})
    with pytest.raises(SystemExit, match="2 "):
        check_cached_files(df, str(tmp_path))
    check_cached_files(df.iloc[:1], str(tmp_path))


def test_inner_split_on_committed_fold0():
    from pathlib import Path
    root = Path(__file__).resolve().parent.parent
    df = pd.read_csv(root / "results/prepared/rop_staging/fold_0.csv")
    out = inner_probe_manifest(df, seed=0)
    tr, va = out[out.split == "train"], out[out.split == "val"]
    assert len(tr) and len(va)
    assert not set(tr.patient_id) & set(va.patient_id)
    fold0_val = set(df[df.split == "val"].image_path)
    assert not fold0_val & (set(tr.image_path) | set(va.image_path))


def test_duplicates_dropped_from_train_only_on_committed_folds():
    from pathlib import Path
    from models.common.v3_launch import drop_duplicate_train_rows
    prep = Path(__file__).resolve().parent.parent / "results/prepared/rop_staging"
    ex = pd.read_csv(prep / "exclude_duplicates.csv")
    dups = set(ex.image_path)
    dropped = 0
    for f in ["fold_0", "fold_1", "fold_2", "fold_3", "fold_4", "fold_site"]:
        df = pd.read_csv(prep / f"{f}.csv")
        out = drop_duplicate_train_rows(df, ex)
        assert not dups & set(out[out.split == "train"].image_path)
        # val rows untouched, same order: val_predictions.npz must still align with the pool
        assert out[out.split == "val"].image_path.tolist() == df[df.split == "val"].image_path.tolist()
        dropped += len(df) - len(out)
    assert dropped == 12 + 9 + 11 + 10 + 14 + 2


# ── Task 7 (Amendment 5): the POOL and LOSO knobs ───────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
KERNEL = ROOT / "kaggle_kernels/rop_v3_finetune.py"
PREP = "results/prepared/rop_staging"


def test_job_list_order_and_loso():
    from models.common.v3_launch import job_list
    five = ["fold0", "fold1", "fold2", "fold3", "fold4"]
    assert job_list([0, 1, 2, 3, 4], True, False) == five + ["foldsite"]        # v3 default
    assert job_list([0, 1, 2, 3, 4], True, True) == five + [
        "foldsite", "foldloso_ostrava", "foldloso_ropvl"]
    assert job_list([3, 4], False, False) == ["fold3", "fold4"]
    assert job_list([0, 1], False, True) == ["fold0", "fold1", "foldloso_ostrava", "foldloso_ropvl"]
    assert job_list([], True, True) == ["foldsite", "foldloso_ostrava", "foldloso_ropvl"]
    # arm B's LOSO-only kernel
    assert job_list([], False, True) == ["foldloso_ostrava", "foldloso_ropvl"]
    assert job_list([], False, False) == []


def test_fold_file_paths_for_v3_and_v4_pools_and_loso():
    from models.common.v3_launch import fold_file, job_list
    assert fold_file("fold0", PREP, "v3") == f"{PREP}/fold_0.csv"
    assert fold_file("fold4", PREP, "v3") == f"{PREP}/fold_4.csv"
    assert fold_file("foldsite", PREP, "v3") == f"{PREP}/fold_site.csv"
    assert fold_file("fold0", PREP, "v4") == f"{PREP}/v4/fold_0.csv"
    assert fold_file("fold3", PREP, "v4") == f"{PREP}/v4/fold_3.csv"
    assert fold_file("foldsite", PREP, "v4") == f"{PREP}/v4/fold_site.csv"
    # LOSO routes by pool like the others: arm B (v3) trains its LOSO folds on the v3 pool
    assert fold_file("foldloso_ostrava", PREP, "v3") == f"{PREP}/fold_loso_ostrava.csv"
    assert fold_file("foldloso_ropvl", PREP, "v3") == f"{PREP}/fold_loso_ropvl.csv"
    assert fold_file("foldloso_ostrava", PREP, "v4") == f"{PREP}/v4/fold_loso_ostrava.csv"
    assert fold_file("foldloso_ropvl", PREP, "v4") == f"{PREP}/v4/fold_loso_ropvl.csv"
    # every job a launch can ask for maps to a committed file, for both pools
    for pool in ("v3", "v4"):
        for name in job_list([0, 1, 2, 3, 4], True, True):
            assert (ROOT / fold_file(name, PREP, pool)).is_file(), (pool, name)


def test_fold_file_rejects_unknown_pool_and_name():
    from models.common.v3_launch import check_pool, fold_file
    for pool in ("", "v5", "V4", "v4 "):
        with pytest.raises(ValueError, match="pool"):
            fold_file("fold0", PREP, pool)
        with pytest.raises(ValueError, match="pool"):
            fold_file("foldloso_ostrava", PREP, pool)
        with pytest.raises(SystemExit, match="pool"):
            check_pool(pool)
    check_pool("v3")
    check_pool("v4")
    for name in ("inner", "fold", "foldx", "fold_0", "fold-1", "foldsite2", "foldloso_",
                 "foldloso_shenzhen", "fold_loso_ostrava", "dev"):
        with pytest.raises(ValueError, match="job"):
            fold_file(name, PREP, "v3")


def test_kernel_has_exactly_one_pool_and_one_loso_knob():
    """The pusher rewrites each knob with one regex; it must match one module-level line."""
    import re
    src = KERNEL.read_text()
    assert re.findall(r'^POOL = ".*?"$', src, re.M) == ['POOL = "v3"']      # v3 by default
    assert re.findall(r"^LOSO = (?:True|False)$", src, re.M) == ["LOSO = False"]


def test_kernel_wires_pool_and_loso_through_the_helpers():
    import ast
    src = KERNEL.read_text()
    tree = ast.parse(src)
    for call in ("check_pool(POOL)", "check_loso(VARIANT, LOSO)",
                 "job_list(FOLDS, ALSO_SITE_FOLD, LOSO)"):
        assert call in src, call
    reads = {ast.unparse(n.args[0]) for n in ast.walk(tree) if isinstance(n, ast.Call)
             and ast.unparse(n.func) == "pd.read_csv"}
    # dev is the same for every pool; fold files only through fold_file
    assert reads == {"f'{PREP}/test_dev.csv'", "f'{PREP}/exclude_duplicates.csv'",
                     "fold_file('fold0', PREP, POOL)", "fold_file(name, PREP, POOL)"}
    assert any(isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == "PREP"
               and ast.unparse(n.value) == f"'{PREP}'" for n in tree.body)
    summary = next(n.value for n in tree.body if isinstance(n, ast.Assign)
                   and ast.unparse(n.targets[0]) == "summary")
    keys = {ast.unparse(k): ast.unparse(v) for k, v in zip(summary.keys, summary.values)}
    assert keys["'pool'"] == "POOL" and keys["'loso'"] == "LOSO"


def test_kernel_cv_summary_counts_the_five_cv_folds_only():
    """cv_macro_f1_mean is over fold0..fold4: never the site, LOSO or probe jobs."""
    import ast
    from models.common.v3_launch import job_list
    tree = ast.parse(KERNEL.read_text())
    cv = next(n for n in tree.body
              if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == "cv")
    cond = " and ".join(ast.unparse(c) for c in cv.value.generators[0].ifs)
    keep = eval(f"lambda k, v: {cond}")
    names = job_list([0, 1, 2, 3, 4], True, True) + ["inner"]
    assert [k for k in names if keep(k, {"macro_f1": 0.5})] == [f"fold{i}" for i in range(5)]
    assert not keep("fold0", {"macro_f1": None})
