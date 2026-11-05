"""The v3 pusher rewrites module-level knobs in the kernel; each must change exactly one line."""
import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import push_v3_kernels as P  # noqa: E402


def _build(tmp_path, **kw):
    kw.setdefault("arch", "retfound_green")
    kw.setdefault("seed", 42)
    slug = P.build(out=tmp_path, **kw)
    return slug, (tmp_path / "rop_v3_finetune.py").read_text()


def _diff_lines(tmp_path, **kw):
    orig = P.KERNEL.read_text().splitlines()
    _, new = _build(tmp_path, **kw)
    new = new.splitlines()
    assert len(orig) == len(new)
    return [(o, n) for o, n in zip(orig, new) if o != n]


def test_each_knob_changes_exactly_one_line(tmp_path):
    changed = _diff_lines(tmp_path, arch="dinov2_s", seed=7, folds="0,1,2", no_site_fold=True,
                          lr="3e-5", epochs=5)
    assert {n.split(" =")[0] for _, n in changed} == {
        "ARCH", "SEED", "FOLDS", "ALSO_SITE_FOLD", "LR", "EPOCHS", "CONFIG"}
    assert len(changed) == 7


def test_values_land_in_the_assignments(tmp_path):
    _, src = _build(tmp_path, arch="dinov2_s", seed=7, folds="0,1", no_site_fold=True,
                    lr="3e-5", epochs=5)
    for line in ('ARCH = "dinov2_s"', "SEED = 7", "FOLDS = [0,1]", "ALSO_SITE_FOLD = False",
                 'LR = "3e-5"', "EPOCHS = 5", 'CONFIG = "configs/rop_staging_dinov2.yaml"'):
        assert re.search("^" + re.escape(line) + r"\s*(#.*)?$", src, re.M), line


def test_probe_forces_fold0_and_no_site_fold(tmp_path):
    slug, src = _build(tmp_path, probe="inner", folds="0,1,2", lr="1e-4", epochs=1)
    assert re.search(r'^PROBE = "inner"', src, re.M)
    assert re.search(r"^FOLDS = \[0\]", src, re.M)
    assert re.search(r"^ALSO_SITE_FOLD = False", src, re.M)
    assert "probe" in slug and "1e-4" in slug and "retfound-green" in slug and "s42" in slug


def test_slug_suffix(tmp_path):
    slug, _ = _build(tmp_path, suffix="a")
    assert slug.endswith("-a")


def test_metadata_attaches_only_the_three_datasets(tmp_path):
    import json
    _build(tmp_path)
    meta = json.loads((tmp_path / "kernel-metadata.json").read_text())
    assert meta["dataset_sources"] == [
        "praharshnagpure/retinal-framework-code", "praharshnagpure/rop-staging-cache768",
        "praharshnagpure/rop-v3-weights"]
    assert meta["enable_gpu"] and meta["is_private"]


def test_duplicate_assignment_is_refused(tmp_path, monkeypatch):
    fake = tmp_path / "k.py"
    fake.write_text(P.KERNEL.read_text() + '\nARCH = "x"\n')
    monkeypatch.setattr(P, "KERNEL", fake)
    with pytest.raises(SystemExit):
        P.build(arch="retfound_green", seed=1, out=tmp_path / "o")


def test_main_runs_drop_duplicates_from_every_training_manifest():
    import ast
    src = P.KERNEL.read_text()
    reads = [n for n in ast.walk(ast.parse(src)) if isinstance(n, ast.Call)
             and ast.unparse(n.func) == "pd.read_csv" and "fold" in ast.unparse(n.args[0])]
    wrapped = [n for n in ast.walk(ast.parse(src)) if isinstance(n, ast.Call)
               and ast.unparse(n.func) == "drop_duplicate_train_rows"]
    # every main-run job (CV, site and LOSO folds, either pool) is read through fold_file and goes
    # through the filter; fold0 for the inner probe is the one exception
    assert {ast.unparse(w.args[0]) for w in wrapped} == {
        "pd.read_csv(fold_file(name, PREP, POOL))"}
    assert sorted(ast.unparse(r.args[0]) for r in reads) == [
        "fold_file('fold0', PREP, POOL)", "fold_file(name, PREP, POOL)"]


def test_main_runs_take_the_probe_lr(tmp_path):
    import json
    pj = tmp_path / "lr_probe.json"
    with pytest.raises(SystemExit, match="missing"):
        P.protocol_lr("dinov2_s", None, pj)
    pj.write_text(json.dumps({"arms": {"dinov2_s": {"chosen_lr": "2e-4"},
                                       "retfound_green": {"chosen_lr": "1e-4"}}}))
    assert P.protocol_lr("dinov2_s", None, pj) == "2e-4"
    assert P.protocol_lr("dinov2_s", "0.0002", pj) == "2e-4"
    assert P.protocol_lr("retfound_green", None, pj) == "1e-4"
    with pytest.raises(SystemExit, match="contradicts"):
        P.protocol_lr("retfound_green", "2e-4", pj)


def test_variant_bag_sets_the_variant_and_the_bag_config(tmp_path):
    changed = _diff_lines(tmp_path, arch="dinov2_s", variant="bag", lr="1e-4")
    assert {n.split(" =")[0] for _, n in changed} == {"ARCH", "VARIANT", "CONFIG"}
    _, src = _build(tmp_path, arch="dinov2_s", variant="bag", lr="1e-4")
    for line in ('VARIANT = "bag"', 'CONFIG = "configs/rop_staging_dinov2_bag.yaml"'):
        assert re.search("^" + re.escape(line) + r"\s*(#.*)?$", src, re.M), line


def test_variant_slugs(tmp_path):
    assert _build(tmp_path, arch="dinov2_s", variant="bag", lr="1e-4")[0] == \
        "rop-v3-dinov2-s-bag-s42-lr1e-4"
    assert _build(tmp_path, arch="dinov2_s", seed=1337, variant="bag", lr="1e-4")[0] == \
        "rop-v3-dinov2-s-bag-s1337-lr1e-4"
    assert _build(tmp_path, arch="dinov2_s", variant="bag", lr="1e-4", probe="inner", epochs=2,
                  suffix="smoke")[0] == "rop-v3-dinov2-s-bag-s42-probe-lr1e-4-smoke"
    assert _build(tmp_path, arch="dinov2_s", lr="1e-4")[0] == "rop-v3-dinov2-s-s42-lr1e-4"


def test_variant_is_refused_where_no_config_exists(tmp_path):
    with pytest.raises(SystemExit, match="variant"):
        _build(tmp_path, arch="retfound_green", variant="bag")
    with pytest.raises(SystemExit, match="variant"):
        _build(tmp_path, arch="dinov2_s", variant="mil")


def test_variant_runs_take_the_protocol_lr_even_as_a_probe(capsys):
    assert P.main(["--arch", "dinov2_s", "--variant", "bag", "--probe", "inner", "--epochs", "2",
                   "--slug-suffix", "smoke", "--dry-run"]) == 0
    assert "[dry-run] rop-v3-dinov2-s-bag-s42-probe-lr1e-4-smoke" in capsys.readouterr().out
    with pytest.raises(SystemExit, match="contradicts"):
        P.main(["--arch", "dinov2_s", "--variant", "bag", "--lr", "2e-4", "--dry-run"])


def test_kernel_checks_its_variant_against_the_config():
    from models.common.config import load_config
    from models.common.v3_launch import check_variant
    root = P.ROOT
    d = load_config(str(root / "configs/rop_staging_dinov2_bag.yaml"))
    b = load_config(str(root / "configs/rop_staging_dinov2.yaml"))
    check_variant("bag", d)
    check_variant("", b)
    with pytest.raises(SystemExit, match="bag_loss"):
        check_variant("bag", b)
    with pytest.raises(SystemExit, match="bag_loss"):
        check_variant("", d)
    with pytest.raises(SystemExit, match="variant"):
        check_variant("mil", d)


# ── arm G (PROTOCOL_V2 Amendment 4a): B at 518 px, five CV folds only, split over two kernels ──
def test_variant_r518_sets_the_variant_and_the_518_config(tmp_path):
    changed = _diff_lines(tmp_path, arch="dinov2_s", variant="r518", lr="1e-4", folds="0,1,2",
                          no_site_fold=True)
    assert {n.split(" =")[0] for _, n in changed} == {
        "ARCH", "VARIANT", "CONFIG", "FOLDS", "ALSO_SITE_FOLD"}
    _, src = _build(tmp_path, arch="dinov2_s", variant="r518", lr="1e-4", folds="3,4",
                    no_site_fold=True)
    for line in ('VARIANT = "r518"', 'CONFIG = "configs/rop_staging_dinov2_518.yaml"',
                 "FOLDS = [3,4]", "ALSO_SITE_FOLD = False"):
        assert re.search("^" + re.escape(line) + r"\s*(#.*)?$", src, re.M), line


def test_r518_slugs_split_the_five_folds_over_an_a_and_a_b_kernel(tmp_path):
    kw = dict(arch="dinov2_s", variant="r518", lr="1e-4", no_site_fold=True)
    assert _build(tmp_path, folds="0,1,2", suffix="a", **kw)[0] == "rop-v3-dinov2-s-r518-s42-lr1e-4-a"
    assert _build(tmp_path, seed=1337, folds="3,4", suffix="b", **kw)[0] == \
        "rop-v3-dinov2-s-r518-s1337-lr1e-4-b"
    assert _build(tmp_path, arch="dinov2_s", variant="r518", lr="1e-4", probe="inner", epochs=2,
                  suffix="smoke")[0] == "rop-v3-dinov2-s-r518-s42-probe-lr1e-4-smoke"


def test_five_fold_only_variant_refuses_a_site_fold(tmp_path, capsys):
    with pytest.raises(SystemExit, match="site fold"):
        _build(tmp_path, arch="dinov2_s", variant="r518", lr="1e-4", folds="0,1,2")
    with pytest.raises(SystemExit, match="site fold"):
        P.main(["--arch", "dinov2_s", "--variant", "r518", "--folds", "0,1,2", "--dry-run"])
    assert P.main(["--arch", "dinov2_s", "--variant", "r518", "--seed", "1337", "--folds", "3,4",
                   "--no-site-fold", "--slug-suffix", "b", "--dry-run"]) == 0
    assert "[dry-run] rop-v3-dinov2-s-r518-s1337-lr1e-4-b" in capsys.readouterr().out
    _build(tmp_path, arch="dinov2_s", variant="bag", lr="1e-4")       # D keeps its site fold


def test_kernel_checks_the_r518_variant_and_its_site_fold():
    from models.common.config import load_config
    from models.common.v3_launch import check_site_fold, check_variant
    g = load_config(str(P.ROOT / "configs/rop_staging_dinov2_518.yaml"))
    b = load_config(str(P.ROOT / "configs/rop_staging_dinov2.yaml"))
    check_variant("r518", g)
    with pytest.raises(SystemExit, match="image_size"):
        check_variant("", g)
    with pytest.raises(SystemExit, match="image_size"):
        check_variant("r518", b)
    check_site_fold("r518", False)
    check_site_fold("", True)
    check_site_fold("bag", True)
    with pytest.raises(SystemExit, match="site fold"):
        check_site_fold("r518", True)
    src = P.KERNEL.read_text()
    assert "check_site_fold(VARIANT, ALSO_SITE_FOLD)" in src and "check_variant(VARIANT, cfg0)" in src


# ── arm F (PROTOCOL_V2 Amendment 4b): self-ensembling on unlabelled dev photos, five CV folds ──
def test_variant_uda_sets_the_variant_and_the_uda_config(tmp_path):
    changed = _diff_lines(tmp_path, arch="dinov2_s", variant="uda", lr="1e-4", folds="3,4",
                          no_site_fold=True)
    assert {n.split(" =")[0] for _, n in changed} == {
        "ARCH", "VARIANT", "CONFIG", "FOLDS", "ALSO_SITE_FOLD"}
    _, src = _build(tmp_path, arch="dinov2_s", variant="uda", lr="1e-4", folds="0,1,2",
                    no_site_fold=True)
    for line in ('VARIANT = "uda"', 'CONFIG = "configs/rop_staging_dinov2_uda.yaml"'):
        assert re.search("^" + re.escape(line) + r"\s*(#.*)?$", src, re.M), line


def test_uda_slugs_and_its_site_fold_refusal(tmp_path):
    kw = dict(arch="dinov2_s", variant="uda", lr="1e-4", no_site_fold=True)
    assert _build(tmp_path, folds="0,1,2", suffix="a", **kw)[0] == "rop-v3-dinov2-s-uda-s42-lr1e-4-a"
    assert _build(tmp_path, seed=1337, folds="3,4", suffix="b", **kw)[0] == \
        "rop-v3-dinov2-s-uda-s1337-lr1e-4-b"
    assert _build(tmp_path, arch="dinov2_s", variant="uda", lr="1e-4", probe="inner", epochs=2,
                  suffix="smoke")[0] == "rop-v3-dinov2-s-uda-s42-probe-lr1e-4-smoke"
    with pytest.raises(SystemExit, match="site fold"):
        _build(tmp_path, arch="dinov2_s", variant="uda", lr="1e-4", folds="0,1,2")


def test_pusher_and_kernel_agree_on_the_five_fold_only_variants():
    from models.common.v3_launch import FIVE_FOLD_ONLY, VARIANTS
    assert set(P.FIVE_FOLD_ONLY) == set(FIVE_FOLD_ONLY) == {"r518", "uda"}
    assert {v for _, v in P.VARIANT_CONFIGS} | {""} == set(VARIANTS)


def test_kernel_checks_the_uda_variant():
    from models.common.config import load_config
    from models.common.v3_launch import check_site_fold, check_variant
    f = load_config(str(P.ROOT / "configs/rop_staging_dinov2_uda.yaml"))
    b = load_config(str(P.ROOT / "configs/rop_staging_dinov2.yaml"))
    check_variant("uda", f)
    with pytest.raises(SystemExit, match="uda"):
        check_variant("", f)
    with pytest.raises(SystemExit, match="uda"):
        check_variant("uda", b)
    with pytest.raises(SystemExit, match="site fold"):
        check_site_fold("uda", True)


# ── arms H and I (PROTOCOL_V2 Amendment 5): the v4 pool and the leave-one-source-out folds ──
V4 = dict(arch="dinov2_s", pool="v4", lr="1e-4")


def _knob_names(changed):
    return {n.split(" =")[0] for _, n in changed}


def test_pool_and_loso_knobs_change_one_line_each(tmp_path):
    changed = _diff_lines(tmp_path, variant="v4data", loso=True, folds="", **V4)
    assert _knob_names(changed) == {"ARCH", "VARIANT", "CONFIG", "FOLDS", "POOL", "LOSO"}
    assert len(changed) == 6
    _, src = _build(tmp_path, variant="v4recipe", loso=True, folds="", **V4)
    for line in ('VARIANT = "v4recipe"', 'CONFIG = "configs/rop_staging_dinov2_v4recipe.yaml"',
                 'POOL = "v4"', "LOSO = True", "FOLDS = []", "ALSO_SITE_FOLD = True"):
        assert re.search("^" + re.escape(line) + r"\s*(#.*)?$", src, re.M), line
    _, src = _build(tmp_path, variant="v4data", folds="0,1,2,3,4", no_site_fold=True, **V4)
    assert re.search(r'^CONFIG = "configs/rop_staging_dinov2_v4data.yaml"', src, re.M)
    assert re.search(r"^LOSO = False", src, re.M) and re.search(r"^FOLDS = \[0,1,2,3,4\]", src, re.M)
    # arm B's LOSO-only kernel: the v3 pool (POOL untouched), no CV fold, no site fold
    changed = _diff_lines(tmp_path, arch="dinov2_s", lr="1e-4", loso=True, folds="",
                          no_site_fold=True)
    assert _knob_names(changed) == {"ARCH", "CONFIG", "FOLDS", "ALSO_SITE_FOLD", "LOSO"}
    # the defaults leave both knobs alone
    assert _knob_names(_diff_lines(tmp_path, arch="dinov2_s", lr="1e-4")) == {"ARCH", "CONFIG"}


def test_probe_forces_loso_off(tmp_path):
    _, src = _build(tmp_path, variant="v4recipe", loso=True, probe="inner", epochs=2, **V4)
    assert re.search(r"^LOSO = False", src, re.M) and re.search(r"^FOLDS = \[0\]", src, re.M)
    assert re.search(r'^POOL = "v4"', src, re.M)


def test_v4_slug(tmp_path):
    assert _build(tmp_path, variant="v4data", folds="0,1,2,3,4", no_site_fold=True, suffix="a",
                  **V4)[0] == "rop-v4-dinov2-s-v4data-s42-lr1e-4-a"
    assert _build(tmp_path, variant="v4recipe", seed=1337, folds="", loso=True, suffix="b",
                  **V4)[0] == "rop-v4-dinov2-s-v4recipe-s1337-lr1e-4-b"
    assert _build(tmp_path, variant="v4recipe", probe="inner", epochs=2, suffix="smoke",
                  **V4)[0] == "rop-v4-dinov2-s-v4recipe-s42-probe-lr1e-4-smoke"
    assert _build(tmp_path, arch="dinov2_s", lr="1e-4", folds="", no_site_fold=True, loso=True,
                  suffix="loso")[0] == "rop-v3-dinov2-s-s42-lr1e-4-loso"       # B stays v3


def test_cli_passes_pool_loso_and_an_empty_fold_list(monkeypatch, capsys):
    seen = {}
    real = P.build

    def spy(arch, seed, out, **kw):
        slug = real(arch, seed, out, **kw)
        seen["src"] = (out / P.KERNEL.name).read_text()
        return slug

    monkeypatch.setattr(P, "build", spy)
    assert P.main(["--arch", "dinov2_s", "--variant", "v4recipe", "--pool", "v4", "--folds", "",
                   "--loso", "--slug-suffix", "b", "--dry-run"]) == 0
    assert "[dry-run] rop-v4-dinov2-s-v4recipe-s42-lr1e-4-b" in capsys.readouterr().out
    for line in ("FOLDS = []", 'POOL = "v4"', "LOSO = True", "ALSO_SITE_FOLD = True"):
        assert re.search("^" + re.escape(line) + r"\s*(#.*)?$", seen["src"], re.M), line
    assert P.main(["--arch", "dinov2_s", "--folds", "", "--no-site-fold", "--loso",
                   "--slug-suffix", "loso", "--dry-run"]) == 0
    assert "[dry-run] rop-v3-dinov2-s-s42-lr1e-4-loso" in capsys.readouterr().out


def test_pusher_refuses_pool_variant_mismatch(tmp_path):
    for kw in (dict(variant="v4data"), dict(variant="v4recipe", pool="v3"), dict(pool="v4"),
               dict(variant="bag", pool="v4"), dict(variant="v4data", pool="v5")):
        with pytest.raises(SystemExit, match="pool"):
            _build(tmp_path, **{**dict(arch="dinov2_s", lr="1e-4"), **kw})
    with pytest.raises(SystemExit, match="pool"):
        P.main(["--arch", "dinov2_s", "--pool", "v4", "--dry-run"])
    with pytest.raises(SystemExit, match="pool"):
        P.main(["--arch", "dinov2_s", "--variant", "v4data", "--dry-run"])


def test_pusher_refuses_loso_outside_B_H_and_I(tmp_path):
    with pytest.raises(SystemExit, match="LOSO"):
        _build(tmp_path, arch="dinov2_s", variant="bag", lr="1e-4", loso=True)
    with pytest.raises(SystemExit, match="LOSO"):
        _build(tmp_path, arch="dinov2_s", variant="uda", lr="1e-4", loso=True, no_site_fold=True)
    with pytest.raises(SystemExit, match="LOSO"):                     # arm A: no LOSO folds
        _build(tmp_path, arch="retfound_green", loso=True, folds="", no_site_fold=True)
    # arm B's CV and site folds are collected already: its LOSO kernel trains the LOSO folds only
    with pytest.raises(SystemExit, match="LOSO-only"):
        _build(tmp_path, arch="dinov2_s", lr="1e-4", loso=True, folds="")
    with pytest.raises(SystemExit, match="LOSO-only"):
        _build(tmp_path, arch="dinov2_s", lr="1e-4", loso=True, no_site_fold=True)
