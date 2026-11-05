import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from models.common.config import load_config  # noqa: E402
from scripts.collect_v3_run import FOLDS, collect, v4_config_drift  # noqa: E402
from scripts.push_v3_kernels import VARIANT_CONFIGS  # noqa: E402

SIX = [n for n in FOLDS if not n.startswith("foldloso_")]    # fold0..fold4, foldsite (v3 arms)
LOSO = ["foldloso_ostrava", "foldloso_ropvl"]


def _as_run_v4(variant, seed, manifest, edits=()):
    """Arm H's or I's config.yaml as the kernel writes it: the registered YAML loaded with the
    kernel's per-run --set list (kaggle_kernels/rop_v3_finetune.py), plus ``edits``."""
    sets = [f"data.manifest={manifest}", "train.epochs=20", f"seed={seed}", "train.lr=1e-4",
            "train.num_workers=2", "preprocess.cached=true",
            "preprocess.cache_root=/kaggle/working/rop_cache"]
    return load_config(ROOT / VARIANT_CONFIGS[("dinov2_s", variant)],
                       overrides=sets + list(edits)).to_dict()


def _kernel_output(tmp_path, seed=1337, bag=None, folds=None, preprocess=None, uda=None,
                   arm_tag=None, arch="dinov2_s", trained_pool=None, cfg_edits=(), **summary_over):
    """folds: the fold names this kernel completed (default: the six v3 ones, no LOSO fold);
    preprocess / uda: extra config.yaml content (arm G's image_size, arm F's train.uda block).
    arm_tag "v4data" / "v4recipe": config.yaml is that arm's registered YAML as the kernel runs it
    (_as_run_v4), with cfg_edits ("key=value" overrides) on top. prep/v4 holds the v4 pool's fold
    files: the same val rows as prep's, a different train row. The kernel trains on the pool its
    summary names (summary_over "pool", default v3) unless trained_pool says otherwise."""
    prep = tmp_path / "prep"
    (prep / "v4").mkdir(parents=True)
    trained_pool = trained_pool or summary_over.get("pool", "v3")
    dev = pd.DataFrame({"image_path": [f"data/rop/mv/d{i}.jpg" for i in range(4)], "label": [0, 1, 2, 3]})
    dev.to_csv(prep / "test_dev.csv", index=False)
    pd.DataFrame({"image_path": ["dup_a", "dup_b"], "md5": ["m", "m"], "group": [0, 0]}).to_csv(
        prep / "exclude_duplicates.csv", index=False)
    (tmp_path / "lr_probe.json").write_text(json.dumps(
        {"arms": {"dinov2_s": {"chosen_lr": "1e-4"}, "retfound_green": {"chosen_lr": "1e-4"}}}))
    src = tmp_path / "out"
    summary = {"arch": arch, "seed": seed, "epochs": 20, "lr": "1e-4", "probe": "",
               "folds": {}}
    for i, (name, fold) in enumerate(FOLDS.items()):
        man = pd.DataFrame({"image_path": [f"p{i}_{j}" for j in range(5)],
                            "label": [0, 1, 2, i % 6, 5], "split": ["train", "val", "val", "val", "val"]})
        man.to_csv(prep / f"{fold}.csv", index=False)
        man4 = man.assign(image_path=[f"q{i}_0"] + man.image_path.tolist()[1:])
        man4.to_csv(prep / "v4" / f"{fold}.csv", index=False)
        if name not in (SIX if folds is None else folds):
            continue
        man = man4 if trained_pool == "v4" else man
        used = src / "code" / "results" / "prepared" / "rop_staging" / f"_v3_{name}.csv"
        used.parent.mkdir(parents=True, exist_ok=True)
        man.to_csv(used, index=False)
        exp = src / "code" / "results" / f"experiment_00{i}_v3_{arch}_{name}_s{seed}"
        exp.mkdir(parents=True)
        yv = man[man.split == "val"].label.values
        np.savez(exp / "val_predictions.npz", y_true=yv, y_pred=yv, prob=np.eye(6)[yv])
        np.savez(exp / "dev_predictions.npz", prob=np.eye(6)[dev.label.values],
                 y_true=dev.label.values, image_path=np.asarray(dev.image_path.tolist()))
        (exp / "metrics.json").write_text("{}")
        manifest = f"results/prepared/rop_staging/_v3_{name}.csv"
        if arm_tag is not None:
            cfg = _as_run_v4(arm_tag, seed, manifest, cfg_edits)
        else:
            train = {"lr": 0.0001} | ({} if bag is None else {"bag_loss": {"enabled": bag, "k": 8}})
            train |= {} if uda is None else {"uda": {"enabled": uda, "method": "self_ensembling"}}
            cfg = ({"seed": seed, "model": {"arch": arch}, "train": train, "data": {"manifest": manifest}}
                   | ({} if preprocess is None else {"preprocess": preprocess}))
        (exp / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
        (exp / "weights.pth").write_bytes(b"w")
        summary["folds"][name] = {"macro_f1": 0.7, "qwk": 0.8, "dir": str(exp.relative_to(src / "code"))}
    summary.update(summary_over)
    (src / "summary.json").write_text(json.dumps(summary))
    return src, prep


def test_collect_copies_the_four_files_per_fold_and_no_weights(tmp_path):
    src, prep = _kernel_output(tmp_path)
    out = collect([src], tmp_path / "runs", prep, tmp_path / "lr_probe.json")
    assert out.name == "dinov2_s_s1337"
    assert sorted(p.name for p in out.iterdir()) == sorted([FOLDS[n] for n in SIX] + ["summary.json"])
    for fold in (FOLDS[n] for n in SIX):
        assert sorted(p.name for p in (out / fold).iterdir()) == sorted(
            ["val_predictions.npz", "dev_predictions.npz", "metrics.json", "config.yaml"])
    assert not list(out.rglob("weights.pth"))
    with pytest.raises(FileExistsError):
        collect([src], tmp_path / "runs", prep, tmp_path / "lr_probe.json")


def test_collect_refuses_probe_and_missing_folds(tmp_path):
    src, prep = _kernel_output(tmp_path, probe="inner")
    with pytest.raises(ValueError, match="not a main run"):
        collect([src], tmp_path / "runs", prep, tmp_path / "lr_probe.json")
    s = json.loads((src / "summary.json").read_text())
    s["probe"] = ""
    s["folds"]["fold3"] = {"status": "failed"}
    (src / "summary.json").write_text(json.dumps(s))
    with pytest.raises(ValueError, match="fold3"):
        collect([src], tmp_path / "runs", prep, tmp_path / "lr_probe.json")
    assert not (tmp_path / "runs").exists()


def test_collect_refuses_misaligned_val_and_dev(tmp_path):
    src, prep = _kernel_output(tmp_path)
    exp = src / "code" / json.loads((src / "summary.json").read_text())["folds"]["fold2"]["dir"]
    np.savez(exp / "val_predictions.npz", y_true=np.array([5, 4, 3, 2]), y_pred=np.zeros(4), prob=np.zeros((4, 6)))
    with pytest.raises(ValueError, match="val labels"):
        collect([src], tmp_path / "runs", prep, tmp_path / "lr_probe.json")

    src, prep = _kernel_output(tmp_path / "b")
    exp = src / "code" / json.loads((src / "summary.json").read_text())["folds"]["foldsite"]["dir"]
    z = dict(np.load(exp / "dev_predictions.npz"))
    z["image_path"] = z["image_path"][::-1]
    np.savez(exp / "dev_predictions.npz", **z)
    with pytest.raises(ValueError, match="test_dev.csv order"):
        collect([src], tmp_path / "b" / "runs", prep, tmp_path / "b" / "lr_probe.json")


def test_collect_refuses_config_disagreeing_with_summary(tmp_path):
    src, prep = _kernel_output(tmp_path, lr="2e-4")
    with pytest.raises(ValueError, match="disagrees"):
        collect([src], tmp_path / "runs", prep, tmp_path / "lr_probe.json")


def test_collect_merges_a_relaunch_and_refuses_a_fold_trained_twice(tmp_path):
    src, prep = _kernel_output(tmp_path)
    s = json.loads((src / "summary.json").read_text())
    first = {**s, "folds": {k: (v if k in ("fold0", "fold1", "fold2") else {"status": "failed"})
                            for k, v in s["folds"].items()}}
    second = {**s, "folds": {k: v for k, v in s["folds"].items() if k not in ("fold0", "fold1", "fold2")}}
    b = tmp_path / "out_b"
    b.mkdir()
    (b / "code").symlink_to(src / "code")
    (src / "summary.json").write_text(json.dumps(first))
    (b / "summary.json").write_text(json.dumps(second))
    out = collect([src, b], tmp_path / "runs", prep, tmp_path / "lr_probe.json")
    m = json.loads((out / "summary.json").read_text())
    assert sorted(m["folds"]) == sorted(SIX)
    assert m["folds"]["fold0"]["kernel"] == "out" and m["folds"]["foldsite"]["kernel"] == "out_b"

    (b / "summary.json").write_text(json.dumps(s))          # fold0 now completed in both
    with pytest.raises(ValueError, match="two kernels"):
        collect([src, b], tmp_path / "runs2", prep, tmp_path / "lr_probe.json")
    other, _ = _kernel_output(tmp_path / "s42", seed=42)
    with pytest.raises(ValueError, match="different arm-seed-lr"):
        collect([src, other], tmp_path / "runs3", prep, tmp_path / "lr_probe.json")


def test_collect_refuses_wrong_lr_and_undeduplicated_training(tmp_path):
    src, prep = _kernel_output(tmp_path)
    (tmp_path / "lr_probe.json").write_text(json.dumps({"arms": {"dinov2_s": {"chosen_lr": "2e-4"}}}))
    with pytest.raises(ValueError, match="probe's choice"):
        collect([src], tmp_path / "runs", prep, tmp_path / "lr_probe.json")

    src, prep = _kernel_output(tmp_path / "d")
    used = src / "code" / "results" / "prepared" / "rop_staging" / "_v3_fold1.csv"
    m = pd.read_csv(used)
    m.loc[0, "image_path"] = "dup_b"                       # a train row
    m.to_csv(used, index=False)
    with pytest.raises(ValueError, match="excluded duplicate"):
        collect([src], tmp_path / "d" / "runs", prep, tmp_path / "d" / "lr_probe.json")
    assert not (tmp_path / "d" / "runs").exists()


def test_collect_names_the_bag_variant_run_and_checks_its_config(tmp_path):
    src, prep = _kernel_output(tmp_path, bag=True, variant="bag")
    out = collect([src], tmp_path / "runs", prep, tmp_path / "lr_probe.json")
    assert out.name == "dinov2_s_bag_s1337"
    assert json.loads((out / "summary.json").read_text())["variant"] == "bag"


@pytest.mark.parametrize("bag,variant", [(None, "bag"), (False, "bag"), (True, ""), (True, None)])
def test_collect_refuses_a_variant_its_config_contradicts(tmp_path, bag, variant):
    over = {} if variant is None else {"variant": variant}    # None: a summary with no variant key
    src, prep = _kernel_output(tmp_path, bag=bag, **over)
    with pytest.raises(ValueError, match="bag_loss"):
        collect([src], tmp_path / "runs", prep, tmp_path / "lr_probe.json")
    assert not (tmp_path / "runs").exists()


def test_collect_refuses_to_merge_arm_B_with_arm_D(tmp_path):
    b, prep = _kernel_output(tmp_path / "b")
    d, _ = _kernel_output(tmp_path / "d", bag=True, variant="bag")
    s = json.loads((d / "summary.json").read_text())
    s["folds"] = {k: v for k, v in s["folds"].items() if k == "foldsite"}
    (d / "summary.json").write_text(json.dumps(s))
    s = json.loads((b / "summary.json").read_text())
    s["folds"]["foldsite"] = {"status": "failed"}
    (b / "summary.json").write_text(json.dumps(s))
    with pytest.raises(ValueError, match="different arm"):
        collect([b, d], tmp_path / "runs", prep, tmp_path / "b" / "lr_probe.json")


def test_existing_A_and_B_kernels_without_a_variant_key_still_collect(tmp_path):
    src, prep = _kernel_output(tmp_path)                    # summary has no "variant" key
    out = collect([src], tmp_path / "runs", prep, tmp_path / "lr_probe.json")
    assert out.name == "dinov2_s_s1337"


# ── arm G (PROTOCOL_V2 Amendment 4a): five CV folds, no site fold, over an a and a b kernel ──
CV5 = [f"fold{k}" for k in range(5)]


def _g_pair(tmp_path, seed=1337):
    """Arm G's two kernels for one seed: a = folds 0, 1, 2 and b = folds 3, 4 (no site fold)."""
    kw = dict(seed=seed, preprocess={"image_size": 518}, variant="r518")
    a, prep = _kernel_output(tmp_path / "a", folds=CV5[:3], **kw)
    b, _ = _kernel_output(tmp_path / "b", folds=CV5[3:], **kw)
    return a, b, prep, tmp_path / "a" / "lr_probe.json"


def test_collect_merges_the_r518_a_and_b_kernels_without_a_site_fold(tmp_path):
    a, b, prep, lp = _g_pair(tmp_path)
    out = collect([a, b], tmp_path / "runs", prep, lp)
    assert out.name == "dinov2_s_r518_s1337"
    assert sorted(p.name for p in out.iterdir()) == sorted([f"fold_{k}" for k in range(5)] + ["summary.json"])
    m = json.loads((out / "summary.json").read_text())
    assert m["variant"] == "r518" and sorted(m["folds"]) == CV5
    assert {m["folds"][f]["kernel"] for f in CV5[:3]} == {"out"} and m["folds"]["fold3"]["kernel"] == "out"
    with pytest.raises(ValueError, match="fold3"):                    # the b kernel is needed
        collect([a], tmp_path / "runs2", prep, lp)


def test_collect_refuses_a_site_fold_in_a_five_fold_variant(tmp_path):
    src, prep = _kernel_output(tmp_path, preprocess={"image_size": 518}, variant="r518")
    with pytest.raises(ValueError, match="site fold"):
        collect([src], tmp_path / "runs", prep, tmp_path / "lr_probe.json")
    assert not (tmp_path / "runs").exists()


@pytest.mark.parametrize("size,variant", [(392, "r518"), (None, "r518"), (518, ""), (518, "bag")])
def test_collect_refuses_an_image_size_the_variant_contradicts(tmp_path, size, variant):
    pre = None if size is None else {"image_size": size}
    src, prep = _kernel_output(tmp_path, preprocess=pre, variant=variant,
                               bag=True if variant == "bag" else None,
                               folds=CV5 if variant == "r518" else None)
    with pytest.raises(ValueError, match="image_size"):
        collect([src], tmp_path / "runs", prep, tmp_path / "lr_probe.json")
    assert not (tmp_path / "runs").exists()


def test_arm_B_still_needs_its_site_fold(tmp_path):
    src, prep = _kernel_output(tmp_path, folds=CV5, preprocess={"image_size": 392})
    with pytest.raises(ValueError, match="foldsite"):
        collect([src], tmp_path / "runs", prep, tmp_path / "lr_probe.json")


# ── arm F (PROTOCOL_V2 Amendment 4b): train.uda, five CV folds, a and b kernels ──
def test_collect_names_the_uda_run_and_checks_its_config(tmp_path):
    kw = dict(uda=True, variant="uda", preprocess={"image_size": 392})
    a, prep = _kernel_output(tmp_path / "a", folds=CV5[:3], **kw)
    b, _ = _kernel_output(tmp_path / "b", folds=CV5[3:], **kw)
    out = collect([a, b], tmp_path / "runs", prep, tmp_path / "a" / "lr_probe.json")
    assert out.name == "dinov2_s_uda_s1337" and not (out / "fold_site").exists()
    assert json.loads((out / "summary.json").read_text())["variant"] == "uda"


@pytest.mark.parametrize("uda,variant", [(None, "uda"), (False, "uda"), (True, "")])
def test_collect_refuses_a_uda_block_the_variant_contradicts(tmp_path, uda, variant):
    src, prep = _kernel_output(tmp_path, uda=uda, variant=variant, folds=CV5)
    with pytest.raises(ValueError, match="uda"):
        collect([src], tmp_path / "runs", prep, tmp_path / "lr_probe.json")
    assert not (tmp_path / "runs").exists()


# ── arms H and I (PROTOCOL_V2 Amendment 5): the v4 pool, five CV folds + site + both LOSO folds ──
def _hi_pair(tmp_path, variant="v4data", seed=1337, b_edits=()):
    """An H (or I) arm-seed as launched: a = the five CV folds; b = the site and LOSO folds
    (b_edits: config overrides in the b kernel's folds only)."""
    kw = dict(seed=seed, arm_tag=variant, variant=variant, pool="v4")
    a, prep = _kernel_output(tmp_path / "a", folds=CV5, loso=False, **kw)
    b, _ = _kernel_output(tmp_path / "b", folds=["foldsite"] + LOSO, loso=True, cfg_edits=b_edits, **kw)
    return a, b, prep, tmp_path / "a" / "lr_probe.json"


@pytest.mark.parametrize("variant", ["v4data", "v4recipe"])
def test_collect_v4_run_with_loso_folds(tmp_path, variant):
    a, b, prep, lp = _hi_pair(tmp_path, variant)
    out = collect([a, b], tmp_path / "runs", prep, lp)
    assert out.name == f"dinov2_s_{variant}_s1337"
    assert sorted(p.name for p in out.iterdir()) == sorted(list(FOLDS.values()) + ["summary.json"])
    assert {p.name for p in (out / "fold_loso_ropvl").iterdir()} == {
        "val_predictions.npz", "dev_predictions.npz", "metrics.json", "config.yaml"}
    m = json.loads((out / "summary.json").read_text())
    assert m["variant"] == variant and m["pool"] == "v4" and sorted(m["folds"]) == sorted(FOLDS)
    # the LOSO folds are required for H and I
    b_site, _ = _kernel_output(tmp_path / "b_site", folds=["foldsite"], seed=1337, arm_tag=variant,
                               variant=variant, pool="v4")
    with pytest.raises(ValueError, match="foldloso_ostrava"):
        collect([a, b_site], tmp_path / "runs2", prep, lp)
    assert not (tmp_path / "runs2").exists()


@pytest.mark.parametrize("edit,key", [
    ("train.ema_decay=0.99", "train.ema_decay"),                  # a recipe value altered
    ("train.aug_v4.fourier_p=0.3", "train.aug_v4.fourier_p"),     # ... inside the aug block
    ("train.label_smoothing=0.05", "train.label_smoothing"),
    ("model.drop_path=0.2", "model.drop_path"),
    ("preprocess.clahe=false", "preprocess.clahe"),
    ("train.mixup=true", "train.mixup"),                          # a key the YAML does not have
])
def test_collect_refuses_a_v4recipe_config_that_left_its_yaml(tmp_path, edit, key):
    """Arm I's as-run config.yaml must be configs/rop_staging_dinov2_v4recipe.yaml in train,
    preprocess and model, the kernel's per-run keys aside; one altered fold refuses the run."""
    a, b, prep, lp = _hi_pair(tmp_path, "v4recipe", b_edits=[edit])
    with pytest.raises(ValueError, match=rf"rop_staging_dinov2_v4recipe\.yaml at {re.escape(key)}$"):
        collect([a, b], tmp_path / "runs", prep, lp)
    assert not (tmp_path / "runs").exists()
    a, b, prep, lp = _hi_pair(tmp_path / "unaltered", "v4recipe")         # the same, unaltered
    assert collect([a, b], tmp_path / "runs", prep, lp).name == "dinov2_s_v4recipe_s1337"


def test_v4_config_check_ignores_only_the_per_run_keys():
    yml = yaml.safe_load((ROOT / VARIANT_CONFIGS[("dinov2_s", "v4recipe")]).read_text())
    as_run = _as_run_v4("v4recipe", 42, "results/prepared/rop_staging/_v3_fold0.csv",
                        ["model.weights_path=/kaggle/input/w.pth", "train.lr=2e-4"])
    for k in ("train.num_workers", "preprocess.cached", "preprocess.cache_root", "model.weights_path"):
        sec, key = k.split(".")
        assert as_run[sec][key] != yml[sec][key], k                # the fixture really differs there
    assert v4_config_drift(as_run, yml) == []                       # unaltered: collects
    as_run["train"]["ema_decay"] = 0.99
    as_run["data"]["presplit"] = False                              # data.* is not compared
    assert v4_config_drift(as_run, yml) == ["train.ema_decay"]
    del as_run["train"]["head_warmup_epochs"]                       # a key the run lacks
    assert v4_config_drift(as_run, yml) == ["train.ema_decay", "train.head_warmup_epochs"]


@pytest.mark.parametrize("how", ["train_rows", "val_labels"])
def test_collect_refuses_v4_run_validated_against_v3_folds(tmp_path, how):
    """A kernel claiming pool v4 must line up with prep/v4's fold files, not prep's."""
    if how == "train_rows":           # trained on the v3 pool's rows (the val sides are the same)
        a, prep = _kernel_output(tmp_path, folds=CV5, arm_tag="v4data", variant="v4data",
                                 pool="v4", trained_pool="v3")
        match = "training rows"
    else:                             # val labels align with prep/fold_2.csv only
        a, prep = _kernel_output(tmp_path, folds=CV5, arm_tag="v4data", variant="v4data", pool="v4")
        f = prep / "v4" / "fold_2.csv"
        m = pd.read_csv(f)
        m.loc[m.split == "val", "label"] = m.loc[m.split == "val", "label"].values[::-1]
        m.to_csv(f, index=False)
        match = "val labels"
    with pytest.raises(ValueError, match=match):
        collect([a], tmp_path / "runs", prep, tmp_path / "lr_probe.json")
    assert not (tmp_path / "runs").exists()


@pytest.mark.parametrize("variant,pool,arm_tag,bag", [
    ("v4data", "v3", "v4data", None),       # arm H on the v3 pool
    ("v4recipe", None, "v4recipe", None),   # arm I with no pool key (an old kernel's default v3)
    ("", "v4", None, None),                 # arm B on the v4 pool
    ("bag", "v4", None, True),              # arm D on the v4 pool
])
def test_collect_refuses_a_pool_its_variant_does_not_train_on(tmp_path, variant, pool, arm_tag, bag):
    over = {} if pool is None else {"pool": pool}
    src, prep = _kernel_output(tmp_path, folds=CV5, arm_tag=arm_tag, bag=bag, variant=variant,
                               trained_pool=pool or "v3", **over)
    with pytest.raises(ValueError, match="pool"):
        collect([src], tmp_path / "runs", prep, tmp_path / "lr_probe.json")
    assert not (tmp_path / "runs").exists()


@pytest.mark.parametrize("arch,variant,extra", [
    ("dinov2_s", "bag", dict(bag=True)),                                       # arm D
    ("dinov2_s", "r518", dict(preprocess={"image_size": 518})),                # arm G
    ("dinov2_s", "uda", dict(uda=True)),                                       # arm F
    ("retfound_green", "", {}),                                                # arm A
])
def test_collect_refuses_loso_folds_outside_B_H_and_I(tmp_path, arch, variant, extra):
    src, prep = _kernel_output(tmp_path, folds=LOSO, loso=True, arch=arch, variant=variant, **extra)
    with pytest.raises(ValueError, match="LOSO"):
        collect([src], tmp_path / "runs", prep, tmp_path / "lr_probe.json")
    assert not (tmp_path / "runs").exists()


def _files(run):
    return {p.relative_to(run): p.read_bytes() for p in sorted(run.rglob("*")) if p.is_file()}


def test_collect_adds_loso_folds_to_existing_B_run_without_overwrite(tmp_path):
    full, prep = _kernel_output(tmp_path / "full")                 # arm B, collected in v3
    lp, runs = tmp_path / "full" / "lr_probe.json", tmp_path / "runs"
    out = collect([full], runs, prep, lp)
    before, s0 = _files(out), json.loads((out / "summary.json").read_text())
    # arm B's LOSO-only kernel (v3 pool): FOLDS = [], no site fold, LOSO = True
    loso, _ = _kernel_output(tmp_path / "loso", folds=LOSO, loso=True, pool="v3")
    assert collect([loso], runs, prep, lp) == out
    assert sorted(p.name for p in runs.iterdir()) == ["dinov2_s_s1337"]          # no .partial left
    assert sorted(p.name for p in out.iterdir()) == sorted(list(FOLDS.values()) + ["summary.json"])
    after = _files(out)
    kept = {k: v for k, v in after.items() if k.parts[0] not in ("fold_loso_ostrava", "fold_loso_ropvl")}
    assert {k: v for k, v in kept.items() if k.name != "summary.json"} == {
        k: v for k, v in before.items() if k.name != "summary.json"}            # byte for byte
    assert sorted(set(after) - set(before)) == sorted(
        Path(f"fold_loso_{s}") / f for s in ("ostrava", "ropvl")
        for f in ("val_predictions.npz", "dev_predictions.npz", "metrics.json", "config.yaml"))
    s1 = json.loads((out / "summary.json").read_text())
    assert {k: v for k, v in s1.items() if k != "folds"} == {k: v for k, v in s0.items() if k != "folds"}
    assert {k: s1["folds"][k] for k in s0["folds"]} == s0["folds"]
    assert sorted(s1["folds"]) == sorted(FOLDS) and s1["folds"]["foldloso_ropvl"]["kernel"] == "out"
    # a second collect of the same LOSO kernel overwrites nothing
    with pytest.raises(FileExistsError):
        collect([loso], runs, prep, lp)
    assert _files(out) == after
    assert sorted(p.name for p in runs.iterdir()) == ["dinov2_s_s1337"]


def test_B_loso_folds_need_the_collected_run_and_arrive_alone(tmp_path):
    loso, prep = _kernel_output(tmp_path / "loso", folds=LOSO, loso=True, pool="v3")
    lp, runs = tmp_path / "loso" / "lr_probe.json", tmp_path / "runs"
    with pytest.raises(FileNotFoundError, match="dinov2_s_s1337"):        # B's run not collected
        collect([loso], runs, prep, lp)
    assert not runs.exists()
    mixed, _ = _kernel_output(tmp_path / "mixed", folds=["foldsite"] + LOSO, loso=True)
    with pytest.raises(ValueError, match="LOSO-only"):
        collect([mixed], runs, prep, lp)
    no_flag, _ = _kernel_output(tmp_path / "no_flag", folds=LOSO)          # summary loso false
    with pytest.raises(ValueError, match="loso"):
        collect([no_flag], runs, prep, lp)
    assert not runs.exists()
