"""The two ViT arms must match arm C's AS-RUN recipe on everything except the designed differences."""
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from models.common.config import load_config  # noqa: E402

BASELINE = ROOT / "configs" / "baselines" / "rop_staging_flat_natural_asrun.yaml"
ARMS = [("rop_staging_rfg", "retfound_green", "half"),
        ("rop_staging_dinov2", "dinov2_s", "imagenet"),
        ("rop_staging_dinov2_v4data", "dinov2_s", "imagenet"),
        ("rop_staging_dinov2_v4recipe", "dinov2_s", "imagenet")]

ALLOWED_TO_DIFFER = {
    "model.arch", "model.weights_path", "model.drop_path", "preprocess.image_size",
    "preprocess.normalize", "preprocess.cached", "preprocess.cache_root", "train.lr",
    "train.layer_decay", "data.manifest", "seed", "train.arm_tag",
}
# Arm I (Amendment 5b) is a pre-registered exception: its recipe keys (incl. label_smoothing,
# a PARITY key) may differ. Per arm, so no other arm gets the exception.
ARM_I = "rop_staging_dinov2_v4recipe"
ARM_I_EXTRA = {"train.label_smoothing", "train.ema_decay", "train.head_warmup_epochs"}
RECIPE_I = {"train.ema_decay": 0.999, "train.head_warmup_epochs": 1, "train.label_smoothing": 0.10,
            "train.aug_v4.enabled": True, "train.aug_v4.rotation": 180,
            "train.aug_v4.min_crop_scale": 0.85, "train.aug_v4.gamma": [0.7, 1.4],
            "train.aug_v4.gamma_p": 0.5, "train.aug_v4.noise_sigma": 0.02,
            "train.aug_v4.noise_p": 0.3, "train.aug_v4.fourier_block": 3,
            "train.aug_v4.fourier_p": 0.5}

PARITY = {"train.class_balanced": True, "train.class_balanced_beta": 0.999,
          "train.label_smoothing": 0.05, "train.strong_aug": True,
          "train.site_randomization": True, "train.weighted_sampler": False,
          "train.batch_size": 16, "train.epochs": 20, "train.amp": True,
          "train.weight_decay": 0.05, "eval.early_stopping": False,
          "preprocess.circle_crop": True, "preprocess.letterbox": True,
          "preprocess.clahe": True, "preprocess.ben_graham": False}


def _flat(d, prefix=""):
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flat(v, key + "."))
        else:
            out[key] = v
    return out


def _get(cfg, dotted):
    node = cfg
    for part in dotted.split("."):
        node = node[part]
    return node


@pytest.mark.parametrize("name,arch,norm", ARMS)
def test_arm_config_matches_baseline_on_every_parity_key(name, arch, norm):
    cfg = load_config(str(ROOT / f"configs/{name}.yaml"))
    for key, val in PARITY.items():
        if name == ARM_I and key in ARM_I_EXTRA:
            continue
        assert _get(cfg, key) == val, key
    assert cfg.model.arch == arch and cfg.preprocess.normalize == norm
    assert cfg.preprocess.image_size == 392 and cfg.train.layer_decay == 0.8


@pytest.mark.parametrize("name,arch,norm", ARMS)
def test_arm_config_differs_from_asrun_baseline_only_in_allowed_keys(name, arch, norm):
    base = _flat(yaml.safe_load(BASELINE.read_text()))
    arm = _flat(yaml.safe_load((ROOT / f"configs/{name}.yaml").read_text()))
    bad = sorted(k for k in set(base) | set(arm)
                 if k not in ALLOWED_TO_DIFFER
                 and not (name == ARM_I and (k in ARM_I_EXTRA or k.startswith("train.aug_v4.")))
                 and base.get(k, "<missing>") != arm.get(k, "<missing>"))
    assert not bad, {k: (base.get(k, "<missing>"), arm.get(k, "<missing>")) for k in bad}


def test_both_arms_build_and_layer_decay_gives_multiple_groups():
    import torch
    from models.common.architectures import build_from_cfg
    from models.common.optim import build_optimizer_and_scheduler
    for name, arch, _ in ARMS:
        cfg = load_config(str(ROOT / f"configs/{name}.yaml"))
        model = build_from_cfg(cfg, pretrained=False).eval()
        if name == ARMS[0][0]:
            with torch.no_grad():
                assert model(torch.zeros(1, 3, 392, 392)).shape == (1, 6)
        opt, _sched = build_optimizer_and_scheduler(cfg, model, steps_per_epoch=10)
        assert len(opt.param_groups) > 1, name


def test_arm_D_config_is_B_plus_exactly_the_bag_loss_block():
    """PROTOCOL_V2 Amendment 3c: D = B's recipe; only train.bag_loss may differ, and only as specified."""
    b = _flat(yaml.safe_load((ROOT / "configs/rop_staging_dinov2.yaml").read_text()))
    d = _flat(yaml.safe_load((ROOT / "configs/rop_staging_dinov2_bag.yaml").read_text()))
    diff = {k: (b.get(k, "<missing>"), d.get(k, "<missing>")) for k in set(b) | set(d)
            if b.get(k, "<missing>") != d.get(k, "<missing>")}
    assert diff == {"train.bag_loss.enabled": ("<missing>", True),
                    "train.bag_loss.source": ("<missing>", "ostrava"),
                    "train.bag_loss.k": ("<missing>", 8),
                    "train.bag_loss.tau": ("<missing>", 1.0)}


def test_arm_G_config_is_B_at_518_px_and_nothing_else():
    """PROTOCOL_V2 Amendment 4(a): G = B at DINOv2's native 518 px; no other key may differ."""
    b = _flat(yaml.safe_load((ROOT / "configs/rop_staging_dinov2.yaml").read_text()))
    g = _flat(yaml.safe_load((ROOT / "configs/rop_staging_dinov2_518.yaml").read_text()))
    diff = {k: (b.get(k, "<missing>"), g.get(k, "<missing>")) for k in set(b) | set(g)
            if b.get(k, "<missing>") != g.get(k, "<missing>")}
    assert diff == {"preprocess.image_size": (392, 518)}
    assert 518 % 14 == 0 and 518 // 14 == 37                 # the 37x37 pre-training patch grid


def test_arm_F_config_is_B_plus_exactly_the_uda_block():
    """PROTOCOL_V2 Amendment 4(b): F = B's recipe plus self-ensembling; only train.uda may differ."""
    b = _flat(yaml.safe_load((ROOT / "configs/rop_staging_dinov2.yaml").read_text()))
    f = _flat(yaml.safe_load((ROOT / "configs/rop_staging_dinov2_uda.yaml").read_text()))
    diff = {k: (b.get(k, "<missing>"), f.get(k, "<missing>")) for k in set(b) | set(f)
            if b.get(k, "<missing>") != f.get(k, "<missing>")}
    want = {"enabled": True, "method": "self_ensembling", "ema_decay": 0.99, "conf_threshold": 0.90,
            "weight": 1.0, "ramp_epochs": 2, "target_manifest": "results/prepared/rop_staging/test_dev.csv",
            "target_batch": 16}
    assert diff == {f"train.uda.{k}": ("<missing>", v) for k, v in want.items()}


def _diff(a, b):
    return {k: (a.get(k, "<missing>"), b.get(k, "<missing>")) for k in set(a) | set(b)
            if a.get(k, "<missing>") != b.get(k, "<missing>")}


def _load(name):
    return _flat(yaml.safe_load((ROOT / f"configs/{name}.yaml").read_text()))


def test_arm_H_config_is_B_plus_only_manifest_and_tag():
    diff = _diff(_load("rop_staging_dinov2"), _load("rop_staging_dinov2_v4data"))
    assert diff == {"data.manifest": ("results/prepared/rop_staging/fold_0.csv",
                                      "results/prepared/rop_staging/v4/fold_0.csv"),
                    "train.arm_tag": ("<missing>", "v4data")}


def test_arm_I_config_is_H_plus_exactly_the_recipe_keys():
    from models.common.preprocessing import aug_v4_settings
    diff = _diff(_load("rop_staging_dinov2_v4data"), _load("rop_staging_dinov2_v4recipe"))
    want = {k: v for k, v in RECIPE_I.items() if k != "train.label_smoothing"}
    exp = {k: ("<missing>", v) for k, v in want.items()}
    exp["train.label_smoothing"] = (0.05, 0.10)
    exp["train.arm_tag"] = ("v4data", "v4recipe")
    assert diff == exp
    assert aug_v4_settings(load_config(str(ROOT / "configs/rop_staging_dinov2_v4recipe.yaml"))) is not None


def test_config_variant_reads_arm_tag():
    from models.common.v3_launch import config_variant
    assert config_variant({"train": {"arm_tag": "v4data"}}) == "v4data"
    assert config_variant({"train": {"arm_tag": "v4recipe"}}) == "v4recipe"
    assert config_variant({"train": {}}) == ""
    with pytest.raises(ValueError):
        config_variant({"train": {"arm_tag": "v4data", "bag_loss": {"enabled": True}}})
    with pytest.raises(ValueError):
        config_variant({"train": {"arm_tag": "bogus"}})
    for f in ("rop_staging_dinov2_v4data", "rop_staging_dinov2_v4recipe"):
        cfg = load_config(str(ROOT / f"configs/{f}.yaml"))
        assert config_variant(cfg.to_dict()) == f.split("_")[-1]


def test_check_loso_rules():
    from models.common.v3_launch import check_loso, VARIANTS
    for v in ("", "v4data", "v4recipe"):
        check_loso(v, True)
    for v in ("bag", "r518", "uda"):
        with pytest.raises(SystemExit):
            check_loso(v, True)
    for v in VARIANTS:
        check_loso(v, False)
