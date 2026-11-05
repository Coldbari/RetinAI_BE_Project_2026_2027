"""Task 5: per-group learning rates (layer decay) that OneCycleLR actually applies."""
import pytest

from models.common.architectures import build_from_cfg, build_vit_arm
from models.common.optim import (build_optimizer_and_scheduler, build_param_groups,
                                 format_group_lrs)
from tests.v3_helpers import _cfg


@pytest.fixture(scope="module")
def vit():
    return build_vit_arm("dinov2_s", 6, 392, 0.3, 0.1, None, pretrained=False)


def test_layer_decay_groups_scale_geometrically(vit):
    groups = build_param_groups(vit, 1e-4, 0.05, 0.8)
    lrs = {g["name"]: g["lr"] for g in groups}
    assert lrs["head"] == pytest.approx(1e-4)
    assert lrs["blocks.11"] == pytest.approx(1e-4 * 0.8)
    assert lrs["blocks.0"] == pytest.approx(1e-4 * 0.8 ** 12)
    assert lrs["embed"] == pytest.approx(1e-4 * 0.8 ** 13)
    assert all(g["weight_decay"] == 0.05 for g in groups)


def test_every_parameter_in_exactly_one_group(vit):
    groups = build_param_groups(vit, 1e-4, 0.05, 0.8)
    ids = [id(p) for g in groups for p in g["params"]]
    assert len(ids) == len(set(ids)) == len(list(vit.parameters()))
    by_name = {g["name"]: {id(p) for p in g["params"]} for g in groups}
    named = dict(vit.named_parameters())
    assert id(named["norm.weight"]) in by_name["head"]
    assert id(named["head.weight"]) in by_name["head"]
    assert id(named["cls_token"]) in by_name["embed"]
    assert id(named["reg_token"]) in by_name["embed"]
    assert id(named["pos_embed"]) in by_name["embed"]


def test_onecycle_receives_a_max_lr_per_group(vit):
    cfg = _cfg(arch="dinov2_s", image_size=392, layer_decay=0.8, lr=1e-4, epochs=3)
    opt, sched = build_optimizer_and_scheduler(cfg, vit, steps_per_epoch=10)
    ratios = {round(g["max_lr"] / 1e-4, 4) for g in opt.param_groups}
    assert len(ratios) > 1                     # not one scalar for everything
    top = next(g for g in opt.param_groups if g["name"] == "head")
    assert top["max_lr"] == pytest.approx(1e-4)


def test_no_layer_decay_keeps_single_group():
    cfg = _cfg(arch="effnetv2s", epochs=3)
    model = build_from_cfg(cfg, pretrained=False)
    opt, _ = build_optimizer_and_scheduler(cfg, model, 10)
    assert len(opt.param_groups) == 1
    g = opt.param_groups[0]
    lr = float(cfg.train.lr)
    assert g["initial_lr"] == pytest.approx(lr / 25)     # OneCycle default div_factor, as before
    assert g["max_lr"] == lr
    assert g["weight_decay"] == float(cfg.train.weight_decay)
    assert len(g["params"]) == len(list(model.parameters()))


def test_layer_decay_one_is_single_group(vit):
    cfg = _cfg(arch="dinov2_s", image_size=392, layer_decay=1.0, epochs=3)
    opt, _ = build_optimizer_and_scheduler(cfg, vit, 10)
    assert len(opt.param_groups) == 1


def test_format_group_lrs(vit):
    cfg = _cfg(arch="dinov2_s", image_size=392, layer_decay=0.8, lr=1e-4, epochs=3)
    opt, _ = build_optimizer_and_scheduler(cfg, vit, 10)
    s = format_group_lrs(opt)
    assert s.startswith("[lr] head=") and "blocks.0=" in s
    cfg1 = _cfg(arch="effnetv2s", epochs=3)
    opt1, _ = build_optimizer_and_scheduler(cfg1, build_from_cfg(cfg1, pretrained=False), 10)
    assert format_group_lrs(opt1).startswith("[lr] all=")
