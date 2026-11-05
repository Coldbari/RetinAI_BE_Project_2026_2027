"""Task 4: the two ViT-S/14 arms (retina-pretrained RETFound-Green, DINOv2 control)."""
from pathlib import Path

import pytest
import torch

from models.common.architectures import (VIT_ARMS, build_from_cfg, build_vit_arm,
                                         get_target_layer, load_rfg_backbone)
from tests.v3_helpers import _cfg

RFG = (Path(__file__).resolve().parent.parent
       / "experiments/eye_spike/weights/retfound_green/retfoundgreen_statedict.pth")


@pytest.fixture(scope="module")
def fake_rfg(tmp_path_factory):
    """A backbone-only state dict (no head) with RETFound-Green's key layout, random values."""
    m = build_vit_arm("retfound_green", 6, 392, 0.3, 0.1, None, pretrained=False)
    sd = {k: v for k, v in m.state_dict().items() if not k.startswith("head.")}
    p = tmp_path_factory.mktemp("rfg") / "rfg.pth"
    torch.save(sd, p)
    return p, sd


def test_dinov2_s_builds_at_392_with_six_outputs():
    m = build_vit_arm("dinov2_s", 6, 392, 0.3, 0.1, None, pretrained=False)
    assert m.pos_embed.shape[1] == 784 and m(torch.zeros(1, 3, 392, 392)).shape == (1, 6)
    assert m.global_pool == "avg"


@pytest.mark.skipif(not RFG.exists(), reason="RETFound-Green weights not on this machine")
def test_retfound_green_loads_every_backbone_key():
    m = build_vit_arm("retfound_green", 6, 392, 0.3, 0.1, str(RFG), pretrained=True)
    rep = load_rfg_backbone(m, str(RFG))
    assert rep["missing"] == [] and rep["unexpected"] == []
    file_pe = torch.load(RFG, map_location="cpu", weights_only=True)["pos_embed"]
    assert torch.equal(m.pos_embed, file_pe)  # a load that did nothing would fail here


def test_build_from_cfg_reads_weights_path_and_drop_path(fake_rfg):
    path, sd = fake_rfg
    cfg = _cfg(arch="dinov2_s", image_size=392, drop_path=0.2)
    m = build_from_cfg(cfg, pretrained=False)
    assert any(getattr(b.drop_path1, "drop_prob", 0) > 0 for b in m.blocks)
    # weights_path is read: the loaded tensor is the file's, not a fresh random init
    cfg = _cfg(arch="retfound_green", image_size=392, weights_path=str(path))
    m = build_from_cfg(cfg, pretrained=True)
    assert torch.equal(m.cls_token, sd["cls_token"])


def test_weights_path_falls_back_to_env(fake_rfg, monkeypatch):
    path, sd = fake_rfg
    monkeypatch.setenv("RETINAI_RFG_WEIGHTS", str(path))
    m = build_from_cfg(_cfg(arch="retfound_green", image_size=392), pretrained=True)
    assert torch.equal(m.cls_token, sd["cls_token"])


def test_retfound_green_pretrained_without_weights_is_an_error(monkeypatch):
    monkeypatch.delenv("RETINAI_RFG_WEIGHTS", raising=False)
    with pytest.raises(ValueError, match="weights"):
        build_from_cfg(_cfg(arch="retfound_green", image_size=392), pretrained=True)


def test_load_rfg_backbone_raises_listing_bad_keys(fake_rfg, tmp_path):
    _, sd = fake_rfg
    bad = {k: v for k, v in sd.items() if k != "cls_token"}
    bad["bogus.weight"] = torch.zeros(1)
    p = tmp_path / "bad.pth"
    torch.save(bad, p)
    m = build_vit_arm("retfound_green", 6, 392, 0.3, 0.1, None, pretrained=False)
    with pytest.raises(RuntimeError) as e:
        load_rfg_backbone(m, str(p))
    assert "cls_token" in str(e.value) and "bogus.weight" in str(e.value)


def test_grad_cam_target_resolves_for_vit():
    m = build_vit_arm("dinov2_s", 6, 392, 0.3, 0.1, None, pretrained=False)
    assert set(VIT_ARMS) == {"retfound_green", "dinov2_s"}
    for arch in VIT_ARMS:
        assert get_target_layer(m, arch) is m.blocks[-1]
