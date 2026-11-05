"""Task 5 of the v4 plan: arm I's trainer additions (Amendment 5b), each off by default (arm B).

- ``train.ema_decay`` > 0: an exponential moving average of the weights (``uda.EMATeacher``),
  updated after every optimizer step, is what is validated and saved (weights.pth), as arm F's
  teacher is; last_checkpoint.pth carries it under ``"ema"``.
- ``train.head_warmup_epochs`` = n > 0: in epochs 1..n only ``optim.head_param_names`` train; from
  epoch n+1 every parameter has its own flag back. The OneCycle schedule is still built once for all
  epochs and steps on every batch (Review Focus 4).
- ``train.aug_v4.fourier_p`` > 0: ``fourier_swap_batch`` on every training batch, on the device.

End to end on CPU with tiny cached images, as in test_v3_uda.py.
"""
import sys
from pathlib import Path

import pandas as pd
import pytest

torch = pytest.importorskip("torch")
yaml = pytest.importorskip("yaml")

import numpy as np  # noqa: E402
from torch import nn  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import models.common.train_utils as T  # noqa: E402
from models.common import uda as U  # noqa: E402
from models.common.config import load_config  # noqa: E402

B_CFG = ROOT / "configs/rop_staging_dinov2.yaml"
UDA_CFG = ROOT / "configs/rop_staging_dinov2_uda.yaml"
BAG_CFG = ROOT / "configs/rop_staging_dinov2_bag.yaml"
V4_BLOCK = {"enabled": True, "rotation": 180, "min_crop_scale": 0.85, "gamma": [0.7, 1.4],
            "gamma_p": 0.5, "noise_sigma": 0.02, "noise_p": 0.3, "fourier_block": 3,
            "fourier_p": 0.5}
CPU = torch.device("cpu")


class _TinyViT(nn.Module):
    """The parameter layout ``build_param_groups`` reads off a timm ViT: embeddings, .blocks, norm, head."""

    def __init__(self, dim=8, n_classes=6):
        super().__init__()
        self.cls_token = nn.Parameter(torch.zeros(1, dim))
        self.patch_embed = nn.Conv2d(3, dim, 7, stride=7)          # 14 x 14 -> 2 x 2
        self.blocks = nn.ModuleList([nn.Linear(dim, dim) for _ in range(2)])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, n_classes)

    def forward(self, x):
        x = self.patch_embed(x).flatten(2).mean(2) + self.cls_token
        for b in self.blocks:
            x = x + torch.tanh(b(x))
        return self.head(self.norm(x))


HEAD = {"norm.weight", "norm.bias", "head.weight", "head.bias"}


class _Log:
    def __init__(self, d):
        self.dir, self.saved = d, []

    def save_weights(self, sd):
        self.saved.append({k: v.clone() for k, v in sd.items()})


def _cfg(tmp_path, *overrides, base=B_CFG, aug_v4=None):
    """``base`` (arm B by default) with ``train.aug_v4 = aug_v4`` when given, sized for CPU."""
    data = yaml.safe_load(base.read_text())
    if aug_v4 is not None:
        data["train"]["aug_v4"] = aug_v4
    path = tmp_path / f"cfg_{len(list(tmp_path.glob('cfg_*.yaml')))}.yaml"
    path.write_text(yaml.safe_dump(data))
    return load_config(path, overrides=[
        "train.epochs=2", "train.batch_size=4", "train.num_workers=0", "train.amp=false",
        "preprocess.image_size=14", "preprocess.cached=true",
        f"preprocess.cache_root={tmp_path / 'cache'}", *overrides])


def _loaders(tmp_path, cfg):
    """10 train + 4 val cached 20 x 20 images -> 3 train batches of (4, 4, 2)."""
    from PIL import Image
    rows = [{"image_path": f"data/rop/shenzhen/x/img{j}.jpg", "label": j % 6, "patient_id": f"shenzhen:{j}",
             "source": "shenzhen", "split": "train" if j < 10 else "val"} for j in range(14)]
    man = pd.DataFrame(rows)
    rng = np.random.default_rng(0)
    for p in man.image_path:
        f = tmp_path / "cache" / "rop" / Path(p).relative_to("data/rop").with_suffix(".jpg")
        f.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(rng.integers(0, 255, (20, 20, 3), dtype=np.uint8)).save(f)
    loaders = T.build_dataloaders(cfg, man)
    assert len(loaders["train"]) == 3
    return loaders


def _record_emas(monkeypatch):
    """Patch ``uda.EMATeacher`` to record each instance and count its ``update`` calls."""
    made = []
    Base = U.EMATeacher

    class Rec(Base):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            self.updates = 0
            made.append(self)

        def update(self, student):
            self.updates += 1
            super().update(student)

    monkeypatch.setattr(U, "EMATeacher", Rec)
    return made


# ── head warm-up ─────────────────────────────────────────────────────────────
def test_head_param_names_is_build_param_groups_head_group():
    from models.common.architectures import build_model, build_vit_arm
    from models.common.optim import build_param_groups, head_param_names

    vit = build_vit_arm("dinov2_s", 6, 392, 0.3, 0.1, None, pretrained=False)
    names = head_param_names(vit)
    assert names == HEAD                                     # norm + head, after the 12 blocks
    by_id = {id(p): n for n, p in vit.named_parameters()}
    (group,) = [g for g in build_param_groups(vit, 1e-4, 0.05, 0.8) if g["name"] == "head"]
    assert {by_id[id(p)] for p in group["params"]} == names
    assert head_param_names(_TinyViT()) == HEAD
    # no .blocks (one "all" param group there): the top-level classifier module only
    assert head_param_names(build_model("efficientnet_b0", 6, pretrained=False)) == {
        "classifier.1.weight", "classifier.1.bias"}
    assert head_param_names(build_model("resnet50", 6, pretrained=False)) == {"fc.1.weight", "fc.1.bias"}
    assert head_param_names(nn.Sequential(nn.Flatten(), nn.Linear(3, 6))) == set()


def test_head_warmup_freezes_backbone_in_epoch_1_and_unfreezes_after(tmp_path, capsys, monkeypatch):
    cfg = _cfg(tmp_path, "train.head_warmup_epochs=1")
    loaders = _loaders(tmp_path, cfg)
    steps = len(loaders["train"])
    torch.manual_seed(0)
    model = _TinyViT()
    model.patch_embed.bias.requires_grad_(False)      # frozen by the model itself: must stay frozen
    start = {n: p.detach().clone() for n, p in model.named_parameters()}

    scheds = []
    real_build = T.build_optimizer_and_scheduler
    monkeypatch.setattr(T, "build_optimizer_and_scheduler",
                        lambda *a, **k: (r := real_build(*a, **k), scheds.append(r[1]))[0])
    flags, states, evaluated = [], [], []
    real_eval = T.evaluate

    def spy(m, *a, **k):              # runs after each epoch's training, before the next epoch
        evaluated.append(m)
        flags.append({n: p.requires_grad for n, p in model.named_parameters()})
        states.append({n: p.detach().clone() for n, p in model.named_parameters()})
        return real_eval(m, *a, **k)

    monkeypatch.setattr(T, "evaluate", spy)
    T.train_model(cfg, model, loaders, _Log(tmp_path), CPU)

    (sched,) = scheds                                        # built once, for all epochs
    assert sched.total_steps == 2 * steps
    assert sched.last_epoch == 2 * steps                     # stepped on every batch, warm-up included
    assert evaluated == [model, model]                       # no EMA: the student is validated
    e1, e2 = flags
    assert {n for n, g in e1.items() if g} == HEAD           # epoch 1: the head only
    assert e2 == {n: n != "patch_embed.bias" for n in e2}    # epoch 2: the model's own flags back
    s1, s2 = states
    for n in start:
        if n in HEAD:
            assert not torch.equal(s1[n], start[n]), n       # the head trained in epoch 1
        else:
            assert torch.equal(s1[n], start[n]), n           # the backbone did not move
    for n in set(start) - HEAD - {"patch_embed.bias"}:
        assert not torch.equal(s2[n], s1[n]), n              # and trains from epoch 2
    assert torch.equal(s2["patch_embed.bias"], start["patch_embed.bias"])
    out = capsys.readouterr().out
    assert "[v4] epoch 1: warmup=True ema_decay=0.0 fourier_p=0.0" in out
    assert "[v4] epoch 2: warmup=False ema_decay=0.0 fourier_p=0.0" in out


# ── EMA weights ──────────────────────────────────────────────────────────────
def test_ema_weights_are_saved_and_differ_from_student(tmp_path, capsys, monkeypatch):
    cfg = _cfg(tmp_path, "train.ema_decay=0.9")
    loaders = _loaders(tmp_path, cfg)
    emas = _record_emas(monkeypatch)
    evaluated = []
    real_eval = T.evaluate
    monkeypatch.setattr(T, "evaluate", lambda m, *a, **k: evaluated.append(m) or real_eval(m, *a, **k))
    torch.manual_seed(0)
    model = _TinyViT()
    log = _Log(tmp_path)
    T.train_model(cfg, model, loaders, log, CPU)

    (ema,) = emas
    assert ema.decay == 0.9 and ema.updates == 2 * len(loaders["train"])   # after every step
    assert evaluated == [ema.model, ema.model]               # validation = the EMA
    assert len(log.saved) == 2
    for k, v in ema.model.state_dict().items():              # weights.pth = the EMA
        assert torch.equal(log.saved[-1][k], v)
    student = model.state_dict()
    assert not torch.equal(log.saved[-1]["head.weight"], student["head.weight"])   # not the student
    ck = torch.load(tmp_path / "last_checkpoint.pth")
    assert "ema" in ck and "teacher" not in ck
    for k, v in ema.model.state_dict().items():
        assert torch.equal(ck["ema"][k], v)
    for k, v in student.items():
        assert torch.equal(ck["model"][k], v)
    assert "[v4] epoch 2: warmup=False ema_decay=0.9 fourier_p=0.0" in capsys.readouterr().out


# ── Fourier swap ─────────────────────────────────────────────────────────────
def test_fourier_swap_runs_on_every_training_batch_on_the_device(tmp_path, capsys, monkeypatch):
    cfg = _cfg(tmp_path, aug_v4=V4_BLOCK)
    loaders = _loaders(tmp_path, cfg)
    calls = []
    real_swap = T.fourier_swap_batch

    def spy(x, block, p, *a, **k):
        out = real_swap(x, block, p, *a, **k)
        calls.append((x, block, p, out))
        return out

    monkeypatch.setattr(T, "fourier_swap_batch", spy)
    torch.manual_seed(0)
    model = _TinyViT()
    seen = []
    model.register_forward_pre_hook(lambda m, args: seen.append(args[0]))
    T.train_model(cfg, model, loaders, _Log(tmp_path), CPU)

    assert len(calls) == 2 * len(loaders["train"])
    assert all(b == 3 and p == 0.5 and x.device == CPU for x, b, p, _ in calls)
    assert sum(any(t is out for t in seen) for *_, out in calls) == len(calls)   # what the model saw
    assert any(not torch.equal(x, out) for x, _, _, out in calls)
    assert "[v4] epoch 1: warmup=False ema_decay=0.0 fourier_p=0.5" in capsys.readouterr().out


# ── arm B: none of the three keys ────────────────────────────────────────────
def test_without_the_v4_keys_the_trainer_is_arm_B(tmp_path, capsys, monkeypatch):
    cfg = _cfg(tmp_path)
    assert not any(k in cfg.train for k in ("ema_decay", "head_warmup_epochs", "aug_v4"))
    loaders = _loaders(tmp_path, cfg)
    emas = _record_emas(monkeypatch)
    monkeypatch.setattr(T, "fourier_swap_batch", lambda *a, **k: pytest.fail("swap without aug_v4"))
    torch.manual_seed(0)
    model = _TinyViT()
    flags, evaluated = [], []
    real_eval = T.evaluate

    def spy(m, *a, **k):
        evaluated.append(m)
        flags.append(all(p.requires_grad for p in model.parameters()))
        return real_eval(m, *a, **k)

    monkeypatch.setattr(T, "evaluate", spy)
    T.train_model(cfg, model, loaders, _Log(tmp_path), CPU)
    assert emas == [] and evaluated == [model, model] and flags == [True, True]
    assert set(torch.load(tmp_path / "last_checkpoint.pth")) == {"epoch", "model", "optimizer", "best"}
    assert "[v4]" not in capsys.readouterr().out


# ── refusals ─────────────────────────────────────────────────────────────────
_STUB = {"train": [None] * 3, "val": [], "counts": [2] * 6, "target": []}   # refused before any batch


def test_ema_and_uda_cannot_combine(tmp_path):
    cfg = _cfg(tmp_path, "train.ema_decay=0.999", base=UDA_CFG)
    with pytest.raises(ValueError, match="uda.*ema_decay"):
        T.train_model(cfg, _TinyViT(), dict(_STUB), _Log(tmp_path), CPU)


@pytest.mark.parametrize("base, aug_v4, overrides, match", [
    (UDA_CFG, V4_BLOCK, [], "uda.*aug_v4"),
    (BAG_CFG, None, ["train.head_warmup_epochs=1"], "bag_loss.*head_warmup_epochs"),
    (BAG_CFG, V4_BLOCK, [], "bag_loss.*aug_v4"),
    (B_CFG, None, ["train.ema_decay=1.0"], "ema_decay"),
    (B_CFG, None, ["train.ema_decay=-0.1"], "ema_decay"),
    (B_CFG, None, ["train.head_warmup_epochs=-1"], "head_warmup_epochs"),
])
def test_other_v4_combinations_and_values_are_refused(tmp_path, base, aug_v4, overrides, match):
    cfg = _cfg(tmp_path, *overrides, base=base, aug_v4=aug_v4)
    with pytest.raises(ValueError, match=match):
        T.train_model(cfg, _TinyViT(), dict(_STUB), _Log(tmp_path), CPU)


def test_head_warmup_refuses_a_model_with_no_head_parameters(tmp_path):
    cfg = _cfg(tmp_path, "train.head_warmup_epochs=1")
    model = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(3, 6))
    with pytest.raises(ValueError, match="head_warmup_epochs"):
        T.train_model(cfg, model, dict(_STUB), _Log(tmp_path), CPU)
