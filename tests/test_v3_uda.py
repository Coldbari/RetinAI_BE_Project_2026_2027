"""Arm F (PROTOCOL_V2 Amendment 4b): self-ensembling adaptation on unlabelled Multi-View dev photos."""
import ast
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from models.common import uda as U  # noqa: E402
from models.common.config import load_config  # noqa: E402

PREP = ROOT / "results/prepared/rop_staging"
UDA_CFG = ROOT / "configs/rop_staging_dinov2_uda.yaml"
# Every file whose code runs inside a training step of a v3 kernel.
TRAINING_CODE = ["models/common/uda.py", "models/common/train_utils.py", "models/common/bag_loss.py",
                 "models/common/datasets.py", "models/common/preprocessing.py", "train.py",
                 "kaggle_kernels/rop_v3_finetune.py"]


# ── the target data: dev IMAGES, never dev labels ────────────────────────────
def test_target_frame_carries_image_paths_only():
    frame = U.load_target_frame(PREP / "test_dev.csv")
    assert list(frame.columns) == ["image_path"]
    assert len(frame) == 663 and frame.image_path.str.startswith("data/rop/multiview/").all()


def test_target_dataset_refuses_a_frame_with_any_label_column():
    for col in ("label", "cls6", "split"):
        frame = pd.DataFrame({"image_path": ["data/rop/multiview/a.jpg"], col: [1]})
        with pytest.raises(ValueError, match="image paths only"):
            U.TargetDataset(frame, None, None)


def test_every_target_read_csv_parses_the_image_path_column_only():
    """AST: no read_csv in uda.py without usecols=TARGET_COLUMNS, so a label column is never parsed."""
    tree = ast.parse((ROOT / "models/common/uda.py").read_text())
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and ast.unparse(n.func).endswith("read_csv")]
    assert calls
    for c in calls:
        kw = {k.arg: ast.unparse(k.value) for k in c.keywords}
        assert kw.get("usecols") == "TARGET_COLUMNS", ast.unparse(c)
    assert U.TARGET_COLUMNS == ["image_path"]


def _docstrings(tree):
    out = set()
    for n in ast.walk(tree):
        if isinstance(n, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and n.body:
            first = n.body[0]
            if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant):
                out.add(id(first.value))
    return out


@pytest.mark.parametrize("rel", TRAINING_CODE)
def test_training_code_never_names_test_locked(rel):
    """AST: no string in the training code (docstrings and comments aside) names test_locked."""
    tree = ast.parse((ROOT / rel).read_text())
    docs = _docstrings(tree)
    hits = [n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, str)
            and id(n) not in docs and "test_locked" in n.value]
    hits += [ast.unparse(n) for n in ast.walk(tree) if isinstance(n, ast.Name) and "locked" in n.id]
    assert not hits, (rel, hits)


def test_settings_refuse_a_locked_or_non_dev_target():
    cfg = load_config(str(UDA_CFG))
    assert U.uda_settings(cfg)["target_manifest"] == "results/prepared/rop_staging/test_dev.csv"
    for bad in ("results/prepared/rop_staging/test_locked.csv", "results/prepared/rop_staging/fold_0.csv"):
        with pytest.raises(ValueError, match="test_dev.csv"):
            U.uda_settings(load_config(str(UDA_CFG), overrides=[f"train.uda.target_manifest={bad}"]))


def test_settings_read_from_the_config():
    assert U.uda_settings(load_config(str(UDA_CFG))) == {
        "method": "self_ensembling", "ema_decay": 0.99, "conf_threshold": 0.90, "weight": 1.0,
        "ramp_epochs": 2, "target_manifest": "results/prepared/rop_staging/test_dev.csv",
        "target_batch": 16}
    assert U.uda_settings(load_config(str(ROOT / "configs/rop_staging_dinov2.yaml"))) is None
    with pytest.raises(ValueError, match="true or false"):
        U.uda_settings(load_config(str(UDA_CFG), overrides=["train.uda.enabled=yes"]))
    with pytest.raises(ValueError, match="method"):
        U.uda_settings(load_config(str(UDA_CFG), overrides=["train.uda.method=mean_teacher"]))
    with pytest.raises(ValueError, match="ema_decay"):
        U.uda_settings(load_config(str(UDA_CFG), overrides=["train.uda.ema_decay=1.0"]))


def _images(tmp_path, paths, size=20, seed=0):
    from PIL import Image
    rng = np.random.default_rng(seed)
    for p in paths:
        f = tmp_path / "cache" / "rop" / Path(p).relative_to("data/rop").with_suffix(".jpg")
        f.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(rng.integers(0, 255, (size, size, 3), dtype=np.uint8)).save(f)


def _cfg(tmp_path, target_csv, *extra):
    return load_config(str(UDA_CFG), overrides=[
        "train.epochs=2", "train.batch_size=4", "train.num_workers=0", "train.amp=false",
        "preprocess.image_size=14", "preprocess.cached=true", f"preprocess.cache_root={tmp_path / 'cache'}",
        "train.uda.target_batch=2", f"train.uda.target_manifest={target_csv}", *extra])


def _dev_like(tmp_path, n=5):
    """A dev-shaped manifest (label columns included, as on disk) under a test_dev.csv name."""
    paths = [f"data/rop/multiview/ROP-DATASET/dataset/9/x.{i}.jpg" for i in range(n)]
    _images(tmp_path, paths, seed=1)
    csv = tmp_path / "prep" / "test_dev.csv"
    csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"image_path": paths, "label": [3] * n, "split": "test", "patient_id": "multiview:9",
                  "source": "multiview", "cls6": "stage_3"}).to_csv(csv, index=False)
    return csv


def test_target_loader_yields_a_weak_and_a_strong_view_and_no_label(tmp_path):
    csv = _dev_like(tmp_path)
    cfg = _cfg(tmp_path, csv)
    loader = U.target_loader(cfg, U.uda_settings(cfg), num_workers=0, pin_memory=False)
    assert len(loader) == 2                                # 5 photos, batches of 2, last one dropped
    batch = next(iter(loader))
    assert isinstance(batch, (list, tuple)) and len(batch) == 2
    weak, strong = batch
    assert weak.shape == strong.shape == (2, 3, 14, 14) and weak.dtype == strong.dtype == torch.float32
    assert not hasattr(loader.dataset, "targets") and not hasattr(loader.dataset, "df")


def test_weak_view_is_resize_and_flips_only(tmp_path):
    cfg = _cfg(tmp_path, _dev_like(tmp_path))
    names = [type(t).__name__ for t in U.weak_transform(cfg).transforms]
    assert names == ["Resize", "RandomHorizontalFlip", "RandomVerticalFlip", "ToTensor", "Normalize"]
    from models.common.preprocessing import build_transforms
    strong = [type(t).__name__ for t in U.target_loader(cfg, U.uda_settings(cfg), 0, False).dataset.strong.transforms]
    assert strong == [type(t).__name__ for t in build_transforms(cfg, "train").transforms]


# ── the teacher, the loss, the ramp ─────────────────────────────────────────
def test_teacher_starts_as_a_copy_and_follows_the_student_by_ema():
    torch.manual_seed(0)
    student = torch.nn.Sequential(torch.nn.Linear(3, 6), torch.nn.Dropout(0.5)).train()
    t = U.EMATeacher(student, decay=0.9)
    assert not t.model.training and all(not p.requires_grad for p in t.model.parameters())
    for a, b in zip(t.model.parameters(), student.parameters()):
        assert torch.equal(a, b) and a.data_ptr() != b.data_ptr()
    before = [p.detach().clone() for p in t.model.parameters()]
    with torch.no_grad():
        for p in student.parameters():
            p.add_(1.0)
    t.update(student)
    for a, b0, s in zip(t.model.parameters(), before, student.parameters()):
        assert torch.allclose(a, 0.9 * b0 + 0.1 * s)
    assert not t.model.training


def test_consistency_loss_masks_unconfident_teacher_rows():
    s = torch.tensor([[2.0, 0, 0, 0, 0, 0], [0, 1.0, 0, 0, 0, 0]])
    t = torch.tensor([[9.0, 0, 0, 0, 0, 0], [0.1, 0, 0, 0, 0, 0]])          # row 0 confident, row 1 not
    loss, rate = U.consistency_loss(s, t, threshold=0.90)
    ps, pt = torch.softmax(s, 1), torch.softmax(t, 1)
    assert rate.item() == pytest.approx(0.5)
    assert loss.item() == pytest.approx(((ps[0] - pt[0]) ** 2).sum().item() / 2)      # mean over BOTH rows
    zero, _ = U.consistency_loss(t, t, threshold=0.0)
    assert zero.item() == pytest.approx(0.0, abs=1e-12)
    s.requires_grad_(True)
    t.requires_grad_(True)
    U.consistency_loss(s, t, 0.9)[0].backward()             # the teacher side is a fixed target
    assert s.grad is not None and t.grad is None


def test_weight_ramps_linearly_by_step_over_the_first_two_epochs():
    assert U.ramp_weight(0, 200, 1.0) == 0.0
    assert U.ramp_weight(50, 200, 1.0) == pytest.approx(0.25)
    assert U.ramp_weight(200, 200, 1.0) == 1.0 and U.ramp_weight(5000, 200, 1.0) == 1.0
    assert U.ramp_weight(3, 0, 1.0) == 1.0


def test_cycle_restarts_an_exhausted_loader():
    it = U.cycle([1, 2])
    assert [next(it) for _ in range(5)] == [1, 2, 1, 2, 1]


# ── end to end on CPU: build_dataloaders + train_model ──────────────────────
def test_two_epochs_train_with_the_teacher_validated_and_saved(tmp_path, capsys, monkeypatch):
    from torch import nn
    import models.common.train_utils as T

    rows = [{"image_path": f"data/rop/shenzhen/x/img{j}.jpg", "label": j % 6, "patient_id": f"shenzhen:{j}",
             "source": "shenzhen", "split": "train" if j < 10 else "val"} for j in range(14)]
    man = pd.DataFrame(rows)
    _images(tmp_path, man.image_path)
    csv = _dev_like(tmp_path, n=5)
    cfg = _cfg(tmp_path, csv, "train.uda.conf_threshold=0.0", "train.uda.ramp_epochs=1")

    reads = []
    real = pd.read_csv
    monkeypatch.setattr(pd, "read_csv", lambda *a, **k: reads.append((str(a[0]), k)) or real(*a, **k))
    loaders = T.build_dataloaders(cfg, man)
    assert [k for p, k in reads if p.endswith("test_dev.csv")] == [{"usecols": ["image_path"]}]
    assert len(loaders["train"]) == 3 and len(loaders["target"]) == 2   # the target cycles

    teachers, evaluated, saved = [], [], []
    Base = U.EMATeacher
    monkeypatch.setattr(U, "EMATeacher", type("Rec", (Base,), {
        "__init__": lambda self, *a, **k: (Base.__init__(self, *a, **k), teachers.append(self))[0]}))
    real_eval = T.evaluate
    monkeypatch.setattr(T, "evaluate", lambda m, *a, **k: evaluated.append(m) or real_eval(m, *a, **k))

    class Log:
        dir = tmp_path

        def save_weights(self, sd):
            saved.append({k: v.clone() for k, v in sd.items()})

    model = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(3, 6))
    history, _ = T.train_model(cfg, model, loaders, Log(), torch.device("cpu"))
    out = capsys.readouterr().out
    assert "[uda] target: 5 unlabelled photos" in out
    assert "[uda] epoch 1: mask_rate=1.000 target_loss=" in out and "[uda] epoch 2: mask_rate=1.000" in out
    import re
    tmax = [float(x) for x in re.findall(r"teacher_maxprob_mean=([0-9.]+)", out)]
    assert len(tmax) == 2 and all(1 / 6 <= t <= 1 for t in tmax)
    assert np.isfinite(history["train_loss"]).all()
    (teacher,) = teachers
    assert evaluated == [teacher.model, teacher.model]                   # validation = the teacher
    assert len(saved) == 2
    for k, v in teacher.model.state_dict().items():                      # weights.pth = the teacher
        assert torch.equal(saved[-1][k], v)
    assert not torch.equal(saved[-1]["2.weight"], model.state_dict()["2.weight"])   # not the student
    ck = torch.load(tmp_path / "last_checkpoint.pth")
    assert set(ck) >= {"model", "teacher"}


def test_uda_refuses_the_bag_loss_and_mixup(tmp_path):
    import models.common.train_utils as T
    from torch import nn
    csv = _dev_like(tmp_path)
    man = pd.DataFrame([{"image_path": "data/rop/shenzhen/x/a.jpg", "label": 0, "split": "train",
                         "patient_id": "shenzhen:0", "source": "shenzhen"}])
    _images(tmp_path, man.image_path)
    for extra in (["train.bag_loss.enabled=true", "train.bag_loss.k=2", "train.bag_loss.tau=1.0",
                   "train.bag_loss.source=ostrava"], ["train.mixup_alpha=0.2"]):
        cfg = _cfg(tmp_path, csv, *extra)
        with pytest.raises(ValueError, match="uda"):
            loaders = T.build_dataloaders(cfg, man)
            T.train_model(cfg, nn.Linear(1, 6), loaders, None, torch.device("cpu"))
