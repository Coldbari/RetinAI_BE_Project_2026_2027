"""Arm D (PROTOCOL_V2 Amendment 3c): the exam-level LSE bag loss on Ostrava sessions."""
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from models.common.bag_loss import (BagBatchSampler, bag_settings, interleave,  # noqa: E402
                                    pool_bags, session_plan)
from models.common.config import load_config  # noqa: E402

PREP = ROOT / "results/prepared/rop_staging"
BAG_CFG = ROOT / "configs/rop_staging_dinov2_bag.yaml"


def _frame(sessions, others=0, pid0=100):
    """Ostrava rows for sessions given as (n_frames, label); `others` non-Ostrava rows after them."""
    rows = []
    for s, (n, lab) in enumerate(sessions):
        for f in range(n):
            rows.append({"image_path": f"data/rop/ostrava/val/DG0/{pid0 + s}_D1_S01_{f + 1}.jpg",
                         "label": lab, "patient_id": f"ostrava:{pid0 + s}", "source": "ostrava"})
    for j in range(others):
        rows.append({"image_path": f"data/rop/shenzhen/x/img{j}.jpg", "label": j % 4,
                     "patient_id": f"shenzhen:{j}", "source": "shenzhen"})
    return pd.DataFrame(rows)


def _epoch(sampler):
    batches = list(sampler)
    img, bags = [], []
    for b in batches:
        groups = [g for _, g in b]
        if len(set(groups)) == len(groups):                     # every row its own group
            img.append([r for r, _ in b])
        else:
            for g in sorted(set(groups)):
                bags.append([r for r, gg in b if gg == g])
    return batches, img, bags


def test_session_plan_splits_singletons_from_multi_frame_sessions():
    df = _frame([(1, 0), (3, 1), (8, 2), (12, 5)], others=5)
    image_rows, bags = session_plan(df, "ostrava")
    assert sorted(image_rows.tolist()) == [0] + list(range(24, 29))   # the singleton + 5 others
    assert [len(b) for b in bags] == [3, 8, 12]
    assert sorted(np.concatenate([image_rows, *bags]).tolist()) == list(range(len(df)))


def test_sampler_visits_every_image_and_every_bag_once_per_epoch():
    df = _frame([(1, 0), (3, 1), (8, 2), (12, 5), (2, 0), (20, 3), (5, 1)], others=37)
    image_rows, bags = session_plan(df, "ostrava")
    s = BagBatchSampler(image_rows, bags, batch_size=16, k=8, seed=0)
    draws = []
    for _ in range(3):
        batches, img, got = _epoch(s)
        assert len(batches) == len(s)
        flat = [r for b in img for r in b]
        assert sorted(flat) == sorted(image_rows.tolist())                # each image once
        assert all(len(b) == 16 for b in img[:-1]) and 0 < len(img[-1]) <= 16
        bag_batches = [b for b in batches if len({g for _, g in b}) < len(b)]
        assert all(len({g for _, g in b}) <= 2 for b in bag_batches)     # <= 2 bags each
        assert len(bag_batches) == math.ceil(len(bags) / 2)
        owner = {r: i for i, b in enumerate(bags) for r in b}
        assert sorted(Counter(owner[b[0]] for b in got).values()) == [1] * len(bags)  # bag once
        for b in got:
            sess = bags[owner[b[0]]]
            assert len(set(b)) == len(b) == min(8, len(sess))             # no replacement, K cap
            assert set(b) <= set(sess.tolist())
        draws.append(sorted(next(b for b in got if len(bags[owner[b[0]]]) == 20)))
    assert len({tuple(d) for d in draws}) > 1          # a 20-frame session is re-drawn each epoch


def test_sampler_interleaves_the_two_kinds_proportionally():
    assert [k for k, _ in interleave(6, 3)] == list("IBIIBIIBI")
    assert [k for k, _ in interleave(2, 0)] == ["I", "I"]
    assert sorted(j for k, j in interleave(5, 4) if k == "B") == [0, 1, 2, 3]


def test_mixed_label_session_raises():
    df = _frame([(4, 1), (3, 0)])
    df.loc[1, "label"] = 2
    with pytest.raises(ValueError, match="mixed labels"):
        session_plan(df, "ostrava")


@pytest.mark.parametrize("fold", ["fold_0", "fold_1", "fold_2", "fold_3", "fold_4", "fold_site"])
def test_committed_folds_have_no_mixed_label_ostrava_session(fold):
    from models.common.v3_launch import drop_duplicate_train_rows, inner_probe_manifest
    dups = pd.read_csv(PREP / "exclude_duplicates.csv")
    man = pd.read_csv(PREP / f"{fold}.csv")
    manifests = [drop_duplicate_train_rows(man, dups)]
    if fold == "fold_0":
        manifests.append(inner_probe_manifest(man, seed=0))         # the smoke run's manifest
    for m in manifests:
        train = m[m.split == "train"].reset_index(drop=True)
        image_rows, bags = session_plan(train, "ostrava")           # raises on a mixed session
        assert bags, fold
        assert len(image_rows) + sum(map(len, bags)) == len(train)
        assert (train.source.values[np.concatenate(bags)] == "ostrava").all()


def test_lse_of_a_one_frame_bag_is_its_logits():
    torch.manual_seed(0)
    logits = torch.randn(5, 6)
    labels = torch.tensor([3, 1, 1, 1, 4])
    pooled, y = pool_bags(logits, labels, torch.tensor([0, 1, 1, 1, 2]), tau=1.0)
    assert torch.equal(pooled[0], logits[0]) and torch.equal(pooled[2], logits[4])
    assert torch.allclose(pooled[1], torch.logsumexp(logits[1:4], dim=0))
    assert y.tolist() == [3, 1, 4]
    same, y2 = pool_bags(logits, labels, torch.arange(5), tau=1.0)     # an image batch
    assert torch.equal(same, logits) and torch.equal(y2, labels)


def test_tau_scales_the_pool():
    x = torch.randn(4, 6)
    pooled, _ = pool_bags(x, torch.zeros(4, dtype=torch.long), torch.zeros(4, dtype=torch.long), tau=0.5)
    assert torch.allclose(pooled[0], 0.5 * torch.logsumexp(x / 0.5, dim=0))


def test_bag_loss_on_identical_frames_equals_image_ce():
    from models.common.losses import build_loss
    cfg = load_config(str(BAG_CFG))
    loss_fn, _, _ = build_loss(cfg, [900, 300, 400, 200, 60, 150])    # class-balanced + smoothing
    x = torch.randn(1, 6)
    y = torch.tensor([2])
    pooled, yb = pool_bags(x.repeat(8, 1), y.repeat(8), torch.zeros(8, dtype=torch.long), tau=1.0)
    assert torch.allclose(loss_fn(pooled, yb), loss_fn(x, y), atol=1e-6)


def test_pool_refuses_a_bag_with_two_labels():
    with pytest.raises(ValueError, match="label"):
        pool_bags(torch.randn(3, 6), torch.tensor([1, 1, 2]), torch.tensor([0, 0, 0]), tau=1.0)


def test_bag_settings_read_from_the_config():
    assert bag_settings(load_config(str(BAG_CFG))) == {"source": "ostrava", "k": 8, "tau": 1.0}
    assert bag_settings(load_config(str(ROOT / "configs/rop_staging_dinov2.yaml"))) is None
    with pytest.raises(ValueError, match="true or false"):
        bag_settings(load_config(str(BAG_CFG), overrides=["train.bag_loss.enabled=yes"]))


def test_one_epoch_trains_through_bags_and_spends_the_onecycle_schedule(tmp_path, capsys):
    """build_dataloaders + train_model end to end on tiny cached images, CPU."""
    from PIL import Image
    from torch import nn
    from models.common.train_utils import build_dataloaders, train_model

    df = _frame([(1, 0), (3, 1), (5, 2), (12, 5), (2, 0)], others=9)
    df["split"] = "train"
    val = _frame([(2, 0), (2, 1)], others=4, pid0=900).assign(split="val")
    man = pd.concat([df, val], ignore_index=True)
    rng = np.random.default_rng(0)
    for p in man.image_path:
        f = tmp_path / "cache" / "rop" / Path(p).relative_to("data/rop").with_suffix(".jpg")
        f.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(rng.integers(0, 255, (20, 20, 3), dtype=np.uint8)).save(f)
    cfg = load_config(str(BAG_CFG), overrides=[
        "train.epochs=1", "train.batch_size=4", "train.bag_loss.k=2", "train.num_workers=0",
        "train.amp=false", "preprocess.image_size=14", "preprocess.cached=true",
        f"preprocess.cache_root={tmp_path / 'cache'}"])
    loaders = build_dataloaders(cfg, man)
    # image rows: 1 singleton + 9 others -> 3 batches of 4; bags: 4 sessions -> 2 batches of 2
    assert len(loaders["train"]) == 5

    class Log:
        dir = tmp_path

        def save_weights(self, sd):
            pass

    model = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(3, 6))
    history, _ = train_model(cfg, model, loaders, Log(), torch.device("cpu"))
    out = capsys.readouterr().out
    assert "[bag] epoch 1: 3 image batches" in out and "2 bag batches = 4 sessions / 8 frames" in out
    assert np.isfinite(history["train_loss"][0])
    # steps_per_epoch counted both kinds: the run's lr is where a 5-step OneCycle is after 5 steps
    ref = torch.optim.AdamW([torch.zeros(1, requires_grad=True)])
    sched = torch.optim.lr_scheduler.OneCycleLR(ref, max_lr=float(cfg.train.lr), total_steps=5,
                                                pct_start=0.1, anneal_strategy="cos")
    for _ in range(5):
        ref.step()
        sched.step()
    assert history["lr"][-1] == pytest.approx(sched.get_last_lr()[0], rel=1e-9)
