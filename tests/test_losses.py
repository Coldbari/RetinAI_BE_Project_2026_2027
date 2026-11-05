"""Class-balanced weights when a class has no training rows (PROTOCOL_V2, Amendment 5 pre-run note).

``fold_loso_ropvl`` trains on Ostrava + Shenzhen, which have no stage 4/5 photo. The old function
clamped that zero count to 1, which made stage 4/5 the LARGEST weight; ``CrossEntropyLoss`` applies
the class weights to the label-smoothing term of every sample, so the loss was minimised by
predicting stage 4/5 for every true class. An absent class now gets weight 0, and the present
classes keep exactly the old weights whenever every class is present (every v3 fold, arms A-G).
"""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from pathlib import Path  # noqa: E402

from models.common.config import load_config  # noqa: E402
from models.common.losses import build_loss, class_balanced_weights  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
I_CFG = ROOT / "configs" / "rop_staging_dinov2_v4recipe.yaml"   # arm I: smoothing 0.10, beta 0.999
# fold_loso_ropvl's training counts (normal, stage 1-3, stage 4/5, AP-ROP): no stage 4/5 row.
LOSO_ROPVL = [486, 261, 349, 292, 0, 20]


def _old_class_balanced_weights(class_counts, beta=0.9999):
    """The function before the fix, verbatim: a zero count is clamped to 1."""
    counts = np.asarray(class_counts, dtype=np.float64)
    counts = np.maximum(counts, 1.0)
    eff = 1.0 - np.power(beta, counts)
    w = (1.0 - beta) / eff
    w = w / w.sum() * len(counts)
    return torch.tensor(w, dtype=torch.float32)


def _fitted_probs(loss_fn, targets, steps=400):
    """The softmax of free logits fitted to ``targets`` by Adam: what the loss asks the model for."""
    logits = torch.zeros(len(targets), 6, requires_grad=True)
    opt = torch.optim.Adam([logits], lr=0.1)
    for _ in range(steps):
        opt.zero_grad()
        loss_fn(logits, targets).backward()
        opt.step()
    return logits.detach().softmax(1)


@pytest.mark.parametrize("beta", [0.999, 0.9999, 0.99])
@pytest.mark.parametrize("counts", [[900, 300, 400, 200, 60, 150],     # test_v3_bag_loss's counts
                                    [3000, 700, 900, 125, 40, 260],
                                    [1, 1, 1, 1, 1, 1], [5, 1, 2, 3, 1, 1]])
def test_with_every_class_present_the_weights_are_the_old_ones(counts, beta):
    new, old = class_balanced_weights(counts, beta), _old_class_balanced_weights(counts, beta)
    assert new.dtype == old.dtype == torch.float32
    assert torch.equal(new, old)                                       # bit for bit


def test_an_absent_class_gets_zero_and_the_present_ones_are_normalised_among_themselves():
    for beta in (0.999, 0.9999):
        w = class_balanced_weights(LOSO_ROPVL, beta)
        assert w[4] == 0.0 and (w[[0, 1, 2, 3, 5]] > 0).all()
        present = [c for c in LOSO_ROPVL if c > 0]
        assert torch.equal(w[[0, 1, 2, 3, 5]], _old_class_balanced_weights(present, beta))
        assert w.sum().item() == pytest.approx(5.0, rel=1e-6)          # = the number present
        # the old function made the absent class the largest weight
        assert int(_old_class_balanced_weights(LOSO_ROPVL, beta).argmax()) == 4
    with pytest.raises(ValueError, match="no class"):
        class_balanced_weights([0] * 6)


@pytest.mark.parametrize("eps", [0.05, 0.10])
def test_on_fold_loso_ropvl_the_loss_asks_for_the_true_class(eps):
    targets = torch.tensor([0, 1, 2, 3])
    fixed = torch.nn.CrossEntropyLoss(weight=class_balanced_weights(LOSO_ROPVL, 0.999),
                                      label_smoothing=eps)
    p = _fitted_probs(fixed, targets)
    assert p.argmax(1).tolist() == targets.tolist()
    assert (p[:, 4] < 0.01).all()                       # zero weight: nothing pulls toward 4/5
    # The reviewer's finding, reproduced with the old weights: stage 4/5 for every true class.
    broken = torch.nn.CrossEntropyLoss(weight=_old_class_balanced_weights(LOSO_ROPVL, 0.999),
                                       label_smoothing=eps)
    q = _fitted_probs(broken, targets)
    assert q.argmax(1).tolist() == [4, 4, 4, 4] and (q[:, 4] > 0.6).all()


def test_build_loss_on_fold_loso_ropvl_counts():
    cfg = load_config(I_CFG)
    assert cfg.train.label_smoothing == 0.10 and cfg.train.class_balanced is True
    loss_fn, decode, needs_float = build_loss(cfg, LOSO_ROPVL)
    assert isinstance(loss_fn, torch.nn.CrossEntropyLoss) and not needs_float
    assert loss_fn.label_smoothing == 0.10
    assert loss_fn.weight[4] == 0.0
    assert torch.equal(loss_fn.weight, class_balanced_weights(LOSO_ROPVL, 0.999))
    # a training batch of that fold never holds class 4; its loss is finite and ...
    x, y = torch.randn(8, 6), torch.tensor([0, 1, 2, 3, 5, 0, 1, 2])
    assert torch.isfinite(loss_fn(x, y))
    p = _fitted_probs(loss_fn, torch.tensor([0, 1, 2, 3]))
    assert p.argmax(1).tolist() == [0, 1, 2, 3]          # ... minimised at the true class
