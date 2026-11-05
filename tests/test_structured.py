"""Unit tests for the structured ROP head — the maths most likely to be silently wrong.

None of these need a GPU or images; they pin the CORN estimator, the ordinal mask, the
gradient-reversal sign, and the 6-class probability composition. A mistake in any of them
would not crash training — it would train something subtly different and report a
plausible number, which is the failure mode this project keeps finding the hard way.
"""
import math

import pytest

torch = pytest.importorskip("torch")

from models.common.structured import (AROP_LABEL, NUM_RANKS, GradientReversal,
                                      StructuredROPModel, corn_loss, corn_predict,
                                      corn_rank_probs, dann_lambda, decode_6class,
                                      structured_losses)


def test_corn_rank_probs_sum_to_one_and_are_nonnegative():
    logits = torch.randn(64, NUM_RANKS - 1) * 3
    p = corn_rank_probs(logits)
    assert p.shape == (64, NUM_RANKS)
    assert torch.all(p >= -1e-6)
    assert torch.allclose(p.sum(dim=1), torch.ones(64), atol=1e-5)


def test_corn_predict_is_rank_consistent_with_probs():
    # The argmax-free CORN decision (count of cumulative probs > 0.5) must agree with the
    # highest-mass rank in the easy, confident cases.
    strong = torch.tensor([[8.0, 8.0, -8.0, -8.0]])   # certain: y > 0, y > 1, not y > 2
    assert corn_predict(strong).item() == 2
    assert corn_rank_probs(strong).argmax(dim=1).item() == 2


def test_corn_loss_drives_ordering():
    # One gradient step on a rank-3 sample must raise the early conditional logits
    # (P(y>0), P(y>1), P(y>2) up) and lower the last (P(y>3) down).
    logits = torch.zeros(1, NUM_RANKS - 1, requires_grad=True)
    corn_loss(logits, torch.tensor([3])).backward()
    g = logits.grad[0]
    assert torch.all(g[:3] < 0), "gradient should push the first three logits up"
    assert g[3] > 0, "gradient should push the last logit down"


def test_corn_loss_conditional_masking():
    # Task k trains only on samples with rank > k-1: a batch of pure rank-0 samples
    # contributes ONLY to task 0, so the loss must not depend on the later logits.
    a = torch.tensor([[0.0, 5.0, -5.0, 2.0]], requires_grad=True)
    b = torch.tensor([[0.0, -3.0, 9.0, -1.0]], requires_grad=True)
    ranks = torch.tensor([0])
    assert torch.allclose(corn_loss(a, ranks), corn_loss(b, ranks))


def test_structured_losses_mask_arop_out_of_stage_loss():
    out = {"stage": torch.randn(4, NUM_RANKS - 1, requires_grad=True),
           "arop": torch.randn(4), "site": torch.randn(4, 3)}
    # Two AP-ROP samples, two staged. The stage loss must equal the loss computed on the
    # staged pair alone — AP-ROP images have no stage and must contribute no stage gradient.
    label = torch.tensor([AROP_LABEL, 2, AROP_LABEL, 0])
    site = torch.tensor([0, 1, 2, 1])
    ls, la, lsite = structured_losses(out, label, site)
    expected = corn_loss(out["stage"][[1, 3]], label[[1, 3]])
    assert torch.allclose(ls, expected)
    assert la.item() > 0 and lsite.item() > 0


def test_gradient_reversal_flips_and_scales():
    grl = GradientReversal()
    grl.lam = 0.7
    x = torch.ones(3, requires_grad=True)
    grl(x).sum().backward()
    assert torch.allclose(x.grad, torch.full((3,), -0.7))


def test_dann_lambda_ramp():
    assert dann_lambda(0.0, 1.0) == pytest.approx(0.0)
    assert dann_lambda(1.0, 1.0) == pytest.approx(1.0, abs=1e-4)
    assert 0.4 < dann_lambda(0.5, 1.0) < 1.0     # sigmoid ramp, already high at midpoint
    assert dann_lambda(1.0, 0.5) == pytest.approx(0.5, abs=1e-4)


def test_decode_6class_composition():
    B = 8
    out = {"stage": torch.randn(B, NUM_RANKS - 1),
           "arop": torch.randn(B), "site": torch.randn(B, 3)}
    pred, prob = decode_6class(out, arop_threshold=0.5)
    assert prob.shape == (B, 6)
    assert torch.allclose(prob.sum(dim=1), torch.ones(B), atol=1e-5)
    # Wherever the AP-ROP branch fires, the composite prediction must be AP-ROP; the last
    # probability column must be exactly the branch's sigmoid.
    pa = torch.sigmoid(out["arop"])
    assert torch.all(pred[pa > 0.5] == AROP_LABEL)
    assert torch.all(pred[pa <= 0.5] < NUM_RANKS)
    assert torch.allclose(prob[:, AROP_LABEL], pa, atol=1e-6)


def test_structured_model_forward_shapes():
    m = StructuredROPModel("efficientnet_b0", n_sites=3, pretrained=False)
    out = m(torch.zeros(2, 3, 224, 224))
    assert out["stage"].shape == (2, NUM_RANKS - 1)
    assert out["arop"].shape == (2,)
    assert out["site"].shape == (2, 3)
    assert out["embedding"].shape[0] == 2


# ── ablation variants ────────────────────────────────────────────────────────
def test_no_ordinal_forward_and_decode():
    m = StructuredROPModel("efficientnet_b0", n_sites=3, pretrained=False,
                           variant="no_ordinal")
    out = m(torch.zeros(2, 3, 224, 224))
    assert out["stage"].shape == (2, NUM_RANKS)      # softmax logits over the 5 stages
    assert out["arop"].shape == (2,)
    pred, prob = decode_6class(out, variant="no_ordinal")
    assert prob.shape == (2, 6)
    assert torch.allclose(prob.sum(dim=1), torch.ones(2), atol=1e-5)
    pa = torch.sigmoid(out["arop"])
    assert torch.allclose(prob[:, AROP_LABEL], pa, atol=1e-6)
    assert torch.all(pred[pa > 0.5] == AROP_LABEL)
    # Where the branch does not fire, the prediction is the softmax argmax.
    quiet = pa <= 0.5
    assert torch.all(pred[quiet] == out["stage"].argmax(dim=1)[quiet])


def test_no_ordinal_stage_loss_is_masked_cross_entropy():
    out = {"stage": torch.randn(4, NUM_RANKS, requires_grad=True),
           "arop": torch.randn(4), "site": torch.randn(4, 3)}
    label = torch.tensor([AROP_LABEL, 2, AROP_LABEL, 0])
    site = torch.tensor([0, 1, 2, 1])
    ls, la, lsite = structured_losses(out, label, site, variant="no_ordinal")
    expected = torch.nn.functional.cross_entropy(out["stage"][[1, 3]], label[[1, 3]])
    assert torch.allclose(ls, expected)
    assert la.item() > 0 and lsite.item() > 0


def test_no_branch_forward_decode_and_loss():
    m = StructuredROPModel("efficientnet_b0", n_sites=3, pretrained=False,
                           variant="no_branch")
    out = m(torch.zeros(2, 3, 224, 224))
    assert "arop" not in out                          # the branch does not exist
    assert out["stage"].shape == (2, NUM_RANKS)       # (6 ranks) - 1 conditional logits
    pred, prob = decode_6class(out, variant="no_branch")
    assert prob.shape == (2, 6)
    assert torch.allclose(prob.sum(dim=1), torch.ones(2), atol=1e-5)
    assert torch.all((pred >= 0) & (pred <= AROP_LABEL))
    # A confident full cascade must decode to AP-ROP = rank 5 on the forced scale.
    sure = {"stage": torch.full((1, NUM_RANKS), 8.0)}
    assert decode_6class(sure, variant="no_branch")[0].item() == AROP_LABEL


def test_no_branch_trains_arop_as_rank_5_unmasked():
    # The whole point of this arm: AP-ROP samples DO flow through the ordinal loss.
    out = {"stage": torch.randn(4, NUM_RANKS, requires_grad=True),
           "site": torch.randn(4, 3)}
    label = torch.tensor([AROP_LABEL, 2, AROP_LABEL, 0])
    site = torch.tensor([0, 1, 2, 1])
    ls, la, lsite = structured_losses(out, label, site, variant="no_branch")
    assert torch.allclose(ls, corn_loss(out["stage"], label))   # all four samples, no mask
    assert la.item() == 0.0                                     # no branch, no branch loss
    assert lsite.item() > 0


def test_variant_rejects_unknown_name():
    with pytest.raises(ValueError):
        StructuredROPModel("efficientnet_b0", n_sites=3, pretrained=False,
                           variant="no_such_arm")
