"""Clinically-structured ROP staging head: ordinal stages + AP-ROP branch + site adversary.

Why this exists — each branch answers a MEASURED defect, not a hunch:

  ORDINAL STAGE BRANCH (CORN).  Flat softmax treats Stage 2 -> Normal as the same error as
  Stage 2 -> Stage 3, and Stage 2 recall sat at 0.45-0.50 across all five benchmark
  backbones. Stages are ordered (ICROP), so the head should be too. CORN (Shi, Cao &
  Raschka, 2021) gives rank-consistent predictions from K-1 conditional binary classifiers.

  AP-ROP BRANCH (binary).  ICROP-3 defines AP-ROP as an aggressive FORM, not a point on the
  stage scale — but the 6-way softmax forced it to compete with the stages, and every one of
  the five backbones produced the same signature: precision 0.89-0.97 with recall 0.16-0.22.
  The model only names AP-ROP when overwhelmingly certain because any probability it assigns
  AP-ROP is probability taken from the stage classes. A separate branch decouples the two
  questions ("how far has it progressed" / "is the aggressive form present").

  SITE-ADVERSARIAL BRANCH (DANN, gradient reversal).  The G5 audit measured hospital
  identity linearly decodable from trained embeddings at 0.82 balanced accuracy (0.95
  untrained). The adversary pushes the representation toward site-invariance during
  training, and the SAME G5 probe quantifies before/after — the number the paper reports.

Ordinal targets are MASKED on AP-ROP images: ICROP-3 says AP-ROP cannot be reliably staged,
and our labels carry no stage for them, so the stage branch simply receives no gradient
from those samples (mask, not a fabricated stage).

The 6-class decode used for every headline metric keeps results directly comparable with
the flat-softmax benchmark: predict AP-ROP when its branch fires, else the ordinal rank.
Probabilities compose as p(arop) and (1 - p(arop)) * P(rank = k), which sums to 1.

ABLATION VARIANTS (the paper's component-contribution table, not alternative products):

  "no_ordinal"  softmax over the 5 stage classes instead of CORN, AP-ROP branch kept —
                isolates what the ordinal head contributes.
  "no_branch"   CORN over 6 forced ranks with AP-ROP as rank 5, no binary branch —
                isolates what the branch contributes, and tests ICROP-3's claim that
                forcing the aggressive form onto the stage scale is a modelling error.

Both keep the identical backbone, site adversary, masking rule (where applicable) and
decode-to-6-classes contract, so their numbers drop into the same comparison table.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .architectures import build_model

# Ordinal axis: normal(0) < stage_1(1) < stage_2(2) < stage_3(3) < stage_4_5(4).
NUM_RANKS = 5
AROP_LABEL = 5          # index of arop in the 6-class manifests (class_index.json)


# ── CORN ─────────────────────────────────────────────────────────────────────
def corn_loss(logits: torch.Tensor, ranks: torch.Tensor) -> torch.Tensor:
    """CORN loss over K-1 conditional binary tasks.

    logits: (B, K-1); ranks: (B,) in [0, K-1]. Task k is trained only on samples with
    rank > k-1 (the conditional subset), with binary target rank > k. Summing the per-task
    BCE and dividing by the total number of conditional samples is the paper's estimator.
    """
    total, n = logits.new_zeros(()), 0
    for k in range(logits.shape[1]):
        mask = ranks > (k - 1)          # rank >= k
        if not bool(mask.any()):
            continue
        target = (ranks[mask] > k).float()
        total = total + F.binary_cross_entropy_with_logits(
            logits[mask, k], target, reduction="sum")
        n += int(mask.sum())
    return total / max(n, 1)


def corn_rank_probs(logits: torch.Tensor) -> torch.Tensor:
    """(B, K) probability of each rank from the conditional logits.

    P(y > k) is the cumulative product of the conditional sigmoids, which is monotonically
    decreasing by construction — so the differences below are non-negative and sum to 1.
    """
    exceed = torch.cumprod(torch.sigmoid(logits), dim=1)          # (B, K-1): P(y > k)
    ones = exceed.new_ones(exceed.shape[0], 1)
    upper = torch.cat([ones, exceed], dim=1)                      # P(y > k-1)
    lower = torch.cat([exceed, exceed.new_zeros(exceed.shape[0], 1)], dim=1)  # P(y > k)
    return upper - lower                                          # P(y = k)


def corn_predict(logits: torch.Tensor) -> torch.Tensor:
    """Rank = number of conditional probabilities whose cumulative product exceeds 0.5."""
    return (torch.cumprod(torch.sigmoid(logits), dim=1) > 0.5).sum(dim=1)


# ── gradient reversal ────────────────────────────────────────────────────────
class _GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lam):
        ctx.lam = lam
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad):
        return grad.neg() * ctx.lam, None


class GradientReversal(nn.Module):
    """Identity forward; backward multiplies the gradient by -lam.

    lam is a plain attribute the training loop ramps per epoch (DANN schedule) — starting
    at 0 so the site head first LEARNS to read site before the backbone is asked to fight
    it. Reversing a randomly-initialised adversary just injects noise.
    """

    def __init__(self):
        super().__init__()
        self.lam = 0.0

    def forward(self, x):
        return _GradReverse.apply(x, self.lam)


# ── the model ────────────────────────────────────────────────────────────────
VARIANTS = ("full", "no_ordinal", "no_branch")


class StructuredROPModel(nn.Module):
    """Shared pretrained backbone, three heads. See module docstring for the why."""

    def __init__(self, arch: str, n_sites: int, pretrained: bool = True,
                 dropout: float = 0.3, variant: str = "full"):
        super().__init__()
        if variant not in VARIANTS:
            raise ValueError(f"variant {variant!r} not in {VARIANTS}")
        self.variant = variant
        # Reuse the benchmark's factory, then strip its classifier: the comparison against
        # the five-backbone baseline is only clean if the feature extractor is IDENTICAL.
        net = build_model(arch, num_classes=6, pretrained=pretrained, dropout=dropout)
        for attr in ("classifier", "fc", "head"):
            if hasattr(net, attr):
                setattr(net, attr, nn.Identity())
                break
        else:  # pragma: no cover - every supported arch has one of the three
            raise ValueError(f"cannot strip the classifier from arch '{arch}'")
        self.backbone = net
        with torch.no_grad():
            feat = self.backbone(torch.zeros(1, 3, 224, 224)).flatten(1).shape[1]

        self.dropout = nn.Dropout(dropout)
        if variant == "full":
            self.stage_head = nn.Linear(feat, NUM_RANKS - 1)  # CORN: K-1 conditional logits
            self.arop_head = nn.Linear(feat, 1)
        elif variant == "no_ordinal":
            self.stage_head = nn.Linear(feat, NUM_RANKS)      # plain softmax over stages
            self.arop_head = nn.Linear(feat, 1)
        else:  # no_branch: CORN over all 6 ranks, AP-ROP forced onto the scale as rank 5
            self.stage_head = nn.Linear(feat, NUM_RANKS)      # (6 ranks) - 1 conditionals
            self.arop_head = None
        self.grl = GradientReversal()
        # Small MLP, not a linear probe: a too-weak adversary under-reports the site signal
        # it is supposed to be removing.
        self.site_head = nn.Sequential(
            nn.Linear(feat, 256), nn.ReLU(inplace=True), nn.Dropout(0.2),
            nn.Linear(256, n_sites))

    def forward(self, x):
        z = self.dropout(self.backbone(x).flatten(1))
        out = {
            "stage": self.stage_head(z),                 # (B, K-1) / (B, 5) by variant
            "site": self.site_head(self.grl(z)),         # (B, n_sites)
            "embedding": z,                              # for the G5 probe
        }
        if self.arop_head is not None:
            out["arop"] = self.arop_head(z).squeeze(1)   # (B,)
        return out


# ── composite decode: back to 6 classes for comparable metrics ───────────────
def decode_6class(out: dict, arop_threshold: float = 0.5, variant: str = "full"):
    """(pred6, prob6) so every variant scores on the exact metrics the flat benchmark
    used. Branched variants: p6 = [ (1-pa) * P(stage k) ... , pa ]; no_branch reads the
    6 ranks straight off the CORN cascade."""
    if variant == "no_branch":
        return corn_predict(out["stage"]), corn_rank_probs(out["stage"])   # rank 5 = AP-ROP
    pa = torch.sigmoid(out["arop"])                      # (B,)
    if variant == "no_ordinal":
        rank_p = F.softmax(out["stage"], dim=1)          # (B, K)
        rank = out["stage"].argmax(dim=1)
    else:
        rank_p = corn_rank_probs(out["stage"])           # (B, K)
        rank = corn_predict(out["stage"])
    prob6 = torch.cat([(1 - pa).unsqueeze(1) * rank_p, pa.unsqueeze(1)], dim=1)
    pred6 = torch.where(pa > arop_threshold,
                        torch.full_like(rank, AROP_LABEL), rank)
    return pred6, prob6


def structured_losses(out: dict, label6: torch.Tensor, site: torch.Tensor,
                      arop_pos_weight: torch.Tensor | None = None,
                      site_weight: torch.Tensor | None = None,
                      variant: str = "full"):
    """(stage_loss, arop_loss, site_loss). Stage loss is masked to non-AP-ROP samples —
    except in no_branch, which by design trains AP-ROP as rank 5 on the ordinal scale
    (the mask is the branch's job, and this variant has no branch)."""
    site_loss = F.cross_entropy(out["site"], site, weight=site_weight)
    if variant == "no_branch":
        return corn_loss(out["stage"], label6), out["stage"].sum() * 0.0, site_loss
    is_arop = label6 == AROP_LABEL
    stageable = ~is_arop
    if bool(stageable.any()):
        stage_loss = (F.cross_entropy(out["stage"][stageable], label6[stageable])
                      if variant == "no_ordinal"
                      else corn_loss(out["stage"][stageable], label6[stageable]))
    else:  # a batch of pure AP-ROP — rare but possible under the weighted sampler
        stage_loss = out["stage"].sum() * 0.0
    arop_loss = F.binary_cross_entropy_with_logits(
        out["arop"], is_arop.float(), pos_weight=arop_pos_weight)
    return stage_loss, arop_loss, site_loss


def dann_lambda(progress: float, max_lam: float) -> float:
    """DANN ramp: 0 -> max_lam along training progress in [0, 1] (Ganin & Lempitsky)."""
    import math
    return max_lam * (2.0 / (1.0 + math.exp(-10.0 * progress)) - 1.0)
