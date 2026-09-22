"""RLCD (reinforcement learning against proper scoring rules) and its multi-label form.

`rlcd_loss` is the training step of `notebooks/laya_finetune_typed_decisions_2xT4_kaggle.ipynb`
as a function: a Gaussian policy over the option logits, G sampled reports per question scored
with `laya.common.proper_reward`, a group-relative (GRPO-style) advantage, and soft
cross-entropy guidance.

The multi-label form changes what a question is, not the algorithm. Marker 0 of a sequence is a
threshold option; label i is the two-way decision between the threshold and its own marker,

    P(label i) = softmax([z_threshold, z_i])[1] = sigmoid(z_i - z_threshold)

which is precisely a `noul` question with logits [false, true] = [z_threshold, z_i]. One
sequence with K labels is therefore K noul questions answered in the same forward pass, and
each gets its own reward, its own baseline and its own policy-gradient term.
"""
from typing import Dict, Optional, Tuple

import torch

from ..common import QTYPES, proper_reward


def rlcd_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    qtype: torch.Tensor,
    mask: torch.Tensor,
    sigma: float,
    group_size: int = 4,
    w_sph: float = 0.75,
    w_rps: float = 1.0,
    rl_weight: float = 1.0,
    ce_weight: float = 1.0,
    weight: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """logits/target/mask: [N, K]; qtype: [N]; weight: optional per-question weight [N]."""
    k = mask.sum(-1, keepdim=True).float()
    w = torch.ones_like(logits[:, 0]) if weight is None else weight
    w = w / w.sum().clamp_min(1e-9)

    # 1. Sample G noisy reports around the current logits. The noise is projected to zero mean
    #    because softmax cannot see a common shift, so that direction carries no signal.
    eps = torch.randn((group_size,) + logits.shape, device=logits.device) * sigma * mask
    eps = (eps - eps.sum(-1, keepdim=True) / k) * mask
    z = logits.detach().unsqueeze(0) + eps
    q = torch.softmax(z.masked_fill(~mask, -1e4), -1)

    # 2. Strictly proper scoring-rule reward, baselined within each question's group
    with torch.no_grad():
        r = proper_reward(q, target.unsqueeze(0), qtype, mask, w_sph=w_sph, w_rps=w_rps)
        adv = r - r.mean(0, keepdim=True)
        adv = adv / (adv.std() + 1e-6)

    # 3. Policy gradient through the Gaussian log-density, plus soft cross-entropy guidance
    logp = -(((z - logits.unsqueeze(0)) ** 2) * mask).sum(-1) / (2 * sigma ** 2)
    loss_rl = -((adv * logp).mean(0) * w).sum()
    ce = -(target * torch.log_softmax(logits.masked_fill(~mask, -1e4), -1)).sum(-1)
    loss_ce = (ce * w).sum()
    loss = rl_weight * loss_rl + ce_weight * loss_ce
    return loss, {"loss_rl": loss_rl.item(), "loss_ce": loss_ce.item(), "reward": r.mean().item()}


def label_logits(logits: torch.Tensor) -> torch.Tensor:
    """Per-label decision logit: label marker minus the threshold marker. [N, 1+K] -> [N, K]."""
    return logits[:, 1:] - logits[:, :1]


def multilabel_pairs(logits: torch.Tensor, marker_mask: torch.Tensor, target: torch.Tensor):
    """Unroll [N, 1+K] marker logits into one [threshold, label] noul question per valid label."""
    lmask = marker_mask[:, 1:]
    z_lab = logits[:, 1:]
    pair = torch.stack([logits[:, :1].expand_as(z_lab), z_lab], -1)[lmask]
    y = target[:, 1:][lmask]
    return pair, torch.stack([1.0 - y, y], -1)


def multilabel_rlcd_loss(
    logits: torch.Tensor,
    marker_mask: torch.Tensor,
    target: torch.Tensor,
    sigma: float,
    group_size: int = 4,
    w_sph: float = 0.75,
    rl_weight: float = 1.0,
    ce_weight: float = 1.0,
    pos_weight: float = 1.0,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """RLCD over every (sequence, label) decision of a batch.

    `target` is aligned with the markers (column 0, the threshold, is ignored) and may be soft.
    `pos_weight` > 1 up-weights decisions whose label is present, for sparse label sets.
    """
    pair, tpair = multilabel_pairs(logits, marker_mask, target)
    qtype = torch.full((pair.size(0),), QTYPES["noul"], device=pair.device, dtype=torch.long)
    mask = torch.ones_like(pair, dtype=torch.bool)
    weight = 1.0 + (pos_weight - 1.0) * tpair[:, 1] if pos_weight != 1.0 else None
    return rlcd_loss(pair, tpair, qtype, mask, sigma, group_size, w_sph=w_sph,
                     rl_weight=rl_weight, ce_weight=ce_weight, weight=weight)
