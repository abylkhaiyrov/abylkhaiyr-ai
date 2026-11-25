from typing import NamedTuple

import torch
from torch import Tensor
from torch.nn import functional

IGNORE_INDEX = -100


def compute_dtype(tensor: Tensor) -> torch.dtype:
    return torch.promote_types(tensor.dtype, torch.float32)


def causal_lm_loss_sum(logits: Tensor, targets: Tensor) -> tuple[Tensor, int]:
    flat_logits = logits.reshape(-1, logits.shape[-1]).to(compute_dtype(logits))
    flat_targets = targets.reshape(-1)
    total = functional.cross_entropy(
        flat_logits, flat_targets, ignore_index=IGNORE_INDEX, reduction="sum"
    )
    return total, int((flat_targets != IGNORE_INDEX).sum().item())


def sequence_log_probs(logits: Tensor, targets: Tensor) -> Tensor:
    log_probs = torch.log_softmax(logits.to(compute_dtype(logits)), dim=-1)
    mask = targets != IGNORE_INDEX
    gathered = log_probs.gather(-1, targets.clamp(min=0).unsqueeze(-1)).squeeze(-1)
    return gathered.masked_fill(~mask, 0).sum(dim=-1)


def dpo_loss(
    policy_chosen: Tensor,
    policy_rejected: Tensor,
    reference_chosen: Tensor,
    reference_rejected: Tensor,
    beta: float,
) -> Tensor:
    margin = (policy_chosen - policy_rejected) - (reference_chosen - reference_rejected)
    return -functional.logsigmoid(beta * margin).mean()


class PairTerms(NamedTuple):
    loss: Tensor
    valid: Tensor
    delta: Tensor


def divergent_masks(chosen_targets: Tensor, rejected_targets: Tensor) -> tuple[Tensor, Tensor]:
    width = min(chosen_targets.shape[1], rejected_targets.shape[1])
    same = chosen_targets[:, :width] == rejected_targets[:, :width]
    prefix = torch.cumprod(same.to(torch.int64), dim=1).bool()
    chosen_mask = chosen_targets != IGNORE_INDEX
    rejected_mask = rejected_targets != IGNORE_INDEX
    chosen_mask[:, :width] &= ~prefix
    rejected_mask[:, :width] &= ~prefix
    return chosen_mask, rejected_mask


def masked_mean_log_prob(logits: Tensor, targets: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
    log_probs = torch.log_softmax(logits.to(compute_dtype(logits)), dim=-1)
    gathered = log_probs.gather(-1, targets.clamp(min=0).unsqueeze(-1)).squeeze(-1)
    counts = mask.sum(dim=-1)
    totals = gathered.masked_fill(~mask, 0).sum(dim=-1)
    return totals / counts.clamp(min=1), counts


def contrastive_pair_terms(
    chosen_logits: Tensor,
    chosen_targets: Tensor,
    rejected_logits: Tensor,
    rejected_targets: Tensor,
    beta: float,
    margin: float,
) -> PairTerms:
    chosen_mask, rejected_mask = divergent_masks(chosen_targets, rejected_targets)
    chosen_mean, chosen_count = masked_mean_log_prob(chosen_logits, chosen_targets, chosen_mask)
    rejected_mean, rejected_count = masked_mean_log_prob(
        rejected_logits, rejected_targets, rejected_mask
    )
    delta = chosen_mean - rejected_mean
    valid = (chosen_count > 0) & (rejected_count > 0)
    return PairTerms(-functional.logsigmoid(beta * (delta - margin)), valid, delta.detach())
