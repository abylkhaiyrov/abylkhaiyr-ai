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
