from __future__ import annotations

from typing import Protocol

import torch
from torch import Tensor, nn

from llm.core.config import ContrastiveConfig
from llm.optimization.precision import autocast_context
from llm.training.losses import (
    IGNORE_INDEX,
    causal_lm_loss_sum,
    contrastive_pair_terms,
    dpo_loss,
    sequence_log_probs,
)

Batch = tuple[Tensor, ...]


class Objective(Protocol):
    def weight(self, batch: Batch) -> int: ...
    def loss_sum(self, model: nn.Module, batch: Batch) -> Tensor: ...
    def drain_stats(self) -> dict[str, float]: ...


class CausalLMObjective:
    def __init__(self, device: torch.device, dtype: torch.dtype) -> None:
        self.device = device
        self.dtype = dtype

    def weight(self, batch: Batch) -> int:
        return int((batch[1] != IGNORE_INDEX).sum().item())

    def drain_stats(self) -> dict[str, float]:
        return {}

    def loss_sum(self, model: nn.Module, batch: Batch) -> Tensor:
        inputs, targets, *segments = batch
        with autocast_context(self.device, self.dtype):
            logits = model(inputs, segment_ids=segments[0]) if segments else model(inputs)
        return causal_lm_loss_sum(logits, targets)[0]


class DPOObjective:
    def __init__(
        self, reference: nn.Module, beta: float, device: torch.device, dtype: torch.dtype
    ) -> None:
        self.reference = reference.eval()
        for parameter in self.reference.parameters():
            parameter.requires_grad_(False)
        self.beta = beta
        self.device = device
        self.dtype = dtype
        self._margin_sum = torch.zeros((), device=device)
        self._correct = torch.zeros((), device=device)
        self._pairs = 0

    def drain_stats(self) -> dict[str, float]:
        if self._pairs == 0:
            return {}
        stats = {
            "dpo/reward_margin": self.beta * float(self._margin_sum.item()) / self._pairs,
            "dpo/reward_accuracy": float(self._correct.item()) / self._pairs,
        }
        self._margin_sum.zero_()
        self._correct.zero_()
        self._pairs = 0
        return stats

    def weight(self, batch: Batch) -> int:
        return batch[0].shape[0]

    def _log_probs(self, model: nn.Module, inputs: Tensor, targets: Tensor) -> Tensor:
        with autocast_context(self.device, self.dtype):
            logits = model(inputs)
        return sequence_log_probs(logits, targets)

    def loss_sum(self, model: nn.Module, batch: Batch) -> Tensor:
        chosen_in, chosen_tg, rejected_in, rejected_tg = batch
        with torch.no_grad():
            ref_chosen = self._log_probs(self.reference, chosen_in, chosen_tg)
            ref_rejected = self._log_probs(self.reference, rejected_in, rejected_tg)
        policy_chosen = self._log_probs(model, chosen_in, chosen_tg)
        policy_rejected = self._log_probs(model, rejected_in, rejected_tg)
        mean_loss = dpo_loss(policy_chosen, policy_rejected, ref_chosen, ref_rejected, self.beta)
        margin = ((policy_chosen - policy_rejected) - (ref_chosen - ref_rejected)).detach()
        self._margin_sum += margin.sum().float()
        self._correct += (margin > 0).sum()
        self._pairs += margin.numel()
        return mean_loss * chosen_in.shape[0]


class ContrastiveObjective:
    def __init__(self, device: torch.device, dtype: torch.dtype, config: ContrastiveConfig) -> None:
        self.device = device
        self.dtype = dtype
        self.config = config
        self._margin_sum = torch.zeros((), device=device)
        self._correct = torch.zeros((), device=device)
        self._pairs = torch.zeros((), device=device)

    def weight(self, batch: Batch) -> int:
        return int((batch[1] != IGNORE_INDEX).sum().item())

    def drain_stats(self) -> dict[str, float]:
        pairs = float(self._pairs.item())
        if pairs == 0:
            return {}
        stats = {
            "contrastive/margin": float(self._margin_sum.item()) / pairs,
            "contrastive/accuracy": float(self._correct.item()) / pairs,
        }
        self._margin_sum.zero_()
        self._correct.zero_()
        self._pairs.zero_()
        return stats

    def loss_sum(self, model: nn.Module, batch: Batch) -> Tensor:
        chosen_in, chosen_tg, rejected_in, rejected_tg = batch[:4]
        with autocast_context(self.device, self.dtype):
            chosen_logits = model(chosen_in)
            rejected_logits = model(rejected_in) if self.config.weight > 0 else None
        total = causal_lm_loss_sum(chosen_logits, chosen_tg)[0]
        if rejected_logits is None:
            return total
        terms = contrastive_pair_terms(
            chosen_logits,
            chosen_tg,
            rejected_logits,
            rejected_tg,
            self.config.beta,
            self.config.margin,
        )
        valid = terms.valid
        pair_mean = (terms.loss * valid).sum() / valid.sum().clamp(min=1)
        self._margin_sum += (terms.delta * valid).sum().float()
        self._correct += ((terms.delta > 0) & valid).sum()
        self._pairs += valid.sum()
        return total + self.config.weight * self.weight(batch) * pair_mean
