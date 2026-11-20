from __future__ import annotations

from typing import Protocol

import torch
from torch import Tensor, nn

from llm.optimization.precision import autocast_context
from llm.training.losses import (
    IGNORE_INDEX,
    causal_lm_loss_sum,
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
        inputs, targets = batch
        with autocast_context(self.device, self.dtype):
            logits = model(inputs)
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
        self._margin_sum = 0.0
        self._correct = 0
        self._pairs = 0

    def drain_stats(self) -> dict[str, float]:
        if self._pairs == 0:
            return {}
        stats = {
            "dpo/reward_margin": self.beta * self._margin_sum / self._pairs,
            "dpo/reward_accuracy": self._correct / self._pairs,
        }
        self._margin_sum, self._correct, self._pairs = 0.0, 0, 0
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
        self._margin_sum += float(margin.sum().item())
        self._correct += int((margin > 0).sum().item())
        self._pairs += int(margin.numel())
        return mean_loss * chosen_in.shape[0]
