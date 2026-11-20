from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import Tensor

MIN_TEMPERATURE = 1e-5


@dataclass
class SamplingParams:
    max_new_tokens: int = 128
    temperature: float = 1.0
    top_k: int = 0
    top_p: float = 1.0
    repetition_penalty: float = 1.0
    penalize_prompt: bool = False
    greedy: bool = False
    beam_width: int = 1
    length_penalty: float = 1.0
    stop_sequences: list[str] = field(default_factory=list)
    use_cache: bool = True
    seed: int | None = None


def apply_repetition_penalty(logits: Tensor, history: Tensor, penalty: float) -> Tensor:
    if penalty == 1.0 or history.numel() == 0:
        return logits
    scores = logits.gather(-1, history)
    adjusted = torch.where(scores > 0, scores / penalty, scores * penalty)
    return logits.scatter(-1, history, adjusted)


def top_k_filter(logits: Tensor, k: int) -> Tensor:
    if k <= 0 or k >= logits.shape[-1]:
        return logits
    threshold = torch.topk(logits, k, dim=-1).values[..., -1:]
    return logits.masked_fill(logits < threshold, float("-inf"))


def top_p_filter(logits: Tensor, p: float) -> Tensor:
    if p >= 1.0:
        return logits
    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    probabilities = torch.softmax(sorted_logits, dim=-1)
    remove_sorted = probabilities.cumsum(dim=-1) - probabilities >= p
    remove = torch.zeros_like(remove_sorted).scatter(-1, sorted_indices, remove_sorted)
    return logits.masked_fill(remove, float("-inf"))


def select_token(
    logits: Tensor, params: SamplingParams, generator: torch.Generator | None
) -> Tensor:
    if params.greedy or params.temperature <= MIN_TEMPERATURE:
        return logits.argmax(dim=-1)
    scaled = logits / params.temperature
    filtered = top_p_filter(top_k_filter(scaled, params.top_k), params.top_p)
    probabilities = torch.softmax(filtered, dim=-1)
    return torch.multinomial(probabilities, 1, generator=generator).squeeze(-1)
