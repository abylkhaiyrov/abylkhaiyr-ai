from __future__ import annotations

import math
import re
from collections.abc import Callable, Hashable, Sequence

import torch
from torch import Tensor

from llm.training.losses import IGNORE_INDEX, causal_lm_loss_sum

NGRAM_SIZE = 3
WHITESPACE = re.compile(r"\s+")


def pass_at_k(samples: int, correct: int, k: int) -> float:
    if samples - correct < k:
        return 1.0
    return 1.0 - math.prod(1.0 - k / (samples - i) for i in range(correct))


def normalize_answer(text: str) -> str:
    return WHITESPACE.sub(" ", text.strip().lower()).strip(" .")


def exact_match(prediction: str, reference: str) -> bool:
    return normalize_answer(prediction) == normalize_answer(reference)


def repeated_ngram_ratio(token_ids: Sequence[Hashable], size: int = NGRAM_SIZE) -> float:
    grams = [tuple(token_ids[i : i + size]) for i in range(len(token_ids) - size + 1)]
    return 0.0 if not grams else 1.0 - len(set(grams)) / len(grams)


def batch_loss_and_accuracy(logits: Tensor, targets: Tensor) -> tuple[float, int, int]:
    total, count = causal_lm_loss_sum(logits, targets)
    mask = targets != IGNORE_INDEX
    correct = int(((logits.argmax(-1) == targets) & mask).sum().item())
    return float(total.item()), count, correct


def perplexity_from(total_loss: float, count: int) -> float:
    return math.exp(total_loss / max(count, 1))


@torch.inference_mode()
def score_token_stream(
    model: torch.nn.Module,
    windows: list[Tensor],
    device: torch.device,
    on_window: Callable[[int], None] | None = None,
) -> tuple[float, int, int]:
    total, count, correct = 0.0, 0, 0
    for window in windows:
        batch = window.unsqueeze(0).to(device)
        loss, tokens, hits = batch_loss_and_accuracy(model(batch[:, :-1]), batch[:, 1:])
        total, count, correct = total + loss, count + tokens, correct + hits
        if on_window is not None:
            on_window(1)
    return total, count, correct
