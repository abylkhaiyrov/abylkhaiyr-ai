import math


def learning_rate_at(
    step: int, max_steps: int, base_lr: float, warmup_steps: int, min_lr_ratio: float
) -> float:
    if warmup_steps > 0 and step < warmup_steps:
        return base_lr * (step + 1) / warmup_steps
    decay_span = max(max_steps - warmup_steps, 1)
    progress = min(max((step - warmup_steps) / decay_span, 0.0), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    min_lr = base_lr * min_lr_ratio
    return min_lr + (base_lr - min_lr) * cosine
