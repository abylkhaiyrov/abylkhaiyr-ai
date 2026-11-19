from collections.abc import Iterable

import torch
from torch import Tensor, nn


def global_grad_norm(parameters: Iterable[nn.Parameter]) -> Tensor:
    norms = [torch.linalg.vector_norm(p.grad.float()) for p in parameters if p.grad is not None]
    if not norms:
        return torch.zeros(())
    return torch.linalg.vector_norm(torch.stack(norms))


def clip_grad_norm(parameters: Iterable[nn.Parameter], max_norm: float) -> Tensor:
    params = [p for p in parameters if p.grad is not None]
    total = global_grad_norm(params)
    if torch.isfinite(total):
        scale = torch.clamp(max_norm / (total + 1e-6), max=1.0)
        for parameter in params:
            parameter.grad.mul_(scale.to(parameter.grad.device))
    return total
