from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch
from torch import Tensor
from torch.optim import Optimizer


class AdamW(Optimizer):
    def __init__(
        self,
        params: Iterable[Any],
        lr: float,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
    ) -> None:
        defaults = {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay}
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Any = None) -> None:
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                self._update(parameter, parameter.grad, group, beta1, beta2)

    def _update(self, parameter: Tensor, grad: Tensor, group: dict, beta1: float, beta2: float):
        state = self.state[parameter]
        if not state:
            state["step"] = 0
            state["exp_avg"] = torch.zeros_like(parameter)
            state["exp_avg_sq"] = torch.zeros_like(parameter)
        state["step"] += 1
        step = state["step"]
        state["exp_avg"].mul_(beta1).add_(grad, alpha=1 - beta1)
        state["exp_avg_sq"].mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
        bias_correction1 = 1 - beta1**step
        bias_correction2 = 1 - beta2**step
        denominator = (state["exp_avg_sq"].sqrt() / bias_correction2**0.5).add_(group["eps"])
        parameter.mul_(1 - group["lr"] * group["weight_decay"])
        parameter.addcdiv_(state["exp_avg"], denominator, value=-group["lr"] / bias_correction1)
