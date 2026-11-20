from __future__ import annotations

from collections.abc import Callable, Iterable
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
        foreach: bool = False,
    ) -> None:
        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
            "foreach": foreach,
        }
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Callable[[], float] | None = None) -> float | None:
        for group in self.param_groups:
            active = [p for p in group["params"] if p.grad is not None]
            if not active:
                continue
            for parameter in active:
                self._advance(parameter)
            if group["foreach"]:
                self._update_foreach(active, group)
            else:
                for parameter in active:
                    self._update(parameter, group)
        return None

    def _advance(self, parameter: Tensor) -> None:
        state = self.state[parameter]
        if not state:
            state["step"] = 0
            state["exp_avg"] = torch.zeros_like(parameter)
            state["exp_avg_sq"] = torch.zeros_like(parameter)
        state["step"] += 1

    def _update(self, parameter: Tensor, group: dict) -> None:
        beta1, beta2 = group["betas"]
        state = self.state[parameter]
        grad, step = parameter.grad, state["step"]
        state["exp_avg"].mul_(beta1).add_(grad, alpha=1 - beta1)
        state["exp_avg_sq"].mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
        bias_correction1 = 1 - beta1**step
        bias_correction2 = 1 - beta2**step
        denominator = (state["exp_avg_sq"].sqrt() / bias_correction2**0.5).add_(group["eps"])
        parameter.mul_(1 - group["lr"] * group["weight_decay"])
        parameter.addcdiv_(state["exp_avg"], denominator, value=-group["lr"] / bias_correction1)

    def _update_foreach(self, parameters: list[Tensor], group: dict) -> None:
        beta1, beta2 = group["betas"]
        states = [self.state[p] for p in parameters]
        grads = [grad for p in parameters if (grad := p.grad) is not None]
        averages = [s["exp_avg"] for s in states]
        squares = [s["exp_avg_sq"] for s in states]
        steps = [s["step"] for s in states]
        torch._foreach_mul_(averages, beta1)
        torch._foreach_add_(averages, grads, alpha=1 - beta1)
        torch._foreach_mul_(squares, beta2)
        torch._foreach_addcmul_(squares, grads, grads, value=1 - beta2)
        denominators = torch._foreach_sqrt(squares)
        torch._foreach_div_(denominators, [(1 - beta2**step) ** 0.5 for step in steps])
        torch._foreach_add_(denominators, group["eps"])
        torch._foreach_mul_(parameters, 1 - group["lr"] * group["weight_decay"])
        rates = [-group["lr"] / (1 - beta1**step) for step in steps]
        torch._foreach_addcdiv_(parameters, averages, denominators, rates)
