import torch
from torch import Tensor, nn


def rms_norm(x: Tensor, weight: Tensor, eps: float) -> Tensor:
    x32 = x.float()
    scale = torch.rsqrt(x32.pow(2).mean(dim=-1, keepdim=True) + eps)
    return (x32 * scale).to(x.dtype) * weight


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def forward(self, x: Tensor) -> Tensor:
        return rms_norm(x, self.weight, self.eps)
