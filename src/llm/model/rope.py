import torch
from torch import Tensor, nn

ROPE_DIM_STEP = 2


def rotate_half(x: Tensor) -> Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, max_positions: int, theta: float) -> None:
        super().__init__()
        exponents = torch.arange(0, head_dim, ROPE_DIM_STEP, dtype=torch.float32) / head_dim
        inverse_frequencies = 1.0 / (theta**exponents)
        angles = torch.outer(torch.arange(max_positions, dtype=torch.float32), inverse_frequencies)
        embedding = torch.cat((angles, angles), dim=-1)
        self.register_buffer("cos", embedding.cos(), persistent=False)
        self.register_buffer("sin", embedding.sin(), persistent=False)

    def forward(self, q: Tensor, k: Tensor, position_ids: Tensor) -> tuple[Tensor, Tensor]:
        cos = self.cos[position_ids].unsqueeze(1).to(q.dtype)
        sin = self.sin[position_ids].unsqueeze(1).to(q.dtype)
        return q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin
