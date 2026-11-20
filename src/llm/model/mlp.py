from torch import Tensor, nn
from torch.nn import functional

from llm.core.config import ModelConfig


def silu(x: Tensor) -> Tensor:
    return functional.silu(x)


def swiglu(gate: Tensor, value: Tensor) -> Tensor:
    return silu(gate) * value


class SwiGLUFeedForward(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: Tensor) -> Tensor:
        hidden = swiglu(self.gate_proj(x), self.up_proj(x))
        return self.dropout(self.down_proj(hidden))
