from __future__ import annotations

from torch import Tensor, nn

from llm.core.config import ModelConfig
from llm.model.attention import CausalSelfAttention
from llm.model.cache import LayerKVCache
from llm.model.mlp import SwiGLUFeedForward
from llm.model.norm import RMSNorm
from llm.model.rope import RotaryEmbedding


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig, rotary: RotaryEmbedding, use_reference: bool = False):
        super().__init__()
        self.attention_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.attention = CausalSelfAttention(config, rotary, use_reference)
        self.ffn_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.feed_forward = SwiGLUFeedForward(config)
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        x: Tensor,
        position_ids: Tensor,
        mask: Tensor | None,
        cache: LayerKVCache | None = None,
        is_causal: bool = False,
    ) -> Tensor:
        attended = self.attention(self.attention_norm(x), position_ids, mask, cache, is_causal)
        x = x + self.dropout(attended)
        return x + self.feed_forward(self.ffn_norm(x))
