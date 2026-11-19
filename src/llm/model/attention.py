from __future__ import annotations

import math

import torch
import torch.nn.functional as functional
from torch import Tensor, nn

from llm.core.config import ModelConfig
from llm.model.cache import LayerKVCache
from llm.model.rope import RotaryEmbedding


def stable_softmax(scores: Tensor, dim: int = -1) -> Tensor:
    shifted = scores - scores.amax(dim=dim, keepdim=True)
    exponentials = shifted.exp()
    return exponentials / exponentials.sum(dim=dim, keepdim=True)


def build_attention_mask(
    query_length: int,
    key_length: int,
    device: torch.device,
    key_padding_mask: Tensor | None = None,
) -> Tensor:
    offset = key_length - query_length
    query_positions = torch.arange(query_length, device=device).unsqueeze(1) + offset
    key_positions = torch.arange(key_length, device=device).unsqueeze(0)
    mask = (key_positions <= query_positions)[None, None]
    if key_padding_mask is not None:
        mask = mask & key_padding_mask[:, None, None, :].bool()
    return mask


def reference_attention(q: Tensor, k: Tensor, v: Tensor, mask: Tensor | None) -> Tensor:
    compute = torch.promote_types(q.dtype, torch.float32)
    scores = (q.to(compute) @ k.to(compute).transpose(-2, -1)) / math.sqrt(q.shape[-1])
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))
    return (stable_softmax(scores) @ v.to(compute)).to(q.dtype)


def sdpa_attention(
    q: Tensor, k: Tensor, v: Tensor, mask: Tensor | None, dropout_p: float = 0.0
) -> Tensor:
    return functional.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=dropout_p)


def repeat_kv(x: Tensor, groups: int) -> Tensor:
    if groups == 1:
        return x
    batch, heads, length, dim = x.shape
    return x[:, :, None].expand(batch, heads, groups, length, dim).reshape(
        batch, heads * groups, length, dim
    )


class CausalSelfAttention(nn.Module):
    def __init__(self, config: ModelConfig, rotary: RotaryEmbedding, use_reference: bool = False):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.groups = config.num_kv_groups
        self.dropout = config.dropout
        self.use_reference = use_reference
        self.rotary = rotary
        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=False)

    def _split(self, x: Tensor, heads: int) -> Tensor:
        batch, length, _ = x.shape
        return x.view(batch, length, heads, self.head_dim).transpose(1, 2)

    def forward(
        self,
        x: Tensor,
        position_ids: Tensor,
        mask: Tensor | None,
        cache: LayerKVCache | None = None,
    ) -> Tensor:
        batch, length, _ = x.shape
        q = self._split(self.q_proj(x), self.num_heads)
        k = self._split(self.k_proj(x), self.num_kv_heads)
        v = self._split(self.v_proj(x), self.num_kv_heads)
        q, k = self.rotary(q, k, position_ids)
        if cache is not None:
            k, v = cache.update(k, v)
        k, v = repeat_kv(k, self.groups), repeat_kv(v, self.groups)
        if self.use_reference:
            context = reference_attention(q, k, v, mask)
        else:
            context = sdpa_attention(q, k, v, mask, self.dropout if self.training else 0.0)
        return self.o_proj(context.transpose(1, 2).reshape(batch, length, -1))
