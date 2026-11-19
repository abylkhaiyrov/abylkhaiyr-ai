from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from llm.core.config import ModelConfig
from llm.model.attention import build_attention_mask
from llm.model.block import TransformerBlock
from llm.model.cache import KVCache
from llm.model.norm import RMSNorm
from llm.model.rope import RotaryEmbedding

RESIDUAL_SCALE_EXEMPT = ("o_proj.weight", "down_proj.weight")


class CausalLM(nn.Module):
    def __init__(self, config: ModelConfig, use_reference_attention: bool = False) -> None:
        super().__init__()
        self.config = config
        self.activation_checkpointing = False
        self.embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        rotary = RotaryEmbedding(config.head_dim, config.max_position_embeddings, config.rope_theta)
        self.blocks = nn.ModuleList(
            TransformerBlock(config, rotary, use_reference_attention)
            for _ in range(config.num_hidden_layers)
        )
        self.final_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.embedding.weight
        self.apply(self._init_weights)
        self._scale_residual_projections()

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)

    def _scale_residual_projections(self) -> None:
        std = self.config.initializer_range / (2 * self.config.num_hidden_layers) ** 0.5
        for name, parameter in self.named_parameters():
            if name.endswith(RESIDUAL_SCALE_EXEMPT):
                nn.init.normal_(parameter, mean=0.0, std=std)

    def num_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def new_cache(self) -> KVCache:
        return KVCache(self.config.num_hidden_layers)

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
        position_ids: Tensor | None = None,
        cache: KVCache | None = None,
    ) -> Tensor:
        batch, length = input_ids.shape
        past = cache.length if cache is not None else 0
        if past + length > self.config.max_position_embeddings:
            raise ValueError("sequence exceeds max_position_embeddings")
        if position_ids is None:
            if attention_mask is not None:
                position_ids = (attention_mask.long().cumsum(-1) - 1).clamp(min=0)[:, past:]
            else:
                position_ids = torch.arange(past, past + length, device=input_ids.device)
                position_ids = position_ids.unsqueeze(0).expand(batch, -1)
        mask = self._mask(length, past + length, input_ids.device, attention_mask, cache)
        hidden = self.embedding(input_ids)
        for index, block in enumerate(self.blocks):
            layer_cache = cache.layers[index] if cache is not None else None
            if self.activation_checkpointing and self.training and cache is None:
                hidden = checkpoint(block, hidden, position_ids, mask, use_reentrant=False)
            else:
                hidden = block(hidden, position_ids, mask, layer_cache)
        return self.lm_head(self.final_norm(hidden))

    @staticmethod
    def _mask(
        length: int,
        total: int,
        device: torch.device,
        attention_mask: Tensor | None,
        cache: KVCache | None,
    ) -> Tensor | None:
        if attention_mask is None and total == length:
            return build_attention_mask(length, total, device)
        if attention_mask is None and length == 1:
            return None
        return build_attention_mask(length, total, device, attention_mask)
