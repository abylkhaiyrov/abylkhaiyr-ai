from __future__ import annotations

from dataclasses import dataclass

from llm.core.config import ModelConfig
from llm.core.profiles import count_parameters, estimate_training_memory_gb

SECONDS_PER_HOUR = 3600
CHINCHILLA_TOKENS_PER_PARAMETER = 20
BYTES_PER_PARAMETER_INFERENCE = 4
ATTENTION_FLOP_FACTOR = 12
PARAMETER_FLOP_FACTOR = 6
BYTES_PER_GB = 1024**3
KV_BYTES = 4


def flops_per_token(config: ModelConfig, sequence_length: int) -> float:
    attention = (
        ATTENTION_FLOP_FACTOR * config.num_hidden_layers * sequence_length * config.hidden_size
    )
    return PARAMETER_FLOP_FACTOR * count_parameters(config) + attention


def calibrate_flops(config: ModelConfig, sequence_length: int, tokens_per_second: float) -> float:
    return flops_per_token(config, sequence_length) * tokens_per_second


def estimated_tokens_per_second(
    config: ModelConfig, sequence_length: int, achieved_flops: float
) -> float:
    return achieved_flops / flops_per_token(config, sequence_length)


def kv_cache_gb(config: ModelConfig, batch: int = 1) -> float:
    per_token = 2 * config.num_hidden_layers * config.num_key_value_heads * config.head_dim
    return per_token * config.max_position_embeddings * batch * KV_BYTES / BYTES_PER_GB


@dataclass(frozen=True)
class ScalingRow:
    name: str
    parameters_m: float
    flops_per_token_m: float
    training_memory_gb: float
    kv_cache_mb: float
    tokens_per_second: float
    chinchilla_tokens_b: float
    chinchilla_hours: float
    hours_per_billion_tokens: float


def scaling_row(
    name: str,
    config: ModelConfig,
    sequence_length: int,
    micro_batch: int,
    achieved_flops: float,
) -> ScalingRow:
    parameters = count_parameters(config)
    speed = estimated_tokens_per_second(config, sequence_length, achieved_flops)
    chinchilla = CHINCHILLA_TOKENS_PER_PARAMETER * parameters
    return ScalingRow(
        name,
        parameters / 1e6,
        flops_per_token(config, sequence_length) / 1e6,
        estimate_training_memory_gb(config, micro_batch, sequence_length),
        kv_cache_gb(config) * 1024,
        speed,
        chinchilla / 1e9,
        chinchilla / speed / SECONDS_PER_HOUR,
        1e9 / speed / SECONDS_PER_HOUR,
    )
