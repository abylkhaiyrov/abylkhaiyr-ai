from __future__ import annotations

from dataclasses import dataclass

from llm.core.config import ModelConfig
from llm.core.errors import ConfigError
from llm.core.hardware import HardwareInfo

PARAMS_BYTES_TRAINING = 16
ACTIVATION_SAFETY_FACTOR = 2.0
MEMORY_BUDGET_FRACTION = 0.7
MIN_HEAD_DIM = 32


@dataclass(frozen=True)
class Profile:
    name: str
    min_params: int
    max_params: int
    context_length: int
    hidden_size: int
    layers: int
    heads: int
    kv_heads: int
    ffn_multiplier: float


PROFILES: dict[str, Profile] = {
    "debug": Profile("debug", 1_000_000, 5_000_000, 128, 128, 4, 4, 2, 2.67),
    "tiny": Profile("tiny", 10_000_000, 30_000_000, 512, 384, 8, 6, 2, 2.67),
    "small": Profile("small", 50_000_000, 150_000_000, 1024, 768, 12, 12, 4, 2.67),
    "medium": Profile("medium", 300_000_000, 500_000_000, 2048, 1024, 24, 16, 4, 2.67),
    "large": Profile("large", 1_000_000_000, 3_000_000_000, 2048, 2048, 24, 16, 8, 2.67),
}

FFN_ROUNDING = 64


def build_model_config(profile_name: str, vocab_size: int) -> ModelConfig:
    if profile_name not in PROFILES:
        raise ConfigError(f"unknown profile: {profile_name}")
    profile = PROFILES[profile_name]
    intermediate = int(profile.hidden_size * profile.ffn_multiplier)
    intermediate = (intermediate + FFN_ROUNDING - 1) // FFN_ROUNDING * FFN_ROUNDING
    return ModelConfig(
        vocab_size=vocab_size,
        hidden_size=profile.hidden_size,
        num_hidden_layers=profile.layers,
        num_attention_heads=profile.heads,
        num_key_value_heads=profile.kv_heads,
        intermediate_size=intermediate,
        max_position_embeddings=profile.context_length,
    )


def count_parameters(config: ModelConfig) -> int:
    head_dim = config.head_dim
    attention = (
        config.hidden_size
        * head_dim
        * (config.num_attention_heads * 2 + config.num_key_value_heads * 2)
    )
    ffn = 3 * config.hidden_size * config.intermediate_size
    norms = 2 * config.hidden_size
    embeddings = config.vocab_size * config.hidden_size
    head = 0 if config.tie_word_embeddings else embeddings
    return (
        config.num_hidden_layers * (attention + ffn + norms)
        + embeddings
        + head
        + config.hidden_size
    )


def estimate_training_memory_gb(
    config: ModelConfig, micro_batch: int, sequence_length: int, bytes_per_activation: int = 4
) -> float:
    params = count_parameters(config) * PARAMS_BYTES_TRAINING
    per_token = (
        config.num_hidden_layers * (config.hidden_size * 12 + config.intermediate_size * 3)
        + config.vocab_size * 3
    )
    activations = micro_batch * sequence_length * per_token * bytes_per_activation
    return (params + activations * ACTIVATION_SAFETY_FACTOR) / 1024**3


def fits_hardware(profile_name: str, vocab_size: int, hardware: HardwareInfo) -> bool:
    config = build_model_config(profile_name, vocab_size)
    required = estimate_training_memory_gb(config, 1, PROFILES[profile_name].context_length)
    return required <= hardware.device_memory_gb * MEMORY_BUDGET_FRACTION


def available_profiles(vocab_size: int, hardware: HardwareInfo) -> list[str]:
    return [name for name in PROFILES if fits_hardware(name, vocab_size, hardware)]


def suggest_micro_batch(
    config: ModelConfig, sequence_length: int, hardware: HardwareInfo, ceiling: int = 64
) -> int:
    budget = hardware.device_memory_gb * MEMORY_BUDGET_FRACTION
    batch = 1
    while (
        batch < ceiling
        and estimate_training_memory_gb(config, batch * 2, sequence_length) <= budget
    ):
        batch *= 2
    return batch


def suggest_compact_profile(profile_name: str, vocab_size: int, hardware: HardwareInfo) -> str:
    candidates = available_profiles(vocab_size, hardware)
    if not candidates:
        raise ConfigError("no profile fits available hardware")
    return profile_name if profile_name in candidates else candidates[-1]
