from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from llm.core.errors import ConfigError

PrecisionName = Literal["fp32", "fp16", "bf16", "auto"]
DeviceName = Literal["cuda", "mps", "cpu", "auto"]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class ModelConfig(StrictModel):
    vocab_size: int = Field(gt=0)
    hidden_size: int = Field(gt=0)
    num_hidden_layers: int = Field(gt=0)
    num_attention_heads: int = Field(gt=0)
    num_key_value_heads: int = Field(gt=0)
    intermediate_size: int = Field(gt=0)
    max_position_embeddings: int = Field(gt=0)
    rope_theta: float = Field(default=10000.0, gt=0)
    rms_norm_eps: float = Field(default=1e-5, gt=0)
    dropout: float = Field(default=0.0, ge=0.0, lt=1.0)
    initializer_range: float = Field(default=0.02, gt=0)
    tie_word_embeddings: bool = True

    @model_validator(mode="after")
    def check_compatibility(self) -> ModelConfig:
        if self.hidden_size % self.num_attention_heads:
            raise ValueError("hidden_size must be divisible by num_attention_heads")
        if self.num_attention_heads % self.num_key_value_heads:
            raise ValueError("num_attention_heads must be divisible by num_key_value_heads")
        if self.head_dim % 2:
            raise ValueError("head_dim must be even for rotary embeddings")
        return self

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads

    @property
    def num_kv_groups(self) -> int:
        return self.num_attention_heads // self.num_key_value_heads


class TokenizerConfig(StrictModel):
    vocab_size: int = Field(default=8192, gt=0)
    min_frequency: int = Field(default=2, ge=1)
    path: str = "data/tokenizer"


class DataSourceConfig(StrictModel):
    name: str
    dataset: str
    data_files: str | list[str] | None = None
    config: str | None = None
    split: str = "train"
    text_field: str | None = None
    template: str | None = None
    response_field: str | None = None
    kind: Literal["text", "instruction", "math", "code"] = "text"
    language: str | None = None
    license: str
    max_documents: int | None = None
    max_chars: int = Field(default=20000, gt=0)
    min_chars: int | None = Field(default=None, ge=1)
    check_language: bool = True
    group: str = "general"
    license_field: str | None = None
    row_filter: dict[str, str] | None = None


PERMISSIVE_LICENSES = (
    "mit",
    "apache-2.0",
    "bsd-2-clause",
    "bsd-3-clause",
    "isc",
    "unlicense",
    "cc0-1.0",
)


class DataConfig(StrictModel):
    version: str = "v1"
    sources: list[DataSourceConfig]
    allowed_licenses: list[str] = Field(default_factory=lambda: list(PERMISSIVE_LICENSES))
    decontaminate: bool = False
    decontamination_ngram: int = Field(default=8, ge=3)
    output_dir: str = "data/processed"
    raw_dir: str = "data/raw"
    validation_fraction: float = Field(default=0.01, gt=0, lt=0.5)
    test_fraction: float = Field(default=0.01, gt=0, lt=0.5)
    min_chars: int = Field(default=200, ge=1)
    max_symbol_ratio: float = Field(default=0.3, gt=0, le=1)
    max_repeated_line_ratio: float = Field(default=0.5, gt=0, le=1)
    near_duplicate_threshold: float = Field(default=0.8, gt=0, le=1)
    minhash_permutations: int = Field(default=64, ge=8)
    shingle_size: int = Field(default=5, ge=1)
    detect_language: bool = True
    language_sample_chars: int = Field(default=2000, gt=0)
    allowed_languages: list[str] = Field(default_factory=lambda: ["ru", "en", "kk"])
    seed: int = 1234
    min_free_disk_gb: float = Field(default=5.0, ge=0)


class OptimizerConfig(StrictModel):
    learning_rate: float = Field(default=3e-4, gt=0)
    min_lr_ratio: float = Field(default=0.1, ge=0, le=1)
    weight_decay: float = Field(default=0.1, ge=0)
    beta1: float = Field(default=0.9, gt=0, lt=1)
    beta2: float = Field(default=0.95, gt=0, lt=1)
    eps: float = Field(default=1e-8, gt=0)
    max_grad_norm: float = Field(default=1.0, gt=0)
    warmup_steps: int = Field(default=100, ge=0)
    foreach: bool = False


class SamplerConfig(StrictModel):
    refusal_share_cap: float | None = Field(default=None, gt=0, lt=1)


class TrainConfig(StrictModel):
    stage: Literal["pretrain", "sft", "dpo"] = "pretrain"
    data_dir: str = "data/processed"
    output_dir: str = "experiments/run"
    checkpoint_dir: str = "checkpoints/run"
    init_from: str | None = None
    restore_optimizer: bool = False
    max_steps: int = Field(default=1000, gt=0)
    micro_batch_size: int = Field(default=8, gt=0)
    gradient_accumulation_steps: int = Field(default=4, gt=0)
    sequence_length: int = Field(default=256, gt=0)
    eval_interval: int = Field(default=100, gt=0)
    eval_batches: int = Field(default=20, gt=0)
    log_interval: int = Field(default=10, gt=0)
    checkpoint_interval: int = Field(default=200, gt=0)
    keep_checkpoints: int = Field(default=3, gt=0)
    protected_steps: list[int] = Field(default_factory=list)
    checkpoint_alias: str | None = None
    early_stopping_patience: int | None = Field(default=None, gt=0)
    precision: PrecisionName = "auto"
    device: DeviceName = "auto"
    activation_checkpointing: bool = False
    pack_sequences: bool = False
    freeze_layers: int = Field(default=0, ge=0)
    layer_lr_decay: float = Field(default=1.0, gt=0, le=1)
    pad_multiple: int = Field(default=64, gt=0)
    bucket_window: int = Field(default=0, ge=0)
    curriculum_start: dict[str, float] = Field(default_factory=dict)
    curriculum_batches: int = Field(default=0, ge=0)
    native_gqa: bool = False
    fast_causal: bool = False
    dynamic_batch_size: bool = True
    min_micro_batch_size: int = Field(default=1, gt=0)
    seed: int = 1234
    num_threads: int | None = Field(default=None, gt=0)
    max_ram_fraction: float = Field(default=0.85, gt=0, le=1)
    min_free_disk_gb: float = Field(default=2.0, ge=0)
    dpo_beta: float = Field(default=0.1, gt=0)
    mixture: dict[str, float] = Field(default_factory=dict)
    dataset_version: str | None = None
    sft_replay_dir: str | None = None
    sft_replay_fraction: float = Field(default=0.0, ge=0, lt=1)
    sampler: SamplerConfig = Field(default_factory=SamplerConfig)
    optimizer: OptimizerConfig = Field(default_factory=OptimizerConfig)


class ExperimentConfig(StrictModel):
    name: str
    model: ModelConfig
    tokenizer: TokenizerConfig = Field(default_factory=TokenizerConfig)
    data: DataConfig | None = None
    train: TrainConfig = Field(default_factory=TrainConfig)


def load_yaml(path: str | Path) -> dict[str, Any]:
    file_path = Path(path)
    if not file_path.is_file():
        raise ConfigError(f"config file not found: {file_path}")
    with file_path.open(encoding="utf-8") as handle:
        content = yaml.safe_load(handle)
    if not isinstance(content, dict):
        raise ConfigError(f"config must be a mapping: {file_path}")
    return content


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    try:
        return ExperimentConfig.model_validate(load_yaml(path))
    except ValidationError as error:
        raise ConfigError(str(error)) from error
