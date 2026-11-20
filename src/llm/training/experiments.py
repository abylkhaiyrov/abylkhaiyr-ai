from __future__ import annotations

import tempfile
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from llm.core.config import ExperimentConfig, OptimizerConfig
from llm.data.binary import ConversationLoader, MixtureLoader, token_dtype
from llm.evaluation.comparison import FIXED_PARAMS, word_list
from llm.evaluation.conversation import engine_responder, run_conversation_benchmark
from llm.inference.engine import InferenceEngine
from llm.optimization.memory import release_device_memory
from llm.tokenizer.bpe import BPETokenizer
from llm.training.engine import Trainer
from llm.training.losses import IGNORE_INDEX
from llm.training.runner import _build_loaders, build_trainer

PROBE_STEPS = 12
FIXED_VALIDATION_BATCHES = 60


@dataclass(frozen=True)
class EfficiencyVariant:
    name: str
    mixture: Mapping[str, float]
    pack: bool = False
    bucket_window: int = 0
    curriculum_start: Mapping[str, float] = field(default_factory=dict)
    curriculum_batches: int = 0
    learning_rate: float | None = None


def configure(
    base: ExperimentConfig, variant: EfficiencyVariant, steps: int, workdir: Path
) -> ExperimentConfig:
    optimizer = base.train.optimizer.model_dump()
    if variant.learning_rate is not None:
        optimizer["learning_rate"] = variant.learning_rate
    optimizer["warmup_steps"] = max(min(optimizer["warmup_steps"], steps // 10), 1)
    train = base.train.model_copy(
        update={
            "mixture": dict(variant.mixture),
            "pack_sequences": variant.pack,
            "bucket_window": variant.bucket_window,
            "curriculum_start": dict(variant.curriculum_start),
            "curriculum_batches": variant.curriculum_batches,
            "max_steps": steps,
            "eval_interval": max(steps, 1),
            "checkpoint_interval": steps + 1,
            "protected_steps": [],
            "max_ram_fraction": 1.0,
            "dynamic_batch_size": False,
            "output_dir": str(workdir / "experiment"),
            "checkpoint_dir": str(workdir / "checkpoints"),
            "checkpoint_alias": None,
            "optimizer": OptimizerConfig(**optimizer),
        }
    )
    return base.model_copy(update={"train": train})


def supervised_tokens_per_step(config: ExperimentConfig, tokenizer: BPETokenizer) -> float:
    train_source, _ = _build_loaders(config, tokenizer)
    accumulation = config.train.gradient_accumulation_steps
    counts = [
        int((train_source.next_batch()[1] != IGNORE_INDEX).sum())
        for _ in range(PROBE_STEPS * accumulation)
    ]
    return float(np.sum(counts)) / PROBE_STEPS


def steps_for_budget(
    base: ExperimentConfig, variant: EfficiencyVariant, tokenizer: BPETokenizer, budget: int
) -> int:
    with tempfile.TemporaryDirectory() as workdir:
        probe = configure(base, variant, 1000, Path(workdir))
        return max(round(budget / supervised_tokens_per_step(probe, tokenizer)), 1)


def fixed_validation(trainer: Trainer, config: ExperimentConfig, tokenizer: BPETokenizer) -> float:
    root = Path(config.train.data_dir) / "sft_groups"
    dtype = token_dtype(tokenizer.vocab_size)
    groups = sorted(
        {p.name.split(".")[1] for p in root.glob("validation.*.jsonl")} & set(config.train.mixture)
    )
    loaders = {
        name: ConversationLoader(
            root / f"validation.{name}",
            dtype,
            config.train.sequence_length + 1,
            config.train.micro_batch_size,
            tokenizer.pad_id,
            0,
            False,
        )
        for name in groups
    }
    fixed = MixtureLoader(loaders, dict.fromkeys(loaders, 1.0), 0)
    trainer.validation_source = fixed
    trainer.config.eval_batches = FIXED_VALIDATION_BATCHES
    return trainer.evaluate()


def conversational_quality(
    trainer: Trainer, tokenizer: BPETokenizer, data_dir: str
) -> dict[str, float]:
    engine = InferenceEngine(trainer.model, tokenizer, trainer.device)
    report = run_conversation_benchmark(engine_responder(engine, FIXED_PARAMS), word_list(data_dir))
    trainer.model.train()
    return report.metrics


def run_variant(
    base: ExperimentConfig,
    variant: EfficiencyVariant,
    budget_tokens: int,
    tokenizer: BPETokenizer,
    quality: bool,
    corpus_dir: str,
) -> dict[str, float | str]:
    steps = steps_for_budget(base, variant, tokenizer, budget_tokens)
    with tempfile.TemporaryDirectory() as workdir:
        config = configure(base, variant, steps, Path(workdir))
        trainer = build_trainer(config, resume=False)
        result = trainer.train()
        validation = fixed_validation(trainer, config, tokenizer)
        metrics: dict[str, float | str] = {
            "variant": variant.name,
            "steps": steps,
            "supervised_tokens": result.tokens_seen,
            "duration_seconds": result.duration_seconds,
            "useful_tokens_per_second": result.tokens_per_second,
            "train_loss": result.final_train_loss,
            "fixed_validation_loss": validation,
            "device_memory_gb": result.device_memory_gb,
        }
        if quality:
            metrics.update(conversational_quality(trainer, tokenizer, corpus_dir))
        del trainer
    release_device_memory(torch.device(base.train.device))
    return metrics
