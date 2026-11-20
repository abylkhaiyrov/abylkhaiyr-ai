from __future__ import annotations

import math
import tempfile
from pathlib import Path

import torch

from llm.core.config import ExperimentConfig, OptimizerConfig
from llm.core.profiles import build_model_config, count_parameters
from llm.optimization.memory import release_device_memory
from llm.training.runner import build_trainer

WARMUP_FRACTION = 0.1


def probe_config(
    base: ExperimentConfig, profile: str, budget_tokens: int, workdir: Path
) -> ExperimentConfig:
    model = build_model_config(profile, base.model.vocab_size)
    tokens_per_step = (
        base.train.micro_batch_size
        * base.train.gradient_accumulation_steps
        * base.train.sequence_length
    )
    steps = max(math.ceil(budget_tokens / tokens_per_step), 2)
    optimizer = OptimizerConfig(
        **{
            **base.train.optimizer.model_dump(),
            "warmup_steps": max(int(steps * WARMUP_FRACTION), 1),
        }
    )
    train = base.train.model_copy(
        update={
            "init_from": None,
            "restore_optimizer": False,
            "max_steps": steps,
            "eval_interval": steps,
            "eval_batches": 40,
            "checkpoint_interval": steps + 1,
            "protected_steps": [],
            "activation_checkpointing": False,
            "dynamic_batch_size": False,
            "max_ram_fraction": 1.0,
            "output_dir": str(workdir / "experiment"),
            "checkpoint_dir": str(workdir / "checkpoints"),
            "checkpoint_alias": None,
            "optimizer": optimizer,
        }
    )
    return base.model_copy(update={"model": model, "train": train})


def run_probe(base: ExperimentConfig, profile: str, budget_tokens: int) -> dict[str, float | str]:
    with tempfile.TemporaryDirectory() as workdir:
        config = probe_config(base, profile, budget_tokens, Path(workdir))
        trainer = build_trainer(config, resume=False)
        result = trainer.train()
        metrics: dict[str, float | str] = {
            "profile": profile,
            "parameters": count_parameters(config.model),
            "steps": result.final_step,
            "tokens": result.tokens_seen,
            "validation_loss": result.last_validation_loss,
            "train_loss": result.final_train_loss,
            "tokens_per_second": result.tokens_per_second,
            "duration_seconds": result.duration_seconds,
            "device_memory_gb": result.device_memory_gb,
        }
        del trainer
    release_device_memory(torch.device(base.train.device))
    return metrics
