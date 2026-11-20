from __future__ import annotations

import tempfile
import time
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, replace
from functools import partial
from pathlib import Path
from typing import TypeVar

import psutil
import torch

from llm.core.config import ExperimentConfig, OptimizerConfig
from llm.optimization.benchmark import BYTES_PER_GB, peak_device_gb, synchronize
from llm.optimization.grad import clip_grad_norm
from llm.optimization.memory import release_device_memory
from llm.training.runner import build_trainer

WARMUP_STEPS = 2
T = TypeVar("T")
PHASES = ("data", "transfer", "forward", "backward", "clip", "optimizer")


@dataclass(frozen=True)
class Variant:
    name: str
    micro_batch: int = 8
    accumulation: int = 4
    pack: bool = False
    native_gqa: bool = False
    fast_causal: bool = False
    foreach: bool = True
    precision: str = "fp32"
    checkpointing: bool = False


@dataclass(frozen=True)
class Throughput:
    variant: str
    steps_per_second: float
    useful_tokens_per_second: float
    slot_tokens_per_second: float
    useful_fraction: float
    peak_device_gb: float
    swap_gb: float


def apply_variant(config: ExperimentConfig, variant: Variant, workdir: Path) -> ExperimentConfig:
    train = config.train.model_copy(
        update={
            "micro_batch_size": variant.micro_batch,
            "gradient_accumulation_steps": variant.accumulation,
            "pack_sequences": variant.pack,
            "native_gqa": variant.native_gqa,
            "fast_causal": variant.fast_causal,
            "precision": variant.precision,
            "activation_checkpointing": variant.checkpointing,
            "init_from": None,
            "restore_optimizer": False,
            "max_ram_fraction": 1.0,
            "dynamic_batch_size": False,
            "output_dir": str(workdir / "experiment"),
            "checkpoint_dir": str(workdir / "checkpoints"),
            "optimizer": OptimizerConfig(
                **{**config.train.optimizer.model_dump(), "foreach": variant.foreach}
            ),
        }
    )
    return config.model_copy(update={"train": train})


def slots_per_step(trainer, steps: int) -> int:
    source = trainer.train_source
    sample = source.next_batch()
    source.load_state_dict({"epoch": 0, "cursor": 0})
    return int(sample[0].numel()) * trainer.state.gradient_accumulation_steps


def measure_throughput(config: ExperimentConfig, variant: Variant, steps: int) -> Throughput:
    with tempfile.TemporaryDirectory() as workdir:
        trainer = build_trainer(apply_variant(config, variant, Path(workdir)), resume=False)
        device = trainer.device
        release_device_memory(device)
        for _ in range(WARMUP_STEPS):
            trainer.run_step()
        synchronize(device)
        started = time.perf_counter()
        useful = 0
        for _ in range(steps):
            useful += trainer.run_step()[1]
        synchronize(device)
        elapsed = time.perf_counter() - started
        slots = slots_per_step(trainer, steps) * steps
        peak = peak_device_gb(device)
        swap = psutil.swap_memory().used / BYTES_PER_GB
        del trainer
    return Throughput(
        variant.name, steps / elapsed, useful / elapsed, slots / elapsed, useful / slots, peak, swap
    )


class PhaseTimer:
    def __init__(self, device: torch.device) -> None:
        self.device = device
        self.totals: dict[str, float] = defaultdict(float)

    def run(self, name: str, work: Callable[[], T]) -> T:
        synchronize(self.device)
        started = time.perf_counter()
        result = work()
        synchronize(self.device)
        self.totals[name] += time.perf_counter() - started
        return result

    def reset(self) -> None:
        self.totals.clear()


def backpropagate(loss: torch.Tensor, total: int) -> None:
    (loss / total).backward()


def profile_step(trainer, timer: PhaseTimer, accumulation: int, max_norm: float) -> None:
    trainer.optimizer.zero_grad(set_to_none=True)
    host = timer.run(
        "data", lambda: [trainer.train_source.next_batch() for _ in range(accumulation)]
    )
    total = sum(trainer.objective.weight(batch) for batch in host)
    batches = timer.run("transfer", lambda: [trainer._to_device(batch) for batch in host])
    for batch in batches:
        loss = timer.run("forward", partial(trainer._forward_loss, batch))
        timer.run("backward", partial(backpropagate, loss, total))
    timer.run("clip", lambda: clip_grad_norm(trainer.model.parameters(), max_norm))
    timer.run("optimizer", trainer.optimizer.step)


def profile_phases(config: ExperimentConfig, variant: Variant, steps: int) -> dict[str, float]:
    with tempfile.TemporaryDirectory() as workdir:
        trainer = build_trainer(apply_variant(config, variant, Path(workdir)), resume=False)
        timer = PhaseTimer(trainer.device)
        max_norm = config.train.optimizer.max_grad_norm
        for _ in range(WARMUP_STEPS):
            profile_step(trainer, timer, variant.accumulation, max_norm)
        timer.reset()
        for _ in range(steps):
            profile_step(trainer, timer, variant.accumulation, max_norm)
        result = {name: timer.totals[name] / steps for name in PHASES}
        started = time.perf_counter()
        trainer.save_checkpoint()
        result["checkpoint_save"] = time.perf_counter() - started
        del trainer
    return result


def with_overrides(variant: Variant, **changes) -> Variant:
    return replace(variant, **changes)
