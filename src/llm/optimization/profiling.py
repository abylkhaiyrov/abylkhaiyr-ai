from __future__ import annotations

import math
import tempfile
import time
from collections import defaultdict
from collections.abc import Callable, Iterator
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, replace
from functools import partial
from pathlib import Path
from typing import TypeVar, cast

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from llm.core.config import ExperimentConfig, OptimizerConfig
from llm.model.transformer import CausalLM
from llm.optimization.benchmark import peak_device_gb, synchronize
from llm.optimization.grad import clip_grad_norm
from llm.optimization.memory import release_device_memory, swap_used_gb
from llm.training.engine import Trainer
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
    pad_multiple: int = 64
    tf32: bool = False
    compile: bool = False
    sdpa: str = ""


@dataclass(frozen=True)
class Throughput:
    variant: str
    steps_per_second: float
    useful_tokens_per_second: float
    slot_tokens_per_second: float
    useful_fraction: float
    peak_device_gb: float
    swap_gb: float | None


def apply_variant(config: ExperimentConfig, variant: Variant, workdir: Path) -> ExperimentConfig:
    train = config.train.model_copy(
        update={
            "micro_batch_size": variant.micro_batch,
            "gradient_accumulation_steps": variant.accumulation,
            "pack_sequences": variant.pack,
            "pad_multiple": variant.pad_multiple,
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


SDPA_BACKENDS = {"math": "MATH", "efficient": "EFFICIENT_ATTENTION", "flash": "FLASH_ATTENTION"}


def build_variant_trainer(config: ExperimentConfig, variant: Variant, workdir: Path) -> Trainer:
    trainer = build_trainer(apply_variant(config, variant, workdir), resume=False)
    if variant.compile:
        trainer.model = cast(CausalLM, torch.compile(trainer.model))
    return trainer


@contextmanager
def runtime(variant: Variant, device: torch.device) -> Iterator[None]:
    with ExitStack() as stack:
        if device.type == "cuda":
            previous = torch.backends.cuda.matmul.allow_tf32
            torch.backends.cuda.matmul.allow_tf32 = variant.tf32
            stack.callback(setattr, torch.backends.cuda.matmul, "allow_tf32", previous)
            if variant.sdpa:
                backend = getattr(SDPBackend, SDPA_BACKENDS[variant.sdpa])
                stack.enter_context(sdpa_kernel([backend]))
        yield


def count_slots(source) -> dict[str, int]:
    counter = {"slots": 0}
    original = source.next_batch

    def counting():
        batch = original()
        counter["slots"] += int(batch[0].numel())
        return batch

    source.next_batch = counting
    return counter


def measure_throughput(config: ExperimentConfig, variant: Variant, steps: int) -> Throughput:
    with tempfile.TemporaryDirectory() as workdir:
        trainer = build_variant_trainer(config, variant, Path(workdir))
        runtime_stack = ExitStack()
        runtime_stack.enter_context(runtime(variant, trainer.device))
        device = trainer.device
        release_device_memory(device)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        for _ in range(WARMUP_STEPS):
            trainer.run_step()
        counter = count_slots(trainer.train_source)
        synchronize(device)
        started = time.perf_counter()
        useful = 0
        for _ in range(steps):
            useful += trainer.run_step()[1]
        synchronize(device)
        elapsed = time.perf_counter() - started
        slots = counter["slots"]
        peak = peak_device_gb(device)
        swap = swap_used_gb()
        runtime_stack.close()
        del trainer
    return Throughput(
        variant.name,
        steps / elapsed,
        useful / elapsed,
        slots / elapsed,
        useful / slots,
        peak,
        swap if math.isfinite(swap) else None,
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
        trainer = build_variant_trainer(config, variant, Path(workdir))
        runtime_stack = ExitStack()
        runtime_stack.enter_context(runtime(variant, trainer.device))
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
        runtime_stack.close()
        del trainer
    return result


def with_overrides(variant: Variant, **changes) -> Variant:
    return replace(variant, **changes)


def loss_trajectory(config: ExperimentConfig, variant: Variant, steps: int) -> list[float]:
    with tempfile.TemporaryDirectory() as workdir:
        trainer = build_variant_trainer(config, variant, Path(workdir))
        runtime_stack = ExitStack()
        runtime_stack.enter_context(runtime(variant, trainer.device))
        losses = [trainer.run_step()[0] for _ in range(steps)]
        runtime_stack.close()
        del trainer
    return losses


def max_trajectory_gap(reference: list[float], candidate: list[float]) -> float:
    return max(abs(a - b) for a, b in zip(reference, candidate, strict=True))
