from __future__ import annotations

import time
from dataclasses import dataclass

import psutil
import torch

from llm.core.config import ModelConfig
from llm.model.transformer import CausalLM
from llm.optimization.adamw import AdamW
from llm.optimization.memory import (
    device_memory_used_gb,
    is_out_of_memory,
    release_device_memory,
)
from llm.optimization.precision import autocast_context
from llm.training.losses import causal_lm_loss_sum

WARMUP_STEPS = 2
BYTES_PER_GB = 1024**3


@dataclass(frozen=True)
class BenchmarkCase:
    micro_batch: int
    sequence_length: int
    dtype: torch.dtype = torch.float32
    activation_checkpointing: bool = False


@dataclass(frozen=True)
class BenchmarkResult:
    case: BenchmarkCase
    tokens_per_second: float
    peak_device_gb: float
    swap_used_gb: float
    ok: bool


def synchronize(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


def peak_device_gb(device: torch.device) -> float:
    if device.type == "mps":
        return torch.mps.driver_allocated_memory() / BYTES_PER_GB
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated(device) / BYTES_PER_GB
    return device_memory_used_gb(device)


def run_case(
    config: ModelConfig, case: BenchmarkCase, device: torch.device, steps: int
) -> BenchmarkResult:
    release_device_memory(device)
    model = CausalLM(config).to(device).train()
    model.activation_checkpointing = case.activation_checkpointing
    optimizer = AdamW(model.parameters(), lr=1e-4)
    shape = (case.micro_batch, case.sequence_length + 1)
    try:
        for step in range(WARMUP_STEPS + steps):
            if step == WARMUP_STEPS:
                synchronize(device)
                started = time.perf_counter()
            ids = torch.randint(0, config.vocab_size, shape, device=device)
            with autocast_context(device, case.dtype):
                logits = model(ids[:, :-1])
            loss_sum, count = causal_lm_loss_sum(logits, ids[:, 1:])
            (loss_sum / count).backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        synchronize(device)
        elapsed = time.perf_counter() - started
    except (RuntimeError, torch.OutOfMemoryError) as error:
        if not is_out_of_memory(error):
            raise
        return BenchmarkResult(
            case, 0.0, peak_device_gb(device), psutil.swap_memory().used / BYTES_PER_GB, False
        )
    tokens = steps * case.micro_batch * case.sequence_length
    return BenchmarkResult(
        case,
        tokens / elapsed,
        peak_device_gb(device),
        psutil.swap_memory().used / BYTES_PER_GB,
        True,
    )
