from __future__ import annotations

import statistics
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass

import torch
from torch import Tensor

from llm.model.attention import (
    build_attention_mask,
    reference_attention,
    repeat_kv,
    sdpa_attention,
)
from llm.optimization.benchmark import peak_device_gb, synchronize
from llm.optimization.memory import release_device_memory

WARMUP = 3
INNER_ITERATIONS = 5


@dataclass(frozen=True)
class AttentionShape:
    batch: int
    heads: int
    kv_heads: int
    length: int
    head_dim: int

    @property
    def tokens(self) -> int:
        return self.batch * self.length


@dataclass(frozen=True)
class AttentionTiming:
    variant: str
    forward_ms: float
    forward_spread_ms: float
    backward_ms: float
    backward_spread_ms: float
    peak_device_gb: float
    max_abs_difference: float


Operation = Callable[[Tensor, Tensor, Tensor], Tensor]


def variants(shape: AttentionShape, device: torch.device) -> dict[str, Operation]:
    groups = shape.heads // shape.kv_heads
    mask = build_attention_mask(shape.length, shape.length, device)

    def masked_repeat(q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        return sdpa_attention(q, repeat_kv(k, groups), repeat_kv(v, groups), mask)

    def masked_native(q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        return sdpa_attention(q, k, v, mask, enable_gqa=True)

    def causal_repeat(q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        return sdpa_attention(q, repeat_kv(k, groups), repeat_kv(v, groups), None, is_causal=True)

    def causal_native(q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        return sdpa_attention(q, k, v, None, is_causal=True, enable_gqa=True)

    def reference(q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        return reference_attention(q, repeat_kv(k, groups), repeat_kv(v, groups), mask)

    return {
        "reference": reference,
        "sdpa_mask_repeat": masked_repeat,
        "sdpa_mask_native_gqa": masked_native,
        "sdpa_causal_repeat": causal_repeat,
        "sdpa_causal_native_gqa": causal_native,
    }


def make_inputs(shape: AttentionShape, device: torch.device, seed: int) -> tuple[Tensor, ...]:
    generator = torch.Generator().manual_seed(seed)

    def sample(heads: int) -> Tensor:
        values = torch.randn(shape.batch, heads, shape.length, shape.head_dim, generator=generator)
        return values.to(device).requires_grad_(True)

    return sample(shape.heads), sample(shape.kv_heads), sample(shape.kv_heads)


def time_calls(call: Callable[[], object], device: torch.device, repeats: int) -> list[float]:
    for _ in range(WARMUP):
        call()
    samples = []
    for _ in range(repeats):
        synchronize(device)
        started = time.perf_counter()
        for _ in range(INNER_ITERATIONS):
            call()
        synchronize(device)
        samples.append((time.perf_counter() - started) / INNER_ITERATIONS * 1000)
    return samples


def spread(samples: list[float]) -> float:
    return max(samples) - min(samples)


def measure_attention(
    shape: AttentionShape, device: torch.device, repeats: int, seed: int = 0
) -> list[AttentionTiming]:
    q, k, v = make_inputs(shape, device, seed)
    operations = variants(shape, device)
    baseline = operations["sdpa_mask_repeat"](q, k, v).detach()
    results = []
    for name, operation in operations.items():
        release_device_memory(device)
        with torch.no_grad():
            output = operation(q, k, v)
        difference = float((output - baseline).abs().max())

        def forward_call(op: Operation = operation) -> Tensor:
            return op(q, k, v)

        def backward(op: Operation = operation) -> None:
            op(q, k, v).sum().backward()

        forward = time_calls(forward_call, device, repeats)

        backward_samples = time_calls(backward, device, repeats)
        results.append(
            AttentionTiming(
                name,
                statistics.median(forward),
                spread(forward),
                statistics.median(backward_samples),
                spread(backward_samples),
                peak_device_gb(device),
                difference,
            )
        )
    return results


def report(
    timings: list[AttentionTiming], baseline: str = "sdpa_mask_repeat"
) -> list[dict[str, float | str]]:
    reference = next(t for t in timings if t.variant == baseline)
    return [
        {
            **asdict(t),
            "forward_vs_baseline": t.forward_ms / reference.forward_ms,
            "backward_vs_baseline": t.backward_ms / reference.backward_ms,
        }
        for t in timings
    ]
