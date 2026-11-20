from __future__ import annotations

import statistics
import time
from dataclasses import dataclass

import torch
from torch import Tensor

from llm.inference.engine import InferenceEngine
from llm.model.cache import KVCache
from llm.optimization.benchmark import peak_device_gb, synchronize

WARMUP_RUNS = 1
PROMPT_TOKEN_ID = 5


class ConcatLayerCache:
    def __init__(self) -> None:
        self.keys: Tensor | None = None
        self.values: Tensor | None = None

    @property
    def length(self) -> int:
        return 0 if self.keys is None else self.keys.shape[2]

    def update(self, keys: Tensor, values: Tensor) -> tuple[Tensor, Tensor]:
        if self.keys is None or self.values is None:
            self.keys, self.values = keys, values
        else:
            self.keys = torch.cat((self.keys, keys), dim=2)
            self.values = torch.cat((self.values, values), dim=2)
        return self.keys, self.values


def concat_cache(num_layers: int) -> KVCache:
    cache = KVCache(num_layers)
    cache.layers = [ConcatLayerCache() for _ in range(num_layers)]  # type: ignore[misc]
    return cache


@dataclass(frozen=True)
class LatencyResult:
    cache: str
    prompt_tokens: int
    prefill_ms: float
    first_token_ms: float
    decode_ms_per_token: float
    tokens_per_second: float
    peak_device_gb: float


@torch.inference_mode()
def measure_once(
    engine: InferenceEngine, prompt_tokens: int, new_tokens: int, cache: KVCache
) -> tuple[float, float, float]:
    model, device = engine.model, engine.device
    prompt = torch.full((1, prompt_tokens), PROMPT_TOKEN_ID, device=device)
    synchronize(device)
    started = time.perf_counter()
    logits = model(prompt, cache=cache)[0, -1]
    synchronize(device)
    prefill = time.perf_counter() - started
    token = int(logits.argmax().item())
    first = time.perf_counter() - started
    decode_started = time.perf_counter()
    for _ in range(new_tokens - 1):
        logits = model(torch.tensor([[token]], device=device), cache=cache)[0, -1]
        token = int(logits.argmax().item())
    decode = (time.perf_counter() - decode_started) / max(new_tokens - 1, 1)
    return prefill, first, decode


def measure_latency(
    engine: InferenceEngine,
    prompt_tokens: int,
    new_tokens: int,
    repeats: int,
    cache_kind: str,
) -> LatencyResult:
    layers = engine.model.config.num_hidden_layers
    make = {
        "preallocated": engine.model.new_cache,
        "concat": lambda: concat_cache(layers),
    }[cache_kind]
    runs = [
        measure_once(engine, prompt_tokens, new_tokens, make())
        for _ in range(WARMUP_RUNS + repeats)
    ][WARMUP_RUNS:]
    prefill, first, decode = (statistics.median(values) for values in zip(*runs, strict=True))
    return LatencyResult(
        cache_kind,
        prompt_tokens,
        prefill * 1000,
        first * 1000,
        decode * 1000,
        1.0 / decode,
        peak_device_gb(engine.device),
    )
