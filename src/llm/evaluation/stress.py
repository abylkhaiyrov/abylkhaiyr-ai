from __future__ import annotations

import gc
from dataclasses import dataclass

import psutil
import torch

from llm.inference.engine import InferenceEngine
from llm.inference.sampling import SamplingParams
from llm.optimization.benchmark import peak_device_gb
from llm.tokenizer.bpe import Message

BYTES_PER_MB = 1024**2
WORDS = ("alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta")


@dataclass(frozen=True)
class MemorySample:
    request: int
    rss_mb: float
    device_gb: float
    live_tensors: int


def live_tensor_count() -> int:
    gc.collect()
    return sum(1 for obj in gc.get_objects() if isinstance(obj, torch.Tensor))


def prompt_of(length: int, seed: int) -> str:
    return " ".join(WORDS[(seed + i) % len(WORDS)] for i in range(length))


def inference_memory_profile(
    engine: InferenceEngine, requests: int, max_prompt_words: int, every: int
) -> list[MemorySample]:
    params = SamplingParams(max_new_tokens=24, temperature=0.8, top_p=0.9, seed=0)
    process = psutil.Process()
    samples: list[MemorySample] = []
    for index in range(1, requests + 1):
        words = 4 + (index * 37) % max_prompt_words
        engine.chat([Message("user", prompt_of(words, index))], params)
        if index % every == 0:
            samples.append(
                MemorySample(
                    index,
                    process.memory_info().rss / BYTES_PER_MB,
                    peak_device_gb(engine.device),
                    live_tensor_count(),
                )
            )
    return samples


def growth(samples: list[MemorySample], field: str) -> float:
    half = len(samples) // 2
    first = sum(getattr(s, field) for s in samples[:half]) / max(half, 1)
    second = sum(getattr(s, field) for s in samples[half:]) / max(len(samples) - half, 1)
    return second - first
