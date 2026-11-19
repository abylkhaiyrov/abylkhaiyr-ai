from __future__ import annotations

import json
import math
import time
from pathlib import Path

import torch
from torch.utils.tensorboard import SummaryWriter

from llm.optimization.memory import device_memory_used_gb, ram_used_gb

METRICS_FILE = "metrics.jsonl"
PERPLEXITY_CAP = 20.0


def perplexity(loss: float) -> float:
    return math.exp(min(loss, PERPLEXITY_CAP))


class TrainingMonitor:
    def __init__(self, directory: str | Path, device: torch.device, total_steps: int) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(str(self.directory / "tensorboard"))
        self.metrics_path = self.directory / METRICS_FILE
        self.device = device
        self.total_steps = total_steps
        self.started = time.monotonic()
        self.start_step = 0

    def begin(self, step: int) -> None:
        self.start_step = step
        self.started = time.monotonic()

    def system_metrics(self) -> dict[str, float]:
        elapsed = time.monotonic() - self.started
        return {
            "ram_gb": ram_used_gb(),
            "device_memory_gb": device_memory_used_gb(self.device),
            "elapsed_seconds": elapsed,
        }

    def eta_seconds(self, step: int) -> float:
        completed = step - self.start_step
        if completed <= 0:
            return float("nan")
        elapsed = time.monotonic() - self.started
        return elapsed / completed * (self.total_steps - step)

    def log(self, step: int, values: dict[str, float]) -> None:
        for name, value in values.items():
            if math.isfinite(value):
                self.writer.add_scalar(name, value, step)
        record = {"step": step, **values}
        with self.metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")

    def close(self) -> None:
        self.writer.flush()
        self.writer.close()
