from __future__ import annotations

import shutil
import statistics
import subprocess
import threading
from dataclasses import dataclass
from typing import Self

QUERY = ["--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits"]
POLL_SECONDS = 0.5


@dataclass(frozen=True)
class GpuUsage:
    mean_utilization: float
    max_utilization: float
    max_memory_mb: float
    samples: int


def parse_sample(line: str) -> tuple[float, float] | None:
    parts = [part.strip() for part in line.split(",")]
    if len(parts) != 2:
        return None
    try:
        return float(parts[0]), float(parts[1])
    except ValueError:
        return None


class GpuSampler:
    def __init__(self) -> None:
        self.executable = shutil.which("nvidia-smi")
        self.readings: list[tuple[float, float]] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._poll, daemon=True)

    def _poll(self) -> None:
        assert self.executable is not None
        while not self._stop.wait(POLL_SECONDS):
            output = subprocess.run(
                [self.executable, *QUERY], capture_output=True, text=True, check=False, timeout=5
            ).stdout
            reading = parse_sample(output.strip().splitlines()[0]) if output.strip() else None
            if reading is not None:
                self.readings.append(reading)

    def __enter__(self) -> Self:
        if self.executable is not None:
            self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join()

    def usage(self) -> GpuUsage | None:
        if not self.readings:
            return None
        utilization = [u for u, _ in self.readings]
        return GpuUsage(
            statistics.fmean(utilization),
            max(utilization),
            max(m for _, m in self.readings),
            len(self.readings),
        )
