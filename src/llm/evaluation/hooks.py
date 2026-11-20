from __future__ import annotations

import time
from typing import Any, Protocol

from llm.progress import StageHandle

MAX_INTERIM_METRICS = 64


class EvaluationHooks(Protocol):
    def suite_started(self, name: str, index: int, total: int) -> None: ...
    def items_planned(self, total: int) -> None: ...
    def item_done(self, count: int = 1) -> None: ...
    def suite_finished(self, name: str, metrics: dict[str, float], seconds: float) -> None: ...


class NullHooks:
    def suite_started(self, name: str, index: int, total: int) -> None:
        return None

    def items_planned(self, total: int) -> None:
        return None

    def item_done(self, count: int = 1) -> None:
        return None

    def suite_finished(self, name: str, metrics: dict[str, float], seconds: float) -> None:
        return None


class StageEvaluationHooks:
    def __init__(
        self, handle: StageHandle, checkpoint_total: int, suites_per_checkpoint: int
    ) -> None:
        self.handle = handle
        self.checkpoint_total = checkpoint_total
        self.suites_per_checkpoint = suites_per_checkpoint
        self.checkpoint_name = ""
        self.checkpoint_index = 0
        self.suite_name = ""
        self.suite_index = 0
        self.suite_total = suites_per_checkpoint
        self.items_total: int | None = None
        self.items_done = 0
        self.tasks_done = 0
        self.checkpoint_started = time.monotonic()
        self.checkpoint_durations: list[float] = []
        self.interim: dict[str, dict[str, float]] = {}
        self.stage_started = time.monotonic()
        handle.set_manual_eta()

    def begin_checkpoint(self, name: str, index: int) -> None:
        self.checkpoint_name = name
        self.checkpoint_index = index
        self.checkpoint_started = time.monotonic()
        self.interim.setdefault(name, {})
        self._publish()

    def end_checkpoint(self) -> None:
        self.checkpoint_durations.append(time.monotonic() - self.checkpoint_started)
        remaining = self.checkpoint_total - len(self.checkpoint_durations)
        average = sum(self.checkpoint_durations) / len(self.checkpoint_durations)
        self.handle.set_eta(average * remaining if remaining > 0 else 0.0)
        self._publish()

    def suite_started(self, name: str, index: int, total: int) -> None:
        self.suite_name = name
        self.suite_index = index
        self.suite_total = total
        self.items_total = None
        self.items_done = 0
        self._publish()

    def items_planned(self, total: int) -> None:
        self.items_total = total
        self._publish()

    def item_done(self, count: int = 1) -> None:
        self.items_done += count
        self.tasks_done += count
        self._publish()

    def suite_finished(self, name: str, metrics: dict[str, float], seconds: float) -> None:
        bucket = self.interim.setdefault(self.checkpoint_name, {})
        bucket.update(metrics)
        self.handle.set_done(
            self.checkpoint_index * self.suites_per_checkpoint + self.suite_index + 1,
            **self._detail(),
            interim=self.interim,
            last_benchmark_seconds=seconds,
        )

    def _detail(self) -> dict[str, Any]:
        elapsed = max(time.monotonic() - self.stage_started, 1e-9)
        remaining = None if self.items_total is None else max(self.items_total - self.items_done, 0)
        return {
            "checkpoint": self.checkpoint_name,
            "checkpoint_index": self.checkpoint_index + 1,
            "checkpoint_total": self.checkpoint_total,
            "benchmark": self.suite_name,
            "benchmark_index": self.suite_index + 1,
            "benchmark_total": self.suite_total,
            "items_done": self.items_done,
            "items_total": self.items_total,
            "items_remaining": remaining,
            "tasks_done": self.tasks_done,
            "tasks_per_second": self.tasks_done / elapsed,
        }

    def _publish(self) -> None:
        self.handle.update(**self._detail())
