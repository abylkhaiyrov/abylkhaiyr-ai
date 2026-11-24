from __future__ import annotations

import json
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path

from llm.core.config import StrictModel
from llm.core.errors import ResourceError

Clock = Callable[[], float]
RESUME_SLACK_SECONDS = 2.0
LEDGER_SUFFIX = ".budget.json"


class BudgetLimits(StrictModel):
    active_seconds: float
    absolute_seconds: float
    max_resource_wait_seconds: float = 600.0
    max_gap_seconds: float = 30.0


@dataclass
class TimeLedger:
    active_seconds: float = 0.0
    idle_seconds: float = 0.0
    wait_seconds: float = 0.0
    runtime_seconds: float = 0.0
    started_at: float = 0.0
    wakeups: int = 0

    def wall_seconds(self, now: float) -> float:
        return now - self.started_at

    @staticmethod
    def load(path: Path) -> TimeLedger | None:
        if not path.exists():
            return None
        return TimeLedger(**json.loads(path.read_text(encoding="utf-8")))

    def save(self, path: Path) -> None:
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(asdict(self), indent=2) + "\n", encoding="utf-8")
        temporary.replace(path)


class BudgetClock:
    def __init__(
        self,
        limits: BudgetLimits,
        ledger: TimeLedger,
        monotonic: Clock = time.monotonic,
        wall: Clock = time.time,
    ) -> None:
        self.limits = limits
        self.ledger = ledger
        self._monotonic = monotonic
        self._wall = wall
        self._last_monotonic = monotonic()
        self._last_wall = wall()
        self._waiting = False
        if not ledger.started_at:
            ledger.started_at = self._last_wall

    @contextmanager
    def waiting(self) -> Iterator[None]:
        self.tick()
        self._waiting = True
        try:
            yield
        finally:
            self.tick()
            self._waiting = False

    def tick(self) -> float:
        now_monotonic, now_wall = self._monotonic(), self._wall()
        elapsed = max(now_monotonic - self._last_monotonic, 0.0)
        wall = max(now_wall - self._last_wall, 0.0)
        self._last_monotonic, self._last_wall = now_monotonic, now_wall
        self.ledger.runtime_seconds += elapsed
        suspended = max(wall - elapsed, 0.0)
        if suspended > RESUME_SLACK_SECONDS:
            self.ledger.idle_seconds += suspended
        if elapsed > self.limits.max_gap_seconds:
            self.ledger.idle_seconds += elapsed
            gap = max(suspended, elapsed)
        elif self._waiting:
            self.ledger.wait_seconds += elapsed
            gap = suspended if suspended > RESUME_SLACK_SECONDS else 0.0
        else:
            self.ledger.active_seconds += elapsed
            gap = suspended if suspended > RESUME_SLACK_SECONDS else 0.0
        if gap:
            self.ledger.wakeups += 1
        return gap

    @property
    def remaining_active(self) -> float:
        return max(self.limits.active_seconds - self.ledger.active_seconds, 0.0)

    def exhausted(self) -> str | None:
        if self.ledger.active_seconds >= self.limits.active_seconds:
            return "active budget exhausted"
        if self.ledger.wall_seconds(self._wall()) >= self.limits.absolute_seconds:
            return "absolute wall-clock guard reached"
        return None

    def check(self) -> None:
        reason = self.exhausted()
        if reason is not None:
            raise ResourceError(reason)
