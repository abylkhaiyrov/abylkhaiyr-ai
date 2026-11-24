from __future__ import annotations

import signal
import subprocess
from collections.abc import Callable, Sequence
from pathlib import Path
from threading import Event

from llm.core.errors import ResourceError
from llm.training.budget import BudgetClock
from llm.training.session import graceful_shutdown

POLL_SECONDS = 0.5
LEDGER_SAVE_SECONDS = 5.0


def run_series(
    jobs: Sequence[str],
    clock: BudgetClock,
    command: Callable[[str, float], list[str]],
    completed: Callable[[str], bool],
    healthy: Callable[[str], bool] = lambda job: True,
    ledger_path: Path | None = None,
    poll_seconds: float = POLL_SECONDS,
) -> str | None:
    stop = Event()
    saved_at = 0.0

    def persist() -> None:
        nonlocal saved_at
        if (
            ledger_path is not None
            and clock.ledger.runtime_seconds - saved_at >= LEDGER_SAVE_SECONDS
        ):
            clock.ledger.save(ledger_path)
            saved_at = clock.ledger.runtime_seconds

    with graceful_shutdown(stop):
        for job in jobs:
            if completed(job):
                continue
            clock.tick()
            if stop.is_set() or clock.exhausted():
                return clock.exhausted() or "stopped"
            reason = launch(job, clock, command, completed, healthy, stop, persist, poll_seconds)
            if ledger_path is not None:
                clock.ledger.save(ledger_path)
            if reason is not None:
                return reason
    return None


def launch(
    job: str,
    clock: BudgetClock,
    command: Callable[[str, float], list[str]],
    completed: Callable[[str], bool],
    healthy: Callable[[str], bool],
    stop: Event,
    persist: Callable[[], None],
    poll_seconds: float,
) -> str | None:
    while True:
        arguments = command(job, clock.remaining_active)
        with subprocess.Popen(arguments) as child:
            outcome = supervise(child, job, clock, healthy, stop, persist, poll_seconds)
        if outcome == "finished":
            break
        if outcome != "resource":
            return outcome
        if not wait_for_resources(job, clock, healthy, stop, poll_seconds):
            return "resources unavailable after the wait limit"
    if child.returncode:
        raise subprocess.CalledProcessError(child.returncode, arguments)
    if not completed(job):
        raise ResourceError(f"job {job} stopped before the comparison point")
    return None


def supervise(
    child: subprocess.Popen[bytes],
    job: str,
    clock: BudgetClock,
    healthy: Callable[[str], bool],
    stop: Event,
    persist: Callable[[], None],
    poll_seconds: float,
) -> str:
    while child.poll() is None:
        if stop.wait(poll_seconds):
            terminate(child)
            return "stopped"
        gap = clock.tick()
        persist()
        reason = clock.exhausted()
        if reason is not None:
            terminate(child)
            return reason
        if gap and not healthy(job):
            terminate(child)
            return "resource"
    return "finished"


def terminate(child: subprocess.Popen[bytes]) -> None:
    child.send_signal(signal.SIGTERM)
    child.wait()


def wait_for_resources(
    job: str,
    clock: BudgetClock,
    healthy: Callable[[str], bool],
    stop: Event,
    poll_seconds: float,
) -> bool:
    with clock.waiting():
        waited = 0.0
        while not healthy(job):
            if (
                stop.wait(max(poll_seconds, 1.0))
                or waited >= clock.limits.max_resource_wait_seconds
            ):
                return False
            waited += max(poll_seconds, 1.0)
            clock.tick()
    return True
