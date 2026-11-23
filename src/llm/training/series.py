from __future__ import annotations

import signal
import subprocess
import time
from collections.abc import Callable, Sequence
from threading import Event

from llm.core.errors import ResourceError
from llm.training.session import graceful_shutdown


def run_series(
    jobs: Sequence[str],
    deadline: float,
    command: Callable[[str], list[str]],
    completed: Callable[[str], bool],
) -> None:
    stop = Event()
    with graceful_shutdown(stop):
        for job in jobs:
            if completed(job):
                continue
            if stop.is_set() or time.time() >= deadline:
                return
            arguments = command(job)
            with subprocess.Popen(arguments) as child:
                while child.poll() is None:
                    if stop.wait(0.5):
                        child.send_signal(signal.SIGTERM)
                        child.wait()
                        return
                if child.returncode:
                    raise subprocess.CalledProcessError(child.returncode, arguments)
            if not completed(job):
                raise ResourceError(f"job {job} stopped before the comparison point")
