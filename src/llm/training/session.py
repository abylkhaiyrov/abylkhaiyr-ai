from __future__ import annotations

import logging
import os
import signal
import sys
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from types import FrameType
from typing import IO

if sys.platform == "win32":
    import msvcrt
else:
    import fcntl

from llm.core.config import ExperimentConfig
from llm.core.errors import LLMError, ResourceError
from llm.evaluation.registry import REGISTRY_FILE, record_experiment, training_record
from llm.training.engine import TrainingResult
from llm.training.runner import build_trainer

LOCK_FILE = Path("experiments/.training.lock")
LOG_FILE_NAME = "train.log"
HANDLED_SIGNALS = (signal.SIGINT, signal.SIGTERM)
RESTART_BACKOFF_SECONDS = 30.0
LOG_FORMAT = "%(asctime)s %(levelname)s %(message)s"

logger = logging.getLogger("llm.training")


@contextmanager
def exclusive_session(lock_path: Path = LOCK_FILE) -> Iterator[None]:
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    handle: IO[str] = lock_path.open("a+")
    try:
        try:
            if sys.platform == "win32":
                handle.seek(0, os.SEEK_END)
                if handle.tell() == 0:
                    handle.write(" ")
                    handle.flush()
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            raise ResourceError("another training session is already running") from error
        handle.seek(0)
        handle.truncate()
        handle.write(str(time.time()))
        handle.flush()
        yield
    finally:
        handle.close()


@contextmanager
def graceful_shutdown(stop_event: threading.Event) -> Iterator[None]:
    def request_stop(signum: int, frame: FrameType | None) -> None:
        logger.warning("signal %s received, saving state after the current step", signum)
        stop_event.set()

    previous = {sig: signal.signal(sig, request_stop) for sig in HANDLED_SIGNALS}
    try:
        yield
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


def configure_logging(output_dir: str) -> None:
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    for handler in (logging.FileHandler(Path(output_dir) / LOG_FILE_NAME), logging.StreamHandler()):
        handler.setFormatter(logging.Formatter(LOG_FORMAT))
        logger.addHandler(handler)


def run_training(
    config: ExperimentConfig,
    resume: bool,
    stop_step: int | None = None,
    max_restarts: int = 0,
    backoff_seconds: float = RESTART_BACKOFF_SECONDS,
    stop_file: Path | None = None,
) -> TrainingResult:
    configure_logging(config.train.output_dir)
    attempt = 0
    with exclusive_session():
        while True:
            trainer = build_trainer(config, resume or attempt > 0)
            trainer.stop_file = stop_file
            with graceful_shutdown(trainer.stop_event):
                try:
                    logger.info("starting at step %d", trainer.state.step)
                    result = trainer.train(stop_step=stop_step)
                    registry = Path(config.train.output_dir).parent / REGISTRY_FILE.name
                    record_experiment(config.name, *training_record(config, result), path=registry)
                    return result
                except LLMError as error:
                    if attempt >= max_restarts or not isinstance(error, ResourceError):
                        raise
                    logger.error("resource failure: %s; restarting from checkpoint", error)
            attempt += 1
            time.sleep(backoff_seconds)
