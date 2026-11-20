from __future__ import annotations

import signal
import subprocess
import sys
import threading
import time
import uuid
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

import psutil
from sqlalchemy import select

from llm.core.config import load_experiment_config
from llm.studio.models import ManagedProcess, ProcessEvent

RESOURCE_INTENSIVE_KINDS = {"pretrain", "sft", "dpo", "evaluation", "sweep"}
RUNNING_STATUSES = {"pending", "running", "pausing", "paused"}
LIST_TTL_SECONDS = 2.0


class ProcessError(Exception):
    pass


class ProcessManager:
    def __init__(self, session_factory, root: Path) -> None:
        self.session_factory = session_factory
        self.root = root.resolve()
        self._cache: tuple[float, list[dict]] = (0.0, [])
        self._cache_lock = threading.Lock()

    def invalidate(self) -> None:
        with self._cache_lock:
            self._cache = (0.0, [])

    def list(self) -> list[dict]:
        with self._cache_lock:
            stamp, cached = self._cache
            if time.monotonic() - stamp < LIST_TTL_SECONDS:
                return [dict(item) for item in cached]
            self._refresh_managed_processes()
            self._discover_cli_processes()
            with self.session_factory() as database:
                rows = database.scalars(
                    select(ManagedProcess).order_by(ManagedProcess.started_at.desc())
                ).all()
                serialized = [self._serialize(row) for row in rows]
            self._cache = (time.monotonic(), serialized)
            return [dict(item) for item in serialized]

    def start_training(self, config_path: str, stage: str | None = None) -> dict:
        config_file = self._safe_config_path(config_path)
        config = load_experiment_config(config_file)
        requested_kind = stage or config.train.stage
        if requested_kind not in {"pretrain", "sft", "dpo"}:
            raise ProcessError("Unsupported training stage")
        self._ensure_capacity(requested_kind)
        command = [sys.executable, "-m", "llm.cli.main", "train", "--config", str(config_file)]
        if stage is not None:
            command.extend(["--stage", stage])
        process = subprocess.Popen(
            command,
            cwd=self.root,
            start_new_session=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        started_at = psutil.Process(process.pid).create_time()
        record = ManagedProcess(
            id=str(uuid.uuid4()),
            pid=process.pid,
            pid_started_at=started_at,
            kind=requested_kind,
            status="running",
            command=" ".join(command),
            config_path=str(config_file.relative_to(self.root)),
            output_dir=config.train.output_dir,
        )
        with self.session_factory.begin() as database:
            database.add(record)
            database.add(
                ProcessEvent(process_id=record.id, level="info", message="Process started")
            )
        self.invalidate()
        return self._serialize(record)

    def pause(self, process_id: str) -> dict:
        return self._send_signal(
            process_id, signal.SIGINT, "paused", "Checkpoint requested before pause"
        )

    def stop(self, process_id: str) -> dict:
        return self._send_signal(process_id, signal.SIGTERM, "stopped", "Graceful stop requested")

    def _send_signal(
        self, process_id: str, signum: signal.Signals, status: str, message: str
    ) -> dict:
        with self.session_factory.begin() as database:
            record = database.get(ManagedProcess, process_id)
            if record is None:
                raise ProcessError("Process not found")
            process = self._owned_process(record)
            if process is None:
                raise ProcessError("The managed process is no longer running")
            process.send_signal(signum)
            self.invalidate()
            record.status = "pausing" if status == "paused" else status
            database.add(ProcessEvent(process_id=record.id, level="info", message=message))
            database.flush()
            return self._serialize(record)

    def _ensure_capacity(self, kind: str) -> None:
        if kind not in RESOURCE_INTENSIVE_KINDS:
            return
        with self.session_factory() as database:
            active = database.scalars(
                select(ManagedProcess).where(
                    ManagedProcess.kind.in_(RESOURCE_INTENSIVE_KINDS),
                    ManagedProcess.status.in_(RUNNING_STATUSES),
                )
            ).all()
        if any(self._owned_process(item) is not None for item in active):
            raise ProcessError("Another resource-intensive Studio process is already active")

    def _refresh_managed_processes(self) -> None:
        with self.session_factory.begin() as database:
            rows = database.scalars(
                select(ManagedProcess).where(ManagedProcess.status.in_(RUNNING_STATUSES))
            ).all()
            for row in rows:
                process = self._owned_process(row)
                if process is None:
                    row.status = "completed" if row.status == "pausing" else "failed"
                    row.completed_at = datetime.now(UTC).replace(tzinfo=None)
                    row.error_reason = (
                        None if row.status == "completed" else "Process is no longer available"
                    )
                    database.add(
                        ProcessEvent(
                            process_id=row.id,
                            level="info" if row.status == "completed" else "error",
                            message="Process ended",
                        )
                    )

    def _discover_cli_processes(self) -> None:
        with self.session_factory.begin() as database:
            known = {
                pid for pid in database.scalars(select(ManagedProcess.pid)).all() if pid is not None
            }
            for process in psutil.process_iter(["pid", "create_time", "cmdline"]):
                details = process.info
                command = details.get("cmdline") or []
                if details["pid"] in known or not self._is_project_cli(command):
                    continue
                kind = self._kind_from_command(command)
                record = ManagedProcess(
                    id=str(uuid.uuid4()),
                    pid=details["pid"],
                    pid_started_at=details["create_time"],
                    kind=kind,
                    status="running",
                    command=" ".join(command),
                )
                database.add(record)
                database.add(
                    ProcessEvent(
                        process_id=record.id,
                        level="info",
                        message="Existing project CLI process discovered",
                    )
                )

    def _owned_process(self, record: ManagedProcess) -> psutil.Process | None:
        if record.pid is None:
            return None
        try:
            process = psutil.Process(record.pid)
            if (
                record.pid_started_at is not None
                and abs(process.create_time() - record.pid_started_at) > 0.1
            ):
                return None
            if not self._is_project_cli(process.cmdline()):
                return None
            return process
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return None

    def _safe_config_path(self, value: str) -> Path:
        path = (self.root / value).resolve()
        configs = (self.root / "configs").resolve()
        if configs not in path.parents or path.suffix not in {".yaml", ".yml"}:
            raise ProcessError("Configuration must be a YAML file inside configs/")
        if not path.is_file():
            raise ProcessError("Configuration file does not exist")
        return path

    def _is_project_cli(self, command: Sequence[str]) -> bool:
        joined = " ".join(command)
        return "llm.cli.main" in joined and any(
            token in command
            for token in (
                "train",
                "resume",
                "finetune",
                "evaluate",
                "prepare-data",
                "train-tokenizer",
            )
        )

    @staticmethod
    def _kind_from_command(command: Sequence[str]) -> str:
        for kind in ("train", "resume", "finetune", "evaluate", "prepare-data", "train-tokenizer"):
            if kind in command:
                return {
                    "train": "pretrain",
                    "resume": "pretrain",
                    "finetune": "sft",
                    "prepare-data": "dataset",
                    "train-tokenizer": "tokenizer",
                }.get(kind, kind)
        return "unknown"

    @staticmethod
    def _serialize(record: ManagedProcess) -> dict:
        cpu_percent = None
        memory_bytes = None
        if record.pid is not None:
            try:
                process = psutil.Process(record.pid)
                cpu_percent = process.cpu_percent(interval=None)
                memory_bytes = process.memory_info().rss
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        return {
            "id": record.id,
            "pid": record.pid,
            "kind": record.kind,
            "status": record.status,
            "command": record.command,
            "config_path": record.config_path,
            "output_dir": record.output_dir,
            "started_at": record.started_at.isoformat(),
            "completed_at": (
                None if record.completed_at is None else record.completed_at.isoformat()
            ),
            "exit_code": record.exit_code,
            "error_reason": record.error_reason,
            "cpu_percent": cpu_percent,
            "memory_bytes": memory_bytes,
        }
