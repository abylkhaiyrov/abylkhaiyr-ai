from __future__ import annotations

import json
import re
import threading
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import psutil

from llm.core.config import load_experiment_config
from llm.core.errors import ConfigError
from llm.studio.pipeline_view import (
    FINISHED,
    StageConfig,
    build_pipeline_view,
    overall_progress,
    parse_time,
)
from llm.training.checkpoint import CheckpointManager

PROCESS_SCAN_TTL = 3.0
SNAPSHOT_TTL = 1.0
TRAIN_KEYS = ("pretrain", "sft", "dpo")


class RuntimeState:
    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self._lock = threading.RLock()
        self._scan: tuple[float, dict[str, Any] | None] = (0.0, None)
        self._snapshot: tuple[float, dict[str, Any]] = (0.0, {})
        self._processes: dict[int, psutil.Process] = {}
        self._config_cache: dict[tuple[str, float], StageConfig | None] = {}
        self._checkpoint_cache: dict[Path, tuple[tuple[tuple[str, int], ...], Path | None]] = {}

    def invalidate(self) -> None:
        with self._lock:
            self._scan = (0.0, None)
            self._snapshot = (0.0, {})

    def training_active(self) -> bool:
        return self._running_process() is not None

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            stamp, cached = self._snapshot
            if cached and time.monotonic() - stamp < SNAPSHOT_TTL:
                return cached
            value = self._build_snapshot()
            self._snapshot = (time.monotonic(), value)
            return value

    def pipeline(self) -> dict[str, Any] | None:
        return self.snapshot()["pipeline"]

    def active_pipeline(self) -> dict[str, Any] | None:
        pipeline = self.pipeline()
        return pipeline if pipeline is not None and pipeline["running"] else None

    def latest_checkpoint(self, directory: Path) -> Path | None:
        metadata = tuple(
            sorted(
                (str(path), path.stat().st_mtime_ns) for path in directory.glob("step-*/meta.json")
            )
        )
        cached = self._checkpoint_cache.get(directory)
        if cached is None or cached[0] != metadata:
            checkpoint = CheckpointManager(directory, keep=1).latest_valid()
            self._checkpoint_cache[directory] = (metadata, checkpoint)
            return checkpoint
        return cached[1]

    def _build_snapshot(self) -> dict[str, Any]:
        now = datetime.now(UTC)
        running = self._running_process()
        state_dir = self._state_dir(running)
        pipeline = None if state_dir is None else self._pipeline_view(state_dir, running, now)
        return {"generated_at": now.isoformat(), "pipeline": pipeline}

    def _pipeline_view(
        self, state_dir: Path, running: dict[str, Any] | None, now: datetime
    ) -> dict[str, Any] | None:
        try:
            state = json.loads((state_dir / "state.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        command = running["command"] if running else []
        alive = running is not None
        configs = {key: self._stage_config(key, command, state_dir, state) for key in TRAIN_KEYS}
        stages = build_pipeline_view(self.root, state, state_dir, configs, alive, now)
        current = next((item for item in stages if item["status"] == "running"), None)
        pending = [item for item in stages if item["status"] == "pending"]
        current_key = current["key"] if current else None
        finished = [item for item in stages if item["status"] in FINISHED and item["finished_at"]]
        last = max(finished, key=lambda item: parse_time(item["finished_at"]) or now, default=None)
        pretrain_config = configs["pretrain"]
        return {
            "id": state_dir.name,
            "state_dir": str(state_dir.relative_to(self.root)),
            "running": alive,
            "pid": running["pid"] if running else None,
            "memory_bytes": running["memory_bytes"] if running else None,
            "cpu_percent": running["cpu_percent"] if running else None,
            "model_name": self._model_name(state_dir, pretrain_config),
            "started_at": state.get("created_at"),
            "updated_at": state.get("updated_at"),
            "stages": stages,
            "current_stage": current_key,
            "next_stage": pending[0]["key"] if pending else None,
            "last_result_stage": last["key"] if last else None,
            "overall": overall_progress(stages, alive, now),
            "artifacts": self._artifacts(state),
            "checkpoint": self._current_checkpoint(configs, current_key),
        }

    def _artifacts(self, state: dict[str, Any]) -> dict[str, Any]:
        artifacts = state.get("artifacts", {})
        keys = (
            "pretrain_alias",
            "best_pretrain_checkpoint",
            "sft_checkpoint",
            "dpo_checkpoint",
            "final_checkpoint",
            "release_path",
        )
        return {key: artifacts[key] for key in keys if artifacts.get(key)}

    def _current_checkpoint(
        self, configs: dict[str, StageConfig | None], current: str | None
    ) -> str | None:
        config = configs.get(current) if current in TRAIN_KEYS else None
        if config is None:
            return None
        checkpoint = self.latest_checkpoint(self.root / config.checkpoint_dir)
        return str(checkpoint.relative_to(self.root)) if checkpoint else None

    def _model_name(self, state_dir: Path, pretrain: StageConfig | None) -> str:
        if pretrain is not None:
            return Path(pretrain.output_dir).name
        return re.sub(r"^pipeline_", "", state_dir.name)

    def _stage_config(
        self, key: str, command: list[str], state_dir: Path, state: dict[str, Any]
    ) -> StageConfig | None:
        path = self._config_path(key, command, state_dir, state)
        if path is None or not path.is_file():
            return None
        signature = (str(path), path.stat().st_mtime)
        if signature not in self._config_cache:
            try:
                config = load_experiment_config(path)
                self._config_cache[signature] = StageConfig(
                    config.train.output_dir, config.train.checkpoint_dir, config.train.max_steps
                )
            except ConfigError:
                self._config_cache[signature] = None
        return self._config_cache[signature]

    def _config_path(
        self, key: str, command: list[str], state_dir: Path, state: dict[str, Any]
    ) -> Path | None:
        recorded = state.get("config", {}).get(f"{key}_config")
        if recorded:
            return self.root / recorded
        flag = f"--{key}"
        if flag in command and command.index(flag) + 1 < len(command):
            return self.root / command[command.index(flag) + 1]
        suffix = re.sub(r"^pipeline_", "", state_dir.name)
        stage_suffix = "" if key == "pretrain" else f"_{key}"
        return self.root / "configs" / f"{suffix}{stage_suffix}.yaml"

    def _state_dir(self, running: dict[str, Any] | None) -> Path | None:
        if running is not None:
            command = running["command"]
            if "--state-dir" in command:
                index = command.index("--state-dir") + 1
                if index < len(command):
                    candidate = (self.root / command[index]).resolve()
                    if self.root in candidate.parents:
                        return candidate
        states = sorted(
            (self.root / "experiments").glob("*/state.json"), key=lambda item: item.stat().st_mtime
        )
        return states[-1].parent if states else None

    def _running_process(self) -> dict[str, Any] | None:
        with self._lock:
            stamp, cached = self._scan
            if time.monotonic() - stamp < PROCESS_SCAN_TTL:
                return cached
            found = self._scan_processes()
            self._scan = (time.monotonic(), found)
            return found

    def _scan_processes(self) -> dict[str, Any] | None:
        script = (self.root / "scripts" / "run_pipeline.py").resolve()
        for process in psutil.process_iter(["pid", "cmdline", "create_time"]):
            try:
                command = process.info["cmdline"] or []
                if not any((self.root / arg).resolve() == script for arg in command[1:]):
                    continue
                if Path(process.cwd()).resolve() != self.root:
                    continue
                tracked = self._processes.setdefault(process.pid, process)
                return {
                    "pid": process.pid,
                    "command": command,
                    "create_time": process.info["create_time"],
                    "memory_bytes": tracked.memory_info().rss,
                    "cpu_percent": tracked.cpu_percent(interval=None),
                }
            except (psutil.Error, OSError, ValueError):
                continue
        self._processes.clear()
        return None
