from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import psutil

from llm.core.config import load_experiment_config
from llm.core.errors import ConfigError
from llm.studio.telemetry import read_metrics
from llm.training.checkpoint import CheckpointManager


class RuntimeState:
    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self._checkpoint_cache: dict[Path, tuple[tuple[tuple[str, int], ...], Path | None]] = {}

    def pipeline(self) -> dict[str, Any] | None:
        for process in psutil.process_iter(["pid", "cmdline", "create_time"]):
            try:
                command = process.info["cmdline"] or []
                script = (self.root / "scripts" / "run_pipeline.py").resolve()
                if not any((self.root / arg).resolve() == script for arg in command[1:]):
                    continue
                if Path(process.cwd()).resolve() != self.root:
                    continue
                state_dir = self._argument(command, "--state-dir", "experiments/pipeline_tiny_v2")
                state_path = (self.root / state_dir / "state.json").resolve()
                if self.root not in state_path.parents:
                    continue
                state = json.loads(state_path.read_text(encoding="utf-8"))
                stages = state.get("stages", {})
                stage = next(
                    (name for name, item in stages.items() if item.get("status") == "running"), None
                )
                config_key = stage if stage in {"pretrain", "sft", "dpo"} else "pretrain"
                config_path = self._argument(
                    command,
                    f"--{config_key}",
                    f"configs/tiny_v2{'' if config_key == 'pretrain' else '_' + config_key}.yaml",
                )
                config = load_experiment_config(self.root / config_path)
                metrics: dict[str, float] = {}
                if stage in {"pretrain", "sft", "dpo"}:
                    for record in read_metrics(
                        self.root / config.train.output_dir / "metrics.jsonl"
                    ):
                        metrics.update(record)
                checkpoint = self._latest_checkpoint(self.root / config.train.checkpoint_dir)
                return {
                    "id": f"pipeline-{process.pid}-{process.info['create_time']}",
                    "pid": process.pid,
                    "kind": stage or "pipeline",
                    "status": "running",
                    "started_at": datetime.fromtimestamp(
                        process.info["create_time"], UTC
                    ).isoformat(),
                    "config_path": config_path,
                    "output_dir": config.train.output_dir,
                    "model_name": config.name,
                    "max_steps": config.train.max_steps
                    if stage in {"pretrain", "sft", "dpo"}
                    else None,
                    "metrics": metrics,
                    "checkpoint": str(checkpoint) if checkpoint else None,
                    "stages": stages,
                    "artifacts": state.get("artifacts", {}),
                    "memory_bytes": process.memory_info().rss,
                }
            except (psutil.Error, OSError, ValueError, TypeError, ConfigError):
                continue
        return None

    def _latest_checkpoint(self, directory: Path) -> Path | None:
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

    @staticmethod
    def _argument(command: list[str], name: str, default: str) -> str:
        try:
            return command[command.index(name) + 1]
        except (ValueError, IndexError):
            return default
