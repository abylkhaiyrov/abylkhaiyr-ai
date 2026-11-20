from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from llm.core.config import ExperimentConfig, load_experiment_config
from llm.core.errors import ConfigError
from llm.studio.pipeline_view import finite
from llm.studio.runtime import RuntimeState
from llm.studio.telemetry import read_metrics

MAX_LOG_LINES = 1000
LOG_TIME = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})")
SECRET_PATTERN = re.compile(r"(?i)(authorization|password|token|secret)(\s*[=:]\s*)([^\s,]+)")
SAFE_NAME = re.compile(r"^[A-Za-z0-9_.-]+$")
STAGE_BY_NAME = (("dpo", "dpo"), ("sft", "sft"))


@dataclass(frozen=True)
class RunConfig:
    path: Path
    config: ExperimentConfig


def downsample(rows: list[dict[str, float]], limit: int) -> list[dict[str, float]]:
    if len(rows) <= limit:
        return rows
    keep = {
        index for index, row in enumerate(rows) if any(key.startswith("validation/") for key in row)
    }
    keep.add(len(rows) - 1)
    stride = max(len(rows) // max(limit - len(keep), 1), 1)
    keep.update(range(0, len(rows), stride))
    return [rows[index] for index in sorted(keep)]


def merge_by_step(records: list[dict[str, float]]) -> list[dict[str, float]]:
    merged: dict[int, dict[str, float]] = {}
    for record in records:
        merged.setdefault(int(record["step"]), {"step": record["step"]}).update(record)
    return [merged[step] for step in sorted(merged)]


class ExperimentCatalog:
    def __init__(self, root: Path, runtime: RuntimeState) -> None:
        self.root = root.resolve()
        self.runtime = runtime
        self._configs: dict[str, RunConfig] = {}
        self._signature: tuple[tuple[str, int], ...] = ()

    def configs(self) -> dict[str, RunConfig]:
        paths = sorted((self.root / "configs").glob("*.yaml"))
        signature = tuple((str(path), path.stat().st_mtime_ns) for path in paths)
        if signature != self._signature:
            found: dict[str, RunConfig] = {}
            for path in paths:
                try:
                    config = load_experiment_config(path)
                except ConfigError:
                    continue
                found.setdefault(config.train.output_dir, RunConfig(path, config))
            self._configs, self._signature = found, signature
        return self._configs

    def runs(self) -> list[dict[str, Any]]:
        names = sorted(
            path.parent.name for path in (self.root / "experiments").glob("*/metrics.jsonl")
        )
        pipeline = self.runtime.pipeline()
        return sorted(
            (self._summary(name, pipeline) for name in names),
            key=lambda item: item["updated_at"] or "",
            reverse=True,
        )

    def detail(self, name: str) -> dict[str, Any] | None:
        directory = self._directory(name)
        if directory is None:
            return None
        summary = self._summary(name, self.runtime.pipeline())
        config = self.configs().get(f"experiments/{name}")
        return {
            **summary,
            "checkpoints": self.checkpoints(name),
            "config": None if config is None else self._config_payload(config),
            "evaluations": self._evaluations(name),
        }

    def metrics(self, name: str, max_points: int) -> list[dict[str, float]] | None:
        directory = self._directory(name)
        if directory is None:
            return None
        records = read_metrics(directory / "metrics.jsonl", 200000)
        return downsample(merge_by_step(records), max(max_points, 10))

    def logs(self, name: str, limit: int) -> list[str] | None:
        directory = self._directory(name)
        if directory is None:
            return None
        lines: list[str] = []
        for file_name in ("train.log", "run.log"):
            path = directory / file_name
            if path.is_file():
                lines.extend(path.read_text(encoding="utf-8", errors="replace").splitlines())
        return [
            SECRET_PATTERN.sub(r"\1\2[redacted]", line)
            for line in lines[-min(limit, MAX_LOG_LINES) :]
        ]

    def checkpoints(self, name: str) -> list[dict[str, Any]]:
        config = self.configs().get(f"experiments/{name}")
        if config is None:
            return []
        directory = self.root / config.config.train.checkpoint_dir
        results: list[dict[str, Any]] = []
        for meta_path in sorted(directory.glob("*/meta.json")):
            if meta_path.parent.is_symlink():
                continue
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            results.append(
                {
                    "id": str(meta_path.parent.relative_to(self.root)),
                    "label": meta_path.parent.name,
                    "step": meta.get("step"),
                    "tokens_seen": meta.get("tokens_seen"),
                    "created_at": datetime.fromtimestamp(
                        meta_path.stat().st_mtime, UTC
                    ).isoformat(),
                }
            )
        return sorted(results, key=lambda item: item["step"] or 0)

    def _directory(self, name: str) -> Path | None:
        if not SAFE_NAME.match(name):
            return None
        directory = self.root / "experiments" / name
        return directory if (directory / "metrics.jsonl").is_file() else None

    def _summary(self, name: str, pipeline: dict[str, Any] | None) -> dict[str, Any]:
        output_dir = f"experiments/{name}"
        run_config = self.configs().get(output_dir)
        latest: dict[str, float] = {}
        best: float | None = None
        first: dict[str, float] | None = None
        for record in read_metrics(self.root / output_dir / "metrics.jsonl", 200000):
            first = first or record
            latest.update(record)
            value = finite(record.get("validation/loss"))
            if value is not None:
                best = value if best is None else min(best, value)
        max_steps = None if run_config is None else run_config.config.train.max_steps
        step = int(latest["step"]) if "step" in latest else None
        stage = self._stage(name, run_config)
        metrics_file = self.root / output_dir / "metrics.jsonl"
        status = self._status(step, max_steps, pipeline, output_dir)
        return {
            "name": name,
            "output_dir": output_dir,
            "stage": stage,
            "status": status,
            "step": step,
            "max_steps": max_steps,
            "percent": None if not step or not max_steps else min(100.0, step / max_steps * 100),
            "train_loss": finite(latest.get("train/loss")),
            "validation_loss": finite(latest.get("validation/loss")),
            "validation_perplexity": finite(latest.get("validation/perplexity")),
            "best_validation_loss": best,
            "first_train_loss": None if first is None else finite(first.get("train/loss")),
            "tokens_seen": finite(latest.get("perf/tokens_seen")),
            "config_path": None
            if run_config is None
            else str(run_config.path.relative_to(self.root)),
            "started_at": self._started_at(self.root / output_dir),
            "updated_at": datetime.fromtimestamp(metrics_file.stat().st_mtime, UTC).isoformat(),
        }

    def _stage(self, name: str, run_config: RunConfig | None) -> str:
        if run_config is not None:
            return run_config.config.train.stage
        for token, stage in STAGE_BY_NAME:
            if token in name:
                return stage
        return "pretrain"

    @staticmethod
    def _status(
        step: int | None,
        max_steps: int | None,
        pipeline: dict[str, Any] | None,
        output_dir: str,
    ) -> str:
        if pipeline is not None and pipeline["running"]:
            current = next(
                (item for item in pipeline["stages"] if item["key"] == pipeline["current_stage"]),
                None,
            )
            if current is not None and current["output_dir"] == output_dir:
                return "running"
        if step is not None and max_steps is not None and step >= max_steps:
            return "completed"
        return "stopped"

    @staticmethod
    def _started_at(directory: Path) -> str | None:
        log = directory / "train.log"
        if not log.is_file():
            return None
        with log.open(encoding="utf-8", errors="replace") as handle:
            match = LOG_TIME.match(handle.readline())
        if match is None:
            return None
        local = datetime.strptime(match.group(1), "%Y-%m-%d %H:%M:%S")
        return local.astimezone().astimezone(UTC).isoformat()

    def _config_payload(self, run_config: RunConfig) -> dict[str, Any]:
        return {
            "path": str(run_config.path.relative_to(self.root)),
            "text": SECRET_PATTERN.sub(
                r"\1\2[redacted]", run_config.path.read_text(encoding="utf-8")
            ),
            "parsed": run_config.config.model_dump(mode="json"),
        }

    def _evaluations(self, name: str) -> list[dict[str, Any]]:
        registry = self.root / "experiments" / "registry.jsonl"
        if not registry.is_file():
            return []
        config = self.configs().get(f"experiments/{name}")
        if config is None:
            return []
        prefix = config.config.train.checkpoint_dir
        results: list[dict[str, Any]] = []
        for line in registry.read_text(encoding="utf-8").splitlines():
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            checkpoint = str(entry.get("checkpoint") or "")
            if prefix in checkpoint:
                results.append(
                    {
                        "name": entry.get("name"),
                        "checkpoint": checkpoint.split(prefix, 1)[1].strip("/"),
                        "results": entry.get("results", {}),
                    }
                )
        return results
