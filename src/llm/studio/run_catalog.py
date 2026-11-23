from __future__ import annotations

import re
import threading
import time
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

from llm.studio.pagination import matches, paginate
from llm.studio.pipeline_view import FINISHED, parse_time

CACHE_TTL = 2.0
ACTIVE = "running"
TERMINAL = frozenset({"completed", "stopped", "failed"})
VERSION = re.compile(r"v(\d+(?:[._]\d+)*)", re.IGNORECASE)
EPOCH = datetime.min.replace(tzinfo=UTC)


def version_of(name: str) -> str | None:
    match = VERSION.search(name)
    return None if match is None else f"v{match.group(1).replace('_', '.')}"


def pipeline_status(pipeline: dict[str, Any]) -> str:
    if pipeline["running"]:
        return ACTIVE
    statuses = [stage["status"] for stage in pipeline["stages"]]
    if "failed" in statuses:
        return "failed"
    started = [status for status in statuses if status != "pending"]
    if started and all(status in FINISHED for status in started) and statuses[-1] in FINISHED:
        return "completed"
    return "stopped"


def pipeline_failure(pipeline: dict[str, Any]) -> str | None:
    failed = next((stage for stage in pipeline["stages"] if stage["status"] == "failed"), None)
    return (
        None
        if failed is None
        else str(failed.get("error") or failed.get("note_code") or failed["key"])
    )


def pipeline_entry(pipeline: dict[str, Any]) -> dict[str, Any]:
    status = pipeline_status(pipeline)
    stage = next(
        (item for item in pipeline["stages"] if item["key"] == pipeline["current_stage"]), None
    )
    return {
        "id": f"pipeline:{pipeline['id']}",
        "kind": "pipeline",
        "name": pipeline["id"],
        "model_name": pipeline["model_name"],
        "version": version_of(pipeline["id"]),
        "stage": "pipeline",
        "status": status,
        "origin": "pipeline_state",
        "created_at": pipeline["started_at"],
        "updated_at": pipeline["updated_at"],
        "step": None,
        "max_steps": None,
        "percent": pipeline["overall"].get("percent")
        if isinstance(pipeline["overall"], dict)
        else None,
        "current_stage": pipeline["current_stage"] if status == ACTIVE else None,
        "output_dir": None if stage is None else stage["output_dir"],
        "state_dir": pipeline["state_dir"],
        "config_path": None,
        "checkpoint": pipeline["checkpoint"],
        "failure_reason": pipeline_failure(pipeline) if status == "failed" else None,
        "validation_loss": None,
        "best_validation_loss": None,
        "validation_perplexity": None,
        "train_loss": None,
        "tokens_seen": None,
        "tokens_per_second": None,
        "dataset": pipeline.get("dataset"),
        "duration_seconds": duration(pipeline["started_at"], pipeline["updated_at"]),
    }


def training_entry(run: dict[str, Any], status: str, failure: str | None) -> dict[str, Any]:
    return {
        "id": f"run:{run['name']}",
        "kind": "training",
        "name": run["name"],
        "model_name": run["name"],
        "version": version_of(run["name"]),
        "stage": run["stage"],
        "status": status,
        "origin": "metrics",
        "created_at": run["started_at"] or run["updated_at"],
        "updated_at": run["updated_at"],
        "step": run["step"],
        "max_steps": run["max_steps"],
        "percent": run["percent"],
        "current_stage": run["stage"] if status == ACTIVE else None,
        "output_dir": run["output_dir"],
        "state_dir": None,
        "config_path": run["config_path"],
        "checkpoint": None,
        "failure_reason": failure,
        "validation_loss": run["validation_loss"],
        "best_validation_loss": run["best_validation_loss"],
        "validation_perplexity": run["validation_perplexity"],
        "train_loss": run["train_loss"],
        "tokens_seen": run["tokens_seen"],
        "tokens_per_second": run.get("tokens_per_second"),
        "dataset": run.get("data_dir"),
        "duration_seconds": duration(run["started_at"], run["updated_at"]),
    }


def duration(start: str | None, end: str | None) -> float | None:
    first, last = parse_time(start), parse_time(end)
    return None if first is None or last is None else max((last - first).total_seconds(), 0.0)


def sort_key(entry: dict[str, Any]) -> datetime:
    return parse_time(entry["updated_at"]) or EPOCH


def select_default(entries: list[dict[str, Any]]) -> dict[str, Any] | None:
    active = [item for item in entries if item["status"] == ACTIVE]
    if active:
        return max(active, key=sort_key)
    finished = [item for item in entries if item["status"] in TERMINAL]
    return max(finished, key=sort_key, default=None)


class RunCatalog:
    def __init__(
        self,
        pipelines: Callable[[], list[dict[str, Any]]],
        runs: Callable[[], list[dict[str, Any]]],
        processes: Callable[[], list[dict[str, Any]]],
    ) -> None:
        self._pipelines = pipelines
        self._runs = runs
        self._processes = processes
        self._lock = threading.Lock()
        self._cache: tuple[float, list[dict[str, Any]]] = (0.0, [])

    def invalidate(self) -> None:
        with self._lock:
            self._cache = (0.0, [])

    def entries(self) -> list[dict[str, Any]]:
        with self._lock:
            stamp, cached = self._cache
            if cached and time.monotonic() - stamp < CACHE_TTL:
                return cached
            built = self._build()
            self._cache = (time.monotonic(), built)
            return built

    def get(self, run_id: str) -> dict[str, Any] | None:
        return next((item for item in self.entries() if item["id"] == run_id), None)

    def default(self) -> dict[str, Any] | None:
        return select_default(self.entries())

    def summary(self) -> dict[str, Any]:
        entries = self.entries()
        active = [item for item in entries if item["status"] == ACTIVE]
        return {
            "active": max(active, key=sort_key, default=None),
            "latest": select_default(entries),
            "completed_count": sum(item["status"] == "completed" for item in entries),
            "total": len(entries),
        }

    def page(
        self,
        *,
        status: str | None = None,
        stage: str | None = None,
        version: str | None = None,
        query: str | None = None,
        oldest_first: bool = False,
        limit: int = 50,
        offset: int = 0,
    ) -> dict[str, Any]:
        matched = [
            item
            for item in self.entries()
            if (status is None or item["status"] == status)
            and (stage is None or item["stage"] == stage)
            and (version is None or item["version"] == version)
            and matches(query, item["name"], item["model_name"])
        ]
        matched.sort(key=sort_key, reverse=not oldest_first)
        return paginate(matched, offset, limit)

    def _build(self) -> list[dict[str, Any]]:
        pipelines = self._pipelines()
        entries = [pipeline_entry(item) for item in pipelines]
        covered = {
            stage["output_dir"]
            for pipeline in pipelines
            for stage in pipeline["stages"]
            if stage["output_dir"]
        }
        by_output: dict[str, dict[str, Any]] = {}
        for record in self._processes():
            if record["output_dir"] and record["output_dir"] not in by_output:
                by_output[record["output_dir"]] = record
        for run in self._runs():
            if run["output_dir"] in covered:
                continue
            process = by_output.get(run["output_dir"])
            status, failure = self._training_state(run, process)
            entries.append(training_entry(run, status, failure))
        return sorted(entries, key=sort_key, reverse=True)

    @staticmethod
    def _training_state(
        run: dict[str, Any], process: dict[str, Any] | None
    ) -> tuple[str, str | None]:
        if process is not None and process["status"] in {"running", "pausing", "pending"}:
            return ACTIVE, None
        if run["status"] == "completed":
            return "completed", None
        if process is not None and process["status"] == "failed":
            return "failed", process["error_reason"]
        return "stopped", None
