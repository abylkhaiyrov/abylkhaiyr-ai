from __future__ import annotations

import json
import math
import threading
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PROGRESS_FILE = "progress.json"
SCHEMA_VERSION = 1
MIN_WRITE_INTERVAL = 0.5


def _iso(timestamp: float) -> str:
    return datetime.fromtimestamp(timestamp, UTC).isoformat()


def rate_and_eta(
    done: float, total: float | None, elapsed: float
) -> tuple[float | None, float | None]:
    if elapsed <= 0 or done <= 0:
        return None, None
    rate = done / elapsed
    if total is None or total < done:
        return rate, None
    return rate, (total - done) / rate


class StageHandle:
    def __init__(self, tracker: ProgressTracker, name: str, entry: dict[str, Any]) -> None:
        self._tracker = tracker
        self.name = name
        self._entry = entry
        self._started = time.monotonic()

    def advance(self, amount: float = 1, **detail: Any) -> None:
        self._entry["done"] += amount
        self.update(**detail)

    def set_done(self, done: float, **detail: Any) -> None:
        self._entry["done"] = done
        self.update(**detail)

    def set_manual_eta(self) -> None:
        self._entry["manual_eta"] = True

    def set_eta(self, seconds: float | None) -> None:
        self._entry["manual_eta"] = True
        self._entry["eta_seconds"] = seconds
        self._tracker.flush(self.name)

    def set_total(self, total: float | None) -> None:
        self._entry["total"] = total
        self._tracker.flush(self.name)

    def update(self, **detail: Any) -> None:
        self._entry["detail"].update(detail)
        self._tracker.flush(self.name)

    def record(self, key: str, value: Any) -> None:
        self._entry["results"][key] = value
        self._tracker.flush(self.name)

    def finish(self, **results: Any) -> None:
        self._entry["status"] = "completed"
        self._entry["results"].update(results)
        if self._entry["total"] is not None:
            self._entry["done"] = self._entry["total"]
        self._tracker.flush(self.name, force=True)

    def stop(self) -> None:
        self._entry["status"] = "stopped"
        self._tracker.flush(self.name, force=True)

    def fail(self, error: str) -> None:
        self._entry["status"] = "failed"
        self._entry["error"] = error
        self._tracker.flush(self.name, force=True)

    def elapsed(self) -> float:
        return time.monotonic() - self._started


class ProgressTracker:
    def __init__(self, directory: str | Path, min_interval: float = MIN_WRITE_INTERVAL) -> None:
        self.path = Path(directory) / PROGRESS_FILE
        self.min_interval = min_interval
        self._lock = threading.Lock()
        self._last_write = 0.0
        self._entries: dict[str, dict[str, Any]] = {}
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def begin(self, name: str, unit: str, total: float | None = None, **detail: Any) -> StageHandle:
        now = time.time()
        entry: dict[str, Any] = {
            "status": "running",
            "unit": unit,
            "total": total,
            "done": 0,
            "started_at": _iso(now),
            "updated_at": _iso(now),
            "rate": None,
            "eta_seconds": None,
            "manual_eta": False,
            "detail": dict(detail),
            "results": {},
            "error": None,
        }
        with self._lock:
            self._entries[name] = entry
        handle = StageHandle(self, name, entry)
        self.flush(name, force=True)
        return handle

    def flush(self, name: str, force: bool = False) -> None:
        now = time.monotonic()
        with self._lock:
            if not force and now - self._last_write < self.min_interval:
                return
            entry = self._entries[name]
            started = datetime.fromisoformat(entry["started_at"]).timestamp()
            rate, eta = rate_and_eta(entry["done"], entry["total"], time.time() - started)
            entry["rate"] = rate
            if entry["status"] != "running":
                entry["eta_seconds"] = None
            elif not entry["manual_eta"]:
                entry["eta_seconds"] = eta
            entry["updated_at"] = _iso(time.time())
            payload = {
                "version": SCHEMA_VERSION,
                "updated_at": entry["updated_at"],
                "stages": self._entries,
            }
            self._write(payload)
            self._last_write = now

    def _write(self, payload: dict[str, Any]) -> None:
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(json.dumps(_sanitize(payload)), encoding="utf-8")
        temporary.replace(self.path)


def _sanitize(value: Any) -> Any:
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(key): _sanitize(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_sanitize(item) for item in value]
    if value is None or isinstance(value, str | int | bool):
        return value
    return str(value)


def read_progress(directory: str | Path) -> dict[str, Any]:
    path = Path(directory) / PROGRESS_FILE
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    stages = payload.get("stages") if isinstance(payload, dict) else None
    return stages if isinstance(stages, dict) else {}
