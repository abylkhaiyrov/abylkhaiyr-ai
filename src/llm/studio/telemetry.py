from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import psutil
import torch

from llm.core.hardware import detect_hardware, resolve_device


def _size_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_file():
        return path.stat().st_size
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def _mps_metrics() -> dict[str, float | None]:
    backend = getattr(torch, "mps", None)
    if backend is None or not torch.backends.mps.is_available():
        return {"allocated_bytes": None, "driver_allocated_bytes": None, "peak_bytes": None}
    try:
        return {
            "allocated_bytes": float(backend.current_allocated_memory()),
            "driver_allocated_bytes": float(backend.driver_allocated_memory()),
            "peak_bytes": None,
        }
    except RuntimeError:
        return {"allocated_bytes": None, "driver_allocated_bytes": None, "peak_bytes": None}


def storage_snapshot(root: Path) -> dict[str, int]:
    return {
        "datasets_bytes": _size_bytes(root / "data"),
        "checkpoints_bytes": _size_bytes(root / "checkpoints"),
        "database_bytes": _size_bytes(root / "studio" / "studio.db"),
        "logs_bytes": _size_bytes(root / "experiments"),
    }


def system_snapshot(root: Path, storage: dict[str, int] | None = None) -> dict[str, Any]:
    memory = psutil.virtual_memory()
    swap = psutil.swap_memory()
    disk = psutil.disk_usage(str(root))
    device = resolve_device("auto")
    gpu: dict[str, Any] = {
        "device": device.type,
        "supported_dtypes": ["float32"],
        "allocated_bytes": None,
        "driver_allocated_bytes": None,
        "peak_bytes": None,
    }
    if device.type == "cuda":
        gpu.update(
            {
                "allocated_bytes": torch.cuda.memory_allocated(device),
                "driver_allocated_bytes": torch.cuda.memory_reserved(device),
                "peak_bytes": torch.cuda.max_memory_allocated(device),
                "supported_dtypes": ["float32", "float16", "bfloat16"],
            }
        )
    elif device.type == "mps":
        gpu.update(_mps_metrics())
        gpu["supported_dtypes"] = ["float32"]
    process = psutil.Process(os.getpid())
    try:
        cpu_frequency = psutil.cpu_freq()
    except (OSError, RuntimeError, SystemError):
        cpu_frequency = None
    return {
        "recorded_at": datetime.now(UTC).isoformat(),
        "cpu": {
            "percent": psutil.cpu_percent(interval=0.05),
            "per_core_percent": psutil.cpu_percent(interval=None, percpu=True),
            "logical_cores": psutil.cpu_count(logical=True),
            "physical_cores": psutil.cpu_count(logical=False),
            "frequency_mhz": None if cpu_frequency is None else cpu_frequency.current,
        },
        "memory": {
            "total_bytes": memory.total,
            "used_bytes": memory.used,
            "available_bytes": memory.available,
            "python_process_bytes": process.memory_info().rss,
            "swap_total_bytes": swap.total,
            "swap_used_bytes": swap.used,
            "pressure": None,
        },
        "gpu": gpu,
        "disk": {
            "total_bytes": disk.total,
            "used_bytes": disk.used,
            "free_bytes": disk.free,
            **(storage or storage_snapshot(root)),
        },
        "hardware": detect_hardware().as_dict(),
    }


def read_metrics(path: Path, limit: int = 3000) -> list[dict[str, float]]:
    if not path.is_file():
        return []
    records: list[dict[str, float]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(item, dict) and isinstance(item.get("step"), int):
                records.append(item)
    return records[-limit:]


def run_metrics(
    root: Path, run_name: str | None = None, limit: int = 3000
) -> list[dict[str, float]]:
    if run_name is not None:
        return read_metrics(root / "experiments" / run_name / "metrics.jsonl", limit)
    paths = list((root / "experiments").glob("*/metrics.jsonl"))
    if not paths:
        return []
    latest = max(paths, key=lambda item: item.stat().st_mtime)
    return read_metrics(latest, limit)


def latest_metrics(root: Path, run_name: str | None = None) -> dict[str, float]:
    records = run_metrics(root, run_name, 1)
    return records[0] if records else {}


def checkpoint_inventory(root: Path) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for meta_path in sorted((root / "checkpoints").glob("**/meta.json")):
        try:
            with meta_path.open(encoding="utf-8") as handle:
                metadata = json.load(handle)
        except (OSError, json.JSONDecodeError):
            continue
        directory = meta_path.parent
        weight_path = directory / "model.safetensors"
        results.append(
            {
                "id": str(directory.relative_to(root)),
                "name": directory.parent.name,
                "path": str(directory),
                "step": metadata.get("step"),
                "tokens_seen": metadata.get("tokens_seen"),
                "created_at": datetime.fromtimestamp(meta_path.stat().st_mtime, UTC).isoformat(),
                "weights_bytes": weight_path.stat().st_size if weight_path.is_file() else None,
                "metadata": metadata,
            }
        )
    return results
