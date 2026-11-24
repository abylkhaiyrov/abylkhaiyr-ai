from __future__ import annotations

from pathlib import Path

import psutil
import torch

from llm.core.config import StrictModel
from llm.core.errors import ResourceError
from llm.core.hardware import free_disk_gb

GIB = 2**30


class ResourceBudget(StrictModel):
    ram_margin_gib: float = 0.75
    disk_reserve_gib: float = 20.0
    disk_per_job_gib: float = 0.6
    disk_slack_gib: float = 0.5
    require_mps: bool = True


def resource_snapshot(
    expected_peak: float, jobs: int, budget: ResourceBudget, directory: Path
) -> dict[str, float]:
    ram = psutil.virtual_memory()
    swap = psutil.swap_memory()
    return {
        "unified_ram_gib": ram.total / GIB,
        "available_ram_gib": ram.available / GIB,
        "swap_used_gib": swap.used / GIB,
        "swap_total_gib": swap.total / GIB,
        "disk_free_gib": free_disk_gb(directory),
        "disk_required_gib": budget.disk_reserve_gib
        + jobs * budget.disk_per_job_gib
        + budget.disk_slack_gib,
        "calibrated_mps_peak_gib": expected_peak,
    }


def resource_gate(
    expected_peak: float, jobs: int, budget: ResourceBudget, directory: Path
) -> dict[str, float]:
    snapshot = resource_snapshot(expected_peak, jobs, budget, directory)
    if budget.require_mps and not torch.backends.mps.is_available():
        raise ResourceError("MPS preflight failed")
    if snapshot["available_ram_gib"] < expected_peak + budget.ram_margin_gib:
        raise ResourceError("RAM preflight failed")
    if snapshot["disk_free_gib"] < snapshot["disk_required_gib"]:
        raise ResourceError(f"disk preflight needs {snapshot['disk_required_gib']:.2f} GiB")
    return snapshot


def resources_available(expected_peak: float, budget: ResourceBudget, directory: Path) -> bool:
    try:
        resource_gate(
            expected_peak, 0, budget.model_copy(update={"disk_slack_gib": 0.0}), directory
        )
    except ResourceError:
        return False
    return True
