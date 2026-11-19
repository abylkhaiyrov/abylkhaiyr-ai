from __future__ import annotations

import gc
import time

import psutil
import torch

from llm.core.errors import ResourceError

BYTES_PER_GB = 1024**3
RAM_CHECK_ATTEMPTS = 5
RAM_CHECK_WAIT_SECONDS = 2.0
OOM_MARKERS = ("out of memory", "invalid buffer size", "mps backend out of memory")


def ram_used_gb() -> float:
    return psutil.Process().memory_info().rss / BYTES_PER_GB


def swap_used_gb() -> float:
    return psutil.swap_memory().used / BYTES_PER_GB


def ram_fraction_used() -> float:
    return psutil.virtual_memory().percent / 100.0


def device_memory_used_gb(device: torch.device) -> float:
    if device.type == "cuda":
        return torch.cuda.memory_allocated(device) / BYTES_PER_GB
    if device.type == "mps":
        return torch.mps.current_allocated_memory() / BYTES_PER_GB
    return 0.0


def is_out_of_memory(error: BaseException) -> bool:
    if isinstance(error, torch.OutOfMemoryError):
        return True
    return isinstance(error, RuntimeError) and any(m in str(error).lower() for m in OOM_MARKERS)


def release_device_memory(device: torch.device) -> None:
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        torch.mps.empty_cache()


def ensure_ram_budget(max_fraction: float) -> None:
    for attempt in range(RAM_CHECK_ATTEMPTS):
        if ram_fraction_used() <= max_fraction:
            return
        gc.collect()
        if attempt + 1 < RAM_CHECK_ATTEMPTS:
            time.sleep(RAM_CHECK_WAIT_SECONDS)
    raise ResourceError(f"system RAM usage stayed above {max_fraction:.0%}")


def configure_threads(num_threads: int | None) -> None:
    if num_threads is not None:
        torch.set_num_threads(num_threads)
