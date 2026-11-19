from __future__ import annotations

import os
import platform
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path

import psutil
import torch

from llm.core.errors import HardwareError

BYTES_PER_GB = 1024**3
MPS_MEMORY_FRACTION = 0.6


@dataclass(frozen=True)
class HardwareInfo:
    system: str
    machine: str
    cpu_cores: int
    ram_gb: float
    device: str
    device_name: str
    device_memory_gb: float
    cuda_available: bool
    mps_available: bool
    bf16_supported: bool
    free_disk_gb: float
    torch_version: str

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def mps_available() -> bool:
    return bool(getattr(torch.backends, "mps", None)) and torch.backends.mps.is_available()


def resolve_device(requested: str = "auto") -> torch.device:
    cuda, mps = torch.cuda.is_available(), mps_available()
    if requested == "auto":
        return torch.device("cuda" if cuda else "mps" if mps else "cpu")
    if requested == "cuda" and not cuda:
        raise HardwareError("CUDA requested but not available")
    if requested == "mps" and not mps:
        raise HardwareError("MPS requested but not available")
    return torch.device(requested)


def resolve_dtype(precision: str, device: torch.device) -> torch.dtype:
    mapping = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
    if precision != "auto":
        dtype = mapping[precision]
        if device.type == "cpu" and dtype == torch.float16:
            raise HardwareError("fp16 is not supported on CPU")
        if device.type == "cuda" and dtype == torch.bfloat16 and not bf16_supported(device):
            raise HardwareError("bf16 is not supported on this CUDA device")
        return dtype
    if device.type == "cuda":
        return torch.bfloat16 if bf16_supported(device) else torch.float16
    return torch.float32


def bf16_supported(device: torch.device) -> bool:
    if device.type == "cuda":
        return torch.cuda.is_bf16_supported()
    return device.type == "cpu"


def device_memory_gb(device: torch.device, ram_gb: float) -> float:
    if device.type == "cuda":
        return torch.cuda.get_device_properties(device).total_memory / BYTES_PER_GB
    if device.type == "mps":
        return ram_gb * MPS_MEMORY_FRACTION
    return ram_gb


def device_name(device: torch.device) -> str:
    if device.type == "cuda":
        return torch.cuda.get_device_name(device)
    if device.type == "mps":
        return platform.processor() or "Apple Silicon"
    return platform.processor() or platform.machine()


def free_disk_gb(path: str | Path = ".") -> float:
    target = Path(path)
    while not target.exists() and target != target.parent:
        target = target.parent
    return shutil.disk_usage(target).free / BYTES_PER_GB


def detect_hardware(requested_device: str = "auto") -> HardwareInfo:
    device = resolve_device(requested_device)
    ram_gb = psutil.virtual_memory().total / BYTES_PER_GB
    return HardwareInfo(
        system=platform.system(),
        machine=platform.machine(),
        cpu_cores=os.cpu_count() or 1,
        ram_gb=round(ram_gb, 2),
        device=device.type,
        device_name=device_name(device),
        device_memory_gb=round(device_memory_gb(device, ram_gb), 2),
        cuda_available=torch.cuda.is_available(),
        mps_available=mps_available(),
        bf16_supported=bf16_supported(device),
        free_disk_gb=round(free_disk_gb(), 2),
        torch_version=torch.__version__,
    )
