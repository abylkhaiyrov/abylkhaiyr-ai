from __future__ import annotations

from contextlib import AbstractContextManager, nullcontext

import torch


def autocast_context(device: torch.device, dtype: torch.dtype) -> AbstractContextManager:
    if dtype == torch.float32:
        return nullcontext()
    return torch.autocast(device_type=device.type, dtype=dtype)


def make_grad_scaler(device: torch.device, dtype: torch.dtype) -> torch.amp.GradScaler | None:
    if device.type == "cuda" and dtype == torch.float16:
        return torch.amp.GradScaler("cuda")
    return None
