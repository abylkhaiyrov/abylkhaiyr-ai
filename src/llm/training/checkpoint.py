from __future__ import annotations

import hashlib
import json
import os
import random
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from safetensors.torch import load_file, save_file
from torch import nn

from llm.core.config import ModelConfig
from llm.core.errors import CheckpointError

MODEL_FILE = "model.safetensors"
STATE_FILE = "training_state.pt"
META_FILE = "meta.json"
STEP_PREFIX = "step-"
STEP_DIGITS = 8
TEMP_SUFFIX = ".tmp"
HASH_CHUNK = 1 << 20
TIED_HEAD_KEY = "lm_head.weight"


@dataclass
class TrainingState:
    step: int = 0
    tokens_seen: int = 0
    best_validation_loss: float = float("inf")
    evaluations_without_improvement: int = 0
    metrics: list[dict[str, float]] = field(default_factory=list)
    micro_batch_size: int | None = None
    gradient_accumulation_steps: int | None = None


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(HASH_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def capture_rng_state(device: torch.device) -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if device.type == "cuda":
        state["cuda"] = torch.cuda.get_rng_state_all()
    elif device.type == "mps":
        state["mps"] = torch.mps.get_rng_state()
    return state


def restore_rng_state(state: dict[str, Any], device: torch.device) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if device.type == "cuda" and "cuda" in state:
        torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda"]])
    elif device.type == "mps" and "mps" in state:
        torch.mps.set_rng_state(state["mps"].cpu())


def model_state_for_saving(model: nn.Module) -> dict[str, torch.Tensor]:
    tied = getattr(getattr(model, "config", None), "tie_word_embeddings", False)
    return {
        key: value.detach().cpu().contiguous()
        for key, value in model.state_dict().items()
        if not (tied and key == TIED_HEAD_KEY)
    }


def save_model_weights(model: nn.Module, path: Path) -> None:
    save_file(model_state_for_saving(model), str(path))


def load_model_weights(model: nn.Module, path: Path) -> None:
    result = model.load_state_dict(load_file(str(path)), strict=False)
    tied = getattr(getattr(model, "config", None), "tie_word_embeddings", False)
    missing = [key for key in result.missing_keys if not (tied and key == TIED_HEAD_KEY)]
    if missing or result.unexpected_keys:
        raise CheckpointError(f"weights mismatch: missing={missing} unexpected={result.unexpected_keys}")


class CheckpointManager:
    def __init__(self, directory: str | Path, keep: int) -> None:
        self.directory = Path(directory)
        self.keep = keep
        self.directory.mkdir(parents=True, exist_ok=True)

    def step_dir(self, step: int) -> Path:
        return self.directory / f"{STEP_PREFIX}{step:0{STEP_DIGITS}d}"

    def list_steps(self) -> list[Path]:
        return sorted(
            p
            for p in self.directory.glob(f"{STEP_PREFIX}*")
            if p.is_dir() and not p.name.endswith(TEMP_SUFFIX)
        )

    def save(
        self,
        model: nn.Module,
        model_config: ModelConfig,
        optimizer: torch.optim.Optimizer,
        scaler: torch.amp.GradScaler | None,
        state: TrainingState,
        loader_state: dict[str, int],
        tokenizer_version: str,
        device: torch.device,
    ) -> Path:
        target = self.step_dir(state.step)
        temporary = target.with_name(target.name + TEMP_SUFFIX)
        if temporary.exists():
            shutil.rmtree(temporary)
        temporary.mkdir(parents=True)
        save_model_weights(model, temporary / MODEL_FILE)
        payload = {
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict() if scaler is not None else None,
            "state": state.__dict__,
            "loader": loader_state,
            "rng": capture_rng_state(device),
        }
        torch.save(payload, temporary / STATE_FILE)
        meta = {
            "step": state.step,
            "tokens_seen": state.tokens_seen,
            "model_config": model_config.model_dump(),
            "tokenizer_version": tokenizer_version,
            "files": {name: file_sha256(temporary / name) for name in (MODEL_FILE, STATE_FILE)},
        }
        (temporary / META_FILE).write_text(json.dumps(meta, indent=2), encoding="utf-8")
        if target.exists():
            shutil.rmtree(target)
        os.replace(temporary, target)
        self._prune()
        return target

    def _prune(self) -> None:
        for stale in self.list_steps()[: -self.keep]:
            shutil.rmtree(stale)

    @staticmethod
    def verify(path: Path) -> dict[str, Any]:
        meta_file = path / META_FILE
        if not meta_file.is_file():
            raise CheckpointError(f"missing metadata: {path}")
        meta = json.loads(meta_file.read_text(encoding="utf-8"))
        for name, expected in meta["files"].items():
            file = path / name
            if not file.is_file() or file_sha256(file) != expected:
                raise CheckpointError(f"corrupted checkpoint file: {file}")
        return meta

    def latest_valid(self) -> Path | None:
        for path in reversed(self.list_steps()):
            try:
                self.verify(path)
            except CheckpointError:
                continue
            return path
        return None

    def restore(
        self,
        path: Path,
        model: nn.Module,
        optimizer: torch.optim.Optimizer | None,
        scaler: torch.amp.GradScaler | None,
        device: torch.device,
    ) -> tuple[TrainingState, dict[str, int]]:
        self.verify(path)
        load_model_weights(model, path / MODEL_FILE)
        payload = torch.load(path / STATE_FILE, map_location="cpu", weights_only=False)
        if optimizer is not None:
            optimizer.load_state_dict(payload["optimizer"])
        if scaler is not None and payload["scaler"] is not None:
            scaler.load_state_dict(payload["scaler"])
        restore_rng_state(payload["rng"], device)
        return TrainingState(**payload["state"]), payload["loader"]


def load_model_for_inference(path: str | Path) -> tuple[nn.Module, ModelConfig, dict[str, Any]]:
    from llm.model.transformer import CausalLM

    directory = Path(path)
    meta = CheckpointManager.verify(directory)
    config = ModelConfig.model_validate(meta["model_config"])
    model = CausalLM(config)
    load_model_weights(model, directory / MODEL_FILE)
    return model, config, meta
