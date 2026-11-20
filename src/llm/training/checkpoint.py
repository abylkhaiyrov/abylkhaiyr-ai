from __future__ import annotations

import json
import os
import random
import shutil
import stat
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from importlib import import_module
from pathlib import Path
from pickle import UnpicklingError
from typing import Any

import numpy as np
import torch
from safetensors.torch import load_file, save_file
from torch import nn

from llm.core.checksums import file_sha256
from llm.core.config import ModelConfig
from llm.core.errors import CheckpointError
from llm.model.transformer import CausalLM

MODEL_FILE = "model.safetensors"
STATE_FILE = "training_state.pt"
META_FILE = "meta.json"
STEP_PREFIX = "step-"
STEP_DIGITS = 8
TEMP_SUFFIX = ".tmp"
BACKUP_SUFFIX = ".old"
TIED_HEAD_KEY = "lm_head.weight"


def is_junction(path: Path) -> bool:
    try:
        details = path.lstat()
    except FileNotFoundError:
        return False
    tag = getattr(stat, "IO_REPARSE_TAG_MOUNT_POINT", None)
    return tag is not None and getattr(details, "st_reparse_tag", None) == tag


@dataclass
class TrainingState:
    step: int = 0
    tokens_seen: int = 0
    best_validation_loss: float = float("inf")
    evaluations_without_improvement: int = 0
    metrics: list[dict[str, float]] = field(default_factory=list)
    micro_batch_size: int = 1
    gradient_accumulation_steps: int = 1


def capture_rng_state(device: torch.device) -> dict[str, Any]:
    name, keys, position, has_gauss, cached_gaussian = np.random.get_state()
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": (name, keys.tolist(), position, has_gauss, cached_gaussian),
        "torch": torch.get_rng_state(),
    }
    if device.type == "cuda":
        state["cuda"] = torch.cuda.get_rng_state_all()
    elif device.type == "mps":
        state["mps"] = torch.mps.get_rng_state()
    return state


def restore_rng_state(state: dict[str, Any], device: torch.device) -> None:
    random.setstate(state["python"])
    name, keys, position, has_gauss, cached_gaussian = state["numpy"]
    np.random.set_state(
        (name, np.asarray(keys, dtype=np.uint32), position, has_gauss, cached_gaussian)
    )
    torch.set_rng_state(state["torch"].cpu())
    if device.type == "cuda" and "cuda" in state:
        torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda"]])
    elif device.type == "mps" and "mps" in state:
        torch.mps.set_rng_state(state["mps"].cpu())


def load_training_state(path: Path) -> dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except UnpicklingError:
        try:
            multiarray = import_module("numpy._core.multiarray")
        except ModuleNotFoundError:
            multiarray = import_module("numpy.core.multiarray")
        reconstruct = multiarray._reconstruct
        legacy_rng: list[Callable[..., Any] | tuple[Callable[..., Any], str]] = [
            np.ndarray,
            np.dtype,
            type(np.dtype("uint32")),
            (reconstruct, "numpy.core.multiarray._reconstruct"),
            (reconstruct, "numpy._core.multiarray._reconstruct"),
        ]
        try:
            with torch.serialization.safe_globals(legacy_rng):
                return torch.load(path, map_location="cpu", weights_only=True)
        except UnpicklingError as error:
            raise CheckpointError(f"unsupported training state: {path}") from error


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
        raise CheckpointError(
            f"weights mismatch: missing={missing} unexpected={result.unexpected_keys}"
        )


class CheckpointManager:
    def __init__(
        self,
        directory: str | Path,
        keep: int,
        protected_steps: Iterable[int] = (),
        alias_prefix: str | None = None,
    ) -> None:
        self.directory = Path(directory)
        self.keep = keep
        self.protected_steps = frozenset(protected_steps)
        self.alias_prefix = alias_prefix
        self.directory.mkdir(parents=True, exist_ok=True)
        self._recover_interrupted_replacements()

    def _recover_interrupted_replacements(self) -> None:
        for backup in self.directory.glob(f"*{BACKUP_SUFFIX}"):
            target = backup.with_name(backup.name.removesuffix(BACKUP_SUFFIX))
            if target.exists():
                shutil.rmtree(backup, ignore_errors=True)
            else:
                os.replace(backup, target)

    def step_dir(self, step: int) -> Path:
        return self.directory / f"{STEP_PREFIX}{step:0{STEP_DIGITS}d}"

    def list_steps(self) -> list[Path]:
        return sorted(
            (
                path
                for path in self.directory.glob(f"{STEP_PREFIX}*")
                if path.name.removeprefix(STEP_PREFIX).isdecimal()
                and path.is_dir()
                and not path.is_symlink()
            ),
            key=lambda path: int(path.name.removeprefix(STEP_PREFIX)),
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
        label: str | None = None,
    ) -> Path:
        target = self.directory / label if label else self.step_dir(state.step)
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
        self._publish(temporary, target)
        if not label:
            self._publish_aliases(target, state.step)
            self._prune()
        return target

    @staticmethod
    def _publish(temporary: Path, target: Path) -> None:
        backup = target.with_name(target.name + BACKUP_SUFFIX)
        if target.exists():
            shutil.rmtree(backup, ignore_errors=True)
            os.replace(target, backup)
        os.replace(temporary, target)
        shutil.rmtree(backup, ignore_errors=True)

    def best_label(self) -> str:
        return f"{self.alias_prefix}_best" if self.alias_prefix else "best"

    def alias_path(self, suffix: str) -> Path:
        return self.directory / f"{self.alias_prefix}_{suffix}"

    def _link(self, alias: Path, target: Path) -> None:
        if os.name == "nt":
            junction = alias.with_name(alias.name + TEMP_SUFFIX)
            if is_junction(junction):
                junction.rmdir()
            elif junction.is_dir():
                shutil.rmtree(junction)
            winapi = import_module("_winapi")
            try:
                winapi.CreateJunction(str(target.resolve()), str(junction.absolute()))
            except PermissionError:
                shutil.copytree(target, junction)
            if is_junction(alias):
                alias.rmdir()
            elif alias.is_symlink():
                alias.unlink()
            elif alias.is_dir():
                shutil.rmtree(alias)
            os.replace(junction, alias)
        else:
            temporary = alias.with_name(alias.name + TEMP_SUFFIX)
            temporary.unlink(missing_ok=True)
            temporary.symlink_to(target.name, target_is_directory=True)
            os.replace(temporary, alias)

    def _publish_aliases(self, target: Path, step: int) -> None:
        if not self.alias_prefix:
            return
        self._link(self.alias_path("latest"), target)
        if step in self.protected_steps:
            self._link(self.alias_path(f"step_{step}"), target)

    def _is_protected(self, path: Path) -> bool:
        return int(path.name.removeprefix(STEP_PREFIX)) in self.protected_steps

    def _prune(self) -> None:
        for stale in self.list_steps()[: -self.keep]:
            if not self._is_protected(stale):
                shutil.rmtree(stale)

    @staticmethod
    def verify(path: Path, *, require_training_state: bool = True) -> dict[str, Any]:
        meta_file = path / META_FILE
        try:
            meta = json.loads(meta_file.read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            raise CheckpointError(f"unreadable checkpoint metadata: {path}") from error
        if not isinstance(meta, dict) or not isinstance(meta.get("model_config"), dict):
            raise CheckpointError(f"invalid checkpoint metadata: {path}")
        if type(meta.get("step")) is not int or meta["step"] < 0:
            raise CheckpointError(f"invalid checkpoint step: {path}")
        inference_only = meta.get("kind") == "inference"
        if inference_only and require_training_state:
            raise CheckpointError(f"checkpoint has no training state: {path}")
        expected_files = {MODEL_FILE} if inference_only else {MODEL_FILE, STATE_FILE}
        files = meta.get("files")
        if not isinstance(files, dict) or set(files) != expected_files:
            raise CheckpointError(f"invalid checkpoint file manifest: {path}")
        for name, expected in files.items():
            file = path / name
            try:
                if not file.is_file() or file.is_symlink() or file_sha256(file) != expected:
                    raise CheckpointError(f"corrupted checkpoint file: {file}")
            except OSError as error:
                raise CheckpointError(f"unreadable checkpoint file: {file}") from error
        return meta

    def latest_valid(self) -> Path | None:
        for path in reversed(self.list_steps()):
            try:
                self.verify(path)
            except CheckpointError:
                continue
            return path
        return None

    @staticmethod
    def load_optimizer_state(path: Path, optimizer: torch.optim.Optimizer) -> None:
        CheckpointManager.verify(path)
        payload = load_training_state(path / STATE_FILE)
        optimizer.load_state_dict(payload["optimizer"])

    def restore(
        self,
        path: Path,
        model: nn.Module,
        optimizer: torch.optim.Optimizer | None,
        scaler: torch.amp.GradScaler | None,
        device: torch.device,
        *,
        tokenizer_version: str | None = None,
    ) -> tuple[TrainingState, dict[str, int]]:
        meta = self.verify(path)
        if tokenizer_version is not None and meta.get("tokenizer_version") != tokenizer_version:
            raise CheckpointError("checkpoint tokenizer fingerprint differs")
        payload = load_training_state(path / STATE_FILE)
        load_model_weights(model, path / MODEL_FILE)
        if optimizer is not None:
            optimizer.load_state_dict(payload["optimizer"])
        if scaler is not None and payload["scaler"] is not None:
            scaler.load_state_dict(payload["scaler"])
        restore_rng_state(payload["rng"], device)
        return TrainingState(**payload["state"]), payload["loader"]


def load_model_for_inference(path: str | Path) -> tuple[CausalLM, ModelConfig, dict[str, Any]]:
    directory = Path(path)
    meta = CheckpointManager.verify(directory, require_training_state=False)
    config = ModelConfig.model_validate(meta["model_config"])
    model = CausalLM(config)
    load_model_weights(model, directory / MODEL_FILE)
    return model, config, meta
