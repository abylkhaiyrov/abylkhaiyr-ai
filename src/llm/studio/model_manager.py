from __future__ import annotations

import gc
import json
import threading
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import psutil
import torch
from pydantic import ValidationError

from llm.core.config import ModelConfig, load_experiment_config
from llm.core.errors import CheckpointError, ConfigError, TokenizerError
from llm.core.hardware import resolve_device
from llm.inference.engine import InferenceEngine
from llm.studio.model_registry import annotate_models
from llm.studio.runtime import RuntimeState
from llm.studio.telemetry import latest_metrics
from llm.tokenizer.bpe import BPETokenizer
from llm.training.checkpoint import MODEL_FILE, CheckpointManager, load_model_for_inference

MIN_CPU_HEADROOM_BYTES = 512 * 1024 * 1024
INFERENCE_MEMORY_MULTIPLIER = 3


class ModelManagerError(RuntimeError):
    def __init__(self, code: str, detail: str) -> None:
        super().__init__(detail)
        self.code = code
        self.detail = detail


@dataclass
class LoadedModelState:
    engine: InferenceEngine
    model_id: str
    checkpoint_id: str
    stage: str
    device: str
    config: dict[str, Any]
    tokenizer_version: str


class ModelManager:
    def __init__(
        self,
        root: Path,
        runtime: RuntimeState,
        inference_lock: threading.Lock,
        initial_engine: InferenceEngine | None = None,
        initial_model_id: str | None = None,
        initial_checkpoint_id: str | None = None,
        tokenizer_path: Path | None = None,
    ) -> None:
        self.root = root.resolve()
        self.runtime = runtime
        self.inference_lock = inference_lock
        self.tokenizer_path = (tokenizer_path or (self.root / "data" / "tokenizer")).resolve()
        self._lock = threading.RLock()
        self._active: LoadedModelState | None = None
        self._catalog_cache: tuple[tuple[Any, ...], list[dict[str, Any]]] | None = None
        if initial_engine is not None:
            resolved_checkpoint = initial_checkpoint_id or self._match_initial_checkpoint(
                initial_model_id
            )
            resolved_model_id = (
                initial_model_id
                or (Path(resolved_checkpoint).parent.name if resolved_checkpoint else None)
                or "local-model"
            )
            resolved_stage = self._detect_stage(
                resolved_model_id,
                {},
                resolved_checkpoint,
            )
            self._active = LoadedModelState(
                engine=initial_engine,
                model_id=resolved_model_id,
                checkpoint_id=resolved_checkpoint or resolved_model_id,
                stage=resolved_stage,
                device=initial_engine.device.type,
                config=initial_engine.model.config.model_dump(),
                tokenizer_version=initial_engine.tokenizer.version,
            )

    @property
    def active(self) -> LoadedModelState | None:
        with self._lock:
            return self._active

    def active_info(self) -> dict[str, Any] | None:
        with self._lock:
            if self._active is None:
                return None
            config = self._active.config
            return {
                "id": self._active.checkpoint_id,
                "model_id": self._active.model_id,
                "checkpoint_id": self._active.checkpoint_id,
                "name": self._active.model_id,
                "stage": self._active.stage,
                "device": self._active.device,
                "loaded": True,
                "available": True,
                "compatible": True,
                "compatibility_issues": [],
                "availability_reason": None,
                "parameters": self.parameter_count(config),
                "vocab_size": config.get("vocab_size"),
                "context_length": config.get("max_position_embeddings"),
                "architecture": config,
            }

    def registry(self) -> dict[str, Any]:
        models = self.list_models()
        active = self.active
        assignments = annotate_models(
            self.root,
            self.runtime.pipeline(),
            models,
            active.checkpoint_id if active is not None else None,
        )
        return {"roles": assignments, "models": models}

    def list_models(self) -> list[dict[str, Any]]:
        signature = self._catalog_signature()
        cached = self._catalog_cache
        if cached is not None and cached[0] == signature:
            return [dict(item) for item in cached[1]]
        models = self._scan_models()
        self._catalog_cache = (signature, models)
        return [dict(item) for item in models]

    def catalog_signature(self) -> tuple[Any, ...]:
        return self._catalog_signature()

    def _catalog_signature(self) -> tuple[Any, ...]:
        checkpoints_root = self.root / "checkpoints"
        stamps = tuple(
            sorted(
                (str(path), path.stat().st_mtime_ns)
                for path in checkpoints_root.glob("**/meta.json")
            )
        )
        active = self._active
        return (
            stamps,
            None if active is None else (active.checkpoint_id, active.device),
            self.runtime.training_active(),
        )

    def _scan_models(self) -> list[dict[str, Any]]:
        metrics_by_run = {
            path.parent.name: latest_metrics(self.root, path.parent.name)
            for path in (self.root / "experiments").glob("*/metrics.jsonl")
        }
        tokenizer = self._load_tokenizer()
        training_active = self.runtime.training_active()
        active = self.active
        results: list[dict[str, Any]] = []
        seen_resolved: set[Path] = set()
        checkpoints_root = self.root / "checkpoints"
        if checkpoints_root.is_dir():
            for meta_path in sorted(checkpoints_root.glob("**/meta.json")):
                directory = meta_path.parent
                if directory.is_symlink():
                    continue
                resolved = directory.resolve()
                if resolved in seen_resolved:
                    continue
                seen_resolved.add(resolved)
                try:
                    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    continue
                checkpoint_id = str(directory.relative_to(self.root))
                run_name = directory.parent.name
                weight_path = directory / MODEL_FILE
                weights_bytes = weight_path.stat().st_size if weight_path.is_file() else None
                compatible, issues, parsed_config = self._check_compatibility(
                    directory, metadata, tokenizer
                )
                config_dict = (
                    parsed_config.model_dump()
                    if parsed_config is not None
                    else metadata.get("model_config", {})
                )
                stage = self._detect_stage(run_name, metadata, checkpoint_id)
                is_loaded = active is not None and active.checkpoint_id == checkpoint_id
                can_cpu, available_ram, required_ram = self.can_run_on_cpu(
                    weights_bytes, config_dict
                )
                available = compatible and (is_loaded or not training_active or can_cpu)
                availability_reason: str | None = None
                if not compatible:
                    availability_reason = issues[0] if issues else "incompatible_model"
                elif training_active and not can_cpu and not is_loaded:
                    availability_reason = (
                        f"insufficient_memory_during_training ({available_ram} < {required_ram})"
                    )
                results.append(
                    {
                        "id": checkpoint_id,
                        "model_id": run_name,
                        "checkpoint_id": checkpoint_id,
                        "name": run_name,
                        "path": str(directory),
                        "step": metadata.get("step"),
                        "tokens_seen": metadata.get("tokens_seen"),
                        "stage": stage,
                        "compatible": compatible,
                        "compatibility_issues": issues,
                        "available": available,
                        "availability_reason": availability_reason,
                        "loaded": is_loaded,
                        "device": active.device if is_loaded and active is not None else None,
                        "created_at": datetime.fromtimestamp(
                            meta_path.stat().st_mtime, UTC
                        ).isoformat(),
                        "weights_bytes": weights_bytes,
                        "parameters": self.parameter_count(config_dict),
                        "vocab_size": config_dict.get("vocab_size"),
                        "context_length": config_dict.get("max_position_embeddings"),
                        "architecture": config_dict,
                        "metrics": metrics_by_run.get(run_name, {}),
                        "metadata": metadata,
                    }
                )
        if active is not None and not any(item["loaded"] for item in results):
            results.insert(
                0,
                {
                    "id": active.checkpoint_id,
                    "model_id": active.model_id,
                    "checkpoint_id": active.checkpoint_id,
                    "name": active.model_id,
                    "path": active.checkpoint_id,
                    "step": None,
                    "tokens_seen": None,
                    "stage": active.stage,
                    "compatible": True,
                    "compatibility_issues": [],
                    "available": True,
                    "availability_reason": None,
                    "loaded": True,
                    "device": active.device,
                    "created_at": datetime.now(UTC).isoformat(),
                    "weights_bytes": None,
                    "parameters": self.parameter_count(active.config),
                    "vocab_size": active.config.get("vocab_size"),
                    "context_length": active.config.get("max_position_embeddings"),
                    "architecture": active.config,
                    "metrics": {},
                    "metadata": {},
                },
            )
        return results

    def load(self, checkpoint_id: str, device: str | None = None) -> dict[str, Any]:
        directory = self._resolve_checkpoint_path(checkpoint_id)
        tokenizer = self._load_tokenizer()
        if tokenizer is None:
            raise ModelManagerError("tokenizer_missing", "Tokenizer file is not available")
        try:
            meta = CheckpointManager.verify(directory)
        except CheckpointError as error:
            raise ModelManagerError("corrupted_checkpoint", str(error)) from error
        compatible, issues, _ = self._check_compatibility(directory, meta, tokenizer)
        if not compatible:
            raise ModelManagerError("incompatible_model", "; ".join(issues))
        weight_path = directory / MODEL_FILE
        weights_bytes = weight_path.stat().st_size if weight_path.is_file() else None
        target_device = self._select_device(device, weights_bytes, meta.get("model_config", {}))
        try:
            model, config, meta = load_model_for_inference(directory)
            engine = InferenceEngine(model, tokenizer, target_device)
        except Exception as error:
            raise ModelManagerError("load_failed", str(error)) from error
        canonical_id = str(directory.relative_to(self.root))
        run_name = directory.parent.name
        stage = self._detect_stage(run_name, meta, canonical_id)
        new_state = LoadedModelState(
            engine=engine,
            model_id=run_name,
            checkpoint_id=canonical_id,
            stage=stage,
            device=target_device.type,
            config=config.model_dump(),
            tokenizer_version=tokenizer.version,
        )
        with self.inference_lock, self._lock:
            previous = self._active
            self._active = new_state
            del previous
            self._release_memory()
        info = self.active_info()
        assert info is not None
        return info

    def unload(self) -> dict[str, Any]:
        with self.inference_lock, self._lock:
            previous = self._active
            self._active = None
            del previous
            self._release_memory()
        return {"loaded": False, "active_model": None}

    def switch(self, checkpoint_id: str, device: str | None = None) -> dict[str, Any]:
        with self._lock:
            if (
                self._active is not None
                and self._active.checkpoint_id == checkpoint_id
                and (device is None or device == "auto" or self._active.device == device)
            ):
                info = self.active_info()
                assert info is not None
                return info
        return self.load(checkpoint_id, device)

    def ensure_cpu_fallback(self) -> LoadedModelState:
        with self.inference_lock, self._lock:
            if self._active is None:
                raise ModelManagerError("model_not_loaded", "No inference model is loaded")
            if self._active.device == "cpu":
                return self._active
            can_cpu, available_ram, required_ram = self.can_run_on_cpu(None, self._active.config)
            if not can_cpu:
                raise ModelManagerError(
                    "insufficient_memory_during_training",
                    f"Not enough free RAM for CPU inference during active training ({available_ram} < {required_ram})",
                )
            self._active.engine.model.to(torch.device("cpu"))
            self._active.engine.device = torch.device("cpu")
            self._active.device = "cpu"
            self._release_memory()
            return self._active

    def can_run_on_cpu(
        self, weights_bytes: int | None = None, config: dict[str, Any] | None = None
    ) -> tuple[bool, int, int]:
        available = int(psutil.virtual_memory().available)
        estimated = self.estimate_inference_memory_bytes(weights_bytes, config or {})
        required = max(MIN_CPU_HEADROOM_BYTES, estimated * INFERENCE_MEMORY_MULTIPLIER)
        return available >= required, available, required

    @classmethod
    def estimate_inference_memory_bytes(
        cls, weights_bytes: int | None, config: dict[str, Any]
    ) -> int:
        if weights_bytes is not None and weights_bytes > 0:
            return weights_bytes
        params = cls.parameter_count(config)
        if params is not None and params > 0:
            return params * 4
        return 64 * 1024 * 1024

    @staticmethod
    def parameter_count(config: dict[str, Any]) -> int | None:
        values = ("vocab_size", "hidden_size", "num_hidden_layers", "intermediate_size")
        if not all(isinstance(config.get(key), int) for key in values):
            return None
        return int(
            config["vocab_size"] * config["hidden_size"]
            + config["num_hidden_layers"]
            * (
                4 * config["hidden_size"] ** 2
                + 3 * config["hidden_size"] * config["intermediate_size"]
            )
        )

    def _select_device(
        self,
        requested: str | None,
        weights_bytes: int | None,
        config: dict[str, Any],
    ) -> torch.device:
        if requested == "cpu":
            can_cpu, available_ram, required_ram = self.can_run_on_cpu(weights_bytes, config)
            if not can_cpu:
                raise ModelManagerError(
                    "insufficient_memory",
                    f"Insufficient RAM for CPU inference ({available_ram} < {required_ram})",
                )
            return torch.device("cpu")
        training_active = self.runtime.training_active()
        auto_device = resolve_device(requested or "auto")
        if training_active and auto_device.type == "mps":
            can_cpu, available_ram, required_ram = self.can_run_on_cpu(weights_bytes, config)
            if not can_cpu:
                raise ModelManagerError(
                    "insufficient_memory_during_training",
                    f"MPS is busy with active training and free RAM is below safety threshold ({available_ram} < {required_ram})",
                )
            return torch.device("cpu")
        return auto_device

    def _resolve_checkpoint_path(self, checkpoint_id: str) -> Path:
        raw = Path(checkpoint_id)
        candidate = raw.resolve() if raw.is_absolute() else (self.root / raw).resolve()
        checkpoints_root = (self.root / "checkpoints").resolve()
        if checkpoints_root not in candidate.parents:
            raise ModelManagerError("checkpoint_not_found", f"Invalid checkpoint: {checkpoint_id}")
        if not candidate.is_dir() or not (candidate / "meta.json").is_file():
            raise ModelManagerError(
                "checkpoint_not_found", f"Checkpoint not found: {checkpoint_id}"
            )
        return candidate

    def _load_tokenizer(self) -> BPETokenizer | None:
        if self._active is not None:
            return self._active.engine.tokenizer
        try:
            return BPETokenizer.load(self.tokenizer_path)
        except TokenizerError:
            return None

    def _check_compatibility(
        self,
        directory: Path,
        metadata: dict[str, Any],
        tokenizer: BPETokenizer | None,
    ) -> tuple[bool, list[str], ModelConfig | None]:
        issues: list[str] = []
        if not (directory / MODEL_FILE).is_file():
            issues.append("missing_weights")
        parsed: ModelConfig | None = None
        raw_config = metadata.get("model_config")
        if not isinstance(raw_config, dict):
            issues.append("missing_model_config")
        else:
            try:
                parsed = ModelConfig.model_validate(raw_config)
            except (ValidationError, TypeError, ValueError):
                issues.append("invalid_model_architecture")
        if tokenizer is None:
            issues.append("tokenizer_unavailable")
        else:
            expected_version = metadata.get("tokenizer_version")
            if expected_version and expected_version != tokenizer.version:
                issues.append("tokenizer_version_mismatch")
            if parsed is not None and parsed.vocab_size != tokenizer.vocab_size:
                issues.append("vocab_size_mismatch")
        return len(issues) == 0, issues, parsed

    def _detect_stage(
        self, run_name: str, metadata: dict[str, Any], checkpoint_id: str | None
    ) -> str:
        stage = metadata.get("stage")
        if isinstance(stage, str) and stage in {"pretrain", "sft", "dpo"}:
            return stage
        config_path = self.root / "configs" / f"{run_name}.yaml"
        if config_path.is_file():
            try:
                cfg = load_experiment_config(config_path)
                if cfg.train.stage in {"pretrain", "sft", "dpo"}:
                    return cfg.train.stage
            except ConfigError:
                pass
        lowered = f"{run_name} {checkpoint_id or ''}".lower()
        if "dpo" in lowered:
            return "dpo"
        if "sft" in lowered:
            return "sft"
        if (self.root / "checkpoints" / run_name).is_dir():
            return "pretrain"
        return "pretrain"

    def _match_initial_checkpoint(self, model_id: str | None) -> str | None:
        checkpoints_root = self.root / "checkpoints"
        if not checkpoints_root.is_dir():
            return None
        if model_id:
            candidate = checkpoints_root / model_id
            if candidate.is_dir():
                latest = CheckpointManager(candidate, keep=1).latest_valid()
                if latest is not None:
                    return str(latest.relative_to(self.root))
        return None

    @staticmethod
    def _release_memory() -> None:
        gc.collect()
        if torch.backends.mps.is_available():
            try:
                torch.mps.empty_cache()
            except RuntimeError:
                pass
