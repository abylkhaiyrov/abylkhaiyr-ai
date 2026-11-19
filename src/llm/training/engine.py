from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass
from typing import Protocol

import numpy as np
import torch
from torch import Tensor, nn

from llm.core.config import ModelConfig, TrainConfig
from llm.core.errors import ResourceError, TrainingError
from llm.core.hardware import free_disk_gb
from llm.model.transformer import CausalLM
from llm.optimization.adamw import AdamW
from llm.optimization.grad import clip_grad_norm
from llm.optimization.memory import (
    ensure_ram_budget,
    is_out_of_memory,
    release_device_memory,
)
from llm.optimization.precision import make_grad_scaler
from llm.optimization.schedule import learning_rate_at
from llm.training.checkpoint import CheckpointManager, TrainingState
from llm.training.monitor import TrainingMonitor, perplexity
from llm.training.objectives import Batch, CausalLMObjective, Objective

MAX_CONSECUTIVE_SKIPS = 5
NO_DECAY_NDIM = 1


class BatchSource(Protocol):
    def next_batch(self) -> Batch: ...
    def set_batch_size(self, batch_size: int) -> None: ...
    def state_dict(self) -> dict[str, int]: ...
    def load_state_dict(self, state: dict[str, int]) -> None: ...


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_optimizer(model: nn.Module, config: TrainConfig) -> AdamW:
    settings = config.optimizer
    decay = [p for p in model.parameters() if p.requires_grad and p.ndim > NO_DECAY_NDIM]
    plain = [p for p in model.parameters() if p.requires_grad and p.ndim <= NO_DECAY_NDIM]
    groups = [
        {"params": decay, "weight_decay": settings.weight_decay},
        {"params": plain, "weight_decay": 0.0},
    ]
    return AdamW(
        groups,
        lr=settings.learning_rate,
        betas=(settings.beta1, settings.beta2),
        eps=settings.eps,
    )


@dataclass
class TrainingResult:
    final_step: int
    tokens_seen: int
    last_validation_loss: float
    best_validation_loss: float
    stopped_early: bool


class Trainer:
    def __init__(
        self,
        model: CausalLM,
        model_config: ModelConfig,
        config: TrainConfig,
        train_source: BatchSource,
        validation_source: BatchSource,
        device: torch.device,
        dtype: torch.dtype,
        checkpoints: CheckpointManager,
        monitor: TrainingMonitor,
        tokenizer_version: str,
        objective: Objective | None = None,
    ) -> None:
        self.model = model.to(device)
        self.objective = objective or CausalLMObjective(device, dtype)
        self.model_config = model_config
        self.config = config
        self.train_source = train_source
        self.validation_source = validation_source
        self.device = device
        self.dtype = dtype
        self.checkpoints = checkpoints
        self.monitor = monitor
        self.tokenizer_version = tokenizer_version
        self.optimizer = build_optimizer(self.model, config)
        self.scaler = make_grad_scaler(device, dtype)
        self.state = TrainingState(
            micro_batch_size=config.micro_batch_size,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
        )
        self.model.activation_checkpointing = config.activation_checkpointing
        self.train_source.set_batch_size(self.state.micro_batch_size)

    def resume(self) -> bool:
        path = self.checkpoints.latest_valid()
        if path is None:
            return False
        self.state, loader_state = self.checkpoints.restore(
            path, self.model, self.optimizer, self.scaler, self.device
        )
        self.train_source.set_batch_size(self.state.micro_batch_size)
        self.train_source.load_state_dict(loader_state)
        return True

    def _set_learning_rate(self) -> float:
        settings = self.config.optimizer
        rate = learning_rate_at(
            self.state.step,
            self.config.max_steps,
            settings.learning_rate,
            settings.warmup_steps,
            settings.min_lr_ratio,
        )
        for group in self.optimizer.param_groups:
            group["lr"] = rate
        return rate

    def _forward_loss(self, batch: Batch) -> Tensor:
        return self.objective.loss_sum(self.model, batch)

    def _to_device(self, batch: Batch) -> Batch:
        return tuple(t.to(self.device) for t in batch)

    def _accumulate(self) -> tuple[float, int]:
        micro_batches = [
            self._to_device(self.train_source.next_batch())
            for _ in range(self.state.gradient_accumulation_steps)
        ]
        total_tokens = sum(self.objective.weight(batch) for batch in micro_batches)
        if total_tokens == 0:
            raise TrainingError("batch contains no target tokens")
        loss_total = 0.0
        for batch in micro_batches:
            loss_sum = self._forward_loss(batch)
            scaled = loss_sum / total_tokens
            (self.scaler.scale(scaled) if self.scaler else scaled).backward()
            loss_total += float(loss_sum.detach().item())
        return loss_total / total_tokens, total_tokens

    def _reduce_micro_batch(self) -> None:
        current = self.state.micro_batch_size
        if not self.config.dynamic_batch_size or current <= self.config.min_micro_batch_size:
            raise ResourceError(
                "out of memory at the minimum micro batch size; use a more compact model profile"
            )
        reduced = max(current // 2, self.config.min_micro_batch_size)
        self.state.gradient_accumulation_steps *= math.ceil(current / reduced)
        self.state.micro_batch_size = reduced
        self.train_source.set_batch_size(reduced)

    def _optimizer_step(self) -> tuple[float, bool]:
        if self.scaler:
            self.scaler.unscale_(self.optimizer)
        grad_norm = float(
            clip_grad_norm(self.model.parameters(), self.config.optimizer.max_grad_norm)
        )
        if not math.isfinite(grad_norm):
            return grad_norm, False
        if self.scaler:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()
        return grad_norm, True

    def _train_step(self) -> tuple[float, int, float, bool]:
        snapshot = self.train_source.state_dict()
        while True:
            self.optimizer.zero_grad(set_to_none=True)
            try:
                loss, tokens = self._accumulate()
                break
            except (RuntimeError, torch.OutOfMemoryError) as error:
                if not is_out_of_memory(error):
                    raise
                self.optimizer.zero_grad(set_to_none=True)
                release_device_memory(self.device)
                self._reduce_micro_batch()
                self.train_source.load_state_dict(snapshot)
        grad_norm, applied = self._optimizer_step()
        return loss, tokens, grad_norm, applied

    @torch.inference_mode()
    def evaluate(self) -> float:
        was_training = self.model.training
        self.model.eval()
        saved = self.validation_source.state_dict()
        self.validation_source.load_state_dict({"epoch": 0, "cursor": 0})
        self.validation_source.set_batch_size(self.state.micro_batch_size)
        total, count = 0.0, 0
        for _ in range(self.config.eval_batches):
            batch = self._to_device(self.validation_source.next_batch())
            total += float(self._forward_loss(batch).item())
            count += self.objective.weight(batch)
        self.validation_source.load_state_dict(saved)
        self.model.train(was_training)
        return total / max(count, 1)

    def save_checkpoint(self) -> None:
        self.checkpoints.save(
            self.model,
            self.model_config,
            self.optimizer,
            self.scaler,
            self.state,
            self.train_source.state_dict(),
            self.tokenizer_version,
            self.device,
        )

    def train(self, stop_step: int | None = None) -> TrainingResult:
        config = self.config
        limit = min(config.max_steps, stop_step or config.max_steps)
        self.model.train()
        self.monitor.begin(self.state.step)
        window_loss, window_tokens, window_steps = 0.0, 0, 0
        window_start = time.monotonic()
        skips, last_validation, stopped_early = 0, float("nan"), False
        while self.state.step < limit:
            if free_disk_gb(config.checkpoint_dir) < config.min_free_disk_gb:
                raise ResourceError("not enough free disk space to continue training")
            ensure_ram_budget(config.max_ram_fraction)
            learning_rate = self._set_learning_rate()
            loss, tokens, grad_norm, applied = self._train_step()
            if not math.isfinite(loss) or not applied:
                skips += 1
                if skips >= MAX_CONSECUTIVE_SKIPS:
                    raise TrainingError("training diverged: repeated non-finite loss or gradients")
                continue
            skips = 0
            self.state.step += 1
            self.state.tokens_seen += tokens
            window_loss += loss * tokens
            window_tokens += tokens
            window_steps += 1
            if self.state.step % config.log_interval == 0:
                elapsed = time.monotonic() - window_start
                mean_loss = window_loss / window_tokens
                self.monitor.log(
                    self.state.step,
                    {
                        "train/loss": mean_loss,
                        "train/perplexity": perplexity(mean_loss),
                        "train/learning_rate": learning_rate,
                        "train/grad_norm": grad_norm,
                        "perf/tokens_per_second": window_tokens / elapsed,
                        "perf/steps_per_second": window_steps / elapsed,
                        "perf/tokens_seen": float(self.state.tokens_seen),
                        "perf/eta_seconds": self.monitor.eta_seconds(self.state.step),
                        "perf/micro_batch_size": float(self.state.micro_batch_size),
                        **{f"system/{k}": v for k, v in self.monitor.system_metrics().items()},
                    },
                )
                window_loss, window_tokens, window_steps = 0.0, 0, 0
                window_start = time.monotonic()
            if self.state.step % config.eval_interval == 0 or self.state.step == limit:
                last_validation = self.evaluate()
                self.monitor.log(
                    self.state.step,
                    {
                        "validation/loss": last_validation,
                        "validation/perplexity": perplexity(last_validation),
                    },
                )
                if self._should_stop(last_validation):
                    stopped_early = True
                    self.save_checkpoint()
                    break
            if self.state.step % config.checkpoint_interval == 0:
                self.save_checkpoint()
        if not stopped_early:
            self.save_checkpoint()
        self.monitor.close()
        return TrainingResult(
            self.state.step,
            self.state.tokens_seen,
            last_validation,
            self.state.best_validation_loss,
            stopped_early,
        )

    def _should_stop(self, validation_loss: float) -> bool:
        if validation_loss < self.state.best_validation_loss:
            self.state.best_validation_loss = validation_loss
            self.state.evaluations_without_improvement = 0
            return False
        self.state.evaluations_without_improvement += 1
        patience = self.config.early_stopping_patience
        return patience is not None and self.state.evaluations_without_improvement >= patience
