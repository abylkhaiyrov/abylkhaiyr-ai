from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import yaml
from pydantic import Field, model_validator

from llm.core.config import (
    ExperimentConfig,
    StrictModel,
    TrainConfig,
    load_experiment_config,
)
from llm.core.errors import ConfigError
from llm.evaluation.plan import EvaluationPlan
from llm.training.budget import BudgetLimits
from llm.training.preflight import CorpusCheck
from llm.training.resources import ResourceBudget


class ArmSpec(StrictModel):
    data_dir: str | None = None
    dataset_version: str | None = None
    mixture: dict[str, float] = Field(default_factory=dict)
    optimizer: dict[str, Any] = Field(default_factory=dict)
    train: dict[str, Any] = Field(default_factory=dict)


class StoppingCriteria(StrictModel):
    max_steps: int | None = Field(default=None, gt=0)
    require_complete_seeds: bool = True
    stages: list[int] = Field(default_factory=list)

    @model_validator(mode="after")
    def check_stages(self) -> StoppingCriteria:
        if self.stages != sorted(set(self.stages)) or any(step <= 0 for step in self.stages):
            raise ValueError("stages must be positive and strictly increasing")
        if self.stages and self.max_steps is not None and self.stages[-1] > self.max_steps:
            raise ValueError("stages must not exceed max_steps")
        return self


class ExperimentSpec(StrictModel):
    name: str
    base_config: str
    initial_checkpoint: str | None = None
    job_template: str = "{arm}-{seed}"
    seeds: list[int] = Field(min_length=1)
    arms: dict[str, ArmSpec] = Field(min_length=1)
    order: list[str] = Field(default_factory=list)
    reuse: dict[str, str] = Field(default_factory=dict)
    token_budget: int = Field(gt=0)
    tokens_per_step: int = Field(gt=0)
    train: dict[str, Any] = Field(default_factory=dict)
    optimizer: dict[str, Any] = Field(default_factory=dict)
    output_root: str
    checkpoint_root: str
    state_file: str = "series.json"
    frozen_inputs: dict[str, str] = Field(default_factory=dict)
    lock_file: str | None = None
    lock_labels: list[str] = Field(default_factory=list)
    corpus: CorpusCheck | None = None
    calibration_file: str | None = None
    evaluation: EvaluationPlan | None = None
    stopping: StoppingCriteria = Field(default_factory=StoppingCriteria)
    resources: ResourceBudget = Field(default_factory=ResourceBudget)
    limits: BudgetLimits

    @model_validator(mode="after")
    def check_references(self) -> ExperimentSpec:
        names = set(self.job_pairs())
        if self.order and set(self.order) != names:
            raise ValueError("order must list every job exactly once")
        if len(self.order) != len(set(self.order)):
            raise ValueError("order contains duplicates")
        if not set(self.reuse) <= names:
            raise ValueError("reuse refers to an unknown job")
        if set(self.lock_labels) - set(self.frozen_inputs):
            raise ValueError("lock_labels must be frozen inputs")
        return self

    def job_pairs(self) -> dict[str, tuple[str, int]]:
        return {
            self.job_template.format(arm=arm, seed=seed): (arm, seed)
            for seed in self.seeds
            for arm in self.arms
        }

    def job_names(self) -> list[str]:
        names = list(self.job_pairs())
        return list(self.order) if self.order else names

    @property
    def target_steps(self) -> int:
        derived = math.ceil(self.token_budget / self.tokens_per_step)
        return self.stopping.max_steps or derived

    def goal(self, until: int | None = None) -> int:
        if until is None:
            return self.target_steps
        if until not in {*self.stopping.stages, self.target_steps}:
            raise ConfigError(f"step {until} is not a registered stage")
        return until

    def stage_checkpoint(self, job: str, until: int | None) -> Path:
        base = self.checkpoint_dir(job)
        return base if until is None else base / f"step-{until:08d}"

    def checkpoint_dir(self, job: str) -> Path:
        return Path(self.reuse[job]) if job in self.reuse else Path(self.checkpoint_root) / job

    def output_dir(self, job: str) -> Path:
        return Path(self.output_root) / job

    @property
    def state_path(self) -> Path:
        return Path(self.output_root) / self.state_file

    @property
    def ledger_path(self) -> Path:
        return self.state_path.with_suffix(".budget.json")

    def runnable_jobs(self) -> list[str]:
        return [job for job in self.job_names() if job not in self.reuse]

    def job_configuration(self, job: str) -> ExperimentConfig:
        arm_name, seed = self.job_pairs()[job]
        arm = self.arms[arm_name]
        base = load_experiment_config(self.base_config)
        optimizer = base.train.optimizer.model_copy(update={**self.optimizer, **arm.optimizer})
        update: dict[str, Any] = {
            **self.train,
            **arm.train,
            "seed": seed,
            "init_from": self.initial_checkpoint,
            "max_steps": self.target_steps,
            "checkpoint_dir": str(self.checkpoint_dir(job)),
            "output_dir": str(self.output_dir(job)),
            "optimizer": optimizer,
        }
        if arm.data_dir is not None:
            update["data_dir"] = arm.data_dir
        if arm.dataset_version is not None:
            update["dataset_version"] = arm.dataset_version
        if arm.mixture:
            update["mixture"] = dict(arm.mixture)
        try:
            train = TrainConfig.model_validate({**base.train.model_dump(), **update})
        except ValueError as error:
            raise ConfigError(f"invalid training configuration for {job}: {error}") from error
        return base.model_copy(update={"name": f"{self.name}-{job}", "train": train})


def load_spec(path: str | Path) -> ExperimentSpec:
    try:
        payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        return ExperimentSpec.model_validate(payload)
    except (OSError, ValueError, yaml.YAMLError) as error:
        raise ConfigError(f"invalid experiment specification {path}: {error}") from error


def spec_digest(spec: ExperimentSpec) -> str:
    text = json.dumps(spec.model_dump(mode="json"), sort_keys=True)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
