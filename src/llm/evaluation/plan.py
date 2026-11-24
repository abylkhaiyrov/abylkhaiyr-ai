from __future__ import annotations

from typing import Any

from pydantic import Field, model_validator

from llm.core.config import StrictModel
from llm.evaluation.campaign_suites import SUITES
from llm.evaluation.gates import Gate
from llm.evaluation.independence import IndependenceCheck

BASELINE = "baseline"
REPETITIONS = 10_000


class SuiteSpec(StrictModel):
    name: str
    label: str | None = None
    options: dict[str, Any] = Field(default_factory=dict)

    @property
    def key(self) -> str:
        return self.label or self.name


class EvaluationPlan(StrictModel):
    baseline: str
    suites: list[SuiteSpec] = Field(min_length=1)
    gates: list[Gate] = Field(default_factory=list)
    device: str = "mps"
    tokenizer: str = "data/tokenizer"
    repetitions: int = Field(default=REPETITIONS, ge=100)
    independence: IndependenceCheck | None = None

    @model_validator(mode="after")
    def check_suites(self) -> EvaluationPlan:
        unknown = [suite.name for suite in self.suites if suite.name not in SUITES]
        if unknown:
            raise ValueError(f"unknown evaluation suite: {', '.join(unknown)}")
        labels = [suite.key for suite in self.suites]
        if len(labels) != len(set(labels)):
            raise ValueError("evaluation suite labels must be unique")
        return self
