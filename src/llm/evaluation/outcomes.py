from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from llm.evaluation.diagnostic_controls import Choice

WIN = 1


@dataclass(frozen=True)
class Outcome:
    id: str
    cluster: str
    group: str
    value: int

    def as_choice(self) -> Choice:
        return Choice(self.id, self.group, self.cluster, WIN, self.value)


@dataclass
class SuiteOutput:
    metrics: dict[str, float] = field(default_factory=dict)
    items: dict[str, list[list[Outcome]]] = field(default_factory=dict)
    inputs: dict[str, str] = field(default_factory=dict)
    records: list[dict[str, Any]] = field(default_factory=list)

    def add_rate(self, name: str, runs: Sequence[Sequence[Outcome]]) -> None:
        pooled = [outcome.value for run in runs for outcome in run]
        if not pooled:
            return
        self.metrics[name] = sum(pooled) / len(pooled)
        self.items[name] = [list(run) for run in runs]

    def to_json(self) -> dict[str, Any]:
        return {
            "metrics": self.metrics,
            "inputs": self.inputs,
            "records": self.records,
            "items": {
                name: [[vars(outcome) for outcome in run] for run in runs]
                for name, runs in self.items.items()
            },
        }

    @staticmethod
    def from_json(payload: dict[str, Any]) -> SuiteOutput:
        return SuiteOutput(
            payload["metrics"],
            {
                name: [[Outcome(**row) for row in run] for run in runs]
                for name, runs in payload["items"].items()
            },
            payload["inputs"],
            payload.get("records", []),
        )
