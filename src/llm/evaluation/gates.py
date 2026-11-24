from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from llm.core.config import StrictModel

GateKind = Literal["min_delta", "max_delta", "max_ratio", "min_value", "max_value"]
Bound = Literal["estimate", "interval"]


class Gate(StrictModel):
    name: str
    metric: str
    arm: str
    against: str
    kind: GateKind
    value: float
    bound: Bound = "estimate"
    required: bool = True


@dataclass(frozen=True)
class Measurement:
    candidate: float
    reference: float
    interval: tuple[float, float] | None = None
    n: int = 0


@dataclass(frozen=True)
class GateResult:
    gate: Gate
    passed: bool
    observed: float | None
    detail: str


def judge_gate(gate: Gate, measurement: Measurement | None) -> GateResult:
    if measurement is None:
        return GateResult(gate, False, None, f"{gate.metric}: no measurement for {gate.arm}")
    delta = measurement.candidate - measurement.reference
    use_interval = gate.bound == "interval" and measurement.interval is not None
    low, high = measurement.interval if measurement.interval is not None else (delta, delta)
    if gate.kind == "min_delta":
        observed = low if use_interval else delta
        passed = observed >= gate.value
        relation = ">="
    elif gate.kind == "max_delta":
        observed = high if use_interval else delta
        passed = observed <= gate.value
        relation = "<="
    elif gate.kind == "max_ratio":
        observed = measurement.candidate / measurement.reference
        passed = observed <= gate.value
        relation = "<="
    elif gate.kind == "min_value":
        observed = measurement.candidate
        passed = observed >= gate.value
        relation = ">="
    else:
        observed = measurement.candidate
        passed = observed <= gate.value
        relation = "<="
    label = "95% bound" if use_interval else "estimate"
    detail = (
        f"{gate.metric} {gate.kind} ({label}) {observed:+.4g} {relation} {gate.value:g}: "
        f"{'passed' if passed else 'failed'} (candidate {measurement.candidate:.4g}, "
        f"{gate.against} {measurement.reference:.4g}, n={measurement.n})"
    )
    return GateResult(gate, passed, observed, detail)


def decide(results: list[GateResult]) -> tuple[bool, list[str]]:
    required = [r for r in results if r.gate.required]
    failed = [r.detail for r in required if not r.passed]
    return not failed, failed or ["every required gate passed"]
