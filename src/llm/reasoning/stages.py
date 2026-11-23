from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

PREPARATION = "preparation"
GENERATION = "generation"
TOOL = "tool"
FORMAT_CHECK = "format_check"
FACTUAL_CHECK = "factual_check"
SEMANTIC_CHECK = "semantic_check"

FORMAT = "format"
FACTUAL = "factual"

STAGE_PHASES = {
    "model_loading": PREPARATION,
    "task_classification": PREPARATION,
    "context_preparation": PREPARATION,
    "decomposition": PREPARATION,
    "clarification": PREPARATION,
    "candidate_generation": GENERATION,
    "answer_formation": GENERATION,
    "generation_timeout": GENERATION,
    "math_computation": TOOL,
    "tool_execution": TOOL,
}


def phase_of(stage: str, data: Mapping[str, Any]) -> str:
    if stage == "verification":
        return FACTUAL_CHECK if data.get("check") == FACTUAL else FORMAT_CHECK
    return STAGE_PHASES.get(stage, GENERATION)


def executed_checks(events: Iterable[Mapping[str, Any]]) -> dict[str, bool]:
    completed = [
        event
        for event in events
        if event.get("stage") == "verification" and event.get("status") == "completed"
    ]
    kinds = {event.get("data", {}).get("check") for event in completed}
    return {FORMAT: FORMAT in kinds, FACTUAL: FACTUAL in kinds, "semantic": False}
