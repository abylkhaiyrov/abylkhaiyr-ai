from __future__ import annotations

from dataclasses import dataclass

from llm.reasoning.analysis import clarification_message, missing_information
from llm.reasoning.solvers import ToolSolution, solve_with_tools

CLARIFICATION = "clarification"
SOLVED = "solved"
MODEL_NEEDED = "model_needed"


@dataclass(frozen=True)
class Triage:
    status: str
    answer: str | None = None
    text: str = ""
    tool: str | None = None
    missing: tuple[str, ...] = ()


def triage(question: str, clarify: bool = True, use_tools: bool = True) -> Triage:
    if clarify:
        missing = missing_information(question)
        if missing:
            return Triage(
                CLARIFICATION, None, clarification_message(question, missing), None, tuple(missing)
            )
    if use_tools:
        solution: ToolSolution | None = solve_with_tools(question)
        if solution is not None:
            return Triage(solution.status, solution.answer, solution.text(), solution.tool)
    return Triage(MODEL_NEEDED)
