import pytest
import torch

from llm.evaluation.suites import TOOL_CASES, TOOL_HOLDOUT_CASES
from llm.inference.engine import InferenceEngine
from llm.model.transformer import CausalLM
from llm.reasoning.engine import Budget, Problem, ReasoningEngine
from llm.reasoning.solvers import solve_with_tools
from llm.reasoning.tools import answers_match


def correct(case) -> bool:
    solution = solve_with_tools(case.prompt)
    return solution is not None and (
        answers_match(solution.answer, case.reference)
        or solution.answer.strip().lower() == case.reference.lower()
    )


def test_solvers_answer_independent_phrasings_in_three_languages():
    assert sum(map(correct, TOOL_CASES)) / len(TOOL_CASES) >= 0.9
    assert sum(map(correct, TOOL_HOLDOUT_CASES)) / len(TOOL_HOLDOUT_CASES) >= 0.9


def test_solvers_never_return_a_wrong_answer_when_they_are_unsure():
    wrong = [
        case
        for case in (*TOOL_CASES, *TOOL_HOLDOUT_CASES)
        if solve_with_tools(case.prompt) is not None and not correct(case)
    ]
    assert not wrong


@pytest.mark.parametrize(
    ("question", "answer"),
    [
        ("Solve for x: 36*x + 19 = 2575.", "71"),
        ("Find the smaller root of x^2 - 7*x - 78 = 0.", "-6"),
        ("Solve the system 3*x + 1*y = -16, 1*x - 3*y = 8. Give the value of x.", "-4"),
        ("What is the smallest integer x satisfying 6*x - 13 > -24?", "-1"),
        ("Compute the definite integral of 8*x^2 from 0 to 1.", "8/3"),
        ("Let f(x) = 2*x^3 + 2*x. Compute f'(2).", "26"),
        ("Compute 8/9 - 6/11 and give the result as a simplified fraction.", "34/99"),
        ("What is the mean of the numbers 6, 25, 12, 31, 48, 5?", "127/6"),
    ],
)
def test_solver_known_answers(question, answer):
    assert solve_with_tools(question).answer == answer


def test_solvers_abstain_on_non_math_and_malformed_input():
    assert solve_with_tools("Tell me a story about a dragon") is None
    assert solve_with_tools("Solve for x: ((3x + = 5") is None
    assert solve_with_tools("x" * 2000) is None


def test_engine_uses_tool_without_sampling_and_falls_back_to_model(model, tokenizer, monkeypatch):
    engine = InferenceEngine(CausalLM(model.config), tokenizer, torch.device("cpu"))
    calls = []
    monkeypatch.setattr(engine, "chat", lambda *args, **kwargs: calls.append(1))
    events = []
    reasoner = ReasoningEngine(engine)
    solution = reasoner.solve(Problem("What is 23 + 19?", "math"), Budget(), on_event=events.append)
    assert solution.answer == "42" and solution.verified and solution.tool == "calculator"
    assert not calls and solution.tokens_used == 0
    assert any(event["stage"] == "tool_execution" for event in events)
    assert "#### 42" in solution.text
    assert (
        ReasoningEngine(engine, use_solvers=False)._solve_with_tool(
            Problem("What is 23 + 19?", "math"), None
        )
        is None
    )
