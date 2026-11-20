from __future__ import annotations

import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from llm.agent.tools import (
    RUNNER,
    TOOL_ARGUMENTS,
    Runner,
    ToolBox,
    parse_tool_call,
    run_in_sandbox,
)
from llm.inference.constrained import ToolGrammar
from llm.inference.engine import Generation, TokenConstraint
from llm.inference.sampling import SamplingParams
from llm.tokenizer.bpe import BPETokenizer, Message

SOLUTION = "solution.py"
TESTS = "test_solution.py"
STEP_TOKENS = 260
INVALID_CALL = '{"ok": false, "output": "invalid tool call"}'


class ChatEngine(Protocol):
    tokenizer: BPETokenizer

    def chat(
        self,
        messages: list[Message],
        params: SamplingParams,
        constraint: TokenConstraint | None = None,
    ) -> Generation: ...


@dataclass(frozen=True)
class AgentTask:
    identifier: str
    instruction: str
    visible_tests: str
    hidden_tests: str
    initial_solution: str | None = None


@dataclass
class AgentRun:
    task: str
    calls: int = 0
    valid_calls: int = 0
    tools: list[str] = field(default_factory=list)
    finished: bool = False
    visible_passed: bool = False
    hidden_passed: bool = False
    failure: str | None = None


def hidden_verdict(workspace: Path, hidden_tests: str, runner: Runner) -> bool:
    solution = workspace / SOLUTION
    if not solution.is_file():
        return False
    result = runner(solution.read_text(encoding="utf-8") + RUNNER.format(tests=hidden_tests))
    return result.passed


def run_agent(
    engine: ChatEngine,
    task: AgentTask,
    constrained: bool,
    max_steps: int = 8,
    runner: Runner = run_in_sandbox,
) -> AgentRun:
    run = AgentRun(task.identifier)
    messages = [Message("user", task.instruction)]
    params = SamplingParams(greedy=True, max_new_tokens=STEP_TOKENS)
    with tempfile.TemporaryDirectory(prefix="agent-run-") as directory:
        workspace = Path(directory)
        (workspace / TESTS).write_text(task.visible_tests, encoding="utf-8")
        if task.initial_solution is not None:
            (workspace / SOLUTION).write_text(task.initial_solution, encoding="utf-8")
        toolbox = ToolBox(workspace, frozenset({TESTS}), runner)
        for _ in range(max_steps):
            grammar = ToolGrammar(engine.tokenizer, TOOL_ARGUMENTS) if constrained else None
            generation = engine.chat(messages, params, grammar)
            run.calls += 1
            try:
                call = parse_tool_call(generation.text)
            except ValueError as error:
                run.failure = str(error)
                messages += [Message("assistant", generation.text), Message("tool", INVALID_CALL)]
                continue
            run.valid_calls += 1
            run.tools.append(call.tool)
            outcome = toolbox.call(call)
            if call.tool == "run_tests":
                run.visible_passed = outcome.ok
            messages += [Message("assistant", call.text()), Message("tool", outcome.text())]
            if outcome.finished:
                run.finished = True
                break
        run.hidden_passed = hidden_verdict(workspace, task.hidden_tests, runner)
    return run
