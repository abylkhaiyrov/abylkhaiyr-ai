from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import Any, Protocol

from llm.evaluation.metrics import prompt_echo, repeated_ngram_ratio
from llm.inference.engine import InferenceEngine
from llm.inference.sampling import SamplingParams
from llm.reasoning.intent import classify_intent, classify_task
from llm.reasoning.sandbox import SandboxLimits, run_program
from llm.reasoning.tools import (
    answers_match,
    extract_code,
    extract_final_answer,
    extract_text_answer,
    is_valid_python,
    repair_arithmetic,
    run_code_tests,
    verify_annotations,
)
from llm.reasoning.triage import CLARIFICATION, MODEL_NEEDED, triage
from llm.tokenizer.bpe import Message

DECOMPOSE_INSTRUCTION = "Break the problem into short numbered steps. Do not solve it yet.\n\n"
STEP_PREFIX_CHARS = ".)- 0123456789"
FAILED_ANSWER = None
LONG_QUESTION_CHARS = 200
GENERAL_CANDIDATES = 2
EXECUTION_LIMITS = SandboxLimits(timeout_seconds=5.0, cpu_seconds=5, memory_mb=256)
TOOL_KINDS = frozenset({"math", "logic"})


@dataclass(frozen=True)
class Problem:
    question: str
    kind: str = "math"
    tests: str | None = None


@dataclass
class Budget:
    max_candidates: int = 4
    max_retries: int = 2
    max_new_tokens_total: int = 4096
    max_seconds: float = 60.0
    max_new_tokens_per_candidate: int = 256


@dataclass
class Candidate:
    text: str
    answer: str | None
    score: float
    verified: bool


@dataclass
class Solution:
    answer: str | None
    text: str
    verified: bool
    candidates: list[Candidate] = field(default_factory=list)
    subtasks: list[str] = field(default_factory=list)
    tokens_used: int = 0
    stopped_by_budget: bool = False
    tool: str | None = None
    status: str = "unverified"


GENERAL_MIN_QUALITY = 0.7
GENERAL_MIN_TOKENS = 3


class Verifier(Protocol):
    def verify(self, problem: Problem, text: str) -> Candidate: ...


class MathVerifier:
    def verify(self, problem: Problem, text: str) -> Candidate:
        answer = extract_final_answer(text)
        correct, total = verify_annotations(text)
        if answer is None:
            return Candidate(text, None, 0.0, False)
        consistency = correct / total if total else 0.5
        return Candidate(text, answer, consistency, total > 0 and correct == total)


class LogicVerifier:
    def verify(self, problem: Problem, text: str) -> Candidate:
        answer = extract_text_answer(text)
        if answer is None:
            return Candidate(text, None, 0.0, False)
        mentioned = answer.lower() in problem.question.lower()
        return Candidate(text, answer if mentioned else None, float(mentioned), mentioned)


class CodeVerifier:
    def verify(self, problem: Problem, text: str) -> Candidate:
        code = extract_code(text)
        if not is_valid_python(code):
            return Candidate(text, None, 0.0, False)
        if problem.tests is None:
            runs = run_program("python", code, EXECUTION_LIMITS).passed
            return Candidate(text, code if runs else None, 0.75 if runs else 0.1, False)
        result = run_code_tests(code, problem.tests)
        return Candidate(text, code if result.passed else None, float(result.passed), result.passed)


class GeneralVerifier:
    def verify(self, problem: Problem, text: str) -> Candidate:
        quality = 1.0 - repeated_ngram_ratio(text.split(), 2)
        usable = (
            len(text.split()) >= GENERAL_MIN_TOKENS
            and quality >= GENERAL_MIN_QUALITY
            and not prompt_echo(problem.question, text)
        )
        return Candidate(text, text if usable else None, quality if usable else 0.0, False)


class ReasoningEngine:
    def __init__(
        self,
        engine: InferenceEngine,
        verifiers: dict[str, Verifier] | None = None,
        use_solvers: bool = True,
        clarify: bool = True,
    ) -> None:
        self.engine = engine
        self.use_solvers = use_solvers
        self.clarify = clarify
        self.verifiers = verifiers or {
            "math": MathVerifier(),
            "code": CodeVerifier(),
            "logic": LogicVerifier(),
            "general": GeneralVerifier(),
        }

    def _triaged(
        self,
        problem: Problem,
        history: list[Message] | None,
        on_event: Callable[[dict[str, Any]], None] | None,
    ) -> Solution | None:
        tool_kind = self.use_solvers and problem.kind in TOOL_KINDS
        result = triage(problem.question, self.clarify and not history, tool_kind)
        if result.status == MODEL_NEEDED:
            if tool_kind:
                self._emit(on_event, "tool_execution", "completed", tool="solver", solved=False)
            return None
        if result.status == CLARIFICATION:
            self._emit(on_event, "clarification", "completed", missing=list(result.missing))
            return Solution(None, result.text, False, status=CLARIFICATION)
        self._emit(on_event, "tool_execution", "running", tool="solver")
        self._emit(on_event, "tool_execution", "completed", tool=result.tool, solved=True)
        candidate = Candidate(result.text, result.answer, 1.0, True)
        return Solution(
            result.answer, result.text, True, [candidate], tool=result.tool, status=result.status
        )

    def decompose(self, problem: Problem, budget: Budget) -> list[str]:
        params = SamplingParams(
            max_new_tokens=budget.max_new_tokens_per_candidate // 2, greedy=True
        )
        reply = self.engine.chat(
            [Message("user", DECOMPOSE_INSTRUCTION + problem.question)], params
        )
        steps = [line.strip(STEP_PREFIX_CHARS).strip() for line in reply.text.splitlines()]
        return [step for step in steps if step]

    def _sample(
        self,
        problem: Problem,
        params: SamplingParams,
        tokens_used: int,
        history: list[Message] | None = None,
    ) -> tuple[str, int, bool]:
        messages = list(history) if history else [Message("user", problem.question)]
        reply = self.engine.chat(messages, params)
        return reply.text, tokens_used + len(reply.token_ids), reply.finish_reason == "stop"

    @staticmethod
    def _emit(
        on_event: Callable[[dict[str, Any]], None] | None,
        stage: str,
        status: str,
        **data: Any,
    ) -> None:
        if on_event is not None:
            on_event({"stage": stage, "status": status, "data": data})

    def solve_auto(
        self,
        question: str,
        tests: str | None = None,
        budget: Budget | None = None,
        history: list[Message] | None = None,
        on_event: Callable[[dict[str, Any]], None] | None = None,
        cancel: threading.Event | None = None,
        temperature: float = 0.7,
        top_p: float = 0.95,
        top_k: int = 0,
        repetition_penalty: float = 1.0,
    ) -> Solution:
        kind = classify_task(question)
        self._emit(
            on_event,
            "task_classification",
            "completed",
            kind=kind,
            intent=classify_intent(question),
        )
        budget = budget or Budget()
        if kind == "general":
            budget = replace(
                budget,
                max_candidates=min(budget.max_candidates, GENERAL_CANDIDATES),
                max_retries=min(budget.max_retries, 1),
            )
        long_math = kind == "math" and len(question) > LONG_QUESTION_CHARS
        return self.solve(
            Problem(question, kind, tests),
            budget,
            decompose=long_math,
            history=history,
            on_event=on_event,
            cancel=cancel,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
        )

    def solve(
        self,
        problem: Problem,
        budget: Budget | None = None,
        decompose: bool = False,
        history: list[Message] | None = None,
        on_event: Callable[[dict[str, Any]], None] | None = None,
        cancel: threading.Event | None = None,
        temperature: float = 0.7,
        top_p: float = 0.95,
        top_k: int = 0,
        repetition_penalty: float = 1.0,
    ) -> Solution:
        budget = budget or Budget()
        verifier = self.verifiers[problem.kind]
        started = time.monotonic()
        solution = Solution(None, "", False)
        prompt_messages = list(history) if history else [Message("user", problem.question)]
        self._emit(
            on_event,
            "context_preparation",
            "completed",
            kind=problem.kind,
            message_count=len(prompt_messages),
            max_candidates=budget.max_candidates,
        )
        tooled = self._triaged(problem, history, on_event)
        if tooled is not None:
            self._emit(
                on_event,
                "answer_formation",
                "completed",
                verified=tooled.verified,
                candidates=len(tooled.candidates),
                tokens_used=0,
                stopped_by_budget=False,
                tool=tooled.tool,
                outcome=tooled.status,
            )
            return tooled
        if decompose and not (cancel is not None and cancel.is_set()):
            self._emit(on_event, "decomposition", "running")
            solution.subtasks = self.decompose(problem, budget)
            self._emit(
                on_event,
                "decomposition",
                "completed",
                subtasks=solution.subtasks,
                count=len(solution.subtasks),
            )
        attempts = budget.max_candidates + budget.max_retries
        for attempt in range(attempts):
            if cancel is not None and cancel.is_set():
                break
            over_time = time.monotonic() - started > budget.max_seconds
            over_tokens = solution.tokens_used >= budget.max_new_tokens_total
            if over_time or over_tokens:
                solution.stopped_by_budget = True
                break
            if attempt >= budget.max_candidates and any(c.verified for c in solution.candidates):
                break
            self._emit(
                on_event,
                "candidate_generation",
                "running",
                attempt=attempt + 1,
                total=attempts,
                greedy=attempt == 0,
            )
            params = SamplingParams(
                max_new_tokens=min(
                    budget.max_new_tokens_per_candidate,
                    budget.max_new_tokens_total - solution.tokens_used,
                ),
                greedy=attempt == 0,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                seed=attempt,
            )
            text, solution.tokens_used, finished = self._sample(
                problem, params, solution.tokens_used, prompt_messages
            )
            self._emit(
                on_event,
                "candidate_generation",
                "completed",
                attempt=attempt + 1,
                total=attempts,
                tokens_used=solution.tokens_used,
            )
            if problem.kind == "math":
                self._emit(on_event, "math_computation", "running", attempt=attempt + 1)
                repaired = repair_arithmetic(text)
                correct, total = verify_annotations(repaired)
                self._emit(
                    on_event,
                    "math_computation",
                    "completed",
                    attempt=attempt + 1,
                    annotations_correct=correct,
                    annotations_total=total,
                    repaired=repaired != text,
                )
                text = repaired
            elif problem.kind == "code":
                self._emit(
                    on_event, "tool_execution", "running", attempt=attempt + 1, tool="python"
                )
                code = extract_code(text)
                valid_code = is_valid_python(code)
                self._emit(
                    on_event,
                    "tool_execution",
                    "completed",
                    attempt=attempt + 1,
                    tool="python",
                    valid_syntax=valid_code,
                    has_tests=problem.tests is not None,
                )
            self._emit(on_event, "verification", "running", attempt=attempt + 1, kind=problem.kind)
            candidate = verifier.verify(problem, text)
            if problem.kind == "general" and not finished:
                candidate = replace(candidate, score=candidate.score * 0.5)
            solution.candidates.append(candidate)
            self._emit(
                on_event,
                "verification",
                "completed",
                attempt=attempt + 1,
                verified=candidate.verified,
                score=round(candidate.score, 4),
            )
        self._select(solution, problem)
        solution.status = "solved" if solution.verified else "unverified"
        self._emit(
            on_event,
            "answer_formation",
            "completed",
            verified=solution.verified,
            candidates=len(solution.candidates),
            tokens_used=solution.tokens_used,
            stopped_by_budget=solution.stopped_by_budget,
        )
        return solution

    def _select(self, solution: Solution, problem: Problem) -> None:
        answered = [c for c in solution.candidates if c.answer is not FAILED_ANSWER]
        if not answered:
            solution.text = solution.candidates[0].text if solution.candidates else ""
            return
        verified = [c for c in answered if c.verified] or answered
        if problem.kind in {"code", "general"}:
            best = max(verified, key=lambda c: c.score)
        else:
            votes: dict[str, float] = {}
            for candidate in verified:
                answer = candidate.answer or ""
                key = next((k for k in votes if answers_match(answer, k)), answer)
                votes[key] = votes.get(key, 0.0) + 1 + candidate.score
            winner = max(votes, key=lambda key: votes[key])
            best = next(
                (
                    c
                    for c in verified
                    if (c.answer or "") == winner or answers_match(c.answer, winner)
                ),
                verified[0],
            )
        solution.answer, solution.text, solution.verified = best.answer, best.text, best.verified
