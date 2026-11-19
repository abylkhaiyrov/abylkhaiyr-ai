from __future__ import annotations

import re
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import Any, Protocol

from llm.evaluation.metrics import repeated_ngram_ratio
from llm.inference.engine import InferenceEngine
from llm.inference.sampling import SamplingParams
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
from llm.tokenizer.bpe import Message

DECOMPOSE_INSTRUCTION = "Break the problem into short numbered steps. Do not solve it yet.\n\n"
STEP_PREFIX_CHARS = ".)- 0123456789"
FAILED_ANSWER = None
LONG_QUESTION_CHARS = 200
GENERAL_CANDIDATES = 2


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


MATH_HINTS = re.compile(
    r"\d\s*[-+*/^=]\s*\d|\b(solve|compute|integral|derivative|probability|mean|determinant|equation|percent|сколько|вычисли|реши|есеп|тап)\b",
    re.IGNORECASE,
)
CODE_HINTS = re.compile(
    r"```|\b(function|def |class |SQL|query|python|java|javascript|typescript|функци|код)\b",
    re.IGNORECASE,
)
LOGIC_HINTS = re.compile(
    r"\b(taller|shorter|older|younger|heavier|faster|выше|ниже|старше)\b", re.IGNORECASE
)
GENERAL_MIN_QUALITY = 0.7
GENERAL_MIN_TOKENS = 3


def classify_task(question: str) -> str:
    if LOGIC_HINTS.search(question):
        return "logic"
    if CODE_HINTS.search(question):
        return "code"
    if MATH_HINTS.search(question):
        return "math"
    return "general"


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
            return Candidate(text, code, 0.5, False)
        result = run_code_tests(code, problem.tests)
        return Candidate(text, code if result.passed else None, float(result.passed), result.passed)


class GeneralVerifier:
    def verify(self, problem: Problem, text: str) -> Candidate:
        quality = 1.0 - repeated_ngram_ratio(text.split(), 2)
        usable = len(text.split()) >= GENERAL_MIN_TOKENS and quality >= GENERAL_MIN_QUALITY
        return Candidate(text, text if usable else None, quality, usable)


class ReasoningEngine:
    def __init__(
        self, engine: InferenceEngine, verifiers: dict[str, Verifier] | None = None
    ) -> None:
        self.engine = engine
        self.verifiers = verifiers or {
            "math": MathVerifier(),
            "code": CodeVerifier(),
            "logic": LogicVerifier(),
            "general": GeneralVerifier(),
        }

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
    ) -> tuple[str, int]:
        messages = list(history) if history else [Message("user", problem.question)]
        reply = self.engine.chat(messages, params)
        return reply.text, tokens_used + len(reply.token_ids)

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
        self._emit(on_event, "task_classification", "completed", kind=kind)
        budget = budget or Budget()
        if kind == "general":
            budget = replace(budget, max_candidates=GENERAL_CANDIDATES, max_retries=1)
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
                max_new_tokens=budget.max_new_tokens_per_candidate,
                greedy=attempt == 0,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                seed=attempt,
            )
            text, solution.tokens_used = self._sample(
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
                self._emit(on_event, "tool_execution", "running", attempt=attempt + 1, tool="python")
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
        if problem.kind == "code":
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
