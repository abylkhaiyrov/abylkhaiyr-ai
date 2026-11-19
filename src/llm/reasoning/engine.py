from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Protocol

from llm.inference.engine import InferenceEngine
from llm.inference.sampling import SamplingParams
from llm.reasoning.tools import (
    answers_match,
    extract_final_answer,
    run_code_tests,
    verify_annotations,
)
from llm.tokenizer.bpe import Message

DECOMPOSE_INSTRUCTION = "Break the problem into short numbered steps. Do not solve it yet.\n\n"
STEP_PREFIX_CHARS = ".)- 0123456789"
FAILED_ANSWER = None


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


class CodeVerifier:
    def verify(self, problem: Problem, text: str) -> Candidate:
        if problem.tests is None:
            return Candidate(text, None, 0.0, False)
        result = run_code_tests(text, problem.tests)
        return Candidate(text, text if result.passed else None, float(result.passed), result.passed)


class ReasoningEngine:
    def __init__(
        self, engine: InferenceEngine, verifiers: dict[str, Verifier] | None = None
    ) -> None:
        self.engine = engine
        self.verifiers = verifiers or {"math": MathVerifier(), "code": CodeVerifier()}

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
        self, problem: Problem, params: SamplingParams, tokens_used: int
    ) -> tuple[str, int]:
        reply = self.engine.chat([Message("user", problem.question)], params)
        return reply.text, tokens_used + len(reply.token_ids)

    def solve(
        self, problem: Problem, budget: Budget | None = None, decompose: bool = False
    ) -> Solution:
        budget = budget or Budget()
        verifier = self.verifiers[problem.kind]
        started = time.monotonic()
        solution = Solution(None, "", False)
        if decompose:
            solution.subtasks = self.decompose(problem, budget)
        attempts = budget.max_candidates + budget.max_retries
        for attempt in range(attempts):
            over_time = time.monotonic() - started > budget.max_seconds
            over_tokens = solution.tokens_used >= budget.max_new_tokens_total
            if over_time or over_tokens:
                solution.stopped_by_budget = True
                break
            if attempt >= budget.max_candidates and any(c.verified for c in solution.candidates):
                break
            params = SamplingParams(
                max_new_tokens=budget.max_new_tokens_per_candidate,
                greedy=attempt == 0,
                temperature=0.7,
                top_p=0.95,
                seed=attempt,
            )
            text, solution.tokens_used = self._sample(problem, params, solution.tokens_used)
            solution.candidates.append(verifier.verify(problem, text))
        self._select(solution, problem)
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
            best = next(c for c in verified if answers_match(c.answer, winner))
        solution.answer, solution.text, solution.verified = best.answer, best.text, best.verified
