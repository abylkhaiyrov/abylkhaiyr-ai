from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from llm.inference.engine import InferenceEngine
from llm.inference.sampling import SamplingParams
from llm.reasoning.container import run_isolated
from llm.tokenizer.bpe import Message

FENCE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)


class AcceptanceCase(BaseModel):
    model_config = ConfigDict(extra="forbid")
    stdin: str = Field(max_length=4096)
    stdout: str = Field(max_length=4096)


class DeveloperTask(BaseModel):
    model_config = ConfigDict(extra="forbid")
    instruction: str = Field(min_length=1, max_length=2200)
    cases: list[AcceptanceCase] = Field(min_length=1, max_length=10)
    attempts: int = Field(default=3, ge=1, le=8)
    seconds: float = Field(default=90, ge=5, le=300)

    @model_validator(mode="after")
    def distinct_inputs(self) -> DeveloperTask:
        if len({case.stdin for case in self.cases}) != len(self.cases):
            raise ValueError("acceptance inputs must be distinct")
        return self


class Attempt(BaseModel):
    number: int
    passed_cases: int = 0
    failure: str | None = None


class DeveloperResult(BaseModel):
    status: Literal["passed", "budget_exhausted", "unavailable"]
    file: str | None = None
    attempts: list[Attempt]


def solve(
    engine: InferenceEngine, task: DeveloperTask, output: Path, image: str
) -> DeveloperResult:
    output.mkdir(parents=True, exist_ok=True)
    destination = output / "solution.py"
    if destination.exists():
        raise FileExistsError(destination)
    deadline = time.monotonic() + task.seconds
    attempts: list[Attempt] = []
    feedback = ""
    for number in range(1, task.attempts + 1):
        if time.monotonic() >= deadline:
            break
        prompt = (
            "Write a complete Python 3 command-line program. Standard library only. "
            "Return only Python code. Read stdin and write stdout. "
            f"{task.instruction}\n{feedback}"
        )
        params = SamplingParams(max_new_tokens=192, temperature=0.25, seed=number)
        chunks = []
        for piece in engine.stream_chat([Message("user", prompt)], params):
            chunks.append(piece)
            if time.monotonic() >= deadline:
                return DeveloperResult(status="budget_exhausted", attempts=attempts)
        response = "".join(chunks).strip()
        fence = FENCE.search(response)
        code = fence[1].strip() if fence else response
        attempt = Attempt(number=number)
        attempts.append(attempt)
        try:
            compile(code, "solution.py", "exec")
        except (SyntaxError, ValueError) as error:
            attempt.failure = f"Invalid Python: {error}"
        else:
            for case in task.cases:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    attempt.failure = "Time budget reached"
                    break
                result = run_isolated(code, case.stdin, min(remaining, 25), image)
                if result.status == "unavailable":
                    attempt.failure = result.stderr
                    return DeveloperResult(status="unavailable", attempts=attempts)
                wrong_output = result.stdout.strip() != case.stdout.strip()
                if result.status != "passed" or wrong_output:
                    reason = "unexpected stdout" if result.status == "passed" else result.status
                    attempt.failure = (
                        f"Acceptance case {attempt.passed_cases + 1} failed ({reason})."
                    )
                    break
                attempt.passed_cases += 1
        if attempt.passed_cases == len(task.cases):
            with destination.open("x", encoding="utf-8") as handle:
                handle.write(code + "\n")
            return DeveloperResult(status="passed", file=str(destination), attempts=attempts)
        feedback = f"Previous attempt failed. {attempt.failure or ''} Correct the program."
    return DeveloperResult(status="budget_exhausted", attempts=attempts)
