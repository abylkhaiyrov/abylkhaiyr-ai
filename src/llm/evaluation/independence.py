from __future__ import annotations

import re
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

from llm.core.config import StrictModel
from llm.core.errors import ConfigError
from llm.data.binary import read_jsonl

WORD = re.compile(r"\w+", re.UNICODE)
SHINGLE = 4


class IndependenceCheck(StrictModel):
    evaluation: list[str]
    training: list[str]
    max_overlap: float = 0.5


@dataclass(frozen=True)
class Overlap:
    source: str
    question: str
    fraction: float
    exact: bool


def words(text: str) -> list[str]:
    return [word.casefold() for word in WORD.findall(text)]


def shingles(tokens: Sequence[str]) -> frozenset[tuple[str, ...]]:
    size = min(SHINGLE, len(tokens))
    return frozenset(tuple(tokens[i : i + size]) for i in range(len(tokens) - size + 1))


def user_messages(path: Path) -> Iterator[str]:
    for row in read_jsonl(path):
        for message in row["messages"]:
            if message["role"] == "user":
                yield message["content"]


def expand(paths: Iterable[str]) -> list[Path]:
    found: list[Path] = []
    for pattern in paths:
        base = Path(pattern)
        matches = sorted(base.parent.glob(base.name)) if "*" in pattern else [base]
        found.extend(matches)
    return found


def overlaps(check: IndependenceCheck) -> list[Overlap]:
    exact: set[tuple[str, ...]] = set()
    grams: set[tuple[str, ...]] = set()
    for path in expand(check.training):
        for text in user_messages(path):
            tokens = words(text)
            exact.add(tuple(tokens))
            grams |= shingles(tokens)
    found = []
    for path in expand(check.evaluation):
        for row in read_jsonl(path):
            question = row["question"] if "question" in row else row["messages"][-1]["content"]
            tokens = words(question)
            parts = shingles(tokens)
            fraction = len(parts & grams) / len(parts) if parts else 0.0
            found.append(Overlap(str(path), question, fraction, tuple(tokens) in exact))
    return found


def verify_independence(check: IndependenceCheck) -> None:
    offenders = [o for o in overlaps(check) if o.exact or o.fraction > check.max_overlap]
    if offenders:
        shown = "; ".join(f"{o.question!r} ({o.fraction:.2f})" for o in offenders[:3])
        raise ConfigError(f"evaluation questions overlap with training data: {shown}")
