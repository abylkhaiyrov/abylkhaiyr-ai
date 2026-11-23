from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from itertools import chain
from typing import Any

import numpy as np

from llm.evaluation.capacity_metrics import Judged, paired_summary, winner
from llm.evaluation.diagnostic import EncodedOption, encode_options, score_encoded
from llm.evaluation.diagnostic_bank import Item
from llm.evaluation.v4_review import proportion
from llm.inference.engine import InferenceEngine

METHODOLOGY = "v3"
NEUTRAL_PREFIX = {"en": "Answer:", "ru": "Ответ:", "kk": "Жауап:"}
METHODS = (
    "normalized",
    "total",
    "option_only",
    "neutral",
    "calibrated",
    "mismatched",
    "mismatched_calibrated",
    "unigram",
    "shortest",
    "longest",
)
OPTIONS = 4


@dataclass(frozen=True)
class ControlRow:
    id: str
    task: str
    language: str
    family: str
    answer: int
    normalized: int
    total: int
    option_only: int
    neutral: int
    calibrated: int
    mismatched: int
    mismatched_calibrated: int
    unigram: int
    shortest: int
    longest: int
    lengths: tuple[int, ...]


@dataclass(frozen=True)
class Choice:
    id: str
    task: str
    family: str
    answer: int
    choice: int


def foreign_prompts(items: Sequence[Item]) -> dict[str, Item]:
    cells: dict[tuple[str, str], list[Item]] = defaultdict(list)
    for item in sorted(items, key=lambda entry: entry.id):
        cells[(item.language, item.task)].append(item)
    result: dict[str, Item] = {}
    for members in cells.values():
        for index, item in enumerate(members):
            result[item.id] = members[(index + 1) % len(members)]
    return result


def extremes(lengths: Sequence[int]) -> tuple[int, int]:
    return (
        min(range(len(lengths)), key=lengths.__getitem__),
        max(range(len(lengths)), key=lengths.__getitem__),
    )


def score_controls(
    engine: InferenceEngine, items: Sequence[Item], unigram: np.ndarray, batch_size: int = 8
) -> Iterable[ControlRow]:
    foreign = foreign_prompts(items)
    bos = [engine.tokenizer.bos_id]
    for start in range(0, len(items), batch_size):
        batch = items[start : start + batch_size]
        variants = {
            "matched": [encode_options(engine, i.prefix, i.options, i.separator) for i in batch],
            "neutral": [
                encode_options(engine, NEUTRAL_PREFIX[i.language], i.options, i.separator)
                for i in batch
            ],
            "mismatched": [
                encode_options(engine, foreign[i.id].prefix, i.options, i.separator) for i in batch
            ],
        }
        variants["option_only"] = [
            [EncodedOption(bos, option.continuation) for option in group]
            for group in variants["matched"]
        ]
        scored = {
            name: score_encoded(engine, list(chain.from_iterable(groups)))
            for name, groups in variants.items()
        }
        for index, item in enumerate(batch):
            window = slice(index * OPTIONS, (index + 1) * OPTIONS)
            cell = {name: values[window] for name, values in scored.items()}
            totals = {name: [value for value, _ in values] for name, values in cell.items()}
            lengths = tuple(count for _, count in cell["matched"])
            shortest, longest = extremes(lengths)
            means = {name: [v / c for v, c in values] for name, values in cell.items()}
            frequencies = [
                float(unigram[o.continuation].mean()) for o in variants["matched"][index]
            ]
            yield ControlRow(
                item.id,
                item.task,
                item.language,
                item.family,
                item.answer,
                winner(means["matched"]),
                winner(totals["matched"]),
                winner(means["option_only"]),
                winner(means["neutral"]),
                winner([m - n for m, n in zip(totals["matched"], totals["neutral"], strict=True)]),
                winner(means["mismatched"]),
                winner(
                    [m - n for m, n in zip(totals["mismatched"], totals["neutral"], strict=True)]
                ),
                winner(frequencies),
                shortest,
                longest,
                lengths,
            )


def choices(rows: Sequence[ControlRow], method: str) -> list[Choice]:
    return [Choice(r.id, r.task, r.family, r.answer, getattr(r, method)) for r in rows]


def accuracy(rows: Sequence[ControlRow], method: str) -> dict[str, Any]:
    wins = proportion(sum(getattr(r, method) == r.answer for r in rows), len(rows))
    wins["chance"] = 1 / OPTIONS
    return wins


def within_model(rows: Sequence[ControlRow], first: str, second: str, repetitions: int) -> dict:
    pair: tuple[Sequence[Judged], Sequence[Judged]] = (choices(rows, first), choices(rows, second))
    return paired_summary([pair], method="choice", repetitions=repetitions)


def across_models(
    first: Sequence[ControlRow], second: Sequence[ControlRow], method: str, repetitions: int
) -> dict:
    pair: tuple[Sequence[Judged], Sequence[Judged]] = (
        choices(first, method),
        choices(second, method),
    )
    return paired_summary([pair], method="choice", repetitions=repetitions)


def control_report(rows: Sequence[ControlRow], repetitions: int = 2_000) -> dict[str, Any]:
    by_task: dict[str, list[ControlRow]] = defaultdict(list)
    for row in rows:
        by_task[row.task].append(row)
    return {
        "methodology": METHODOLOGY,
        "items": len(rows),
        "accuracy": {method: accuracy(rows, method) for method in METHODS},
        "by_task": {
            task: {method: accuracy(group, method)["rate"] for method in METHODS}
            for task, group in sorted(by_task.items())
        },
        "question_use": within_model(rows, "mismatched", "normalized", repetitions),
        "prompt_gain_over_option_only": within_model(
            rows, "option_only", "normalized", repetitions
        ),
        "neutral_gain_over_option_only": within_model(rows, "option_only", "neutral", repetitions),
        "calibrated_vs_normalized": within_model(rows, "normalized", "calibrated", repetitions),
        "agreement_normalized_total": float(np.mean([r.normalized == r.total for r in rows])),
        "chosen_shortest": {
            method: float(np.mean([getattr(r, method) == r.shortest for r in rows]))
            for method in ("normalized", "total", "calibrated", "option_only")
        },
        "chosen_longest": {
            method: float(np.mean([getattr(r, method) == r.longest for r in rows]))
            for method in ("normalized", "total", "calibrated", "option_only")
        },
    }
