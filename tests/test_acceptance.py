from __future__ import annotations

from pathlib import Path

import pytest

from llm.evaluation.acceptance import (
    LANGUAGES,
    AcceptanceItem,
    Grade,
    aggregate,
    auto_check,
    load_benchmark,
)
from llm.reasoning.sandbox import run_python

BENCHMARK = Path("configs/acceptance/benchmark_v1.json")


def item(scenario: str, language: str = "en") -> AcceptanceItem:
    items, _ = load_benchmark(BENCHMARK)
    return next(entry for entry in items if entry.item_id == f"{scenario}/{language}")


def test_benchmark_is_balanced_and_stable() -> None:
    items, digest = load_benchmark(BENCHMARK)
    again, same = load_benchmark(BENCHMARK)
    assert digest == same and items == again
    assert len(items) == 99
    assert len({entry.item_id for entry in items}) == 99
    for language in LANGUAGES:
        assert sum(entry.language == language for entry in items) == 33


def test_every_check_is_defined_for_every_language() -> None:
    items, _ = load_benchmark(BENCHMARK)
    for entry in items:
        result = auto_check(entry, "x")
        assert result.kind == entry.check["kind"]


def test_lexical_check_accepts_variants_and_rejects_forbidden() -> None:
    assert auto_check(item("fact_france", "ru"), "Столица Франции — Париж.").passed
    assert not auto_check(item("fact_france", "ru"), "Столица Франции — Лондон.").passed
    assert auto_check(item("mt_pet", "en"), "Your dog is Max.").passed
    assert not auto_check(item("mt_pet", "en"), "Your dog is Max and the cat is Luna.").passed


def test_number_check_uses_word_boundaries_and_words() -> None:
    assert auto_check(item("math_add", "en"), "17 + 25 = 42.").passed
    assert not auto_check(item("math_add", "en"), "17 + 25 = 420.").passed
    assert auto_check(item("fact_week", "en"), "There are seven days.").passed
    assert auto_check(item("ctx_time", "ru"), "Библиотека открывается в 9 утра.").passed


def test_count_check_requires_exact_repetitions() -> None:
    assert auto_check(item("repeat_word", "en"), "hello, hello, hello").passed
    assert not auto_check(item("repeat_word", "en"), "hello, hello").passed
    assert not auto_check(item("repeat_word", "en"), "hello hello hello hello").passed


def test_refusal_check_flags_invented_numbers() -> None:
    assert auto_check(item("ans_missing", "en"), "The text does not say how old Anna is.").passed
    assert not auto_check(item("ans_missing", "en"), "Anna is 25. The text does not say.").passed
    assert not auto_check(item("ans_missing", "en"), "Anna lives in Greenfield.").passed
    assert auto_check(item("ans_fictional", "kk"), "Мен ол елді білмеймін.").passed


@pytest.mark.skipif(not run_python("print(1)").passed, reason="sandbox unavailable")
def test_code_check_executes_the_answer() -> None:
    good = "```python\ndef f(a):\n    return sorted(a)\n```"
    bad = "```python\ndef f(a):\n    return a\n```"
    broken = "```python\ndef f(a:\n```"
    assert auto_check(item("code_sort"), good).passed
    assert not auto_check(item("code_sort"), bad).passed
    assert not auto_check(item("code_sort"), broken).passed
    assert auto_check(item("code_add"), "def add(x, y):\n    return x + y").passed


def test_aggregate_reports_rates_by_language_and_cause() -> None:
    rows = [
        {"item_id": "a/ru", "language": "ru", "category": "fact", "auto": {"passed": True}},
        {"item_id": "b/en", "language": "en", "category": "fact", "auto": {"passed": None}},
    ]
    grades = {
        "a/ru": Grade("a/ru", {"relevance": 2, "correctness": 1}, "", "", True),
        "b/en": Grade("b/en", {"relevance": 0, "correctness": 0}, "echo_or_repeat", "", False),
    }
    summary = aggregate(rows, grades)
    assert summary["manual_pass_rate"] == 0.5
    assert summary["manual_pass_by_language"] == {"en": 0.0, "ru": 1.0}
    assert summary["causes"] == {"echo_or_repeat": 1}
    assert summary["auto_n"] == 1
