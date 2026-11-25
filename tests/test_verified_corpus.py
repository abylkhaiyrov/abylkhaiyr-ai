from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path

import pytest

from llm.data.verified_common import LANGUAGES, exchange, split_for
from llm.data.verified_corpus import (
    collect_rows,
    enforce_split_integrity,
    language_issues,
    language_ok,
    prompt_key,
    protected_keys,
    user_sequence_key,
    verify,
)
from llm.data.verified_knowledge import knowledge_rows
from llm.data.verified_tasks import HELD_OUT_ARITHMETIC, task_rows

BENCHMARK = Path("configs/acceptance/benchmark_v1.json")


@pytest.fixture(scope="module")
def verified() -> list[dict]:
    kept, _ = verify(collect_rows(), protected_keys(BENCHMARK))
    final, _ = enforce_split_integrity(kept)
    return final


def test_language_detector_separates_kazakh_from_ukrainian_and_kyrgyz() -> None:
    assert language_ok("Бүгін ауа райы өте жақсы, күн ашық.", "kk")
    assert not language_ok("Хто відкрив мікроскоп? Мікроскоп відкрив Роберт Гук у 1665 році.", "kk")
    assert not language_ok("Темир жөнүндө макал билесизби? Темирди кенден изде.", "kk")
    assert language_ok("Жылда неше ай бар?", "kk")
    assert not language_ok("Сәлем, қалайсың?", "ru")
    assert language_ok("Привет, как дела?", "ru")
    assert not language_ok("Привет, как дела?", "en")
    assert language_ok('How do you say "дом" in Russian?', "en")


def test_generators_are_deterministic() -> None:
    assert knowledge_rows() == knowledge_rows()
    assert task_rows() == task_rows()


def test_verify_removes_conflicts_overlap_and_echo() -> None:
    protected = {prompt_key("What can you do?")}
    rows = [
        exchange("g", "c", "en", "a", "What is two plus two?", "Four.", "t"),
        exchange("g", "c", "en", "b", "What is two plus two?", "Five.", "t"),
        exchange("g", "c", "en", "c", "What can you do?", "Many things.", "t"),
        exchange("g", "c", "en", "d", "Say it again please", "say it again please!", "t"),
        exchange("g", "c", "en", "e", "Что такое снег?", "Снег — это замёрзшая вода.", "t"),
    ]
    kept, rejected = verify(rows, protected)
    assert [row["entity"] for row in kept] == ["a"]
    assert rejected == {
        "conflicting_answer": 1,
        "benchmark_overlap": 1,
        "answer_echoes_question": 1,
        "language_user": 1,
    }


def test_no_benchmark_turn_sequence_is_in_the_corpus(verified: list[dict]) -> None:
    protected = protected_keys(BENCHMARK)
    assert all(user_sequence_key(row["messages"]) not in protected for row in verified)


def test_splits_are_independent_by_user_sequence(verified: list[dict]) -> None:
    owners: dict[str, set[str]] = defaultdict(set)
    for row in verified:
        owners[user_sequence_key(row["messages"])].add(row["split"])
    assert all(len(splits) == 1 for splits in owners.values())
    assert split_for("capital:Germany") == split_for("capital:Germany")


def test_every_message_passes_the_language_check(verified: list[dict]) -> None:
    assert not [row for row in verified if language_issues(row)]
    assert {row["language"] for row in verified} == set(LANGUAGES)


def test_rows_have_provenance_and_license(verified: list[dict]) -> None:
    assert all(row["provenance"] and row["license"] and row["source"] for row in verified)


def test_arithmetic_answers_are_computed_and_held_out_pairs_absent() -> None:
    symbols = {"×": lambda a, b: a * b, "+": lambda a, b: a + b, "-": lambda a, b: a - b}
    pattern = re.compile(r"(\d+) ([×+-]) (\d+) = (-?\d+)\.")
    checked = 0
    for row in task_rows():
        if not row["category"].startswith("arithmetic"):
            continue
        a, symbol, b, result = pattern.fullmatch(row["messages"][1]["content"]).groups()
        assert symbols[symbol](int(a), int(b)) == int(result)
        key = ("*" if symbol == "×" else symbol, int(a), int(b))
        assert key not in HELD_OUT_ARITHMETIC
        checked += 1
    assert checked > 1000


def test_memory_answers_reproduce_the_value_stated_earlier() -> None:
    names = 0
    for row in task_rows():
        if row["category"] != "remember_name":
            continue
        statement, _, _, answer = (message["content"] for message in row["messages"])
        value = re.search(r"(?:\w+)\W*$", answer.rstrip(".")).group(0)
        assert value in statement
        names += 1
    assert names > 100


def test_context_answers_follow_the_asked_entity() -> None:
    checked = 0
    for row in task_rows():
        if row["category"] != "profession":
            continue
        question, answer = row["messages"][0]["content"], row["messages"][1]["content"]
        if row["language"] == "en":
            asked = re.search(r"What is (\w+)'s job", question).group(1)
            job = re.search(rf"{asked} works as (an? \w+)", question).group(1)
            assert answer == f"{asked} works as {job}."
            checked += 1
    assert checked > 50


def test_missing_information_rows_never_state_a_value() -> None:
    for row in task_rows():
        if row["category"] == "answerability_missing":
            assert not re.search(r"\d", row["messages"][1]["content"])
            assert row["messages"][1]["content"].count(" ") > 3
