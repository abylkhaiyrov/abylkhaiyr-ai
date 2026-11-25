from __future__ import annotations

import re
from pathlib import Path

import pytest

from llm.data.verified_contrast import contrastive_pairs
from llm.data.verified_corpus import collect_rows, enforce_split_integrity, protected_keys, verify

BENCHMARK = Path("configs/acceptance/benchmark_v1.json")


@pytest.fixture(scope="module")
def pairs() -> list[dict]:
    kept, _ = verify(collect_rows(), protected_keys(BENCHMARK))
    rows, _ = enforce_split_integrity(kept)
    return contrastive_pairs(rows)


def of_kind(pairs: list[dict], kind: str, category: str | None = None) -> list[dict]:
    return [
        item
        for item in pairs
        if item["kind"] == kind and (category is None or item["category"].startswith(category))
    ]


def test_every_pair_shares_the_context_and_differs_in_the_answer(pairs: list[dict]) -> None:
    assert len(pairs) > 5000
    for item in pairs:
        assert item["chosen"][:-1] == item["rejected"][:-1]
        assert item["chosen"][-1]["content"] != item["rejected"][-1]["content"]
        assert item["chosen"][-1]["role"] == item["rejected"][-1]["role"] == "assistant"
    kinds = {item["kind"] for item in pairs}
    assert kinds == {
        "wrong_value",
        "other_entity",
        "other_relation",
        "distractor",
        "hallucination",
        "unjustified_refusal",
    }


def test_arithmetic_negatives_state_a_false_result(pairs: list[dict]) -> None:
    pattern = re.compile(r"(\d+) ([×+-]) (\d+) = (-?\d+)\.")
    checked = 0
    for item in of_kind(pairs, "wrong_value", "arithmetic"):
        a, symbol, b, wrong = pattern.fullmatch(item["rejected"][-1]["content"]).groups()
        truth = {"+": int(a) + int(b), "-": int(a) - int(b), "×": int(a) * int(b)}[symbol]
        assert int(wrong) != truth
        checked += 1
    assert checked > 100


def test_pet_distractor_names_the_other_pet(pairs: list[dict]) -> None:
    items = of_kind(pairs, "distractor", "remember_pet")
    assert items
    for item in items:
        statement = item["chosen"][0]["content"]
        right, wrong = item["chosen"][-1]["content"], item["rejected"][-1]["content"]
        names = [name for name in re.findall(r"[A-ZА-ЯӘҒҚҢӨҰҮҺІ]\w+", statement) if name in wrong]
        assert names and not set(names) <= set(re.findall(r"\w+", right))


def test_missing_information_negatives_invent_a_value(pairs: list[dict]) -> None:
    items = of_kind(pairs, "hallucination")
    assert items
    for item in items:
        assert "answerability" in item["category"]
        assert item["rejected"][-1]["content"] != item["chosen"][-1]["content"]


def test_answerable_contexts_get_a_refusal_negative(pairs: list[dict]) -> None:
    items = of_kind(pairs, "unjustified_refusal")
    assert items
    assert {item["category"] for item in items} <= {"profession", "opening_hours"}


def test_pairs_respect_the_entity_split(pairs: list[dict]) -> None:
    splits: dict[str, set[str]] = {}
    for item in pairs:
        key = " || ".join(m["content"] for m in item["chosen"] if m["role"] == "user")
        splits.setdefault(key, set()).add(item["split"])
    assert all(len(value) == 1 for value in splits.values())
