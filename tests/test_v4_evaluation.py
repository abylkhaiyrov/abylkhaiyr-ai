from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest

from llm.evaluation.v4_review import Review, accepted, metrics, proportion, script_matches


def test_wilson_reporting_includes_sample_size_and_handles_zero() -> None:
    assert proportion(0, 0)["wilson_95"] is None
    estimate = proportion(0, 5)
    assert estimate["n"] == 5
    assert estimate["wilson_95"][0] == 0
    assert estimate["wilson_95"][1] == pytest.approx(0.43448, abs=1e-5)


def test_manual_metrics_require_a_matching_review() -> None:
    rows = [
        {
            "id": "example",
            "language": "en",
            "category": "greeting",
            "stop": "stop",
            "response": "A real answer.",
            "response_sha256": "current",
            "prompt_echo": False,
            "repetition": 0.0,
        }
    ]
    assert metrics(rows)["human_rubric"]["conversational_pass"]["n"] == 0
    review = Review(
        id="example",
        response_sha256="current",
        ratings={"conversational_pass": True, "code_pass_at_1": None},
    )
    results = metrics(rows, [review])
    assert results["human_rubric"]["conversational_pass"]["successes"] == 1
    assert results["human_rubric"]["code_pass_at_1"]["n"] == 0
    stale = review.model_copy(update={"response_sha256": "old"})
    with pytest.raises(ValueError, match="review does not match prediction"):
        metrics(rows, [stale])


def test_single_answer_matching_requires_every_group() -> None:
    groups = [["walk"], ["evening", "night"]]
    assert accepted("I like to Walk in the evening.", groups)
    assert not accepted("I like to walk.", groups)


def test_distractor_echo_is_not_credited() -> None:
    assert accepted("carrot", [["carrot"]], ["apple", "pear"])
    assert not accepted("apple, carrot, pear", [["carrot"]], ["apple", "pear"])


def test_script_match_separates_languages() -> None:
    assert script_matches("en", "Hello there")
    assert not script_matches("en", "Привет")
    assert script_matches("ru", "Привет, как дела")
    assert not script_matches("ru", "Сәлем, қалайсың")
    assert script_matches("kk", "Сәлем, қалайсың")
    assert not script_matches("kk", "")


def test_natural_evaluation_set_is_balanced_and_unique() -> None:
    rows = [
        json.loads(line) for line in Path("configs/natural_eval_v4.jsonl").read_text().splitlines()
    ]
    assert len({row["id"] for row in rows}) == len(rows)
    cells = Counter((row["language"], row["category"]) for row in rows)
    assert set(cells.values()) == {3}
    assert {language for language, _ in cells} == {"en", "ru", "kk"}
    assert all(bool(row["accept"]) == (row["kind"] == "single") for row in rows)


def test_kappa_is_one_for_identical_and_zero_for_chance_level_ratings() -> None:
    from llm.evaluation.v4_review import cohens_kappa

    assert cohens_kappa([True, False, True, False], [True, False, True, False]) == 1.0
    assert cohens_kappa([True, True, False, False], [True, False, True, False]) == 0.0
    assert cohens_kappa([True, True], [True, True]) is None


def test_rater_agreement_uses_only_items_rated_by_both() -> None:
    from llm.evaluation.v4_review import rater_agreement

    first = {"a": {"coherence": True}, "b": {"coherence": False}, "c": {"coherence": True}}
    second = {"a": {"coherence": True}, "b": {"coherence": False}, "d": {"coherence": None}}
    result = rater_agreement(first, second)
    assert result["coherence"]["n"] == 2
    assert result["coherence"]["raw_agreement"] == 1.0


def test_verbatim_leak_needs_a_long_enough_exact_phrase() -> None:
    from llm.evaluation.v4_review import verbatim_leak

    answers = ["Red and blue. Red is warm while blue is cold."]
    assert verbatim_leak("Sure: red and blue. Red is warm while blue is cold!", answers)
    assert not verbatim_leak("Red and blue.", answers)
    assert not verbatim_leak("Something unrelated entirely.", answers)
