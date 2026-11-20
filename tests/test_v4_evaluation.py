from __future__ import annotations

import pytest

from llm.evaluation.v4_review import Review, metrics, proportion


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
