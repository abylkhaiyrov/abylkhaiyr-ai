from __future__ import annotations

import json
from pathlib import Path

import pytest

from llm.data.feedback_curation import (
    FeedbackStore,
    StoreError,
    Verification,
    approval_decision,
    attach_correction,
    auto_error_class,
    capture,
    classify,
    curate,
    scrub,
    verify,
)
from llm.data.verified_corpus import prompt_key

CHECKPOINT = "checkpoints/tiny_v31_sft_stage11/step-00000120"
HUMAN = Verification(kind="human_reviewer", verifier="reviewer-1", evidence="checked against atlas")


def messages(text: str) -> list[dict[str, str]]:
    return [{"role": "user", "content": text}]


def verified_record(query: str, answer: str, correction: str, kind: str = "human_reviewer"):
    record = capture(messages(query), answer, CHECKPOINT, "medium", -1, True)
    assert record is not None
    record = classify(record, "wrong_fact")
    verification = (
        HUMAN
        if kind == "human_reviewer"
        else Verification(kind=kind, verifier="checker", evidence="rule")
    )
    return verify(attach_correction(record, correction, verification))


def test_capture_requires_explicit_consent() -> None:
    assert capture(messages("What is 2+2?"), "5", CHECKPOINT, "medium", -1, consent=False) is None


@pytest.mark.parametrize(
    "text",
    [
        "write to ivan.petrov@example.com",
        "call +7 701 123 45 67 now",
        "my password: hunter2hunter2",
        "key sk-abcdEFGH1234567890zz",
        "card 4111 1111 1111 1111",
        "IIN 960115300123",
        "see https://example.com/reset?token=abc",
    ],
)
def test_scrub_removes_personal_data_and_secrets(text: str) -> None:
    cleaned, count = scrub(text)
    assert count >= 1 and cleaned != text


def test_scrub_keeps_arithmetic_and_plain_text() -> None:
    for text in ("Compute 24894 - 90406 - 83567.", "17 + 25 = 42", "Столица Франции — Париж."):
        assert scrub(text) == (text, 0)


def test_heavy_personal_data_blocks_the_record() -> None:
    text = "a@b.co c@d.co e@f.co g@h.co"
    record = capture(messages(text), "ok", CHECKPOINT, "medium", -1, True)
    assert record is not None and record.stage == "blocked"


def test_a_model_cannot_verify_its_own_answer() -> None:
    record = capture(messages("Capital of France?"), "Berlin", CHECKPOINT, "medium", -1, True)
    classified = classify(record, "wrong_fact")
    bad = Verification(kind="human_reviewer", verifier=CHECKPOINT, evidence="self")
    with pytest.raises(ValueError):
        attach_correction(classified, "Paris", bad)
    with pytest.raises(ValueError):
        attach_correction(
            classified,
            "Paris",
            Verification(kind="human_reviewer", verifier="model:x", evidence="x"),
        )


def test_state_machine_rejects_skipped_stages() -> None:
    record = capture(messages("q"), "a", CHECKPOINT, "medium", -1, True)
    with pytest.raises(ValueError):
        attach_correction(record, "fixed", HUMAN)
    with pytest.raises(ValueError):
        classify(record, "not_a_class")
    with pytest.raises(ValueError):
        verify(classify(record, "other"))


def test_automatic_classes_are_limited_to_echo_and_language() -> None:
    echo = capture(messages("Describe autumn."), "Describe autumn", CHECKPOINT, "medium", -1, True)
    assert auto_error_class(echo, "en") == "echo_or_repeat"
    wrong = capture(messages("Hi"), "Привет, как дела", CHECKPOINT, "medium", -1, True)
    assert auto_error_class(wrong, "en") == "language_mismatch"
    fine = capture(messages("Hi"), "Hello there", CHECKPOINT, "medium", -1, True)
    assert auto_error_class(fine, "en") is None


def test_programmatic_verifiers_reject_wrong_corrections() -> None:
    assert (
        verified_record("What is 17 + 25?", "56", "17 + 25 = 42.", "arithmetic").stage == "verified"
    )
    rejected = verified_record("What is 17 + 25?", "56", "17 + 25 = 41.", "arithmetic")
    assert rejected.stage == "rejected" and rejected.block_reason == "arithmetic_failed"
    record = capture(messages("Capital of Italy?"), "Paris", CHECKPOINT, "medium", -1, True)
    corrected = attach_correction(
        classify(record, "wrong_fact"),
        "Rome",
        Verification(kind="knowledge_table", verifier="checker", evidence="table"),
    )
    assert verify(corrected, {prompt_key("Capital of Italy?"): "Rome"}).stage == "verified"
    assert verify(corrected, {}).stage == "rejected"


def test_curation_exports_only_verified_clean_non_benchmark_rows(tmp_path: Path) -> None:
    store = FeedbackStore(tmp_path / "feedback.jsonl")
    good = verified_record("Capital of Italy?", "Paris", "The capital of Italy is Rome.")
    store.append(good)
    store.append(verified_record("What can you do?", "x", "I answer questions."))
    unverified = capture(messages("Capital of Spain?"), "Paris", CHECKPOINT, "medium", -1, True)
    store.append(classify(unverified, "wrong_fact"))
    store.append(verified_record("Capital of Peru?", "Rome", "The capital of Peru is Lima."))
    store.append(verified_record("Capital of Peru?", "Madrid", "Peru's capital: Cusco."))
    manifest = curate(store, tmp_path, "v1", {prompt_key("What can you do?")})
    rows = [
        json.loads(line) for line in (tmp_path / "corrections.v1.jsonl").read_text().splitlines()
    ]
    assert [row["messages"][0]["content"] for row in rows] == ["Capital of Italy?"]
    assert rows[0]["provenance"]["consent"] is True
    assert rows[0]["provenance"]["verification"]["verifier"] == "reviewer-1"
    assert manifest["rejected"]["benchmark_overlap"] == 1
    assert manifest["rejected"]["not_verified_or_no_consent"] == 1
    assert manifest["rejected"]["conflicting_corrections"] == 1
    with pytest.raises(StoreError):
        curate(store, tmp_path, "v1", set())


def evaluation(rate: float, ru: float, en: float, kk: float, sha: str = "abc") -> dict:
    return {
        "benchmark_sha256": sha,
        "manual_pass_rate": rate,
        "manual_pass_by_language": {"ru": ru, "en": en, "kk": kk},
    }


def test_approval_requires_independent_improvement_and_no_regression() -> None:
    baseline = evaluation(0.10, 0.1, 0.1, 0.1)
    assert approval_decision(baseline, evaluation(0.15, 0.2, 0.15, 0.1), 1.1)["approved"]
    assert (
        "no_improvement"
        in approval_decision(baseline, evaluation(0.10, 0.1, 0.1, 0.1), 1.0)["reasons"]
    )
    assert (
        "language_regression_kk"
        in approval_decision(baseline, evaluation(0.2, 0.3, 0.3, 0.0), 1.0)["reasons"]
    )
    assert (
        "retention_regression"
        in approval_decision(baseline, evaluation(0.2, 0.2, 0.2, 0.2), 1.4)["reasons"]
    )
    assert (
        "benchmark_hash_mismatch"
        in approval_decision(baseline, evaluation(0.2, 0.2, 0.2, 0.2, sha="other"), 1.0)["reasons"]
    )
