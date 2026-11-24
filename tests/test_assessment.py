from __future__ import annotations

import json

import pytest
import yaml

from llm.cli.main import main
from llm.core.errors import ConfigError
from llm.evaluation.gates import Gate, Measurement, decide, judge_gate
from llm.evaluation.outcomes import Outcome, SuiteOutput
from llm.evaluation.registry import load_registry
from llm.training.assessment import assess
from llm.training.spec import load_spec
from tests.test_campaign_integration import write_project

METRIC = "chat.unit.correct"
ITEMS = 40


def gate(kind: str, value: float, **changes) -> Gate:
    payload = {
        "name": "g",
        "metric": "m",
        "arm": "B",
        "against": "A",
        "kind": kind,
        "value": value,
    }
    return Gate.model_validate({**payload, **changes})


def test_minimum_improvement_gate_uses_the_interval_bound_when_requested():
    measurement = Measurement(0.6, 0.5, (-0.02, 0.22), 40)
    assert judge_gate(gate("min_delta", 0.05), measurement).passed
    assert not judge_gate(gate("min_delta", 0.05, bound="interval"), measurement).passed
    assert judge_gate(gate("min_delta", -0.05, bound="interval"), measurement).passed


def test_regression_ratio_and_value_gates():
    measurement = Measurement(0.31, 0.30, (-0.05, 0.07), 100)
    assert judge_gate(gate("max_delta", 0.02), measurement).passed
    assert not judge_gate(gate("max_delta", 0.02, bound="interval"), measurement).passed
    assert judge_gate(gate("max_ratio", 1.05), measurement).passed
    assert not judge_gate(gate("max_ratio", 1.02), measurement).passed
    assert judge_gate(gate("min_value", 0.3), measurement).passed
    assert judge_gate(gate("max_value", 0.4), measurement).passed


def test_missing_measurements_fail_and_optional_gates_do_not_block():
    missing = judge_gate(gate("min_delta", 0.0), None)
    assert not missing.passed and "no measurement" in missing.detail
    optional = judge_gate(gate("min_delta", 0.0, required=False), None)
    accepted, reasons = decide([optional])
    assert accepted and reasons == ["every required gate passed"]
    accepted, reasons = decide([missing])
    assert not accepted and "no measurement" in reasons[0]


def outputs_by_call(rates: list[float], inputs: dict[str, str] | None = None):
    calls = []

    def suite(engine, options) -> SuiteOutput:
        rate = rates[len(calls)]
        calls.append(rate)
        wins = round(rate * ITEMS)
        outcomes = [Outcome(f"c{i}", f"c{i}", "en", int(i < wins)) for i in range(ITEMS)]
        output = SuiteOutput(inputs=inputs or {"set": "hash"})
        output.add_rate(METRIC, [outcomes])
        output.metrics["chat.unit.repetition"] = 0.1
        return output

    return suite, calls


def evaluated_spec(tmp_path, tokenizer, model_config, monkeypatch, gates):
    monkeypatch.chdir(tmp_path)
    path = write_project(tmp_path, tokenizer, model_config)
    assert main(["experiment", "run", str(path)]) == 0
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    payload["evaluation"] = {
        "baseline": str(tmp_path / "ckpt" / "A-1"),
        "device": "cpu",
        "tokenizer": str(tmp_path / "tokenizer"),
        "repetitions": 200,
        "suites": [{"name": "chat_natural", "options": {}}],
        "gates": gates,
    }
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return load_spec(path)


def gate_payload(name: str, arm: str, against: str, kind: str, value: float, **extra):
    return {
        "name": name,
        "metric": METRIC,
        "arm": arm,
        "against": against,
        "kind": kind,
        "value": value,
        **extra,
    }


def test_assessment_accepts_only_when_every_registered_gate_passes(
    tmp_path, tokenizer, model_config, monkeypatch
):
    gates = [
        gate_payload("improves", "B", "A", "min_delta", 0.2, bound="interval"),
        gate_payload("no_regression", "A", "baseline", "min_delta", -0.01),
    ]
    spec = evaluated_spec(tmp_path, tokenizer, model_config, monkeypatch, gates)
    suite, calls = outputs_by_call([0.5, 0.5, 1.0])
    report = assess(spec, {"chat_natural": suite})
    assert report["accepted"] and report["promoted"] is False
    assert [g["passed"] for g in report["gates"]] == [True, True]
    assert report["arms"]["B"][METRIC] == pytest.approx(1.0)
    assert report["per_seed"]["improves"] == [
        {"seed": 1, "reference": 0.5, "candidate": 1.0, "delta": 0.5}
    ]
    assert report["arms"]["baseline"][METRIC] == pytest.approx(0.5)
    decision = json.loads((tmp_path / "out" / "evaluation" / "decision.json").read_text())
    assert decision["accepted"] is True
    registry = load_registry(tmp_path / "out" / "registry.jsonl")
    assert registry[-1]["promoted"] is False and registry[-1]["accepted"] is True
    again, replay = outputs_by_call([0.0, 0.0, 0.0])
    assert assess(spec, {"chat_natural": again})["accepted"] and not replay
    assert len(calls) == 3


def test_assessment_names_the_gate_that_rejects_the_result(
    tmp_path, tokenizer, model_config, monkeypatch
):
    gates = [
        gate_payload("improves", "B", "A", "min_delta", 0.2, bound="interval"),
        gate_payload("hold_quality", "B", "baseline", "min_delta", 0.0),
    ]
    spec = evaluated_spec(tmp_path, tokenizer, model_config, monkeypatch, gates)
    suite, _ = outputs_by_call([0.5, 0.5, 0.5])
    report = assess(spec, {"chat_natural": suite})
    assert not report["accepted"]
    assert len(report["reasons"]) == 1 and METRIC in report["reasons"][0]
    assert "failed" in report["reasons"][0] and "A" in report["reasons"][0]


def test_assessment_refuses_a_model_scored_on_different_evaluation_inputs(
    tmp_path, tokenizer, model_config, monkeypatch
):
    spec = evaluated_spec(
        tmp_path,
        tokenizer,
        model_config,
        monkeypatch,
        [gate_payload("g", "B", "A", "min_delta", 0.0)],
    )
    seen = iter([{"set": "one"}, {"set": "one"}, {"set": "two"}])

    def suite(engine, options) -> SuiteOutput:
        output = SuiteOutput(inputs=next(seen))
        output.add_rate(METRIC, [[Outcome("c0", "c0", "en", 1)]])
        return output

    with pytest.raises(ConfigError, match="inputs differ"):
        assess(spec, {"chat_natural": suite})


def test_unknown_suites_and_unregistered_gate_metrics_are_rejected(
    tmp_path, tokenizer, model_config, monkeypatch
):
    spec = evaluated_spec(tmp_path, tokenizer, model_config, monkeypatch, [])
    payload = spec.model_dump(mode="json")
    payload["evaluation"]["suites"] = [{"name": "invented", "options": {}}]
    with pytest.raises(ValueError, match="unknown evaluation suite"):
        type(spec).model_validate(payload)
    payload["evaluation"]["suites"] = [{"name": "perplexity", "options": {}}]
    payload["evaluation"]["gates"] = [gate_payload("g", "B", "A", "min_delta", 0.0)]
    with pytest.raises(ConfigError, match="unregistered suite"):
        assess(type(spec).model_validate(payload), {"perplexity": lambda e, o: SuiteOutput()})


def write_questions(path, questions):
    rows = [{"question": q, "messages": [{"role": "user", "content": q}]} for q in questions]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_independence_check_flags_exact_and_near_duplicate_questions(tmp_path):
    from llm.evaluation.independence import IndependenceCheck, overlaps, verify_independence

    training = tmp_path / "train.jsonl"
    write_questions(training, ["What is the capital of Germany?", "How tall is the Eiffel Tower?"])
    evaluation = tmp_path / "eval.jsonl"
    write_questions(
        evaluation,
        [
            "what is the capital of germany",
            "How tall is the Eiffel Tower in metres?",
            "Who painted the ceiling of the Sistine Chapel?",
        ],
    )
    check = IndependenceCheck(evaluation=[str(evaluation)], training=[str(training)])
    found = overlaps(check)
    assert [o.exact for o in found] == [True, False, False]
    assert found[1].fraction > 0.5 and found[2].fraction == 0.0
    with pytest.raises(ConfigError, match="overlap"):
        verify_independence(check)
    write_questions(evaluation, ["Who painted the ceiling of the Sistine Chapel?"])
    verify_independence(check)


def test_assessment_stops_before_scoring_when_evaluation_overlaps_training(
    tmp_path, tokenizer, model_config, monkeypatch
):
    spec = evaluated_spec(
        tmp_path,
        tokenizer,
        model_config,
        monkeypatch,
        [gate_payload("g", "B", "A", "min_delta", 0.0)],
    )
    questions = tmp_path / "q.jsonl"
    write_questions(questions, ["What is the capital of Germany?"])
    payload = spec.model_dump(mode="json")
    payload["evaluation"]["independence"] = {
        "evaluation": [str(questions)],
        "training": [str(questions)],
    }
    suite, calls = outputs_by_call([0.5, 0.5, 0.5])
    with pytest.raises(ConfigError, match="overlap"):
        assess(type(spec).model_validate(payload), {"chat_natural": suite})
    assert not calls


def test_cached_results_are_discarded_when_an_evaluation_file_changes(
    tmp_path, tokenizer, model_config, monkeypatch
):
    spec = evaluated_spec(
        tmp_path,
        tokenizer,
        model_config,
        monkeypatch,
        [gate_payload("g", "B", "A", "min_delta", 0.0)],
    )
    questions = tmp_path / "bank.jsonl"
    questions.write_text("one\n", encoding="utf-8")
    payload = spec.model_dump(mode="json")
    payload["evaluation"]["suites"][0]["options"] = {"cases": str(questions)}
    changed = type(spec).model_validate(payload)
    suite, calls = outputs_by_call([0.5] * 12)
    assess(changed, {"chat_natural": suite})
    first = len(calls)
    assess(changed, {"chat_natural": suite})
    assert len(calls) == first
    questions.write_text("two\n", encoding="utf-8")
    assess(changed, {"chat_natural": suite})
    assert len(calls) == 2 * first
