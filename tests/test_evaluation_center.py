import json

import pytest

from llm.studio.evaluation_center import (
    IncompatibleEvaluations,
    compare,
    group_detail,
    infer_stage,
    list_groups,
    quality_gates,
    verdict,
)
from llm.studio.evaluation_metrics import BY_KEY, METRICS
from tests.test_accounts import csrf, make_app, register, sign_in  # noqa: F401


def write_report(root, name, entries):
    path = root / "experiments" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(entries))


@pytest.fixture
def project(tmp_path):
    write_report(
        tmp_path,
        "stage/final_eval.json",
        {
            "base": {
                "checkpoint": "checkpoints/a/base",
                "split": "test",
                "metrics": {"test_perplexity": 10.0, "conv_pass_rate": 0.1},
            },
            "sft": {
                "checkpoint": "checkpoints/a_sft/best",
                "split": "test",
                "metrics": {
                    "test_perplexity": 12.0,
                    "conv_pass_rate": 0.7,
                    "conv_repetition_ratio": 0.2,
                },
            },
            "other": {
                "checkpoint": "checkpoints/a_sft/last",
                "split": "test",
                "metrics": {"test_perplexity": 11.0, "conv_pass_rate": 0.69},
            },
        },
    )
    write_report(
        tmp_path,
        "alone_eval.json",
        {"checkpoint": "checkpoints/b/x", "metrics": {"test_perplexity": 9.0}},
    )
    return tmp_path


def test_metric_registry_has_unique_keys_and_directions():
    assert len({item.key for item in METRICS}) == len(METRICS)
    assert BY_KEY["test_perplexity"].direction == "lower"
    assert BY_KEY["conv_pass_rate"].direction == "higher"
    assert BY_KEY["math_bench_tool_accuracy"].tools is True


def test_reports_of_one_experiment_form_one_group_regardless_of_file(project):
    groups = list_groups(project, None, 0, 50)
    assert groups["total"] == 2
    stage = next(item for item in groups["items"] if item["experiment"] == "stage")
    assert stage["checkpoints"] == 3
    assert list_groups(project, "alone", 0, 50)["total"] == 1
    assert list_groups(project, None, 1, 1)["items"][0]["experiment"] in {"stage", "alone_eval"}


def test_missing_metrics_stay_missing_and_gates_report_unverified(project):
    stage = next(
        item for item in list_groups(project, None, 0, 50)["items"] if item["experiment"] == "stage"
    )
    detail = group_detail(project, stage["id"])
    base = next(row for row in detail["rows"] if row["checkpoint"].endswith("base"))
    assert "conv_repetition_ratio" not in base["results"]
    states = {gate["id"]: gate["status"] for gate in base["gates"]}
    assert states == {
        "instruction_stage": "not_verified",
        "conversational_pass_rate": "failed",
        "repetition": "not_verified",
    }
    sft = next(row for row in detail["rows"] if row["checkpoint"].endswith("best"))
    assert sft["stage"] == "sft"
    assert sft["gate_status"] == "passed"


def test_compare_computes_direction_aware_changes(project):
    stage = next(
        item for item in list_groups(project, None, 0, 50)["items"] if item["experiment"] == "stage"
    )
    result = compare(project, stage["id"], ["checkpoints/a/base", "checkpoints/a_sft/best"])
    metrics = {item["key"]: item for item in result["metrics"]}
    perplexity = metrics["test_perplexity"]["entries"][1]
    assert perplexity["delta"] == pytest.approx(2.0)
    assert perplexity["improved"] is False
    assert metrics["conv_pass_rate"]["entries"][1]["improved"] is True
    assert metrics["conv_pass_rate"]["entries"][0]["delta"] == 0
    assert metrics["conv_repetition_ratio"]["entries"][0]["value"] is None
    assert metrics["conv_repetition_ratio"]["entries"][1]["delta"] is None
    assert "confidence_intervals_not_computed" in result["caveats"]
    assert result["verdict"]["status"] == "leader"
    assert result["verdict"]["checkpoint"] == "checkpoints/a_sft/best"


def test_perplexity_alone_never_decides_a_winner():
    low = {"checkpoint": "a", "results": {"test_perplexity": 5.0}, "gate_status": "passed"}
    high = {"checkpoint": "b", "results": {"test_perplexity": 50.0}, "gate_status": "passed"}
    assert verdict([low, high])["status"] == "insufficient_data"
    close = [
        {"checkpoint": "a", "results": {"conv_pass_rate": 0.6}, "gate_status": "passed"},
        {"checkpoint": "b", "results": {"conv_pass_rate": 0.62}, "gate_status": "passed"},
    ]
    assert verdict(close)["status"] == "inconclusive"
    failing = [
        {"checkpoint": "a", "results": {"conv_pass_rate": 0.3}, "gate_status": "failed"},
        {"checkpoint": "b", "results": {"conv_pass_rate": 0.1}, "gate_status": "failed"},
    ]
    assert verdict(failing)["reason"] == "leader_fails_quality_gates"


def test_unknown_checkpoint_is_incompatible(project):
    stage = next(
        item for item in list_groups(project, None, 0, 50)["items"] if item["experiment"] == "stage"
    )
    with pytest.raises(IncompatibleEvaluations):
        compare(project, stage["id"], ["checkpoints/a/base", "checkpoints/b/x"])
    assert compare(project, "missing:00000000", ["a", "b"]) is None


def test_stage_inference_and_gates():
    assert infer_stage("checkpoints/tiny_sft_best/x") == "sft"
    assert infer_stage("checkpoints/tiny/step-1") is None
    assert (
        quality_gates("dpo", {"conv_pass_rate": 0.9, "conv_repetition_ratio": 0.9})[2]["status"]
        == "failed"
    )


def test_endpoints_enforce_admin_and_validate_input(tmp_path, model_config, tokenizer):
    write_report(
        tmp_path,
        "x/evaluation.json",
        {"checkpoint": "checkpoints/m/a", "metrics": {"test_perplexity": 3.0}},
    )
    app = make_app(model_config, tokenizer, tmp_path)
    admin = sign_in(app, "admin", "admin")
    user, _ = register(app)
    assert user.get("/api/evaluation/groups").status_code == 403
    groups = admin.get("/api/evaluation/groups").json()
    assert groups["total"] == 1
    identifier = groups["items"][0]["id"]
    assert admin.get(f"/api/evaluation/groups/{identifier}").json()["rows"][0]["results"] == {
        "test_perplexity": 3.0
    }
    assert admin.get("/api/evaluation/groups/none").status_code == 404
    assert (
        admin.get(
            "/api/evaluation/compare",
            params={"group": identifier, "checkpoint": ["checkpoints/m/a"]},
        ).status_code
        == 422
    )
    conflict = admin.get(
        "/api/evaluation/compare",
        params={"group": identifier, "checkpoint": ["checkpoints/m/a", "zzz"]},
    )
    assert conflict.status_code == 409
    assert admin.get("/api/evaluation/metrics").json()[0]["key"] == "test_perplexity"
    assert admin.get("/api/evaluation").status_code == 200
