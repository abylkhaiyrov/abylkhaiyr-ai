import json

import pytest

from llm.studio.run_catalog import RunCatalog, pipeline_status, select_default, version_of
from tests.test_studio import studio_client
from tests.test_studio_api import login


def pipeline(name, running, statuses, updated):
    return {
        "id": name,
        "running": running,
        "model_name": name,
        "started_at": "2026-01-01T00:00:00+00:00",
        "updated_at": updated,
        "stages": [
            {"key": f"s{index}", "status": status, "output_dir": f"experiments/{name}_{index}"}
            for index, status in enumerate(statuses)
        ],
        "current_stage": "s0" if running else None,
        "overall": {"percent": 40.0},
        "state_dir": f"experiments/{name}",
        "checkpoint": None,
        "pid": 1,
    }


def run(name, updated, status="completed", max_steps=10, step=10):
    return {
        "name": name,
        "output_dir": f"experiments/{name}",
        "stage": "sft",
        "status": status,
        "step": step,
        "max_steps": max_steps,
        "percent": 100.0,
        "started_at": None,
        "updated_at": updated,
        "config_path": None,
        "validation_loss": 1.0,
        "best_validation_loss": 0.9,
    }


def build(pipelines=(), runs=(), processes=()):
    return RunCatalog(lambda: list(pipelines), lambda: list(runs), lambda: list(processes))


def test_version_extraction():
    assert version_of("tiny_v4_1_sft") == "v4.1"
    assert version_of("pipeline_tiny_v2") == "v2"
    assert version_of("baseline") is None


@pytest.mark.parametrize(
    ("running", "statuses", "expected"),
    [
        (True, ["running"], "running"),
        (False, ["completed", "skipped"], "completed"),
        (False, ["completed", "failed", "pending"], "failed"),
        (False, ["completed", "interrupted", "pending"], "stopped"),
        (False, ["completed", "completed", "pending"], "stopped"),
    ],
)
def test_pipeline_status(running, statuses, expected):
    assert (
        pipeline_status(pipeline("p", running, statuses, "2026-01-02T00:00:00+00:00")) == expected
    )


def test_newer_training_run_wins_over_stale_pipeline():
    catalog = build(
        [pipeline("pipeline_tiny_v2", False, ["completed"], "2026-01-02T00:00:00+00:00")],
        [run("tiny_v4_sft", "2026-03-01T00:00:00+00:00")],
    )
    assert catalog.default()["name"] == "tiny_v4_sft"


def test_active_run_beats_newer_finished_run():
    catalog = build(
        [pipeline("pipeline_tiny_v2", True, ["running"], "2026-01-02T00:00:00+00:00")],
        [run("tiny_v4_sft", "2026-03-01T00:00:00+00:00")],
    )
    assert catalog.default()["name"] == "pipeline_tiny_v2"
    assert catalog.summary()["active"]["status"] == "running"


def test_empty_catalog_has_no_default():
    assert select_default([]) is None
    assert build().summary() == {"active": None, "latest": None, "completed_count": 0, "total": 0}


def test_pipeline_stage_runs_are_not_duplicated():
    catalog = build(
        [pipeline("pipeline_x", False, ["completed", "completed"], "2026-01-02T00:00:00+00:00")],
        [
            run("pipeline_x_0", "2026-01-01T00:00:00+00:00"),
            run("other", "2026-01-01T00:00:00+00:00"),
        ],
    )
    assert sorted(item["name"] for item in catalog.entries()) == ["other", "pipeline_x"]


def test_managed_process_state_overrides_metrics_status():
    active = {"output_dir": "experiments/live", "status": "running", "error_reason": None}
    failed = {"output_dir": "experiments/broken", "status": "failed", "error_reason": "oom"}
    catalog = build(
        runs=[
            run("live", "2026-01-01T00:00:00+00:00", status="stopped", step=3),
            run("broken", "2026-01-01T00:00:00+00:00", status="stopped", step=3),
        ],
        processes=[active, failed],
    )
    by_name = {item["name"]: item for item in catalog.entries()}
    assert by_name["live"]["status"] == "running"
    assert (by_name["broken"]["status"], by_name["broken"]["failure_reason"]) == ("failed", "oom")


def test_page_filters_and_pagination():
    catalog = build(
        runs=[
            run("tiny_v3_sft", "2026-01-01T00:00:00+00:00"),
            run("tiny_v4_sft", "2026-02-01T00:00:00+00:00"),
            run("tiny_v4_1_sft", "2026-03-01T00:00:00+00:00"),
        ]
    )
    assert [i["name"] for i in catalog.page()["items"]] == [
        "tiny_v4_1_sft",
        "tiny_v4_sft",
        "tiny_v3_sft",
    ]
    assert catalog.page(version="v4")["total"] == 1
    assert catalog.page(query="V3")["items"][0]["name"] == "tiny_v3_sft"
    assert catalog.page(limit=1, offset=1)["items"][0]["name"] == "tiny_v4_sft"
    assert catalog.page(oldest_first=True)["items"][0]["name"] == "tiny_v3_sft"
    assert catalog.page(status="failed")["total"] == 0


def test_catalog_endpoints_and_runtime(model_config, tokenizer, tmp_path):
    first = tmp_path / "experiments" / "demo_v3"
    first.mkdir(parents=True)
    (first / "metrics.jsonl").write_text(json.dumps({"step": 10, "train/loss": 3.0}) + "\n")
    client = studio_client(model_config, tokenizer, tmp_path)
    login(client)
    page = client.get("/api/catalog/runs", params={"version": "v3"}).json()
    assert page["total"] == 1
    entry = page["items"][0]
    assert entry["id"] == "run:demo_v3"
    detail = client.get(f"/api/catalog/runs/{entry['id']}").json()
    assert detail["run"]["name"] == "demo_v3"
    assert client.get("/api/catalog/runs/run:missing").status_code == 404
    runtime = client.get("/api/runtime").json()
    assert runtime["runs"]["latest"]["id"] == "run:demo_v3"
    assert runtime["runs"]["active"] is None
    assert client.get("/api/catalog/runs", params={"limit": 0}).status_code == 422


def test_catalog_requires_admin(model_config, tokenizer, tmp_path):
    client = studio_client(model_config, tokenizer, tmp_path)
    assert client.get("/api/catalog/runs").status_code == 401


def test_finished_process_releases_stale_pid(model_config, tokenizer, tmp_path):
    from datetime import UTC, datetime

    from llm.studio.models import ManagedProcess

    client = studio_client(model_config, tokenizer, tmp_path)
    login(client)
    studio = client.app.state.studio
    with studio.session_factory.begin() as database:
        database.add(
            ManagedProcess(
                id="old",
                pid=2_000_000_000,
                pid_started_at=1.0,
                kind="sft",
                status="running",
                command="python -m llm.cli.main train",
                output_dir="experiments/old",
                started_at=datetime(2026, 1, 1, tzinfo=UTC).replace(tzinfo=None),
            )
        )
    items = {item["id"]: item for item in client.get("/api/processes").json()}
    assert items["old"]["status"] == "failed"
    assert items["old"]["pid"] is None
    assert items["old"]["completed_at"] is not None
    assert items["old"]["error_reason"]
    events = client.get("/api/processes/old/events").json()
    assert events[-1]["message"] == "Process ended"
    assert client.get("/api/processes/missing/events").status_code == 404
