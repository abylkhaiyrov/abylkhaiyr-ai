import json
import os
from datetime import UTC, datetime
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from llm.studio.models import ManagedProcess
from tests.test_accounts import csrf, make_app, register, sign_in
from tests.test_studio_runtime import write_project

SAMPLES = Path("frontend/src/lib/contract-samples.json")
UPDATE = os.environ.get("UPDATE_CONTRACTS") == "1"


def signature(value):
    if isinstance(value, dict):
        return {key: signature(item) for key, item in sorted(value.items())}
    if isinstance(value, list):
        merged = {}
        for item in value:
            merged = merge(merged, signature(item))
        return [merged] if merged else []
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, (int, float)):
        return "number"
    return "string"


def merge(left, right):
    if not left:
        return right
    if isinstance(left, dict) and isinstance(right, dict):
        return {
            key: merge(left.get(key, "absent"), right.get(key, "absent"))
            for key in sorted({*left, *right})
        }
    if left == right:
        return left
    parts = {
        *(left.split("|") if isinstance(left, str) else [json.dumps(left)]),
        *(right.split("|") if isinstance(right, str) else [json.dumps(right)]),
    }
    return "|".join(sorted(parts))


@pytest.fixture
def project(tmp_path, model_config, tokenizer):
    write_project(
        tmp_path,
        {
            "stages": {
                "pretrain": {"status": "completed", "updated_at": "2026-01-01T00:00:00+00:00"}
            }
        },
    )
    flat = tmp_path / "experiments" / "demo_v3"
    nested = tmp_path / "experiments" / "v4" / "pretrain"
    for directory in (flat, nested):
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "metrics.jsonl").write_text(
            json.dumps({"step": 10, "train/loss": 3.0, "perf/tokens_seen": 100.0})
            + "\n"
            + json.dumps({"step": 10, "validation/loss": 3.1, "validation/perplexity": 22.0})
            + "\n"
        )
    report = tmp_path / "experiments" / "stage" / "final_eval.json"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(
        json.dumps(
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
            }
        )
    )
    (tmp_path / "experiments" / "registry.jsonl").write_text(
        json.dumps({"name": "probe", "results": {"validation_loss": 3.0}}) + "\n"
    )
    return make_app(model_config, tokenizer, tmp_path)


def collect(project, root):
    admin = sign_in(project, "admin", "admin")
    user, _ = register(project)
    studio = project.state.studio
    with studio.session_factory.begin() as database:
        database.add(
            ManagedProcess(
                id="proc-1",
                pid=None,
                kind="sft",
                status="completed",
                command="python -m llm.cli.main train",
                output_dir="experiments/demo_v3",
                started_at=datetime(2026, 1, 1, tzinfo=UTC).replace(tzinfo=None),
                completed_at=datetime(2026, 1, 1, 1, tzinfo=UTC).replace(tzinfo=None),
            )
        )
    groups = admin.get("/api/evaluation/groups").json()["items"]
    group = groups[0]["id"]
    detail = admin.get(f"/api/evaluation/groups/{group}").json()
    checkpoints = [row["checkpoint"] for row in detail["rows"]][:2]
    identifier = admin.get("/api/models/catalog").json()["items"][0]["id"]
    admin.put("/api/models/allowed", json={"ids": [identifier]}, headers=csrf(admin))
    calls = {
        "admin GET /api/runtime": admin.get("/api/runtime"),
        "admin GET /api/catalog/runs": admin.get("/api/catalog/runs"),
        "admin GET /api/catalog/runs/{id}": admin.get("/api/catalog/runs/run:demo_v3"),
        "admin GET /api/catalog/runs/{id} pipeline": admin.get(
            "/api/catalog/runs/pipeline:pipeline_tiny_v2"
        ),
        "admin GET /api/runs": admin.get("/api/runs"),
        "admin GET /api/pipelines": admin.get("/api/pipelines"),
        "admin GET /api/experiments": admin.get("/api/experiments"),
        "admin GET /api/processes": admin.get("/api/processes"),
        "admin GET /api/processes/{id}/events": admin.get("/api/processes/proc-1/events"),
        "admin GET /api/models/registry": admin.get("/api/models/registry"),
        "admin GET /api/models/catalog": admin.get("/api/models/catalog"),
        "admin GET /api/models/allowed": admin.get("/api/models/allowed"),
        "admin GET /api/processes/page": admin.get(
            "/api/processes/page", params={"group": "completed"}
        ),
        "admin GET /api/experiments/page": admin.get("/api/experiments/page"),
        "admin GET /api/models/checkpoints": admin.get("/api/models/checkpoints"),
        "admin GET /api/inference/profiles": admin.get("/api/inference/profiles"),
        "admin GET /api/evaluation/metrics": admin.get("/api/evaluation/metrics"),
        "admin GET /api/evaluation/groups": admin.get("/api/evaluation/groups"),
        "admin GET /api/evaluation/groups/{id}": admin.get(f"/api/evaluation/groups/{group}"),
        "admin GET /api/evaluation/compare": admin.get(
            "/api/evaluation/compare", params={"group": group, "checkpoint": checkpoints}
        ),
        "user GET /api/chat/models": user.get("/api/chat/models"),
        "user GET /api/chat/model": user.get("/api/chat/model"),
        "user GET /api/inference/profiles": user.get("/api/inference/profiles"),
    }
    for name, response in calls.items():
        assert response.status_code == 200, f"{name}: {response.status_code} {response.text[:200]}"
    return json.loads(
        json.dumps({name: response.json() for name, response in calls.items()}).replace(
            str(root), "<root>"
        )
    )


def test_http_contracts_match_the_committed_samples(project, tmp_path):
    current = collect(project, tmp_path)
    if UPDATE or not SAMPLES.is_file():
        SAMPLES.write_text(json.dumps(current, indent=1, sort_keys=True, default=str) + "\n")
    committed = json.loads(SAMPLES.read_text())
    assert sorted(committed) == sorted(current)
    drift = {
        name: (signature(committed[name]), signature(current[name]))
        for name in current
        if signature(committed[name]) != signature(current[name])
    }
    assert (
        drift == {}
    ), "API shape changed: regenerate with UPDATE_CONTRACTS=1 and verify the frontend guards"


def test_error_contracts(project):
    admin = sign_in(project, "admin", "admin")
    user, _ = register(project)
    anonymous = TestClient(project)
    for path in (
        "/api/runtime",
        "/api/catalog/runs",
        "/api/evaluation/groups",
        "/api/models/catalog",
    ):
        assert anonymous.get(path).status_code == 401
        assert user.get(path).status_code == 403
    assert admin.get("/api/catalog/runs/run:missing").json() == {"detail": "run_not_found"}
    assert admin.get("/api/evaluation/groups/none").json() == {"detail": "group_not_found"}
    assert admin.get("/api/processes/none/events").json() == {"detail": "process_not_found"}
    assert admin.get("/api/catalog/runs", params={"limit": 0}).status_code == 422
    assert admin.get("/api/catalog/runs", params={"limit": 999}).status_code == 422
    assert admin.get("/api/models/checkpoints", params={"offset": -1}).status_code == 422
    assert admin.put("/api/models/allowed", json={"ids": ["x"]}).status_code in {401, 403}
    assert (
        admin.put("/api/models/allowed", json={"ids": ["missing"]}, headers=csrf(admin)).status_code
        == 422
    )
    assert (
        admin.put(
            "/api/models/allowed", json={"ids": [], "extra": 1}, headers=csrf(admin)
        ).status_code
        == 422
    )
    assert (
        user.post(
            "/api/chats", headers=csrf(user), json={"generation_settings": {"profile": "ultra"}}
        ).status_code
        == 422
    )
    assert user.post("/api/chats", headers=csrf(user), json={"model_name": "x"}).json() == {
        "detail": "model_not_allowed"
    }


WS_SAMPLES = Path("frontend/src/lib/ws-samples.json")


def collect_events(project):
    admin = sign_in(project, "admin", "admin")
    seen = {}
    with admin.websocket_connect("/api/ws", headers={"origin": "http://testserver"}) as ws:
        ws.send_json({"type": "subscribe", "channels": ["studio", "chat"]})
        ws.send_json({"type": "ping"})
        for _ in range(40):
            event = ws.receive_json()
            seen.setdefault(event["type"], event)
            if {
                "hello",
                "chat.ready",
                "runtime.update",
                "models.update",
                "system.update",
                "pong",
            } <= set(seen):
                break
    return json.loads(json.dumps(seen))


def test_websocket_events_match_the_committed_samples(project):
    current = collect_events(project)
    assert {
        "hello",
        "chat.ready",
        "runtime.update",
        "models.update",
        "system.update",
        "pong",
    } <= set(current)
    if UPDATE or not WS_SAMPLES.is_file():
        WS_SAMPLES.write_text(json.dumps(current, indent=1, sort_keys=True, default=str) + "\n")
    committed = json.loads(WS_SAMPLES.read_text())
    drift = {
        name: item
        for name, item in current.items()
        if name in committed and signature(committed[name]) != signature(item)
    }
    assert drift == {}, "WebSocket event shape changed: regenerate with UPDATE_CONTRACTS=1"
