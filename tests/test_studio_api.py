import asyncio
from unittest.mock import AsyncMock, MagicMock

from llm.studio.events import make_event
from llm.studio.websocket import OUTBOX_LIMIT, Connection, WebSocketManager
from tests.test_studio import studio_client


def login(client):
    client.post("/api/auth/login", json={"username": "admin", "password": "admin"})
    return {"X-CSRF-Token": client.cookies.get("abylkhaiyr_csrf")}


def test_chat_pin_export_search_and_title(model_config, tokenizer, tmp_path):
    client = studio_client(model_config, tokenizer, tmp_path)
    headers = login(client)
    chat = client.post("/api/chats", headers=headers, json={"title": "Draft"}).json()
    assert chat["pinned"] is False
    pinned = client.patch(f"/api/chats/{chat['id']}", headers=headers, json={"pinned": True}).json()
    assert pinned["pinned"] is True
    studio = client.app.state.studio
    studio.add_message(1, chat["id"], "user", "needle in a haystack")
    found = client.get("/api/chats", params={"query": "needle"}).json()
    assert [item["id"] for item in found] == [chat["id"]]
    assert client.get("/api/chats", params={"query": "absent"}).json() == []
    markdown = client.get(f"/api/chats/{chat['id']}/export", params={"format": "md"})
    assert markdown.status_code == 200
    assert "needle in a haystack" in markdown.text
    assert "attachment" in markdown.headers["content-disposition"]
    exported = client.get(f"/api/chats/{chat['id']}/export", params={"format": "json"}).json()
    assert exported["messages"][0]["content"] == "needle in a haystack"


def test_runtime_runs_and_registry_endpoints(model_config, tokenizer, tmp_path):
    run = tmp_path / "experiments" / "demo"
    run.mkdir(parents=True)
    (run / "metrics.jsonl").write_text(
        '{"step":10,"train/loss":3.0}\n{"step":10,"validation/loss":3.1}\n{"step":20,"train/loss":2.5}\n'
    )
    client = studio_client(model_config, tokenizer, tmp_path)
    login(client)
    runs = client.get("/api/runs").json()
    assert runs[0]["name"] == "demo"
    assert runs[0]["best_validation_loss"] == 3.1
    rows = client.get("/api/runs/demo/metrics").json()
    assert rows[0] == {"step": 10, "train/loss": 3.0, "validation/loss": 3.1}
    assert client.get("/api/runs/missing/metrics").status_code == 404
    assert client.get("/api/runs/..%2Fetc/metrics").status_code in {404, 422}
    registry = client.get("/api/models/registry").json()
    assert set(registry["roles"]) == {"production", "best_evaluated", "latest_trained", "loaded"}
    runtime = client.get("/api/runtime").json()
    assert runtime["compute"]["busy"] is False
    assert set(runtime["models"]) == set(registry["roles"])


def test_model_switch_is_rejected_during_generation(model_config, tokenizer, tmp_path):
    client = studio_client(model_config, tokenizer, tmp_path)
    headers = login(client)
    client.app.state.model_manager.inference_lock.acquire()
    try:
        response = client.post(
            "/api/models/switch", headers=headers, json={"checkpoint_id": "checkpoints/x/step-1"}
        )
        assert response.status_code == 409
        assert response.json()["detail"] == "generation_active"
    finally:
        client.app.state.model_manager.inference_lock.release()


def test_outbox_drops_telemetry_but_disconnects_slow_consumers():
    async def scenario():
        manager = WebSocketManager(MagicMock(), MagicMock(), MagicMock(), scheduler=MagicMock())
        connection = Connection(MagicMock(close=AsyncMock()), 1)
        manager.connections.add(connection)
        for _ in range(OUTBOX_LIMIT):
            manager.enqueue(connection, make_event("pong"))
        manager.enqueue(connection, make_event("system.update", data={}))
        assert connection.outbox.qsize() == OUTBOX_LIMIT
        assert connection.closed is False
        manager.enqueue(connection, make_event("generation.delta", text="x"))
        assert connection.closed is True
        assert connection not in manager.connections

    asyncio.run(scenario())


def test_unknown_api_routes_return_json_not_the_spa_shell(model_config, tokenizer, tmp_path):
    dist = tmp_path / "frontend" / "dist"
    (dist / "assets").mkdir(parents=True)
    (dist / "index.html").write_text("<!doctype html><html></html>")
    client = studio_client(model_config, tokenizer, tmp_path)
    for method, path in (
        ("get", "/api"),
        ("get", "/v1"),
        ("get", "/api/nope"),
        ("post", "/api/nope"),
        ("get", "/v1/nope"),
        ("delete", "/api/chats/x/y/z"),
    ):
        response = getattr(client, method)(path)
        assert response.status_code == 404, path
        assert response.headers["content-type"].startswith("application/json")
        assert response.json() == {"detail": "endpoint_not_found"}
    method = client.post("/api/runtime")
    assert method.status_code == 405
    assert method.headers["allow"] == "GET"
    assert method.json() == {"detail": "method_not_allowed"}
    assert client.get("/api/runtime").status_code == 401
    assert client.get("/missing.js").status_code == 404
    assert client.get("/training", headers={"Accept": "application/json"}).status_code == 404
    page = client.get("/training")
    assert page.status_code == 200
    assert page.headers["content-type"].startswith("text/html")
    assert page.headers["cache-control"] == "no-store"


def test_unhandled_api_errors_are_json(model_config, tokenizer, tmp_path, monkeypatch):
    import torch
    from fastapi.testclient import TestClient

    from llm.api.server import create_app
    from llm.inference.engine import InferenceEngine
    from llm.model.transformer import CausalLM

    engine = InferenceEngine(CausalLM(model_config), tokenizer, device=torch.device("cpu"))
    app = create_app(engine, "test-model", studio_root=tmp_path)
    client = TestClient(app, raise_server_exceptions=False)
    login(client)
    monkeypatch.setattr(app.state.studio, "runtime_state", lambda: 1 / 0)
    response = client.get("/api/runtime")
    assert response.status_code == 500
    assert response.headers["content-type"].startswith("application/json")
    assert response.json() == {"detail": "internal_error"}


def test_regeneration_keeps_previous_answers_as_selectable_variants(
    model_config, tokenizer, tmp_path
):
    client = studio_client(model_config, tokenizer, tmp_path)
    headers = login(client)
    chat = client.post("/api/chats", headers=headers, json={}).json()
    studio = client.app.state.studio
    studio.add_message(1, chat["id"], "user", "question")
    first = studio.add_message(
        1, chat["id"], "assistant", "first answer", model_id="m", checkpoint_id="c1"
    )
    saved = studio.start_chat_generation(
        1, chat["id"], action="regenerate", target_message_id=first["id"]
    )
    assert saved is not None
    current = saved[1]
    studio.finalize_message(
        1, chat["id"], current["id"], "second answer", "completed", model_id="m", checkpoint_id="c2"
    )
    message = client.get(f"/api/chats/{chat['id']}").json()["messages"][-1]
    assert message["content"] == "second answer"
    assert [item["content"] for item in message["variants"]] == ["first answer"]
    selected = client.post(
        f"/api/chats/{chat['id']}/messages/{message['id']}/variants/0", headers=headers
    )
    assert selected.status_code == 200
    body = selected.json()
    assert body["content"] == "first answer"
    assert body["checkpoint_id"] == "c1"
    assert [item["content"] for item in body["variants"]] == ["second answer"]
    missing = client.post(
        f"/api/chats/{chat['id']}/messages/{message['id']}/variants/5", headers=headers
    )
    assert missing.status_code == 404


def test_frontend_only_calls_existing_endpoints(model_config, tokenizer, tmp_path):
    import re
    from pathlib import Path

    client = studio_client(model_config, tokenizer, tmp_path)
    paths = {
        (method.upper(), re.sub(r"\{[^/}]+\}", "{}", path))
        for path, item in client.app.openapi()["paths"].items()
        for method in item
    }
    source = Path("frontend/src/lib/api.ts").read_text()
    missing = []
    for line in source.splitlines():
        match = re.search(r"(?:request|http)<.*?>\(\s*[`\"]([^`\"]+)[`\"]", line)
        if not match:
            continue
        raw = re.sub(r"\$\{(query|category)[^`]*$", "", match.group(1))
        path = re.sub(r"\{[^/}]+\}", "{}", re.sub(r"\$\{[^}]*\}", "{}", "/api" + raw.split("?")[0]))
        verb = re.search(r"body\(\"(\w+)\"", line)
        key = (verb.group(1) if verb else "GET", path)
        if key not in paths:
            missing.append(key)
    assert missing == []


def test_session_discovery_exposes_api_version(model_config, tokenizer, tmp_path):
    client = studio_client(model_config, tokenizer, tmp_path)
    unauthorized = client.get("/api/auth/me")
    assert unauthorized.status_code == 401
    assert unauthorized.headers["x-studio-api-version"] == "1"

    login(client)
    session = client.get("/api/auth/me")
    assert session.status_code == 200
    assert session.headers["x-studio-api-version"] == "1"
    assert session.headers["cache-control"] == "no-store"
