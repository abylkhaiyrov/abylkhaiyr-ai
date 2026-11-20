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
