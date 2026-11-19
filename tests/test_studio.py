import torch
from fastapi.testclient import TestClient

from llm.api.server import create_app
from llm.inference.engine import InferenceEngine
from llm.model.transformer import CausalLM


def studio_client(model_config, tokenizer, tmp_path):
    engine = InferenceEngine(CausalLM(model_config), tokenizer, device=torch.device("cpu"))
    return TestClient(create_app(engine, "test-model", studio_root=tmp_path))


def test_studio_requires_authentication(model_config, tokenizer, tmp_path):
    client = studio_client(model_config, tokenizer, tmp_path)
    response = client.get("/api/dashboard")
    assert response.status_code == 401, response.text


def test_training_start_rejects_active_pipeline(model_config, tokenizer, tmp_path, monkeypatch):
    client = studio_client(model_config, tokenizer, tmp_path)
    monkeypatch.setattr(client.app.state.studio.runtime, "pipeline", lambda: {"pid": 42})
    client.post("/api/auth/login", json={"username": "admin", "password": "admin"})
    token = client.cookies.get("abylkhaiyr_csrf")
    response = client.post(
        "/api/training",
        headers={"X-CSRF-Token": token},
        json={"config_path": "configs/debug.yaml"},
    )
    assert response.status_code == 400


def test_dashboard_does_not_report_old_metrics_as_active(model_config, tokenizer, tmp_path):
    metrics_dir = tmp_path / "experiments" / "old"
    metrics_dir.mkdir(parents=True)
    (metrics_dir / "metrics.jsonl").write_text('{"step": 99, "train/loss": 1.2}\n')
    client = studio_client(model_config, tokenizer, tmp_path)
    client.post("/api/auth/login", json={"username": "admin", "password": "admin"})
    dashboard = client.get("/api/dashboard").json()
    assert dashboard["inference_model"] == "test-model"
    assert dashboard["training"]["step"] is None
    assert dashboard["training"]["train_loss"] is None


def test_studio_login_csrf_and_password_change(model_config, tokenizer, tmp_path):
    client = studio_client(model_config, tokenizer, tmp_path)
    invalid = client.post("/api/auth/login", json={"username": "admin", "password": "wrong"})
    assert invalid.status_code == 401
    login = client.post("/api/auth/login", json={"username": "admin", "password": "admin"})
    assert login.status_code == 200
    assert login.json()["requires_password_change"] is True
    assert client.get("/api/dashboard").status_code == 200
    assert client.post("/api/auth/logout").status_code == 403
    csrf_token = client.cookies.get("abylkhaiyr_csrf")
    changed = client.put(
        "/api/auth/password",
        headers={"X-CSRF-Token": csrf_token},
        json={"current_password": "admin", "new_password": "a secure local password"},
    )
    assert changed.status_code == 204
    assert (
        client.post("/api/auth/login", json={"username": "admin", "password": "admin"}).status_code
        == 401
    )
    assert (
        client.post(
            "/api/auth/login",
            json={"username": "admin", "password": "a secure local password"},
        ).status_code
        == 200
    )


def test_websocket_streams_and_restores_history(model_config, tokenizer, tmp_path, monkeypatch):
    client = studio_client(model_config, tokenizer, tmp_path)
    client.post("/api/auth/login", json={"username": "admin", "password": "admin"})
    csrf = client.cookies.get("abylkhaiyr_csrf")
    chat = client.post("/api/chats", headers={"X-CSRF-Token": csrf}, json={}).json()
    monkeypatch.setattr(InferenceEngine, "stream_chat", lambda *args: iter(["Привет", " мир"]))
    with client.websocket_connect("/api/ws", headers={"origin": "http://testserver"}) as ws:
        ws.send_json({"type": "subscribe", "channels": ["chat"]})
        ws.send_json(
            {
                "type": "generation.start",
                "request_id": "request-1",
                "chat_id": chat["id"],
                "content": "Hello",
            }
        )
        events = []
        while not events or events[-1]["type"] != "generation.completed":
            events.append(ws.receive_json())
        assert (
            "".join(event["text"] for event in events if event["type"] == "generation.delta")
            == "Привет мир"
        )
        assert [event["sequence_id"] for event in events] == list(range(1, len(events) + 1))
        ws.send_json(
            {
                "type": "generation.start",
                "request_id": "request-1",
                "chat_id": chat["id"],
                "content": "Hello",
            }
        )
        assert ws.receive_json()["type"] == "generation.snapshot"
    messages = client.get(f"/api/chats/{chat['id']}").json()["messages"]
    assert [message["content"] for message in messages] == ["Hello", "Привет мир"]


def test_websocket_rejects_cross_origin(model_config, tokenizer, tmp_path):
    import pytest
    from starlette.websockets import WebSocketDisconnect

    client = studio_client(model_config, tokenizer, tmp_path)
    client.post("/api/auth/login", json={"username": "admin", "password": "admin"})
    with (
        pytest.raises(WebSocketDisconnect),
        client.websocket_connect("/api/ws", headers={"origin": "https://example.com"}),
    ):
        pass


def test_websocket_cancellation_keeps_partial_answer(
    model_config, tokenizer, tmp_path, monkeypatch
):
    import time

    client = studio_client(model_config, tokenizer, tmp_path)
    client.post("/api/auth/login", json={"username": "admin", "password": "admin"})
    csrf = client.cookies.get("abylkhaiyr_csrf")
    chat = client.post("/api/chats", headers={"X-CSRF-Token": csrf}, json={}).json()

    def generate(*args):
        yield "first"
        time.sleep(0.2)
        yield "second"

    monkeypatch.setattr(InferenceEngine, "stream_chat", generate)
    with client.websocket_connect("/api/ws", headers={"origin": "http://testserver"}) as ws:
        ws.send_json({"type": "subscribe", "channels": ["chat"]})
        ws.send_json(
            {
                "type": "generation.start",
                "request_id": "cancel-1",
                "chat_id": chat["id"],
                "content": "Hello",
            }
        )
        while ws.receive_json()["type"] != "generation.delta":
            pass
        ws.send_json({"type": "generation.cancel", "request_id": "cancel-1"})
        result = ws.receive_json()
        assert result["type"] == "generation.cancelled"
        assert result["text"] == "first"
    messages = client.get(f"/api/chats/{chat['id']}").json()["messages"]
    assert messages[-1]["content"] == "first"
