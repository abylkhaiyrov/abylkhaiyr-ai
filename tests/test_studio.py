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


def test_model_manager_lifecycle_and_compatibility(model_config, tokenizer, tmp_path):
    from llm.training.checkpoint import CheckpointManager, TrainingState

    tokenizer.save(tmp_path / "data" / "tokenizer")
    model = CausalLM(model_config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    sft_dir = CheckpointManager(tmp_path / "checkpoints" / "demo_sft", keep=2).save(
        model,
        model_config,
        optimizer,
        None,
        TrainingState(step=20, tokens_seen=1000),
        {},
        tokenizer.version,
        torch.device("cpu"),
    )
    dpo_dir = CheckpointManager(tmp_path / "checkpoints" / "demo_dpo", keep=2).save(
        model,
        model_config,
        optimizer,
        None,
        TrainingState(step=40, tokens_seen=2000),
        {},
        tokenizer.version,
        torch.device("cpu"),
    )
    CheckpointManager(tmp_path / "checkpoints" / "bad_tok", keep=2).save(
        model,
        model_config,
        optimizer,
        None,
        TrainingState(step=10, tokens_seen=500),
        {},
        "wrong-version-hash",
        torch.device("cpu"),
    )
    client = studio_client(model_config, tokenizer, tmp_path)
    client.post("/api/auth/login", json={"username": "admin", "password": "admin"})
    csrf = client.cookies.get("abylkhaiyr_csrf")
    listed = client.get("/api/models").json()
    by_id = {item["checkpoint_id"]: item for item in listed}
    sft_id = str(sft_dir.relative_to(tmp_path))
    dpo_id = str(dpo_dir.relative_to(tmp_path))
    assert by_id[sft_id]["stage"] == "sft"
    assert by_id[sft_id]["compatible"] is True
    assert by_id[dpo_id]["stage"] == "dpo"
    bad_item = next(item for item in listed if item["name"] == "bad_tok")
    assert bad_item["compatible"] is False
    assert "tokenizer_version_mismatch" in bad_item["compatibility_issues"]

    switched = client.post(
        "/api/models/switch",
        headers={"X-CSRF-Token": csrf},
        json={"checkpoint_id": dpo_id, "device": "cpu"},
    )
    assert switched.status_code == 200
    assert switched.json()["checkpoint_id"] == dpo_id
    assert switched.json()["stage"] == "dpo"
    assert client.get("/api/models/active").json()["checkpoint_id"] == dpo_id

    unloaded = client.post("/api/models/unload", headers={"X-CSRF-Token": csrf})
    assert unloaded.status_code == 200
    assert unloaded.json()["loaded"] is False
    assert client.get("/api/models/active").json() is None


def test_websocket_persistent_idempotency_across_restart(
    model_config, tokenizer, tmp_path, monkeypatch
):
    client_one = studio_client(model_config, tokenizer, tmp_path)
    client_one.post("/api/auth/login", json={"username": "admin", "password": "admin"})
    csrf = client_one.cookies.get("abylkhaiyr_csrf")
    chat = client_one.post("/api/chats", headers={"X-CSRF-Token": csrf}, json={}).json()
    monkeypatch.setattr(InferenceEngine, "stream_chat", lambda *args: iter(["hello", " world"]))
    with client_one.websocket_connect("/api/ws", headers={"origin": "http://testserver"}) as ws:
        ws.send_json({"type": "subscribe", "channels": ["chat"]})
        ws.send_json(
            {
                "type": "generation.start",
                "request_id": "persist-req-1",
                "chat_id": chat["id"],
                "content": "Hi",
            }
        )
        while ws.receive_json()["type"] != "generation.completed":
            pass
    client_two = studio_client(model_config, tokenizer, tmp_path)
    client_two.post("/api/auth/login", json={"username": "admin", "password": "admin"})
    with client_two.websocket_connect("/api/ws", headers={"origin": "http://testserver"}) as ws:
        ws.send_json({"type": "subscribe", "channels": ["chat"]})
        ws.send_json(
            {
                "type": "generation.start",
                "request_id": "persist-req-1",
                "chat_id": chat["id"],
                "content": "Hi",
            }
        )
        snapshot = ws.receive_json()
        assert snapshot["type"] == "generation.snapshot"
        assert snapshot["status"] == "completed"
        assert snapshot["text"] == "hello world"
    messages = client_two.get(f"/api/chats/{chat['id']}").json()["messages"]
    assert len(messages) == 2
    assert messages[1]["model_id"] == "test-model"
    assert messages[1]["checkpoint_id"] is not None
    assert messages[1]["generation_params"] is not None
    assert messages[1]["model_config"] is not None


def test_scheduler_queue_cancel_and_ram_guard_during_training(
    model_config, tokenizer, tmp_path, monkeypatch
):
    import types

    client = studio_client(model_config, tokenizer, tmp_path)
    client.post("/api/auth/login", json={"username": "admin", "password": "admin"})
    csrf = client.cookies.get("abylkhaiyr_csrf")
    chat = client.post("/api/chats", headers={"X-CSRF-Token": csrf}, json={}).json()
    monkeypatch.setattr(client.app.state.studio.runtime, "pipeline", lambda: {"pid": 99})
    monkeypatch.setattr(
        "llm.studio.model_manager.psutil.virtual_memory",
        lambda: types.SimpleNamespace(available=32 * 1024 * 1024),
    )
    with client.websocket_connect("/api/ws", headers={"origin": "http://testserver"}) as ws:
        ws.send_json({"type": "subscribe", "channels": ["chat"]})
        ws.send_json(
            {
                "type": "generation.start",
                "request_id": "ram-guard-1",
                "chat_id": chat["id"],
                "content": "Hello",
            }
        )
        failed = ws.receive_json()
        assert failed["type"] == "generation.failed"
        assert failed["code"] == "insufficient_memory_during_training"

    monkeypatch.setattr(
        "llm.studio.model_manager.psutil.virtual_memory",
        lambda: types.SimpleNamespace(available=8 * 1024 * 1024 * 1024),
    )
    monkeypatch.setattr(InferenceEngine, "stream_chat", lambda *args: iter(["cpu ", "reply"]))
    with client.websocket_connect("/api/ws", headers={"origin": "http://testserver"}) as ws:
        ws.send_json({"type": "subscribe", "channels": ["chat"]})
        ws.send_json(
            {
                "type": "generation.start",
                "request_id": "cpu-fallback-1",
                "chat_id": chat["id"],
                "content": "Hello on CPU",
            }
        )
        events = []
        while not events or events[-1]["type"] != "generation.completed":
            events.append(ws.receive_json())
        assert any(
            e["type"] == "generation.queued" and e["queue_reason"] == "cpu_fallback_training_active"
            for e in events
        )
        assert events[-1]["device"] == "cpu"
        assert events[-1]["text"] == "cpu reply"


def test_reasoning_mode_and_chat_edit_regenerate_continue(
    model_config, tokenizer, tmp_path, monkeypatch
):
    from llm.inference.engine import Generation

    client = studio_client(model_config, tokenizer, tmp_path)
    client.post("/api/auth/login", json={"username": "admin", "password": "admin"})
    csrf = client.cookies.get("abylkhaiyr_csrf")
    chat = client.post("/api/chats", headers={"X-CSRF-Token": csrf}, json={}).json()
    monkeypatch.setattr(
        InferenceEngine,
        "chat",
        lambda *args, **kwargs: Generation("2 + 2 = <<2+2=4>>4\n#### 4", [1, 2, 3], "stop"),
    )
    with client.websocket_connect("/api/ws", headers={"origin": "http://testserver"}) as ws:
        ws.send_json({"type": "subscribe", "channels": ["chat"]})
        ws.send_json(
            {
                "type": "generation.start",
                "request_id": "reason-1",
                "chat_id": chat["id"],
                "content": "Compute 2 + 2",
                "mode": "reasoning",
            }
        )
        events = []
        while not events or events[-1]["type"] != "generation.completed":
            events.append(ws.receive_json())
        stages = [e["stage"] for e in events if e["type"] == "generation.stage"]
        assert "task_classification" in stages
        assert "candidate_generation" in stages
        assert "math_computation" in stages
        assert "verification" in stages
        assert "answer_formation" in stages

    detail = client.get(f"/api/chats/{chat['id']}").json()
    user_id = detail["messages"][0]["id"]
    assistant_id = detail["messages"][1]["id"]
    assert len(detail["messages"][1]["reasoning_trace"]) >= 5

    monkeypatch.setattr(
        InferenceEngine,
        "stream_chat_continuation",
        lambda *args, **kwargs: iter([" more"]),
    )
    with client.websocket_connect("/api/ws", headers={"origin": "http://testserver"}) as ws:
        ws.send_json({"type": "subscribe", "channels": ["chat"]})
        ws.send_json(
            {
                "type": "generation.start",
                "request_id": "continue-1",
                "chat_id": chat["id"],
                "action": "continue",
                "target_message_id": assistant_id,
            }
        )
        while ws.receive_json()["type"] != "generation.completed":
            pass
    continued = client.get(f"/api/chats/{chat['id']}").json()["messages"][1]["content"]
    assert continued.endswith(" more")

    monkeypatch.setattr(InferenceEngine, "stream_chat", lambda *args, **kwargs: iter(["edited!"]))
    with client.websocket_connect("/api/ws", headers={"origin": "http://testserver"}) as ws:
        ws.send_json({"type": "subscribe", "channels": ["chat"]})
        ws.send_json(
            {
                "type": "generation.start",
                "request_id": "edit-1",
                "chat_id": chat["id"],
                "action": "edit",
                "target_message_id": user_id,
                "content": "Updated question",
            }
        )
        while ws.receive_json()["type"] != "generation.completed":
            pass
    after_edit = client.get(f"/api/chats/{chat['id']}").json()["messages"]
    assert [m["content"] for m in after_edit] == ["Updated question", "edited!"]

    monkeypatch.setattr(InferenceEngine, "stream_chat", lambda *args, **kwargs: iter(["regen!"]))
    with client.websocket_connect("/api/ws", headers={"origin": "http://testserver"}) as ws:
        ws.send_json({"type": "subscribe", "channels": ["chat"]})
        ws.send_json(
            {
                "type": "generation.start",
                "request_id": "regen-1",
                "chat_id": chat["id"],
                "action": "regenerate",
                "target_message_id": after_edit[1]["id"],
            }
        )
        while ws.receive_json()["type"] != "generation.completed":
            pass
    after_regen = client.get(f"/api/chats/{chat['id']}").json()["messages"]
    assert [m["content"] for m in after_regen] == ["Updated question", "regen!"]


def test_scheduler_cancels_queued_request(model_config, tokenizer, tmp_path, monkeypatch):
    import threading
    import time

    client = studio_client(model_config, tokenizer, tmp_path)
    client.post("/api/auth/login", json={"username": "admin", "password": "admin"})
    csrf = client.cookies.get("abylkhaiyr_csrf")
    chat = client.post("/api/chats", headers={"X-CSRF-Token": csrf}, json={}).json()
    gate = threading.Event()

    def slow_generate(*args, **kwargs):
        yield "part1"
        gate.wait(timeout=2.0)
        yield "part2"

    monkeypatch.setattr(InferenceEngine, "stream_chat", slow_generate)
    with client.websocket_connect("/api/ws", headers={"origin": "http://testserver"}) as ws:
        ws.send_json({"type": "subscribe", "channels": ["chat"]})
        ws.send_json(
            {
                "type": "generation.start",
                "request_id": "first-running",
                "chat_id": chat["id"],
                "content": "First",
            }
        )
        while ws.receive_json()["type"] != "generation.delta":
            pass
        ws.send_json(
            {
                "type": "generation.start",
                "request_id": "second-queued",
                "chat_id": chat["id"],
                "content": "Second",
            }
        )
        queued_event = ws.receive_json()
        while queued_event["type"] != "generation.queued":
            queued_event = ws.receive_json()
        assert queued_event["request_id"] == "second-queued"
        assert queued_event["queue_reason"] == "another_generation_running"
        ws.send_json({"type": "generation.cancel", "request_id": "second-queued"})
        cancelled_event = ws.receive_json()
        while (
            cancelled_event["type"] != "generation.cancelled"
            or cancelled_event["request_id"] != "second-queued"
        ):
            cancelled_event = ws.receive_json()
        assert cancelled_event["reason"] == "cancelled_in_queue"
        gate.set()
        done_event = ws.receive_json()
        while (
            done_event["type"] != "generation.completed"
            or done_event["request_id"] != "first-running"
        ):
            done_event = ws.receive_json()
        time.sleep(0.05)
    messages = client.get(f"/api/chats/{chat['id']}").json()["messages"]
    assert [m["content"] for m in messages] == ["First", "part1part2"]
