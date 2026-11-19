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
    assert client.post("/api/auth/login", json={"username": "admin", "password": "admin"}).status_code == 401
    assert (
        client.post(
            "/api/auth/login",
            json={"username": "admin", "password": "a secure local password"},
        ).status_code
        == 200
    )
