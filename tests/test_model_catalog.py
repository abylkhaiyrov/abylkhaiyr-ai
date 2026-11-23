import json

import pytest

from llm.studio.model_catalog import build_catalog, checkpoint_status, user_view
from tests.test_accounts import csrf, make_app, register, sign_in


@pytest.fixture
def app(model_config, tokenizer, tmp_path):
    return make_app(model_config, tokenizer, tmp_path)


def checkpoint(model_id, step, **override):
    base = {
        "id": f"checkpoints/{model_id}/step-{step}",
        "model_id": model_id,
        "step": step,
        "stage": "sft",
        "roles": [],
        "quality_failed": False,
        "verified": False,
        "evaluated": False,
        "parameters": 1000,
        "context_length": 256,
        "compatible": True,
        "available": True,
        "availability_reason": None,
    }
    return {**base, **override}


@pytest.mark.parametrize(
    ("override", "expected"),
    [
        ({"roles": ["production"], "verified": True}, "production"),
        ({"quality_failed": True, "evaluated": True}, "rejected"),
        ({"verified": True, "evaluated": True}, "validated"),
        ({"evaluated": True}, "candidate"),
        ({"roles": ["latest_trained"]}, "candidate"),
        ({}, "experimental"),
    ],
)
def test_checkpoint_status(override, expected):
    assert checkpoint_status(checkpoint("m", 1, **override)) == expected


def test_catalog_groups_checkpoints_and_picks_best_representative():
    models = [
        {**checkpoint("tiny_v4_sft", 100), "status": "experimental"},
        {**checkpoint("tiny_v4_sft", 50), "status": "validated"},
        {**checkpoint("tiny_v4_sft", 200), "status": "validated"},
        {**checkpoint("tiny_v3_sft", 10), "status": "rejected"},
    ]
    entries = {item["id"]: item for item in build_catalog(models, {"tiny_v4_sft", "tiny_v3_sft"})}
    assert len(entries) == 2
    best = entries["tiny_v4_sft"]
    assert best["representative_checkpoint"].endswith("step-200")
    assert best["checkpoint_count"] == 3
    assert best["quality_status"] == "validated"
    assert best["display_name"] == "Tiny v4"
    assert entries["tiny_v3_sft"]["allowed"] is False


def test_default_policy_exposes_only_production_models():
    models = [
        {**checkpoint("a", 1), "status": "validated"},
        {**checkpoint("b", 1), "status": "production"},
    ]
    assert [item["id"] for item in user_view(build_catalog(models, None))] == ["b"]


def test_user_view_hides_unavailable_and_technical_fields():
    models = [{**checkpoint("a", 1, available=False), "status": "validated"}]
    assert user_view(build_catalog(models, {"a"})) == []
    visible = user_view(build_catalog([{**checkpoint("a", 1), "status": "validated"}], {"a"}))
    assert "representative_checkpoint" not in visible[0]
    assert "checkpoint_count" not in visible[0]


def test_admin_controls_what_users_see(app):
    admin = sign_in(app, "admin", "admin")
    user, response = register(app)
    assert response.status_code == 201, response.text
    catalog = admin.get("/api/models/catalog").json()
    assert catalog
    identifier = catalog[0]["id"]
    assert user.get("/api/chat/models").json() == []
    assert user.get("/api/models/catalog").status_code == 403
    assert (
        user.put("/api/models/allowed", json={"ids": [identifier]}, headers=csrf(user)).status_code
        == 403
    )
    denied = admin.put("/api/models/allowed", json={"ids": ["missing"]}, headers=csrf(admin))
    assert denied.status_code == 422
    assert json.loads(denied.json()["detail"])["unknown"] == ["missing"]
    allowed = admin.put("/api/models/allowed", json={"ids": [identifier]}, headers=csrf(admin))
    assert allowed.status_code == 200
    assert next(item for item in allowed.json() if item["id"] == identifier)["allowed"] is True
    visible = user.get("/api/chat/models").json()
    assert [item["id"] for item in visible] == [identifier]
    assert "representative_checkpoint" not in visible[0]
    admin.put("/api/models/allowed", json={"ids": []}, headers=csrf(admin))
    assert user.get("/api/chat/models").json() == []


def test_registry_checkpoints_carry_lifecycle_status(app):
    admin = sign_in(app, "admin", "admin")
    models = admin.get("/api/models/registry").json()["models"]
    assert models
    assert all(
        item["status"] in {"production", "validated", "candidate", "experimental", "rejected"}
        for item in models
    )


def test_users_can_only_pick_models_the_admin_allowed(app):
    admin = sign_in(app, "admin", "admin")
    user, _ = register(app)
    identifier = admin.get("/api/models/catalog").json()[0]["id"]
    headers = csrf(user)
    denied = user.post("/api/chats", headers=headers, json={"model_name": identifier})
    assert (denied.status_code, denied.json()["model_name"]) == (201, None)
    admin.put("/api/models/allowed", json={"ids": [identifier]}, headers=csrf(admin))
    created = user.post("/api/chats", headers=headers, json={"model_name": identifier})
    assert created.status_code == 201
    assert created.json()["model_name"] == identifier
    studio = app.state.studio
    assert studio.resolve_user_checkpoint(identifier)
    admin.put("/api/models/allowed", json={"ids": []}, headers=csrf(admin))
    assert studio.resolve_user_checkpoint(identifier) is None
    missing = user.post("/api/chats", headers=headers, json={"model_name": "missing"})
    assert missing.json()["model_name"] is None
    assert user.post("/api/chats", headers=headers, json={}).status_code == 201
