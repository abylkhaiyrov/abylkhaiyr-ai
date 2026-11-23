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
    assert best["display_name"] == "Tiny v4 sft"
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
    catalog = admin.get("/api/models/catalog").json()["items"]
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
    assert allowed.json() == {"ids": [identifier]}
    page = admin.get("/api/models/catalog", params={"q": identifier[:4], "limit": 1}).json()
    assert page["total"] >= 1 and len(page["items"]) == 1
    assert admin.get("/api/models/allowed").json() == {"ids": [identifier]}
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
    identifier = admin.get("/api/models/catalog").json()["items"][0]["id"]
    headers = csrf(user)
    denied = user.post("/api/chats", headers=headers, json={"model_name": identifier})
    assert (denied.status_code, denied.json()["detail"]) == (403, "model_not_allowed")
    admin.put("/api/models/allowed", json={"ids": [identifier]}, headers=csrf(admin))
    created = user.post("/api/chats", headers=headers, json={"model_name": identifier})
    assert created.status_code == 201
    assert created.json()["model_name"] == identifier
    studio = app.state.studio
    assert studio.resolve_user_checkpoint(identifier)
    admin.put("/api/models/allowed", json={"ids": []}, headers=csrf(admin))
    assert studio.resolve_user_checkpoint(identifier) is None
    missing = user.post("/api/chats", headers=headers, json={"model_name": "missing"})
    assert (missing.status_code, missing.json()["detail"]) == (403, "model_not_allowed")
    assert user.post("/api/chats", headers=headers, json={}).status_code == 201


def test_checkpoint_page_filters_sorts_and_paginates():
    from llm.studio.model_catalog import checkpoint_page

    models = [
        {
            "id": f"checkpoints/m{i}/step-1",
            "name": f"m{i}",
            "status": "validated" if i % 2 else "experimental",
            "stage": "sft",
            "created_at": f"2026-01-0{i + 1}",
        }
        for i in range(6)
    ]
    page = checkpoint_page(models, query=None, status=None, stage=None, offset=0, limit=4)
    assert page["total"] == 6
    assert [item["name"] for item in page["items"]] == ["m5", "m4", "m3", "m2"]
    rest = checkpoint_page(models, query=None, status=None, stage=None, offset=4, limit=4)
    assert [item["name"] for item in rest["items"]] == ["m1", "m0"]
    assert (
        checkpoint_page(models, query="M3", status=None, stage=None, offset=0, limit=10)["total"]
        == 1
    )
    assert (
        checkpoint_page(models, query=None, status="validated", stage=None, offset=0, limit=10)[
            "total"
        ]
        == 3
    )
    assert (
        checkpoint_page(models, query=None, status=None, stage="dpo", offset=0, limit=10)["items"]
        == []
    )


def test_checkpoint_endpoint_is_admin_only_and_paginated(app):
    admin = sign_in(app, "admin", "admin")
    user, _ = register(app)
    assert user.get("/api/models/checkpoints").status_code == 403
    page = admin.get("/api/models/checkpoints", params={"limit": 1}).json()
    assert len(page["items"]) <= 1 and page["total"] >= 1
    assert admin.get("/api/models/checkpoints", params={"limit": 0}).status_code == 422


def test_scheduler_rejects_forbidden_model_use_explicitly(app):
    import asyncio
    import threading

    from llm.studio.scheduler import GenerationRequest, GenerationScheduler

    admin = sign_in(app, "admin", "admin")
    user, _ = register(app)
    studio = app.state.studio
    identifier = admin.get("/api/models/catalog").json()["items"][0]["id"]
    admin.put("/api/models/allowed", json={"ids": [identifier]}, headers=csrf(admin))
    chat = user.post("/api/chats", headers=csrf(user), json={"model_name": identifier}).json()
    scheduler = GenerationScheduler(studio, app.state.model_manager, threading.Lock())
    record = studio.chat(2, chat["id"], None, False)
    other = GenerationRequest(
        "r", 2, chat["id"], content="hi", checkpoint_id="checkpoints/x/step-1"
    )
    assert asyncio.run(scheduler._model_rejection(other, record))["code"] == "model_not_allowed"
    allowed = GenerationRequest("r", 2, chat["id"], content="hi")
    assert asyncio.run(scheduler._model_rejection(allowed, record)) is None
    admin.put("/api/models/allowed", json={"ids": []}, headers=csrf(admin))
    rejection = asyncio.run(scheduler._model_rejection(allowed, record))
    assert rejection == {
        "code": "model_not_allowed",
        "detail": "This model is not enabled for your account",
    }
    privileged = GenerationRequest(
        "r", 1, chat["id"], content="hi", privileged=True, checkpoint_id="x"
    )
    assert asyncio.run(scheduler._model_rejection(privileged, record)) is None


def test_pagination_helpers_and_new_pages(app):
    from llm.studio.pagination import MAX_PAGE, matches, paginate
    from llm.studio.processes import process_group

    assert paginate(list(range(5)), 1, 2) == {"items": [1, 2], "total": 5}
    assert paginate(list(range(5)), -3, 0)["items"] == [0]
    assert len(paginate(list(range(1000)), 0, 10_000)["items"]) == MAX_PAGE
    assert matches(None, "x") and matches("AB", "cab") and not matches("zz", "cab", None)
    assert [
        process_group(item)
        for item in ("running", "pending", "failed", "interrupted", "completed", "stopped")
    ] == [
        "active",
        "queued",
        "failed",
        "failed",
        "completed",
        "completed",
    ]
    admin = sign_in(app, "admin", "admin")
    user, _ = register(app)
    for path in ("/api/processes/page", "/api/experiments/page", "/api/models/allowed"):
        assert user.get(path).status_code == 403
    assert admin.get("/api/processes/page", params={"group": "bogus"}).status_code == 422
    assert admin.get("/api/processes/page", params={"limit": 201}).status_code == 422
    page = admin.get("/api/experiments/page", params={"limit": 5}).json()
    assert set(page) == {"items", "total"}
