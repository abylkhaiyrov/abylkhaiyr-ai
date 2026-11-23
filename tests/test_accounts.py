import sqlite3
from datetime import UTC, datetime, timedelta

import pytest
import torch
from fastapi.testclient import TestClient

from llm.api.server import create_app
from llm.inference.engine import InferenceEngine
from llm.model.transformer import CausalLM
from llm.studio.database import create_sqlite_engine, ensure_studio_schema
from llm.studio.models import SessionRecord

ORIGIN = {"origin": "http://testserver"}
PASSWORD = "a long enough password"


def make_app(model_config, tokenizer, tmp_path):
    engine = InferenceEngine(CausalLM(model_config), tokenizer, device=torch.device("cpu"))
    return create_app(engine, "test-model", studio_root=tmp_path)


def csrf(client):
    return {"X-CSRF-Token": client.cookies.get("abylkhaiyr_csrf")}


def sign_in(app, identifier, password):
    client = TestClient(app)
    response = client.post("/api/auth/login", json={"username": identifier, "password": password})
    assert response.status_code == 200, response.text
    return client


def register(app, username="alice", email="Alice@Example.com", **override):
    client = TestClient(app)
    payload = {
        "firstName": "Alice",
        "lastName": "Doe",
        "email": email,
        "username": username,
        "password": PASSWORD,
        **override,
    }
    response = client.post("/api/auth/register", json=payload)
    return client, response


@pytest.fixture
def app(model_config, tokenizer, tmp_path):
    return make_app(model_config, tokenizer, tmp_path)


@pytest.fixture
def admin(app):
    return sign_in(app, "admin", "admin")


def until(ws, event_type, limit=60):
    for _ in range(limit):
        event = ws.receive_json()
        if event["type"] == event_type:
            return event
    raise AssertionError(f"{event_type} not received")


def test_registration_creates_a_user_role_without_leaking_secrets(app):
    client, response = register(app)
    assert response.status_code == 201, response.text
    body = response.json()
    assert body["role"] == "USER"
    assert body["email"] == "alice@example.com"
    assert body["username"] == "alice"
    assert not {"password", "password_hash"} & set(body)
    assert client.get("/api/auth/me").json()["id"] == body["id"]
    assert "HttpOnly" in response.headers["set-cookie"]


def test_registration_rejects_role_selection_and_invalid_input(app):
    assert register(app, role="ADMIN")[1].status_code == 422
    assert register(app, email="not-an-email")[1].json()["detail"] == "invalid_email"
    assert register(app, username="Bad Name!")[1].json()["detail"] == "invalid_username"
    assert register(app, password="short")[1].json()["detail"] == "weak_password"


def test_duplicate_email_and_username_are_conflicts_after_normalization(app):
    assert register(app)[1].status_code == 201
    email = register(app, username="bob", email=" ALICE@example.com ")[1]
    assert (email.status_code, email.json()["detail"]) == (409, "email_taken")
    username = register(app, username="Alice", email="other@example.com")[1]
    assert (username.status_code, username.json()["detail"]) == (409, "username_taken")


def test_registration_is_rate_limited_per_address(app):
    statuses = [
        register(app, f"user{index}", f"user{index}@example.com")[1].status_code
        for index in range(6)
    ]
    assert statuses[:5] == [201] * 5
    assert statuses[5] == 429


def test_login_by_username_or_email_and_bad_password(app):
    register(app)
    sign_in(app, "alice", PASSWORD)
    sign_in(app, "ALICE@example.com", PASSWORD)
    client = TestClient(app)
    response = client.post(
        "/api/auth/login", json={"username": "alice", "password": "wrong password!!"}
    )
    assert (response.status_code, response.json()["detail"]) == (401, "invalid_credentials")


def test_repeated_failures_lock_the_account_temporarily(app):
    register(app)
    client = TestClient(app)
    codes = [
        client.post(
            "/api/auth/login", json={"username": "alice", "password": "wrong password!!"}
        ).status_code
        for _ in range(6)
    ]
    assert codes == [401] * 5 + [429]
    assert (
        client.post("/api/auth/login", json={"username": "alice", "password": PASSWORD}).status_code
        == 429
    )


def test_login_issues_a_new_session_and_drops_the_presented_one(app):
    client, _ = register(app)
    old = client.cookies.get("abylkhaiyr_session")
    client.post("/api/auth/login", json={"username": "alice", "password": PASSWORD})
    assert client.cookies.get("abylkhaiyr_session") != old
    stale = TestClient(app)
    stale.cookies.set("abylkhaiyr_session", old)
    assert stale.get("/api/auth/me").status_code == 401


def test_expired_sessions_are_rejected(app):
    client, _ = register(app)
    assert client.get("/api/auth/me").status_code == 200
    studio = app.state.studio
    with studio.session_factory.begin() as database:
        for record in database.query(SessionRecord).all():
            record.expires_at = datetime.now(UTC).replace(tzinfo=None) - timedelta(minutes=1)
    assert client.get("/api/auth/me").status_code == 401


def test_existing_admin_is_preserved_across_restarts(model_config, tokenizer, tmp_path):
    first = make_app(model_config, tokenizer, tmp_path)
    admin_client = sign_in(first, "admin", "admin")
    assert admin_client.get("/api/auth/me").json()["role"] == "ADMIN"
    assert admin_client.get("/api/auth/me").json()["requires_password_change"] is True
    changed = admin_client.put(
        "/api/auth/password",
        headers=csrf(admin_client),
        json={"current_password": "admin", "new_password": "a brand new admin password"},
    )
    assert changed.status_code == 204
    second = make_app(model_config, tokenizer, tmp_path)
    assert (
        TestClient(second)
        .post("/api/auth/login", json={"username": "admin", "password": "admin"})
        .status_code
        == 401
    )
    restored = sign_in(second, "admin", "a brand new admin password")
    assert restored.get("/api/auth/me").json()["requires_password_change"] is False
    with second.state.studio.session_factory() as database:
        from llm.studio.models import User

        assert database.query(User).filter(User.username == "admin").count() == 1


def test_user_cannot_reach_admin_apis(app):
    client, _ = register(app)
    for path in (
        "/api/runtime",
        "/api/system",
        "/api/processes",
        "/api/logs",
        "/api/datasets",
        "/api/models/registry",
        "/api/runs",
        "/api/users",
        "/api/experiments",
        "/api/evaluation",
    ):
        assert client.get(path).status_code == 403, path
    assert (
        client.post(
            "/api/training", headers=csrf(client), json={"config_path": "configs/debug.yaml"}
        ).status_code
        == 403
    )
    assert client.post("/api/pipeline/stop", headers=csrf(client)).status_code == 403
    assert (
        client.post(
            "/api/models/switch", headers=csrf(client), json={"checkpoint_id": "x"}
        ).status_code
        == 403
    )
    assert TestClient(app).get("/api/runtime").status_code == 401


def test_user_cannot_pick_models_and_sees_only_the_active_one(app):
    client, _ = register(app)
    denied = client.post(
        "/api/chats",
        headers=csrf(client),
        json={"checkpoint": "checkpoints/x/step-1", "model_name": "x"},
    )
    assert (denied.status_code, denied.json()["detail"]) == (403, "model_not_allowed")
    only_checkpoint = client.post(
        "/api/chats", headers=csrf(client), json={"checkpoint": "checkpoints/x/step-1"}
    )
    assert only_checkpoint.status_code == 403
    created = client.post("/api/chats", headers=csrf(client), json={}).json()
    assert created["checkpoint"] is None and created["model_name"] is None
    patched = client.patch(
        f"/api/chats/{created['id']}",
        headers=csrf(client),
        json={"checkpoint": "checkpoints/y/step-9"},
    )
    assert (patched.status_code, patched.json()["detail"]) == (403, "model_not_allowed")
    unknown = client.patch(
        f"/api/chats/{created['id']}", headers=csrf(client), json={"model_name": "missing"}
    )
    assert unknown.status_code == 403
    info = client.get("/api/chat/model").json()
    assert info["id"] and "weights_bytes" not in info


def test_chats_are_isolated_between_users(app):
    alice, _ = register(app)
    bob, _ = register(app, "bob", "bob@example.com")
    chat = alice.post("/api/chats", headers=csrf(alice), json={"title": "private"}).json()
    path = f"/api/chats/{chat['id']}"
    assert bob.get(path).status_code == 404
    assert bob.patch(path, headers=csrf(bob), json={"title": "hijack"}).status_code == 404
    assert bob.delete(path, headers=csrf(bob)).status_code == 404
    assert bob.get(path + "/export").status_code == 404
    assert bob.get("/api/chats").json() == []
    assert [item["id"] for item in alice.get("/api/chats").json()] == [chat["id"]]
    assert (
        alice.patch(path, headers=csrf(alice), json={"archived": True}).json()["archived"] is True
    )
    assert alice.get("/api/chats").json() == []
    assert len(alice.get("/api/chats", params={"archived": True}).json()) == 1


def test_websocket_generation_against_foreign_chat_is_rejected(app):
    alice, _ = register(app)
    bob, _ = register(app, "bob", "bob@example.com")
    chat = alice.post("/api/chats", headers=csrf(alice), json={}).json()
    with bob.websocket_connect("/api/ws", headers=ORIGIN) as ws:
        ws.send_json({"type": "subscribe", "channels": ["chat"]})
        ws.send_json(
            {
                "type": "generation.start",
                "request_id": "r-foreign",
                "chat_id": chat["id"],
                "content": "hi",
            }
        )
        failed = until(ws, "generation.failed")
        assert failed["code"] == "chat_not_found"
        ws.send_json({"type": "generation.cancel", "request_id": "r-foreign"})


def test_websocket_studio_channel_is_admin_only(app, admin):
    user, _ = register(app)
    with user.websocket_connect("/api/ws", headers=ORIGIN) as ws:
        ws.send_json({"type": "subscribe", "channels": ["studio", "chat"]})
        ws.send_json({"type": "ping"})
        seen = []
        while True:
            event = ws.receive_json()
            seen.append(event["type"])
            if event["type"] == "pong":
                break
        assert not {"runtime.update", "system.update", "models.update"} & set(seen)
    with admin.websocket_connect("/api/ws", headers=ORIGIN) as ws:
        ws.send_json({"type": "subscribe", "channels": ["studio"]})
        assert until(ws, "runtime.update")["data"]["compute"] is not None


def test_notifications_crud_pagination_and_ownership(app):
    alice, _ = register(app)
    bob, _ = register(app, "bob", "bob@example.com")
    service = app.state.studio.notifications
    created = [service.create(2, "SYSTEM", f"notice-{index}", "body") for index in range(3)]
    assert alice.get("/api/notifications/unread-count").json() == {"unread": 3}
    page = alice.get("/api/notifications", params={"limit": 2, "offset": 0}).json()
    assert (len(page["items"]), page["total"], page["unread"]) == (2, 3, 3)
    assert page["items"][0]["title"] == "notice-2"
    ascending = alice.get("/api/notifications", params={"sort": "asc"}).json()["items"]
    assert ascending[0]["title"] == "notice-0"
    target = created[0]["id"]
    assert bob.get(f"/api/notifications/{target}").status_code == 404
    assert bob.post(f"/api/notifications/{target}/read", headers=csrf(bob)).status_code == 404
    assert bob.delete(f"/api/notifications/{target}", headers=csrf(bob)).status_code == 404
    assert alice.get(f"/api/notifications/{target}").json()["is_read"] is False
    assert (
        alice.post(f"/api/notifications/{target}/read", headers=csrf(alice)).json()["is_read"]
        is True
    )
    assert alice.get("/api/notifications", params={"unread": True}).json()["total"] == 2
    assert alice.post("/api/notifications/read-all", headers=csrf(alice)).json() == {"updated": 2}
    assert alice.get("/api/notifications/unread-count").json() == {"unread": 0}
    assert alice.delete(f"/api/notifications/{target}", headers=csrf(alice)).status_code == 204
    assert alice.get("/api/notifications").json()["total"] == 2
    assert bob.get("/api/notifications").json()["total"] == 0


def test_users_cannot_create_notifications_over_the_api(app):
    client, _ = register(app)
    assert client.post(
        "/api/notifications", headers=csrf(client), json={"user_id": 1}
    ).status_code in {404, 405}


def test_generation_notifications_are_unique_and_delivered_only_to_the_owner(app, monkeypatch):
    from llm.inference.engine import InferenceEngine as Engine

    monkeypatch.setattr(Engine, "stream_chat", lambda *args: iter(["Hello", " there"]))
    alice, _ = register(app)
    bob, _ = register(app, "bob", "bob@example.com")
    chat = alice.post("/api/chats", headers=csrf(alice), json={}).json()
    with (
        alice.websocket_connect("/api/ws", headers=ORIGIN) as ws_a,
        bob.websocket_connect("/api/ws", headers=ORIGIN) as ws_b,
    ):
        for ws in (ws_a, ws_b):
            ws.send_json({"type": "subscribe", "channels": ["chat"]})
        message = {
            "type": "generation.start",
            "request_id": "gen-1",
            "chat_id": chat["id"],
            "content": "Hi",
        }
        ws_a.send_json(message)
        done = until(ws_a, "generation.completed")
        assert done["task_status"] == "COMPLETED"
        created = until(ws_a, "notification.created")
        assert created["notification"]["type"] == "GENERATION_COMPLETED"
        assert created["notification"]["chat_id"] == chat["id"]
        assert until(ws_a, "notification.count.updated")["unread"] == 1
        ws_a.send_json(message)
        ws_b.send_json({"type": "ping"})
        types = []
        while True:
            event = ws_b.receive_json()
            types.append(event["type"])
            if event["type"] == "pong":
                break
        assert not any(name.startswith(("notification", "generation")) for name in types)
    listed = alice.get("/api/notifications").json()
    assert listed["total"] == 1
    assert bob.get("/api/notifications").json()["total"] == 0
    job = app.state.studio.get_generation_job(2, "gen-1")
    assert job["task_status"] == "COMPLETED" and job["started_at"] and job["completed_at"]


def test_notifications_survive_offline_period(app, monkeypatch):
    from llm.inference.engine import InferenceEngine as Engine

    monkeypatch.setattr(Engine, "stream_chat", lambda *args: iter(["ok"]))
    alice, _ = register(app)
    chat = alice.post("/api/chats", headers=csrf(alice), json={}).json()
    with alice.websocket_connect("/api/ws", headers=ORIGIN) as ws:
        ws.send_json({"type": "subscribe", "channels": ["chat"]})
        ws.send_json(
            {
                "type": "generation.start",
                "request_id": "gen-off",
                "chat_id": chat["id"],
                "content": "x",
            }
        )
        until(ws, "generation.completed")
    reconnected = TestClient(app)
    reconnected.cookies.update(alice.cookies)
    assert reconnected.get("/api/notifications/unread-count").json()["unread"] == 1


def test_notification_for_generation_is_never_duplicated(app):
    register(app)
    service = app.state.studio.notifications
    assert (
        service.create(
            2, "GENERATION_COMPLETED", "generation_completed", chat_id=None, generation_id="g"
        )
        is not None
    )
    assert (
        service.create(
            2, "GENERATION_COMPLETED", "generation_completed", chat_id=None, generation_id="g"
        )
        is None
    )
    assert service.list(2, False, None, 10, 0, False)["total"] == 1


def test_blocking_a_user_revokes_sessions_and_closes_websockets(app, admin):
    user, response = register(app)
    user_id = response.json()["id"]
    with user.websocket_connect("/api/ws", headers=ORIGIN) as ws:
        ws.send_json({"type": "subscribe", "channels": ["chat"]})
        blocked = admin.patch(
            f"/api/users/{user_id}", headers=csrf(admin), json={"isActive": False}
        )
        assert blocked.status_code == 200 and blocked.json()["is_active"] is False
        assert until(ws, "session.revoked")["type"] == "session.revoked"
    assert user.get("/api/auth/me").status_code == 401
    login = TestClient(app).post(
        "/api/auth/login", json={"username": "alice", "password": PASSWORD}
    )
    assert (login.status_code, login.json()["detail"]) == (403, "account_disabled")
    admin.patch(f"/api/users/{user_id}", headers=csrf(admin), json={"isActive": True})
    sign_in(app, "alice", PASSWORD)


def test_session_revocation_by_user_and_admin(app, admin):
    user, response = register(app)
    second = sign_in(app, "alice", PASSWORD)
    sessions = user.get("/api/auth/sessions").json()
    assert len(sessions) >= 2 and sum(item["current"] for item in sessions) == 1
    assert user.post("/api/auth/sessions/revoke-others", headers=csrf(user)).json()["revoked"] >= 1
    assert second.get("/api/auth/me").status_code == 401
    assert user.get("/api/auth/me").status_code == 200
    revoked = admin.post(f"/api/users/{response.json()['id']}/revoke-sessions", headers=csrf(admin))
    assert revoked.json()["revoked"] == 1
    assert user.get("/api/auth/me").status_code == 401


def test_the_last_admin_cannot_be_demoted_or_blocked(app, admin):
    me = admin.get("/api/auth/me").json()["id"]
    for change in ({"role": "USER"}, {"isActive": False}):
        response = admin.patch(f"/api/users/{me}", headers=csrf(admin), json=change)
        assert (response.status_code, response.json()["detail"]) == (409, "last_admin"), change
    _, other = register(app)
    promoted = admin.patch(
        f"/api/users/{other.json()['id']}", headers=csrf(admin), json={"role": "ADMIN"}
    )
    assert promoted.json()["role"] == "ADMIN"
    assert (
        admin.patch(f"/api/users/{me}", headers=csrf(admin), json={"role": "USER"}).status_code
        == 200
    )


def test_admin_password_reset_forces_a_change_and_is_audited(app, admin):
    user, response = register(app)
    reset = admin.post(f"/api/users/{response.json()['id']}/reset-password", headers=csrf(admin))
    temporary = reset.json()["temporary_password"]
    assert user.get("/api/auth/me").status_code == 401
    fresh = sign_in(app, "alice", temporary)
    assert fresh.get("/api/auth/me").json()["requires_password_change"] is True
    assert (
        TestClient(app)
        .post("/api/auth/login", json={"username": "alice", "password": PASSWORD})
        .status_code
        == 401
    )
    with app.state.studio.session_factory() as database:
        from llm.studio.models import AuditLog

        actions = {
            row.action for row in database.query(AuditLog).filter(AuditLog.category == "admin")
        }
    assert "password_reset" in actions
    notifications = fresh.get("/api/notifications").json()["items"]
    assert notifications[0]["type"] == "ACCOUNT"


def test_user_directory_search_filter_and_pagination(app, admin):
    for index in range(4):
        register(app, f"member{index}", f"member{index}@example.com", firstName=f"Member{index}")
    page = admin.get("/api/users", params={"limit": 2, "offset": 0}).json()
    assert page["total"] == 5 and len(page["items"]) == 2
    assert not any("hash" in key for item in page["items"] for key in item)
    found = admin.get("/api/users", params={"query": "member3"}).json()
    assert [item["username"] for item in found["items"]] == ["member3"]
    admins = admin.get("/api/users", params={"role": "ADMIN"}).json()
    assert admins["total"] == 1 and admins["items"][0]["username"] == "admin"
    assert admin.get("/api/users/unknown-id").status_code == 404


def test_profile_update_and_password_change(app):
    client, _ = register(app)
    updated = client.patch(
        "/api/auth/profile",
        headers=csrf(client),
        json={
            "firstName": "Alicia",
            "lastName": "Doe",
            "email": "alicia@example.com",
            "username": "alicia",
        },
    )
    assert updated.json()["username"] == "alicia"
    register(app, "carol", "carol@example.com")
    clash = client.patch(
        "/api/auth/profile",
        headers=csrf(client),
        json={
            "firstName": "A",
            "lastName": "D",
            "email": "carol@example.com",
            "username": "alicia",
        },
    )
    assert (clash.status_code, clash.json()["detail"]) == (409, "email_taken")
    changed = client.put(
        "/api/auth/password",
        headers=csrf(client),
        json={"current_password": PASSWORD, "new_password": "another long password"},
    )
    assert changed.status_code == 204
    assert client.get("/api/auth/me").status_code == 401
    sign_in(app, "alicia", "another long password")


def test_mutations_require_csrf(app):
    client, _ = register(app)
    assert client.post("/api/chats", json={}).status_code == 403
    assert client.post("/api/notifications/read-all").status_code == 403
    assert (
        client.patch(
            "/api/auth/profile",
            json={"firstName": "A", "lastName": "B", "email": "a@b.co", "username": "abc"},
        ).status_code
        == 403
    )


def test_message_pagination_loads_only_recent_messages(app):
    client, _ = register(app)
    chat = client.post("/api/chats", headers=csrf(client), json={}).json()
    studio = app.state.studio
    for index in range(7):
        studio.add_message(2, chat["id"], "user", f"m{index}")
    page = client.get(f"/api/chats/{chat['id']}", params={"limit": 3}).json()
    assert [item["content"] for item in page["messages"]] == ["m4", "m5", "m6"]
    assert page["total_messages"] == 7


def test_legacy_database_is_migrated_without_losing_data(tmp_path):
    path = tmp_path / "studio" / "studio.db"
    path.parent.mkdir(parents=True)
    connection = sqlite3.connect(path)
    connection.executescript(
        """
        CREATE TABLE users (id INTEGER PRIMARY KEY, username VARCHAR(128) NOT NULL UNIQUE,
            password_hash VARCHAR(512) NOT NULL, requires_password_change BOOLEAN, created_at DATETIME, updated_at DATETIME);
        CREATE TABLE chats (id VARCHAR(36) PRIMARY KEY, user_id INTEGER, title VARCHAR(256), model_name VARCHAR(256),
            checkpoint VARCHAR(1024), system_prompt TEXT, generation_settings TEXT, created_at DATETIME, updated_at DATETIME);
        INSERT INTO users VALUES (1, 'admin', 'hash', 1, '2026-01-01', '2026-01-01');
        INSERT INTO chats VALUES ('c1', 1, 'history', NULL, NULL, '', '{}', '2026-01-01', '2026-01-01');
        """
    )
    connection.commit()
    connection.close()
    engine = create_sqlite_engine(path)
    ensure_studio_schema(engine)
    ensure_studio_schema(engine)
    with engine.connect() as database:
        from sqlalchemy import text

        row = database.execute(text("SELECT role, uuid, is_active, password_hash FROM users")).one()
        assert (
            row.role == "ADMIN" and row.uuid and row.is_active == 1 and row.password_hash == "hash"
        )
        assert database.execute(text("SELECT title, archived FROM chats")).one() == ("history", 0)
        assert database.execute(text("SELECT COUNT(*) FROM notifications")).scalar() == 0
