import json
import threading
from itertools import pairwise

import pytest

from llm.studio.inference_profiles import (
    PROFILE_IDS,
    PROFILES,
    effective_output_tokens,
    profile_view,
    unavailable_reason,
)
from llm.studio.scheduler import GenerationRequest, GenerationScheduler
from tests.test_accounts import csrf, make_app, register, sign_in


@pytest.fixture
def app(model_config, tokenizer, tmp_path):
    return make_app(model_config, tokenizer, tmp_path)


def test_registry_is_complete_and_monotonic():
    assert PROFILE_IDS == ("lite", "medium", "high", "very_high")
    ordered = [PROFILES[item] for item in PROFILE_IDS]
    for lower, higher in pairwise(ordered):
        assert lower.max_output_tokens <= higher.max_output_tokens
        assert lower.context_budget < higher.context_budget
        assert lower.generation_timeout_seconds < higher.generation_timeout_seconds
        assert lower.limits.max_total_new_tokens <= higher.limits.max_total_new_tokens
    assert not PROFILES["lite"].reasoning.enabled
    assert (
        PROFILES["very_high"].reasoning.verification_attempts
        > PROFILES["high"].reasoning.verification_attempts
    )


def test_reasoning_budget_is_derived_from_the_profile():
    budget = PROFILES["high"].budget(200)
    assert budget.max_candidates == 2
    assert budget.max_retries == 1
    assert budget.max_seconds == PROFILES["high"].tool_budget_seconds
    assert budget.max_new_tokens_total <= PROFILES["high"].limits.max_total_new_tokens


def test_small_context_marks_profiles_unavailable():
    assert unavailable_reason(PROFILES["lite"], None) == "model_not_loaded"
    assert unavailable_reason(PROFILES["very_high"], 256) == "context_too_small"
    assert unavailable_reason(PROFILES["lite"], 512) is None
    assert effective_output_tokens(PROFILES["very_high"], 256) < 256
    view = profile_view(PROFILES["very_high"], 256, {"lite"})
    assert (view["available"], view["allowed"]) == (False, False)


def test_sampling_params_respect_the_profile():
    chat = {"generation_settings": {}}
    request = GenerationRequest("r", 1, "c", content="hi", profile="lite")
    params, described = GenerationScheduler._resolve_sampling_params(request, chat, 1024)
    assert params.max_new_tokens == PROFILES["lite"].max_output_tokens
    assert described["profile"] == "lite"
    request = GenerationRequest("r", 1, "c", content="hi", profile="lite", max_tokens=900)
    assert (
        GenerationScheduler._resolve_sampling_params(request, chat, 1024)[0].max_new_tokens == 128
    )
    saved = {"generation_settings": {"profile": "very_high"}}
    request = GenerationRequest("r", 1, "c", content="hi")
    params, described = GenerationScheduler._resolve_sampling_params(request, saved, 300)
    assert described["profile"] == "very_high"
    assert params.max_new_tokens == effective_output_tokens(PROFILES["very_high"], 300)
    plain = GenerationScheduler._resolve_sampling_params(
        GenerationRequest("r", 1, "c"), chat, 1024
    )[1]
    assert plain["profile"] is None


def test_admin_controls_profiles_for_users(app):
    admin = sign_in(app, "admin", "admin")
    user, response = register(app)
    assert response.status_code == 201
    everyone = user.get("/api/inference/profiles").json()
    assert [item["id"] for item in everyone] == list(PROFILE_IDS)
    assert (
        user.put(
            "/api/inference/profiles/allowed", json={"ids": ["lite"]}, headers=csrf(user)
        ).status_code
        == 403
    )
    bad = admin.put("/api/inference/profiles/allowed", json={"ids": ["ultra"]}, headers=csrf(admin))
    assert bad.status_code == 422
    assert json.loads(bad.json()["detail"])["unknown"] == ["ultra"]
    result = admin.put(
        "/api/inference/profiles/allowed", json={"ids": ["lite", "medium"]}, headers=csrf(admin)
    )
    assert result.status_code == 200
    assert {item["id"] for item in result.json() if item["allowed"]} == {"lite", "medium"}
    assert [item["id"] for item in user.get("/api/inference/profiles").json()] == ["lite", "medium"]
    assert len(admin.get("/api/inference/profiles").json()) == 4


def test_chat_settings_reject_unknown_profile(app):
    user, _ = register(app)
    headers = csrf(user)
    ok = user.post(
        "/api/chats", headers=headers, json={"generation_settings": {"profile": "medium"}}
    )
    assert ok.status_code == 201
    assert ok.json()["generation_settings"]["profile"] == "medium"
    bad = user.post(
        "/api/chats", headers=headers, json={"generation_settings": {"profile": "ultra"}}
    )
    assert bad.status_code == 422
    patched = user.patch(
        f"/api/chats/{ok.json()['id']}",
        headers=headers,
        json={"generation_settings": {"profile": "nope"}},
    )
    assert patched.status_code == 422


def test_scheduler_enforces_profile_rbac_and_availability(app):
    studio, manager = app.state.studio, app.state.model_manager
    scheduler = GenerationScheduler(studio, manager, threading.Lock())
    request = GenerationRequest("r", 2, "c", content="hi", profile="lite")
    assert scheduler._profile_rejection(request)["code"] == "profile_unavailable"
    studio.set_allowed_profiles(["lite"])
    denied = GenerationRequest("r", 2, "c", content="hi", profile="medium")
    assert scheduler._profile_rejection(denied)["code"] == "profile_not_allowed"
    admin = GenerationRequest("r", 1, "c", content="hi", profile="medium", privileged=True)
    assert scheduler._profile_rejection(admin)["code"] == "profile_unavailable"
    unknown = GenerationRequest("r", 1, "c", profile="x", privileged=True)
    assert scheduler._profile_rejection(unknown)["code"] == "invalid_profile"
    assert scheduler._profile_rejection(GenerationRequest("r", 1, "c")) is None


def test_budgets_of_profiles_without_reasoning_keep_a_usable_time_limit():
    for name in ("lite", "medium"):
        profile = PROFILES[name]
        assert not profile.reasoning.enabled
        assert profile.budget(64).max_seconds == profile.generation_timeout_seconds > 0
