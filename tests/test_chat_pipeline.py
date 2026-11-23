from __future__ import annotations

import logging

import pytest
import torch

from llm.api.server import create_app
from llm.inference.engine import InferenceEngine
from llm.inference.sampling import SamplingParams
from llm.model.transformer import CausalLM
from llm.reasoning.engine import Budget, Problem, ReasoningEngine
from llm.reasoning.stages import (
    FACTUAL_CHECK,
    FORMAT_CHECK,
    GENERATION,
    PREPARATION,
    TOOL,
    executed_checks,
    phase_of,
)
from llm.tokenizer.bpe import Message
from tests.test_accounts import ORIGIN, csrf, register
from tests.test_reasoning import ScriptedEngine

SECRET = "my private question about taxes"


def stage_event(stage: str, status: str = "completed", **data):
    return {"stage": stage, "status": status, "data": data}


def test_phases_separate_preparation_generation_tools_and_checks():
    assert phase_of("task_classification", {}) == PREPARATION
    assert phase_of("context_preparation", {}) == PREPARATION
    assert phase_of("answer_formation", {}) == GENERATION
    assert phase_of("candidate_generation", {}) == GENERATION
    assert phase_of("math_computation", {}) == TOOL
    assert phase_of("verification", {"check": "format"}) == FORMAT_CHECK
    assert phase_of("verification", {"check": "factual"}) == FACTUAL_CHECK


def test_only_completed_checks_are_reported_and_semantic_is_never_claimed():
    events = [
        stage_event("task_classification"),
        stage_event("context_preparation"),
        stage_event("answer_formation"),
    ]
    assert executed_checks(events) == {"format": False, "factual": False, "semantic": False}
    running = [stage_event("verification", "running", check="factual")]
    assert executed_checks(running)["factual"] is False
    done = [
        stage_event("verification", check="format"),
        stage_event("verification", check="factual"),
    ]
    assert executed_checks(done) == {"format": True, "factual": True, "semantic": False}


def test_reasoning_reports_which_kind_of_check_ran():
    for problem, reply, expected in (
        (
            Problem("tell me about the weather today", "general"),
            "A fine long enough answer",
            "format",
        ),
        (Problem("3*3?", "math"), "<<3*3=9>> #### 9", "factual"),
    ):
        events: list[dict] = []
        ReasoningEngine(ScriptedEngine([reply]), use_solvers=False).solve(
            problem, Budget(max_candidates=1, max_retries=0), on_event=events.append
        )
        verification = [e for e in events if e["stage"] == "verification"]
        assert verification and all(e["data"]["check"] == expected for e in verification)


@pytest.fixture
def studio(model_config, tokenizer, tmp_path):
    torch.manual_seed(0)
    engine = InferenceEngine(CausalLM(model_config), tokenizer, device=torch.device("cpu"))
    app = create_app(engine, "test-model", studio_root=tmp_path)
    user, response = register(app)
    assert response.status_code == 201, response.text
    return engine, user


def run_generation(
    user, content: str, request_id: str, chat_settings: dict | None = None, **settings
):
    chat = user.post("/api/chats", headers=csrf(user), json=chat_settings or {}).json()
    with user.websocket_connect("/api/ws", headers=ORIGIN) as socket:
        socket.send_json({"type": "subscribe", "channels": ["chat"]})
        socket.send_json(
            {
                "type": "generation.start",
                "request_id": request_id,
                "chat_id": chat["id"],
                "content": content,
                **settings,
            }
        )
        stages = []
        for _ in range(500):
            event = socket.receive_json()
            if event["type"] == "generation.stage" and event["stage_status"] == "completed":
                stages.append(event["reasoning_events"][-1])
            if event["type"] in {"generation.completed", "generation.failed"}:
                return event, stages
    raise AssertionError("generation did not finish")


def test_studio_without_reasoning_reports_no_checks_and_labels_each_stage(studio):
    _, user = studio
    done, stages = run_generation(user, "hello there", "r-plain", temperature=0.0)
    assert done["type"] == "generation.completed"
    assert done["checks"] == {"format": False, "factual": False, "semantic": False}
    assert {item["stage"]: item["phase"] for item in stages} == {
        "task_classification": PREPARATION,
        "context_preparation": PREPARATION,
        "answer_formation": GENERATION,
    }


def test_studio_greedy_generation_matches_the_direct_engine(studio):
    engine, user = studio
    done, _ = run_generation(
        user, "tell me about rivers", "r-same", temperature=0.0, repetition_penalty=1.1
    )
    params = SamplingParams(
        max_new_tokens=done["output_tokens"] + 8, greedy=True, repetition_penalty=1.1
    )
    direct = engine.chat([Message("user", "tell me about rivers")], params)
    assert done["text"] == direct.text


def test_generation_log_has_a_prompt_digest_but_no_private_text(studio, caplog):
    engine, user = studio
    caplog.set_level(logging.INFO, logger="llm.studio.scheduler")
    run_generation(user, SECRET, "r-log", temperature=0.0)
    lines = [
        record.getMessage() for record in caplog.records if "prompt_digest" in record.getMessage()
    ]
    assert len(lines) == 1
    assert f"prompt_digest={engine.chat_digest([Message('user', SECRET)])}" in lines[0]
    assert SECRET not in caplog.text


def test_natural_chat_evaluation_is_protected_from_training_and_covers_three_languages():
    import json
    from collections import Counter
    from pathlib import Path

    from llm.data.dedup import stable_hash
    from llm.data.v4_splits import legacy_prompts, normalized

    path = Path("configs/chat_eval_natural_v1.jsonl")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert Counter(row["language"] for row in rows) == {"ru": 12, "en": 12, "kk": 12}
    assert len({row["id"] for row in rows}) == 36
    assert all(row["messages"][-1] == {"role": "user", "content": row["question"]} for row in rows)
    assert stable_hash(normalized("что умеешь делать?")) in legacy_prompts([path])


def logged_digests(caplog) -> list[str]:
    return [
        record.getMessage().split("prompt_digest=")[1].split()[0]
        for record in caplog.records
        if "prompt_digest" in record.getMessage()
    ]


def test_each_chat_sends_only_its_own_messages_without_duplicates(studio, caplog):
    engine, user = studio
    caplog.set_level(logging.INFO, logger="llm.studio.scheduler")
    run_generation(user, "first chat question", "r-first", temperature=0.0)
    run_generation(
        user,
        "second chat question",
        "r-second",
        chat_settings={"system_prompt": "Be brief."},
        temperature=0.0,
    )
    first, second = logged_digests(caplog)
    assert first == engine.chat_digest([Message("user", "first chat question")])
    assert second == engine.chat_digest(
        [Message("system", "Be brief."), Message("user", "second chat question")]
    )
