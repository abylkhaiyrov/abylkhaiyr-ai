from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

from llm.api.server import create_app
from llm.evaluation.comparison import load_engine
from llm.evaluation.metrics import normalize_words, prompt_echo, repeated_ngram_ratio
from llm.inference.engine import InferenceEngine
from llm.inference.sampling import SamplingParams
from llm.reasoning.engine import ReasoningEngine
from llm.studio.inference_profiles import PROFILES
from llm.tokenizer.bpe import Message

ORIGIN = {"origin": "http://testserver"}
PROMPTS = (
    ("ru", "что умеешь делать?"),
    ("ru", "Как дела?"),
    ("ru", "Кто ты?"),
    ("ru", "Чем можешь помочь?"),
    ("ru", "Что такое искусственный интеллект?"),
    ("ru", "Почему идёт дождь?"),
    ("ru", "Объясни разницу между Java и Python."),
    ("ru", "Что такое PostgreSQL?"),
    ("ru", "Сколько будет 17 × 24?"),
    ("ru", "Как отсортировать список в Python?"),
    ("ru", "Переведи на английский: Я иду домой."),
    ("en", "What can you do?"),
    ("en", "Who are you?"),
    ("en", "What is PostgreSQL?"),
    ("kk", "не істей аласың?"),
    ("kk", "Сен кімсің?"),
    ("kk", "Жасанды интеллект дегеніміз не?"),
)
HISTORY_CASE = (
    "ru",
    [("user", "Меня зовут Дамир, я живу в Семее."), ("assistant", "Приятно познакомиться, Дамир.")],
    "В каком городе я живу?",
)


def metrics(question: str, text: str, finished: bool) -> dict[str, Any]:
    words = normalize_words(text)
    return {
        "text": text,
        "finished": finished,
        "echo": prompt_echo(question, text),
        "repetition": repeated_ngram_ratio(words) if len(words) >= 6 else 0.0,
    }


def direct(engine: InferenceEngine, messages: list[Message], params: SamplingParams) -> dict:
    reply = engine.chat(messages, params)
    return metrics(messages[-1].content, reply.text, reply.finish_reason == "stop")


def reasoned(engine: InferenceEngine, messages: list[Message], params: SamplingParams) -> dict:
    profile = PROFILES["medium"]
    events: list[str] = []
    solution = ReasoningEngine(engine).solve_auto(
        messages[-1].content,
        budget=profile.budget(params.max_new_tokens),
        history=messages,
        on_event=lambda event: events.append(f"{event['stage']}:{event['status']}"),
        temperature=params.temperature,
        top_p=params.top_p,
        top_k=params.top_k,
        repetition_penalty=params.repetition_penalty,
    )
    return {**metrics(messages[-1].content, solution.text, True), "events": events}


class Studio:
    def __init__(self, engine: InferenceEngine, root: Path) -> None:
        self.app = create_app(engine, "diagnostic", studio_root=root)
        self.client = TestClient(self.app)
        response = self.client.post(
            "/api/auth/login", json={"username": "admin", "password": "admin"}
        )
        response.raise_for_status()
        self.csrf = {"X-CSRF-Token": self.client.cookies.get("abylkhaiyr_csrf") or ""}
        self.requests = 0

    def ask(
        self,
        socket: Any,
        history: list[tuple[str, str]],
        question: str,
        profile: str,
        **settings: Any,
    ) -> dict:
        chat = self.client.post("/api/chats", headers=self.csrf, json={}).json()
        for role, content in history:
            self.app.state.studio.add_message(1, chat["id"], role, content)
        self.requests += 1
        socket.send_json(
            {
                "type": "generation.start",
                "request_id": f"diagnostic-{self.requests}",
                "chat_id": chat["id"],
                "content": question,
                "profile": profile,
                **settings,
            }
        )
        stages: list[str] = []
        prefill: dict[str, Any] = {}
        for _ in range(2000):
            event = socket.receive_json()
            if event["type"] == "generation.stage" and event["stage_status"] == "completed":
                stages.append(event["stage"])
            if event["type"] == "generation.prefill":
                prefill = event
            if event["type"] in {"generation.completed", "generation.failed"}:
                break
        return {
            **metrics(question, event.get("text", ""), event["type"] == "generation.completed"),
            "stages": stages,
            "input_tokens": prefill.get("input_tokens"),
            "checks": event.get("checks"),
            "event": event["type"],
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("--profile", default="medium")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    engine = load_engine(args.checkpoint, "data/tokenizer", args.device)
    profile = PROFILES[args.profile]
    greedy = SamplingParams(
        max_new_tokens=profile.max_output_tokens, greedy=True, repetition_penalty=1.1
    )
    sampled = SamplingParams(
        max_new_tokens=profile.max_output_tokens,
        seed=0,
        temperature=0.7,
        top_p=0.9,
        top_k=40,
        repetition_penalty=1.1,
    )
    cases: list[tuple[str, list[tuple[str, str]], str]] = [
        (lang, [], question) for lang, question in PROMPTS
    ] + [HISTORY_CASE]
    rows = []
    with tempfile.TemporaryDirectory() as root:
        studio = Studio(engine, Path(root))
        socket_context = studio.client.websocket_connect("/api/ws", headers=ORIGIN)
        socket = socket_context.__enter__()
        socket.send_json({"type": "subscribe", "channels": ["chat"]})
        for language, history, question in cases:
            messages = [Message(role, content) for role, content in history] + [
                Message("user", question)
            ]
            template = engine.tokenizer.decode(
                engine.tokenizer.encode_conversation(messages, add_generation_prompt=True)
            )
            rows.append(
                {
                    "language": language,
                    "question": question,
                    "prompt_template": template,
                    "prompt_tokens": engine.chat_prompt_tokens(messages, profile.max_output_tokens),
                    "A_direct_greedy": direct(engine, messages, greedy),
                    "A_direct_sampled": direct(engine, messages, sampled),
                    "B_reasoning_greedy": reasoned(engine, messages, greedy),
                    "C_studio_greedy": studio.ask(
                        socket,
                        history,
                        question,
                        args.profile,
                        temperature=0.0,
                        repetition_penalty=1.1,
                    ),
                    "C_studio_default": studio.ask(socket, history, question, args.profile),
                    "C_studio_high_greedy": studio.ask(
                        socket, history, question, "high", temperature=0.0, repetition_penalty=1.1
                    ),
                }
            )
        socket_context.__exit__(None, None, None)
    args.output.write_text(json.dumps(rows, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"{len(rows)} cases written")


if __name__ == "__main__":
    main()
