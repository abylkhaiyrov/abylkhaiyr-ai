from __future__ import annotations

import argparse
import json
from contextlib import ExitStack
from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import uuid4

import torch
from fastapi.testclient import TestClient

from llm.api.server import create_app
from llm.evaluation.comparison import load_engine


def probe(checkpoint: str, question: str) -> dict[str, str]:
    engine = load_engine(checkpoint, "data/tokenizer", "cpu")
    with TemporaryDirectory() as directory, ExitStack() as cleanup:
        app = create_app(
            engine, "dialogue-probe", studio_root=Path(directory), checkpoint_id=checkpoint
        )
        cleanup.callback(app.state.studio.engine.dispose)
        with TestClient(app) as client:
            client.post(
                "/api/auth/login", json={"username": "admin", "password": "admin"}
            ).raise_for_status()
            active = client.get("/api/models/active").json()
            if active["checkpoint_id"] != checkpoint:
                raise ValueError("unexpected active checkpoint")
            headers = {"X-CSRF-Token": client.cookies["abylkhaiyr_csrf"]}
            response = client.post(
                "/v1/chat/completions",
                json={
                    "messages": [{"role": "user", "content": question}],
                    "temperature": 0.3,
                    "top_k": 40,
                    "top_p": 0.9,
                    "repetition_penalty": 1.1,
                    "max_tokens": 64,
                    "seed": 42,
                },
            )
            response.raise_for_status()
            chat = client.post("/api/chats", headers=headers, json={})
            chat.raise_for_status()
            with client.websocket_connect("/api/ws", headers={"origin": "http://testserver"}) as ws:
                ws.send_json({"type": "subscribe", "channels": ["chat"]})
                ws.send_json(
                    {
                        "type": "generation.start",
                        "request_id": uuid4().hex,
                        "chat_id": chat.json()["id"],
                        "content": question,
                        "temperature": 0.0,
                        "max_tokens": 64,
                        "repetition_penalty": 1.1,
                    }
                )
                for _ in range(128):
                    event = ws.receive_json()
                    if event["type"] == "generation.failed" or event["type"] == "error":
                        raise RuntimeError(str(event))
                    if event["type"] == "generation.completed":
                        return {
                            "checkpoint": active["checkpoint_id"],
                            "api": response.json()["choices"][0]["message"]["content"],
                            "studio_websocket": str(event["text"]),
                        }
            raise RuntimeError("generation did not finish within the event budget")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoints", nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    results = [probe(checkpoint, "Как дела?") for checkpoint in args.checkpoints]
    args.output.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
