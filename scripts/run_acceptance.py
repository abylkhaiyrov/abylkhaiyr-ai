from __future__ import annotations

import argparse
import asyncio
import json
import secrets
import time
from pathlib import Path
from typing import Any

import httpx
import websockets

from llm.evaluation.acceptance import AcceptanceItem, auto_check, load_benchmark
from llm.tokenizer.bpe import BPETokenizer, Message

CONFIGS: dict[str, dict[str, Any]] = {
    "greedy": {"profile": "medium", "temperature": 0.0, "repetition_penalty": 1.1},
    "default": {"profile": "medium"},
    "high": {"profile": "high", "temperature": 0.0, "repetition_penalty": 1.1},
}
TERMINAL = {"generation.completed", "generation.failed", "generation.cancelled"}


def login(client: httpx.Client, base: str, password: str) -> dict[str, str]:
    headers = {"origin": base}
    response = client.post(
        "/api/auth/login", json={"identifier": "admin", "password": password}, headers=headers
    )
    if response.status_code != 200:
        response = client.post(
            "/api/auth/login", json={"identifier": "admin", "password": "admin"}, headers=headers
        )
        response.raise_for_status()
        csrf = client.cookies.get("abylkhaiyr_csrf") or ""
        changed = client.put(
            "/api/auth/password",
            json={"current_password": "admin", "new_password": password},
            headers={**headers, "X-CSRF-Token": csrf},
        )
        changed.raise_for_status()
        response = client.post(
            "/api/auth/login", json={"identifier": "admin", "password": password}, headers=headers
        )
        response.raise_for_status()
    csrf = client.cookies.get("abylkhaiyr_csrf") or ""
    return {**headers, "X-CSRF-Token": csrf}


async def generate(
    socket: Any, chat_id: str, content: str, settings: dict[str, Any]
) -> dict[str, Any]:
    request_id = secrets.token_hex(8)
    await socket.send(
        json.dumps(
            {
                "type": "generation.start",
                "request_id": request_id,
                "chat_id": chat_id,
                "content": content,
                "mode": "standard",
                **settings,
            }
        )
    )
    stages: list[str] = []
    started: dict[str, Any] = {}
    deadline = time.monotonic() + 240
    while time.monotonic() < deadline:
        event = json.loads(await asyncio.wait_for(socket.recv(), timeout=240))
        if event.get("request_id") != request_id:
            continue
        kind = event["type"]
        if kind == "generation.started":
            started = event
        if kind == "generation.stage":
            stages.append(f"{event.get('stage')}:{event.get('status')}")
        if kind in TERMINAL:
            return {
                "request_id": request_id,
                "terminal": event,
                "started": started,
                "stages": stages,
            }
    raise TimeoutError("generation did not finish")


def finish_reason(terminal: dict[str, Any], max_tokens: int, stages: list[str]) -> str:
    if terminal["type"] != "generation.completed":
        return terminal["type"].split(".")[1]
    if any(stage.startswith("generation_timeout") for stage in stages):
        return "timeout"
    return "length" if terminal.get("output_tokens", 0) >= max_tokens else "stop"


async def run_item(
    item: AcceptanceItem,
    base: str,
    http: httpx.Client,
    headers: dict[str, str],
    socket: Any,
    settings: dict[str, Any],
    tokenizer: BPETokenizer,
    system_prompt: str | None,
) -> dict[str, Any]:
    chat = http.post(
        "/api/chats",
        json={"title": item.item_id, "system_prompt": system_prompt or ""},
        headers=headers,
    )
    chat.raise_for_status()
    chat_id = chat.json()["id"]
    history: list[dict[str, str]] = []
    turns: list[dict[str, Any]] = []
    for text in item.turns:
        history.append({"role": "user", "content": text})
        result = await generate(socket, chat_id, text, settings)
        terminal = result["terminal"]
        answer = terminal.get("text", "")
        messages = [Message(entry["role"], entry["content"]) for entry in history]
        if system_prompt:
            messages.insert(0, Message("system", system_prompt))
        ids = tokenizer.encode_conversation(messages, add_generation_prompt=True)
        turns.append(
            {
                "request_id": result["request_id"],
                "history_before": list(history[:-1]),
                "user": text,
                "prompt_text": tokenizer.decode(ids),
                "prompt_tokens": len(ids),
                "answer": answer,
                "terminal": terminal["type"],
                "output_tokens": terminal.get("output_tokens"),
                "finish_reason": finish_reason(
                    terminal,
                    int(result["started"]["generation_params"]["max_tokens"]),
                    result["stages"],
                ),
                "elapsed_ms": terminal.get("elapsed_ms"),
                "checkpoint_id": terminal.get("checkpoint_id"),
                "generation_params": result["started"]["generation_params"],
                "stages": result["stages"],
                "checks": terminal.get("checks"),
                "server_input_tokens": terminal.get("input_tokens"),
                "prompt_matches_server": terminal.get("input_tokens") == len(ids),
            }
        )
        history.append({"role": "assistant", "content": answer})
    final = turns[-1]
    auto = auto_check(item, final["answer"])
    return {
        "item_id": item.item_id,
        "scenario": item.scenario,
        "category": item.category,
        "language": item.language,
        "chat_id": chat_id,
        "key_points": list(item.key_points),
        "turns": turns,
        "answer": final["answer"],
        "auto": {"kind": auto.kind, "passed": auto.passed, "detail": auto.detail},
    }


async def main_async(args: argparse.Namespace) -> None:
    items, digest = load_benchmark(args.benchmark)
    if args.limit:
        items = items[: args.limit]
    tokenizer = BPETokenizer.load(args.tokenizer)
    settings = CONFIGS[args.config]
    base = f"http://127.0.0.1:{args.port}"
    with httpx.Client(base_url=base) as http:
        headers = login(http, base, args.password)
        cookie = "; ".join(f"{key}={value}" for key, value in http.cookies.items())
        async with websockets.connect(
            f"ws://127.0.0.1:{args.port}/api/ws",
            additional_headers={"Cookie": cookie, "Origin": base},
            max_size=None,
        ) as socket:
            await socket.send(json.dumps({"type": "subscribe", "channels": ["chat"]}))
            rows = []
            for index, item in enumerate(items, 1):
                row = await run_item(
                    item, base, http, headers, socket, settings, tokenizer, args.system_prompt
                )
                rows.append(row)
                print(f"{index}/{len(items)} {item.item_id}: {row['answer'][:70]!r}", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "benchmark_sha256": digest,
        "config": args.config,
        "settings": settings,
        "system_prompt": args.system_prompt,
        "rows": rows,
    }
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--benchmark", type=Path, default=Path("configs/acceptance/benchmark_v1.json")
    )
    parser.add_argument("--tokenizer", type=Path, default=Path("data/tokenizer"))
    parser.add_argument("--config", choices=sorted(CONFIGS), default="greedy")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--password", required=True)
    parser.add_argument("--system-prompt", default=None)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    asyncio.run(main_async(parser.parse_args()))


if __name__ == "__main__":
    main()
