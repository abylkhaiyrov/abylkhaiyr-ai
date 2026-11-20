from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
from pathlib import Path
from typing import Any

import torch

from llm.evaluation.comparison import load_engine
from llm.inference.sampling import SamplingParams
from llm.tokenizer.bpe import Message

PROMPTS = ["Как дела?", "Что такое искусственный интеллект?", "Сколько будет 25 умножить на 48?"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def raw_greedy(engine, ids: list[int], limit: int) -> tuple[list[int], str]:
    stops = {engine.tokenizer.eos_id, *engine._chat_stop_tokens()}
    history = list(ids)
    out: list[int] = []
    with torch.inference_mode():
        for _ in range(limit):
            logits = engine.model(torch.tensor([history], device=engine.device))[0, -1]
            token = int(logits.argmax())
            if token in stops:
                return out, "stop"
            out.append(token)
            history.append(token)
    return out, "length"


def api_call(url: str, prompt: str, limit: int) -> dict:
    body = json.dumps(
        {"messages": [{"role": "user", "content": prompt}], "max_tokens": limit, "temperature": 0}
    ).encode()
    request = urllib.request.Request(
        f"{url}/v1/chat/completions", body, {"content-type": "application/json"}
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        return json.load(response)


def main() -> int:
    parser = argparse.ArgumentParser(prog="diagnose_model")
    parser.add_argument("checkpoint")
    parser.add_argument("--tokenizer", default="data/tokenizer")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--api", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    engine = load_engine(args.checkpoint, args.tokenizer, args.device)
    tokenizer = engine.tokenizer
    report: dict[str, Any] = {
        "checkpoint": args.checkpoint,
        "weights_sha256": sha256(Path(args.checkpoint) / "model.safetensors"),
        "tokenizer_version": tokenizer.version,
        "cases": [],
    }
    greedy = SamplingParams(max_new_tokens=args.max_new_tokens, temperature=0.0)
    for prompt in PROMPTS:
        messages = [Message("user", prompt)]
        ids = tokenizer.encode_conversation(messages, add_generation_prompt=True)
        raw_ids, raw_reason = raw_greedy(engine, ids, args.max_new_tokens)
        generation = engine.chat(messages, greedy)
        case: dict[str, Any] = {
            "prompt": prompt,
            "input_tokens": ids,
            "input_decoded": tokenizer.decode(ids),
            "raw": {
                "tokens": raw_ids,
                "text": tokenizer.decode(raw_ids, skip_special_tokens=True),
                "finish": raw_reason,
            },
            "engine": {
                "tokens": generation.token_ids,
                "text": generation.text,
                "finish": generation.finish_reason,
            },
            "engine_matches_raw": generation.token_ids == raw_ids,
        }
        if args.api:
            reply = api_call(args.api, prompt, args.max_new_tokens)
            api_text = reply["choices"][0]["message"]["content"]
            case["api"] = {
                "text": api_text,
                "finish": reply["choices"][0]["finish_reason"],
                "model": reply["model"],
                "prompt_tokens": reply["usage"]["prompt_tokens"],
            }
            case["api_matches_engine"] = api_text == generation.text
        report["cases"].append(case)
    Path(args.output).write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
