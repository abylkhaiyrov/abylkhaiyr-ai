from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from llm.data.binary import read_jsonl
from llm.evaluation.comparison import load_engine
from llm.evaluation.metrics import normalize_words, prompt_echo, repeated_ngram_ratio
from llm.evaluation.v4_review import script_matches
from llm.inference.engine import InferenceEngine
from llm.inference.sampling import SamplingParams
from llm.tokenizer.bpe import Message

LABELS = {
    "en": ("Question:", "Answer:"),
    "ru": ("Вопрос:", "Ответ:"),
    "kk": ("Сұрақ:", "Жауап:"),
}
PARAMS = SamplingParams(greedy=True, max_new_tokens=64, repetition_penalty=1.1)


def plain_prompt(messages: list[dict], language: str) -> str:
    question, answer = LABELS[language]
    turns = [
        f"{question if item['role'] == 'user' else answer} {item['content']}" for item in messages
    ]
    return "\n".join(turns) + f"\n{answer}"


def generate(engine: InferenceEngine, case: dict, form: str) -> dict:
    messages = case["messages"][:-1]
    if form == "chat":
        result = engine.chat([Message(item["role"], item["content"]) for item in messages], PARAMS)
        prompt = messages[-1]["content"]
    else:
        prompt = plain_prompt(messages, case["language"])
        result = engine.generate(prompt, PARAMS)
    words = normalize_words(result.text)
    return {
        "id": case["id"],
        "language": case["language"],
        "category": case["category"],
        "format": form,
        "prompt": prompt,
        "response": result.text,
        "response_sha256": hashlib.sha256(result.text.encode()).hexdigest(),
        "finished": result.finish_reason == "stop",
        "echo": prompt_echo(messages[-1]["content"], result.text),
        "repetition": repeated_ngram_ratio(words) if len(words) >= 6 else 0.0,
        "script_match": script_matches(case["language"], result.text),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("models", nargs="+", help="name=checkpoint")
    parser.add_argument("--cases", type=Path, default=Path("experiments/stage18/free_cases.jsonl"))
    parser.add_argument("--device", default="mps")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    cases = list(read_jsonl(args.cases))
    for entry in args.models:
        name, checkpoint = entry.split("=", 1)
        engine = load_engine(checkpoint, "data/tokenizer", args.device)
        rows = [generate(engine, case, form) for form in ("chat", "plain") for case in cases]
        (args.output / f"{name}.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
        )
        print(name, "done", flush=True)


if __name__ == "__main__":
    main()
