from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from llm.data.binary import read_jsonl
from llm.data.refusals import is_refusal
from llm.evaluation.comparison import load_engine
from llm.evaluation.metrics import normalize_words, prompt_echo, repeated_ngram_ratio
from llm.evaluation.v4_review import accepted, proportion, script_matches
from llm.inference.engine import InferenceEngine
from llm.inference.sampling import SamplingParams
from llm.tokenizer.bpe import Message

CONFIGS: dict[str, dict[str, Any]] = {
    "greedy_rp1.1": {"greedy": True, "repetition_penalty": 1.1},
    "greedy_rp1.0": {"greedy": True, "repetition_penalty": 1.0},
    "studio_default": {"temperature": 0.7, "top_p": 0.9, "top_k": 40, "repetition_penalty": 1.1},
    "studio_precise": {"temperature": 0.2, "top_p": 0.85, "top_k": 20, "repetition_penalty": 1.1},
    "hot": {"temperature": 1.0, "top_p": 0.95, "top_k": 80, "repetition_penalty": 1.1},
}
SEEDS = (0, 1, 2, 3, 4)
MAX_NEW_TOKENS = 128


def judge(case: dict, text: str, finished: bool) -> dict[str, Any]:
    folded = text.casefold()
    words = normalize_words(text)
    topical = bool(case["topic_any"]) and any(item in folded for item in case["topic_any"])
    topical = topical or (
        bool(case["topic_all"]) and all(item in folded for item in case["topic_all"])
    )
    return {
        "id": case["id"],
        "language": case["language"],
        "scenario": case["scenario"],
        "question": case["question"],
        "response": text,
        "finished": finished,
        "language_match": script_matches(case["language"], text),
        "echo": prompt_echo(case["question"], text),
        "repetition": repeated_ngram_ratio(words) if len(words) >= 6 else 0.0,
        "canned_refusal": is_refusal(text),
        "topic_proxy": topical if (case["topic_any"] or case["topic_all"]) else None,
        "correct": accepted(text, case["accept"]) if case["accept"] else None,
    }


def run(engine: InferenceEngine, cases: list[dict], config: dict, seed: int | None) -> list[dict]:
    params = SamplingParams(max_new_tokens=MAX_NEW_TOKENS, seed=seed, **config)
    rows = []
    for case in cases:
        messages = [
            Message(role, content)
            for pair in case["history"]
            for role, content in zip(("user", "assistant"), pair, strict=True)
        ] + [Message("user", case["question"])]
        reply = engine.chat(messages, params)
        rows.append(judge(case, reply.text, reply.finish_reason == "stop"))
    return rows


def summarize(rows: list[dict]) -> dict[str, Any]:
    def rate(key: str, scope: list[dict]) -> dict[str, Any]:
        values = [row[key] for row in scope if row[key] is not None]
        return proportion(sum(bool(v) for v in values), len(values))

    keys = ("finished", "language_match", "echo", "canned_refusal", "topic_proxy", "correct")
    by_language: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_language[row["language"]].append(row)
    return {
        "overall": {key: rate(key, rows) for key in keys},
        "mean_repetition": sum(row["repetition"] for row in rows) / len(rows),
        "by_language": {
            language: {key: rate(key, scope) for key in keys}
            for language, scope in sorted(by_language.items())
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("models", nargs="+", help="name=checkpoint")
    parser.add_argument("--cases", type=Path, default=Path("configs/chat_eval_natural_v1.jsonl"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    cases = list(read_jsonl(args.cases))
    for entry in args.models:
        name, checkpoint = entry.split("=", 1)
        engine = load_engine(checkpoint, "data/tokenizer", args.device)
        report: dict[str, Any] = {"checkpoint": checkpoint, "configs": {}, "rows": {}}
        for label, config in CONFIGS.items():
            sampled = not config.get("greedy", False)
            runs = [
                run(engine, cases, config, seed if sampled else None)
                for seed in (SEEDS if sampled else (None,))
            ]
            pooled = [row for rows in runs for row in rows]
            report["configs"][label] = {**summarize(pooled), "runs": len(runs)}
            report["rows"][label] = runs[0]
        (args.output / f"{name}.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
        )
        print(name, "done", flush=True)


if __name__ == "__main__":
    main()
