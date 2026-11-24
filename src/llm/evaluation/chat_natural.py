from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from llm.core.checksums import file_sha256
from llm.data.binary import read_jsonl
from llm.data.refusals import is_refusal
from llm.evaluation.metrics import normalize_words, prompt_echo, repeated_ngram_ratio
from llm.evaluation.outcomes import Outcome, SuiteOutput
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
UNANSWERABLE_CATEGORY = "unknown_entity"
RATE_KEYS = (
    "finished",
    "language_match",
    "echo",
    "topic_proxy",
    "correct",
    "unjustified_refusal",
    "justified_refusal",
    "answered_unanswerable",
)


def judge(case: dict, text: str, finished: bool) -> dict[str, Any]:
    folded = text.casefold()
    words = normalize_words(text)
    topical = bool(case["topic_any"]) and any(item in folded for item in case["topic_any"])
    topical = topical or (
        bool(case["topic_all"]) and all(item in folded for item in case["topic_all"])
    )
    unanswerable = case.get("category") == UNANSWERABLE_CATEGORY
    refused = is_refusal(text)
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
        "canned_refusal": refused,
        "unjustified_refusal": None if unanswerable else refused,
        "justified_refusal": refused if unanswerable else None,
        "answered_unanswerable": (not refused) if unanswerable else None,
        "topic_proxy": topical if (case["topic_any"] or case["topic_all"]) else None,
        "correct": accepted(text, case["accept"]) if case["accept"] else None,
    }


def conversation(case: dict) -> list[Message]:
    history = [
        Message(role, content)
        for pair in case["history"]
        for role, content in zip(("user", "assistant"), pair, strict=True)
    ]
    return [*history, Message("user", case["question"])]


def run(engine: InferenceEngine, cases: list[dict], config: dict, seed: int | None) -> list[dict]:
    params = SamplingParams(max_new_tokens=MAX_NEW_TOKENS, seed=seed, **config)
    rows = []
    for case in cases:
        reply = engine.chat(conversation(case), params)
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


def runs_for(
    engine: InferenceEngine, cases: list[dict], config: dict, seeds: Sequence[int]
) -> list[list[dict]]:
    if config.get("greedy", False):
        return [run(engine, cases, config, None)]
    return [run(engine, cases, config, seed) for seed in seeds]


def outcomes(rows: list[dict], key: str) -> list[Outcome]:
    return [
        Outcome(row["id"], row["id"], row["language"], int(bool(row[key])))
        for row in rows
        if row[key] is not None
    ]


def evaluate(engine: InferenceEngine, options: dict[str, Any]) -> SuiteOutput:
    path = Path(options["cases"])
    cases = list(read_jsonl(path))
    output = SuiteOutput(inputs={str(path): file_sha256(path)})
    seeds = tuple(options.get("seeds", SEEDS))
    for label in options.get("configs", list(CONFIGS)):
        runs = runs_for(engine, cases, CONFIGS[label], seeds)
        for key in RATE_KEYS:
            output.add_rate(f"chat.{label}.{key}", [outcomes(rows, key) for rows in runs])
        repetition = [row["repetition"] for rows in runs for row in rows]
        output.metrics[f"chat.{label}.repetition"] = sum(repetition) / len(repetition)
    return output
