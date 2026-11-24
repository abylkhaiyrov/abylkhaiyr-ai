from __future__ import annotations

from pathlib import Path
from typing import Any

from llm.core.checksums import file_sha256
from llm.data.binary import read_jsonl
from llm.data.refusals import is_refusal
from llm.evaluation.outcomes import Outcome, SuiteOutput
from llm.inference.engine import InferenceEngine
from llm.inference.sampling import SamplingParams
from llm.tokenizer.bpe import Message

SOURCES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("configs/natural_eval_v4.jsonl", ("unknown",)),
    ("configs/natural_eval_v5.jsonl", ("unknown_entity",)),
    ("configs/chat_eval_natural_v1.jsonl", ()),
)
SAMPLED_SEEDS = (0, 1, 2)
MAX_NEW_TOKENS = 96
Case = tuple[str, list[Message], bool]


def questions(path: str, justified: frozenset[str]) -> list[Case]:
    result = []
    for index, row in enumerate(read_jsonl(Path(path))):
        messages = row["messages"]
        if messages[-1]["role"] == "assistant":
            messages = messages[:-1]
        expects = row.get("category") in justified
        turns = [Message(item["role"], item["content"]) for item in messages]
        result.append((f"{Path(path).stem}:{index}", turns, expects))
    return result


def settings(mode: str) -> list[SamplingParams]:
    if mode == "greedy":
        return [SamplingParams(max_new_tokens=MAX_NEW_TOKENS, greedy=True, repetition_penalty=1.1)]
    return [
        SamplingParams(
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=0.7,
            top_p=0.9,
            top_k=40,
            repetition_penalty=1.1,
            seed=seed,
        )
        for seed in SAMPLED_SEEDS
    ]


def refusals(engine: InferenceEngine, cases: list[Case], params: SamplingParams) -> list[bool]:
    return [is_refusal(engine.chat(messages, params).text) for _, messages, _ in cases]


def evaluate(engine: InferenceEngine, options: dict[str, Any]) -> SuiteOutput:
    sources = options.get("sources", [[path, list(kinds)] for path, kinds in SOURCES])
    cases = [case for path, kinds in sources for case in questions(path, frozenset(kinds))]
    output = SuiteOutput(inputs={path: file_sha256(Path(path)) for path, _ in sources})
    for mode in options.get("modes", ["greedy", "sampled"]):
        runs = [refusals(engine, cases, params) for params in settings(mode)]
        unjustified, justified, answered = [], [], []
        for flags in runs:
            rows = list(zip(cases, flags, strict=True))
            unjustified.append(
                [Outcome(i, i, "answerable", int(r)) for (i, _, expects), r in rows if not expects]
            )
            justified.append(
                [Outcome(i, i, "unanswerable", int(r)) for (i, _, expects), r in rows if expects]
            )
            answered.append(
                [
                    Outcome(i, i, "unanswerable", int(not r))
                    for (i, _, expects), r in rows
                    if expects
                ]
            )
        output.add_rate(f"refusal.{mode}.unjustified", unjustified)
        output.add_rate(f"refusal.{mode}.justified", justified)
        output.add_rate(f"refusal.{mode}.answered_unanswerable", answered)
        flat = [
            (expects, refused)
            for flags in runs
            for (_, _, expects), refused in zip(cases, flags, strict=True)
        ]
        total = sum(refused for _, refused in flat)
        if total:
            output.metrics[f"refusal.{mode}.precision"] = (
                sum(refused for expects, refused in flat if expects) / total
            )
    return output
