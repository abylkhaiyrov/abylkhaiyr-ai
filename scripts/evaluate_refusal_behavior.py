from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from llm.data.binary import read_jsonl
from llm.data.refusals import is_refusal
from llm.evaluation.comparison import load_engine
from llm.evaluation.v4_review import proportion
from llm.inference.engine import InferenceEngine
from llm.inference.sampling import SamplingParams
from llm.tokenizer.bpe import Message

SOURCES: tuple[tuple[Path, frozenset[str]], ...] = (
    (Path("configs/natural_eval_v4.jsonl"), frozenset({"unknown"})),
    (Path("configs/natural_eval_v5.jsonl"), frozenset({"unknown_entity"})),
    (Path("configs/chat_eval_natural_v1.jsonl"), frozenset()),
)
GREEDY = SamplingParams(max_new_tokens=96, greedy=True, repetition_penalty=1.1)
SAMPLED = tuple(
    SamplingParams(
        max_new_tokens=96, temperature=0.7, top_p=0.9, top_k=40, repetition_penalty=1.1, seed=seed
    )
    for seed in range(3)
)


def questions(path: Path, justified: frozenset[str]) -> list[tuple[list[Message], bool]]:
    result = []
    for row in read_jsonl(path):
        messages = row["messages"]
        if messages[-1]["role"] == "assistant":
            messages = messages[:-1]
        expects = row.get("category") in justified
        result.append(([Message(item["role"], item["content"]) for item in messages], expects))
    return result


def measure(
    engine: InferenceEngine,
    cases: list[tuple[list[Message], bool]],
    params: tuple[SamplingParams, ...],
) -> dict[str, Any]:
    outcomes = [
        (expects, is_refusal(engine.chat(messages, setting).text))
        for messages, expects in cases
        for setting in params
    ]
    answerable = [refused for expects, refused in outcomes if not expects]
    unanswerable = [refused for expects, refused in outcomes if expects]
    refusals = sum(refused for _, refused in outcomes)
    return {
        "unjustified_refusal_rate": proportion(sum(answerable), len(answerable)),
        "justified_refusal_rate": proportion(sum(unanswerable), len(unanswerable)),
        "answered_unanswerable_rate": proportion(
            len(unanswerable) - sum(unanswerable), len(unanswerable)
        ),
        "refusal_precision": sum(refused for expects, refused in outcomes if expects) / refusals
        if refusals
        else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("models", nargs="+", help="name=checkpoint")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cases = [case for path, justified in SOURCES for case in questions(path, justified)]
    report = {}
    for entry in args.models:
        name, checkpoint = entry.split("=", 1)
        engine = load_engine(checkpoint, "data/tokenizer", args.device)
        report[name] = {
            "greedy": measure(engine, cases, (GREEDY,)),
            "sampled": measure(engine, cases, SAMPLED),
        }
        print(name, "done", flush=True)
    args.output.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
