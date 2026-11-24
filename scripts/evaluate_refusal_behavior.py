from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from llm.evaluation.comparison import load_engine
from llm.evaluation.refusal_behavior import SOURCES, Case, questions, refusals, settings
from llm.evaluation.v4_review import proportion
from llm.inference.engine import InferenceEngine


def measure(engine: InferenceEngine, cases: list[Case], mode: str) -> dict[str, Any]:
    flat = [
        (expects, refused)
        for params in settings(mode)
        for (_, _, expects), refused in zip(cases, refusals(engine, cases, params), strict=True)
    ]
    answerable = [refused for expects, refused in flat if not expects]
    unanswerable = [refused for expects, refused in flat if expects]
    total = sum(refused for _, refused in flat)
    return {
        "unjustified_refusal_rate": proportion(sum(answerable), len(answerable)),
        "justified_refusal_rate": proportion(sum(unanswerable), len(unanswerable)),
        "answered_unanswerable_rate": proportion(
            len(unanswerable) - sum(unanswerable), len(unanswerable)
        ),
        "refusal_precision": sum(unanswerable) / total if total else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("models", nargs="+", help="name=checkpoint")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cases = [case for path, kinds in SOURCES for case in questions(path, frozenset(kinds))]
    report = {}
    for entry in args.models:
        name, checkpoint = entry.split("=", 1)
        engine = load_engine(checkpoint, "data/tokenizer", args.device)
        report[name] = {mode: measure(engine, cases, mode) for mode in ("greedy", "sampled")}
        print(name, "done", flush=True)
    args.output.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
