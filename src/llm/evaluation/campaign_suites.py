from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from llm.core.checksums import file_sha256
from llm.data.binary import read_jsonl
from llm.evaluation import chat_natural, refusal_behavior
from llm.evaluation.capacity_metrics import unigram_probabilities
from llm.evaluation.diagnostic_bank import Item
from llm.evaluation.diagnostic_controls import choices, score_controls
from llm.evaluation.engine import EvaluationConfig, Evaluator
from llm.evaluation.outcomes import Outcome, SuiteOutput
from llm.evaluation.v4_language import LANGUAGES, language_losses
from llm.inference.engine import InferenceEngine

Suite = Callable[[InferenceEngine, dict[str, Any]], SuiteOutput]
MCQ_METHODS = ("calibrated", "normalized")


def perplexity(engine: InferenceEngine, options: dict[str, Any]) -> SuiteOutput:
    output = SuiteOutput()
    for name, (root, split, batches) in options["sets"].items():
        for language, values in language_losses(engine, Path(root), batches, split=split).items():
            output.metrics[f"ppl.{name}.{language}"] = float(values["perplexity"])
        output.inputs[f"{root}/{split}"] = file_sha256(Path(root) / f"{split}.{LANGUAGES[0]}.bin")
    return output


def load_items(path: str) -> list[Item]:
    return [Item(**{**row, "options": tuple(row["options"])}) for row in read_jsonl(Path(path))]


def mcq(engine: InferenceEngine, options: dict[str, Any]) -> SuiteOutput:
    output = SuiteOutput(inputs={options["unigram"]: file_sha256(Path(options["unigram"]))})
    unigram = unigram_probabilities(Path(options["unigram"]), engine.tokenizer)
    for bank, path in options["banks"].items():
        output.inputs[path] = file_sha256(Path(path))
        rows = list(score_controls(engine, load_items(path), unigram))
        for method in options.get("methods", MCQ_METHODS):
            outcomes = [
                Outcome(c.id, c.family, c.task, int(c.choice == c.answer))
                for c in choices(rows, method)
            ]
            output.add_rate(f"mcq.{bank}.{method}.all", [outcomes])
            for task in sorted({outcome.group for outcome in outcomes}):
                scoped = [outcome for outcome in outcomes if outcome.group == task]
                output.add_rate(f"mcq.{bank}.{method}.{task}", [scoped])
    return output


def core(engine: InferenceEngine, options: dict[str, Any]) -> SuiteOutput:
    evaluator = Evaluator(engine, EvaluationConfig(data_dir=options.get("data_dir", "data")))
    output = SuiteOutput()
    for method in options["methods"]:
        for name, value in getattr(evaluator, method)().items():
            output.metrics[f"core.{name}"] = float(value)
    return output


SUITES: dict[str, Suite] = {
    "perplexity": perplexity,
    "mcq": mcq,
    "core": core,
    "chat_natural": chat_natural.evaluate,
    "refusal_behavior": refusal_behavior.evaluate,
}
