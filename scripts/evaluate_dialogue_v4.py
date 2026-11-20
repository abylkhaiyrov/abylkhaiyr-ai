from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from statistics import mean
from typing import Any

import torch

from llm.evaluation.comparison import load_engine
from llm.evaluation.conversation import score_case
from llm.evaluation.dialogue_v4 import PROBES, benchmark_cases, loop_rate
from llm.evaluation.engine import EvaluationConfig, Evaluator
from llm.evaluation.metrics import prompt_echo
from llm.inference.engine import InferenceEngine
from llm.inference.sampling import SamplingParams
from llm.training.checkpoint import CheckpointManager
from llm.training.promotion import dialogue_v4_failures

PRESETS = {
    "greedy": SamplingParams(greedy=True, max_new_tokens=64, repetition_penalty=1.1),
    "low": SamplingParams(
        temperature=0.3, top_k=40, top_p=0.9, max_new_tokens=64, repetition_penalty=1.1, seed=42
    ),
    "wide": SamplingParams(
        temperature=0.75, top_k=64, top_p=0.95, max_new_tokens=96, repetition_penalty=1.15, seed=42
    ),
    "short": SamplingParams(
        temperature=0.3, top_k=20, top_p=0.85, max_new_tokens=32, repetition_penalty=1.05, seed=42
    ),
}


def probes(engine: InferenceEngine, params: SamplingParams) -> dict[str, Any]:
    result = {}
    for key, messages in PROBES.items():
        generation = engine.chat(messages, params)
        result[key] = {"answer": generation.text, "finish": generation.finish_reason}
    return result


def benchmark(engine: InferenceEngine, batch_size: int, params: SamplingParams) -> dict[str, Any]:
    cases = benchmark_cases()
    rows = []
    for start in range(0, len(cases), batch_size):
        batch = cases[start : start + batch_size]
        for case, generation in zip(
            batch, engine.chat_batch([item.messages() for item in batch], params), strict=True
        ):
            score = score_case(case, generation.text, generation.finish_reason == "stop", None)
            rows.append(
                {
                    "case": case.id,
                    "category": case.category,
                    "language": case.language,
                    "answer": generation.text,
                    "passed": score.passed,
                    "echo": prompt_echo(case.prompt, generation.text),
                    "loop": loop_rate(generation.text),
                    "finished": score.finished,
                }
            )
    metrics = {
        "pass_rate": mean(row["passed"] for row in rows),
        "echo_rate": mean(row["echo"] for row in rows),
        "repetition_rate": mean(row["loop"] for row in rows),
        "completion_rate": mean(row["finished"] for row in rows),
    }
    for field in ("language", "category"):
        for value in sorted({row[field] for row in rows}):
            metrics[f"{field}_{value}"] = mean(row["passed"] for row in rows if row[field] == value)
    metrics["dialogue_v4_cases"] = float(len(rows))
    metrics["prompt_echo"] = metrics["echo_rate"]
    for language in ("ru", "en", "kk"):
        metrics[f"dialogue_v4_{language}"] = metrics[f"language_{language}"]
    for category, key in (("instruction", "instruction_exact"), ("multi_turn", "multi_turn_exact")):
        metrics[key] = mean(
            row["passed"]
            for row in rows
            if row["category"] == category and row["case"].startswith("v4-")
        )
    return {"metrics": metrics, "cases": rows}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoints", nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch", type=int, default=24)
    parser.add_argument("--probes-only", action="store_true")
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--reference-report", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(4)
    report: dict[str, object] = {}
    reference: dict[str, float] | None = None
    if args.reference_report:
        previous = json.loads(args.reference_report.read_text(encoding="utf-8"))
        reference = next(iter(previous.values()))["metrics"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for name in args.checkpoints:
        path = Path(name)
        meta = CheckpointManager.verify(path)
        engine = load_engine(name, "data/tokenizer", args.device)
        if engine.tokenizer.version != meta["tokenizer_version"]:
            raise ValueError(f"tokenizer mismatch: {path}")
        if engine.tokenizer.vocab_size != engine.model.config.vocab_size:
            raise ValueError(f"vocab mismatch: {path}")
        result: dict[str, Any] = {
            "step": meta["step"],
            "parameters": sum(p.numel() for p in engine.model.parameters()),
            "probes": probes(engine, PRESETS["low"]),
        }
        for key in ("ru_smalltalk", "ru_context", "kk_smalltalk"):
            messages = PROBES[key]
            greedy = PRESETS["greedy"]
            cached = engine.chat(messages, greedy)
            uncached = engine.chat(messages, replace(greedy, use_cache=False))
            result.setdefault("cache_parity", {})[key] = cached.token_ids == uncached.token_ids
        if args.sweep:
            result["sampling"] = {
                key: probes(engine, settings) for key, settings in PRESETS.items()
            }
            result["sampling"]["stop_sequence"] = probes(
                engine, replace(PRESETS["low"], stop_sequences=["\n\n"])
            )
        if not args.probes_only:
            result.update(benchmark(engine, args.batch, PRESETS["greedy"]))
            metrics = result["metrics"]
            if reference is None:
                reference = metrics
            result["gate_failures"] = dialogue_v4_failures(metrics, reference)
            evaluator = Evaluator(engine, EvaluationConfig("data/processed", max_windows=32))
            result["retention"] = {
                **evaluator.language_modeling(),
                **evaluator.multilingual_perplexity(),
            }
            del evaluator
        report[name] = result
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(
            name,
            json.dumps(result.get("metrics", result["probes"]), ensure_ascii=False),
            flush=True,
        )
        del engine


if __name__ == "__main__":
    main()
