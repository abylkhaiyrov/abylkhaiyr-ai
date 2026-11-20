from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from llm.core.hardware import resolve_device
from llm.evaluation.conversation import (
    CASES,
    BenchmarkReport,
    ConversationCase,
    WordList,
    engine_responder,
    run_conversation_benchmark,
)
from llm.inference.engine import InferenceEngine
from llm.inference.sampling import SamplingParams
from llm.tokenizer.bpe import BPETokenizer
from llm.training.checkpoint import CheckpointManager, load_model_for_inference

FIXED_PARAMS = SamplingParams(greedy=True, max_new_tokens=96, repetition_penalty=1.1)
TASK_GROUPS: dict[str, frozenset[str]] = {
    "dialogue": frozenset({"greeting", "identity", "clarify", "unknown", "multi_turn"}),
    "explanation": frozenset({"explain", "context"}),
    "math": frozenset({"math", "logic"}),
    "code": frozenset({"code"}),
    "instruction": frozenset({"instruction", "translation"}),
}
SUMMARY_KEYS = (
    "conv_pass_rate",
    "conv_check_pass_rate",
    "conv_completion_rate",
    "conv_language_stability",
    "conv_repetition_ratio",
    "conv_wellformed_word_rate",
    "conv_non_empty_rate",
)


def load_engine(checkpoint: str, tokenizer_path: str, device: str) -> InferenceEngine:
    resolved = CheckpointManager(checkpoint, keep=1).latest_valid() or Path(checkpoint)
    model, _, _ = load_model_for_inference(resolved)
    return InferenceEngine(model, BPETokenizer.load(tokenizer_path), resolve_device(device))


def word_list(data_dir: str) -> WordList | None:
    corpus = Path(data_dir) / "corpus" / "validation.jsonl"
    return WordList.from_jsonl(corpus) if corpus.is_file() else None


def compare_checkpoints(
    models: Mapping[str, str],
    tokenizer_path: str,
    device: str,
    data_dir: str,
    params: SamplingParams = FIXED_PARAMS,
) -> dict[str, BenchmarkReport]:
    words = word_list(data_dir)
    reports: dict[str, BenchmarkReport] = {}
    for name, checkpoint in models.items():
        engine = load_engine(checkpoint, tokenizer_path, device)
        reports[name] = run_conversation_benchmark(engine_responder(engine, params), words)
        del engine
    return reports


def markdown_table(
    reports: Mapping[str, BenchmarkReport], keys: Iterable[str] = SUMMARY_KEYS
) -> str:
    columns = list(keys)
    lines = ["| model | " + " | ".join(columns) + " |", "|---|" + "---|" * len(columns)]
    for name, report in reports.items():
        cells = " | ".join(f"{report.metrics.get(key, float('nan')):.3f}" for key in columns)
        lines.append(f"| {name} | {cells} |")
    return "\n".join(lines)


def category_table(reports: Mapping[str, BenchmarkReport]) -> str:
    categories = sorted({item.category for item in CASES})
    lines = ["| model | " + " | ".join(categories) + " |", "|---|" + "---|" * len(categories)]
    for name, report in reports.items():
        cells = " | ".join(
            f"{report.metrics.get(f'conv_category_{c}', 0.0):.2f}" for c in categories
        )
        lines.append(f"| {name} | {cells} |")
    return "\n".join(lines)


def save_comparison(reports: Mapping[str, BenchmarkReport], path: str | Path) -> None:
    payload = {
        name: {"metrics": report.metrics, "samples": report.samples()}
        for name, report in reports.items()
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


@dataclass(frozen=True)
class SamplingVariant:
    name: str
    params: SamplingParams


@dataclass
class SweepRow:
    variant: str
    group: str
    pass_rate: float
    completion_rate: float
    repetition: float
    language_stability: float


@dataclass
class SweepResult:
    rows: list[SweepRow] = field(default_factory=list)

    def best_by_group(self) -> dict[str, SweepRow]:
        best: dict[str, SweepRow] = {}
        for row in self.rows:
            current = best.get(row.group)
            if current is None or score(row) > score(current):
                best[row.group] = row
        return best


def score(row: SweepRow) -> float:
    return row.pass_rate + 0.25 * row.completion_rate - 0.5 * row.repetition


def default_variants(max_new_tokens: int = 96) -> list[SamplingVariant]:
    def make(name: str, **kwargs) -> SamplingVariant:
        return SamplingVariant(name, SamplingParams(**{"max_new_tokens": max_new_tokens, **kwargs}))

    return [
        make("greedy_rp1.0", greedy=True, repetition_penalty=1.0),
        make("greedy_rp1.1", greedy=True, repetition_penalty=1.1),
        make("greedy_rp1.2", greedy=True, repetition_penalty=1.2),
        make("t0.3_p0.9", temperature=0.3, top_p=0.9),
        make("t0.5_p0.9", temperature=0.5, top_p=0.9),
        make("t0.5_p0.9_rp1.1", temperature=0.5, top_p=0.9, repetition_penalty=1.1),
        make("t0.7_p0.9", temperature=0.7, top_p=0.9),
        make("t0.7_p0.95_k40", temperature=0.7, top_p=0.95, top_k=40),
        make("t0.7_p0.9_rp1.1", temperature=0.7, top_p=0.9, repetition_penalty=1.1),
        make("t1.0_p0.95", temperature=1.0, top_p=0.95),
        make("t0.5_p0.9_stop", temperature=0.5, top_p=0.9, stop_sequences=["\n\n\n"]),
        make("greedy_rp1.1_len48", greedy=True, repetition_penalty=1.1, max_new_tokens=48),
        make("greedy_rp1.1_len160", greedy=True, repetition_penalty=1.1, max_new_tokens=160),
    ]


def cases_for(group: str) -> list[ConversationCase]:
    return [item for item in CASES if item.category in TASK_GROUPS[group]]


def sweep_sampling(
    engine: InferenceEngine,
    variants: Iterable[SamplingVariant],
    seeds: tuple[int, ...] = (0, 1, 2),
    words: WordList | None = None,
) -> SweepResult:
    result = SweepResult()
    for variant in variants:
        seeded = (
            [variant.params]
            if variant.params.greedy
            else [SamplingParams(**{**variant.params.__dict__, "seed": seed}) for seed in seeds]
        )
        for group in TASK_GROUPS:
            reports = [
                run_conversation_benchmark(
                    engine_responder(engine, params),
                    words,
                    cases_for(group),
                ).metrics
                for params in seeded
            ]
            result.rows.append(
                SweepRow(
                    variant.name,
                    group,
                    float(np.mean([m["conv_pass_rate"] for m in reports])),
                    float(np.mean([m["conv_completion_rate"] for m in reports])),
                    float(np.mean([m["conv_repetition_ratio"] for m in reports])),
                    float(np.mean([m["conv_language_stability"] for m in reports])),
                )
            )
    return result
