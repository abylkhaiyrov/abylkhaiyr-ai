from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from llm.core.config import ExperimentConfig
from llm.core.profiles import count_parameters

REGISTRY_FILE = Path("experiments/registry.jsonl")
COMPARED_METRICS = (
    "validation_loss",
    "test_perplexity",
    "arithmetic_exact_match",
    "gsm8k_exact_match",
    "code_pass_at_1",
    "code_pass_at_4",
    "instruction_following_accuracy",
    "perplexity_kk",
    "perplexity_ru",
    "perplexity_en",
    "generation_tokens_per_second",
    "ram_gb",
)


def record_experiment(
    name: str, details: dict[str, object], results: dict[str, float], path: Path = REGISTRY_FILE
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    entry = {"name": name, **details, "results": results}
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry) + "\n")


def load_registry(path: Path = REGISTRY_FILE) -> list[dict]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def comparison_table(entries: list[dict], metrics: tuple[str, ...] = COMPARED_METRICS) -> str:
    names = [entry["name"] for entry in entries]
    width = max([len(n) for n in names] + [len("metric")])
    lines = ["metric".ljust(34) + " ".join(n.rjust(width) for n in names)]
    for metric in metrics:
        cells = []
        for entry in entries:
            value = entry["results"].get(metric)
            cells.append(("-" if value is None else f"{value:.4g}").rjust(width))
        lines.append(metric.ljust(34) + " ".join(cells))
    return "\n".join(lines)


def training_record(
    config: ExperimentConfig, result: Any, benchmarks: dict[str, float] | None = None
) -> tuple[dict[str, object], dict[str, float]]:
    train = config.train
    details: dict[str, object] = {
        "kind": "training",
        "stage": train.stage,
        "architecture": config.model.model_dump(),
        "parameters": count_parameters(config.model),
        "seed": train.seed,
        "dataset_version": train.dataset_version,
        "data_dir": train.data_dir,
        "init_from": train.init_from,
        "train_config": train.model_dump(),
        "tokens_processed": result.tokens_seen,
        "steps": result.final_step,
        "duration_seconds": result.duration_seconds,
        "interrupted": result.interrupted,
    }
    validation = result.best_validation_loss
    results = {
        "train_loss": result.final_train_loss,
        "validation_loss": validation,
        "perplexity": math.exp(min(validation, 50.0))
        if math.isfinite(validation)
        else float("nan"),
        "tokens_per_second": result.tokens_per_second,
        "device_memory_gb": result.device_memory_gb,
        **(benchmarks or {}),
    }
    return details, results
