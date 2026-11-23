from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

import torch
from torch.nn import functional

from llm.evaluation.v4_review import proportion
from llm.inference.engine import InferenceEngine


@dataclass(frozen=True)
class Scored:
    id: str
    language: str
    task: str
    group: str
    chance: float
    correct_normalized: bool
    correct_total: bool


@torch.inference_mode()
def option_log_probs(
    engine: InferenceEngine, prefix: str, options: list[str]
) -> list[tuple[float, int]]:
    tokenizer = engine.tokenizer
    context = tokenizer.encode(prefix, add_bos=True)
    results = []
    for option in options:
        continuation = tokenizer.encode(f" {option}")
        ids = (context + continuation)[-engine.max_positions :]
        count = min(len(continuation), len(ids) - 1)
        logits = engine.model(torch.tensor([ids[:-1]], device=engine.device))[0].float()
        log_probs = functional.log_softmax(logits[-count:], dim=-1)
        targets = torch.tensor(ids[-count:], device=engine.device)
        results.append((float(log_probs.gather(-1, targets.unsqueeze(-1)).sum()), count))
    return results


def score_item(engine: InferenceEngine, item: Mapping[str, Any]) -> Scored:
    scores = option_log_probs(engine, item["prefix"], item["options"])
    normalized = max(range(len(scores)), key=lambda i: scores[i][0] / scores[i][1])
    total = max(range(len(scores)), key=lambda i: scores[i][0])
    return Scored(
        item["id"],
        item["language"],
        item["task"],
        item["group"],
        1 / len(item["options"]),
        normalized == item["answer"],
        total == item["answer"],
    )


def summarize(scored: Iterable[Scored]) -> dict[str, Any]:
    rows = list(scored)

    def cell(subset: list[Scored]) -> dict[str, Any]:
        wins = proportion(sum(row.correct_normalized for row in subset), len(subset))
        wins["chance"] = sum(row.chance for row in subset) / len(subset)
        wins["above_chance"] = wins["wilson_95"][0] > wins["chance"]
        wins["total_logprob_accuracy"] = sum(row.correct_total for row in subset) / len(subset)
        return wins

    report: dict[str, Any] = {"overall": cell(rows)}
    for key in ("group", "task", "language"):
        buckets: dict[str, list[Scored]] = defaultdict(list)
        for row in rows:
            buckets[getattr(row, key)].append(row)
        report[f"by_{key}"] = {name: cell(items) for name, items in sorted(buckets.items())}
    pairs: dict[str, list[Scored]] = defaultdict(list)
    for row in rows:
        pairs[f"{row.group}/{row.language}"].append(row)
    report["by_group_language"] = {name: cell(items) for name, items in sorted(pairs.items())}
    return report
