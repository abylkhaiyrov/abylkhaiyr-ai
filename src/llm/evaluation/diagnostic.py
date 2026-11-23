from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
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


@dataclass(frozen=True)
class EncodedOption:
    context: list[int]
    continuation: list[int]


def encode_options(
    engine: InferenceEngine,
    prefix: str,
    options: Sequence[str],
    separator: str = " ",
) -> list[EncodedOption]:
    tokenizer = engine.tokenizer
    context = tokenizer.encode(prefix, add_bos=True)
    alternatives = tokenizer.encode_batch([separator + option for option in options])
    if not alternatives or any(not ids for ids in alternatives):
        raise ValueError("diagnostic options must tokenize to non-empty sequences")
    room = engine.max_positions - max(map(len, alternatives))
    if room < 1:
        raise ValueError("diagnostic option exceeds the model window")
    shared = context[-room:]
    return [EncodedOption(shared, ids) for ids in alternatives]


@torch.inference_mode()
def score_encoded(
    engine: InferenceEngine,
    options: Sequence[EncodedOption],
) -> list[tuple[float, int]]:
    if not options:
        return []
    inputs = [
        torch.tensor([*option.context, *option.continuation][:-1], dtype=torch.long)
        for option in options
    ]
    batch = torch.nn.utils.rnn.pad_sequence(
        inputs,
        batch_first=True,
        padding_value=engine.tokenizer.pad_id,
    ).to(engine.device)
    logits = engine.model(batch)
    rows = []
    positions: list[int] = []
    targets = []
    counts = []
    for index, option in enumerate(options):
        count = len(option.continuation)
        last_context = len(option.context) - 1
        rows.extend([index] * count)
        positions.extend(range(last_context, last_context + count))
        targets.extend(option.continuation)
        counts.append(count)
    selection = logits[
        torch.tensor(rows, device=engine.device), torch.tensor(positions, device=engine.device)
    ].float()
    target_ids = torch.tensor(targets, device=engine.device).unsqueeze(-1)
    values = functional.log_softmax(selection, dim=-1).gather(-1, target_ids).squeeze(-1)
    totals = torch.stack([chunk.sum() for chunk in values.split(counts)]).tolist()
    return list(zip(totals, counts, strict=True))


def option_log_probs(
    engine: InferenceEngine,
    prefix: str,
    options: list[str],
) -> list[tuple[float, int]]:
    return score_encoded(engine, encode_options(engine, prefix, options))


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
