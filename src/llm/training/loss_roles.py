from __future__ import annotations

import statistics
from collections import defaultdict
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import torch

from llm.data.binary import IGNORE_INDEX, read_jsonl
from llm.data.grounded import REFUSAL, Example, prompt_of, training_examples
from llm.data.grounded_text import LANGUAGES
from llm.data.tokenizer_audit import encoded
from llm.inference.engine import InferenceEngine
from llm.tokenizer.bpe import Message

ROLES = ("refusal", "template", "entity", "value", "eos")
INDEX_POOL = 7_000


def example_index(language: str, seed: int, missing_share: float) -> dict[str, Example]:
    return {
        prompt_of(example.plan, example.structure): example
        for example in training_examples(language, INDEX_POOL, seed, missing_share=missing_share)
    }


def spans_of(text: str, needle: str) -> list[tuple[int, int]]:
    found, start = [], 0
    while needle and (start := text.find(needle, start)) >= 0:
        found.append((start, start + len(needle)))
        start += len(needle)
    return found


def token_roles(example: Example, answer: str, offsets: Sequence[tuple[int, int]]) -> list[str]:
    plan = example.plan
    if plan.expects == REFUSAL:
        return ["refusal"] * len(offsets)
    value = spans_of(answer, plan.answer)
    entity = [] if plan.answer == plan.entity else spans_of(answer, plan.entity)
    roles = []
    for start, end in offsets:
        if any(a < end and start < b for a, b in value):
            roles.append("value")
        elif any(a < end and start < b for a, b in entity):
            roles.append("entity")
        else:
            roles.append("template")
    return roles


def row_roles(
    engine: InferenceEngine, row: dict[str, Any], example: Example
) -> tuple[list[str], list[float], list[bool]]:
    tokenizer = engine.tokenizer
    messages = [Message(turn["role"], turn["content"], True) for turn in row["messages"]]
    ids, labels = tokenizer.encode_conversation_with_labels(messages, IGNORE_INDEX)
    positions = [i for i, label in enumerate(labels) if label != IGNORE_INDEX]
    answer = row["messages"][-1]["content"]
    _, offsets = encoded(tokenizer, answer)
    roles = token_roles(example, answer, offsets) + ["eos"]
    with torch.no_grad():
        logits = engine.model(torch.tensor([ids[:-1]], device=engine.device)).float()[0]
    logprobs = torch.log_softmax(logits, dim=-1)
    losses = [-logprobs[p - 1, ids[p]].item() for p in positions]
    hits = [int(logprobs[p - 1].argmax().item() == ids[p]) for p in positions]
    return roles, losses, [bool(hit) for hit in hits]


def role_losses(
    engine: InferenceEngine, paths: Iterable[Path], seed: int, missing_share: float, limit: int
) -> dict[str, Any]:
    engine.model.eval()
    indexes = {language: example_index(language, seed, missing_share) for language in LANGUAGES}
    losses: dict[str, list[float]] = defaultdict(list)
    first: list[float] = []
    rest: list[float] = []
    exact: list[float] = []
    for path in paths:
        seen: dict[str, int] = defaultdict(int)
        for row in read_jsonl(path):
            if seen[row["category"]] >= limit:
                continue
            seen[row["category"]] += 1
            example = indexes[row["language"]][row["messages"][0]["content"]]
            roles, token_losses, hits = row_roles(engine, row, example)
            for role, value in zip(roles, token_losses, strict=True):
                losses[role].append(value)
            valued = [i for i, role in enumerate(roles) if role == "value"]
            if valued:
                first.append(token_losses[valued[0]])
                rest.extend(token_losses[i] for i in valued[1:])
                exact.append(float(all(hits[i] for i in valued)))
    report: dict[str, Any] = {
        role: {"tokens": len(values), "loss": statistics.fmean(values)}
        for role, values in losses.items()
    }
    report["value_first_loss"] = statistics.fmean(first) if first else None
    report["value_rest_loss"] = statistics.fmean(rest) if rest else None
    report["value_exact_teacher_forced"] = statistics.fmean(exact) if exact else None
    return report
