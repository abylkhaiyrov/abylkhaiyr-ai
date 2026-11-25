from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from llm.core.config import ExperimentConfig
from llm.data.binary import IGNORE_INDEX, PackedConversationLoader, read_jsonl, with_extension
from llm.data.grounded import prompt_of, training_examples
from llm.data.grounded_text import LANGUAGES
from llm.data.sft_profile import supervised_per_example
from llm.inference.engine import InferenceEngine
from llm.tokenizer.bpe import BPETokenizer, Message
from llm.training.runner import _build_loaders

READING = "reading_"
ANSWERABLE, UNANSWERABLE, SURFACE = "answerable", "unanswerable", "surface"
CHECKPOINTS = (10, 20, 40, 60, 80, 120)
LEGACY_POOL = 7_000


@dataclass(frozen=True)
class Entry:
    kind: str
    task: str
    family: str
    supervised: int


def classify(category: str) -> str:
    if category == "grounded_missing":
        return UNANSWERABLE
    return ANSWERABLE if category.startswith("grounded_") else SURFACE


def legacy_families(language: str, seed: int) -> dict[str, str]:
    return {
        prompt_of(example.plan, example.structure): example.family
        for example in training_examples(language, LEGACY_POOL, seed)
    }


def group_catalog(prefix: Path, language: str, seed: int) -> list[Entry]:
    rows = list(read_jsonl(with_extension(prefix, ".jsonl")))
    supervised = supervised_per_example(prefix)
    legacy: dict[str, str] | None = None
    entries = []
    for row, tokens in zip(rows, supervised, strict=True):
        family = row.get("family", "")
        if not family and classify(row["category"]) != SURFACE:
            legacy = legacy if legacy is not None else legacy_families(language, seed)
            family = legacy[row["messages"][0]["content"]]
        entries.append(Entry(classify(row["category"]), row["category"], family, int(tokens)))
    return entries


def reading_catalogs(root: Path, seed: int) -> dict[str, list[Entry]]:
    return {
        f"{READING}{language}": group_catalog(
            root / "sft_groups" / f"train.{READING}{language}", language, seed
        )
        for language in LANGUAGES
    }


class Recorder:
    def __init__(self, loader: PackedConversationLoader) -> None:
        self.taken: list[int] = []
        self.original = loader._take
        loader.__dict__["_take"] = self.take

    def take(self) -> int:
        index = self.original()
        self.taken.append(index)
        return index


def chosen_group(source: Any) -> str:
    rng = np.random.default_rng((source.seed, source.counter))
    return str(source.names[int(rng.choice(len(source.names), p=source.probabilities))])


def audit(
    config: ExperimentConfig, tokenizer: BPETokenizer, catalogs: dict[str, list[Entry]]
) -> dict[str, Any]:
    source, _ = _build_loaders(config, tokenizer)
    recorders = {
        name: Recorder(loader)
        for name, loader in source.loaders.items()
        if isinstance(loader, PackedConversationLoader)
    }
    steps: list[dict[str, Any]] = []
    totals: dict[str, Counter[str]] = defaultdict(Counter)
    seen: dict[str, Counter[int]] = defaultdict(Counter)
    for step in range(1, config.train.max_steps + 1):
        record: dict[str, Any] = {"step": step, "batches": Counter(), "supervised": Counter()}
        for _ in range(config.train.gradient_accumulation_steps):
            group = chosen_group(source)
            recorder = recorders.get(group)
            before = len(recorder.taken) if recorder else 0
            batch = source.next_batch()
            record["batches"][group] += 1
            if len(batch) == 3:
                record["supervised"][group] += int((batch[1] != IGNORE_INDEX).sum())
            else:
                record["supervised"][f"replay:{group}"] += int((batch[1] != IGNORE_INDEX).sum())
            if recorder and group in catalogs:
                for index in recorder.taken[before:]:
                    entry = catalogs[group][index]
                    seen[group][index] += 1
                    record.setdefault("examples", Counter())[entry.kind] += 1
                    record.setdefault("example_tokens", Counter())[entry.kind] += entry.supervised
                    totals[entry.kind]["examples"] += 1
                    totals[entry.kind]["supervised"] += entry.supervised
        steps.append(
            {
                key: dict(value) if isinstance(value, Counter) else value
                for key, value in record.items()
            }
        )
    return {"steps": steps, "summary": summarize(steps, catalogs, seen, totals)}


def cumulative(steps: list[dict[str, Any]], key: str, kind: str, upto: int) -> int:
    return sum(item.get(key, {}).get(kind, 0) for item in steps[:upto])


def summarize(
    steps: list[dict[str, Any]],
    catalogs: dict[str, list[Entry]],
    seen: dict[str, Counter[int]],
    totals: dict[str, Counter[str]],
) -> dict[str, Any]:
    batches: Counter[str] = Counter()
    supervised: Counter[str] = Counter()
    for item in steps:
        batches.update(item["batches"])
        supervised.update(item["supervised"])
    all_batches = sum(batches.values())
    sft_supervised = sum(v for k, v in supervised.items() if not k.startswith("replay:"))
    groups = {
        name: {
            "batches": batches[name],
            "batch_share": batches[name] / all_batches,
            "supervised": supervised[name],
            "supervised_share": supervised[name] / sft_supervised if sft_supervised else 0.0,
        }
        for name in batches
    }
    reading = [name for name in catalogs]
    families = {
        group: {catalogs[group][index].family for index in counts} for group, counts in seen.items()
    }
    distinct = sum(len(counts) for counts in seen.values())
    taken = sum(sum(counts.values()) for counts in seen.values())
    tasks: Counter[str] = Counter()
    for group, counts in seen.items():
        for index, times in counts.items():
            tasks[catalogs[group][index].task] += times
    return {
        "steps": len(steps),
        "sft_supervised_tokens": sft_supervised,
        "replay_tokens": sum(v for k, v in supervised.items() if k.startswith("replay:")),
        "groups": groups,
        "reading_batch_share": sum(batches[name] for name in reading) / all_batches,
        "reading_supervised_share": sum(supervised[name] for name in reading) / sft_supervised,
        "reading_examples": {kind: dict(value) for kind, value in totals.items()},
        "reading_tasks": dict(tasks),
        "distinct_reading_examples": distinct,
        "reading_examples_taken": taken,
        "repeat_exposure": taken / distinct if distinct else 0.0,
        "families": sum(len({f for f in members if f}) for members in families.values()),
        "cumulative": {
            str(upto): {
                kind: {
                    "examples": cumulative(steps, "examples", kind, upto),
                    "supervised": cumulative(steps, "example_tokens", kind, upto),
                }
                for kind in totals
            }
            for upto in CHECKPOINTS
            if upto <= len(steps)
        },
    }


def dump(report: dict[str, Any], path: Path) -> None:
    path.write_text(json.dumps(report, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")


def attach_metrics(report: dict[str, Any], path: Path) -> None:
    rows = map(json.loads, path.read_text(encoding="utf-8").splitlines())
    logged = {row["step"]: row for row in rows if "train/loss" in row}
    for item in report["steps"]:
        row = logged.get(item["step"])
        if row:
            item["loss"] = row["train/loss"]
            item["grad_norm"] = row["train/grad_norm"]


def row_loss(engine: InferenceEngine, row: dict[str, Any]) -> float:
    messages = [Message(turn["role"], turn["content"], True) for turn in row["messages"]]
    ids, labels = engine.tokenizer.encode_conversation_with_labels(messages, IGNORE_INDEX)
    with torch.no_grad():
        logits = engine.model(torch.tensor([ids[:-1]], device=engine.device)).float()
    targets = torch.tensor(labels[1:], device=engine.device)
    mask = targets != IGNORE_INDEX
    losses = torch.nn.functional.cross_entropy(logits[0][mask], targets[mask], reduction="none")
    return float(losses.mean())


def task_losses(engine: InferenceEngine, paths: list[Path], per_category: int) -> dict[str, float]:
    engine.model.eval()
    by_category: dict[str, list[float]] = defaultdict(list)
    for path in paths:
        counts: Counter[str] = Counter()
        for row in read_jsonl(path):
            if counts[row["category"]] < per_category:
                counts[row["category"]] += 1
                by_category[row["category"]].append(row_loss(engine, row))
    return {name: float(np.mean(values)) for name, values in sorted(by_category.items())}
