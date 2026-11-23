from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

from llm.data.binary import read_jsonl, supervised_token_count, tokenize_sft_split, with_extension
from llm.data.dedup import MinHashDeduplicator, stable_hash
from llm.data.v4_dialogues import Dialogue
from llm.data.v4_quality import LANGUAGES
from llm.tokenizer.bpe import BPETokenizer

WORD = re.compile(r"[^\W_]+")
V4_VALIDATION_SIZE = 540
SEMANTIC = {"reading", "classification", "extraction", "summarization", "causal", "comparison"}


def normalized(text: str) -> str:
    return " ".join(WORD.findall(text.casefold()))


def legacy_prompts(paths: Iterable[Path]) -> set[int]:
    hashes: set[int] = set()
    for path in paths:
        for row in read_jsonl(path):
            users = [item["content"] for item in row["messages"] if item["role"] == "user"]
            hashes.update(stable_hash(normalized(text)) for text in users)
    return hashes


class DialogueDeduplicator:
    def __init__(self, existing: set[int] | None = None) -> None:
        self.prompts = set() if existing is None else set(existing)
        self.families: set[str] = set()
        self.user_roots: set[int] = set()
        self.protected: set[int] = set()
        self.near = MinHashDeduplicator(48, 0.8, 3, 1234)

    def block(self, prompt_hashes: set[int]) -> None:
        self.protected.update(prompt_hashes)

    def add(self, row: Dialogue, *, unique_family: bool = True) -> bool:
        history = normalized(row.prompt)
        key = stable_hash(history)
        if (
            key in self.prompts
            or any(
                stable_hash(normalized(turn.content)) in self.protected
                for turn in row.messages
                if turn.role == "user"
            )
            or (unique_family and row.family in self.families)
            or self.near.is_duplicate(history)
        ):
            return False
        self.prompts.add(key)
        self.user_roots.add(stable_hash(normalized(row.messages[0].content)))
        self.families.add(row.family)
        return True


@dataclass(frozen=True)
class Holdout:
    rows: list[Dialogue]
    index: DialogueDeduplicator


def validation_set(groups: Iterable[Dialogue], excluded: set[int]) -> Holdout:
    cells: dict[str, list[Dialogue]] = defaultdict(list)
    for row in groups:
        if any(
            stable_hash(normalized(turn.content)) in excluded
            for turn in row.messages
            if turn.role == "user"
        ):
            continue
        cells[row.language].append(row)
    index = DialogueDeduplicator()
    selected = []
    required = V4_VALIDATION_SIZE // len(LANGUAGES)
    for language in LANGUAGES:
        ordered = sorted(cells[language], key=lambda row: stable_hash(row.key))
        limits = {"translation": 65, "reading": 110}
        counts: Counter[str] = Counter()
        families: Counter[str] = Counter()
        chosen: list[Dialogue] = []
        for row in ordered:
            group = row.category if row.category in limits else "natural"
            if (
                counts[group] >= limits.get(group, 85)
                or families[row.family] >= 2
                or not index.add(row, unique_family=False)
            ):
                continue
            chosen.append(row)
            counts[group] += 1
            families[row.family] += 1
            if len(chosen) == required:
                break
        if len(chosen) != required:
            raise ValueError(f"{language}: only {len(chosen)} independent validation cases")
        selected.extend(chosen)
    return Holdout(sorted(selected, key=lambda row: stable_hash(row.key)), index)


def select_training(rows: Iterable[Dialogue], blocked: DialogueDeduplicator) -> Iterator[Dialogue]:
    counts: Counter[tuple[str, str]] = Counter()
    prefixes: Counter[tuple[str, str]] = Counter()
    for row in sorted(rows, key=lambda item: stable_hash(item.key)):
        cell = (row.language, row.category)
        prefix = (row.language, " ".join(normalized(row.messages[-2].content).split()[:5]))
        limit = 150 if row.category in {"math", "code"} else 900
        if (
            stable_hash(normalized(row.messages[0].content)) in blocked.user_roots
            or counts[cell] >= limit
            or prefixes[prefix] >= 18
            or not blocked.add(row)
        ):
            continue
        counts[cell] += 1
        prefixes[prefix] += 1
        yield row


def write_rows(path: Path, rows: Iterable[Dialogue]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row.as_row(), ensure_ascii=False) + "\n")
            counts[f"{row.language}.{row.category}"] += 1
    return dict(sorted(counts.items()))


def sft_group(row: Dialogue) -> str:
    if row.source == "curated":
        return f"curated_{row.language}"
    if row.category in {"math", "code"}:
        return row.category
    if row.category in SEMANTIC:
        return f"semantic_{row.language}"
    return f"dialogue_{row.language}"


def write_sft_groups(
    root: Path, rows: Iterable[Dialogue], tokenizer: BPETokenizer
) -> dict[str, dict[str, int]]:
    grouped: dict[str, list[Dialogue]] = defaultdict(list)
    for row in rows:
        grouped[sft_group(row)].append(row)
    directory = root / "sft_groups"
    directory.mkdir()
    report: dict[str, dict[str, int]] = {"train": {}, "validation": {}}
    for name, dialogues in grouped.items():
        if len(dialogues) < 2:
            raise ValueError(f"need at least two dialogs for SFT group {name}")
        ordered = sorted(dialogues, key=lambda row: stable_hash(row.family))
        boundary = max(1, math.floor(len(ordered) * 0.04))
        for split, cases in (("train", ordered[boundary:]), ("validation", ordered[:boundary])):
            prefix = directory / f"{split}.{name}"
            source = with_extension(prefix, ".jsonl")
            write_rows(source, cases)
            tokenize_sft_split(source, prefix, tokenizer)
            report[split][name] = supervised_token_count(prefix)
    return report
