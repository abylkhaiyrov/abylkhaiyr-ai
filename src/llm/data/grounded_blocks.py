from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from llm.data.binary import IGNORE_INDEX, tokenize_sft_split, with_extension
from llm.data.grounded import (
    MISSING_SHARE,
    Case,
    Example,
    eval_cases,
    row_of,
    trained_cases,
    training_examples,
)
from llm.data.grounded_text import LANGUAGES
from llm.tokenizer.bpe import BPETokenizer, Message

POOL_SIZE = 7_000
HELD_SPLITS = {"validation": "dev", "test": "test"}
HELD_SIZE = 200
TOLERANCE = 0.02
REPAIR_PASSES = 20
REPAIR_TOLERANCE = 0.003
READING = "reading_"


@dataclass(frozen=True)
class Target:
    tokens: int
    supervised: int


def group_target(prefix: Path) -> Target:
    lengths = np.load(with_extension(prefix, ".lengths.npy"))
    labels = np.memmap(with_extension(prefix, ".labels.bin"), dtype=np.int32, mode="r")
    return Target(int(lengths.sum()), int((labels != IGNORE_INDEX).sum()))


def measure(row: dict, tokenizer: BPETokenizer) -> tuple[int, int]:
    messages = [Message(turn["role"], turn["content"], True) for turn in row["messages"]]
    ids, labels = tokenizer.encode_conversation_with_labels(messages, IGNORE_INDEX)
    return len(ids), sum(label != IGNORE_INDEX for label in labels)


@dataclass
class Choice:
    options: list[dict]
    sizes: list[tuple[int, int]]
    pick: int


def totals(choices: list[Choice]) -> tuple[int, int]:
    return (
        sum(c.sizes[c.pick][0] for c in choices),
        sum(c.sizes[c.pick][1] for c in choices),
    )


def repair(choices: list[Choice], target: Target, rng: random.Random) -> None:
    order = list(range(len(choices)))
    for _ in range(REPAIR_PASSES):
        tokens, supervised = totals(choices)
        if abs(supervised - target.supervised) <= target.supervised * REPAIR_TOLERANCE:
            return
        rng.shuffle(order)
        for index in order:
            choice = choices[index]
            current = choice.sizes[choice.pick]
            gap = abs(supervised - target.supervised)
            best = min(
                range(len(choice.sizes)),
                key=lambda k: abs(supervised - current[1] + choice.sizes[k][1] - target.supervised),
            )
            replacement = choice.sizes[best]
            if abs(supervised - current[1] + replacement[1] - target.supervised) < gap:
                choice.pick = best
                tokens += replacement[0] - current[0]
                supervised += replacement[1] - current[1]
            if abs(supervised - target.supervised) <= target.supervised * REPAIR_TOLERANCE:
                return


def fit_rows(
    examples: list[Example], target: Target, tokenizer: BPETokenizer, seed: int
) -> list[dict]:
    rng = random.Random(seed)
    ratio = target.supervised / target.tokens
    choices: list[Choice] = []
    tokens = supervised = 0
    for example in examples:
        if tokens >= target.tokens:
            break
        options = [row_of(example, style) for style in range(4)]
        sizes = [measure(option, tokenizer) for option in options]
        deviations = [abs((supervised + s) / (tokens + t) - ratio) for t, s in sizes]
        allowed = [i for i, d in enumerate(deviations) if d <= min(deviations) + TOLERANCE]
        pick = rng.choice(allowed)
        choices.append(Choice(options, sizes, pick))
        tokens += sizes[pick][0]
        supervised += sizes[pick][1]
    repair(choices, target, rng)
    return [choice.options[choice.pick] for choice in choices]


def write_rows(
    directory: Path, split: str, group: str, rows: list[dict], tokenizer: BPETokenizer
) -> None:
    path = directory / f"{split}.{group}.jsonl"
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
    )
    tokenize_sft_split(path, directory / f"{split}.{group}", tokenizer)


def write_reading_block(
    output: Path,
    base: Path,
    tokenizer: BPETokenizer,
    seed: int,
    missing_share: float = MISSING_SHARE,
) -> dict[str, dict[str, int]]:
    groups = output / "sft_groups"
    groups.mkdir(parents=True, exist_ok=True)
    report: dict[str, dict[str, int]] = {}
    for language in LANGUAGES:
        group = f"{READING}{language}"
        target = group_target(base / "sft_groups" / f"train.{group}")
        examples = training_examples(language, POOL_SIZE, seed, missing_share=missing_share)
        rows = fit_rows(examples, target, tokenizer, seed)
        random.Random(seed).shuffle(rows)
        write_rows(groups, "train", group, rows, tokenizer)
        for split, source in HELD_SPLITS.items():
            held = [
                row_of(e, i)
                for i, e in enumerate(training_examples(language, HELD_SIZE, seed, source))
            ]
            write_rows(groups, split, group, held, tokenizer)
        achieved = group_target(groups / f"train.{group}")
        report[group] = {
            "examples": len(rows),
            "tokens": achieved.tokens,
            "supervised": achieved.supervised,
            "target_tokens": target.tokens,
            "target_supervised": target.supervised,
        }
    for entry in sorted((base / "sft_groups").iterdir()):
        link = groups / entry.name
        if not entry.name.split(".")[1].startswith(READING) and not link.exists():
            link.symlink_to(entry.resolve())
    return report


def write_cases(path: Path, cases: list[Case]) -> list[Case]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(asdict(case), ensure_ascii=False) + "\n" for case in cases),
        encoding="utf-8",
    )
    return cases


def write_grounded(output: Path, base: Path, tokenizer: BPETokenizer, seed: int) -> dict:
    report = write_reading_block(output / "reading_b", base, tokenizer, seed)
    sizes = {
        split: len(write_cases(output / f"{split}.jsonl", eval_cases(split)))
        for split in ("dev", "test")
    }
    sizes["trained"] = len(write_cases(output / "trained.jsonl", trained_cases(seed)))
    summary = {"reading_b": report, "cases": sizes, "seed": seed}
    (output / "report.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return summary
