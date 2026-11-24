from __future__ import annotations

import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from llm.data.binary import IGNORE_INDEX, read_jsonl, with_extension
from llm.data.refusals import assistant_answers, refusal_kinds, template_of

KAZAKH_LETTERS = frozenset("әғқңөұүһі")
INTENT_WORDS = 3
CONTEXT_MARKER = "\n\n"
VERIFIED_SOURCES = ("synthetic_", "gsm8k_sft")


def script_of(text: str) -> str:
    letters = [char.casefold() for char in text if char.isalpha()]
    if not letters:
        return "none"
    if any(char in KAZAKH_LETTERS for char in letters):
        return "kk"
    cyrillic = sum("Ѐ" <= char <= "ӿ" for char in letters) / len(letters)
    return "ru" if cyrillic > 0.5 else "en"


def entropy(counts: Counter[str]) -> float:
    total = sum(counts.values())
    return -sum(c / total * math.log2(c / total) for c in counts.values()) if total else 0.0


def intent_of(row: dict) -> str:
    if row.get("category"):
        return str(row["category"])
    words = row["messages"][0]["content"].casefold().split()[:INTENT_WORDS]
    return " ".join(words)


def supervised_per_example(prefix: Path) -> np.ndarray:
    labels = np.memmap(with_extension(prefix, ".labels.bin"), dtype=np.int32, mode="r")
    lengths = np.load(with_extension(prefix, ".lengths.npy"))
    starts = np.concatenate(([0], np.cumsum(lengths)[:-1]))
    return np.add.reduceat((labels != IGNORE_INDEX).astype(np.int64), starts)


def has_context(row: dict) -> bool:
    return CONTEXT_MARKER in row["messages"][0]["content"] and row.get("group", "").startswith(
        "reading"
    )


def group_profile(path: Path) -> dict[str, Any]:
    rows = list(read_jsonl(path))
    prefix = path.with_suffix("")
    tokens = supervised_per_example(prefix)
    answers = [answer for row in rows for answer in assistant_answers(row)]
    templates = Counter(template_of(answer) for answer in answers)
    prompts: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        prompts[row["messages"][0]["content"]].add(row["messages"][-1]["content"])
    intents = Counter(intent_of(row) for row in rows)
    languages = Counter(
        row.get("language") or script_of(row["messages"][0]["content"]) for row in rows
    )
    verified = [str(row.get("source", "")).startswith(VERIFIED_SOURCES) for row in rows]
    refusal = [any(refusal_kinds(row, answer) for answer in assistant_answers(row)) for row in rows]
    return {
        "examples": len(rows),
        "supervised_tokens": int(tokens.sum()),
        "unique_prompts": len(prompts),
        "unique_answers": len(set(answers)),
        "answer_templates": len(templates),
        "top_template_share": templates.most_common(1)[0][1] / len(answers) if answers else 0.0,
        "distinct_intents": len(intents),
        "intent_entropy_bits": entropy(intents),
        "languages": dict(languages),
        "multi_turn_share": sum(len(row["messages"]) > 2 for row in rows) / len(rows),
        "context_share": sum(has_context(row) for row in rows) / len(rows),
        "verified_synthetic_share": sum(verified) / len(rows),
        "verified_synthetic_tokens": int(tokens[np.asarray(verified)].sum()),
        "refusal_share": sum(refusal) / len(rows),
        "refusal_tokens": int(tokens[np.asarray(refusal)].sum()),
        "conflicting_prompts": sum(len(values) > 1 for values in prompts.values()),
    }


def profile_sft(directory: Path) -> dict[str, Any]:
    groups = {
        path.name.split(".")[1]: group_profile(path)
        for path in sorted(directory.glob("train.*.jsonl"))
    }
    total = sum(group["supervised_tokens"] for group in groups.values())
    for group in groups.values():
        group["token_share"] = group["supervised_tokens"] / total
    return {"groups": groups, "supervised_tokens": total}
