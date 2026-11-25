from __future__ import annotations

import hashlib
import json
import random
from functools import cache
from pathlib import Path
from typing import Any

LANGUAGES = ("ru", "en", "kk")
Row = dict[str, Any]
TRAIN_BELOW = 84
VALIDATION_BELOW = 92
CONFIG_DIR = Path("configs/verified")


def bucket(key: str) -> int:
    return int(hashlib.sha256(key.encode("utf-8")).hexdigest(), 16) % 100


def split_for(key: str) -> str:
    value = bucket(key)
    if value < TRAIN_BELOW:
        return "train"
    return "validation" if value < VALIDATION_BELOW else "test"


def rng_for(*parts: object) -> random.Random:
    return random.Random("|".join(str(part) for part in parts))


@cache
def load_config(name: str) -> dict[str, Any]:
    return json.loads((CONFIG_DIR / name).read_text(encoding="utf-8"))


def make_row(
    group: str,
    category: str,
    language: str,
    entity: str,
    messages: list[tuple[str, str]],
    source: str,
    answer_language: str | None = None,
) -> Row:
    spec = load_config("knowledge.json")
    return {
        "source": source,
        "group": group,
        "category": category,
        "language": language,
        "answer_language": answer_language or language,
        "entity": entity,
        "split": split_for(entity),
        "provenance": spec["provenance"],
        "license": spec["license"],
        "messages": [{"role": role, "content": content} for role, content in messages],
    }


def exchange(
    group: str,
    category: str,
    language: str,
    entity: str,
    user: str,
    answer: str,
    source: str,
    answer_language: str | None = None,
) -> Row:
    return make_row(
        group,
        category,
        language,
        entity,
        [("user", user), ("assistant", answer)],
        source,
        answer_language,
    )


def capitalize(text: str) -> str:
    return text[:1].upper() + text[1:]
