from __future__ import annotations

import re
from collections import Counter
from collections.abc import Callable, Iterable, Mapping
from typing import Any

ENTITY_TEMPLATE = re.compile(
    r"не хочу выдумывать|would rather not invent|ойдан шығарғым келмейді", re.IGNORECASE
)
NOT_IN_TEXT = re.compile(
    r"Мәтінде бұл туралы айтылмаған|В тексте (?:об этом )?не говорится|does not (?:say|mention)",
    re.IGNORECASE,
)
NO_ACCESS_CATEGORIES = frozenset({"unknown_knowledge", "refusal"})
PROPER_NAME = re.compile(r"«?[A-ZА-ЯӘҒҚҢӨҰҮҺІ][a-zа-яәғқңөұүһі]{3,12}»?")

NO_ACCESS_PHRASES = re.compile(
    r"нет доступа|не имею доступа|do not have access|don't have access|cannot see|can't see|"
    r"қолжетімділік жоқ|қолжетімділігім жоқ|не вижу|мен сіздің .* көрмеймін",
    re.IGNORECASE,
)
ENTITY = "entity_template"
NO_ACCESS = "no_access"
GROUNDED = "not_in_text"
KINDS = (ENTITY, NO_ACCESS, GROUNDED)

Detector = Callable[[Mapping[str, Any], str], bool]
DETECTORS: dict[str, Detector] = {
    ENTITY: lambda row, answer: bool(ENTITY_TEMPLATE.search(answer)),
    NO_ACCESS: lambda row, answer: row.get("category") in NO_ACCESS_CATEGORIES,
    GROUNDED: lambda row, answer: bool(NOT_IN_TEXT.search(answer)),
}


def assistant_answers(row: Mapping[str, Any]) -> list[str]:
    return [turn["content"] for turn in row["messages"] if turn["role"] == "assistant"]


def refusal_kinds(row: Mapping[str, Any], answer: str) -> list[str]:
    return [kind for kind in KINDS if DETECTORS[kind](row, answer)]


def template_of(answer: str) -> str:
    return PROPER_NAME.sub("X", answer)


def phrasing_counts(rows: Iterable[Mapping[str, Any]], kind: str) -> Counter[str]:
    return Counter(
        template_of(answer)
        for row in rows
        for answer in assistant_answers(row)
        if DETECTORS[kind](row, answer)
    )


def is_refusal(text: str) -> bool:
    return any(
        pattern.search(text) for pattern in (ENTITY_TEMPLATE, NOT_IN_TEXT, NO_ACCESS_PHRASES)
    )
