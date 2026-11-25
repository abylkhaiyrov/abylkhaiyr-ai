from __future__ import annotations

import ast
import hashlib
import json
import re
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from llm.evaluation.conversation import numbers_in
from llm.reasoning.sandbox import SandboxLimits, run_python
from llm.reasoning.tools import extract_code

LANGUAGES = ("ru", "en", "kk")
DIMENSIONS = (
    "relevance",
    "correctness",
    "completeness",
    "coherence",
    "instruction",
    "refusal",
    "history",
)
INSUFFICIENT_MARKERS = {
    "ru": (
        "не указан",
        "не говорится",
        "не сказано",
        "нет информации",
        "нет данных",
        "недостаточно",
        "не хватает",
        "неизвестно",
        "не знаю",
        "не могу",
        "не хватит",
        "невозможно",
        "нет, ",
        "не достаточно",
        "не известно",
    ),
    "en": (
        "not mentioned",
        "not stated",
        "not specified",
        "does not say",
        "doesn't say",
        "no information",
        "not enough",
        "insufficient",
        "don't know",
        "do not know",
        "cannot",
        "can't",
        "unknown",
        "no, ",
        "not provided",
        "isn't enough",
    ),
    "kk": (
        "көрсетілмеген",
        "айтылмаған",
        "жеткіліксіз",
        "мәлімет жоқ",
        "ақпарат жоқ",
        "білмеймін",
        "белгісіз",
        "жеткілікті емес",
        "болмайды",
        "жоқ",
        "мүмкін емес",
        "табу мүмкін емес",
    ),
}
CODE_HARNESS = (
    "import json\n"
    "__results = []\n"
    "__calls = json.loads({calls!r})\n"
    "for __call in __calls:\n"
    "    try:\n"
    "        __results.append(__NAME__(*__call['args']))\n"
    "    except Exception as __error:\n"
    "        __results.append('error:' + type(__error).__name__)\n"
    "print(json.dumps(__results))\n"
)


@dataclass(frozen=True)
class AcceptanceItem:
    item_id: str
    scenario: str
    category: str
    language: str
    turns: tuple[str, ...]
    check: dict[str, Any]
    key_points: tuple[str, ...]


@dataclass(frozen=True)
class AutoResult:
    kind: str
    passed: bool | None
    detail: str = ""


@dataclass
class Grade:
    item_id: str
    scores: dict[str, int | None]
    cause: str = ""
    note: str = ""
    passed: bool = False


def load_benchmark(path: Path) -> tuple[list[AcceptanceItem], str]:
    raw = path.read_bytes()
    spec = json.loads(raw)
    items = [
        AcceptanceItem(
            item_id=f"{scenario['id']}/{language}",
            scenario=scenario["id"],
            category=scenario["category"],
            language=language,
            turns=tuple(scenario["turns"][language]),
            check=scenario["check"],
            key_points=tuple(scenario["key_points"]),
        )
        for scenario in spec["scenarios"]
        for language in LANGUAGES
    ]
    return items, hashlib.sha256(raw).hexdigest()


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower().replace("ё", "е")).strip()


def for_language(value: Any, language: str) -> Any:
    return value[language] if isinstance(value, dict) else value


def lexical_check(item: AcceptanceItem, answer: str) -> AutoResult:
    text = normalize(answer)
    groups = for_language(item.check["all_of"], item.language)
    missing = [group for group in groups if not any(normalize(alt) in text for alt in group)]
    forbidden = [
        alt
        for group in for_language(item.check.get("none_of", {}), item.language) or []
        for alt in group
        if normalize(alt) in text
    ]
    passed = not missing and not forbidden
    return AutoResult("lexical", passed, f"missing={missing} forbidden={forbidden}")


def number_check(item: AcceptanceItem, answer: str) -> AutoResult:
    value = str(for_language(item.check["value"], item.language))
    words = for_language(item.check.get("words", {}), item.language) or []
    text = normalize(answer)
    found = value in numbers_in(answer) or any(normalize(word) in text for word in words)
    return AutoResult("number", found, f"expected={value}")


def count_check(item: AcceptanceItem, answer: str) -> AutoResult:
    term = normalize(for_language(item.check["term"], item.language))
    found = len(re.findall(re.escape(term), normalize(answer)))
    return AutoResult("count", found == item.check["equals"], f"found={found}")


def refusal_check(item: AcceptanceItem, answer: str) -> AutoResult:
    text = normalize(answer)
    markers = [m for m in INSUFFICIENT_MARKERS[item.language] if normalize(m) in text]
    invented = bool(item.check.get("forbid_number")) and bool(re.search(r"\d", answer))
    return AutoResult("refusal", bool(markers) and not invented, f"markers={markers[:3]}")


def first_function_name(code: str) -> str | None:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None
    names = [node.name for node in tree.body if isinstance(node, ast.FunctionDef)]
    return names[0] if names else None


def code_check(item: AcceptanceItem, answer: str) -> AutoResult:
    code = extract_code(answer)
    name = first_function_name(code)
    if name is None:
        return AutoResult("code", False, "no parseable function")
    calls = json.dumps(item.check["calls"])
    harness = CODE_HARNESS.format(calls=calls).replace("__NAME__", name)
    result = run_python(f"{code}\n\n{harness}", SandboxLimits(timeout_seconds=5.0))
    if not result.passed:
        return AutoResult("code", False, (result.stderr or "failed")[-160:])
    try:
        outputs = json.loads(result.stdout.strip().splitlines()[-1])
    except (json.JSONDecodeError, IndexError):
        return AutoResult("code", False, "unreadable output")
    expected = [call["expect"] for call in item.check["calls"]]
    return AutoResult("code", outputs == expected, f"outputs={outputs}")


CHECKS = {
    "lexical": lexical_check,
    "number": number_check,
    "count": count_check,
    "refusal": refusal_check,
    "code": code_check,
}


def auto_check(item: AcceptanceItem, answer: str) -> AutoResult:
    kind = item.check["kind"]
    if kind == "manual":
        return AutoResult("manual", None)
    return CHECKS[kind](item, answer)


def rate(values: Iterable[bool]) -> float | None:
    items = list(values)
    return sum(items) / len(items) if items else None


def aggregate(rows: list[dict[str, Any]], grades: dict[str, Grade]) -> dict[str, Any]:
    summary: dict[str, Any] = {"n": len(rows)}
    graded = [row for row in rows if row["item_id"] in grades]
    summary["graded"] = len(graded)
    summary["manual_pass_rate"] = rate(grades[row["item_id"]].passed for row in graded)
    automatic = [row for row in rows if row["auto"]["passed"] is not None]
    summary["auto_pass_rate"] = rate(row["auto"]["passed"] for row in automatic)
    summary["auto_n"] = len(automatic)
    for key in ("language", "category"):
        buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in graded:
            buckets[row[key]].append(row)
        summary[f"manual_pass_by_{key}"] = {
            name: rate(grades[row["item_id"]].passed for row in bucket)
            for name, bucket in sorted(buckets.items())
        }
    for dimension in DIMENSIONS:
        values = [
            grades[row["item_id"]].scores[dimension]
            for row in graded
            if grades[row["item_id"]].scores.get(dimension) is not None
        ]
        summary[f"mean_{dimension}"] = (
            sum(values) / len(values) if values else None  # type: ignore[arg-type]
        )
    causes: dict[str, int] = defaultdict(int)
    for row in graded:
        cause = grades[row["item_id"]].cause
        if cause:
            causes[cause] += 1
    summary["causes"] = dict(sorted(causes.items(), key=lambda pair: -pair[1]))
    return summary
