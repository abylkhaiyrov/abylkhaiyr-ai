from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from llm.data.preference import PREFERENCE_DIR, tokenize_preference_split
from llm.data.verified_common import Row, load_config, rng_for
from llm.tokenizer.bpe import BPETokenizer

Pair = dict[str, Any]
ARITHMETIC = re.compile(r"^(?P<head>.*?= )(?P<value>-?\d+)(?P<tail>\..*)$", re.DOTALL)
JOB_PATTERNS = {
    "ru": re.compile(r"работает (\w+)"),
    "en": re.compile(r"works as (an? \w+)"),
    "kk": re.compile(r"(\w+) болып жұмыс"),
}
HALLUCINATIONS = {
    "ru": ["{p} — {n} лет.", "{p} любит синий цвет.", "Да, у {p} большая семья."],
    "en": ["{p} is {n} years old.", "{p} likes the color blue.", "Yes, {p} has a big family."],
    "kk": ["{p} {n} жаста.", "{p} көк түсті жақсы көреді.", "Иә, {p} отбасы үлкен."],
}
REFUSAL = {
    "ru": "В тексте об этом не сказано.",
    "en": "The text does not say.",
    "kk": "Мәтінде бұл туралы айтылмаған.",
}
PERSON_AFTER = {
    "ru": re.compile(r"Вопрос: Кем работает (\w+)\?"),
    "en": re.compile(r"What is (\w+)'s job"),
    "kk": re.compile(r"Сұрақ: (\w+) кім"),
}
MIN_DELTA, MAX_DELTA = 1, 9
MISSING_KINDS = {"answerability_missing"}
SKIPPED_GROUPS = {"code"}


def pair(row: Row, negative: str, kind: str) -> Pair | None:
    positive = row["messages"][-1]["content"]
    if negative == positive or not negative.strip():
        return None
    context = row["messages"][:-1]
    return {
        "chosen": [*context, {"role": "assistant", "content": positive}],
        "rejected": [*context, {"role": "assistant", "content": negative}],
        "kind": kind,
        "group": row["group"],
        "category": row["category"],
        "language": row["language"],
        "split": row["split"],
    }


def wrong_number(answer: str, rng, *, all_occurrences: bool = False) -> str | None:
    match = ARITHMETIC.match(answer)
    if match is None:
        return None
    value = int(match["value"])
    wrong = value + rng.choice([-1, 1]) * rng.randint(MIN_DELTA, MAX_DELTA)
    tail = match["tail"]
    if all_occurrences:
        tail = re.sub(rf"(?<![\d-]){value}(?!\d)", str(wrong), tail)
    return f"{match['head']}{wrong}{tail}"


def names_in(text: str, pool: list[str]) -> list[str]:
    return [name for name in pool if re.search(rf"(?<!\w){re.escape(name)}(?!\w)", text)]


def other_name_negative(row: Row, rng) -> tuple[str, str] | None:
    language = row["language"]
    config = load_config("knowledge.json")
    first = row["messages"][0]["content"]
    answer = row["messages"][-1]["content"]
    pets = names_in(first, config["pet_names"][language])
    if row["category"] == "remember_pet" and len(pets) == 2:
        correct = names_in(answer, pets)
        other = [name for name in pets if name not in correct]
        if correct and other:
            return answer.replace(correct[0], other[0]), "distractor"
    people = config["people"][language]
    given = names_in(answer, people)
    if row["category"] == "remember_name" and given:
        replacement = rng.choice([name for name in people if name != given[0]])
        return answer.replace(given[0], replacement), "wrong_value"
    return None


def profession_negative(row: Row) -> tuple[str, str] | None:
    language = row["language"]
    text = row["messages"][0]["content"]
    jobs = JOB_PATTERNS[language].findall(text)
    answered = JOB_PATTERNS[language].findall(row["messages"][-1]["content"])
    if len(jobs) != 2 or len(answered) != 1 or jobs[0] == jobs[1]:
        return None
    other = jobs[1] if answered[0] == jobs[0] else jobs[0]
    return row["messages"][-1]["content"].replace(answered[0], other), "distractor"


def hours_negative(row: Row) -> tuple[str, str] | None:
    numbers = re.findall(r"\d+", row["messages"][0]["content"])
    answer = row["messages"][-1]["content"]
    given = re.findall(r"\d+", answer)
    if len(numbers) != 2 or len(given) != 1:
        return None
    other = numbers[1] if given[0] == numbers[0] else numbers[0]
    return answer.replace(given[0], other, 1), "other_relation"


def missing_negative(row: Row, rng) -> tuple[str, str]:
    language = row["language"]
    question = row["messages"][0]["content"]
    person = (names_in(question, load_config("knowledge.json")["people"][language]) or ["X"])[0]
    index = row["entity"].rsplit(":", 1)[-1]
    template = HALLUCINATIONS[language][int(index) % 3]
    return template.format(p=person, n=rng.randint(18, 70)), "hallucination"


def structured_negative(row: Row, rng) -> tuple[str, str] | None:
    category = row["category"]
    answer = row["messages"][-1]["content"]
    if category.startswith("arithmetic") or category == "carry_over":
        wrong = wrong_number(answer, rng)
    elif category == "basket":
        wrong = wrong_number(answer, rng, all_occurrences=True)
    elif category in {"remember_name", "remember_pet"}:
        return other_name_negative(row, rng)
    elif category == "profession":
        return profession_negative(row)
    elif category == "opening_hours":
        return hours_negative(row)
    elif category in MISSING_KINDS:
        return missing_negative(row, rng)
    else:
        return None
    return (wrong, "wrong_value") if wrong else None


def refusal_for_answerable(row: Row) -> tuple[str, str]:
    return REFUSAL[row["language"]], "unjustified_refusal"


def other_entity_negative(row: Row, pool: list[Row], rng) -> tuple[str, str] | None:
    candidates = [item for item in pool if item["entity"] != row["entity"]]
    if not candidates:
        return None
    return rng.choice(candidates)["messages"][-1]["content"], "other_entity"


def contrastive_pairs(rows: list[Row], seed: str = "contrast-v1") -> list[Pair]:
    pools: dict[tuple[str, str, str], list[Row]] = defaultdict(list)
    for row in rows:
        pools[(row["group"], row["category"], row["language"])].append(row)
    pairs: list[Pair] = []
    for index, row in enumerate(rows):
        if row["group"] in SKIPPED_GROUPS or len(row["messages"]) % 2:
            continue
        rng = rng_for(seed, index, row["entity"])
        negative = structured_negative(row, rng)
        answerable_context = row["category"] in {"profession", "opening_hours"}
        if answerable_context and (negative is None or index % 5 == 0):
            negative = refusal_for_answerable(row)
        if negative is None:
            pool = pools[(row["group"], row["category"], row["language"])]
            negative = other_entity_negative(row, pool, rng)
        if negative is None:
            continue
        built = pair(row, negative[0], negative[1])
        if built is not None:
            pairs.append(built)
    return pairs


def write_pairs(pairs: list[Pair], out_dir: Path, tokenizer: BPETokenizer) -> dict[str, Any]:
    root = out_dir / PREFERENCE_DIR
    root.mkdir(parents=True, exist_ok=True)
    counts: dict[str, Any] = {"splits": {}, "kinds": {}}
    for split in ("train", "validation", "test"):
        items = [item for item in pairs if item["split"] == split]
        if not items:
            continue
        with (root / f"{split}.jsonl").open("w", encoding="utf-8") as handle:
            for item in items:
                handle.write(json.dumps(item, ensure_ascii=False) + "\n")
        if split != "test":
            tokenize_preference_split(out_dir, split, tokenizer)
        counts["splits"][split] = len(items)
    for item in pairs:
        counts["kinds"][item["kind"]] = counts["kinds"].get(item["kind"], 0) + 1
    return counts
