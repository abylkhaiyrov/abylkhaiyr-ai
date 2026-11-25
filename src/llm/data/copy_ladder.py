from __future__ import annotations

import json
import random
import re
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

from llm.tokenizer.bpe import BPETokenizer

LANGUAGES = ("en", "ru", "kk")
LEVELS = (1, 2, 3, 4)
CATEGORIES = ("familiar", "single", "double", "long", "number", "invented")
ITEMS_PER_CELL = 8
LONG_TOKENS = 4
MAX_ATTEMPTS = 200_000
SYLLABLE_RANGE = {"double": (1, 2), "long": (3, 6)}
WIDTH = {1: 1, 2: 1, 3: 2, 4: 4}
ENTITIES = ("A", "B")
RELATIONS = ("river", "mountain")
ANSWER_LABEL = {"en": "Answer:", "ru": "Ответ:", "kk": "Жауап:"}
CAPITALIZED = {
    "en": re.compile(r"[A-Z][a-z]{3,}"),
    "ru": re.compile(r"[А-ЯЁ][а-яё]{3,}"),
    "kk": re.compile(r"[А-ЯЁӘҒҚҢӨҰҮҺІ][а-яёәғқңөұүһі]{3,}"),
}
SYLLABLES = {
    "en": ("bdgklmnprstvz", "aeiou"),
    "ru": ("бвгдзклмнпрстф", "аеиоуя"),
    "kk": ("бгдзклмнпрстқғң", "аеиоуәөұүі"),
}
SUFFIXES = ("a", "en", "is", "ar", "on", "ith", "ova")
FAMILIAR = {
    "en": (
        "table",
        "window",
        "garden",
        "teacher",
        "bread",
        "winter",
        "market",
        "bridge",
        "forest",
        "school",
        "river",
        "mountain",
        "doctor",
        "street",
        "summer",
        "kitchen",
        "letter",
        "horse",
        "story",
        "village",
    ),
    "ru": (
        "стол",
        "окно",
        "сад",
        "учитель",
        "хлеб",
        "зима",
        "рынок",
        "мост",
        "лес",
        "школа",
        "река",
        "гора",
        "врач",
        "улица",
        "лето",
        "кухня",
        "письмо",
        "лошадь",
        "рассказ",
        "село",
    ),
    "kk": (
        "үстел",
        "терезе",
        "бақ",
        "мұғалім",
        "нан",
        "қыс",
        "базар",
        "көпір",
        "орман",
        "мектеп",
        "өзен",
        "тау",
        "дәрігер",
        "көше",
        "жаз",
        "ас үй",
        "хат",
        "жылқы",
        "әңгіме",
        "ауыл",
    ),
}
LEVEL_ONE = {"en": "Repeat exactly: {X}", "ru": "Повтори точно: {X}", "kk": "Дәл қайтала: {X}"}
LEVEL_TWO = {
    "en": "Text: The river is called {X}. Name the river.",
    "ru": "Текст: Река называется {X}. Назови реку.",
    "kk": "Мәтін: Өзеннің аты {X}. Өзеннің атын ата.",
}
LEVEL_THREE = {
    "en": "City A has the river {X1}. City B has the river {X2}. Name the river of city {E}.",
    "ru": "У города A река {X1}. У города B река {X2}. Назови реку города {E}.",
    "kk": "A қаласында өзен {X1}. B қаласында өзен {X2}. {E} қаласының өзенін ата.",
}
LEVEL_FOUR = {
    "en": "City A has the river {R1} and the mountain {M1}. City B has the river {R2} and the "
    "mountain {M2}. Name the {rel} of city {E}.",
    "ru": "У города A река {R1} и гора {M1}. У города B река {R2} и гора {M2}. Назови {rel} города {E}.",
    "kk": "A қаласында өзен {R1} және тау {M1}. B қаласында өзен {R2} және тау {M2}. "
    "{E} қаласының {rel} ата.",
}
ASKED = {
    "en": {"river": "river", "mountain": "mountain"},
    "ru": {"river": "реку", "mountain": "гору"},
    "kk": {"river": "өзенін", "mountain": "тауын"},
}


FRAMES = {
    2: {
        "en": "The river is called",
        "ru": "Река называется",
        "kk": "Өзеннің аты",
    },
    3: {
        "en": "The river of city {E} is",
        "ru": "Река города {E} —",
        "kk": "{E} қаласының өзені —",
    },
    4: {
        "en": "The {rel} of city {E} is",
        "ru": "{rel} города {E} —",
        "kk": "{E} қаласының {rel} —",
    },
}
FRAME_RELATIONS = {
    "en": {"river": "river", "mountain": "mountain"},
    "ru": {"river": "Река", "mountain": "Гора"},
    "kk": {"river": "өзені", "mountain": "тауы"},
}


@dataclass(frozen=True)
class CopyCase:
    id: str
    family: str
    language: str
    level: int
    category: str
    prompt: str
    target: str
    rivals: tuple[str, ...]
    entity: str
    relation: str


def doc_tokens(tokenizer: BPETokenizer, value: str) -> int:
    return len(tokenizer.encode(f" {value}"))


def vocabulary_names(tokenizer: BPETokenizer, language: str) -> list[str]:
    vocabulary = tokenizer.backend.get_vocab()
    found = []
    for token, index in vocabulary.items():
        if token.startswith("Ġ"):
            word = tokenizer.decode([index]).strip()
            if CAPITALIZED[language].fullmatch(word) and doc_tokens(tokenizer, word) == 1:
                found.append(word)
    return sorted(found)


def invented(rng: random.Random, language: str, syllables: int) -> str:
    consonants, vowels = SYLLABLES[language]
    parts = [rng.choice(consonants) + rng.choice(vowels) for _ in range(syllables)]
    return (
        "".join(parts) + rng.choice(SUFFIXES if language == "en" else ("а", "ен", "ис", "ар"))
    ).capitalize()


def latin_name(rng: random.Random) -> str:
    return invented(rng, "en", 3)


def draw_pool(
    tokenizer: BPETokenizer, language: str, category: str, count: int, seed: int
) -> list[str]:
    rng = random.Random(f"{seed}:{language}:{category}")
    if category in ("familiar", "single"):
        source = (
            list(FAMILIAR[language])
            if category == "familiar"
            else vocabulary_names(tokenizer, language)
        )
        rng.shuffle(source)
        return [source[index % len(source)] for index in range(count)]
    if category == "number":
        return [str(rng.randint(100, 999_999)) for _ in range(count)]
    found: list[str] = []
    attempts = 0
    while len(found) < count:
        attempts += 1
        if attempts > MAX_ATTEMPTS:
            raise ValueError(f"cannot draw {count} {category} values for {language}")
        if category == "invented":
            candidate = latin_name(rng)
        else:
            candidate = invented(rng, language, rng.randint(*SYLLABLE_RANGE[category]))
            size = doc_tokens(tokenizer, candidate)
            if (category == "double" and size != 2) or (category == "long" and size < LONG_TOKENS):
                continue
        if candidate not in found:
            found.append(candidate)
    return found


def pools(
    tokenizer: BPETokenizer, per_cell: int = ITEMS_PER_CELL, seed: int = 11
) -> dict[tuple[str, str], list[str]]:
    result = {}
    for language in LANGUAGES:
        for category in CATEGORIES:
            want = per_cell * sum(WIDTH.values())
            drawn = draw_pool(tokenizer, language, category, want, seed)
            result[(language, category)] = drawn
    return result


def level_prompt(
    language: str, level: int, values: Sequence[str], rng: random.Random
) -> tuple[str, str, tuple[str, ...], str, str]:
    entity = rng.choice(ENTITIES)
    relation = rng.choice(RELATIONS)
    if level == 1:
        return LEVEL_ONE[language].format(X=values[0]), values[0], (), "", ""
    if level == 2:
        return LEVEL_TWO[language].format(X=values[0]), values[0], (), "", "river"
    if level == 3:
        asked = values[ENTITIES.index(entity)]
        rivals = tuple(v for v in values[:2] if v != asked)
        text = LEVEL_THREE[language].format(X1=values[0], X2=values[1], E=entity)
        return text, asked, rivals, entity, "river"
    table = {
        ("A", "river"): values[0],
        ("A", "mountain"): values[1],
        ("B", "river"): values[2],
        ("B", "mountain"): values[3],
    }
    asked = table[(entity, relation)]
    rivals = tuple(v for v in values[:4] if v != asked)
    text = LEVEL_FOUR[language].format(
        R1=values[0],
        M1=values[1],
        R2=values[2],
        M2=values[3],
        rel=ASKED[language][relation],
        E=entity,
    )
    return text, asked, rivals, entity, relation


def copy_cases(
    tokenizer: BPETokenizer, per_cell: int = ITEMS_PER_CELL, seed: int = 11
) -> list[CopyCase]:
    drawn = pools(tokenizer, per_cell, seed)
    cases: list[CopyCase] = []
    for language in LANGUAGES:
        rng = random.Random(f"{seed}:{language}:cases")
        for category in CATEGORIES:
            pool = drawn[(language, category)]
            cursor = 0
            for level in LEVELS:
                width = WIDTH[level]
                for number in range(per_cell):
                    if cursor + width > len(pool):
                        cursor = 0
                    values = pool[cursor : cursor + width]
                    cursor += width
                    prompt, target, rivals, entity, relation = level_prompt(
                        language, level, values, rng
                    )
                    name = f"{language}/{category}/L{level}/{number}"
                    cases.append(
                        CopyCase(
                            name,
                            name,
                            language,
                            level,
                            category,
                            prompt,
                            target,
                            rivals,
                            entity,
                            relation,
                        )
                    )
    return cases


def write_copy_cases(path: Path, tokenizer: BPETokenizer, seed: int = 11) -> list[CopyCase]:
    cases = copy_cases(tokenizer, seed=seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(asdict(case), ensure_ascii=False) + "\n" for case in cases),
        encoding="utf-8",
    )
    return cases


def answer_frame(case: CopyCase) -> str:
    if case.level == 1:
        return ""
    relation = FRAME_RELATIONS[case.language].get(case.relation, "")
    return FRAMES[case.level][case.language].format(E=case.entity, rel=relation)
