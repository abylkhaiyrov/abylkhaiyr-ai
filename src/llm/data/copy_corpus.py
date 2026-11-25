from __future__ import annotations

import json
import random
import re
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from functools import cache
from pathlib import Path
from typing import Any

from llm.data.binary import tokenize_sft_split
from llm.data.copy_ladder import (
    CATEGORIES,
    FAMILIAR,
    LANGUAGES,
    CopyCase,
    doc_tokens,
    invented,
    latin_name,
    permuted_cases,
    vocabulary_names,
)
from llm.data.copy_text import (
    ANSWERS,
    FRAMES_DEV,
    FRAMES_HOLDOUT,
    FRAMES_TRAIN,
    INSTRUCTIONS_DEV,
    INSTRUCTIONS_HOLDOUT,
    INSTRUCTIONS_TRAIN,
    LABELS,
    LAYOUTS,
    NOUNS,
    QUESTIONS_DEV,
    QUESTIONS_HOLDOUT,
    QUESTIONS_TRAIN,
)
from llm.data.dedup import stable_hash
from llm.data.grounded_text import FILLERS
from llm.tokenizer.bpe import BPETokenizer

PRIMARY = ("long", "number", "invented")
EVAL_ONLY = ("code", "phrase", "bignumber")
TRAIN_COUNT = 20_000
LEVEL_SHARE = {1: 0.4, 2: 0.6}
WEIGHTS = {
    "long": 0.26,
    "number": 0.20,
    "invented": 0.24,
    "double": 0.12,
    "single": 0.10,
    "familiar": 0.08,
}
SEED = 41
MAX_ATTEMPTS = 400_000
SCRIPT = {
    "en": re.compile(r"[a-z]{4,}"),
    "ru": re.compile(r"[а-яё]{4,}"),
    "kk": re.compile(r"[а-яёәғқңөұүһі]{4,}"),
}
CAPITAL = {
    "en": re.compile(r"[A-Z][a-z]+"),
    "ru": re.compile(r"[А-ЯЁ][а-яё]+"),
    "kk": re.compile(r"[А-ЯЁӘҒҚҢӨҰҮҺІ][а-яёәғқңөұүһі]+"),
}
LOWER = {
    "en": re.compile(r"[a-z]{2,}"),
    "ru": re.compile(r"[а-яё]{2,}"),
    "kk": re.compile(r"[а-яёәғқңөұүһі]{2,}"),
}
CELL_SIZES = {"seen": 16, "wording": 6, "category": 8, "cross": 6}
TWIN = "twin"
TEMPLATES = {
    "train": (INSTRUCTIONS_TRAIN, FRAMES_TRAIN, QUESTIONS_TRAIN),
    "dev": (INSTRUCTIONS_DEV, FRAMES_DEV, QUESTIONS_DEV),
    "holdout": (INSTRUCTIONS_HOLDOUT, FRAMES_HOLDOUT, QUESTIONS_HOLDOUT),
}


@dataclass(frozen=True)
class CorpusCase(CopyCase):
    cell: str = ""
    noun: str = ""


def value_split(value: str) -> str:
    bucket = stable_hash(f"copyvalue:{value}") % 100
    return "train" if bucket < 80 else "dev" if bucket < 90 else "holdout"


@cache
def vocabulary_pieces(tokenizer_version: str, language: str, kind: str) -> tuple[str, ...]:
    raise NotImplementedError


def pieces(tokenizer: BPETokenizer, language: str) -> tuple[list[str], list[str]]:
    firsts: list[str] = []
    seconds: list[str] = []
    for token, index in tokenizer.backend.get_vocab().items():
        text = tokenizer.decode([index])
        if token.startswith("Ġ") and CAPITAL[language].fullmatch(text.strip()):
            firsts.append(text.strip())
        elif not token.startswith("Ġ") and LOWER[language].fullmatch(text):
            seconds.append(text)
    return sorted(firsts), sorted(seconds)


@cache
def finite_pool(tokenizer: BPETokenizer, language: str, category: str) -> tuple[str, ...]:
    if category == "single":
        return tuple(vocabulary_names(tokenizer, language))
    if category == "familiar":
        words = set(FAMILIAR[language])
        for token, index in tokenizer.backend.get_vocab().items():
            text = tokenizer.decode([index]).strip()
            if token.startswith("Ġ") and SCRIPT[language].fullmatch(text):
                words.add(text)
        return tuple(sorted(words))
    firsts, seconds = pieces(tokenizer, language)
    found = []
    for first in firsts:
        for second in seconds:
            if doc_tokens(tokenizer, first + second) == 2:
                found.append(first + second)
    return tuple(sorted(found))


class Values:
    def __init__(self, tokenizer: BPETokenizer, split: str, exclude: frozenset[str]) -> None:
        self.tokenizer = tokenizer
        self.split = split
        self.exclude = exclude
        self.finite: dict[tuple[str, str], list[str]] = {}
        self.used: dict[tuple[str, str], int] = Counter()

    def pool(self, language: str, category: str) -> list[str]:
        key = (language, category)
        if key not in self.finite:
            raw = finite_pool(self.tokenizer, language, category)
            kept = [v for v in raw if value_split(v) == self.split and v not in self.exclude]
            self.finite[key] = kept
        return self.finite[key]

    def draw(self, rng: random.Random, language: str, category: str) -> str:
        if category in ("single", "familiar", "double"):
            pool = self.pool(language, category)
            return pool[rng.randrange(len(pool))]
        for _ in range(MAX_ATTEMPTS):
            candidate = self.generate(rng, language, category)
            if value_split(candidate) == self.split and candidate not in self.exclude:
                return candidate
        raise ValueError(f"cannot draw {category} for {language} in {self.split}")

    def generate(self, rng: random.Random, language: str, category: str) -> str:
        if category == "number":
            return str(rng.randint(100, 999_999))
        if category == "bignumber":
            return str(rng.randint(1_000_000, 999_999_999))
        if category == "invented":
            return latin_name(rng)
        if category == "code":
            return f"{rng.choice('KMRTXZ')}{rng.randint(1, 9)}-{rng.randint(1000, 9999)}"
        if category == "phrase":
            return f"{latin_name(rng)} {latin_name(rng)}"
        candidate = invented(rng, language, rng.randint(3, 6))
        return (
            candidate
            if doc_tokens(self.tokenizer, candidate) >= 4
            else self.generate(rng, language, category)
        )


def render_context(
    language: str, noun: str, value: str, split: str, rng: random.Random
) -> tuple[str, str, str]:
    _, frames, questions = TEMPLATES[split]
    frame = rng.choice(frames[language]).format(noun=noun, Noun=noun.capitalize(), X=value)
    question = rng.choice(questions[language]).format(noun=noun)
    fillers = rng.sample(FILLERS[language], rng.randint(0, 2))
    before = " ".join(fillers[: len(fillers) // 2 + len(fillers) % 2])
    after = " ".join(fillers[len(fillers) // 2 + len(fillers) % 2 :])
    context = " ".join(part for part in (before, frame, after) if part)
    return context, question, frame


def level_prompt(
    language: str, level: int, value: str, split: str, rng: random.Random
) -> tuple[str, str]:
    instructions = TEMPLATES[split][0]
    if level == 1:
        return rng.choice(instructions[language]).format(X=value), ""
    noun = rng.choice(NOUNS[language])
    context, question, _ = render_context(language, noun, value, split, rng)
    layout = rng.choice(LAYOUTS)
    text_label, question_label = LABELS[language]
    if layout == "question_first":
        return f"{question}\n\n{context}", noun
    if layout == "labeled":
        return f"{text_label}: {context}\n{question_label}: {question}", noun
    if layout == "text_inline":
        return f"{text_label}: {context} {question}", noun
    return f"{context} {question}", noun


def answer_of(language: str, level: int, value: str, noun: str, rng: random.Random) -> str:
    if level == 1:
        return value
    style = rng.choices(range(4), weights=(40, 20, 20, 20))[0]
    return ANSWERS[language][style].format(noun=noun, X=value)


def category_plan(count: int, weights: dict[str, float]) -> list[str]:
    ordered = sorted(weights)
    plan: list[str] = []
    for category in ordered:
        plan.extend([category] * round(count * weights[category]))
    while len(plan) < count:
        plan.append(ordered[len(plan) % len(ordered)])
    return plan[:count]


def build_train(
    tokenizer: BPETokenizer, exclude: frozenset[str], count: int = TRAIN_COUNT, seed: int = SEED
) -> list[CorpusCase]:
    rng = random.Random(f"{seed}:train")
    values = Values(tokenizer, "train", exclude)
    cases: list[CorpusCase] = []
    seen_prompts: set[str] = set()
    per_language = count // len(LANGUAGES)
    for language in LANGUAGES:
        wanted = per_language + (
            count - per_language * len(LANGUAGES) if language == LANGUAGES[-1] else 0
        )
        levels = [1] * round(wanted * LEVEL_SHARE[1]) + [2] * (
            wanted - round(wanted * LEVEL_SHARE[1])
        )
        categories = category_plan(wanted, WEIGHTS)
        rng.shuffle(levels)
        rng.shuffle(categories)
        for level, category in zip(levels, categories, strict=True):
            for _ in range(50):
                value = values.draw(rng, language, category)
                prompt, noun = level_prompt(language, level, value, "train", rng)
                if prompt not in seen_prompts:
                    break
            else:
                continue
            seen_prompts.add(prompt)
            name = f"train/{language}/{len(cases)}"
            cases.append(
                CorpusCase(
                    name, name, language, level, category, prompt, value, (), "", "", "train", noun
                )
            )
    return cases


def eval_case(
    name: str,
    family: str,
    language: str,
    level: int,
    category: str,
    prompt: str,
    value: str,
    cell: str,
    noun: str,
) -> CorpusCase:
    return CorpusCase(
        name, family, language, level, category, prompt, value, (), "", "", cell, noun
    )


def build_eval(
    tokenizer: BPETokenizer, split: str, exclude: frozenset[str], seed: int = SEED
) -> list[CorpusCase]:
    rng = random.Random(f"{seed}:{split}")
    values = Values(tokenizer, split, exclude)
    cases: list[CorpusCase] = []

    def add_cell(
        cell: str, categories: Sequence[str], size: int, template: str, foreign: bool = False
    ) -> None:
        for language in LANGUAGES:
            for category in categories:
                for level in (1, 2):
                    count = size // 2 * 2 if level == 2 and cell == "seen" else size
                    step = 2 if level == 2 and cell == "seen" else 1
                    for number in range(0, count, step):
                        source = (
                            [other for other in LANGUAGES if other != language][number % 2]
                            if foreign
                            else language
                        )
                        first = values.draw(rng, source, category)
                        group = f"{split}/{cell}/{language}/{category}/L{level}/{number}"
                        for twin in range(step):
                            value = first
                            while twin and value == first:
                                value = values.draw(rng, source, category)
                            local = random.Random(f"{seed}:{group}")
                            prompt, noun = level_prompt(language, level, value, template, local)
                            name = f"{group}/{twin}"
                            family = group if step == 2 else name
                            cases.append(
                                eval_case(
                                    name,
                                    family,
                                    language,
                                    level,
                                    category,
                                    prompt,
                                    value,
                                    cell,
                                    noun,
                                )
                            )

    add_cell("seen", CATEGORIES, CELL_SIZES["seen"], "train")
    add_cell("wording", CATEGORIES, CELL_SIZES["wording"], split)
    add_cell("category", EVAL_ONLY, CELL_SIZES["category"], "train")
    add_cell("cross", ("long", "single", "double", "familiar"), CELL_SIZES["cross"], "train", True)
    return cases


def exclusions(ladder: Iterable[CopyCase], extra: Iterable[str] = ()) -> frozenset[str]:
    return frozenset([*(case.target for case in ladder), *extra])


def training_rows(cases: Sequence[CorpusCase], seed: int = SEED) -> list[dict[str, Any]]:
    rng = random.Random(f"{seed}:answers")
    rows = []
    for case in cases:
        answer = answer_of(case.language, case.level, case.target, case.noun, rng)
        rows.append(
            {
                "category": f"copy_l{case.level}",
                "value_category": case.category,
                "language": case.language,
                "messages": [
                    {"role": "user", "content": case.prompt},
                    {"role": "assistant", "content": answer},
                ],
            }
        )
    return rows


def write_jsonl(path: Path, items: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in items), encoding="utf-8"
    )


def manifest_of(
    train: Sequence[CorpusCase],
    dev: Sequence[CorpusCase],
    holdout: Sequence[CorpusCase],
    tokenizer: BPETokenizer,
) -> dict[str, Any]:
    def stats(cases: Sequence[CorpusCase]) -> dict[str, Any]:
        return {
            "examples": len(cases),
            "unique_prompts": len({c.prompt for c in cases}),
            "unique_values": len({c.target for c in cases}),
            "by_language": dict(Counter(c.language for c in cases)),
            "by_level": dict(Counter(c.level for c in cases)),
            "by_category": dict(Counter(c.category for c in cases)),
            "unique_values_by_category": {
                category: len({c.target for c in cases if c.category == category})
                for category in sorted({c.category for c in cases})
            },
            "mean_value_tokens": sum(doc_tokens(tokenizer, c.target) for c in cases)
            / max(len(cases), 1),
        }

    train_values = {c.target for c in train}
    train_tokens = {t for c in train for t in tokenizer.encode(f" {c.target}")}
    held = [(name, cases) for name, cases in (("dev", dev), ("holdout", holdout))]
    return {
        "train": stats(train),
        "dev": stats(dev),
        "holdout": stats(holdout),
        "target_overlap": {
            name: len(train_values & {c.target for c in cases}) for name, cases in held
        },
        "dev_holdout_target_overlap": len({c.target for c in dev} & {c.target for c in holdout}),
        "shared_bpe_token_share": {
            name: len({t for c in cases for t in tokenizer.encode(f" {c.target}")} & train_tokens)
            / max(len({t for c in cases for t in tokenizer.encode(f" {c.target}")}), 1)
            for name, cases in held
        },
    }


def write_corpus(
    output: Path,
    tokenizer: BPETokenizer,
    ladder: Sequence[CopyCase],
    earlier: Iterable[str],
    count: int = TRAIN_COUNT,
) -> dict[str, Any]:
    exclude = exclusions(ladder, earlier)
    train = build_train(tokenizer, exclude, count)
    dev = build_eval(tokenizer, "dev", exclude)
    holdout = build_eval(tokenizer, "holdout", exclude)
    validation = [c for c in dev if c.cell == "seen"][:200]
    directory = output / "B" / "sft"
    rows = training_rows(train)
    write_jsonl(directory / "train.jsonl", rows)
    write_jsonl(directory / "validation.jsonl", training_rows(validation))
    tokenize_sft_split(directory / "train.jsonl", directory / "train", tokenizer)
    tokenize_sft_split(directory / "validation.jsonl", directory / "validation", tokenizer)
    write_jsonl(output / "dev.jsonl", (asdict(c) for c in dev))
    write_jsonl(output / "holdout.jsonl", (asdict(c) for c in holdout))
    sampled = train[:: max(len(train) // 36, 1)][:36]
    write_jsonl(output / "trained_B.jsonl", (asdict(c) for c in sampled))
    write_jsonl(output / "permuted_B.jsonl", (asdict(c) for c in permuted_cases(sampled)))
    manifest = manifest_of(train, dev, holdout, tokenizer)
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return manifest


def load_corpus_cases(path: str) -> list[CopyCase]:
    cases: list[CopyCase] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        row["rivals"] = tuple(row["rivals"])
        cases.append(CorpusCase(**row) if "cell" in row else CopyCase(**row))
    return cases
