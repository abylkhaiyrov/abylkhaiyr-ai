from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable
from math import log

from llm.data.cleaning import (
    clean_document,
    repeated_line_ratio,
    symbol_ratio,
)

WORDS = re.compile(r"[^\W\d_]+")
CYRILLIC = re.compile(r"[а-яёғқңөұүһіә]+")
KYRGYZ_MARKERS = frozenset({"эмне", "үчүн", "болгон", "ушул", "кандай", "дегенде", "айтылат"})
CORRUPTION = re.compile(r"\ufffd|\x00|(?:Ã.|Â.|â€){2,}")
LANGUAGES = ("en", "ru", "kk")


def trigrams(text: str) -> Counter[str]:
    result: Counter[str] = Counter()
    for word in CYRILLIC.findall(text[:1600].lower()):
        padded = f"^{word}$"
        result.update(padded[index : index + 3] for index in range(len(padded) - 2))
    return result


class CyrillicLanguageFilter:
    def __init__(self, examples: dict[str, Iterable[str]]) -> None:
        counts: dict[str, Counter[str]] = {}
        for language, texts in examples.items():
            counts[language] = Counter()
            for text in texts:
                counts[language].update(trigrams(text))
        vocabulary = set().union(*(counter.keys() for counter in counts.values()))
        self.profiles: dict[str, dict[str, float]] = {}
        self.defaults: dict[str, float] = {}
        for language, counter in counts.items():
            total = counter.total()
            if not total:
                raise ValueError(f"empty language profile: {language}")
            denominator = total + len(vocabulary)
            self.defaults[language] = -log(denominator)
            self.profiles[language] = {
                key: log(count + 1) - log(denominator) for key, count in counter.items()
            }

    def accepts(self, text: str, expected: str, *, allow_short: bool = False) -> bool:
        letters = [char for char in text.lower() if char.isalpha()]
        if expected not in LANGUAGES or not letters:
            return False
        latin = sum(char.isascii() for char in letters) / len(letters)
        if expected == "en":
            return latin >= 0.9
        cyrillic = sum("\u0400" <= char <= "\u04ff" for char in letters) / len(letters)
        markers = KYRGYZ_MARKERS | {"ооба", "жөнүндө"} if allow_short else KYRGYZ_MARKERS
        if cyrillic < 0.7 or set(WORDS.findall(text.lower())) & markers:
            return False
        features = trigrams(text)
        count = features.total()
        if allow_short and expected == "ru" and any(char in "әғқңөұүһі" for char in letters):
            return False
        if (
            allow_short
            and 2 <= count < 16
            and expected == "kk"
            and any(char in "әғқұһі" for char in letters)
        ):
            return True
        if count < (2 if allow_short else 8):
            return False
        scores = {
            language: sum(
                frequency * profile.get(key, self.defaults[language])
                for key, frequency in features.items()
            )
            / count
            for language, profile in self.profiles.items()
        }
        return all(
            scores[expected] > score + 0.1
            for language, score in scores.items()
            if language != expected
        )


def clean_v4_text(raw: str, minimum: int) -> str | None:
    if CORRUPTION.search(raw):
        return None
    text = clean_document(raw, strip_markup=False)
    if len(text) < minimum or symbol_ratio(text) > 0.3 or repeated_line_ratio(text) > 0.5:
        return None
    return text
