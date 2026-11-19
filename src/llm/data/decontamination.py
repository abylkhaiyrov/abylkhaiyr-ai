from __future__ import annotations

import re
from collections.abc import Iterable

from llm.data.dedup import stable_hash

WORD_PATTERN = re.compile(r"\w+", re.UNICODE)


def word_ngrams(text: str, size: int) -> Iterable[int]:
    words = WORD_PATTERN.findall(text.lower())
    for index in range(max(len(words) - size + 1, 0)):
        yield stable_hash(" ".join(words[index : index + size]))


class Decontaminator:
    def __init__(self, benchmark_texts: Iterable[str], ngram_size: int) -> None:
        self.ngram_size = ngram_size
        self.blocked: set[int] = set()
        for text in benchmark_texts:
            self.blocked.update(word_ngrams(text, ngram_size))

    def is_contaminated(self, text: str) -> bool:
        return any(gram in self.blocked for gram in word_ngrams(text, self.ngram_size))


def benchmark_texts(gsm8k_cases: int | None = None) -> list[str]:
    from datasets import load_dataset

    from llm.evaluation import suites

    texts: list[str] = []
    gsm8k = load_dataset("openai/gsm8k", "main", split="test")
    for row in gsm8k:
        texts.extend((row["question"], row["answer"]))
    texts.extend(case.prompt for case in suites.CODE_CASES)
    texts.extend(case.prompt for case in suites.FACTUAL_CASES)
    texts.extend(case.prompt for case in suites.instruction_cases())
    return texts
