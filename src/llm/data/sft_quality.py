from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field

from llm.data.cleaning import detect_language
from llm.data.decontamination import Decontaminator
from llm.evaluation.metrics import repeated_ngram_ratio
from llm.reasoning.tools import evaluate_expression, numbers_equal, to_number
from llm.tokenizer.bpe import BPETokenizer, Message

KAZAKH_ONLY = frozenset("әғқұіһӘҒҚҰІҺ")
TURKIC_SHARED = frozenset("үөңҮӨҢ")
TERMINAL = (".", "!", "?", "…", "»", '"', ")", "]", "”", "`", "。")
GSM8K_ARTIFACT = re.compile(r"(=\s*)>(?=\d)")
EQUATION = re.compile(r"(?P<lhs>[\d\s.,+\-*/()x×÷]+?)=\s*(?P<rhs>-?\d[\d,]*\.?\d*(?:/\d+)?)")
WORD = re.compile(r"[^\W\d_]+", re.UNICODE)
NORMALIZE = re.compile(r"\s+")
IDENTITY_CONFLICT = re.compile(
    r"open[\s-]?assistant|\blaion\b|as an ai language model|\bchatgpt\b|\bopenai\b"
    r"|sorry for providing|как языковая модель|как ии[- ]модель",
    re.IGNORECASE,
)
KYRGYZ_MARKERS = frozenset({"жана", "үчүн", "канча", "кандай", "эмне", "болгон", "ушул", "менен"})
STEM_LENGTH = 5
MIN_STEM_WORD = 4
OFF_TOPIC_QUESTION_WORDS = 4
OFF_TOPIC_ANSWER_CHARS = 80
PLAIN_GROUPS = frozenset({"math", "word_math", "code", "agent"})
SYNTHETIC_PREFIX = "synthetic_"
EXACT_ONLY_GROUPS = PLAIN_GROUPS
LONG_ANSWER_CHARS = 160
MIN_LANGUAGE_CHECK_CHARS = 60


@dataclass(frozen=True)
class QualityRules:
    max_tokens: int = 511
    min_answer_chars: int = 2
    max_answer_chars: int = 1400
    max_script_mix: float = 0.25
    max_repetition: float = 0.45
    min_alpha_ratio: float = 0.3
    source_caps: dict[str, int] = field(default_factory=dict)
    prefix_words: int = 3
    prefix_cap: int = 600


def normalize(text: str) -> str:
    return NORMALIZE.sub(" ", text.strip().lower())


def repair_gsm8k(text: str) -> str:
    return GSM8K_ARTIFACT.sub(r"\1", text)


def equations_hold(text: str) -> bool:
    for match in EQUATION.finditer(text):
        left = evaluate_expression(
            match["lhs"].replace("x", "*").replace("×", "*").replace("÷", "/")
        )
        right = to_number(match["rhs"])
        if left is None or right is None or not left.is_number:
            continue
        if not numbers_equal(left, right):
            return False
    return True


def script_mix(text: str) -> float:
    letters = [ch for ch in text if ch.isalpha()]
    if not letters:
        return 0.0
    latin = sum(ch.isascii() for ch in letters)
    cyrillic = sum("Ѐ" <= ch <= "ӿ" for ch in letters)
    total = latin + cyrillic
    return 0.0 if total == 0 else min(latin, cyrillic) / total


def looks_kyrgyz(text: str) -> bool:
    cyrillic = [ch for ch in text if "Ѐ" <= ch <= "ӿ"]
    if len(cyrillic) < 30:
        return False
    chars = set(cyrillic)
    if chars & KAZAKH_ONLY:
        return False
    words = set(WORD.findall(text.lower()))
    return bool(chars & TURKIC_SHARED) or bool(words & KYRGYZ_MARKERS)


def stems(text: str) -> set[str]:
    return {word[:STEM_LENGTH] for word in WORD.findall(text.lower()) if len(word) >= MIN_STEM_WORD}


def is_off_topic(question: str, answer: str) -> bool:
    asked = stems(question)
    if len(asked) < OFF_TOPIC_QUESTION_WORDS or len(answer) < OFF_TOPIC_ANSWER_CHARS:
        return False
    return not asked & stems(answer)


def alpha_ratio(text: str) -> float:
    return sum(ch.isalpha() for ch in text) / len(text) if text else 0.0


def is_complete(answer: str) -> bool:
    stripped = answer.rstrip()
    if len(stripped) < LONG_ANSWER_CHARS:
        return True
    if stripped.endswith("```"):
        return True
    return stripped.endswith(TERMINAL)


class SFTQualityFilter:
    def __init__(self, rules: QualityRules, tokenizer: BPETokenizer | None = None) -> None:
        self.rules = rules
        self.tokenizer = tokenizer
        self.source_counts: Counter[str] = Counter()
        self.prefix_counts: Counter[str] = Counter()
        self.rejected: Counter[str] = Counter()

    def _prefix(self, row: dict) -> str:
        words = normalize(row["messages"][0]["content"]).split()
        return " ".join(words[: self.rules.prefix_words])

    def reason(self, row: dict) -> str | None:
        messages = row["messages"]
        if len(messages) < 2 or messages[-1]["role"] != "assistant":
            return "bad_structure"
        code_like = row["group"] in PLAIN_GROUPS or row["source"].startswith(SYNTHETIC_PREFIX)
        for message in messages:
            content = message["content"]
            if not content.strip():
                return "empty_message"
            if looks_kyrgyz(content):
                return "kyrgyz_text"
        for index, message in enumerate(messages):
            if message["role"] != "assistant":
                continue
            answer = message["content"]
            if len(answer.strip()) < self.rules.min_answer_chars:
                return "answer_too_short"
            if code_like:
                continue
            if len(answer) > self.rules.max_answer_chars:
                return "answer_too_long"
            if IDENTITY_CONFLICT.search(answer):
                return "identity_conflict"
            if not is_complete(answer):
                return "incomplete_answer"
            if script_mix(answer) > self.rules.max_script_mix:
                return "mixed_scripts"
            if alpha_ratio(answer) < self.rules.min_alpha_ratio and len(answer) > 40:
                return "low_alpha"
            words = WORD.findall(answer.lower())
            if len(words) > 12 and repeated_ngram_ratio(words, 3) > self.rules.max_repetition:
                return "repetitive_answer"
            question = messages[index - 1]["content"] if index else ""
            if normalize(answer) == normalize(question):
                return "answer_echoes_question"
            if is_off_topic(question, answer):
                return "off_topic"
            if self._language_mismatch(question, answer):
                return "language_mismatch"
        if self.tokenizer is not None and self._too_long(messages):
            return "too_many_tokens"
        return None

    def _language_mismatch(self, question: str, answer: str) -> bool:
        if len(question) < MIN_LANGUAGE_CHECK_CHARS or len(answer) < MIN_LANGUAGE_CHECK_CHARS:
            return False
        asked, replied = detect_language(question), detect_language(answer)
        return "other" not in (asked, replied) and asked != replied and "en" in (asked, replied)

    def _too_long(self, messages: list[dict]) -> bool:
        assert self.tokenizer is not None
        ids = self.tokenizer.encode_conversation(
            [Message(m["role"], m["content"]) for m in messages]
        )
        return len(ids) > self.rules.max_tokens

    def admit(self, row: dict) -> bool:
        reason = self.reason(row)
        if reason is None:
            cap = self.rules.source_caps.get(row["source"])
            if cap is not None and self.source_counts[row["source"]] >= cap:
                reason = "source_cap"
        if reason is None and row["group"].startswith("chat"):
            prefix = self._prefix(row)
            if self.prefix_counts[prefix] >= self.rules.prefix_cap:
                reason = "template_cap"
            self.prefix_counts[prefix] += reason is None
        if reason is not None:
            self.rejected[f"{row['source']}:{reason}"] += 1
            return False
        self.source_counts[row["source"]] += 1
        return True


class PromptBlocklist:
    def __init__(self, prompts: Iterable[str], ngram_size: int = 6) -> None:
        texts = list(prompts)
        self.exact = {normalize(text) for text in texts}
        self.grams = Decontaminator(texts, ngram_size)

    def blocks(self, row: dict) -> bool:
        for message in row["messages"]:
            if message["role"] != "user":
                continue
            if normalize(message["content"]) in self.exact:
                return True
            if row["group"] not in EXACT_ONLY_GROUPS and self.grams.is_contaminated(
                message["content"]
            ):
                return True
        return False
