from __future__ import annotations

import re

from llm.data.cleaning import detect_language
from llm.reasoning.intent import ANALYSIS, CODE, LOGIC, MATH, TRANSLATION, classify_intent

MATH_TOKEN = re.compile(
    r"\d|\b(?:sin|cos|tan|log|ln|exp|sqrt|pi)\b|[a-z]\s*[\^=+\-*/]\s*[a-z\d]|[xyz]\^", re.IGNORECASE
)
LANGUAGE_MENTION = re.compile(
    r"\b(?:russian|english|kazakh|russian|русск\w+|английск\w+|казахск\w+|орыс\w*|ағылшын\w*|қазақ\w*)\b",
    re.IGNORECASE,
)
TRANSLATION_FILLER = re.compile(
    r"\b(?:translate|translation|please|this|it|that|into|to|переведи\w*|пожалуйста|это|эту|"
    r"мынаны|аудар\w*|на|язык\w*|тіліне)\b",
    re.IGNORECASE,
)
QUOTED = re.compile(r"[:«\"]")
MIN_CODE_WORDS = 4
MIN_TEXT_CHARS = 30

MISSING_NAMES = {
    "numbers": {
        "en": "the numbers or the expression",
        "ru": "числа или выражение",
        "kk": "сандар немесе өрнек",
    },
    "text": {"en": "the text", "ru": "текст", "kk": "мәтін"},
    "target_language": {
        "en": "the target language",
        "ru": "язык перевода",
        "kk": "аудару тілі",
    },
    "task": {
        "en": "a description of the task",
        "ru": "описание задачи",
        "kk": "тапсырма сипаттамасы",
    },
}
CLARIFICATION_TEMPLATE = {
    "en": "I need more information to answer. Please provide: {items}.",
    "ru": "Мне не хватает информации для ответа. Пожалуйста, укажите: {items}.",
    "kk": "Жауап беру үшін ақпарат жеткіліксіз. Мына нәрсені көрсетіңіз: {items}.",
}


def has_payload(question: str, minimum: int) -> bool:
    return ":" in question or "«" in question or '"' in question or len(question) >= minimum


def missing_information(question: str) -> list[str]:
    intent = classify_intent(question)
    missing: list[str] = []
    if intent in {MATH, LOGIC} and not MATH_TOKEN.search(question) and len(question.split()) < 12:
        missing.append("numbers")
    if intent == CODE and len(question.split()) < MIN_CODE_WORDS and "```" not in question:
        missing.append("task")
    if intent == ANALYSIS and not has_payload(question, MIN_TEXT_CHARS):
        missing.append("text")
    if intent == TRANSLATION:
        remainder = TRANSLATION_FILLER.sub(" ", LANGUAGE_MENTION.sub(" ", question))
        if not QUOTED.search(question) and not re.search(r"\w", remainder):
            missing.append("text")
        if not LANGUAGE_MENTION.search(question):
            missing.append("target_language")
    return missing


def clarification_message(question: str, missing: list[str]) -> str:
    language = detect_language(question)
    language = language if language in CLARIFICATION_TEMPLATE else "en"
    items = ", ".join(MISSING_NAMES[item][language] for item in missing)
    return CLARIFICATION_TEMPLATE[language].format(items=items)
