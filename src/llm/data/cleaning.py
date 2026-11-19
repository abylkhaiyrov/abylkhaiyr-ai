from __future__ import annotations

import html
import re
import unicodedata
from collections import Counter

TAG_PATTERN = re.compile(r"<(script|style)\b.*?</\1>|<[^>]+>", re.DOTALL | re.IGNORECASE)
EMAIL_PATTERN = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
PHONE_PATTERN = re.compile(r"(?<!\d)(?:\+?\d[\s\-()]{0,2}){10,15}(?!\d)")
IPV4_PATTERN = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
CARD_PATTERN = re.compile(r"\b(?:\d[ -]?){13,19}\b")
BLANK_LINES = re.compile(r"\n{3,}")
INLINE_SPACES = re.compile(r"[^\S\n]+")
CONTROL_CATEGORIES = {"Cc", "Cf", "Cs", "Co"}
KEEP_CONTROL = {"\n", "\t"}

EMAIL_TOKEN = "<email>"
PHONE_TOKEN = "<phone>"
IP_TOKEN = "<ip>"
CARD_TOKEN = "<card>"

KAZAKH_LETTERS = set("әғқңөұүһіӘҒҚҢӨҰҮҺІ")
UKRAINIAN_LETTERS = set("їєґЇЄҐ")
ENGLISH_STOPWORDS = frozenset(
    "the of and to in is that it for as with was on are by this be at from or an have not"
    " but his her they you which their has had were been one all would there what".split()
)
SCRIPT_DOMINANCE = 0.5
KAZAKH_LETTER_RATIO = 0.01
ENGLISH_STOPWORD_RATIO = 0.12
OTHER_LANGUAGE = "other"


def strip_html(text: str) -> str:
    return html.unescape(TAG_PATTERN.sub(" ", text))


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text).replace("\r\n", "\n").replace("\r", "\n")
    text = "".join(
        ch for ch in text if ch in KEEP_CONTROL or unicodedata.category(ch) not in CONTROL_CATEGORIES
    )
    text = INLINE_SPACES.sub(" ", text)
    text = "\n".join(line.strip() for line in text.split("\n"))
    return BLANK_LINES.sub("\n\n", text).strip()


def remove_personal_data(text: str) -> str:
    text = EMAIL_PATTERN.sub(EMAIL_TOKEN, text)
    text = CARD_PATTERN.sub(CARD_TOKEN, text)
    text = PHONE_PATTERN.sub(PHONE_TOKEN, text)
    return IPV4_PATTERN.sub(IP_TOKEN, text)


def clean_document(text: str, strip_markup: bool = True) -> str:
    text = strip_html(text) if strip_markup else text
    return remove_personal_data(normalize_text(text))


def detect_language(text: str) -> str:
    letters = [ch for ch in text if ch.isalpha()]
    if not letters:
        return OTHER_LANGUAGE
    cyrillic = [ch for ch in letters if "Ѐ" <= ch <= "ӿ"]
    latin = [ch for ch in letters if ch.isascii()]
    if len(cyrillic) / len(letters) >= SCRIPT_DOMINANCE:
        if any(ch in UKRAINIAN_LETTERS for ch in cyrillic):
            return OTHER_LANGUAGE
        kazakh = sum(ch in KAZAKH_LETTERS for ch in cyrillic)
        return "kk" if kazakh / len(cyrillic) >= KAZAKH_LETTER_RATIO else "ru"
    if len(latin) / len(letters) >= SCRIPT_DOMINANCE:
        words = re.findall(r"[a-z]+", text.lower())
        if words and sum(w in ENGLISH_STOPWORDS for w in words) / len(words) >= ENGLISH_STOPWORD_RATIO:
            return "en"
    return OTHER_LANGUAGE


def symbol_ratio(text: str) -> float:
    if not text:
        return 1.0
    symbols = sum(not (ch.isalnum() or ch.isspace()) for ch in text)
    return symbols / len(text)


def repeated_line_ratio(text: str) -> float:
    lines = [line for line in text.split("\n") if line.strip()]
    if not lines:
        return 1.0
    most_common = Counter(lines).most_common(1)[0][1]
    return most_common / len(lines) if len(lines) > 1 else 0.0


def clean_for_kind(text: str, kind: str) -> str:
    if kind in {"code", "math"}:
        return EMAIL_PATTERN.sub(EMAIL_TOKEN, normalize_text(text))
    return clean_document(text)
