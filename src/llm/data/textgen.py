from __future__ import annotations

import json
import random
import re
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path

from llm.data.cleaning import detect_language
from llm.data.dedup import stable_hash
from llm.data.lexicon import LANGUAGES

SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-ZА-ЯЁӘҒҚҢӨҰҮҺІ«\"])")
WORD = re.compile(r"[^\W\d_]+(?:-[^\W\d_]+)*", re.UNICODE)
NUMBER = re.compile(r"\d+(?:[.,]\d+)?")
ARTIFACT = re.compile(r"\s{2,}|\s[,;:)]|[(]\s|^[^\w«\"(]|[(][^\w]*[)]|\[|\]|\{|\}")
MIN_SENTENCE_CHARS = 25
MAX_SENTENCE_CHARS = 220
MIN_PASSAGE_CHARS = 90
MAX_PASSAGE_CHARS = 420
MAX_DIGIT_RATIO = 0.12
MIN_LETTER_RATIO = 0.65
MIN_KEYWORD_LENGTH = 6
MAX_LISTED_NUMBERS = 6
SENTENCE_COUNTS = (2, 3, 4)


@dataclass(frozen=True)
class Passage:
    language: str
    sentences: tuple[str, ...]

    @property
    def text(self) -> str:
        return " ".join(self.sentences)


def usable_sentence(sentence: str) -> bool:
    if not MIN_SENTENCE_CHARS <= len(sentence) <= MAX_SENTENCE_CHARS:
        return False
    if sentence[-1] not in ".!?" or "http" in sentence or "\n" in sentence:
        return False
    if ARTIFACT.search(sentence):
        return False
    letters = sum(ch.isalpha() for ch in sentence)
    digits = sum(ch.isdigit() for ch in sentence)
    return letters / len(sentence) >= MIN_LETTER_RATIO and digits / len(sentence) <= MAX_DIGIT_RATIO


def passages_from_document(text: str, language: str, rng: random.Random) -> Iterator[Passage]:
    sentences = [part.strip() for part in SENTENCE_SPLIT.split(text.replace("\n", " ").strip())]
    start = 0
    while start < len(sentences) - 1:
        count = rng.choice(SENTENCE_COUNTS)
        window = tuple(sentences[start : start + count])
        start += count
        if len(window) != count or not all(usable_sentence(s) for s in window):
            continue
        passage = Passage(language, window)
        if (
            MIN_PASSAGE_CHARS <= len(passage.text) <= MAX_PASSAGE_CHARS
            and detect_language(passage.text) == language
        ):
            yield passage
            return


def sample_passages(
    corpus_file: Path, per_language: int, seed: int, stride: int = 7
) -> dict[str, list[Passage]]:
    rng = random.Random(seed)
    found: dict[str, list[Passage]] = {language: [] for language in LANGUAGES}
    with corpus_file.open(encoding="utf-8") as handle:
        for line in handle:
            if all(len(items) >= per_language for items in found.values()):
                break
            if stable_hash(f"{seed}:{line[:200]}") % stride:
                continue
            row = json.loads(line)
            language = row.get("language")
            if language not in found or len(found[language]) >= per_language:
                continue
            found[language].extend(passages_from_document(row["text"], language, rng))
    return found


def words_of(text: str) -> list[str]:
    return WORD.findall(text)


def keyword_candidates(passage: Passage) -> list[str]:
    return [w for w in words_of(passage.text) if len(w) >= MIN_KEYWORD_LENGTH]


COUNT_QUESTION = {
    "en": ("How many sentences are in this text?", "Count the sentences in the text below."),
    "ru": ("Сколько предложений в этом тексте?", "Посчитай предложения в тексте ниже."),
    "kk": ("Бұл мәтінде неше сөйлем бар?", "Төмендегі мәтіндегі сөйлемдерді санап бер."),
}
COUNT_ANSWER = {
    "en": "The text has {n} sentences.",
    "ru": "В тексте {n} предложения.",
    "kk": "Мәтінде {n} сөйлем бар.",
}
FIRST_QUESTION = {
    "en": ("Copy the first sentence of the text.", "What is the first sentence of this text?"),
    "ru": ("Выпиши первое предложение текста.", "Какое первое предложение в этом тексте?"),
    "kk": ("Мәтіннің бірінші сөйлемін көшіріп жаз.", "Бұл мәтіннің бірінші сөйлемі қандай?"),
}
LAST_QUESTION = {
    "en": ("Copy the last sentence of the text.", "What is the last sentence of this text?"),
    "ru": ("Выпиши последнее предложение текста.", "Какое последнее предложение в этом тексте?"),
    "kk": ("Мәтіннің соңғы сөйлемін көшіріп жаз.", "Бұл мәтіннің соңғы сөйлемі қандай?"),
}
LANGUAGE_QUESTION = {
    "en": ("Which language is this text written in?", "Identify the language of the text."),
    "ru": ("На каком языке написан этот текст?", "Определи язык этого текста."),
    "kk": ("Бұл мәтін қай тілде жазылған?", "Мәтіннің тілін анықта."),
}
LANGUAGE_ANSWER = {
    "en": "The text is written in {name}.",
    "ru": "Текст написан на {name} языке.",
    "kk": "Мәтін {name} жазылған.",
}
LANGUAGE_NAME_FORMS = {
    "en": {"ru": "Russian", "en": "English", "kk": "Kazakh"},
    "ru": {"ru": "русском", "en": "английском", "kk": "казахском"},
    "kk": {"ru": "орыс тілінде", "en": "ағылшын тілінде", "kk": "қазақ тілінде"},
}
NUMBERS_QUESTION = {
    "en": ("List all numbers that appear in the text, separated by commas.",),
    "ru": ("Перечисли через запятую все числа, которые встречаются в тексте.",),
    "kk": ("Мәтінде кездесетін барлық сандарды үтір арқылы тізіп жаз.",),
}
LONGEST_QUESTION = {
    "en": ("What is the longest word in the text?",),
    "ru": ("Какое слово в тексте самое длинное?",),
    "kk": ("Мәтіндегі ең ұзын сөз қайсы?",),
}
LONGEST_ANSWER = {
    "en": "The longest word is “{w}”.",
    "ru": "Самое длинное слово — «{w}».",
    "kk": "Ең ұзын сөз — «{w}».",
}
CONTAINS_QUESTION = {
    "en": "Does the text mention the word “{w}”? Answer yes or no.",
    "ru": "Упоминается ли в тексте слово «{w}»? Ответь да или нет.",
    "kk": "Мәтінде «{w}» сөзі кездесе ме? Иә немесе жоқ деп жауап бер.",
}
YES_NO = {"en": ("Yes.", "No."), "ru": ("Да.", "Нет."), "kk": ("Иә.", "Жоқ.")}
WHICH_QUESTION = {
    "en": "Which sentence of the text contains the word “{w}”?",
    "ru": "В каком предложении текста есть слово «{w}»?",
    "kk": "Мәтіннің қай сөйлемінде «{w}» сөзі бар?",
}

TaskResult = tuple[str, str, str] | None
Builder = Callable[[random.Random, Passage, str, list[str]], TaskResult]


def count_task(rng: random.Random, passage: Passage, lang: str, pool: list[str]) -> TaskResult:
    count = len(passage.sentences)
    return rng.choice(COUNT_QUESTION[lang]), COUNT_ANSWER[lang].format(n=count), "count_sentences"


def first_task(rng: random.Random, passage: Passage, lang: str, pool: list[str]) -> TaskResult:
    return rng.choice(FIRST_QUESTION[lang]), passage.sentences[0], "first_sentence"


def last_task(rng: random.Random, passage: Passage, lang: str, pool: list[str]) -> TaskResult:
    return rng.choice(LAST_QUESTION[lang]), passage.sentences[-1], "last_sentence"


def language_task(rng: random.Random, passage: Passage, lang: str, pool: list[str]) -> TaskResult:
    answer = LANGUAGE_ANSWER[lang].format(name=LANGUAGE_NAME_FORMS[lang][passage.language])
    return rng.choice(LANGUAGE_QUESTION[lang]), answer, "language_id"


def numbers_task(rng: random.Random, passage: Passage, lang: str, pool: list[str]) -> TaskResult:
    numbers = NUMBER.findall(passage.text)
    if not numbers or len(numbers) > MAX_LISTED_NUMBERS:
        return None
    return rng.choice(NUMBERS_QUESTION[lang]), ", ".join(numbers), "list_numbers"


def longest_task(rng: random.Random, passage: Passage, lang: str, pool: list[str]) -> TaskResult:
    words = words_of(passage.text)
    longest = max(len(w) for w in words)
    winners = {w for w in words if len(w) == longest}
    if len(winners) != 1:
        return None
    return (
        rng.choice(LONGEST_QUESTION[lang]),
        LONGEST_ANSWER[lang].format(w=winners.pop()),
        "longest_word",
    )


def contains_task(rng: random.Random, passage: Passage, lang: str, pool: list[str]) -> TaskResult:
    present = keyword_candidates(passage)
    if not present:
        return None
    lowered = {w.lower() for w in words_of(passage.text)}
    positive = rng.random() < 0.5
    word = rng.choice(present) if positive else rng.choice(pool)
    if (word.lower() in lowered) != positive:
        return None
    yes, no = YES_NO[lang]
    return CONTAINS_QUESTION[lang].format(w=word), yes if positive else no, "contains_word"


def which_task(rng: random.Random, passage: Passage, lang: str, pool: list[str]) -> TaskResult:
    candidates = keyword_candidates(passage)
    unique = [
        w
        for w in candidates
        if sum(w.lower() in {x.lower() for x in words_of(s)} for s in passage.sentences) == 1
    ]
    if not unique:
        return None
    word = rng.choice(unique)
    sentence = next(
        s for s in passage.sentences if word.lower() in {x.lower() for x in words_of(s)}
    )
    return WHICH_QUESTION[lang].format(w=word), sentence, "which_sentence"


BUILDERS: tuple[Builder, ...] = (
    count_task,
    first_task,
    last_task,
    language_task,
    numbers_task,
    longest_task,
    contains_task,
    which_task,
)


def reading_row(rng: random.Random, passage: Passage, pool: list[str]) -> dict | None:
    lang = rng.choice(LANGUAGES)
    built = rng.choice(BUILDERS)(rng, passage, lang, pool)
    if built is None:
        return None
    question, answer, category = built
    prompt = f"{question}\n\n{passage.text}"
    return {
        "source": "text_reading",
        "group": f"reading_{lang}",
        "category": f"reading_{category}",
        "language": lang,
        "passage_language": passage.language,
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": answer},
        ],
    }


def build_reading_dataset(corpus_file: Path, per_language: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    passages = sample_passages(corpus_file, per_language, seed)
    pools = {
        language: [w for p in items for w in keyword_candidates(p)]
        for language, items in passages.items()
    }
    rows: list[dict] = []
    seen: set[str] = set()
    for language, items in passages.items():
        for passage in items:
            row = None
            for _ in range(6):
                row = reading_row(rng, passage, pools[language])
                if row is not None:
                    break
            if row is None:
                continue
            key = json.dumps(row["messages"], ensure_ascii=False)
            if key not in seen:
                seen.add(key)
                rows.append(row)
    rng.shuffle(rows)
    return rows
