from __future__ import annotations

import json
import re
from collections import defaultdict
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from llm.data.cleaning import detect_language
from llm.evaluation.metrics import repeated_ngram_ratio
from llm.inference.engine import InferenceEngine
from llm.inference.sampling import SamplingParams
from llm.tokenizer.bpe import Message

WORD = re.compile(r"[^\W\d_]+", re.UNICODE)
NUMBER = re.compile(r"-?\d+(?:[.,]\d+)?")
SENTENCE_END = re.compile(r"[.!?…]+(?:\s+|$)")
KAZAKH_LETTERS = frozenset("әғқңөұүһі")
MIN_WELLFORMED_WORDS = 3
ECHO_NGRAM = 3
ECHO_THRESHOLD = 0.6


@dataclass(frozen=True)
class Check:
    kind: str
    value: tuple[str, ...] = ()
    number: int = 0


@dataclass(frozen=True)
class ConversationCase:
    id: str
    category: str
    language: str
    prompt: str
    checks: tuple[Check, ...]
    history: tuple[tuple[str, str], ...] = ()

    @property
    def unanswerable(self) -> bool:
        return self.category == "unknown" or self.id.startswith("context_unknown")

    def messages(self, system_prompt: str | None = None) -> list[Message]:
        messages = [Message("system", system_prompt)] if system_prompt else []
        for user, assistant in self.history:
            messages.extend((Message("user", user), Message("assistant", assistant)))
        messages.append(Message("user", self.prompt))
        return messages


def any_of(*values: str) -> Check:
    return Check("any", tuple(values))


def exact(*values: str) -> Check:
    return Check("exact", tuple(values))


def all_of(*values: str) -> Check:
    return Check("all", tuple(values))


def none_of(*values: str) -> Check:
    return Check("none", tuple(values))


def number(value: str) -> Check:
    return Check("number", (value,))


def max_words(limit: int) -> Check:
    return Check("max_words", number=limit)


def min_items(count: int) -> Check:
    return Check("min_items", number=count)


def sentences(count: int) -> Check:
    return Check("sentences", number=count)


def script(language: str) -> Check:
    return Check("script", (language,))


def case(
    case_id: str,
    category: str,
    language: str,
    prompt: str,
    *checks: Check,
    history: tuple[tuple[str, str], ...] = (),
) -> ConversationCase:
    return ConversationCase(case_id, category, language, prompt, checks, history)


CASES: tuple[ConversationCase, ...] = (
    case(
        "greet_en",
        "greeting",
        "en",
        "Hi there, nice to meet you!",
        any_of("hello", "hi", "help", "nice"),
        script("en"),
    ),
    case(
        "greet_ru",
        "greeting",
        "ru",
        "Привет! Как твои дела?",
        any_of("привет", "здравствуй", "помочь", "дела"),
        script("ru"),
    ),
    case(
        "greet_kk",
        "greeting",
        "kk",
        "Сәлем, қалайсың?",
        any_of("сәлем", "сәлеметсіз", "көмектес", "жақсы"),
        script("kk"),
    ),
    case(
        "id_who_en",
        "identity",
        "en",
        "Could you tell me who you are?",
        any_of("abylkhaiyr"),
        script("en"),
    ),
    case(
        "id_name_ru",
        "identity",
        "ru",
        "Как к тебе можно обращаться?",
        any_of("abylkhaiyr"),
        script("ru"),
    ),
    case(
        "id_who_kk",
        "identity",
        "kk",
        "Өзің кімсің, таныстырып жіберші?",
        any_of("abylkhaiyr"),
        script("kk"),
    ),
    case(
        "id_abilities_en",
        "identity",
        "en",
        "What are you able to help me with?",
        any_of("chat", "answer", "explain", "translate", "question", "help"),
        script("en"),
    ),
    case(
        "id_abilities_ru",
        "identity",
        "ru",
        "Расскажи, что ты умеешь делать.",
        any_of("отвечать", "объяснять", "перевод", "помога", "общаться", "вопрос"),
        script("ru"),
    ),
    case(
        "id_limits_en",
        "identity",
        "en",
        "Is there anything you cannot do?",
        any_of("mistake", "wrong", "limited", "cannot", "can't", "small", "error"),
        script("en"),
    ),
    case(
        "explain_magnet_en",
        "explain",
        "en",
        "What is a magnet?",
        any_of("attract", "iron", "metal", "magnetic", "field"),
        script("en"),
    ),
    case(
        "explain_rainbow_ru",
        "explain",
        "ru",
        "Что такое радуга?",
        any_of("свет", "дожд", "капл", "небе", "цвет"),
        script("ru"),
    ),
    case(
        "explain_bacteria_kk",
        "explain",
        "kk",
        "Бактерия деген не?",
        any_of("организм", "жасуша", "ұсақ", "микро"),
        script("kk"),
    ),
    case(
        "explain_database_en",
        "explain",
        "en",
        "Explain in simple words what a database is.",
        any_of("data", "store", "information", "organized"),
        script("en"),
    ),
    case(
        "explain_fraction_ru",
        "explain",
        "ru",
        "Объясни, что такое дробь.",
        any_of("част", "числ", "знаменател", "делен", "доля", "доли"),
        script("ru"),
    ),
    case(
        "context_city_en",
        "context",
        "en",
        "Bakyt is a nurse from Aktau. Madina is a pilot from Oral. "
        "Which city is Madina from? Answer using the text.",
        any_of("oral"),
        none_of("aktau"),
    ),
    case(
        "context_city_ru",
        "context",
        "ru",
        "Бакыт — пекарь из Актау. Мадина — пилот из Орала. Откуда Бакыт? Ответь по тексту.",
        any_of("актау"),
        none_of("орал "),
    ),
    case(
        "context_city_kk",
        "context",
        "kk",
        "Бақыт Ақтауда тұрады, ал Мадина Оралда тұрады. Мадина қайда тұрады?",
        any_of("орал"),
        none_of("ақтау"),
    ),
    case(
        "context_time_en",
        "context",
        "en",
        "The shop opens at 9 and closes at 18. When does the shop close?",
        any_of("18", "6 pm", "6 p.m", "six"),
    ),
    case(
        "context_books_ru",
        "context",
        "ru",
        "В библиотеке 120 книг по физике и 80 книг по химии. Сколько книг по химии?",
        number("80"),
    ),
    case(
        "context_unknown_en",
        "context",
        "en",
        "Bakyt lives in Aktau and works as a nurse. What is Bakyt's phone number?",
        any_of("not", "no ", "cannot", "don't", "doesn't", "unknown", "does not"),
        none_of("+7"),
    ),
    case(
        "context_unknown_ru",
        "context",
        "ru",
        "Мадина работает пилотом в Орале. Какой у неё любимый фильм? Ответь по тексту.",
        any_of("не сказано", "не указан", "не знаю", "нет информации", "не говорится"),
    ),
    case("math_add_en", "math", "en", "What is 23 + 19?", number("42")),
    case("math_mul_ru", "math", "ru", "Сколько будет 15 умножить на 4?", number("60")),
    case("math_add_kk", "math", "kk", "12 + 30 неше болады?", number("42")),
    case("math_div_en", "math", "en", "What is 144 divided by 12?", number("12")),
    case("math_eq_ru", "math", "ru", "Реши уравнение: 3x + 6 = 21. Чему равен x?", number("5")),
    case(
        "math_word_en",
        "math",
        "en",
        "A box has 8 apples. You add 7 more apples. How many apples are in the box now?",
        number("15"),
    ),
    case(
        "translate_tooth_en_ru",
        "translation",
        "en",
        "Translate into Russian: tooth",
        any_of("зуб"),
    ),
    case(
        "translate_sea_ru_en",
        "translation",
        "ru",
        "Переведи на английский слово: море",
        any_of("sea"),
    ),
    case(
        "translate_salt_en_kk",
        "translation",
        "en",
        "Translate into Kazakh: salt",
        any_of("тұз"),
    ),
    case(
        "translate_sugar_kk_ru",
        "translation",
        "kk",
        "«Қант» сөзін орыс тіліне аудар.",
        any_of("сахар"),
    ),
    case(
        "translate_street_ru_kk",
        "translation",
        "ru",
        "Переведи на казахский слово: улица",
        any_of("көше"),
    ),
    case(
        "translate_ice_en_ru",
        "translation",
        "en",
        "How do you say ice in Russian?",
        any_of("лёд", "лед"),
    ),
    case(
        "follow_upper_en",
        "instruction",
        "en",
        "Write the word hello in capital letters.",
        exact("HELLO"),
    ),
    case(
        "follow_colors_ru",
        "instruction",
        "ru",
        "Назови три цвета.",
        min_items(3),
        script("ru"),
    ),
    case(
        "follow_oneword_en",
        "instruction",
        "en",
        "Answer with one word: what color is grass?",
        max_words(3),
        any_of("green"),
    ),
    case(
        "follow_oneword_ru",
        "instruction",
        "ru",
        "Ответь одним словом: какого цвета снег?",
        max_words(3),
        any_of("бел"),
    ),
    case(
        "follow_oneword_kk",
        "instruction",
        "kk",
        "Бір сөзбен жауап бер: аспан қандай түсті?",
        max_words(3),
        any_of("көк"),
    ),
    case(
        "follow_sentences_en",
        "instruction",
        "en",
        "Write exactly two sentences about the sea.",
        sentences(2),
        script("en"),
    ),
    case(
        "clarify_translate_en",
        "clarify",
        "en",
        "Translate this.",
        any_of("text", "please", "send", "which", "what", "provide"),
        script("en"),
    ),
    case(
        "clarify_fix_ru",
        "clarify",
        "ru",
        "Исправь ошибки.",
        any_of("пришл", "текст", "уточн", "какой", "что именно", "код"),
        script("ru"),
    ),
    case(
        "unknown_weather_en",
        "unknown",
        "en",
        "What will the weather be like in Paris next week?",
        any_of("don't", "do not", "cannot", "can't", "no access", "unable", "not able"),
        script("en"),
    ),
    case(
        "unknown_rate_ru",
        "unknown",
        "ru",
        "Какой сейчас курс евро к тенге?",
        any_of("не знаю", "нет доступа", "не могу", "недоступ", "не имею"),
        script("ru"),
    ),
    case(
        "multi_name_en",
        "multi_turn",
        "en",
        "Do you remember my name?",
        any_of("bolat"),
        history=(("My name is Bolat.", "Nice to meet you, Bolat!"),),
    ),
    case(
        "multi_tea_ru",
        "multi_turn",
        "ru",
        "Что я люблю пить?",
        any_of("чай"),
        history=(("Я люблю чай с мёдом.", "Звучит вкусно!"),),
    ),
    case(
        "multi_color_kk",
        "multi_turn",
        "kk",
        "Менің сүйікті түсім қандай?",
        any_of("жасыл"),
        history=(("Менің сүйікті түсім — жасыл.", "Жақсы түс екен!"),),
    ),
    case(
        "multi_sub_en",
        "multi_turn",
        "en",
        "Now subtract 5 from it.",
        number("10"),
        history=(("What is 9 + 6?", "9 + 6 = 15."),),
    ),
    case(
        "multi_mul_ru",
        "multi_turn",
        "ru",
        "Теперь умножь результат на 2.",
        number("30"),
        history=(("Сколько будет 7 + 8?", "7 + 8 = 15."),),
    ),
    case(
        "logic_tall_en",
        "logic",
        "en",
        "Anna is taller than Ben. Ben is taller than Carl. Who is the shortest?",
        any_of("carl"),
    ),
    case(
        "logic_age_ru",
        "logic",
        "ru",
        "Аня старше Бори, а Боря старше Васи. Кто самый младший?",
        any_of("вас"),
    ),
    case(
        "logic_syllogism_en",
        "logic",
        "en",
        "All cats are animals. Tom is a cat. Is Tom an animal? Answer yes or no.",
        any_of("yes"),
    ),
    case(
        "code_add_en",
        "code",
        "en",
        "Write a Python function add(a, b) that returns the sum.",
        all_of("def add", "return"),
    ),
    case(
        "code_square_ru",
        "code",
        "ru",
        "Напиши на Python функцию, которая возвращает квадрат числа.",
        all_of("def", "return"),
    ),
)


def benchmark_prompts() -> list[str]:
    texts: list[str] = []
    for item in CASES:
        texts.extend(user for user, _ in item.history)
        texts.append(item.prompt)
    return texts


def dominant_script(text: str) -> str | None:
    letters = [ch for ch in text if ch.isalpha()]
    if not letters:
        return None
    cyrillic = sum("Ѐ" <= ch <= "ӿ" for ch in letters)
    return "cyrillic" if cyrillic / len(letters) > 0.5 else "latin"


def word_count(text: str) -> int:
    return len(WORD.findall(text))


def item_count(text: str) -> int:
    lines = [line for line in text.splitlines() if line.strip()]
    if len(lines) > 1:
        return len(lines)
    return len([part for part in re.split(r"[,;]| и | and | және ", text) if part.strip()])


def sentence_count(text: str) -> int:
    return len([part for part in SENTENCE_END.split(text.strip()) if part.strip()])


def numbers_in(text: str) -> set[str]:
    return {item.replace(",", ".").rstrip("0").rstrip(".") for item in NUMBER.findall(text)}


def evaluate_check(check: Check, response: str) -> bool:
    lowered = response.lower()
    if check.kind == "any":
        return any(value.lower() in lowered for value in check.value)
    if check.kind == "exact":
        return any(value in response for value in check.value)
    if check.kind == "all":
        return all(value.lower() in lowered for value in check.value)
    if check.kind == "none":
        return not any(value.lower() in lowered for value in check.value)
    if check.kind == "number":
        return check.value[0] in numbers_in(response)
    if check.kind == "max_words":
        return 0 < word_count(response) <= check.number
    if check.kind == "min_items":
        return item_count(response) >= check.number
    if check.kind == "sentences":
        return sentence_count(response) == check.number
    if check.kind == "script":
        expected = "latin" if check.value[0] == "en" else "cyrillic"
        found = dominant_script(response)
        return found == expected and (
            check.value[0] != "kk"
            or any(ch in KAZAKH_LETTERS for ch in lowered)
            or len(lowered) < 25
        )
    raise ValueError(f"unknown check kind: {check.kind}")


@dataclass
class WordList:
    words: frozenset[str]

    @classmethod
    def from_texts(cls, texts: Iterable[str], min_count: int = 2) -> WordList:
        counts: dict[str, int] = defaultdict(int)
        for text in texts:
            for word in WORD.findall(text.lower()):
                counts[word] += 1
        return cls(frozenset(word for word, count in counts.items() if count >= min_count))

    @classmethod
    def from_jsonl(cls, path: Path, limit: int = 50_000) -> WordList:
        def texts() -> Iterable[str]:
            with path.open(encoding="utf-8") as handle:
                for index, line in enumerate(handle):
                    if index >= limit:
                        return
                    yield json.loads(line)["text"]

        return cls.from_texts(texts())

    def rate(self, text: str) -> float | None:
        words = WORD.findall(text.lower())
        if len(words) < MIN_WELLFORMED_WORDS:
            return None
        return sum(word in self.words for word in words) / len(words)


@dataclass
class CaseResult:
    case: ConversationCase
    response: str
    finished: bool
    passed: bool
    checks_passed: int
    repetition: float
    wellformed: float | None
    detected_language: str


@dataclass
class BenchmarkReport:
    metrics: dict[str, float]
    results: list[CaseResult] = field(default_factory=list)

    def samples(self) -> list[dict[str, str]]:
        return [
            {
                "id": item.case.id,
                "category": item.case.category,
                "prompt": item.case.prompt,
                "response": item.response,
                "passed": str(item.passed),
            }
            for item in self.results
        ]


Responder = Callable[[ConversationCase], tuple[str, bool]]


def engine_responder(
    engine: InferenceEngine, params: SamplingParams, system_prompt: str | None = None
) -> Responder:
    def respond(item: ConversationCase) -> tuple[str, bool]:
        generation = engine.chat(item.messages(system_prompt), params)
        return generation.text.strip(), generation.finish_reason == "stop"

    return respond


def score_case(
    item: ConversationCase, response: str, finished: bool, words: WordList | None
) -> CaseResult:
    passed = [evaluate_check(check, response) for check in item.checks]
    tokens = WORD.findall(response.lower())
    return CaseResult(
        item,
        response,
        finished,
        bool(response) and all(passed),
        sum(passed),
        repeated_ngram_ratio(tokens, 3) if len(tokens) >= 6 else 0.0,
        words.rate(response) if words else None,
        detect_language(response),
    )


def rate(values: Iterable[bool]) -> float:
    items = list(values)
    return float(np.mean(items)) if items else 0.0


def echo_ratio(prompt: str, response: str) -> float:
    prompt_words = [w.lower() for w in WORD.findall(prompt)]
    response_words = [w.lower() for w in WORD.findall(response)]
    size = ECHO_NGRAM
    if len(response_words) < size:
        return 0.0
    prompt_grams = {tuple(prompt_words[i : i + size]) for i in range(len(prompt_words) - size + 1)}
    grams = [tuple(response_words[i : i + size]) for i in range(len(response_words) - size + 1)]
    return sum(gram in prompt_grams for gram in grams) / len(grams)


def summarize(results: list[CaseResult]) -> dict[str, float]:
    metrics: dict[str, float] = {
        "conv_pass_rate": rate(item.passed for item in results),
        "conv_non_empty_rate": rate(bool(item.response) for item in results),
        "conv_completion_rate": rate(item.finished for item in results),
        "conv_repetition_ratio": float(np.mean([item.repetition for item in results])),
        "conv_check_pass_rate": float(
            np.sum([item.checks_passed for item in results])
            / max(np.sum([len(item.case.checks) for item in results]), 1)
        ),
    }
    metrics["conv_prompt_echo_rate"] = rate(
        echo_ratio(item.case.prompt, item.response) >= ECHO_THRESHOLD for item in results
    )
    unanswerable = [item for item in results if item.case.unanswerable]
    metrics["conv_hallucination_rate"] = (
        rate(not item.passed for item in unanswerable) if unanswerable else 0.0
    )
    scripted = [
        item for item in results if any(check.kind == "script" for check in item.case.checks)
    ]
    metrics["conv_language_stability"] = rate(
        evaluate_check(Check("script", (item.case.language,)), item.response) for item in scripted
    )
    wellformed = [item.wellformed for item in results if item.wellformed is not None]
    metrics["conv_wellformed_word_rate"] = float(np.mean(wellformed)) if wellformed else 0.0
    for key, selector in (
        ("category", lambda i: i.case.category),
        ("lang", lambda i: i.case.language),
    ):
        groups: dict[str, list[bool]] = defaultdict(list)
        for item in results:
            groups[selector(item)].append(item.passed)
        metrics.update(
            {f"conv_{key}_{name}": rate(values) for name, values in sorted(groups.items())}
        )
    return metrics


def run_conversation_benchmark(
    respond: Responder,
    words: WordList | None = None,
    cases: Iterable[ConversationCase] = CASES,
) -> BenchmarkReport:
    results = []
    for item in cases:
        response, finished = respond(item)
        results.append(score_case(item, response, finished, words))
    return BenchmarkReport(summarize(results), results)
