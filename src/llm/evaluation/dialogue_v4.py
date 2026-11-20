from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from llm.evaluation.conversation import CASES, ConversationCase, exact
from llm.evaluation.metrics import normalize_words, repeated_ngram_ratio
from llm.tokenizer.bpe import Message


@dataclass(frozen=True)
class Locale:
    remember: tuple[str, ...]
    update: tuple[str, ...]
    recall: tuple[str, ...]
    reply: str
    instruction: tuple[str, ...]
    math: tuple[str, ...]


LOCALES = {
    "ru": Locale(
        ("Код для шкафчика {a}.", "Встреча в комнате {a}.", "На конверте номер {a}."),
        ("Исправление, код {b}.", "Теперь комната {b}.", "На конверте теперь номер {b}."),
        ("Какой действует код?", "Назови актуальный номер комнаты.", "Какой номер на конверте?"),
        "Понял, записал.",
        (
            "Ответь только числом {b}.",
            "Не пиши {a}, напиши только {b}.",
            "Повтори число {b} без пояснений.",
        ),
        (
            "Сколько будет {a} + {b}? Только число.",
            "Вычти {b} из {a}. Только число.",
            "Посчитай {a} × {b}. Только число.",
        ),
    ),
    "en": Locale(
        (
            "My locker code is {a}.",
            "The meeting is in room {a}.",
            "The tag on the envelope reads {a}.",
        ),
        (
            "Correction: the code is {b}.",
            "We moved to room {b}.",
            "The new number on the envelope is {b}.",
        ),
        (
            "What code should I use?",
            "Give the current room number.",
            "Which number is on the envelope now?",
        ),
        "Got it.",
        (
            "Respond with just the number {b}.",
            "Skip {a} and output only {b}.",
            "Repeat {b} without explanation.",
        ),
        (
            "What is {a} plus {b}? Just the number.",
            "Subtract {b} from {a}. Just the number.",
            "Multiply {a} by {b}. Just the number.",
        ),
    ),
    "kk": Locale(
        ("Шкафтың коды {a}.", "Кездесу {a}-бөлмеде өтеді.", "Конверттің нөмірі {a}."),
        (
            "Код өзгерді, жаңа код {b}.",
            "Енді {b}-бөлмеде кездесеміз.",
            "Конверттің жаңа нөмірі {b}.",
        ),
        (
            "Қазір қай кодты қолданамын?",
            "Қазір қай бөлмеде кездесеміз?",
            "Конверттің қазіргі нөмірі қандай?",
        ),
        "Түсіндім.",
        (
            "Тек {b} санын жаз.",
            "{a} санын жазба, тек {b} санын жаз.",
            "{b} санын түсіндірмей қайтала.",
        ),
        (
            "{a} мен {b} сандарын қос. Тек санды жаз.",
            "{a} санынан {b} санын азайт. Тек санды жаз.",
            "{a} санын {b} санына көбейт. Тек санды жаз.",
        ),
    ),
}


def objective_cases() -> Iterable[ConversationCase]:
    for language, locale in LOCALES.items():
        for index in range(42):
            a, b = 37 + index * 13, 11 + (index * 17) % 101
            kind = index % 3
            yield ConversationCase(
                f"v4-{language}-follow-{index}",
                "instruction",
                language,
                locale.instruction[kind].format(a=a, b=b),
                (exact(str(b)),),
            )
            value = (a + b, a - b, a * b)[kind]
            yield ConversationCase(
                f"v4-{language}-math-{index}",
                "math",
                language,
                locale.math[kind].format(a=a, b=b),
                (exact(str(value)),),
            )
            history = ((locale.remember[kind].format(a=a), locale.reply),)
            yield ConversationCase(
                f"v4-{language}-memory-{index}",
                "context",
                language,
                locale.recall[kind],
                (exact(str(a)),),
                history,
            )
            yield ConversationCase(
                f"v4-{language}-update-{index}",
                "multi_turn",
                language,
                locale.recall[kind],
                (exact(str(b)),),
                (*history, (locale.update[kind].format(b=b), locale.reply)),
            )


def benchmark_cases() -> list[ConversationCase]:
    return [*CASES, *objective_cases()]


def loop_rate(response: str) -> float:
    return repeated_ngram_ratio(normalize_words(response))


PROBES = {
    "ru_smalltalk": [Message("user", "Как дела?")],
    "ru_typo": [Message("user", "как дила")],
    "en_smalltalk": [Message("user", "How are you doing?")],
    "kk_smalltalk": [Message("user", "Қалың қалай?")],
    "ru_greet": [Message("user", "Привет!")],
    "ru_identity": [Message("user", "Что ты умеешь?")],
    "ru_developer": [Message("user", "Кто тебя разработал?")],
    "ru_help": [Message("user", "Чем ты можешь помочь?")],
    "ru_explain": [Message("user", "Объясни простыми словами, что такое нейросеть.")],
    "ru_programming": [Message("user", "Чем Java отличается от Python?")],
    "ru_sorting": [Message("user", "Напиши функцию сортировки на Python.")],
    "ru_arithmetic": [Message("user", "Сколько будет 125 * 48?")],
    "ru_context": [
        Message("user", "Меня зовут Аскар."),
        Message("assistant", "Приятно познакомиться, Аскар."),
        Message("user", "Как меня зовут?"),
    ],
    "ru_translation": [
        Message("user", "Закончи фразу: Снег"),
        Message("assistant", "Снег тает на солнце."),
        Message("user", "Переведи предыдущий ответ на казахский."),
    ],
    "ru_continue": [
        Message("user", "Начни рассказ о прогулке."),
        Message("assistant", "Лена вышла из дома и услышала шаги."),
        Message("user", "Продолжи предыдущую мысль."),
    ],
    "ru_correction": [
        Message("user", "Сколько лап у кошки?"),
        Message("assistant", "У кошки шесть лап."),
        Message("user", "Объясни, почему твой ответ неправильный."),
    ],
}
