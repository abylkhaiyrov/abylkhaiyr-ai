from __future__ import annotations

import random
from collections.abc import Sequence

from llm.data import lexicon as lx
from llm.data.identity import IdentityFacts

Row = dict
Turn = tuple[str, str] | tuple[str, str, bool]

RECALL_QUESTIONS = {
    "en": (
        "What did I ask you earlier?",
        "What was my previous question?",
        "Remind me what I asked before.",
    ),
    "ru": (
        "Что я спрашивал ранее?",
        "Какой был мой предыдущий вопрос?",
        "Напомни, о чём я спрашивал.",
    ),
    "kk": (
        "Мен бұрын не сұрадым?",
        "Менің алдыңғы сұрағым қандай болды?",
        "Бұрын не сұрағанымды еске сал.",
    ),
}
RECALL_ANSWER = {
    "en": "You asked: “{q}”",
    "ru": "Вы спросили: «{q}»",
    "kk": "Сіз мынаны сұрадыңыз: «{q}»",
}
CONTINUE_FIRST = {
    "en": "Continue the sequence: {shown}",
    "ru": "Продолжи последовательность: {shown}",
    "kk": "Тізбекті жалғастыр: {shown}",
}
CONTINUE_AGAIN = {"en": "Continue.", "ru": "Продолжай.", "kk": "Жалғастыр."}
FIX_QUESTIONS = {
    "en": "What is {a} {symbol} {b}?",
    "ru": "Сколько будет {a} {symbol} {b}?",
    "kk": "{a} {symbol} {b} неше болады?",
}
FIX_COMPLAINTS = {
    "en": (
        "That is wrong, please correct your answer.",
        "I think you made a mistake. Fix it.",
        "Check your calculation again.",
    ),
    "ru": (
        "Это неверно, исправь свой ответ.",
        "Кажется, ты ошибся. Исправь.",
        "Проверь своё вычисление ещё раз.",
    ),
    "kk": (
        "Бұл дұрыс емес, жауабыңды түзет.",
        "Сен қателескен сияқтысың. Түзет.",
        "Есептеуді қайта тексер.",
    ),
}
FIX_ANSWERS = {
    "en": "You are right, I made a mistake. {a} {symbol} {b} = {result}.",
    "ru": "Вы правы, я ошибся. {a} {symbol} {b} = {result}.",
    "kk": "Дұрыс айтасыз, мен қателестім. {a} {symbol} {b} = {result}.",
}
FIX_WRONG = {
    "en": "{a} {symbol} {b} = {wrong}.",
    "ru": "{a} {symbol} {b} = {wrong}.",
    "kk": "{a} {symbol} {b} = {wrong}.",
}
OPERATIONS = (("+", lambda a, b: a + b), ("-", lambda a, b: a - b), ("×", lambda a, b: a * b))
NOISE_ALPHABET = {
    "en": "abcdefghijklmnopqrstuvwxyz",
    "ru": "абвгдежзийклмнопрстуфхцчшщыэюя",
    "kk": "абвгдежзийклмнопрстуфхцчшщыэюяәғқңөұүһі",
}
FRUITS = (
    (lx.Term("banana", "банан", "банан"), "yellow"),
    (lx.Term("lemon", "лимон", "лимон"), "yellow"),
    (lx.Term("cherry", "вишня", "шие"), "red"),
    (lx.Term("strawberry", "клубника", "құлпынай"), "red"),
    (lx.Term("pear", "груша", "алмұрт"), "green"),
)
COLOR_TERMS = {term.en: term for term in lx.COLORS}
NEGATION_QUESTION = {
    "en": "Name a fruit that is not {color}.",
    "ru": "Назови фрукт или ягоду, которые не бывают цвета «{color}».",
    "kk": "«{color}» түсті болмайтын жеміс немесе жидек ата.",
}
NEGATION_ANSWER = {
    "en": "{fruit} is not {color}. It is {actual}.",
    "ru": "{fruit} — не «{color}», а «{actual}».",
    "kk": "{fruit} «{color}» емес, ол «{actual}».",
}
ACQUAINT_OPENERS = {
    "en": ("Let's get acquainted.", "Can we get to know each other?"),
    "ru": ("Давай познакомимся.", "Давайте знакомиться."),
    "kk": ("Танысайық.", "Бір-бірімізбен танысып алайық."),
}
ACQUAINT_REPLY = {
    "en": "Nice idea! I am {name}, a small language model. What is your name?",
    "ru": "Отличная идея! Я {name}, небольшая языковая модель. А как зовут вас?",
    "kk": "Жақсы ой! Мен {name} — шағын тіл моделімін. Сіздің атыңыз кім?",
}
ACQUAINT_INTRO = {"en": "I am {person}.", "ru": "Я {person}.", "kk": "Мен {person}."}
ACQUAINT_CLOSE = {
    "en": "Nice to meet you, {person}! How can I help you?",
    "ru": "Приятно познакомиться, {person}! Чем могу помочь?",
    "kk": "Танысқаныма қуаныштымын, {person}! Қалай көмектесе аламын?",
}


def conversation_row(category: str, language: str, turns: Sequence[Turn]) -> Row:
    messages: list[dict] = []
    for turn in turns:
        messages.append({"role": "user", "content": turn[0]})
        assistant: dict = {"role": "assistant", "content": turn[1]}
        if len(turn) == 3 and turn[2] is False:
            assistant["supervised"] = False
        messages.append(assistant)
    return {
        "source": f"synthetic_{category}",
        "group": "dialog",
        "category": category,
        "language": language,
        "messages": messages,
    }


def pick_language(rng: random.Random) -> str:
    return rng.choice(lx.LANGUAGES)


def recall_question(rng: random.Random, facts: IdentityFacts) -> Row:
    language = pick_language(rng)
    fact = rng.choice(lx.FACTS)
    question = fact.question.in_language(language)
    turns = [
        (question, fact.answer.in_language(language)),
        (rng.choice(RECALL_QUESTIONS[language]), RECALL_ANSWER[language].format(q=question)),
    ]
    return conversation_row("recall_question", language, turns)


def continue_sequence(rng: random.Random, facts: IdentityFacts) -> Row:
    language = pick_language(rng)
    start, step = rng.randint(1, 30), rng.randint(2, 9)
    values = [start + step * i for i in range(10)]
    shown = ", ".join(map(str, values[:4]))
    first = ", ".join(map(str, values[4:7]))
    second = ", ".join(map(str, values[7:10]))
    turns = [
        (CONTINUE_FIRST[language].format(shown=shown), first + "."),
        (CONTINUE_AGAIN[language], second + "."),
    ]
    return conversation_row("continue_sequence", language, turns)


def fix_mistake(rng: random.Random, facts: IdentityFacts) -> Row:
    language = pick_language(rng)
    symbol, operation = rng.choice(OPERATIONS)
    a, b = rng.randint(3, 40), rng.randint(3, 40)
    result = operation(a, b)
    wrong = result + rng.choice((-10, -2, -1, 1, 2, 10))
    values = {"a": a, "b": b, "symbol": symbol}
    turns: list[Turn] = [
        (
            FIX_QUESTIONS[language].format(**values),
            FIX_WRONG[language].format(**values, wrong=wrong),
            False,
        ),
        (
            rng.choice(FIX_COMPLAINTS[language]),
            FIX_ANSWERS[language].format(**values, result=result),
        ),
    ]
    return conversation_row("fix_mistake", language, turns)


def add_typos(text: str, language: str, rng: random.Random) -> str:
    words = text.split(" ")
    candidates = [i for i, word in enumerate(words) if len(word) > 3 and word.isalpha()]
    for index in rng.sample(candidates, min(2, len(candidates))):
        word = words[index]
        position = rng.randrange(1, len(word) - 1)
        kind = rng.choice(("swap", "drop", "double", "replace"))
        if kind == "swap":
            word = word[:position] + word[position + 1] + word[position] + word[position + 2 :]
        elif kind == "drop":
            word = word[:position] + word[position + 1 :]
        elif kind == "double":
            word = word[:position] + word[position] + word[position:]
        else:
            word = word[:position] + rng.choice(NOISE_ALPHABET[language]) + word[position + 1 :]
        words[index] = word
    return " ".join(words)


def typo_fact(rng: random.Random, facts: IdentityFacts) -> Row | None:
    language = pick_language(rng)
    fact = rng.choice(lx.FACTS)
    clean = fact.question.in_language(language)
    noisy = add_typos(clean, language, rng)
    if noisy == clean:
        return None
    return conversation_row("typo_fact", language, [(noisy, fact.answer.in_language(language))])


def negation_fruit(rng: random.Random, facts: IdentityFacts) -> Row:
    language = pick_language(rng)
    forbidden, _ = rng.choice([(color, c) for color in COLOR_TERMS for c in (color,)])
    options = [(fruit, actual) for fruit, actual in FRUITS if actual != forbidden]
    fruit, actual = rng.choice(options)
    color = COLOR_TERMS[forbidden].in_language(language)
    shown = COLOR_TERMS[actual].in_language(language)
    question = NEGATION_QUESTION[language].format(color=color)
    answer = NEGATION_ANSWER[language].format(
        fruit=fruit.in_language(language).capitalize(), color=color, actual=shown
    )
    return conversation_row("negation_fruit", language, [(question, answer)])


def get_acquainted(rng: random.Random, facts: IdentityFacts) -> Row:
    language = pick_language(rng)
    person = rng.choice(lx.PEOPLE[language])
    turns = [
        (
            rng.choice(ACQUAINT_OPENERS[language]),
            ACQUAINT_REPLY[language].format(name=facts.name),
        ),
        (
            ACQUAINT_INTRO[language].format(person=person),
            ACQUAINT_CLOSE[language].format(person=person),
        ),
    ]
    return conversation_row("get_acquainted", language, turns)
