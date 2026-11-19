from __future__ import annotations

import random
from dataclasses import dataclass, field

import sympy

ARITHMETIC_OPERATORS = ("+", "-", "*")
ARITHMETIC_MAX_OPERAND = 99
INSTRUCTION_WORDS = ("apple", "river", "winter", "garden")


@dataclass(frozen=True)
class PromptCase:
    prompt: str
    reference: str


@dataclass(frozen=True)
class ConstraintCase:
    prompt: str
    constraint: str
    argument: str = ""


@dataclass(frozen=True)
class CodeCase:
    prompt: str
    tests: str
    function_name: str = ""
    extras: dict[str, str] = field(default_factory=dict)


def arithmetic_cases(count: int, seed: int) -> list[PromptCase]:
    rng = random.Random(seed)
    cases = []
    for _ in range(count):
        a, b = rng.randint(1, ARITHMETIC_MAX_OPERAND), rng.randint(1, ARITHMETIC_MAX_OPERAND)
        operator = rng.choice(ARITHMETIC_OPERATORS)
        expression = f"{a} {operator} {b}"
        cases.append(PromptCase(f"What is {expression}?", str(sympy.sympify(expression))))
    return cases


FACTUAL_CASES = [
    PromptCase("What is the capital of France?", "paris"),
    PromptCase("What is the capital of Kazakhstan?", "astana"),
    PromptCase("Столица России?", "москва"),
    PromptCase("Қазақстанның астанасы қай қала?", "астана"),
    PromptCase("How many days are in a week?", "7"),
    PromptCase("What planet do we live on?", "earth"),
    PromptCase("What is the chemical symbol for water?", "h2o"),
    PromptCase("Which language is mostly spoken in Brazil?", "portuguese"),
]

MULTILINGUAL_PROMPTS = {
    "en": "The history of the city begins",
    "ru": "История этого города начинается",
    "kk": "Бұл қаланың тарихы басталады",
}


def instruction_cases() -> list[ConstraintCase]:
    cases = [
        ConstraintCase(f"Write a sentence that contains the word '{w}'.", "contains", w)
        for w in INSTRUCTION_WORDS
    ]
    cases += [
        ConstraintCase("Write a short sentence in lowercase letters only.", "lowercase"),
        ConstraintCase("Answer in at most 5 words: what colour is the sky?", "max_words", "5"),
        ConstraintCase("Write a sentence and finish it with the word 'done'.", "ends_with", "done"),
        ConstraintCase("Reply with a single word: yes or no. Is fire hot?", "max_words", "1"),
    ]
    return cases


def check_constraint(case: ConstraintCase, response: str) -> bool:
    text = response.strip()
    if not text:
        return False
    if case.constraint == "contains":
        return case.argument in text.lower()
    if case.constraint == "lowercase":
        return text == text.lower()
    if case.constraint == "max_words":
        return len(text.split()) <= int(case.argument)
    if case.constraint == "ends_with":
        return text.rstrip(" .!").lower().endswith(case.argument)
    return False


@dataclass(frozen=True)
class MultiTurnCase:
    turns: tuple[tuple[str, str], ...]
    final_prompt: str
    expected_substring: str
    language: str


@dataclass(frozen=True)
class LogicCase:
    prompt: str
    expected: str
    language: str


MULTITURN_CASES = [
    MultiTurnCase(
        (("My name is Arman and I live in Almaty.", "Nice to meet you, Arman."),),
        "What is my name?",
        "arman",
        "en",
    ),
    MultiTurnCase(
        (("Меня зовут Айгуль, мой любимый цвет — синий.", "Приятно познакомиться, Айгуль."),),
        "Какой мой любимый цвет?",
        "син",
        "ru",
    ),
    MultiTurnCase(
        (("Менің атым Данияр, мен Астанада тұрамын.", "Танысқаныма қуаныштымын, Данияр."),),
        "Мен қай қалада тұрамын?",
        "астана",
        "kk",
    ),
    MultiTurnCase(
        (("Remember the code 7429 for later.", "Got it, I will remember 7429."),),
        "What was the code I asked you to remember?",
        "7429",
        "en",
    ),
]

CONTEXT_CASES = [
    PromptCase(
        "Context: The library opens at 9 AM and closes at 6 PM on weekdays. Question: When does the library close?",
        "6",
    ),
    PromptCase(
        "Контекст: Завод в Караганде производит 450 деталей в день. Вопрос: Сколько деталей производит завод в день?",
        "450",
    ),
    PromptCase(
        "Мәтін: Абай университеті 1928 жылы Алматыда ашылды. Сұрақ: Университет қай жылы ашылды?",
        "1928",
    ),
    PromptCase(
        "Context: Project Aurora uses Python and PostgreSQL for its backend. Question: Which database does Project Aurora use?",
        "postgresql",
    ),
]

LOGIC_CASES = [
    LogicCase(
        "Anna is taller than Bota. Bota is taller than Dana. Who is the shortest?", "Dana", "en"
    ),
    LogicCase(
        "All birds have feathers. A sparrow is a bird. Does a sparrow have feathers?", "yes", "en"
    ),
    LogicCase("Асет старше Марата. Марат старше Тимура. Кто самый младший?", "Тимур", "ru"),
    LogicCase("Айдана Сауледен биік. Сауле Мадинадан биік. Ең аласа кім?", "Мадина", "kk"),
]

CODE_CASES = [
    CodeCase(
        "Write a Python function second_largest(xs) that returns the second largest distinct number of a list.",
        "assert second_largest([1, 5, 3, 5]) == 3\nassert second_largest([9, 2]) == 2",
    ),
    CodeCase(
        "Write a Python function digit_sum(n) that returns the sum of the decimal digits of a non-negative integer.",
        "assert digit_sum(123) == 6\nassert digit_sum(0) == 0",
    ),
    CodeCase(
        "Write a Python function remove_duplicates(xs) that removes duplicates from a list keeping the first occurrences in order.",
        "assert remove_duplicates([3, 1, 3, 2, 1]) == [3, 1, 2]",
    ),
    CodeCase(
        "Write a Python function is_sorted(xs) that returns True if the list is in non-decreasing order.",
        "assert is_sorted([1, 2, 2, 5]) and not is_sorted([2, 1])",
    ),
    CodeCase(
        "Write a Python function celsius_to_fahrenheit(c) that converts degrees Celsius to Fahrenheit.",
        "assert celsius_to_fahrenheit(0) == 32\nassert celsius_to_fahrenheit(100) == 212",
    ),
    CodeCase(
        "Write a Python function merge_sorted(a, b) that merges two sorted lists into one sorted list.",
        "assert merge_sorted([1, 4], [2, 3, 9]) == [1, 2, 3, 4, 9]",
    ),
    CodeCase(
        "Write a Python function capitalize_words(s) that capitalizes the first letter of every word of a string.",
        "assert capitalize_words('hello big world') == 'Hello Big World'",
    ),
    CodeCase(
        "Write a Python function count_char(s, ch) that returns how many times the character ch occurs in s.",
        "assert count_char('banana', 'a') == 3\nassert count_char('', 'x') == 0",
    ),
]
