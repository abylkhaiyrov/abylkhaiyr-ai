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


CODE_CASES = [
    CodeCase(
        "Write a Python function add(a, b) that returns the sum of a and b.",
        "assert add(2, 3) == 5\nassert add(-1, 1) == 0",
    ),
    CodeCase(
        "Write a Python function is_even(n) that returns True if n is even.",
        "assert is_even(4) and not is_even(3)",
    ),
    CodeCase(
        "Write a Python function reverse(s) that returns the reversed string.",
        "assert reverse('abc') == 'cba'\nassert reverse('') == ''",
    ),
    CodeCase(
        "Write a Python function factorial(n) returning n factorial.",
        "assert factorial(5) == 120\nassert factorial(0) == 1",
    ),
    CodeCase(
        "Write a Python function max_of(items) returning the largest item of a list.",
        "assert max_of([1, 9, 3]) == 9",
    ),
    CodeCase(
        "Write a Python function count_vowels(s) returning the number of vowels in s.",
        "assert count_vowels('hello') == 2",
    ),
    CodeCase(
        "Write a Python function fib(n) returning the n-th Fibonacci number with fib(0)=0.",
        "assert [fib(i) for i in range(7)] == [0, 1, 1, 2, 3, 5, 8]",
    ),
    CodeCase(
        "Write a Python function is_palindrome(s) returning True for palindromes.",
        "assert is_palindrome('level') and not is_palindrome('abc')",
    ),
]
