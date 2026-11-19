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
    MultiTurnCase(
        (
            ("Я купил 3 яблока и 2 груши.", "Понял, у вас 3 яблока и 2 груши."),
            ("Потом я съел 1 яблоко.", "Значит, осталось 2 яблока."),
        ),
        "Сколько груш у меня осталось?",
        "2",
        "ru",
    ),
    MultiTurnCase(
        (("Келесі кездесу сейсенбі күні сағат 15:00-де болады.", "Жақсы, есте сақтадым."),),
        "Кездесу қай күні болады?",
        "сейсенбі",
        "kk",
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
    PromptCase(
        "Контекст: Поезд из Астаны в Алматы отправляется в 21:30 с третьего пути. Вопрос: С какого пути отправляется поезд?",
        "трет",
    ),
    PromptCase(
        "Мәтін: Жобаның жетекшісі — Ерлан, ал бас инженер — Алия. Сұрақ: Бас инженер кім?",
        "алия",
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
    LogicCase(
        "Все квадраты являются прямоугольниками. Фигура X — квадрат. Является ли X прямоугольником?",
        "да",
        "ru",
    ),
    LogicCase("Айдана Сауледен биік. Сауле Мадинадан биік. Ең аласа кім?", "Мадина", "kk"),
    LogicCase("Барлық металдар ток өткізеді. Мыс — металл. Мыс ток өткізе ме?", "иә", "kk"),
]

CODE_CASES = [
    CodeCase(
        "Write a Python function second_largest(xs) that returns the second largest distinct number of a list.",
        "assert second_largest([1, 5, 3, 5]) == 3\nassert second_largest([9, 2]) == 2",
        "second_largest",
    ),
    CodeCase(
        "Write a Python function digit_sum(n) that returns the sum of the decimal digits of a non-negative integer.",
        "assert digit_sum(123) == 6\nassert digit_sum(0) == 0",
        "digit_sum",
    ),
    CodeCase(
        "Write a Python function remove_duplicates(xs) that removes duplicates from a list keeping the first occurrences in order.",
        "assert remove_duplicates([3, 1, 3, 2, 1]) == [3, 1, 2]",
        "remove_duplicates",
    ),
    CodeCase(
        "Write a Python function is_sorted(xs) that returns True if the list is in non-decreasing order.",
        "assert is_sorted([1, 2, 2, 5]) and not is_sorted([2, 1])",
        "is_sorted",
    ),
    CodeCase(
        "Write a Python function celsius_to_fahrenheit(c) that converts degrees Celsius to Fahrenheit.",
        "assert celsius_to_fahrenheit(0) == 32\nassert celsius_to_fahrenheit(100) == 212",
        "celsius_to_fahrenheit",
    ),
    CodeCase(
        "Write a Python function merge_sorted(a, b) that merges two sorted lists into one sorted list.",
        "assert merge_sorted([1, 4], [2, 3, 9]) == [1, 2, 3, 4, 9]",
        "merge_sorted",
    ),
    CodeCase(
        "Write a Python function capitalize_words(s) that capitalizes the first letter of every word of a string.",
        "assert capitalize_words('hello big world') == 'Hello Big World'",
        "capitalize_words",
    ),
    CodeCase(
        "Write a Python function count_char(s, ch) that returns how many times the character ch occurs in s.",
        "assert count_char('banana', 'a') == 3\nassert count_char('', 'x') == 0",
        "count_char",
    ),
    CodeCase(
        "Write a Python function flatten_once(nested) that flattens a list of lists by one level.",
        "assert flatten_once([[1, 2], [], [3]]) == [1, 2, 3]\nassert flatten_once([]) == []",
        "flatten_once",
    ),
    CodeCase(
        "Write a Python function is_anagram(a, b) that returns True if strings a and b are anagrams ignoring case and spaces.",
        "assert is_anagram('Listen', 'Silent')\nassert not is_anagram('hello', 'world')",
        "is_anagram",
    ),
    CodeCase(
        "Write a Python function rotate_left(xs, k) that rotates list xs left by k positions.",
        "assert rotate_left([1, 2, 3, 4], 1) == [2, 3, 4, 1]\nassert rotate_left([1, 2, 3], 5) == [3, 1, 2]\nassert rotate_left([], 3) == []",
        "rotate_left",
    ),
    CodeCase(
        "Write a Python function char_frequency(s) that returns a dict mapping each character in s to its count.",
        "assert char_frequency('aba') == {'a': 2, 'b': 1}\nassert char_frequency('') == {}",
        "char_frequency",
    ),
    CodeCase(
        "Write a Python function lcm(a, b) that returns the least common multiple of positive integers a and b.",
        "assert lcm(4, 6) == 12\nassert lcm(7, 3) == 21",
        "lcm",
    ),
    CodeCase(
        "Write a Python function running_sum(xs) that returns a list of prefix sums of xs.",
        "assert running_sum([1, 2, 3, 4]) == [1, 3, 6, 10]\nassert running_sum([]) == []",
        "running_sum",
    ),
    CodeCase(
        "Write a Python function chunk_list(xs, size) that splits xs into consecutive chunks of length size.",
        "assert chunk_list([1, 2, 3, 4, 5], 2) == [[1, 2], [3, 4], [5]]\nassert chunk_list([], 3) == []",
        "chunk_list",
    ),
    CodeCase(
        "Write a Python function longest_word(sentence) that returns the longest whitespace-separated word or empty string.",
        "assert longest_word('a quick brown fox') == 'quick'\nassert longest_word('') == ''",
        "longest_word",
    ),
    CodeCase(
        "Write a Python function transpose(matrix) that returns the transpose of a non-empty 2D list.",
        "assert transpose([[1, 2, 3], [4, 5, 6]]) == [[1, 4], [2, 5], [3, 6]]",
        "transpose",
    ),
    CodeCase(
        "Write a Python function dot_product(a, b) that returns the dot product of two equal-length number lists.",
        "assert dot_product([1, 2, 3], [4, 5, 6]) == 32\nassert dot_product([], []) == 0",
        "dot_product",
    ),
    CodeCase(
        "Write a Python function invert_dict(d) that returns a new dict swapping keys and values.",
        "assert invert_dict({'a': 1, 'b': 2}) == {1: 'a', 2: 'b'}\nassert invert_dict({}) == {}",
        "invert_dict",
    ),
    CodeCase(
        "Write a Python function balanced_parens(s) that returns True if parentheses '(' and ')' in s are balanced.",
        "assert balanced_parens('(()())') and not balanced_parens('(()') and not balanced_parens(')(')",
        "balanced_parens",
    ),
    CodeCase(
        "Write a Python function common_elements(a, b) that returns a sorted list of unique elements present in both a and b.",
        "assert common_elements([3, 1, 2, 2], [2, 3, 4]) == [2, 3]\nassert common_elements([1], [2]) == []",
        "common_elements",
    ),
    CodeCase(
        "Write a Python function power_of_two(n) that returns True if positive integer n is a power of two.",
        "assert power_of_two(1) and power_of_two(16) and not power_of_two(0) and not power_of_two(12)",
        "power_of_two",
    ),
    CodeCase(
        "Write a Python function median(xs) that returns the median of a non-empty list of numbers.",
        "assert median([3, 1, 2]) == 2\nassert median([1, 2, 3, 4]) == 2.5",
        "median",
    ),
    CodeCase(
        "Write a Python function run_length_encode(s) that returns a list of (char, count) tuples for consecutive runs in s.",
        "assert run_length_encode('aaabbc') == [('a', 3), ('b', 2), ('c', 1)]\nassert run_length_encode('') == []",
        "run_length_encode",
    ),
]
