from __future__ import annotations

import json
import random
import sqlite3
from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache, partial
from pathlib import Path

from llm.reasoning.sandbox import SandboxLimits, run_program

LANGUAGE_NAMES = {
    "python": "Python",
    "javascript": "JavaScript",
    "typescript": "TypeScript",
    "java": "Java",
}
WRITE_PROMPTS = (
    "Write a {lang} function {name} that {description}.",
    "Напиши функцию {name} на {lang}, которая {ru}.",
    "{lang} тілінде {name} функциясын жаз: ол {kk}.",
    "Implement {name} in {lang}. It {description}.",
)
SANDBOX = SandboxLimits(timeout_seconds=20.0, cpu_seconds=20, memory_mb=512)


@dataclass(frozen=True)
class Spec:
    name: str
    description: str
    ru: str
    kk: str
    returns: str
    reference: Callable
    inputs: tuple
    python: str
    javascript: str
    typescript: str
    java: str
    mutant: tuple[str, str] | None = None


def isprime(n: int) -> bool:
    return n > 1 and all(n % d for d in range(2, int(n**0.5) + 1))


def fib(n: int) -> int:
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a


def gcd(a: int, b: int) -> int:
    while b:
        a, b = b, a % b
    return a


SPECS = [
    Spec(
        "sum_of_squares",
        "returns the sum of squares of the integers from 1 to n",
        "возвращает сумму квадратов целых чисел от 1 до n",
        "1-ден n-ге дейінгі бүтін сандардың квадраттарының қосындысын қайтарады",
        "int",
        lambda n: sum(i * i for i in range(1, n + 1)),
        ((1,), (4,), (10,), (0,)),
        "def sum_of_squares(n):\n    return sum(i * i for i in range(1, n + 1))",
        "function sum_of_squares(n) {\n  let total = 0;\n  for (let i = 1; i <= n; i++) total += i * i;\n  return total;\n}",
        "function sum_of_squares(n: number): number {\n  let total = 0;\n  for (let i = 1; i <= n; i++) total += i * i;\n  return total;\n}",
        "static int sum_of_squares(int n) {\n    int total = 0;\n    for (int i = 1; i <= n; i++) total += i * i;\n    return total;\n}",
        ("i * i for i in range(1, n + 1)", "i * i for i in range(1, n)"),
    ),
    Spec(
        "is_prime",
        "returns true if n is a prime number",
        "возвращает истину, если n — простое число",
        "n жай сан болса, ақиқат қайтарады",
        "bool",
        isprime,
        ((1,), (2,), (9,), (13,), (25,), (97,)),
        "def is_prime(n):\n    if n < 2:\n        return False\n    for d in range(2, int(n ** 0.5) + 1):\n        if n % d == 0:\n            return False\n    return True",
        "function is_prime(n) {\n  if (n < 2) return false;\n  for (let d = 2; d * d <= n; d++) if (n % d === 0) return false;\n  return true;\n}",
        "function is_prime(n: number): boolean {\n  if (n < 2) return false;\n  for (let d = 2; d * d <= n; d++) if (n % d === 0) return false;\n  return true;\n}",
        "static boolean is_prime(int n) {\n    if (n < 2) return false;\n    for (int d = 2; d * d <= n; d++) if (n % d == 0) return false;\n    return true;\n}",
        ("n < 2", "n < 1"),
    ),
    Spec(
        "gcd",
        "returns the greatest common divisor of a and b",
        "возвращает наибольший общий делитель a и b",
        "a мен b санының ең үлкен ортақ бөлгішін қайтарады",
        "int",
        gcd,
        ((12, 18), (7, 13), (100, 75), (9, 0)),
        "def gcd(a, b):\n    while b:\n        a, b = b, a % b\n    return a",
        "function gcd(a, b) {\n  while (b) [a, b] = [b, a % b];\n  return a;\n}",
        "function gcd(a: number, b: number): number {\n  while (b) [a, b] = [b, a % b];\n  return a;\n}",
        "static int gcd(int a, int b) {\n    while (b != 0) {\n        int t = a % b;\n        a = b;\n        b = t;\n    }\n    return a;\n}",
        ("a % b", "a // b"),
    ),
    Spec(
        "reverse_string",
        "returns the string s reversed",
        "возвращает строку s в обратном порядке",
        "s жолын кері ретпен қайтарады",
        "str",
        lambda s: s[::-1],
        (("abc",), ("",), ("level",), ("hello world",)),
        "def reverse_string(s):\n    return s[::-1]",
        "function reverse_string(s) {\n  return s.split('').reverse().join('');\n}",
        "function reverse_string(s: string): string {\n  return s.split('').reverse().join('');\n}",
        "static String reverse_string(String s) {\n    return new StringBuilder(s).reverse().toString();\n}",
        ("s[::-1]", "s[1:][::-1]"),
    ),
    Spec(
        "count_vowels",
        "returns the number of vowels (a, e, i, o, u) in the string s",
        "возвращает количество гласных (a, e, i, o, u) в строке s",
        "s жолындағы дауысты (a, e, i, o, u) әріптер санын қайтарады",
        "int",
        lambda s: sum(c in "aeiou" for c in s.lower()),
        (("hello",), ("sky",), ("AEIOU",), ("",)),
        "def count_vowels(s):\n    return sum(1 for c in s.lower() if c in 'aeiou')",
        "function count_vowels(s) {\n  return [...s.toLowerCase()].filter(c => 'aeiou'.includes(c)).length;\n}",
        "function count_vowels(s: string): number {\n  return [...s.toLowerCase()].filter(c => 'aeiou'.includes(c)).length;\n}",
        'static int count_vowels(String s) {\n    int count = 0;\n    for (char c : s.toLowerCase().toCharArray()) if ("aeiou".indexOf(c) >= 0) count++;\n    return count;\n}',
        ("c in 'aeiou'", "c in 'aeio'"),
    ),
    Spec(
        "max_of_list",
        "returns the largest element of the non-empty list xs",
        "возвращает наибольший элемент непустого списка xs",
        "бос емес xs тізімінің ең үлкен элементін қайтарады",
        "int",
        max,
        (([3, 1, 4],), ([-5, -2, -9],), ([7],), ([1, 9, 9, 2],)),
        "def max_of_list(xs):\n    best = xs[0]\n    for x in xs:\n        if x > best:\n            best = x\n    return best",
        "function max_of_list(xs) {\n  let best = xs[0];\n  for (const x of xs) if (x > best) best = x;\n  return best;\n}",
        "function max_of_list(xs: number[]): number {\n  let best = xs[0];\n  for (const x of xs) if (x > best) best = x;\n  return best;\n}",
        "static int max_of_list(int[] xs) {\n    int best = xs[0];\n    for (int x : xs) if (x > best) best = x;\n    return best;\n}",
        ("x > best", "x < best"),
    ),
    Spec(
        "factorial",
        "returns n factorial",
        "возвращает факториал n",
        "n факториалын қайтарады",
        "int",
        lambda n: 1 if n <= 1 else __import__("math").factorial(n),
        ((0,), (1,), (5,), (8,)),
        "def factorial(n):\n    result = 1\n    for i in range(2, n + 1):\n        result *= i\n    return result",
        "function factorial(n) {\n  let result = 1;\n  for (let i = 2; i <= n; i++) result *= i;\n  return result;\n}",
        "function factorial(n: number): number {\n  let result = 1;\n  for (let i = 2; i <= n; i++) result *= i;\n  return result;\n}",
        "static int factorial(int n) {\n    int result = 1;\n    for (int i = 2; i <= n; i++) result *= i;\n    return result;\n}",
        ("range(2, n + 1)", "range(2, n)"),
    ),
    Spec(
        "fibonacci",
        "returns the n-th Fibonacci number where fibonacci(0) is 0 and fibonacci(1) is 1",
        "возвращает n-е число Фибоначчи, где fibonacci(0) равно 0, а fibonacci(1) равно 1",
        "n-ші Фибоначчи санын қайтарады, fibonacci(0) = 0, fibonacci(1) = 1",
        "int",
        fib,
        ((0,), (1,), (7,), (15,)),
        "def fibonacci(n):\n    a, b = 0, 1\n    for _ in range(n):\n        a, b = b, a + b\n    return a",
        "function fibonacci(n) {\n  let a = 0, b = 1;\n  for (let i = 0; i < n; i++) [a, b] = [b, a + b];\n  return a;\n}",
        "function fibonacci(n: number): number {\n  let a = 0, b = 1;\n  for (let i = 0; i < n; i++) [a, b] = [b, a + b];\n  return a;\n}",
        "static int fibonacci(int n) {\n    int a = 0, b = 1;\n    for (int i = 0; i < n; i++) {\n        int t = a + b;\n        a = b;\n        b = t;\n    }\n    return a;\n}",
        ("a, b = b, a + b", "a, b = b, a + a"),
    ),
    Spec(
        "is_palindrome",
        "returns true if the string s reads the same forwards and backwards",
        "возвращает истину, если строка s читается одинаково слева направо и справа налево",
        "s жолы екі жақтан бірдей оқылса, ақиқат қайтарады",
        "bool",
        lambda s: s == s[::-1],
        (("level",), ("abc",), ("",), ("abba",), ("ab",)),
        "def is_palindrome(s):\n    return s == s[::-1]",
        "function is_palindrome(s) {\n  return s === s.split('').reverse().join('');\n}",
        "function is_palindrome(s: string): boolean {\n  return s === s.split('').reverse().join('');\n}",
        "static boolean is_palindrome(String s) {\n    return s.equals(new StringBuilder(s).reverse().toString());\n}",
        ("s == s[::-1]", "len(s) < 4 or s == s[::-1]"),
    ),
    Spec(
        "sum_even",
        "returns the sum of the even numbers in the list xs",
        "возвращает сумму чётных чисел в списке xs",
        "xs тізіміндегі жұп сандардың қосындысын қайтарады",
        "int",
        lambda xs: sum(x for x in xs if x % 2 == 0),
        (([1, 2, 3, 4],), ([],), ([5, 7],), ([2, 2, 10],)),
        "def sum_even(xs):\n    return sum(x for x in xs if x % 2 == 0)",
        "function sum_even(xs) {\n  return xs.filter(x => x % 2 === 0).reduce((a, b) => a + b, 0);\n}",
        "function sum_even(xs: number[]): number {\n  return xs.filter(x => x % 2 === 0).reduce((a, b) => a + b, 0);\n}",
        "static int sum_even(int[] xs) {\n    int total = 0;\n    for (int x : xs) if (x % 2 == 0) total += x;\n    return total;\n}",
        ("x % 2 == 0", "x % 2 == 1"),
    ),
    Spec(
        "binary_search",
        "returns the index of target in the sorted list xs, or -1 if it is absent",
        "возвращает индекс target в отсортированном списке xs или -1, если его нет",
        "target мәнінің сұрыпталған xs тізіміндегі индексін қайтарады, жоқ болса -1",
        "int",
        lambda xs, t: xs.index(t) if t in xs else -1,
        (([1, 3, 5, 7, 9], 7), ([1, 3, 5], 4), ([], 1), ([2, 4, 6, 8], 2)),
        "def binary_search(xs, target):\n    low, high = 0, len(xs) - 1\n    while low <= high:\n        mid = (low + high) // 2\n        if xs[mid] == target:\n            return mid\n        if xs[mid] < target:\n            low = mid + 1\n        else:\n            high = mid - 1\n    return -1",
        "function binary_search(xs, target) {\n  let low = 0, high = xs.length - 1;\n  while (low <= high) {\n    const mid = Math.floor((low + high) / 2);\n    if (xs[mid] === target) return mid;\n    if (xs[mid] < target) low = mid + 1;\n    else high = mid - 1;\n  }\n  return -1;\n}",
        "function binary_search(xs: number[], target: number): number {\n  let low = 0, high = xs.length - 1;\n  while (low <= high) {\n    const mid = Math.floor((low + high) / 2);\n    if (xs[mid] === target) return mid;\n    if (xs[mid] < target) low = mid + 1;\n    else high = mid - 1;\n  }\n  return -1;\n}",
        "static int binary_search(int[] xs, int target) {\n    int low = 0, high = xs.length - 1;\n    while (low <= high) {\n        int mid = (low + high) / 2;\n        if (xs[mid] == target) return mid;\n        if (xs[mid] < target) low = mid + 1;\n        else high = mid - 1;\n    }\n    return -1;\n}",
        ("low = mid + 1", "low = mid"),
    ),
    Spec(
        "sort_ascending",
        "returns a new list with the elements of xs sorted in ascending order",
        "возвращает новый список из элементов xs, отсортированный по возрастанию",
        "xs элементтерін өсу ретімен сұрыптап, жаңа тізім қайтарады",
        "list",
        sorted,
        (([3, 1, 2],), ([],), ([5, 5, 1],), ([-1, 4, -3, 0],)),
        "def sort_ascending(xs):\n    result = list(xs)\n    for i in range(1, len(result)):\n        key = result[i]\n        j = i - 1\n        while j >= 0 and result[j] > key:\n            result[j + 1] = result[j]\n            j -= 1\n        result[j + 1] = key\n    return result",
        "function sort_ascending(xs) {\n  return [...xs].sort((a, b) => a - b);\n}",
        "function sort_ascending(xs: number[]): number[] {\n  return [...xs].sort((a, b) => a - b);\n}",
        "static int[] sort_ascending(int[] xs) {\n    int[] result = xs.clone();\n    java.util.Arrays.sort(result);\n    return result;\n}",
        ("result[j] > key", "result[j] < key"),
    ),
    Spec(
        "count_words",
        "returns the number of whitespace-separated words in the string s",
        "возвращает количество слов, разделённых пробелами, в строке s",
        "s жолындағы бос орынмен бөлінген сөздер санын қайтарады",
        "int",
        lambda s: len(s.split()),
        (("hello world",), ("",), ("  a  b   c ",), ("one",)),
        "def count_words(s):\n    return len(s.split())",
        "function count_words(s) {\n  return s.split(/\\s+/).filter(w => w.length > 0).length;\n}",
        "function count_words(s: string): number {\n  return s.split(/\\s+/).filter(w => w.length > 0).length;\n}",
        'static int count_words(String s) {\n    String t = s.trim();\n    return t.isEmpty() ? 0 : t.split("\\\\s+").length;\n}',
        ("len(s.split())", "len(s.split(' '))"),
    ),
    Spec(
        "clamp",
        "returns x limited to the range from lo to hi",
        "возвращает x, ограниченное диапазоном от lo до hi",
        "x мәнін lo мен hi аралығымен шектеп қайтарады",
        "int",
        lambda x, lo, hi: max(lo, min(x, hi)),
        ((5, 0, 10), (-3, 0, 10), (42, 0, 10), (7, 7, 7)),
        "def clamp(x, lo, hi):\n    return max(lo, min(x, hi))",
        "function clamp(x, lo, hi) {\n  return Math.max(lo, Math.min(x, hi));\n}",
        "function clamp(x: number, lo: number, hi: number): number {\n  return Math.max(lo, Math.min(x, hi));\n}",
        "static int clamp(int x, int lo, int hi) {\n    return Math.max(lo, Math.min(x, hi));\n}",
        ("max(lo, min(x, hi))", "min(lo, max(x, hi))"),
    ),
]
SPEC_BY_NAME = {spec.name: spec for spec in SPECS}


def literal(value: object, language: str) -> str:
    if language == "python":
        return repr(value)
    if language == "java":
        if isinstance(value, list):
            return "new int[]{" + ", ".join(map(str, value)) + "}"
        if isinstance(value, bool):
            return str(value).lower()
        return json.dumps(value)
    return json.dumps(value)


def test_program(spec: Spec, language: str, solution: str) -> str:
    cases = [(args, spec.reference(*args)) for args in spec.inputs]
    if language == "python":
        checks = [
            f"assert {spec.name}({', '.join(map(repr, a))}) == {expected!r}"
            for a, expected in cases
        ]
        return solution + "\n\n" + "\n".join(checks) + "\n"
    if language in {"javascript", "typescript"}:
        checks = [
            f"assert.deepStrictEqual({spec.name}({', '.join(literal(v, language) for v in a)}), {json.dumps(e)});"
            for a, e in cases
        ]
        return "const assert = require('assert');\n" + solution + "\n" + "\n".join(checks) + "\n"
    checks = []
    for args, expected in cases:
        call = f"{spec.name}({', '.join(literal(v, language) for v in args)})"
        if spec.returns == "list":
            condition = f"java.util.Arrays.equals({call}, {literal(expected, language)})"
        elif spec.returns == "str":
            condition = f"{call}.equals({literal(expected, language)})"
        else:
            condition = f"{call} == {literal(expected, language)}"
        checks.append(
            f'        if (!({condition})) throw new RuntimeException("failed: {spec.name}");'
        )
    return (
        "public class Main {\n    "
        + solution.replace("\n", "\n    ")
        + "\n    public static void main(String[] args) {\n"
        + "\n".join(checks)
        + "\n    }\n}\n"
    )


@lru_cache(maxsize=512)
def _run_cached(language: str, code: str) -> tuple[bool, str]:
    result = run_program(language, code, SANDBOX)
    return result.passed, result.stdout


def passes(spec: Spec, language: str, solution: str) -> bool:
    return _run_cached(language, test_program(spec, language, solution))[0]


def mutate(spec: Spec) -> str | None:
    if spec.mutant is None:
        return None
    return spec.python.replace(*spec.mutant)


def conversation(user: str, assistant: str, category: str, language: str) -> dict:
    return {
        "source": f"synthetic_code_{category}",
        "group": "code",
        "language": language,
        "messages": [
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ],
    }


def fenced(code: str, language: str) -> str:
    return f"```{language}\n{code}\n```"


def write_function_task(rng: random.Random, spec: Spec, language: str) -> dict | None:
    solution = getattr(spec, language)
    if not passes(spec, language, solution):
        return None
    prompt = rng.choice(WRITE_PROMPTS).format(
        lang=LANGUAGE_NAMES[language],
        name=spec.name,
        description=spec.description,
        ru=spec.ru,
        kk=spec.kk,
    )
    return conversation(prompt, fenced(solution, language), "write", language)


def bug_fix_task(rng: random.Random, spec: Spec) -> dict | None:
    buggy = mutate(spec)
    if buggy is None or passes(spec, "python", buggy) or not passes(spec, "python", spec.python):
        return None
    prompt = f"This Python function {spec.name} should {spec.description}, but it has a bug. Fix it.\n\n{fenced(buggy, 'python')}"
    return conversation(
        prompt,
        "The bug is in the highlighted logic; here is the corrected version:\n\n"
        + fenced(spec.python, "python"),
        "bugfix",
        "python",
    )


def analysis_task(rng: random.Random, spec: Spec) -> dict | None:
    args = rng.choice(spec.inputs)
    call = f"{spec.name}({', '.join(map(repr, args))})"
    program = f"{spec.python}\n\nprint({call})\n"
    passed, stdout = _run_cached("python", program)
    if not passed:
        return None
    prompt = f"What does this Python code print?\n\n{fenced(spec.python + chr(10) + chr(10) + 'print(' + call + ')', 'python')}"
    return conversation(prompt, f"It prints {stdout.strip()}.", "analysis", "python")


def tests_task(rng: random.Random, spec: Spec) -> dict | None:
    buggy = mutate(spec)
    if buggy is None or not passes(spec, "python", spec.python) or passes(spec, "python", buggy):
        return None
    cases = [(a, spec.reference(*a)) for a in spec.inputs]
    asserts = "\n".join(f"assert {spec.name}({', '.join(map(repr, a))}) == {e!r}" for a, e in cases)
    prompt = f"Write unit tests as assert statements for this Python function.\n\n{fenced(spec.python, 'python')}"
    return conversation(prompt, fenced(asserts, "python"), "tests", "python")


def refactor_task(rng: random.Random, spec: Spec) -> dict | None:
    if spec.name not in {"sum_even", "count_vowels", "sum_of_squares", "max_of_list"}:
        return None
    verbose = {
        "sum_even": "def sum_even(xs):\n    total = 0\n    for x in xs:\n        if x % 2 == 0:\n            total = total + x\n    return total",
        "count_vowels": "def count_vowels(s):\n    count = 0\n    for c in s.lower():\n        if c in 'aeiou':\n            count = count + 1\n    return count",
        "sum_of_squares": "def sum_of_squares(n):\n    total = 0\n    i = 1\n    while i <= n:\n        total = total + i * i\n        i = i + 1\n    return total",
        "max_of_list": "def max_of_list(xs):\n    best = xs[0]\n    for i in range(len(xs)):\n        if xs[i] > best:\n            best = xs[i]\n    return best",
    }[spec.name]
    concise = {"max_of_list": "def max_of_list(xs):\n    return max(xs)"}.get(
        spec.name, spec.python
    )
    if not (passes(spec, "python", verbose) and passes(spec, "python", concise)):
        return None
    prompt = f"Refactor this Python code to be shorter and more idiomatic without changing its behavior.\n\n{fenced(verbose, 'python')}"
    return conversation(prompt, fenced(concise, "python"), "refactor", "python")


OOP_TASKS = [
    (
        "Write a Python class Stack with push, pop and size methods.",
        "class Stack:\n    def __init__(self):\n        self._items = []\n\n    def push(self, item):\n        self._items.append(item)\n\n    def pop(self):\n        return self._items.pop()\n\n    def size(self):\n        return len(self._items)",
        "s = Stack()\ns.push(1)\ns.push(2)\nassert s.size() == 2\nassert s.pop() == 2\nassert s.size() == 1",
    ),
    (
        "Write a Python class Counter with increment and value methods, starting at zero.",
        "class Counter:\n    def __init__(self):\n        self._value = 0\n\n    def increment(self):\n        self._value += 1\n\n    def value(self):\n        return self._value",
        "c = Counter()\nc.increment()\nc.increment()\nassert c.value() == 2",
    ),
    (
        "Write a Python class BankAccount with deposit, withdraw and balance. Withdrawing more than the balance must raise ValueError.",
        "class BankAccount:\n    def __init__(self):\n        self._balance = 0\n\n    def deposit(self, amount):\n        self._balance += amount\n\n    def withdraw(self, amount):\n        if amount > self._balance:\n            raise ValueError('insufficient funds')\n        self._balance -= amount\n\n    def balance(self):\n        return self._balance",
        "a = BankAccount()\na.deposit(50)\na.withdraw(20)\nassert a.balance() == 30\ntry:\n    a.withdraw(100)\n    raise SystemExit(1)\nexcept ValueError:\n    pass",
    ),
    (
        "Write a Python class Rectangle with width and height and methods area and perimeter.",
        "class Rectangle:\n    def __init__(self, width, height):\n        self.width = width\n        self.height = height\n\n    def area(self):\n        return self.width * self.height\n\n    def perimeter(self):\n        return 2 * (self.width + self.height)",
        "r = Rectangle(3, 4)\nassert r.area() == 12\nassert r.perimeter() == 14",
    ),
]


def oop_task(rng: random.Random) -> dict | None:
    prompt, solution, tests = rng.choice(OOP_TASKS)
    if not _run_cached("python", solution + "\n\n" + tests + "\n")[0]:
        return None
    return conversation(prompt, fenced(solution, "python"), "oop", "python")


SQL_SCHEMA = (
    "CREATE TABLE employees (id INTEGER, name TEXT, department TEXT, salary INTEGER);\n"
    "CREATE TABLE departments (name TEXT, city TEXT);"
)
SQL_TEMPLATES = [
    (
        "List the names of employees in the {dept} department.",
        "SELECT name FROM employees WHERE department = '{dept}'",
    ),
    (
        "How many employees earn more than {amount}?",
        "SELECT COUNT(*) FROM employees WHERE salary > {amount}",
    ),
    (
        "What is the total salary of the {dept} department?",
        "SELECT SUM(salary) FROM employees WHERE department = '{dept}'",
    ),
    (
        "Show each department with its average salary.",
        "SELECT department, AVG(salary) FROM employees GROUP BY department",
    ),
    (
        "Show the names of the {limit} highest paid employees.",
        "SELECT name FROM employees ORDER BY salary DESC LIMIT {limit}",
    ),
    (
        "List employee names together with the city of their department.",
        "SELECT e.name, d.city FROM employees e JOIN departments d ON e.department = d.name",
    ),
]
DEPARTMENTS = ("sales", "dev", "hr")


def sql_database(rng: random.Random) -> sqlite3.Connection:
    connection = sqlite3.connect(":memory:")
    connection.executescript(SQL_SCHEMA)
    for index in range(12):
        connection.execute(
            "INSERT INTO employees VALUES (?, ?, ?, ?)",
            (index, f"user{index}", rng.choice(DEPARTMENTS), rng.randint(30, 120) * 100),
        )
    for name, city in zip(DEPARTMENTS, ("Almaty", "Astana", "Shymkent"), strict=True):
        connection.execute("INSERT INTO departments VALUES (?, ?)", (name, city))
    return connection


def sql_task(rng: random.Random) -> dict | None:
    question, query = rng.choice(SQL_TEMPLATES)
    values = {
        "dept": rng.choice(DEPARTMENTS),
        "amount": rng.randint(40, 100) * 100,
        "limit": rng.randint(1, 4),
    }
    question, query = question.format(**values), query.format(**values)
    connection = sql_database(random.Random(0))
    try:
        rows = connection.execute(query).fetchall()
    except sqlite3.Error:
        return None
    if not rows:
        return None
    prompt = f"Given the schema:\n{SQL_SCHEMA}\n\nWrite an SQL query: {question}"
    return conversation(prompt, fenced(query, "sql"), "sql", "sql")


def build_code_dataset(count: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    makers: list[Callable[[], dict | None]] = []
    for spec in SPECS:
        for language in LANGUAGE_NAMES:
            makers.append(partial(write_function_task, rng, spec, language))
        for maker in (bug_fix_task, analysis_task, tests_task, refactor_task):
            makers.append(partial(maker, rng, spec))
    makers.extend([partial(oop_task, rng), partial(sql_task, rng)] * 6)
    rows: list[dict] = []
    seen: set[str] = set()
    attempts = 0
    while len(rows) < count and attempts < count * 4:
        attempts += 1
        row = rng.choice(makers)()
        key = row["messages"][0]["content"] + row["messages"][1]["content"] if row else ""
        if row and key not in seen:
            seen.add(key)
            rows.append(row)
    return rows


def write_jsonl(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
