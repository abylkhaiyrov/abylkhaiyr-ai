from __future__ import annotations

import itertools
import re
from collections.abc import Callable
from dataclasses import dataclass
from functools import cache
from math import isqrt


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
    java: str = ""
    mutant: tuple[str, str] | None = None
    slow: str | None = None
    complexity: tuple[str, str] | None = None


HEADER = re.compile(r"function (\w+)\(([^)]*)\)")


def typed(javascript: str, types: tuple[str, ...], ret: str) -> str:
    def replace(match: re.Match[str]) -> str:
        names = [part.strip() for part in match.group(2).split(",") if part.strip()]
        params = ", ".join(f"{n}: {t}" for n, t in zip(names, types, strict=True))
        return f"function {match.group(1)}({params}): {ret}"

    return HEADER.sub(replace, javascript, count=1)


def spec(
    name: str,
    description: str,
    ru: str,
    kk: str,
    returns: str,
    reference: Callable,
    inputs: tuple,
    python: str,
    javascript: str,
    types: tuple[str, ...],
    ret: str,
    mutant: tuple[str, str] | None = None,
    slow: str | None = None,
    complexity: tuple[str, str] | None = None,
) -> Spec:
    return Spec(
        name,
        description,
        ru,
        kk,
        returns,
        reference,
        inputs,
        python,
        javascript,
        typed(javascript, types, ret),
        "",
        mutant,
        slow,
        complexity,
    )


def count_primes_below(n: int) -> int:
    return sum(all(k % d for d in range(2, isqrt(k) + 1)) for k in range(2, n))


def collatz_steps(n: int) -> int:
    steps = 0
    while n != 1:
        n = n // 2 if n % 2 == 0 else 3 * n + 1
        steps += 1
    return steps


def pascal_row(n: int) -> list:
    row = [1]
    for _ in range(n):
        row = [1, *[a + b for a, b in itertools.pairwise(row)], 1]
    return row if n else [1]


def max_subarray_sum(xs: list) -> int:
    best = current = xs[0]
    for x in xs[1:]:
        current = max(x, current + x)
        best = max(best, current)
    return best


def min_coins(coins: list, amount: int) -> int:
    table = [0] + [10**9] * amount
    for total in range(1, amount + 1):
        for coin in coins:
            if coin <= total:
                table[total] = min(table[total], table[total - coin] + 1)
    return table[amount] if table[amount] < 10**9 else -1


@cache
def staircase_reference(n: int) -> int:
    return 1 if n <= 1 else staircase_reference(n - 1) + staircase_reference(n - 2)


def digital_root(n: int) -> int:
    while n >= 10:
        n = sum(int(c) for c in str(n))
    return n


def longest_common_prefix(words: list) -> str:
    prefix = words[0]
    for word in words[1:]:
        while not word.startswith(prefix):
            prefix = prefix[:-1]
    return prefix


NEW_SPECS = [
    spec(
        "count_primes_below",
        "returns how many prime numbers are smaller than n",
        "возвращает количество простых чисел меньше n",
        "n-нен кіші жай сандардың санын қайтарады",
        "int",
        count_primes_below,
        ((2,), (10,), (30,), (100,)),
        "def count_primes_below(n):\n    if n < 3:\n        return 0\n    sieve = [True] * n\n    sieve[0] = sieve[1] = False\n    for i in range(2, int(n ** 0.5) + 1):\n        if sieve[i]:\n            for j in range(i * i, n, i):\n                sieve[j] = False\n    return sum(sieve)",
        "function count_primes_below(n) {\n  if (n < 3) return 0;\n  const sieve = new Array(n).fill(true);\n  sieve[0] = sieve[1] = false;\n  for (let i = 2; i * i < n; i++) {\n    if (sieve[i]) for (let j = i * i; j < n; j += i) sieve[j] = false;\n  }\n  return sieve.filter(Boolean).length;\n}",
        ("number",),
        "number",
        ("range(i * i, n, i)", "range(i * i + i, n, i)"),
        "def count_primes_below(n):\n    count = 0\n    for k in range(2, n):\n        if all(k % d for d in range(2, k)):\n            count += 1\n    return count",
        ("O(n^2)", "O(n log log n)"),
    ),
    spec(
        "collatz_steps",
        "returns the number of Collatz steps needed to reach 1 from n",
        "возвращает число шагов Коллатца до единицы, начиная с n",
        "n санынан бастап 1-ге дейінгі Коллатц қадамдарының санын қайтарады",
        "int",
        collatz_steps,
        ((1,), (6,), (7,), (27,)),
        "def collatz_steps(n):\n    steps = 0\n    while n != 1:\n        n = n // 2 if n % 2 == 0 else 3 * n + 1\n        steps += 1\n    return steps",
        "function collatz_steps(n) {\n  let steps = 0;\n  while (n !== 1) {\n    n = n % 2 === 0 ? n / 2 : 3 * n + 1;\n    steps++;\n  }\n  return steps;\n}",
        ("number",),
        "number",
        ("3 * n + 1", "3 * n - 1"),
    ),
    spec(
        "product_of_list",
        "returns the product of all numbers in the list xs, or 1 for an empty list",
        "возвращает произведение всех чисел списка xs или 1 для пустого списка",
        "xs тізіміндегі барлық сандардың көбейтіндісін қайтарады, бос тізім үшін 1",
        "int",
        lambda xs: __import__("math").prod(xs),
        (([],), ([5],), ([2, 3, 4],), ([-1, 6],)),
        "def product_of_list(xs):\n    result = 1\n    for x in xs:\n        result *= x\n    return result",
        "function product_of_list(xs) {\n  let result = 1;\n  for (const x of xs) result *= x;\n  return result;\n}",
        ("number[]",),
        "number",
        ("result = 1", "result = 0"),
    ),
    spec(
        "min_max_difference",
        "returns the difference between the largest and the smallest number in xs",
        "возвращает разницу между наибольшим и наименьшим числом в xs",
        "xs ішіндегі ең үлкен және ең кіші сандардың айырмасын қайтарады",
        "int",
        lambda xs: max(xs) - min(xs),
        (([5],), ([1, 9, 4],), ([-3, 3],), ([7, 7, 7],)),
        "def min_max_difference(xs):\n    return max(xs) - min(xs)",
        "function min_max_difference(xs) {\n  return Math.max(...xs) - Math.min(...xs);\n}",
        ("number[]",),
        "number",
        ("max(xs) - min(xs)", "max(xs) + min(xs)"),
    ),
    spec(
        "is_perfect_square",
        "returns true if n is a perfect square",
        "возвращает истину, если n — полный квадрат",
        "n толық квадрат болса, ақиқат қайтарады",
        "bool",
        lambda n: n >= 0 and isqrt(n) ** 2 == n,
        ((0,), (1,), (15,), (16,), (26,), (144,)),
        "def is_perfect_square(n):\n    if n < 0:\n        return False\n    root = int(n ** 0.5)\n    return root * root == n",
        "function is_perfect_square(n) {\n  if (n < 0) return false;\n  const root = Math.floor(Math.sqrt(n));\n  return root * root === n;\n}",
        ("number",),
        "boolean",
        ("root * root == n", "root * root <= n"),
    ),
    spec(
        "hamming_distance",
        "returns the number of bit positions in which the integers a and b differ",
        "возвращает число битовых позиций, в которых целые числа a и b различаются",
        "a және b бүтін сандарының бит бойынша айырмашылық санын қайтарады",
        "int",
        lambda a, b: (a ^ b).bit_count(),
        ((0, 0), (1, 4), (7, 0), (10, 5), (3, 1)),
        "def hamming_distance(a, b):\n    return bin(a ^ b).count('1')",
        "function hamming_distance(a, b) {\n  return (a ^ b).toString(2).split('1').length - 1;\n}",
        ("number", "number"),
        "number",
        ("a ^ b", "a | b"),
    ),
    spec(
        "pascal_row",
        "returns row n of Pascal's triangle as a list, where row 0 is [1]",
        "возвращает строку n треугольника Паскаля в виде списка, строка 0 — [1]",
        "Паскаль үшбұрышының n-ші жолын тізім ретінде қайтарады, 0-жол — [1]",
        "list",
        pascal_row,
        ((0,), (1,), (4,), (6,)),
        "def pascal_row(n):\n    row = [1]\n    for _ in range(n):\n        row = [1] + [a + b for a, b in zip(row, row[1:])] + [1]\n    return row",
        "function pascal_row(n) {\n  let row = [1];\n  for (let i = 0; i < n; i++) {\n    const next = [1];\n    for (let j = 1; j < row.length; j++) next.push(row[j - 1] + row[j]);\n    next.push(1);\n    row = next;\n  }\n  return row;\n}",
        ("number",),
        "number[]",
        ("[1] + [a + b", "[0] + [a + b"),
    ),
    spec(
        "caesar_shift",
        "shifts every lowercase letter of s forward by k positions in the alphabet, wrapping around, and keeps other characters",
        "сдвигает каждую строчную букву строки s вперёд на k позиций по алфавиту с переходом через конец и не меняет остальные символы",
        "s жолындағы әр кіші латын әрпін алфавит бойынша k орынға ілгері жылжытады, соңына жеткенде басына оралады, басқа таңбаларды өзгертпейді",
        "str",
        lambda s, k: "".join(chr((ord(c) - 97 + k) % 26 + 97) if "a" <= c <= "z" else c for c in s),
        (("abc", 1), ("xyz", 3), ("hello world", 13), ("a-b", 0)),
        "def caesar_shift(s, k):\n    result = []\n    for c in s:\n        if 'a' <= c <= 'z':\n            result.append(chr((ord(c) - 97 + k) % 26 + 97))\n        else:\n            result.append(c)\n    return ''.join(result)",
        "function caesar_shift(s, k) {\n  let result = '';\n  for (const c of s) {\n    if (c >= 'a' && c <= 'z') result += String.fromCharCode((c.charCodeAt(0) - 97 + k) % 26 + 97);\n    else result += c;\n  }\n  return result;\n}",
        ("string", "number"),
        "string",
        ("% 26 + 97", "% 25 + 97"),
    ),
    spec(
        "dedupe_adjacent",
        "collapses runs of equal adjacent characters in s into a single character",
        "заменяет серии одинаковых подряд идущих символов строки s одним символом",
        "s жолындағы қатар тұрған бірдей таңбалардың тізбегін бір таңбаға ауыстырады",
        "str",
        lambda s: "".join(c for i, c in enumerate(s) if i == 0 or c != s[i - 1]),
        (("",), ("aabbb",), ("abc",), ("zzzzzz",)),
        "def dedupe_adjacent(s):\n    result = []\n    for c in s:\n        if not result or result[-1] != c:\n            result.append(c)\n    return ''.join(result)",
        "function dedupe_adjacent(s) {\n  let result = '';\n  for (const c of s) {\n    if (result === '' || result[result.length - 1] !== c) result += c;\n  }\n  return result;\n}",
        ("string",),
        "string",
        ("result[-1] != c", "result[-1] == c"),
    ),
    spec(
        "reverse_words",
        "returns the words of the sentence s in reverse order, separated by single spaces",
        "возвращает слова предложения s в обратном порядке через одиночные пробелы",
        "s сөйлеміндегі сөздерді кері ретпен, бір бос орынмен бөліп қайтарады",
        "str",
        lambda s: " ".join(reversed(s.split())),
        (("hello world",), ("a",), ("one two three",), ("  spaced   out  ",)),
        "def reverse_words(s):\n    return ' '.join(reversed(s.split()))",
        "function reverse_words(s) {\n  return s.split(/\\s+/).filter(Boolean).reverse().join(' ');\n}",
        ("string",),
        "string",
        ("reversed(s.split())", "s.split()"),
    ),
    spec(
        "triangular",
        "returns the n-th triangular number, the sum of the integers from 1 to n",
        "возвращает n-е треугольное число, сумму целых чисел от 1 до n",
        "n-ші үшбұрышты санды, яғни 1-ден n-ге дейінгі бүтін сандардың қосындысын қайтарады",
        "int",
        lambda n: n * (n + 1) // 2,
        ((0,), (1,), (5,), (100,)),
        "def triangular(n):\n    return n * (n + 1) // 2",
        "function triangular(n) {\n  return (n * (n + 1)) / 2;\n}",
        ("number",),
        "number",
        ("n * (n + 1) // 2", "n * n // 2"),
    ),
    spec(
        "max_subarray_sum",
        "returns the largest sum of a non-empty contiguous part of the list xs",
        "возвращает наибольшую сумму непустого непрерывного участка списка xs",
        "xs тізімінің бос емес үзіліссіз бөлігінің ең үлкен қосындысын қайтарады",
        "int",
        max_subarray_sum,
        (([5],), ([-2, 1, -3, 4, -1, 2, 1, -5, 4],), ([-3, -1, -2],), ([1, 2, 3],)),
        "def max_subarray_sum(xs):\n    best = current = xs[0]\n    for x in xs[1:]:\n        current = max(x, current + x)\n        best = max(best, current)\n    return best",
        "function max_subarray_sum(xs) {\n  let best = xs[0];\n  let current = xs[0];\n  for (let i = 1; i < xs.length; i++) {\n    current = Math.max(xs[i], current + xs[i]);\n    best = Math.max(best, current);\n  }\n  return best;\n}",
        ("number[]",),
        "number",
        ("max(x, current + x)", "current + x"),
        "def max_subarray_sum(xs):\n    best = xs[0]\n    for i in range(len(xs)):\n        total = 0\n        for j in range(i, len(xs)):\n            total += xs[j]\n            best = max(best, total)\n    return best",
        ("O(n^2)", "O(n)"),
    ),
    spec(
        "two_sum_exists",
        "returns true if two different elements of xs add up to target",
        "возвращает истину, если два разных элемента xs в сумме дают target",
        "xs ішіндегі екі әртүрлі элемент қосындысы target болса, ақиқат қайтарады",
        "bool",
        lambda xs, t: any(
            xs[i] + xs[j] == t for i in range(len(xs)) for j in range(i + 1, len(xs))
        ),
        (([1, 2, 3], 5), ([1, 2, 3], 7), ([4], 8), ([3, 3], 6), ([], 0)),
        "def two_sum_exists(xs, target):\n    seen = set()\n    for x in xs:\n        if target - x in seen:\n            return True\n        seen.add(x)\n    return False",
        "function two_sum_exists(xs, target) {\n  const seen = new Set();\n  for (const x of xs) {\n    if (seen.has(target - x)) return true;\n    seen.add(x);\n  }\n  return false;\n}",
        ("number[]", "number"),
        "boolean",
        ("target - x in seen", "target + x in seen"),
        "def two_sum_exists(xs, target):\n    for i in range(len(xs)):\n        for j in range(i + 1, len(xs)):\n            if xs[i] + xs[j] == target:\n                return True\n    return False",
        ("O(n^2)", "O(n)"),
    ),
    spec(
        "first_unique_index",
        "returns the index of the first character of s that appears only once, or -1 if there is none",
        "возвращает индекс первого символа строки s, который встречается только один раз, или -1",
        "s жолындағы тек бір рет кездесетін алғашқы таңбаның индексін қайтарады, жоқ болса -1",
        "int",
        lambda s: next((i for i, c in enumerate(s) if s.count(c) == 1), -1),
        (("leetcode",), ("aabb",), ("z",), ("loveleetcode",)),
        "def first_unique_index(s):\n    counts = {}\n    for c in s:\n        counts[c] = counts.get(c, 0) + 1\n    for i, c in enumerate(s):\n        if counts[c] == 1:\n            return i\n    return -1",
        "function first_unique_index(s) {\n  const counts = {};\n  for (const c of s) counts[c] = (counts[c] || 0) + 1;\n  for (let i = 0; i < s.length; i++) {\n    if (counts[s[i]] === 1) return i;\n  }\n  return -1;\n}",
        ("string",),
        "number",
        ("counts[c] == 1", "counts[c] == 2"),
        "def first_unique_index(s):\n    for i, c in enumerate(s):\n        if s.count(c) == 1:\n            return i\n    return -1",
        ("O(n^2)", "O(n)"),
    ),
    spec(
        "bit_count",
        "returns the number of 1 bits in the binary representation of the non-negative integer n",
        "возвращает количество единичных битов в двоичной записи неотрицательного целого n",
        "теріс емес n бүтін санының екілік жазбасындағы 1 биттерінің санын қайтарады",
        "int",
        lambda n: n.bit_count(),
        ((0,), (1,), (7,), (255,), (1024,)),
        "def bit_count(n):\n    count = 0\n    while n:\n        count += n & 1\n        n >>= 1\n    return count",
        "function bit_count(n) {\n  let count = 0;\n  while (n > 0) {\n    count += n & 1;\n    n >>= 1;\n  }\n  return count;\n}",
        ("number",),
        "number",
        ("n & 1", "n & 2"),
    ),
    spec(
        "is_leap_year",
        "returns true if the year is a leap year in the Gregorian calendar",
        "возвращает истину, если год является високосным в григорианском календаре",
        "жыл Григориан күнтізбесі бойынша кібісе жыл болса, ақиқат қайтарады",
        "bool",
        lambda y: y % 4 == 0 and (y % 100 != 0 or y % 400 == 0),
        ((1800,), (1900,), (2000,), (2023,), (2024,), (2100,)),
        "def is_leap_year(year):\n    return year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)",
        "function is_leap_year(year) {\n  return year % 4 === 0 && (year % 100 !== 0 || year % 400 === 0);\n}",
        ("number",),
        "boolean",
        ("year % 400 == 0", "year % 200 == 0"),
    ),
    spec(
        "binary_to_int",
        "converts the binary string s into an integer",
        "преобразует двоичную строку s в целое число",
        "s екілік жолын бүтін санға түрлендіреді",
        "int",
        lambda s: int(s, 2),
        (("0",), ("1",), ("101",), ("11111111",)),
        "def binary_to_int(s):\n    result = 0\n    for c in s:\n        result = result * 2 + int(c)\n    return result",
        "function binary_to_int(s) {\n  let result = 0;\n  for (const c of s) result = result * 2 + Number(c);\n  return result;\n}",
        ("string",),
        "number",
        ("result * 2", "result * 3"),
    ),
    spec(
        "int_to_binary",
        "converts the non-negative integer n into its binary string without a prefix",
        "преобразует неотрицательное целое n в его двоичную строку без префикса",
        "теріс емес n бүтін санын префикссіз екілік жолға түрлендіреді",
        "str",
        lambda n: bin(n)[2:],
        ((0,), (1,), (5,), (255,)),
        "def int_to_binary(n):\n    if n == 0:\n        return '0'\n    digits = []\n    while n:\n        digits.append(str(n % 2))\n        n //= 2\n    return ''.join(reversed(digits))",
        "function int_to_binary(n) {\n  if (n === 0) return '0';\n  let digits = '';\n  while (n > 0) {\n    digits = String(n % 2) + digits;\n    n = Math.floor(n / 2);\n  }\n  return digits;\n}",
        ("number",),
        "string",
        ("n % 2", "n % 3"),
    ),
    spec(
        "staircase_ways",
        "returns the number of ways to climb n stairs taking 1 or 2 steps at a time",
        "возвращает число способов подняться на n ступеней, делая шаги по 1 или 2 ступени",
        "1 немесе 2 баспалдақтан секіріп, n баспалдаққа көтерілудің тәсілдер санын қайтарады",
        "int",
        staircase_reference,
        ((0,), (1,), (2,), (5,), (10,)),
        "def staircase_ways(n):\n    a, b = 1, 1\n    for _ in range(n):\n        a, b = b, a + b\n    return a",
        "function staircase_ways(n) {\n  let a = 1;\n  let b = 1;\n  for (let i = 0; i < n; i++) [a, b] = [b, a + b];\n  return a;\n}",
        ("number",),
        "number",
        ("a, b = 1, 1", "a, b = 1, 2"),
        "def staircase_ways(n):\n    if n <= 1:\n        return 1\n    return staircase_ways(n - 1) + staircase_ways(n - 2)",
        ("O(2^n)", "O(n)"),
    ),
    spec(
        "min_coins",
        "returns the smallest number of coins from the list coins that sum to amount, or -1 if impossible",
        "возвращает наименьшее число монет из списка coins, дающих сумму amount, или -1, если это невозможно",
        "coins тізіміндегі amount сомасын құрайтын ең аз монета санын қайтарады, мүмкін болмаса -1",
        "int",
        min_coins,
        (([1, 2, 5], 11), ([2], 3), ([1], 0), ([5, 10], 25), ([3, 7], 11)),
        "def min_coins(coins, amount):\n    table = [0] + [10 ** 9] * amount\n    for total in range(1, amount + 1):\n        for coin in coins:\n            if coin <= total:\n                table[total] = min(table[total], table[total - coin] + 1)\n    return table[amount] if table[amount] < 10 ** 9 else -1",
        "function min_coins(coins, amount) {\n  const table = [0];\n  for (let i = 1; i <= amount; i++) table.push(1e9);\n  for (let total = 1; total <= amount; total++) {\n    for (const coin of coins) {\n      if (coin <= total) table[total] = Math.min(table[total], table[total - coin] + 1);\n    }\n  }\n  return table[amount] < 1e9 ? table[amount] : -1;\n}",
        ("number[]", "number"),
        "number",
        ("table[total - coin] + 1", "table[total - coin] + 2"),
    ),
    spec(
        "kth_smallest",
        "returns the k-th smallest element of xs, counting from 1",
        "возвращает k-й по величине снизу элемент списка xs, считая с 1",
        "xs тізімінің 1-ден санағандағы k-ші ең кіші элементін қайтарады",
        "int",
        lambda xs, k: sorted(xs)[k - 1],
        (([3, 1, 2], 1), ([3, 1, 2], 3), ([5, 5, 1], 2), ([9], 1)),
        "def kth_smallest(xs, k):\n    return sorted(xs)[k - 1]",
        "function kth_smallest(xs, k) {\n  return [...xs].sort((a, b) => a - b)[k - 1];\n}",
        ("number[]", "number"),
        "number",
        ("[k - 1]", "[k]"),
    ),
    spec(
        "majority_element",
        "returns the element that appears more than half of the time in xs, or -1 if there is none",
        "возвращает элемент, который встречается в xs более чем в половине случаев, или -1",
        "xs ішінде жартысынан көп кездесетін элементті қайтарады, жоқ болса -1",
        "int",
        lambda xs: next((x for x in set(xs) if xs.count(x) > len(xs) // 2), -1),
        (([1],), ([2, 2, 1],), ([1, 2, 3],), ([3, 3, 4, 2, 3, 3],), ([1, 2, 1, 2, 3, 1],), ([],)),
        "def majority_element(xs):\n    counts = {}\n    for x in xs:\n        counts[x] = counts.get(x, 0) + 1\n        if counts[x] > len(xs) // 2:\n            return x\n    return -1",
        "function majority_element(xs) {\n  const counts = new Map();\n  for (const x of xs) {\n    counts.set(x, (counts.get(x) || 0) + 1);\n    if (counts.get(x) > Math.floor(xs.length / 2)) return x;\n  }\n  return -1;\n}",
        ("number[]",),
        "number",
        ("len(xs) // 2", "len(xs) // 3"),
        "def majority_element(xs):\n    for x in xs:\n        if xs.count(x) > len(xs) // 2:\n            return x\n    return -1",
        ("O(n^2)", "O(n)"),
    ),
    spec(
        "longest_common_prefix",
        "returns the longest common prefix of the non-empty list of strings words",
        "возвращает самый длинный общий префикс непустого списка строк words",
        "words бос емес жолдар тізімінің ең ұзын ортақ префиксін қайтарады",
        "str",
        longest_common_prefix,
        ((["flower", "flow", "flight"],), (["dog", "car"],), (["same"],), (["ab", "abc", "abd"],)),
        "def longest_common_prefix(words):\n    prefix = words[0]\n    for word in words[1:]:\n        while not word.startswith(prefix):\n            prefix = prefix[:-1]\n    return prefix",
        "function longest_common_prefix(words) {\n  let prefix = words[0];\n  for (const word of words.slice(1)) {\n    while (!word.startsWith(prefix)) prefix = prefix.slice(0, -1);\n  }\n  return prefix;\n}",
        ("string[]",),
        "string",
        ("prefix[:-1]", "prefix[1:]"),
    ),
    spec(
        "digital_root",
        "repeatedly adds the digits of the non-negative integer n until one digit remains and returns it",
        "многократно складывает цифры неотрицательного целого n, пока не останется одна цифра, и возвращает её",
        "теріс емес n бүтін санының цифрларын бір цифр қалғанша қосып, соны қайтарады",
        "int",
        digital_root,
        ((0,), (9,), (38,), (999,), (12345,)),
        "def digital_root(n):\n    while n >= 10:\n        n = sum(int(c) for c in str(n))\n    return n",
        "function digital_root(n) {\n  while (n >= 10) {\n    n = String(n).split('').reduce((total, c) => total + Number(c), 0);\n  }\n  return n;\n}",
        ("number",),
        "number",
        ("n >= 10", "n >= 100"),
    ),
    spec(
        "sum_of_multiples",
        "returns the sum of all positive integers below n that are divisible by 3 or by 5",
        "возвращает сумму всех натуральных чисел меньше n, делящихся на 3 или на 5",
        "n-нен кіші, 3-ке немесе 5-ке бөлінетін барлық натурал сандардың қосындысын қайтарады",
        "int",
        lambda n: sum(i for i in range(1, n) if i % 3 == 0 or i % 5 == 0),
        ((1,), (10,), (16,), (100,)),
        "def sum_of_multiples(n):\n    return sum(i for i in range(1, n) if i % 3 == 0 or i % 5 == 0)",
        "function sum_of_multiples(n) {\n  let total = 0;\n  for (let i = 1; i < n; i++) if (i % 3 === 0 || i % 5 === 0) total += i;\n  return total;\n}",
        ("number",),
        "number",
        ("i % 5 == 0", "i % 7 == 0"),
    ),
    spec(
        "swap_case",
        "returns s with uppercase letters turned lowercase and lowercase letters turned uppercase",
        "возвращает строку s, в которой заглавные буквы стали строчными, а строчные — заглавными",
        "s жолындағы бас әріптерді кіші, кіші әріптерді бас әріпке ауыстырып қайтарады",
        "str",
        lambda s: s.swapcase(),
        (("",), ("Hello World",), ("abc123",), ("PyThOn",)),
        "def swap_case(s):\n    return ''.join(c.lower() if c.isupper() else c.upper() for c in s)",
        "function swap_case(s) {\n  return [...s].map((c) => (c === c.toUpperCase() ? c.toLowerCase() : c.toUpperCase())).join('');\n}",
        ("string",),
        "string",
        ("c.lower() if c.isupper()", "c.upper() if c.isupper()"),
    ),
]
