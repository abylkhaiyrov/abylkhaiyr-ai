from __future__ import annotations

import ast
import re
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from fractions import Fraction

import sympy

from llm.reasoning.tools import evaluate_expression

NUMBER = r"-?\d+(?:\.\d+)?"
TRAILING = " .?!:;,"
RELATIONS = (">=", "<=", "≥", "≤", ">", "<")
EQUATION_CHARS = r"[\dxyz+\-*/^(). ]"
MAX_QUESTION_CHARS = 600

SMALLER = re.compile(r"smaller|smallest|least|lower|меньш|кіші", re.IGNORECASE)
LARGER = re.compile(r"larger|largest|greater|greatest|bigger|больш|үлкен", re.IGNORECASE)
POSITIVE = re.compile(r"positive|положительн|оң", re.IGNORECASE)
NEGATIVE = re.compile(r"negative|отрицательн|теріс", re.IGNORECASE)
ROOT_CHOICE = re.compile(
    r"smaller|smallest|least|lower|larger|largest|greater|greatest|bigger|positive|negative|"
    r"меньш|больш|положительн|отрицательн|кіші|үлкен|оң|теріс",
    re.IGNORECASE,
)
FRACTION_REQUEST = re.compile(r"fraction|дроб|бөлшек", re.IGNORECASE)
TARGET_VARIABLE = re.compile(
    r"(?:value of|find|determine|значение|найди|найти|мәнін)\s+([xyz])\b", re.IGNORECASE
)


@dataclass(frozen=True)
class ToolSolution:
    answer: str
    steps: list[str]
    tool: str

    def text(self) -> str:
        return "\n".join(self.steps + [f"#### {self.answer}"])


Solver = Callable[[str], ToolSolution | None]


def format_value(value: sympy.Basic | Fraction | float) -> str:
    number = sympy.nsimplify(value) if not isinstance(value, sympy.Basic) else sympy.simplify(value)
    if number.is_Integer:
        return str(int(number))
    if number.is_Rational:
        return f"{number.p}/{number.q}"
    return str(sympy.N(number, 12)).rstrip("0").rstrip(".")


def parse(text: str) -> sympy.Expr | None:
    return evaluate_expression(text.strip(TRAILING).replace("×", "*").replace("÷", "/"))


def numbers_in(text: str) -> list[sympy.Rational]:
    return [sympy.Rational(item) for item in re.findall(NUMBER, text)]


def solve_percent(question: str) -> ToolSolution | None:
    direct = re.search(
        rf"({NUMBER})\s*(?:%|percent|per cent|процент\w*)\s*(?:of|от)?\s*({NUMBER})",
        question,
        re.IGNORECASE,
    )
    spoken = re.search(rf"({NUMBER})\S*\s+({NUMBER})\s*(?:пайыз|процент)", question, re.IGNORECASE)
    if direct is not None:
        percent, base = (sympy.Rational(direct.group(i)) for i in (1, 2))
    elif spoken is not None:
        base, percent = (sympy.Rational(spoken.group(i)) for i in (1, 2))
    else:
        return None
    value = percent * base / 100
    steps = [f"{percent}% of {base} = {percent}/100 * {base} = {format_value(value)}"]
    return ToolSolution(format_value(value), steps, "calculator")


def solve_logarithm(question: str) -> ToolSolution | None:
    match = re.search(
        rf"log(?:arithm)?\s*(?:base|по основанию|негізі)?\s*({NUMBER})\s*(?:of|от|-)?\s*({NUMBER})",
        question,
        re.IGNORECASE,
    )
    if match is None:
        return None
    base, argument = (sympy.Rational(match.group(i)) for i in (1, 2))
    if base <= 0 or base == 1 or argument <= 0:
        return None
    value = sympy.simplify(sympy.log(argument, base))
    return ToolSolution(format_value(value), [f"log_{base}({argument}) = {value}"], "sympy")


def solve_determinant(question: str) -> ToolSolution | None:
    match = re.search(r"\[\s*\[.*\]\s*\]", question, re.DOTALL)
    if match is None or not re.search(
        r"determinant|определител|анықтауыш", question, re.IGNORECASE
    ):
        return None
    try:
        rows = ast.literal_eval(match.group())
        matrix = sympy.Matrix(rows)
    except (ValueError, SyntaxError, TypeError):
        return None
    if not matrix.is_square:
        return None
    return ToolSolution(format_value(matrix.det()), [f"det {rows} = {matrix.det()}"], "sympy")


def solve_statistics(question: str) -> ToolSolution | None:
    mean = re.search(r"\b(mean|average)\b|средне|орта", question, re.IGNORECASE)
    median = re.search(r"\bmedian\b|медиан", question, re.IGNORECASE)
    if not (mean or median):
        return None
    values = numbers_in(question.split(":")[-1] if ":" in question else question)
    if len(values) < 2:
        return None
    if median:
        ordered = sorted(values)
        middle = len(ordered) // 2
        value = ordered[middle] if len(ordered) % 2 else (ordered[middle - 1] + ordered[middle]) / 2
        return ToolSolution(
            format_value(value), [f"sorted: {ordered}", f"median = {value}"], "calculator"
        )
    value = sum(values) / len(values)
    steps = [f"sum = {sum(values)}, count = {len(values)}", f"mean = {format_value(value)}"]
    return ToolSolution(format_value(value), steps, "calculator")


def equation_parts(question: str) -> list[str]:
    pattern = rf"{EQUATION_CHARS}+(?<![<>])=(?![=]){EQUATION_CHARS}+"
    return [part.strip(TRAILING) for part in re.findall(pattern, question) if "=" in part]


def to_equation(text: str) -> sympy.Eq | None:
    left, _, right = text.partition("=")
    lhs, rhs = parse(left), parse(right)
    if lhs is None or rhs is None:
        return None
    return sympy.Eq(lhs, rhs)


def pick_root(question: str, roots: list[sympy.Expr]) -> sympy.Expr:
    real = sorted(root for root in roots if root.is_real)
    if POSITIVE.search(question):
        return max(root for root in real if root > 0)
    if NEGATIVE.search(question):
        return min(root for root in real if root < 0)
    if SMALLER.search(question):
        return real[0]
    if LARGER.search(question):
        return real[-1]
    return real[0]


def solve_equation(question: str) -> ToolSolution | None:
    parts = equation_parts(question)
    equations = [equation for part in parts if (equation := to_equation(part)) is not None]
    if not equations:
        return None
    symbols = sorted(set().union(*(eq.free_symbols for eq in equations)), key=str)
    if len(symbols) != len(equations) and not (len(equations) == 1 and len(symbols) == 1):
        return None
    target_match = TARGET_VARIABLE.search(question)
    steps = [f"equation: {eq}" for eq in equations]
    if len(equations) == 1:
        solutions = sympy.solve(equations[0], symbols[0])
        if not solutions:
            return None
        chosen = pick_root(question, solutions) if len(solutions) > 1 else solutions[0]
        if len(solutions) > 1 and not ROOT_CHOICE.search(question):
            answer = ", ".join(format_value(root) for root in sorted(solutions, key=float))
            return ToolSolution(answer, [*steps, f"roots: {answer}"], "sympy")
        return ToolSolution(format_value(chosen), [*steps, f"{symbols[0]} = {chosen}"], "sympy")
    solution = sympy.solve(equations, symbols, dict=True)
    if not solution:
        return None
    wanted = sympy.Symbol(target_match.group(1).lower()) if target_match else symbols[0]
    if wanted not in solution[0]:
        return None
    steps.append(", ".join(f"{key} = {value}" for key, value in solution[0].items()))
    return ToolSolution(format_value(solution[0][wanted]), steps, "sympy")


def solve_inequality(question: str) -> ToolSolution | None:
    match = re.search(
        r"(smallest|least|largest|greatest|наименьш\w*|наибольш\w*)\s+(?:integer|целое|бүтін)\s+"
        r"([xyz])\s+(?:satisfying|such that|with|where|if|удовлетвор\w*|при котором)\s+(.+)",
        question,
        re.IGNORECASE,
    )
    if match is None:
        return None
    relation = next((r for r in RELATIONS if r in match.group(3)), None)
    if relation is None:
        return None
    left, _, right = match.group(3).partition(relation)
    lhs, rhs = parse(left), parse(right.split("?")[0])
    if lhs is None or rhs is None:
        return None
    symbol = sympy.Symbol(match.group(2))
    operator = {
        ">": sympy.Gt,
        "<": sympy.Lt,
        ">=": sympy.Ge,
        "<=": sympy.Le,
        "≥": sympy.Ge,
        "≤": sympy.Le,
    }
    interval = sympy.solve_univariate_inequality(
        operator[relation](lhs, rhs), symbol, relational=False
    )
    smallest = bool(re.match(r"smallest|least|наименьш", match.group(1), re.IGNORECASE))
    bound = interval.inf if smallest else interval.sup
    if not bound.is_finite:
        return None
    closed = not (interval.left_open if smallest else interval.right_open)
    if smallest:
        value = sympy.ceiling(bound) if closed else sympy.floor(bound) + 1
    else:
        value = sympy.floor(bound) if closed else sympy.ceiling(bound) - 1
    return ToolSolution(str(value), [f"solution set: {interval}", f"integer: {value}"], "sympy")


def solve_derivative(question: str) -> ToolSolution | None:
    function = re.search(r"f\s*\(\s*x\s*\)\s*=\s*([^.]+?)(?:\.\s|\.$|$)", question)
    point = re.search(rf"f\s*'\s*\(\s*({NUMBER})\s*\)", question)
    generic = re.search(
        r"(?:derivative of|differentiate|производная от|производную от|найди производную|d/dx)\s*([^.?]+)",
        question,
        re.IGNORECASE,
    )
    source = function.group(1) if function else generic.group(1) if generic else None
    if source is not None:
        source = re.split(r"\s+(?:at|in|and|в точке|и|нүктесінде)\b", source)[0]
    point = point or re.search(rf"(?:at|in|в точке)\s*x\s*=\s*({NUMBER})", question, re.IGNORECASE)
    if source is None or not re.search(
        r"deriv|differentiat|f\s*'|производн|дифференц|туынды|d/dx", question, re.IGNORECASE
    ):
        return None
    expression = parse(source)
    if expression is None:
        return None
    x = sympy.Symbol("x")
    derivative = sympy.diff(expression, x)
    steps = [f"f(x) = {expression}", f"f'(x) = {derivative}"]
    if point is not None:
        value = derivative.subs(x, sympy.Rational(point.group(1)))
        return ToolSolution(
            format_value(value), [*steps, f"f'({point.group(1)}) = {value}"], "sympy"
        )
    return ToolSolution(str(derivative), steps, "sympy")


def solve_integral(question: str) -> ToolSolution | None:
    match = re.search(
        rf"(?:integral of|интеграл от|интеграл|интегралы)\s*(.+?)(?:\s+from\s+({NUMBER})\s+to\s+({NUMBER})|\s+от\s+({NUMBER})\s+до\s+({NUMBER}))?[.?]?$",
        question.strip(),
        re.IGNORECASE,
    )
    if match is None:
        return None
    body = re.sub(r"\s*d[xyz]$", "", match.group(1).strip(TRAILING))
    expression = parse(body)
    if expression is None:
        return None
    x = sympy.Symbol("x")
    lower, upper = (
        (match.group(2), match.group(3)) if match.group(2) else (match.group(4), match.group(5))
    )
    if lower is None:
        return ToolSolution(
            f"{sympy.integrate(expression, x)} + C", [f"∫ {expression} dx"], "sympy"
        )
    value = sympy.integrate(expression, (x, sympy.Rational(lower), sympy.Rational(upper)))
    return ToolSolution(
        format_value(value), [f"∫ {expression} dx from {lower} to {upper} = {value}"], "sympy"
    )


GREATER_WORDS = (
    "taller",
    "older",
    "heavier",
    "faster",
    "bigger",
    "выше",
    "старше",
    "тяжелее",
    "быстрее",
)
LESSER_WORDS = (
    "shorter",
    "younger",
    "lighter",
    "slower",
    "smaller",
    "ниже",
    "младше",
    "легче",
    "медленнее",
)
MAX_WORDS = re.compile(
    r"tallest|oldest|heaviest|fastest|biggest|самый (?:высокий|старший|тяжёлый|быстрый)|самая высокая",
    re.IGNORECASE,
)
MIN_WORDS = re.compile(
    r"shortest|youngest|lightest|slowest|smallest|самый (?:низкий|младший|лёгкий|медленный)|самая низкая",
    re.IGNORECASE,
)
NAME = r"[A-ZА-ЯЁ][\wёЁ'-]*"


def solve_ordering(question: str) -> ToolSolution | None:
    superlative_max, superlative_min = MAX_WORDS.search(question), MIN_WORDS.search(question)
    if not (superlative_max or superlative_min):
        return None
    beats: dict[str, set[str]] = defaultdict(set)
    people: set[str] = set()
    words = "|".join((*GREATER_WORDS, *LESSER_WORDS))
    for first, word, second in re.findall(
        rf"({NAME})\s+(?:is\s+)?({words})\s+(?:than\s+)?({NAME})", question
    ):
        winner, loser = (first, second) if word in GREATER_WORDS else (second, first)
        beats[winner].add(loser)
        people.update((winner, loser))
    if len(people) < 2:
        return None
    ranked = sorted(people, key=lambda person: -len(closure(beats, person)))
    top, bottom = ranked[0], ranked[-1]
    if len(closure(beats, top)) != len(people) - 1:
        return None
    if any(bottom not in closure(beats, person) for person in people if person != bottom):
        return None
    answer = top if superlative_max else bottom
    return ToolSolution(answer, [f"order from greatest: {', '.join(ranked)}"], "graph")


def closure(beats: dict[str, set[str]], person: str) -> set[str]:
    seen: set[str] = set()
    stack = list(beats[person])
    while stack:
        current = stack.pop()
        if current not in seen:
            seen.add(current)
            stack.extend(beats[current])
    return seen


def solve_probability(question: str) -> ToolSolution | None:
    if not re.search(r"probabilit|вероятност|ықтималдық", question, re.IGNORECASE):
        return None
    pair = re.search(
        r"(\d+)\s+([a-zа-я]+)\s+(?:and|и|,)\s+(\d+)\s+([a-zа-я]+)", question, re.IGNORECASE
    )
    if pair is None:
        return None
    pool = {pair.group(2).lower(): int(pair.group(1)), pair.group(4).lower(): int(pair.group(3))}
    tail = question[pair.end() :].lower()
    chosen = next((color for color in pool if re.search(rf"\b{re.escape(color)}", tail)), None)
    if chosen is None:
        return None
    total = sum(pool.values())
    steps = [f"{pool[chosen]} / {total}"]
    return ToolSolution(format_value(sympy.Rational(pool[chosen], total)), steps, "calculator")


def solve_arithmetic(question: str) -> ToolSolution | None:
    if "=" in question or re.search(r"\b[xyz]\b", question):
        return None
    spans = re.findall(r"[\d(][\d\s+\-*/^().×÷]*[\d)]|\d", question)
    candidates = [span for span in spans if re.search(r"[+\-*/^×÷]", span.strip())]
    if not candidates:
        return None
    span = max(candidates, key=len)
    value = parse(span)
    if value is None or not value.is_number or not value.is_finite:
        return None
    answer = format_value(value)
    if FRACTION_REQUEST.search(question) and "/" not in answer and value.is_Rational:
        answer = format_value(sympy.Rational(value))
    return ToolSolution(answer, [f"{span.strip()} = {answer}"], "calculator")


SOLVERS: tuple[Solver, ...] = (
    solve_determinant,
    solve_derivative,
    solve_integral,
    solve_inequality,
    solve_ordering,
    solve_probability,
    solve_statistics,
    solve_logarithm,
    solve_percent,
    solve_equation,
    solve_arithmetic,
)


def solve_with_tools(question: str) -> ToolSolution | None:
    if len(question) > MAX_QUESTION_CHARS:
        return None
    for solver in SOLVERS:
        try:
            result = solver(question)
        except (sympy.SympifyError, ValueError, ZeroDivisionError, TypeError, NotImplementedError):
            continue
        if result is not None:
            return result
    return None
