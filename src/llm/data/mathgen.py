from __future__ import annotations

import json
import random
from collections.abc import Callable
from dataclasses import asdict, dataclass
from fractions import Fraction
from pathlib import Path

import sympy

from llm.data.dedup import stable_hash
from llm.reasoning.tools import (
    answers_match,
    extract_final_answer,
    extract_text_answer,
    verify_annotations,
)

MAX_LEVEL = 5
MATH_BENCH_FILE = "math_bench.jsonl"
LEVELS = tuple(range(1, MAX_LEVEL + 1))
X = sympy.Symbol("x")
Y = sympy.Symbol("y")


@dataclass(frozen=True)
class MathProblem:
    category: str
    level: int
    question: str
    solution: str
    answer: str
    check: str

    @property
    def key(self) -> int:
        return stable_hash(" ".join(self.question.lower().split()))


def fmt(value: object) -> str:
    number = sympy.nsimplify(value) if not isinstance(value, sympy.Basic) else value
    if isinstance(number, sympy.Rational) and number.q == 1:
        return str(int(number))
    return str(number)


def signed(value: int) -> str:
    return f"+ {value}" if value >= 0 else f"- {-value}"


def bound(level: int) -> int:
    return 10**level


def arithmetic(rng: random.Random, level: int) -> MathProblem:
    a, b, c = (rng.randint(2, bound(level)) for _ in range(3))
    op1, op2 = rng.choice("+-*"), rng.choice("+-")
    first = f"{a}{op1}{b}"
    first_value = sympy.sympify(first)
    total = sympy.sympify(f"{first_value}{op2}{c}")
    question = f"Compute {a} {op1} {b} {op2} {c}."
    solution = (
        f"First {a} {op1} {b} = <<{first}={first_value}>>{first_value}.\n"
        f"Then {first_value} {op2} {c} = <<{first_value}{op2}{c}={total}>>{total}.\n#### {total}"
    )
    return MathProblem("arithmetic", level, question, solution, str(total), f"{a}{op1}{b}{op2}{c}")


def fractions(rng: random.Random, level: int) -> MathProblem:
    top = 3 + level * 6
    a = Fraction(rng.randint(1, top), rng.randint(2, top))
    b = Fraction(rng.randint(1, top), rng.randint(2, top))
    op = rng.choice("+-*")
    expr = f"({a.numerator}/{a.denominator}){op}({b.numerator}/{b.denominator})"
    value = sympy.sympify(expr)
    question = f"Compute {a} {op} {b} and give the result as a simplified fraction."
    solution = f"{a} {op} {b} = <<{expr}={value}>>{value}.\n#### {value}"
    return MathProblem("fractions", level, question, solution, str(value), expr)


def percent(rng: random.Random, level: int) -> MathProblem:
    base = rng.randint(2, 20 * level) * 10
    rate = rng.choice([5, 10, 15, 20, 25, 30, 40, 50, 75])
    value = sympy.Rational(base * rate, 100)
    question = f"What is {rate}% of {base}?"
    solution = f"{rate}% of {base} = <<{base}*{rate}/100={value}>>{value}.\n#### {value}"
    return MathProblem("percent", level, question, solution, str(value), f"{base}*{rate}/100")


def linear_equation(rng: random.Random, level: int) -> MathProblem:
    solution_x = rng.randint(-bound(min(level, 2)), bound(min(level, 2)))
    a = rng.choice([n for n in range(-9 * level, 9 * level + 1) if n not in (0, 1)])
    b = rng.randint(-20 * level, 20 * level)
    rhs = a * solution_x + b
    question = f"Solve for x: {a}*x {signed(b)} = {rhs}."
    step = rhs - b
    solution = (
        f"Subtract {b}: {a}*x = <<{rhs}-({b})={step}>>{step}.\n"
        f"Divide by {a}: x = <<{step}/({a})={solution_x}>>{solution_x}.\n#### {solution_x}"
    )
    return MathProblem(
        "linear_equation", level, question, solution, str(solution_x), f"{a}*x+{b}-({rhs})"
    )


def quadratic(rng: random.Random, level: int) -> MathProblem:
    r1, r2 = sorted(rng.sample(range(-3 * level, 3 * level + 1), 2))
    b, c = -(r1 + r2), r1 * r2
    question = f"Find the smaller root of x^2 {signed(b).replace(' ', '', 1)[0:1]} {abs(b)}*x {signed(c)} = 0."
    solution = (
        f"The roots satisfy (x - {r1})(x - {r2}) = 0, so the roots are {r1} and {r2}.\n"
        f"The smaller root is {r1}.\n#### {r1}"
    )
    return MathProblem("quadratic", level, question, solution, str(r1), f"x**2+({b})*x+({c})")


def system(rng: random.Random, level: int) -> MathProblem:
    x, y = rng.randint(-5 * level, 5 * level), rng.randint(-5 * level, 5 * level)
    a, b, c, d = (rng.randint(1, 3 + level) for _ in range(4))
    while a * d - b * c == 0:
        d = rng.randint(1, 3 + level)
    e, f = a * x + b * y, c * x - d * y
    question = f"Solve the system {a}*x + {b}*y = {e}, {c}*x - {d}*y = {f}. Give the value of x."
    numerator, denominator = e * d + b * f, a * d + b * c
    solution = (
        f"Eliminating y gives x = ({e}*{d}+{b}*({f}))/({a}*{d}+{b}*{c}) = "
        f"<<({numerator})/({denominator})={x}>>{x}.\n#### {x}"
    )
    return MathProblem(
        "system", level, question, solution, str(x), f"{a}*x+{b}*y-({e});{c}*x-{d}*y-({f})"
    )


def inequality(rng: random.Random, level: int) -> MathProblem:
    a = rng.randint(2, 4 + level)
    b = rng.randint(-10 * level, 10 * level)
    rhs = rng.randint(-10 * level, 10 * level)
    bound_value = sympy.Rational(rhs - b, a)
    question = f"What is the smallest integer x satisfying {a}*x {signed(b)} > {rhs}?"
    smallest = int(sympy.floor(bound_value)) + 1
    solution = (
        f"{a}*x > <<{rhs}-({b})={rhs - b}>>{rhs - b}, so x > {bound_value}.\n"
        f"The smallest integer is {smallest}.\n#### {smallest}"
    )
    return MathProblem("inequality", level, question, solution, str(smallest), f"{a}*x+{b}>{rhs}")


def powers_logs(rng: random.Random, level: int) -> MathProblem:
    base, exponent = rng.randint(2, 3 + level), rng.randint(2, 2 + level)
    if rng.random() < 0.5:
        value = base**exponent
        question = f"Compute {base}^{exponent}."
        solution = f"{base}^{exponent} = <<{base}**{exponent}={value}>>{value}.\n#### {value}"
        return MathProblem("powers", level, question, solution, str(value), f"{base}**{exponent}")
    argument = base**exponent
    question = f"Compute log base {base} of {argument}."
    solution = (
        f"Since {base}^{exponent} = {argument}, the logarithm is {exponent}.\n#### {exponent}"
    )
    return MathProblem(
        "logarithms", level, question, solution, str(exponent), f"log({argument},{base})"
    )


def matrices(rng: random.Random, level: int) -> MathProblem:
    size = 2 if level < 4 else 3
    span = 2 + level
    rows = [[rng.randint(-span, span) for _ in range(size)] for _ in range(size)]
    text = "[" + ", ".join("[" + ", ".join(map(str, r)) + "]" for r in rows) + "]"
    determinant = sympy.Matrix(rows).det()
    question = f"Compute the determinant of the matrix {text}."
    if size == 2:
        (p, q), (r, t) = rows
        solution = f"det = {p}*{t} - ({q})*({r}) = <<{p}*{t}-({q})*({r})={determinant}>>{determinant}.\n#### {determinant}"
    else:
        solution = f"Expanding the determinant of {text} gives {determinant}.\n#### {determinant}"
    return MathProblem("matrices", level, question, solution, str(determinant), text)


def derivative(rng: random.Random, level: int) -> MathProblem:
    a, n = rng.randint(1, 3 + level), rng.randint(2, 2 + level)
    point, k = rng.randint(1, 3), rng.randint(1, 9)
    expr = a * X**n + k * X
    function_text = str(expr).replace("**", "^")
    value = sympy.diff(expr, X).subs(X, point)
    question = f"Let f(x) = {function_text}. Compute f'({point})."
    solution = (
        f"f'(x) = {sympy.diff(expr, X)}. At x = {point}: "
        f"<<{n * a}*{point}**{n - 1}+{k}={value}>>{value}.\n#### {value}"
    )
    return MathProblem("derivative", level, question, solution, str(value), f"{expr}|{point}")


def integral(rng: random.Random, level: int) -> MathProblem:
    a, n = rng.randint(1, 3 + level), rng.randint(1, 1 + level)
    upper = rng.randint(1, 3 + level)
    expr = a * X**n
    value = sympy.integrate(expr, (X, 0, upper))
    question = f"Compute the definite integral of {str(expr).replace('**', '^')} from 0 to {upper}."
    solution = (
        f"An antiderivative is {sympy.integrate(expr, X)}. At {upper}: "
        f"<<{a}*{upper}**{n + 1}/{n + 1}={value}>>{value}.\n#### {value}"
    )
    return MathProblem("integral", level, question, solution, str(value), f"{expr}|{upper}")


def probability(rng: random.Random, level: int) -> MathProblem:
    red, blue = rng.randint(1, 3 + level), rng.randint(1, 3 + level)
    total = red + blue
    value = sympy.Rational(red, total)
    question = (
        f"A bag has {red} red and {blue} blue balls. What is the probability of drawing a red ball?"
    )
    solution = f"There are {total} balls, so P = <<{red}/{total}={value}>>{value}.\n#### {value}"
    return MathProblem("probability", level, question, solution, str(value), f"{red}/{total}")


def statistics(rng: random.Random, level: int) -> MathProblem:
    values = [rng.randint(1, 10 * level) for _ in range(rng.randint(3, 4 + level))]
    mean = sympy.Rational(sum(values), len(values))
    question = f"What is the mean of the numbers {', '.join(map(str, values))}?"
    total = "+".join(map(str, values))
    solution = (
        f"The sum is <<{total}={sum(values)}>>{sum(values)}. "
        f"The mean is <<{sum(values)}/{len(values)}={mean}>>{mean}.\n#### {mean}"
    )
    return MathProblem(
        "statistics", level, question, solution, str(mean), ",".join(map(str, values))
    )


def logic(rng: random.Random, level: int) -> MathProblem:
    count = 2 + level
    order = rng.sample(["Anna", "Bota", "Daulet", "Erlan", "Aigul", "Timur", "Dana"], count)
    pairs = [(order[i], order[i + 1]) for i in range(count - 1)]
    rng.shuffle(pairs)
    clues = " ".join(f"{a} is taller than {b}." for a, b in pairs)
    question = f"{clues} Who is the shortest?"
    solution = f"The shortest person is the one who is never taller than anyone: {order[-1]}.\n#### {order[-1]}"
    spec = ";".join(f"{a}>{b}" for a, b in pairs)
    return MathProblem("logic", level, question, solution, order[-1], spec)


GENERATORS: dict[str, Callable[[random.Random, int], MathProblem]] = {
    "arithmetic": arithmetic,
    "fractions": fractions,
    "percent": percent,
    "linear_equation": linear_equation,
    "quadratic": quadratic,
    "system": system,
    "inequality": inequality,
    "powers": powers_logs,
    "matrices": matrices,
    "derivative": derivative,
    "integral": integral,
    "probability": probability,
    "statistics": statistics,
    "logic": logic,
}
TEXT_ANSWER_CATEGORIES = {"logic"}


def sympy_answer(problem: MathProblem) -> str:
    category, spec = problem.category, problem.check
    if category in {"arithmetic", "fractions", "percent", "powers", "probability"}:
        return str(sympy.sympify(spec))
    if category == "logarithms":
        return str(sympy.sympify(spec).doit())
    if category == "linear_equation":
        return str(sympy.solve(sympy.sympify(spec), X)[0])
    if category == "quadratic":
        return str(min(sympy.solve(sympy.sympify(spec), X)))
    if category == "system":
        left, right = (sympy.sympify(part) for part in spec.split(";"))
        return str(sympy.solve([left, right], [X, Y])[X])
    if category == "inequality":
        relation = sympy.sympify(spec, evaluate=False)
        candidate = sympy.solve_univariate_inequality(relation, X, relational=False)
        return str(int(sympy.floor(candidate.inf)) + 1)
    if category == "matrices":
        return str(sympy.Matrix(json.loads(spec)).det())
    if category == "derivative":
        expression, point = spec.split("|")
        return str(sympy.diff(sympy.sympify(expression), X).subs(X, int(point)))
    if category == "integral":
        expression, upper = spec.split("|")
        return str(sympy.integrate(sympy.sympify(expression), (X, 0, int(upper))))
    if category == "statistics":
        numbers = [int(n) for n in spec.split(",")]
        return str(sympy.Rational(sum(numbers), len(numbers)))
    if category == "logic":
        pairs = [clue.split(">") for clue in spec.split(";")]
        taller = {a for a, _ in pairs}
        return next(b for _, b in pairs if b not in taller)
    raise ValueError(f"unknown category {category}")


def verify_problem(problem: MathProblem) -> bool:
    final = extract_final_answer(problem.solution)
    if problem.category in TEXT_ANSWER_CATEGORIES:
        return extract_text_answer(problem.solution) == problem.answer == sympy_answer(problem)
    correct, total = verify_annotations(problem.solution)
    return (
        correct == total
        and answers_match(final, sympy_answer(problem))
        and answers_match(problem.answer, sympy_answer(problem))
    )


def build_dataset(
    count: int, seed: int, exclude: set[int] | None = None, levels: tuple[int, ...] = LEVELS
) -> list[MathProblem]:
    rng = random.Random(seed)
    excluded = set(exclude or ())
    names = list(GENERATORS)
    problems: list[MathProblem] = []
    attempts = 0
    while len(problems) < count and attempts < count * 30:
        attempts += 1
        problem = GENERATORS[names[attempts % len(names)]](rng, rng.choice(levels))
        if problem.key in excluded or not verify_problem(problem):
            continue
        excluded.add(problem.key)
        problems.append(problem)
    return problems


def curriculum_order(problems: list[MathProblem], seed: int) -> list[MathProblem]:
    rng = random.Random(seed)
    count = len(problems)
    by_level = {level: [p for p in problems if p.level == level] for level in LEVELS}
    for items in by_level.values():
        rng.shuffle(items)
    ordered: list[MathProblem] = []
    while any(by_level.values()):
        progress = len(ordered) / max(count, 1)
        weights = [
            (
                max(0.0, 1.0 - abs(level - 1 - progress * (MAX_LEVEL - 1)) / 2)
                if by_level[level]
                else 0.0
            )
            for level in LEVELS
        ]
        if not any(weights):
            weights = [1.0 if by_level[level] else 0.0 for level in LEVELS]
        level = rng.choices(LEVELS, weights)[0]
        ordered.append(by_level[level].pop())
    return ordered


def write_dataset(problems: list[MathProblem], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for problem in problems:
            handle.write(json.dumps(asdict(problem), ensure_ascii=False) + "\n")


def read_dataset(path: Path) -> list[MathProblem]:
    with path.open(encoding="utf-8") as handle:
        return [MathProblem(**json.loads(line)) for line in handle]


def answer_is_correct(problem: MathProblem, answer: str | None) -> bool:
    if answer is None:
        return False
    if problem.category in TEXT_ANSWER_CATEGORIES:
        return answer.strip().lower() == problem.answer.lower()
    return answers_match(answer, problem.answer)
