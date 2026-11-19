from __future__ import annotations

import re
from fractions import Fraction

import sympy
from sympy.parsing.sympy_parser import (
    convert_xor,
    implicit_multiplication_application,
    parse_expr,
    standard_transformations,
)

from llm.reasoning.sandbox import SandboxLimits, SandboxResult, run_python

TRANSFORMATIONS = standard_transformations + (implicit_multiplication_application, convert_xor)
FUNCTION_NAMES = ("sqrt", "sin", "cos", "tan", "exp", "log", "pi", "E", "factorial")
CONSTRUCTOR_NAMES = ("Integer", "Float", "Rational", "Symbol", "Function", "Mul", "Add", "Pow")
SAFE_NAMESPACE = {name: getattr(sympy, name) for name in (*FUNCTION_NAMES, *CONSTRUCTOR_NAMES)}
ANNOTATION_PATTERN = re.compile(r"<<([^<>=]+)=([^<>]+)>>")
FINAL_ANSWER_PATTERN = re.compile(r"####\s*(-?[\d,]*\.?\d+)")
NUMBER_PATTERN = re.compile(r"-?[\d,]*\.?\d+")
SAFE_EXPRESSION = re.compile(r"^[\d\s+\-*/().,^%a-zA-Z_]*$")
MAX_EXPRESSION_CHARS = 200
TOLERANCE = 1e-6


def evaluate_expression(expression: str) -> sympy.Expr | None:
    if len(expression) > MAX_EXPRESSION_CHARS or not SAFE_EXPRESSION.match(expression):
        return None
    try:
        return parse_expr(
            expression.replace(",", ""),
            local_dict=SAFE_NAMESPACE,
            global_dict={"__builtins__": {}},
            transformations=TRANSFORMATIONS,
        )
    except Exception:
        return None


def to_number(value: str) -> Fraction | None:
    cleaned = value.strip().replace(",", "").rstrip(".")
    try:
        return Fraction(cleaned)
    except (ValueError, ZeroDivisionError):
        return None


def numbers_equal(left: sympy.Expr | Fraction, right: sympy.Expr | Fraction) -> bool:
    try:
        return abs(float(left) - float(right)) <= TOLERANCE * max(1.0, abs(float(right)))
    except (TypeError, ValueError):
        return False


def verify_annotations(text: str) -> tuple[int, int]:
    correct = total = 0
    for expression, claimed in ANNOTATION_PATTERN.findall(text):
        total += 1
        value, claim = evaluate_expression(expression), to_number(claimed)
        if value is not None and claim is not None and value.is_number and numbers_equal(value, claim):
            correct += 1
    return correct, total


def extract_final_answer(text: str) -> str | None:
    match = FINAL_ANSWER_PATTERN.search(text)
    if match:
        return match.group(1).replace(",", "")
    numbers = NUMBER_PATTERN.findall(text)
    return numbers[-1].replace(",", "") if numbers else None


def answers_match(candidate: str | None, reference: str | None) -> bool:
    if candidate is None or reference is None:
        return False
    left, right = to_number(candidate), to_number(reference)
    return left is not None and right is not None and numbers_equal(left, right)


def run_code_tests(code: str, tests: str, limits: SandboxLimits | None = None) -> SandboxResult:
    return run_python(f"{code}\n\n{tests}\n", limits)
