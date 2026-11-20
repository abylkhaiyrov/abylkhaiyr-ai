from __future__ import annotations

import itertools
from collections.abc import Callable
from dataclasses import dataclass

from llm.agent.loop import AgentRun, AgentTask, ChatEngine, run_agent
from llm.data.agentgen import CREATE_PROMPT
from llm.data.lexicon import LANGUAGES


@dataclass(frozen=True)
class BenchmarkFunction:
    name: str
    en: str
    ru: str
    kk: str
    reference: Callable
    visible: tuple
    hidden: tuple


FUNCTIONS = (
    BenchmarkFunction(
        "double_all",
        "returns a new list with every number of xs doubled",
        "возвращает новый список, в котором каждое число xs удвоено",
        "xs тізіміндегі әр санды екі еселеп жаңа тізім қайтарады",
        lambda xs: [2 * x for x in xs],
        (([1, 2],), ([],)),
        (([5, -3, 0],), ([7],), ([10, 10],)),
    ),
    BenchmarkFunction(
        "count_negatives",
        "returns how many numbers in xs are negative",
        "возвращает, сколько чисел в xs отрицательные",
        "xs ішінде неше сан теріс екенін қайтарады",
        lambda xs: sum(x < 0 for x in xs),
        (([1, -2],), ([],)),
        (([-1, -1, 3],), ([0, 0],), ([-5],)),
    ),
    BenchmarkFunction(
        "first_and_last",
        "returns a list with the first and the last element of xs",
        "возвращает список из первого и последнего элементов xs",
        "xs тізімінің бірінші және соңғы элементтерінен тұратын тізім қайтарады",
        lambda xs: [xs[0], xs[-1]],
        (([1, 2, 3],),),
        (([9],), ([4, 5],), ([7, 8, 9, 10],)),
    ),
    BenchmarkFunction(
        "abs_diff",
        "returns the absolute difference between a and b",
        "возвращает модуль разности a и b",
        "a мен b айырмасының модулін қайтарады",
        lambda a, b: abs(a - b),
        ((5, 3), (2, 9)),
        ((0, 0), (-4, 6), (10, -10)),
    ),
    BenchmarkFunction(
        "repeat_word",
        "returns the string word repeated n times separated by single spaces",
        "возвращает строку word, повторённую n раз через одиночные пробелы",
        "word жолын n рет бір бос орынмен бөліп қайтарады",
        lambda word, n: " ".join([word] * n),
        (("ha", 3),),
        (("a", 1), ("go", 2), ("x", 4)),
    ),
    BenchmarkFunction(
        "last_char",
        "returns the last character of the non-empty string s",
        "возвращает последний символ непустой строки s",
        "бос емес s жолының соңғы таңбасын қайтарады",
        lambda s: s[-1],
        (("abc",),),
        (("z",), ("hello",), ("12",)),
    ),
    BenchmarkFunction(
        "middle_element",
        "returns the middle element of xs, which always has an odd length",
        "возвращает средний элемент xs, длина которого всегда нечётная",
        "ұзындығы әрқашан тақ xs тізімінің ортаңғы элементін қайтарады",
        lambda xs: xs[len(xs) // 2],
        (([1, 2, 3],),),
        (([7],), ([1, 5, 9, 2, 4],), ([3, 3, 8],)),
    ),
    BenchmarkFunction(
        "swap_halves",
        "returns the string s of even length with its two halves swapped",
        "возвращает строку s чётной длины, у которой две половины поменяны местами",
        "жұп ұзындықты s жолының екі жартысын ауыстырып қайтарады",
        lambda s: s[len(s) // 2 :] + s[: len(s) // 2],
        (("abcd",),),
        (("ab",), ("xyzw",), ("123456",)),
    ),
    BenchmarkFunction(
        "count_long_words",
        "returns how many words of the sentence s have more than n letters",
        "возвращает, сколько слов предложения s длиннее n букв",
        "s сөйлеміндегі неше сөздің ұзындығы n әріптен артық екенін қайтарады",
        lambda s, n: sum(len(w) > n for w in s.split()),
        (("a bb ccc", 1),),
        (("hello big world", 3), ("a b c", 0), ("", 2)),
    ),
    BenchmarkFunction(
        "sum_of_pairs",
        "returns a list with the sums of neighbouring elements of xs",
        "возвращает список сумм соседних элементов xs",
        "xs тізімінің көрші элементтерінің қосындыларынан тұратын тізім қайтарады",
        lambda xs: [a + b for a, b in itertools.pairwise(xs)],
        (([1, 2, 3],),),
        (([5],), ([1, 1, 1, 1],), ([4, -4, 9],)),
    ),
)


def assertions_for(function: BenchmarkFunction, inputs: tuple) -> str:
    lines = [
        f"assert {function.name}({', '.join(map(repr, args))}) == {function.reference(*args)!r}"
        for args in inputs
    ]
    return "\n".join(lines) + "\n"


def benchmark_tasks() -> list[tuple[str, AgentTask]]:
    tasks = []
    for function in FUNCTIONS:
        for language in LANGUAGES:
            instruction = CREATE_PROMPT[language].format(
                name=function.name,
                description=function.en,
                ru=function.ru,
                kk=function.kk,
            )
            tasks.append(
                (
                    language,
                    AgentTask(
                        f"{function.name}-{language}",
                        instruction,
                        assertions_for(function, function.visible),
                        assertions_for(function, function.hidden),
                    ),
                )
            )
    return tasks


def evaluate_agent(
    engine: ChatEngine, constrained: bool, limit: int | None = None
) -> dict[str, float]:
    runs: list[tuple[str, AgentRun]] = [
        (language, run_agent(engine, task, constrained))
        for language, task in benchmark_tasks()[:limit]
    ]
    calls = sum(run.calls for _, run in runs)
    metrics = {
        "agent_valid_call_rate": sum(run.valid_calls for _, run in runs) / max(calls, 1),
        "agent_finish_rate": sum(run.finished for _, run in runs) / len(runs),
        "agent_visible_pass_rate": sum(run.visible_passed for _, run in runs) / len(runs),
        "agent_hidden_pass_rate": sum(run.hidden_passed for _, run in runs) / len(runs),
        "agent_tasks": float(len(runs)),
    }
    for language in LANGUAGES:
        subset = [run for lang, run in runs if lang == language]
        if subset:
            metrics[f"agent_hidden_pass_{language}"] = sum(r.hidden_passed for r in subset) / len(
                subset
            )
    return metrics
