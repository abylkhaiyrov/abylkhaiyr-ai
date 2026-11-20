import json
from types import SimpleNamespace

import pytest

from llm.agent.benchmark import FUNCTIONS, assertions_for, benchmark_tasks
from llm.agent.loop import run_agent
from llm.agent.tools import ToolCall, run_in_sandbox
from llm.data.codegen import ALL_SPECS
from llm.evaluation import suites
from llm.inference.engine import Generation

if not run_in_sandbox("print(1)").passed:
    pytest.skip("sandbox unavailable", allow_module_level=True)


def call(tool, **args):
    return ToolCall(tool=tool, args=args).text()


class ScriptedChat:
    tokenizer = SimpleNamespace()

    def __init__(self, replies):
        self.replies = list(replies)

    def chat(self, messages, params, constraint=None):
        text = self.replies.pop(0) if self.replies else call("finish", summary="stop")
        return Generation(text, [], "stop")


def reference_solution(function) -> str:
    return {
        "double_all": "def double_all(xs):\n    return [2 * x for x in xs]\n",
        "count_negatives": "def count_negatives(xs):\n    return sum(x < 0 for x in xs)\n",
        "first_and_last": "def first_and_last(xs):\n    return [xs[0], xs[-1]]\n",
        "abs_diff": "def abs_diff(a, b):\n    return abs(a - b)\n",
        "repeat_word": "def repeat_word(word, n):\n    return ' '.join([word] * n)\n",
        "last_char": "def last_char(s):\n    return s[-1]\n",
        "middle_element": "def middle_element(xs):\n    return xs[len(xs) // 2]\n",
        "swap_halves": "def swap_halves(s):\n    h = len(s) // 2\n    return s[h:] + s[:h]\n",
        "count_long_words": "def count_long_words(s, n):\n    return sum(len(w) > n for w in s.split())\n",
        "sum_of_pairs": "def sum_of_pairs(xs):\n    return [a + b for a, b in zip(xs, xs[1:])]\n",
    }[function.name]


def oracle(function):
    return ScriptedChat(
        [
            call("write_file", path="solution.py", content=reference_solution(function)),
            call("run_tests"),
            call("finish", summary="done"),
        ]
    )


def test_every_benchmark_task_is_solvable_and_graded_by_hidden_tests():
    by_name = {task.identifier.rsplit("-", 1)[0]: task for _, task in benchmark_tasks()}
    for function in FUNCTIONS:
        run = run_agent(oracle(function), by_name[function.name], constrained=False)
        assert run.finished and run.visible_passed and run.hidden_passed, function.name
        assert run.valid_calls == 3 and run.tools == ["write_file", "run_tests", "finish"]


def test_hardcoding_visible_answers_passes_visible_tests_but_not_hidden_ones():
    function = FUNCTIONS[0]
    task = next(t for _, t in benchmark_tasks() if t.identifier == "double_all-en")
    cheat = "def double_all(xs):\n    return [2, 4] if xs else []\n"
    engine = ScriptedChat(
        [
            call("write_file", path="solution.py", content=cheat),
            call("run_tests"),
            call("finish", summary="x"),
        ]
    )
    run = run_agent(engine, task, constrained=False)
    assert run.visible_passed and run.finished and not run.hidden_passed
    assert assertions_for(function, function.hidden) != task.visible_tests


def test_agent_cannot_weaken_the_visible_tests_and_invalid_calls_are_counted():
    task = next(t for _, t in benchmark_tasks() if t.identifier == "abs_diff-en")
    engine = ScriptedChat(
        [
            call("edit_file", path="test_solution.py", old="== 2", new="== 0"),
            "not json at all",
            call("write_file", path="../outside.py", content="x = 1\n"),
            call("finish", summary="giving up"),
        ]
    )
    run = run_agent(engine, task, constrained=False)
    assert run.calls == 4 and run.valid_calls == 3 and not run.hidden_passed
    assert run.failure is not None


def test_step_budget_is_enforced():
    task = next(t for _, t in benchmark_tasks() if t.identifier == "last_char-en")
    engine = ScriptedChat([call("list_files")] * 20)
    run = run_agent(engine, task, constrained=False, max_steps=5)
    assert run.calls == 5 and not run.finished


def test_benchmark_is_independent_of_training_specs_and_other_benchmarks():
    names = {function.name for function in FUNCTIONS}
    assert not names & {spec.name for spec in ALL_SPECS}
    assert not names & {case.function_name for case in suites.CODE_CASES}
    assert len(benchmark_tasks()) == 3 * len(FUNCTIONS)
    assert json.dumps(assertions_for(FUNCTIONS[0], FUNCTIONS[0].visible))
