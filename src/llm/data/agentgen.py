from __future__ import annotations

import random
import tempfile
from collections.abc import Callable
from pathlib import Path

from llm.agent.tools import ToolBox, ToolCall, ToolName
from llm.data.codegen import ALL_SPECS
from llm.data.codespecs import Spec
from llm.data.lexicon import LANGUAGES

SOLUTION = "solution.py"
TESTS = "test_solution.py"
CREATE_PROMPT = {
    "en": "Create solution.py with a function {name} that {description}. The tests are in test_solution.py. Use the tools and call finish when the tests pass.",
    "ru": "Создай solution.py с функцией {name}, которая {ru}. Тесты лежат в test_solution.py. Используй инструменты и вызови finish, когда тесты пройдут.",
    "kk": "solution.py файлын жасап, ішіне {name} функциясын жаз, ол {kk}. Тесттер test_solution.py ішінде. Құралдарды қолданып, тесттер өткен соң finish шақыр.",
}
FIX_PROMPT = {
    "en": "The function {name} in solution.py has a bug. Make the tests in test_solution.py pass using the tools, then call finish.",
    "ru": "В функции {name} в solution.py есть ошибка. С помощью инструментов добейся, чтобы тесты из test_solution.py прошли, затем вызови finish.",
    "kk": "solution.py ішіндегі {name} функциясында қате бар. Құралдарды қолданып test_solution.py тесттерін өткіз, содан кейін finish шақыр.",
}
SUMMARY = {
    "en": "{name} is implemented and all tests pass.",
    "ru": "Функция {name} реализована, все тесты проходят.",
    "kk": "{name} функциясы жасалды, барлық тесттер өтті.",
}


def tests_text(spec: Spec) -> str:
    lines = [
        f"assert {spec.name}({', '.join(map(repr, args))}) == {spec.reference(*args)!r}"
        for args in spec.inputs
    ]
    return "\n".join(lines) + "\n"


class Recorder:
    def __init__(self, workspace: Path, language: str, prompt: str) -> None:
        self.box = ToolBox(workspace, frozenset({TESTS}))
        self.language = language
        self.messages: list[dict] = [{"role": "user", "content": prompt}]
        self.ok = True
        self.tests_pass = False

    def step(self, tool: ToolName, supervised: bool = True, **args: str) -> None:
        call = ToolCall(tool=tool, args=args)
        outcome = self.box.call(call)
        assistant: dict = {"role": "assistant", "content": call.text()}
        if not supervised:
            assistant["supervised"] = False
        self.messages.extend([assistant, {"role": "tool", "content": outcome.text()}])
        if tool == "run_tests":
            self.tests_pass = outcome.ok
        else:
            self.ok = self.ok and outcome.ok


def prepare(workspace: Path, spec: Spec, initial: str | None) -> None:
    (workspace / TESTS).write_text(tests_text(spec), encoding="utf-8")
    if initial is not None:
        (workspace / SOLUTION).write_text(initial, encoding="utf-8")


def finish(recorder: Recorder, spec: Spec) -> None:
    recorder.step("finish", summary=SUMMARY[recorder.language].format(name=spec.name))


def success(rec: Recorder, spec: Spec, rng: random.Random) -> None:
    rec.step("write_file", path=SOLUTION, content=spec.python)
    rec.step("run_tests")
    finish(rec, spec)


def read_tests_first(rec: Recorder, spec: Spec, rng: random.Random) -> None:
    rec.step("read_file", path=TESTS)
    success(rec, spec, rng)


def fix_preloaded(rec: Recorder, spec: Spec, rng: random.Random) -> None:
    assert spec.mutant is not None
    rec.step("run_tests")
    rec.step("read_file", path=SOLUTION)
    rec.step("edit_file", path=SOLUTION, old=spec.mutant[1], new=spec.mutant[0])
    rec.step("run_tests")
    finish(rec, spec)


def self_correct(rec: Recorder, spec: Spec, rng: random.Random) -> None:
    assert spec.mutant is not None
    rec.step(
        "write_file", supervised=False, path=SOLUTION, content=spec.python.replace(*spec.mutant)
    )
    rec.step("run_tests")
    rec.step("edit_file", path=SOLUTION, old=spec.mutant[1], new=spec.mutant[0])
    rec.step("run_tests")
    finish(rec, spec)


Flow = Callable[[Recorder, Spec, random.Random], None]
FLOWS: dict[str, tuple[Flow, bool, bool]] = {
    "write": (success, False, False),
    "read_then_write": (read_tests_first, False, False),
    "fix_preloaded": (fix_preloaded, True, True),
    "self_correct": (self_correct, True, False),
}


def trajectory(kind: str, spec: Spec, language: str, rng: random.Random) -> dict | None:
    flow, needs_mutant, preload = FLOWS[kind]
    if needs_mutant and (
        spec.mutant is None or spec.python.replace(*spec.mutant).count(spec.mutant[1]) != 1
    ):
        return None
    initial = spec.python.replace(*spec.mutant) if preload and spec.mutant else None
    template = FIX_PROMPT if preload else CREATE_PROMPT
    prompt = template[language].format(
        name=spec.name, description=spec.description, ru=spec.ru, kk=spec.kk
    )
    with tempfile.TemporaryDirectory(prefix="agent-trajectory-") as directory:
        workspace = Path(directory)
        prepare(workspace, spec, initial)
        recorder = Recorder(workspace, language, prompt)
        flow(recorder, spec, rng)
        verified = recorder.ok and recorder.tests_pass
    if not verified:
        return None
    return {
        "source": f"synthetic_agent_{kind}",
        "group": "agent",
        "category": f"agent_{kind}",
        "language": language,
        "messages": recorder.messages[:-1],
    }


def build_agent_dataset(count: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    combos = [
        (kind, spec, language) for kind in FLOWS for spec in ALL_SPECS for language in LANGUAGES
    ]
    rng.shuffle(combos)
    rows: list[dict] = []
    for kind, spec, language in combos:
        if len(rows) >= count:
            break
        row = trajectory(kind, spec, language, rng)
        if row is not None:
            rows.append(row)
    return rows
