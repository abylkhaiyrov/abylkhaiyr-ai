from __future__ import annotations

import ast
import json
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, model_validator

from llm.reasoning.sandbox import SandboxLimits, SandboxResult, run_program

TOOL_ARGUMENTS: dict[str, tuple[str, ...]] = {
    "list_files": (),
    "read_file": ("path",),
    "write_file": ("path", "content"),
    "edit_file": ("path", "old", "new"),
    "run_tests": (),
    "finish": ("summary",),
}
ToolName = Literal["list_files", "read_file", "write_file", "edit_file", "run_tests", "finish"]
FILE_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]{0,31}\.py")
MAX_FILE_CHARS = 3000
MAX_OUTPUT_CHARS = 400
TEST_LIMITS = SandboxLimits(timeout_seconds=10.0, cpu_seconds=10, memory_mb=256)
RUNNER = """
import ast as _ast
_results = []
for _index, _line in enumerate({tests!r}.splitlines(), 1):
    if not _line.strip():
        continue
    _node = _ast.parse(_line).body[0]
    _call = _ast.get_source_segment(_line, _node.test.left)
    try:
        _left = eval(compile(_ast.Expression(_node.test.left), "t", "eval"), globals())
        _right = eval(compile(_ast.Expression(_node.test.comparators[0]), "t", "eval"), globals())
    except Exception as _error:
        _results.append((_index, _call, type(_error).__name__ + ": " + str(_error)[:60], None))
        continue
    if _left != _right:
        _results.append((_index, _call, repr(_left)[:60], repr(_right)[:60]))
_total = sum(1 for _line in {tests!r}.splitlines() if _line.strip())
if _results:
    _i, _c, _got, _want = _results[0]
    _text = "FAILED " + str(len(_results)) + " of " + str(_total) + ". First: " + _c
    _text += " returned " + _got if _want is not None else " raised " + _got
    if _want is not None:
        _text += " but expected " + _want
    print(_text)
    raise SystemExit(1)
print("all " + str(_total) + " tests passed")
"""


class ToolCall(BaseModel):
    model_config = ConfigDict(extra="forbid")
    tool: ToolName
    args: dict[str, str]

    @model_validator(mode="after")
    def exact_arguments(self) -> ToolCall:
        if set(self.args) != set(TOOL_ARGUMENTS[self.tool]):
            raise ValueError(f"{self.tool} takes exactly {list(TOOL_ARGUMENTS[self.tool])}")
        return self

    def text(self) -> str:
        return json.dumps({"tool": self.tool, "args": self.args}, ensure_ascii=False)


class ToolOutcome(BaseModel):
    ok: bool
    output: str
    finished: bool = False

    def text(self) -> str:
        return json.dumps({"ok": self.ok, "output": self.output}, ensure_ascii=False)


Runner = Callable[[str], SandboxResult]


def parse_tool_call(text: str) -> ToolCall:
    try:
        return ToolCall.model_validate_json(text.strip())
    except ValueError as error:
        raise ValueError(str(error).splitlines()[0][:120]) from error


def run_in_sandbox(program: str) -> SandboxResult:
    return run_program("python", program, TEST_LIMITS)


@dataclass
class ToolBox:
    workspace: Path
    protected: frozenset[str] = frozenset()
    runner: Runner = run_in_sandbox
    solution: str = "solution.py"
    tests: str = "test_solution.py"

    def _path(self, name: str) -> Path:
        if not FILE_NAME.fullmatch(name):
            raise ValueError("file names are plain python names without directories")
        return self.workspace / name

    def call(self, call: ToolCall) -> ToolOutcome:
        try:
            return getattr(self, f"_{call.tool}")(**call.args)
        except (ValueError, OSError) as error:
            return ToolOutcome(ok=False, output=str(error)[:MAX_OUTPUT_CHARS])

    def _list_files(self) -> ToolOutcome:
        names = sorted(p.name for p in self.workspace.glob("*.py"))
        return ToolOutcome(ok=True, output=", ".join(names) or "empty")

    def _read_file(self, path: str) -> ToolOutcome:
        target = self._path(path)
        if not target.is_file():
            raise ValueError(f"no such file: {path}")
        return ToolOutcome(ok=True, output=target.read_text(encoding="utf-8")[:MAX_FILE_CHARS])

    def _check_writable(self, path: str, content: str) -> Path:
        if path in self.protected:
            raise ValueError(f"{path} is read-only")
        if len(content) > MAX_FILE_CHARS:
            raise ValueError("content too long")
        try:
            ast.parse(content)
        except SyntaxError as error:
            raise ValueError(f"syntax error on line {error.lineno}") from error
        return self._path(path)

    def _write_file(self, path: str, content: str) -> ToolOutcome:
        self._check_writable(path, content).write_text(content, encoding="utf-8")
        return ToolOutcome(ok=True, output=f"written {len(content)} chars to {path}")

    def _edit_file(self, path: str, old: str, new: str) -> ToolOutcome:
        target = self._path(path)
        if path in self.protected or not target.is_file():
            raise ValueError(f"cannot edit {path}")
        text = target.read_text(encoding="utf-8")
        if text.count(old) != 1:
            raise ValueError("old text must occur exactly once")
        updated = text.replace(old, new)
        self._check_writable(path, updated).write_text(updated, encoding="utf-8")
        return ToolOutcome(ok=True, output=f"edited {path}")

    def _run_tests(self) -> ToolOutcome:
        solution, tests = self._path(self.solution), self._path(self.tests)
        if not solution.is_file() or not tests.is_file():
            raise ValueError("solution and tests must both exist")
        result = self.runner(
            solution.read_text(encoding="utf-8")
            + RUNNER.format(tests=tests.read_text(encoding="utf-8"))
        )
        text = (result.stdout or result.stderr).strip().splitlines()
        return ToolOutcome(
            ok=result.passed, output=(text[-1] if text else "no output")[:MAX_OUTPUT_CHARS]
        )

    def _finish(self, summary: str) -> ToolOutcome:
        return ToolOutcome(ok=True, output=summary[:MAX_OUTPUT_CHARS], finished=True)
