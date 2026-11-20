import pytest

from llm.agent.tools import ToolBox, ToolCall, parse_tool_call, run_in_sandbox

TESTS = "assert add(1, 2) == 3\nassert add(5, -1) == 4\n"
GOOD = "def add(a, b):\n    return a + b\n"
BAD = "def add(a, b):\n    return a - b\n"


@pytest.fixture()
def box(tmp_path):
    probe = run_in_sandbox("print(1)")
    if not probe.passed:
        pytest.skip(f"sandbox unavailable: {probe.stderr}")
    (tmp_path / "test_solution.py").write_text(TESTS)
    return ToolBox(tmp_path, frozenset({"test_solution.py"}))


def call(box, tool, **args):
    return box.call(ToolCall(tool=tool, args=args))


def test_write_run_and_failure_message_name_the_failing_call(box):
    assert call(box, "write_file", path="solution.py", content=BAD).ok
    failed = call(box, "run_tests")
    assert not failed.ok
    assert (
        "add(1, 2)" in failed.output
        and "returned -1" in failed.output
        and "expected 3" in failed.output
    )
    assert call(box, "edit_file", path="solution.py", old="a - b", new="a + b").ok
    passed = call(box, "run_tests")
    assert passed.ok and passed.output == "all 2 tests passed"
    assert call(box, "finish", summary="done").finished


def test_protected_files_paths_and_invalid_code_are_rejected(box):
    for bad_path in ("../escape.py", "/etc/passwd", "sub/dir.py", "no_extension", "a b.py"):
        assert not call(box, "write_file", path=bad_path, content=GOOD).ok
        assert not call(box, "read_file", path=bad_path).ok
    assert not call(box, "write_file", path="test_solution.py", content="assert True\n").ok
    assert not call(box, "edit_file", path="test_solution.py", old="== 3", new="== 4").ok
    assert not call(box, "write_file", path="solution.py", content="def broken(:\n").ok
    assert not call(box, "write_file", path="solution.py", content="x = 1\n" * 2000).ok
    assert (box.workspace / "test_solution.py").read_text() == TESTS
    assert not list(box.workspace.parent.glob("escape.py"))


def test_edit_requires_a_unique_match_and_run_tests_needs_both_files(box):
    assert not call(box, "run_tests").ok
    call(
        box, "write_file", path="solution.py", content="def add(a, b):\n    return a + b + 0 + 0\n"
    )
    assert not call(box, "edit_file", path="solution.py", old="+ 0", new="").ok
    assert not call(box, "read_file", path="missing.py").ok
    assert call(box, "list_files").output == "solution.py, test_solution.py"


def test_tool_call_schema_is_strict():
    assert parse_tool_call('{"tool": "run_tests", "args": {}}').tool == "run_tests"
    for text in (
        '{"tool": "run_tests", "args": {"x": "1"}}',
        '{"tool": "rm_rf", "args": {}}',
        '{"tool": "write_file", "args": {"path": "a.py"}}',
        "not json",
        '{"tool": "finish", "args": {"summary": "ok"}, "extra": 1}',
    ):
        with pytest.raises(ValueError):
            parse_tool_call(text)
    call_ = ToolCall(tool="write_file", args={"path": "a.py", "content": "print('é')\n"})
    assert parse_tool_call(call_.text()) == call_
