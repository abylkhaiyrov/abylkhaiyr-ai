import pytest

from llm.core.errors import SandboxError
from llm.reasoning.engine import Budget, MathVerifier, Problem, ReasoningEngine
from llm.reasoning.sandbox import SandboxLimits, run_python
from llm.reasoning.tools import (
    answers_match,
    evaluate_expression,
    extract_final_answer,
    run_code_tests,
    verify_annotations,
)


def test_sandbox_runs_and_captures_output():
    result = run_python("print(6 * 7)")
    assert result.passed and result.stdout.strip() == "42"


def test_sandbox_enforces_timeout():
    result = run_python("while True: pass", SandboxLimits(timeout_seconds=1.0, cpu_seconds=1))
    assert not result.passed


def test_sandbox_blocks_network_and_shell():
    assert not run_python("import socket; socket.socket().connect(('127.0.0.1', 9))").passed
    assert not run_python("import os; os.system('echo hi')").passed
    assert not run_python("import subprocess; subprocess.Popen(['ls'])").passed


def test_sandbox_runs_in_isolated_directory_without_environment(monkeypatch):
    monkeypatch.setenv("SECRET_TOKEN", "hunter2")
    result = run_python("import os; print(sorted(os.environ)); print(os.getcwd())")
    assert "SECRET_TOKEN" not in result.stdout and "llm-sandbox-" in result.stdout


def test_sandbox_limits_file_size():
    code = "open('big.bin', 'wb').write(b'x' * 5_000_000)"
    assert not run_python(code, SandboxLimits(max_file_mb=1)).passed


def test_code_tests_detect_wrong_solution():
    assert run_code_tests("def add(a, b): return a + b", "assert add(1, 2) == 3").passed
    assert not run_code_tests("def add(a, b): return a - b", "assert add(1, 2) == 3").passed


def test_expression_evaluation_is_safe():
    assert evaluate_expression("2^3 + sqrt(16)") == 12
    assert evaluate_expression("__import__('os').system('ls')") is None
    assert evaluate_expression("open('x')") is None
    assert evaluate_expression("9" * 300) is None


def test_annotation_verification_and_answer_extraction():
    assert verify_annotations("<<48/2=24>> <<3*4=13>>") == (1, 2)
    assert extract_final_answer("so #### 1,024") == "1024"
    assert extract_final_answer("the answer is 7.") == "7"
    assert (
        answers_match("24.0", "24")
        and not answers_match("25", "24")
        and not answers_match(None, "1")
    )


class ScriptedEngine:
    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = 0

    def chat(self, messages, params):
        from llm.inference.engine import Generation

        reply = self.replies[min(self.calls, len(self.replies) - 1)]
        self.calls += 1
        return Generation(reply, list(range(len(reply.split()))), "stop")


def test_reasoning_prefers_verified_answer_over_majority_of_wrong_ones():
    replies = [
        "<<2+2=5>> #### 5",
        "<<2+2=5>> #### 5",
        "<<2+2=4>> #### 4",
        "<<2+2=5>> #### 5",
    ]
    engine = ReasoningEngine(ScriptedEngine(replies))
    solution = engine.solve(Problem("2+2?"), Budget(max_candidates=4, max_retries=0))
    assert solution.answer == "4" and solution.verified


def test_reasoning_retries_after_failure_then_stops_when_verified():
    replies = ["no idea", "still unsure", "<<3*3=9>> #### 9", "unused"]
    scripted = ScriptedEngine(replies)
    solution = ReasoningEngine(scripted).solve(
        Problem("3*3?"), Budget(max_candidates=2, max_retries=2)
    )
    assert solution.answer == "9" and solution.verified and scripted.calls == 3


def test_reasoning_respects_token_budget():
    scripted = ScriptedEngine(["a b c d e f g h"])
    solution = ReasoningEngine(scripted).solve(
        Problem("q"), Budget(max_candidates=10, max_new_tokens_total=16)
    )
    assert solution.stopped_by_budget and scripted.calls == 2


def test_code_problem_uses_sandbox_tests():
    problem = Problem("write add", kind="code", tests="assert add(1, 2) == 3")
    good = ScriptedEngine(["def add(a, b):\n    return a + b"])
    assert ReasoningEngine(good).solve(problem, Budget(max_candidates=1, max_retries=0)).verified
    bad = ScriptedEngine(["def add(a, b):\n    return 0"])
    assert not ReasoningEngine(bad).solve(problem, Budget(max_candidates=1, max_retries=0)).verified


def test_math_verifier_without_annotations_is_unverified():
    candidate = MathVerifier().verify(Problem("q"), "the answer is 4")
    assert candidate.answer == "4" and not candidate.verified


def test_sandbox_error_type_exists():
    assert issubclass(SandboxError, Exception)


@pytest.mark.parametrize("language", ["python", "javascript", "typescript", "java"])
def test_all_languages_run_and_block_writes_outside_workdir(language, tmp_path):
    from llm.reasoning.sandbox import run_program

    programs = {
        "python": ("print(6 * 7)", f"open({str(tmp_path / 'x')!r}, 'w').write('x')"),
        "javascript": (
            "console.log(6 * 7)",
            f"require('fs').writeFileSync({str(tmp_path / 'x')!r}, 'x')",
        ),
        "typescript": (
            "const n: number = 6 * 7; console.log(n)",
            f"require('fs').writeFileSync({str(tmp_path / 'x')!r}, 'x')",
        ),
        "java": (
            "public class Main { public static void main(String[] a) { System.out.println(6 * 7); } }",
            "public class Main { public static void main(String[] a) throws Exception { java.nio.file.Files.writeString(java.nio.file.Path.of("
            + '"'
            + str(tmp_path / "x")
            + '"'
            + '), "x"); } }',
        ),
    }
    good, bad = programs[language]
    assert run_program(language, good).stdout.strip() == "42"
    assert not run_program(language, bad).passed
    assert not (tmp_path / "x").exists()


def test_task_classification():
    from llm.reasoning.engine import classify_task

    assert classify_task("Compute 12 + 7 - 3.") == "math"
    assert classify_task("Write a Python function that reverses a string") == "code"
    assert classify_task("Anna is taller than Bota. Who is the shortest?") == "logic"
    assert classify_task("Tell me about the weather") == "general"


def test_repair_arithmetic_fixes_calculator_errors():
    from llm.reasoning.tools import repair_arithmetic

    fixed = repair_arithmetic("48/2 is <<48/2=25>>25. #### 25")
    assert "<<48/2=24>>24" in fixed and fixed.endswith("#### 24")


def test_logic_verifier_requires_answer_from_the_question():
    from llm.reasoning.engine import LogicVerifier

    problem = Problem("Anna is taller than Bota. Who is the shortest?", "logic")
    assert LogicVerifier().verify(problem, "so\n#### Bota").verified
    assert not LogicVerifier().verify(problem, "so\n#### Zhanar").verified


def test_solve_auto_picks_strategy():
    engine = ReasoningEngine(ScriptedEngine(["<<2+2=5>> #### 5"]))
    solution = engine.solve_auto("Compute 2 + 2.")
    assert solution.answer == "4" and solution.verified
