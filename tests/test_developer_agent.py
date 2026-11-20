import os

import pytest

from llm.reasoning.container import run_isolated


def test_container_execution_uses_stdin_and_has_no_host_mount(tmp_path):
    image = os.environ.get("ABYLKHAIYR_TEST_IMAGE")
    if image is None:
        pytest.skip("set ABYLKHAIYR_TEST_IMAGE to test Docker")
    secret = tmp_path / "private.txt"
    secret.write_text("isolated")
    program = (
        "from pathlib import Path\n"
        f"assert not Path({secret.as_posix()!r}).exists()\n"
        "assert not Path('/var/run/docker.sock').exists()\n"
        "assert set(Path('/sys/class/net').iterdir()) == {Path('/sys/class/net/lo')}\n"
        "print(sum(map(int, input().split())))"
    )
    result = run_isolated(program, "11 -4\n", 10, image)
    assert result.status == "passed", result
    assert result.stdout == "7\n"
    assert run_isolated("while True: pass", "", 2, image).status == "timeout"


def test_container_requires_pin():
    with pytest.raises(ValueError):
        run_isolated("print(1)", "", 1, "python:latest")
