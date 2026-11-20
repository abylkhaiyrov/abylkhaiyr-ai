import os
import shutil
import subprocess
import time

import pytest

from llm.reasoning.container import run_isolated
from tests.conftest import docker_ready

IMAGE = os.environ.get("ABYLKHAIYR_TEST_IMAGE")
pytestmark = [
    pytest.mark.skipif(IMAGE is None, reason="set ABYLKHAIYR_TEST_IMAGE to test Docker"),
    pytest.mark.skipif(not docker_ready(), reason="Docker daemon is not running"),
]


def run(code: str, stdin: str = "", timeout: float = 25):
    return run_isolated(code, stdin, timeout, IMAGE)


def test_network_is_unreachable():
    result = run(
        "import socket\n"
        "try:\n"
        "    socket.create_connection(('1.1.1.1', 53), timeout=3)\n"
        "    print('connected')\n"
        "except OSError:\n"
        "    print('blocked')"
    )
    assert result.status == "passed" and result.stdout.strip() == "blocked"


def test_root_filesystem_is_read_only_and_user_is_unprivileged():
    result = run(
        "import os\n"
        "print(os.getuid())\n"
        "for path in ('/tmp/x', '/x', '/usr/x'):\n"
        "    try:\n"
        "        open(path, 'w').write('1')\n"
        "        print('wrote', path)\n"
        "    except OSError:\n"
        "        print('denied', path)\n"
        "try:\n"
        "    os.setuid(0)\n"
        "    print('escalated')\n"
        "except OSError:\n"
        "    print('no escalation')"
    )
    lines = result.stdout.split()
    assert result.status == "passed"
    assert lines[0] == "65534" and "wrote" not in result.stdout and "escalated" not in result.stdout


def test_host_environment_secrets_are_not_visible(monkeypatch):
    monkeypatch.setenv("ABYLKHAIYR_SECRET_TOKEN", "hunter2")
    result = run(
        "import os\nprint(os.environ.get('ABYLKHAIYR_SECRET_TOKEN'))\nprint(os.path.exists('/Users'))"
    )
    assert result.stdout.split() == ["None", "False"]


def test_fork_bomb_is_contained_by_the_process_limit():
    result = run(
        "import os, sys\n"
        "count = 0\n"
        "try:\n"
        "    for _ in range(500):\n"
        "        if os.fork() == 0:\n"
        "            import time; time.sleep(5); os._exit(0)\n"
        "        count += 1\n"
        "except OSError:\n"
        "    pass\n"
        "print(count < 100)",
        timeout=20,
    )
    assert result.status in {"passed", "failed"} and "True" in result.stdout


def test_memory_bomb_is_killed_without_affecting_the_host():
    started = time.monotonic()
    result = run("data = bytearray(900 * 1024 * 1024)\nprint(len(data))", timeout=25)
    assert result.status == "failed" and result.stdout.strip() == ""
    assert time.monotonic() - started < 25


def test_output_is_capped_and_infinite_loops_are_removed():
    flood = run("print('x' * 20_000_000)", timeout=25)
    assert len(flood.stdout) <= 8193
    before = subprocess.run(
        [shutil.which("docker") or "docker", "ps", "-aq", "--filter", "name=abylkhaiyr-task-"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()
    assert run("while True: pass", "", 2).status == "timeout"
    after = subprocess.run(
        [shutil.which("docker") or "docker", "ps", "-aq", "--filter", "name=abylkhaiyr-task-"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()
    assert sorted(after) == sorted(before)


def test_code_with_shell_metacharacters_is_not_interpreted_by_a_shell():
    result = run("print('$(touch /tmp/pwned); `id` ; && rm -rf /')")
    assert result.status == "passed"
    assert result.stdout.strip() == "$(touch /tmp/pwned); `id` ; && rm -rf /"
