from __future__ import annotations

import os
import resource
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

from llm.core.errors import SandboxError

BYTES_PER_MB = 1024 * 1024
RESTRICTED_ENV = {"PATH": "/usr/bin:/bin", "PYTHONHASHSEED": "0", "PYTHONDONTWRITEBYTECODE": "1"}
MAX_OUTPUT_CHARS = 4000
PREAMBLE = """
import builtins, os, socket, subprocess
def _blocked(*args, **kwargs):
    raise PermissionError("blocked in sandbox")
socket.socket.connect = _blocked
socket.socket.bind = _blocked
subprocess.Popen = _blocked
for _name in ("system", "popen", "execv", "execve", "fork", "forkpty", "spawnv", "kill", "remove", "unlink", "rmdir"):
    if hasattr(os, _name):
        setattr(os, _name, _blocked)
"""


@dataclass(frozen=True)
class SandboxLimits:
    timeout_seconds: float = 5.0
    memory_mb: int = 512
    cpu_seconds: int = 5
    max_file_mb: int = 1
    max_open_files: int = 32


@dataclass(frozen=True)
class SandboxResult:
    passed: bool
    stdout: str
    stderr: str
    timed_out: bool


def _apply_limits(limits: SandboxLimits):
    def apply() -> None:
        resource.setrlimit(resource.RLIMIT_CPU, (limits.cpu_seconds, limits.cpu_seconds))
        resource.setrlimit(resource.RLIMIT_FSIZE, (limits.max_file_mb * BYTES_PER_MB,) * 2)
        resource.setrlimit(resource.RLIMIT_NOFILE, (limits.max_open_files,) * 2)
        if sys.platform.startswith("linux"):
            resource.setrlimit(resource.RLIMIT_AS, (limits.memory_mb * BYTES_PER_MB,) * 2)
        os.setsid()

    return apply


def run_python(code: str, limits: SandboxLimits | None = None) -> SandboxResult:
    limits = limits or SandboxLimits()
    with tempfile.TemporaryDirectory(prefix="llm-sandbox-") as workdir:
        script = Path(workdir) / "main.py"
        script.write_text(PREAMBLE + "\n" + code, encoding="utf-8")
        try:
            completed = subprocess.run(
                [sys.executable, "-I", "-S", str(script)],
                cwd=workdir,
                env=RESTRICTED_ENV,
                capture_output=True,
                check=False,
                text=True,
                timeout=limits.timeout_seconds,
                preexec_fn=_apply_limits(limits),
                stdin=subprocess.DEVNULL,
            )
        except subprocess.TimeoutExpired as error:
            return SandboxResult(False, "", str(error), True)
        except OSError as error:
            raise SandboxError(f"cannot start sandbox: {error}") from error
    return SandboxResult(
        completed.returncode == 0,
        completed.stdout[:MAX_OUTPUT_CHARS],
        completed.stderr[:MAX_OUTPUT_CHARS],
        False,
    )
