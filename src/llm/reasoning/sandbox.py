from __future__ import annotations

import os
import resource
import shutil
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


SANDBOX_EXEC = "sandbox-exec"
PROFILE_TEMPLATE = (
    "(version 1)(allow default)(deny network*)(deny file-write*)"
    '(allow file-write* (subpath "{workdir}") (literal "/dev/null"))'
)
HEAP_MB = 256
LANGUAGES = {
    "python": ("main.py", ["{python}", "-I", "-S", "main.py"]),
    "javascript": ("main.js", ["node", f"--max-old-space-size={HEAP_MB}", "main.js"]),
    "typescript": ("main.ts", ["node", f"--max-old-space-size={HEAP_MB}", "main.ts"]),
    "java": ("Main.java", ["java", f"-Xmx{HEAP_MB}m", "-XX:-UsePerfData", "Main.java"]),
}


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


def _apply_limits(limits: SandboxLimits, language: str = "python"):
    def apply() -> None:
        resource.setrlimit(resource.RLIMIT_CPU, (limits.cpu_seconds, limits.cpu_seconds))
        resource.setrlimit(resource.RLIMIT_FSIZE, (limits.max_file_mb * BYTES_PER_MB,) * 2)
        if language == "python":
            resource.setrlimit(resource.RLIMIT_NOFILE, (limits.max_open_files,) * 2)
        if sys.platform.startswith("linux"):
            resource.setrlimit(resource.RLIMIT_AS, (limits.memory_mb * BYTES_PER_MB,) * 2)
        os.setsid()

    return apply


def os_confinement(workdir: str) -> list[str]:
    if sys.platform == "darwin" and shutil.which(SANDBOX_EXEC):
        return [SANDBOX_EXEC, "-p", PROFILE_TEMPLATE.format(workdir=os.path.realpath(workdir))]
    return []


def run_program(language: str, code: str, limits: SandboxLimits | None = None) -> SandboxResult:
    if language not in LANGUAGES:
        raise SandboxError(f"unsupported language: {language}")
    limits = limits or SandboxLimits()
    filename, template = LANGUAGES[language]
    command = [part.format(python=sys.executable) for part in template]
    source = PREAMBLE + "\n" + code if language == "python" else code
    with tempfile.TemporaryDirectory(prefix="llm-sandbox-") as workdir:
        (Path(workdir) / filename).write_text(source, encoding="utf-8")
        env = {**RESTRICTED_ENV, "TMPDIR": workdir, "HOME": workdir}
        if language != "python":
            env["PATH"] = os.environ.get("PATH", RESTRICTED_ENV["PATH"])
        try:
            completed = subprocess.run(
                [*os_confinement(workdir), *command],
                cwd=workdir,
                env=env,
                capture_output=True,
                text=True,
                timeout=limits.timeout_seconds,
                preexec_fn=_apply_limits(limits, language),
                stdin=subprocess.DEVNULL,
                check=False,
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


def run_python(code: str, limits: SandboxLimits | None = None) -> SandboxResult:
    return run_program("python", code, limits)
