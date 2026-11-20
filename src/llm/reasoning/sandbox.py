from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Protocol, cast

from llm.core.errors import SandboxError
from llm.reasoning.container import run_isolated


class ResourceLimits(Protocol):
    RLIMIT_CPU: int
    RLIMIT_FSIZE: int
    RLIMIT_NOFILE: int
    RLIMIT_AS: int

    def setrlimit(self, resource: int, limits: tuple[int, int]) -> None: ...


resource = None if sys.platform == "win32" else cast(ResourceLimits, import_module("resource"))

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
PRIVATE_READ_ROOTS = (
    "/Users",
    "/Volumes",
    "/private/var/folders",
    "/private/tmp",
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
    module = resource
    if module is None:
        raise SandboxError("Unix resource limits unavailable")

    def apply() -> None:
        module.setrlimit(module.RLIMIT_CPU, (limits.cpu_seconds, limits.cpu_seconds))
        module.setrlimit(module.RLIMIT_FSIZE, (limits.max_file_mb * BYTES_PER_MB,) * 2)
        if language == "python":
            module.setrlimit(module.RLIMIT_NOFILE, (limits.max_open_files,) * 2)
        if sys.platform.startswith("linux"):
            module.setrlimit(module.RLIMIT_AS, (limits.memory_mb * BYTES_PER_MB,) * 2)
        if sys.platform != "win32":
            os.setsid()

    return apply


def os_confinement(workdir: str, runtime: Path) -> list[str]:
    if sys.platform != "darwin" or not shutil.which(SANDBOX_EXEC):
        return []
    directory = json.dumps(os.path.realpath(workdir))
    private_paths = " ".join(f"(subpath {json.dumps(path)})" for path in PRIVATE_READ_ROOTS)
    runtime_path = json.dumps(str(runtime.resolve()))
    profile = (
        "(version 1)(allow default)(deny network*)(deny process-fork)"
        f"(deny file-read-data {private_paths})(deny file-write*)"
        f"(allow file-read-data (subpath {runtime_path}) (subpath {directory}))"
        f'(allow file-write* (subpath {directory}) (literal "/dev/null"))'
    )
    return [SANDBOX_EXEC, "-p", profile]


def interpreter(language: str) -> tuple[str, Path] | None:
    if language == "python":
        return os.path.realpath(sys.executable), Path(sys.base_prefix)
    if language == "java" and sys.platform == "darwin":
        discovered = subprocess.run(
            ["/usr/libexec/java_home"], capture_output=True, text=True, check=False, timeout=2
        )
        if discovered.returncode != 0:
            return None
        runtime = Path(discovered.stdout.strip()).resolve()
        return str(runtime / "bin" / "java"), runtime
    found = shutil.which(LANGUAGES[language][1][0])
    if found is None:
        return None
    command = Path(found).resolve()
    return str(command), command


def run_program(language: str, code: str, limits: SandboxLimits | None = None) -> SandboxResult:
    if language not in LANGUAGES:
        raise SandboxError(f"unsupported language: {language}")
    limits = limits or SandboxLimits()
    image = os.environ.get("ABYLKHAIYR_SANDBOX_IMAGE")
    if language == "python" and image:
        result = run_isolated(PREAMBLE + "\n" + code, "", limits.timeout_seconds, image)
        return SandboxResult(
            result.status == "passed", result.stdout, result.stderr, result.status == "timeout"
        )
    if resource is None:
        return SandboxResult(
            False,
            "",
            "Code execution sandbox requires Unix resource limits; unavailable on Windows.",
            False,
        )
    filename, template = LANGUAGES[language]
    resolved = interpreter(language)
    if resolved is None:
        return SandboxResult(False, "", f"Runtime unavailable for {language}.", False)
    executable, runtime = resolved
    command = [executable, *template[1:]]
    source = PREAMBLE + "\n" + code if language == "python" else code
    with tempfile.TemporaryDirectory(prefix="llm-sandbox-") as workdir:
        (Path(workdir) / filename).write_text(source, encoding="utf-8")
        env = {**RESTRICTED_ENV, "TMPDIR": workdir, "HOME": workdir}
        if language != "python":
            env["PATH"] = os.environ.get("PATH", RESTRICTED_ENV["PATH"])
        confinement = os_confinement(workdir, runtime)
        if not confinement:
            return SandboxResult(
                False, "", "OS isolation unavailable; configure a sandbox image.", False
            )
        try:
            with tempfile.TemporaryFile() as output, tempfile.TemporaryFile() as errors:
                completed = subprocess.run(
                    [*confinement, *command],
                    cwd=workdir,
                    env=env,
                    stdout=output,
                    stderr=errors,
                    timeout=limits.timeout_seconds,
                    preexec_fn=_apply_limits(limits, language),
                    stdin=subprocess.DEVNULL,
                    check=False,
                )
                output.seek(0)
                errors.seek(0)
                stdout = output.read(MAX_OUTPUT_CHARS).decode(errors="replace")
                stderr = errors.read(MAX_OUTPUT_CHARS).decode(errors="replace")
        except subprocess.TimeoutExpired as error:
            return SandboxResult(False, "", str(error), True)
        except OSError as error:
            raise SandboxError(f"cannot start sandbox: {error}") from error
    return SandboxResult(
        completed.returncode == 0,
        stdout,
        stderr,
        False,
    )


def run_python(code: str, limits: SandboxLimits | None = None) -> SandboxResult:
    return run_program("python", code, limits)
