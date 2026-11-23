from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
import threading
import uuid
from typing import BinaryIO, Literal

from pydantic import BaseModel

DAEMON_MARKERS = (
    "cannot connect to the docker daemon",
    "failed to connect to the docker api",
    "is the docker daemon running",
)


class ToolResult(BaseModel):
    status: Literal["passed", "failed", "timeout", "unavailable"]
    exit_code: int | None = None
    stdout: str = ""
    stderr: str = ""


def _drain(stream: BinaryIO, captured: bytearray) -> None:
    while chunk := stream.read(4096):
        captured.extend(chunk[: max(0, 8193 - len(captured))])


def run_isolated(code: str, stdin: str, timeout: float, image: str) -> ToolResult:
    docker = shutil.which("docker")
    if docker is None:
        return ToolResult(status="unavailable", stderr="Docker executable unavailable")
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", image):
        raise ValueError("a local Docker image ID is required")
    name = f"abylkhaiyr-task-{uuid.uuid4().hex}"
    command = [
        docker,
        "run",
        "-i",
        "--pull=never",
        "--rm",
        "--name",
        name,
        "--network=none",
        "--read-only",
        "--user",
        "65534:65534",
        "--cap-drop=ALL",
        "--security-opt=no-new-privileges",
        "--pids-limit=32",
        "--memory=96m",
        "--memory-swap=96m",
        "--cpus=0.5",
        "--log-driver=none",
        image,
        "python",
        "-I",
        "-B",
        "-c",
        code,
    ]

    try:
        with tempfile.TemporaryFile() as input_file:
            input_file.write(stdin.encode())
            input_file.seek(0)
            return _run_container(docker, command, name, input_file, timeout)
    except (OSError, subprocess.TimeoutExpired) as error:
        return ToolResult(status="unavailable", stderr=str(error))


def _run_container(
    docker: str,
    command: list[str],
    name: str,
    input_file: BinaryIO,
    timeout: float,
) -> ToolResult:
    with subprocess.Popen(
        command, stdin=input_file, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    ) as process:
        assert process.stdout and process.stderr
        output, errors = bytearray(), bytearray()
        readers = [
            threading.Thread(target=_drain, args=(process.stdout, output)),
            threading.Thread(target=_drain, args=(process.stderr, errors)),
        ]
        for reader in readers:
            reader.start()
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
            subprocess.run([docker, "rm", "-f", name], timeout=5, capture_output=True, check=False)
            return ToolResult(status="timeout")
        finally:
            for reader in readers:
                reader.join()
        message = errors.decode(errors="replace")
        daemon_down = any(marker in message.lower() for marker in DAEMON_MARKERS)
        status: Literal["passed", "failed", "unavailable"] = (
            "passed"
            if process.returncode == 0
            else "unavailable"
            if process.returncode == 125 or daemon_down
            else "failed"
        )
        return ToolResult(
            status=status,
            exit_code=process.returncode,
            stdout=output.decode(errors="replace"),
            stderr=message,
        )
