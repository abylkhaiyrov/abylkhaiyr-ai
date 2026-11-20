from __future__ import annotations

import re
import shutil
import subprocess
import threading
import uuid
from typing import BinaryIO, Literal

from pydantic import BaseModel


class ToolResult(BaseModel):
    status: Literal["passed", "failed", "timeout", "unavailable"]
    exit_code: int | None = None
    stdout: str = ""
    stderr: str = ""


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

    def drain(stream: BinaryIO, captured: bytearray) -> None:
        while chunk := stream.read(4096):
            captured.extend(chunk[: max(0, 8193 - len(captured))])

    try:
        with subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        ) as process:
            assert process.stdin and process.stdout and process.stderr
            output, errors = bytearray(), bytearray()
            readers = [
                threading.Thread(target=drain, args=(process.stdout, output)),
                threading.Thread(target=drain, args=(process.stderr, errors)),
            ]
            for reader in readers:
                reader.start()
            try:
                try:
                    with process.stdin:
                        process.stdin.write(stdin.encode())
                except BrokenPipeError:
                    pass
                process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
                subprocess.run(
                    [docker, "rm", "-f", name], timeout=5, capture_output=True, check=False
                )
                return ToolResult(status="timeout")
            finally:
                for reader in readers:
                    reader.join()
            status: Literal["passed", "failed", "unavailable"] = (
                "passed"
                if process.returncode == 0
                else "unavailable" if process.returncode == 125 else "failed"
            )
            return ToolResult(
                status=status,
                exit_code=process.returncode,
                stdout=output.decode(errors="replace"),
                stderr=errors.decode(errors="replace"),
            )
    except (OSError, subprocess.TimeoutExpired) as error:
        return ToolResult(status="unavailable", stderr=str(error))
