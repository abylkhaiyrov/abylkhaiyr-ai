from __future__ import annotations

import asyncio
import contextlib
import hashlib
import os
import shutil
import signal
import socket
import subprocess
from collections.abc import Callable, Iterator
from pathlib import Path
from types import FrameType
from typing import Self

import uvicorn
from fastapi import FastAPI

from llm.core.errors import ConfigError
from llm.studio.protocol import API_VERSION

DEV_HOST = "127.0.0.1"
STARTUP_TIMEOUT = 30
SHUTDOWN_SIGNALS = (signal.SIGINT, signal.SIGTERM)
SignalHandler = Callable[[int, FrameType | None], object] | int | None


class StudioServer(uvicorn.Server):
    @contextlib.contextmanager
    def capture_signals(self) -> Iterator[None]:
        yield

    @property
    def exit_code(self) -> int:
        failed = getattr(self.lifespan, "should_exit", False)
        return 0 if self.started and not failed else 1


class GracefulShutdown:
    def __init__(self, server: StudioServer) -> None:
        self.server = server
        self.requested = False
        self.previous: dict[signal.Signals, SignalHandler] = {}

    def __enter__(self) -> Self:
        self.previous = {sig: signal.signal(sig, self.handle) for sig in SHUTDOWN_SIGNALS}
        return self

    def __exit__(self, *_: object) -> None:
        for sig, handler in self.previous.items():
            signal.signal(sig, handler)

    def handle(self, sig: int, _frame: FrameType | None) -> None:
        if not self.requested:
            self.requested = True
            self.server.should_exit = True
            return
        self.server.force_exit = True
        if sig == signal.SIGINT:
            raise KeyboardInterrupt


def check_port(host: str, port: int) -> None:
    if not 1 <= port <= 65535:
        raise ConfigError(f"invalid Studio port: {port}")
    try:
        with socket.create_server(
            (host, port), family=socket.AF_INET6 if ":" in host else socket.AF_INET
        ):
            pass
    except OSError as error:
        raise ConfigError(f"Studio cannot bind {host}:{port}: {error.strerror}") from error


def frontend_signature(frontend: Path) -> str:
    digest = hashlib.sha256(API_VERSION.encode())
    paths = [path for path in (frontend / "src").rglob("*") if path.is_file()]
    paths.extend(
        path for path in frontend.iterdir() if path.is_file() and not path.name.startswith(".")
    )
    for path in sorted(paths):
        digest.update(str(path.relative_to(frontend)).encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def prepare_frontend(frontend: Path, dev: bool) -> None:
    lock = frontend / "package-lock.json"
    if not lock.is_file():
        raise ConfigError(
            "start Studio from the repository root (frontend/package-lock.json missing)"
        )
    if shutil.which("npm") is None or shutil.which("node") is None:
        raise ConfigError("Studio requires Node.js and npm")
    installed = frontend / "node_modules" / ".package-lock.json"
    vite = frontend / "node_modules" / ".bin" / "vite"
    if (
        not vite.is_file()
        or not installed.is_file()
        or installed.stat().st_mtime_ns < lock.stat().st_mtime_ns
    ):
        run_frontend_command(frontend, ["npm", "ci", "--no-audit", "--no-fund"])
    if dev:
        return
    dist = frontend / "dist"
    stamp = dist / ".studio-build"
    signature = frontend_signature(frontend)
    if (
        (dist / "index.html").is_file()
        and any((dist / "assets").glob("*.js"))
        and stamp.is_file()
        and stamp.read_text().strip() == signature
    ):
        return
    run_frontend_command(frontend, ["npm", "run", "build"])
    if not (dist / "index.html").is_file():
        raise ConfigError("Studio build did not produce frontend/dist/index.html")
    stamp.write_text(signature + "\n")


def run_frontend_command(frontend: Path, command: list[str]) -> None:
    # Windows CreateProcess does not resolve npm to npm.cmd automatically.
    executable = shutil.which(command[0])
    if executable is None:
        raise ConfigError(f"Studio frontend executable not found: {command[0]}")
    result = subprocess.run([executable, *command[1:]], cwd=frontend, check=False)
    if result.returncode:
        raise ConfigError(
            f"Studio frontend command failed: {' '.join(command)} (exit {result.returncode})"
        )


async def run_dev_server(
    server: StudioServer, frontend: Path, backend_port: int, frontend_port: int
) -> int:
    backend = asyncio.create_task(server.serve())
    frontend_process: asyncio.subprocess.Process | None = None
    try:
        async with asyncio.timeout(STARTUP_TIMEOUT):
            while not server.started:
                if backend.done():
                    backend.result()
                    if server.should_exit:
                        return 0
                    raise ConfigError("Studio backend exited before it became ready")
                await asyncio.sleep(0.05)
            frontend_process = await asyncio.create_subprocess_exec(
                str(frontend / "node_modules" / ".bin" / "vite"),
                "--host",
                DEV_HOST,
                "--port",
                str(frontend_port),
                "--strictPort",
                cwd=frontend,
                start_new_session=True,
                env={**os.environ, "STUDIO_BACKEND_PORT": str(backend_port)},
            )
            while frontend_process.returncode is None and not server.should_exit:
                try:
                    _, writer = await asyncio.open_connection(DEV_HOST, frontend_port)
                except OSError:
                    await asyncio.sleep(0.05)
                else:
                    writer.close()
                    await writer.wait_closed()
                    break
            else:
                if server.should_exit:
                    return 0
                raise ConfigError("Vite exited before it became ready")
        print(f"Studio development server ready at http://{DEV_HOST}:{frontend_port}", flush=True)
        frontend_exit = asyncio.create_task(frontend_process.wait())
        try:
            done, _ = await asyncio.wait(
                {backend, frontend_exit}, return_when=asyncio.FIRST_COMPLETED
            )
            if frontend_exit in done and not server.should_exit:
                raise ConfigError(f"Vite exited with status {frontend_exit.result()}")
            if backend in done:
                backend.result()
        finally:
            frontend_exit.cancel()
            await asyncio.gather(frontend_exit, return_exceptions=True)
        return server.exit_code
    except TimeoutError as error:
        raise ConfigError("Studio did not become ready within 30 seconds") from error
    finally:
        server.should_exit = True
        try:
            await backend
        finally:
            if frontend_process is not None and frontend_process.returncode is None:
                frontend_process.terminate()
                try:
                    await asyncio.wait_for(frontend_process.wait(), 5)
                except TimeoutError:
                    frontend_process.kill()
                    await frontend_process.wait()


def run_studio(
    app: FastAPI, frontend: Path, host: str, port: int, frontend_port: int | None
) -> int:
    server = StudioServer(uvicorn.Config(app, host=host, port=port, log_level="info"))
    with GracefulShutdown(server):
        if frontend_port is None:
            server.run()
            return server.exit_code
        return asyncio.run(run_dev_server(server, frontend, port, frontend_port))
