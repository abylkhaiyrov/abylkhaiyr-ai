from __future__ import annotations

import asyncio
import os
import signal
import socket
import subprocess
import sys
import textwrap
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from llm.api.server import create_app
from llm.studio.scheduler import GenerationScheduler, GenerationState
from llm.studio.websocket import Connection, WebSocketManager

SERVER = textwrap.dedent(
    """
    import asyncio
    import signal
    import sys
    from contextlib import asynccontextmanager
    from pathlib import Path

    from fastapi import FastAPI

    from llm.studio.launcher import run_studio

    signal.signal(signal.SIGINT, signal.default_int_handler)
    port, mode, marker, behaviour = int(sys.argv[1]), sys.argv[2], Path(sys.argv[3]), sys.argv[4]


    @asynccontextmanager
    async def lifespan(_):
        if behaviour == "startup_fails":
            raise RuntimeError("startup exploded")
        marker.write_text("started")
        yield
        if behaviour == "shutdown_fails":
            raise RuntimeError("shutdown exploded")
        if behaviour == "shutdown_hangs":
            marker.write_text("shutting")
            await asyncio.sleep(60)
        marker.write_text("stopped")


    frontend = Path(sys.argv[5])
    code = run_studio(
        FastAPI(lifespan=lifespan),
        frontend,
        "127.0.0.1",
        port,
        port + 1 if mode == "dev" else None,
    )
    sys.exit(code)
    """
)

FAKE_VITE = textwrap.dedent(
    f"""\
    #!{sys.executable}
    import os
    import signal
    import socket
    import sys
    import time
    from pathlib import Path

    port = int(sys.argv[sys.argv.index("--port") + 1])
    Path(__file__).with_name("vite.pid").write_text(str(os.getpid()))
    listener = socket.create_server(("127.0.0.1", port))
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    while True:
        time.sleep(0.1)
    """
)


def free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        first = probe.getsockname()[1]
    return first


def is_open(port: int) -> bool:
    with socket.socket() as probe:
        probe.settimeout(0.2)
        return probe.connect_ex(("127.0.0.1", port)) == 0


def frontend_with_fake_vite(root: Path) -> Path:
    binary = root / "frontend" / "node_modules" / ".bin"
    binary.mkdir(parents=True)
    vite = binary / "vite"
    vite.write_text(FAKE_VITE)
    vite.chmod(0o755)
    return root / "frontend"


def launch(tmp_path: Path, mode: str, behaviour: str = "ok") -> tuple[subprocess.Popen[str], int]:
    port = free_port()
    while is_open(port + 1) or port + 1 > 65535:
        port = free_port()
    script = tmp_path / "server.py"
    script.write_text(SERVER)
    frontend = frontend_with_fake_vite(tmp_path)
    process = subprocess.Popen(
        [
            sys.executable,
            str(script),
            str(port),
            mode,
            str(tmp_path / "marker"),
            behaviour,
            str(frontend),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return process, port


def wait_until(condition, timeout: float = 20.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if condition():
            return
        time.sleep(0.05)
    raise AssertionError("condition not reached in time")


def wait_ready(process: subprocess.Popen[str], port: int, mode: str) -> None:
    wait_until(
        lambda: (
            process.poll() is not None
            or (is_open(port) and (mode == "normal" or is_open(port + 1)))
        )
    )
    assert process.poll() is None


@pytest.mark.parametrize("mode", ["normal", "dev"])
@pytest.mark.parametrize("sig", [signal.SIGINT, signal.SIGTERM])
def test_graceful_stop_is_quiet_and_releases_ports(tmp_path, mode, sig):
    process, port = launch(tmp_path, mode)
    wait_ready(process, port, mode)
    process.send_signal(sig)
    _, stderr = process.communicate(timeout=30)
    assert process.returncode == 0, stderr
    assert "Traceback" not in stderr
    assert "CancelledError" not in stderr
    assert (tmp_path / "marker").read_text() == "stopped"
    assert not is_open(port)
    assert not is_open(port + 1)
    if mode == "dev":
        pid = int((tmp_path / "frontend" / "node_modules" / ".bin" / "vite.pid").read_text())
        with pytest.raises(ProcessLookupError):
            os.kill(pid, 0)


def test_shutdown_error_is_reported(tmp_path):
    process, port = launch(tmp_path, "normal", "shutdown_fails")
    wait_ready(process, port, "normal")
    process.send_signal(signal.SIGTERM)
    _, stderr = process.communicate(timeout=30)
    assert process.returncode == 1
    assert "shutdown exploded" in stderr
    assert not is_open(port)


def test_startup_error_is_reported_and_port_is_free(tmp_path):
    process, port = launch(tmp_path, "normal", "startup_fails")
    _, stderr = process.communicate(timeout=30)
    assert process.returncode not in {0, None}
    assert "startup exploded" in stderr
    assert not is_open(port)


def test_second_interrupt_forces_exit_without_suppression(tmp_path):
    process, port = launch(tmp_path, "normal", "shutdown_hangs")
    wait_ready(process, port, "normal")
    process.send_signal(signal.SIGINT)
    wait_until(lambda: (tmp_path / "marker").read_text() == "shutting")
    process.send_signal(signal.SIGINT)
    _, stderr = process.communicate(timeout=30)
    assert process.returncode != 0
    assert "KeyboardInterrupt" in stderr
    assert not is_open(port)


def state(user_id: int = 1, request_id: str = "r") -> GenerationState:
    return GenerationState(request_id=request_id, user_id=user_id, chat_id=1, message_id=1)


def test_scheduler_close_cancels_generations_and_waits_for_worker():
    async def scenario() -> None:
        scheduler = GenerationScheduler(MagicMock(), MagicMock(), MagicMock())
        running, queued = state(request_id="a"), state(request_id="b")
        scheduler.states = {(1, "a"): running, (1, "b"): queued}
        scheduler.pending.append((queued, MagicMock(), AsyncMock()))
        finished = asyncio.Event()

        async def worker() -> None:
            while not running.cancel.is_set():
                await asyncio.sleep(0.01)
            finished.set()

        scheduler.worker = asyncio.create_task(worker())
        await scheduler.close(timeout=5)
        assert running.cancel.is_set() and queued.cancel.is_set()
        assert not scheduler.pending
        assert finished.is_set()

    asyncio.run(scenario())


def test_scheduler_close_cancels_a_stuck_worker_after_the_timeout():
    async def scenario() -> None:
        scheduler = GenerationScheduler(MagicMock(), MagicMock(), MagicMock())
        scheduler.worker = asyncio.create_task(asyncio.sleep(60))
        await scheduler.close(timeout=0.05)
        assert scheduler.worker.cancelled()

    asyncio.run(scenario())


def test_websocket_manager_close_notifies_clients_and_stops_tasks():
    async def scenario() -> None:
        scheduler = MagicMock()
        scheduler.close = AsyncMock()
        manager = WebSocketManager(MagicMock(), MagicMock(), MagicMock(), scheduler=scheduler)
        sockets = [MagicMock(close=AsyncMock()) for _ in range(2)]
        connections = [Connection(peer, user_id=index) for index, peer in enumerate(sockets)]
        manager.connections.update(connections)
        manager.broadcaster = asyncio.create_task(asyncio.sleep(60))
        await manager.close()
        scheduler.close.assert_awaited_once()
        for peer in sockets:
            peer.close.assert_awaited_once_with(code=1001)
        assert not manager.connections
        assert all(connection.closed for connection in connections)
        assert manager.broadcaster.cancelled()

    asyncio.run(scenario())


def test_app_shutdown_releases_resources_without_touching_training(
    model_config, tokenizer, tmp_path
):
    import torch

    from llm.inference.engine import InferenceEngine
    from llm.model.transformer import CausalLM

    engine = InferenceEngine(CausalLM(model_config), tokenizer, device=torch.device("cpu"))
    app = create_app(engine, "test-model", studio_root=tmp_path)
    studio, manager = app.state.studio, app.state.model_manager
    studio.engine.dispose = MagicMock(wraps=studio.engine.dispose)
    studio.processes.stop = MagicMock()
    with TestClient(app) as client:
        assert client.get("/health").status_code == 200
        assert manager.active is not None
    assert manager.active is None
    studio.engine.dispose.assert_called_once()
    studio.processes.stop.assert_not_called()
