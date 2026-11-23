from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Literal
from urllib.parse import urlsplit

from fastapi import WebSocket, WebSocketDisconnect
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from llm.studio.events import DROPPABLE_EVENT_TYPES, make_event
from llm.studio.model_manager import ModelManager
from llm.studio.scheduler import GenerationRequest, GenerationScheduler, GenerationState
from llm.studio.service import TASK_STATUS, StudioService

logger = logging.getLogger(__name__)

MAX_CONNECTIONS = 64
MAX_CONNECTIONS_PER_USER = 8
MAX_PAYLOAD_BYTES = 32768
MAX_EVENTS_PER_WINDOW = 50
RATE_WINDOW_SECONDS = 10
OUTBOX_LIMIT = 512
SEND_TIMEOUT_SECONDS = 5
HEARTBEAT_SECONDS = 15
IDLE_TIMEOUT_SECONDS = 60
BROADCAST_SECONDS = 2
SYSTEM_SECONDS = 5
ROW_BATCH_LIMIT = 200
GOING_AWAY = 1001


class ClientEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["subscribe", "ping", "generation.start", "generation.cancel"]
    channels: list[Literal["studio", "chat"]] = Field(default_factory=list, max_length=2)
    resume: list[str] = Field(default_factory=list, max_length=8)
    request_id: str | None = Field(default=None, min_length=1, max_length=64)
    chat_id: str | None = Field(default=None, min_length=1, max_length=64)
    content: str | None = Field(default=None, max_length=10000)
    action: Literal["send", "edit", "regenerate", "continue"] = "send"
    target_message_id: str | None = Field(default=None, min_length=1, max_length=64)
    mode: Literal["standard", "reasoning", "precise", "creative"] = "standard"
    checkpoint_id: str | None = Field(default=None, min_length=1, max_length=1024)
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    top_p: float | None = Field(default=None, gt=0.0, le=1.0)
    top_k: int | None = Field(default=None, ge=0, le=1000)
    max_tokens: int | None = Field(default=None, gt=0, le=8192)
    repetition_penalty: float | None = Field(default=None, gt=0.0, le=2.0)


@dataclass(eq=False)
class Connection:
    socket: WebSocket
    user_id: int
    role: str = "USER"
    token_hash: str = ""
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    channels: set[str] = field(default_factory=set)
    outbox: asyncio.Queue[dict[str, Any]] = field(
        default_factory=lambda: asyncio.Queue(maxsize=OUTBOX_LIMIT)
    )
    seq: int = 0
    closed: bool = False


def fingerprint(payload: Any) -> str:
    return hashlib.sha1(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()


class WebSocketManager:
    def __init__(
        self,
        studio: StudioService,
        model_manager: ModelManager,
        lock: threading.Lock,
        scheduler: GenerationScheduler | None = None,
    ) -> None:
        self.studio = studio
        self.model_manager = model_manager
        self.inference_lock = lock
        self.scheduler = scheduler or GenerationScheduler(studio, model_manager, lock)
        self.connections: set[Connection] = set()
        self.command_lock = asyncio.Lock()
        self.broadcaster: asyncio.Task[None] | None = None
        self.last_runtime = ""
        self.last_models = ""
        self.last_system = 0.0
        self.metric_cursor: tuple[str | None, int, str] = (None, -1, "")
        self.loop: asyncio.AbstractEventLoop | None = None
        studio.notifications.publishers.append(self.publish_to_user)
        studio.auth.listeners.append(self.on_sessions_revoked)

    def publish_to_user(self, user_id: int, event: dict[str, Any]) -> None:
        loop = self.loop
        if loop is None or loop.is_closed():
            return
        payload = make_event(
            event["type"], **{key: value for key, value in event.items() if key != "type"}
        )
        loop.call_soon_threadsafe(self.deliver_to_user, user_id, payload)

    def deliver_to_user(self, user_id: int, payload: dict[str, Any]) -> None:
        for connection in tuple(self.connections):
            if connection.user_id == user_id:
                self.enqueue(connection, payload)

    def on_sessions_revoked(self, user_id: int, hashes: set[str] | None) -> None:
        loop = self.loop
        if loop is None or loop.is_closed():
            return
        loop.call_soon_threadsafe(self.revoke_connections, user_id, hashes)

    def revoke_connections(self, user_id: int, hashes: set[str] | None) -> None:
        for connection in tuple(self.connections):
            if connection.user_id == user_id and (
                hashes is None or connection.token_hash in hashes
            ):
                self.enqueue(connection, make_event("session.revoked"))
                asyncio.get_running_loop().call_later(0.2, self.drop, connection, 1008)

    def enqueue(self, connection: Connection, payload: dict[str, Any]) -> None:
        if connection.closed:
            return
        try:
            connection.outbox.put_nowait(payload)
        except asyncio.QueueFull:
            if payload["type"] in DROPPABLE_EVENT_TYPES:
                return
            self.drop(connection, 1013)

    def drop(self, connection: Connection, code: int) -> None:
        if connection.closed:
            return
        connection.closed = True
        self.connections.discard(connection)
        asyncio.get_running_loop().create_task(self._close_quietly(connection.socket, code))

    @staticmethod
    async def _close_quietly(socket: WebSocket, code: int) -> None:
        with contextlib.suppress(RuntimeError, WebSocketDisconnect, OSError, TimeoutError):
            await asyncio.wait_for(socket.close(code=code), timeout=SEND_TIMEOUT_SECONDS)

    async def close(self) -> None:
        await self.scheduler.close()
        connections = tuple(self.connections)
        self.connections.clear()
        for connection in connections:
            connection.closed = True
        await asyncio.gather(
            *(self._close_quietly(connection.socket, GOING_AWAY) for connection in connections)
        )
        broadcaster = self.broadcaster
        if broadcaster is not None:
            broadcaster.cancel()
            await asyncio.gather(broadcaster, return_exceptions=True)

    async def writer(self, connection: Connection) -> None:
        try:
            while not connection.closed:
                payload = await connection.outbox.get()
                connection.seq += 1
                payload = {**payload, "seq": connection.seq, "ts": time.time()}
                await asyncio.wait_for(
                    connection.socket.send_json(payload), timeout=SEND_TIMEOUT_SECONDS
                )
        except (RuntimeError, WebSocketDisconnect, TimeoutError, OSError):
            self.drop(connection, 1011)

    async def publish(self, state: GenerationState, event_type: str, data: dict[str, Any]) -> None:
        state.sequence += 1
        payload = make_event(
            event_type,
            request_id=state.request_id,
            chat_id=state.chat_id,
            message_id=state.message_id,
            sequence_id=state.sequence,
            task_status=TASK_STATUS.get(state.status, "PROCESSING"),
            **data,
        )
        for connection in tuple(self.connections):
            if connection.user_id == state.user_id and "chat" in connection.channels:
                self.enqueue(connection, payload)

    def subscribers(self, channel: str) -> list[Connection]:
        return [item for item in self.connections if channel in item.channels]

    async def send_studio_state(self, targets: list[Connection], force: bool = False) -> None:
        def collect() -> tuple[dict[str, Any], dict[str, Any]]:
            registry = self.model_manager.registry()
            return self.studio.runtime_state(registry), registry

        runtime, registry = await asyncio.to_thread(collect)
        runtime_print, models_print = (
            fingerprint(runtime | {"generated_at": 0}),
            fingerprint(registry),
        )
        if force or runtime_print != self.last_runtime:
            if not force:
                self.last_runtime = runtime_print
            for item in targets:
                self.enqueue(item, make_event("runtime.update", data=runtime))
        if force or models_print != self.last_models:
            if not force:
                self.last_models = models_print
            for item in targets:
                self.enqueue(item, make_event("models.update", data=registry))
        if force or time.monotonic() - self.last_system >= SYSTEM_SECONDS:
            if not force:
                self.last_system = time.monotonic()
            system = await asyncio.to_thread(self.studio.record_system_metric)
            for item in targets:
                self.enqueue(item, make_event("system.update", data=system))

    async def send_metric_rows(self, targets: list[Connection], reset: bool) -> None:
        rows = await asyncio.to_thread(self.studio.metrics, None, 3000)
        run = self.studio.active_run_name()
        if run is None or not rows:
            self.metric_cursor = (run, -1, "")
            return
        signature = fingerprint(rows[-2:])
        previous_run, cursor, previous_signature = self.metric_cursor
        if not reset and run == previous_run and signature == previous_signature:
            return
        start = cursor if run == previous_run and not reset else -1
        fresh = [row for row in rows if row["step"] >= start][-ROW_BATCH_LIMIT:]
        if not reset:
            self.metric_cursor = (run, int(rows[-1]["step"]), signature)
        for item in targets:
            self.enqueue(item, make_event("metrics.append", run=run, rows=fresh, reset=reset))

    async def broadcast_loop(self) -> None:
        try:
            while self.connections:
                targets = self.subscribers("studio")
                if targets:
                    await self.send_studio_state(targets)
                    await self.send_metric_rows(targets, reset=False)
                await asyncio.sleep(BROADCAST_SECONDS)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Studio broadcaster stopped")
            for target in self.subscribers("studio"):
                self.drop(target, 1011)
        finally:
            self.broadcaster = None

    def ensure_broadcaster(self) -> None:
        if self.broadcaster is None or self.broadcaster.done():
            self.broadcaster = asyncio.create_task(self.broadcast_loop())

    async def authorize(self, socket: WebSocket) -> tuple[int, str, str | None] | None:
        origin = socket.headers.get("origin")
        host = socket.headers.get("host")
        if (
            not origin
            or urlsplit(origin).netloc != host
            or urlsplit(origin).scheme not in {"http", "https"}
        ):
            await socket.close(code=1008)
            return None
        if len(self.connections) >= MAX_CONNECTIONS:
            await socket.close(code=1013)
            return None
        token = socket.cookies.get("abylkhaiyr_session")
        identity = await asyncio.to_thread(self.studio.auth.get_user, token)
        if identity is None:
            await socket.close(code=1008)
            return None
        owned = [item for item in self.connections if item.user_id == identity[0].id]
        if len(owned) >= MAX_CONNECTIONS_PER_USER:
            await socket.close(code=1013)
            return None
        return identity[0].id, identity[0].role, token

    async def handle(self, socket: WebSocket) -> None:
        authorized = await self.authorize(socket)
        if authorized is None:
            return
        user_id, role, token = authorized
        await socket.accept()
        self.loop = asyncio.get_running_loop()
        connection = Connection(
            socket, user_id, role, self.studio.auth.hash_token(token) if token else ""
        )
        self.connections.add(connection)
        writer = asyncio.create_task(self.writer(connection))
        guard = asyncio.create_task(self.heartbeat(connection, token))
        self.enqueue(
            connection,
            make_event(
                "hello", connection_id=connection.id, server_time=datetime.now(UTC).isoformat()
            ),
        )
        recent: list[float] = []
        try:
            while not connection.closed:
                raw = await asyncio.wait_for(socket.receive_text(), timeout=IDLE_TIMEOUT_SECONDS)
                now = time.monotonic()
                recent = [stamp for stamp in recent if now - stamp < RATE_WINDOW_SECONDS]
                if (
                    len(raw.encode("utf-8")) > MAX_PAYLOAD_BYTES
                    or len(recent) >= MAX_EVENTS_PER_WINDOW
                ):
                    self.drop(connection, 1008)
                    break
                recent.append(now)
                try:
                    event = ClientEvent.model_validate_json(raw)
                    async with self.command_lock:
                        await self.dispatch(connection, event)
                except (ValidationError, ValueError):
                    self.enqueue(connection, make_event("error", code="invalid_event"))
        except TimeoutError:
            await self._close_quietly(socket, 1001)
        except (WebSocketDisconnect, RuntimeError, OSError):
            pass
        finally:
            connection.closed = True
            self.connections.discard(connection)
            guard.cancel()
            writer.cancel()
            await asyncio.gather(guard, writer, return_exceptions=True)

    async def heartbeat(self, connection: Connection, token: str | None) -> None:
        try:
            while not connection.closed:
                await asyncio.sleep(HEARTBEAT_SECONDS)
                if await asyncio.to_thread(self.studio.auth.get_user, token) is None:
                    self.drop(connection, 1008)
                    return
                self.enqueue(connection, make_event("heartbeat"))
        except asyncio.CancelledError:
            return

    async def dispatch(self, connection: Connection, event: ClientEvent) -> None:
        if event.type == "ping":
            self.enqueue(connection, make_event("pong"))
        elif event.type == "subscribe":
            allowed = {"chat"} if connection.role != "ADMIN" else {"chat", "studio"}
            connection.channels = {str(name) for name in event.channels} & allowed
            recovered: set[str] = set()
            if "chat" in connection.channels:
                active = self.scheduler.active_states_for_user(connection.user_id)
                recovered = {state.request_id for state in active}
                for state in active:
                    self.enqueue(connection, self.scheduler._state_snapshot(state))
                for request_id in dict.fromkeys(event.resume):
                    if request_id in recovered or not 1 <= len(request_id) <= 64:
                        continue
                    job = await asyncio.to_thread(
                        self.studio.get_generation_job, connection.user_id, request_id
                    )
                    if job is not None:
                        self.enqueue(connection, self.scheduler._job_snapshot(job))
                        recovered.add(request_id)
            if "studio" in connection.channels:
                await self.send_studio_state([connection], force=True)
                await self.send_metric_rows([connection], reset=True)
                self.ensure_broadcaster()
            if "chat" in connection.channels:
                self.enqueue(connection, make_event("chat.ready", request_ids=sorted(recovered)))
        elif event.type == "generation.cancel":
            if event.request_id:
                await self.scheduler.cancel(connection.user_id, event.request_id, self.publish)
        elif event.type == "generation.start":
            if not event.request_id or not event.chat_id:
                raise ValueError("Missing generation fields")
            if event.action in {"send", "edit"} and (
                not event.content or not event.content.strip()
            ):
                raise ValueError("Missing content")
            request = GenerationRequest(
                request_id=event.request_id,
                user_id=connection.user_id,
                chat_id=event.chat_id,
                content=event.content,
                action=event.action,
                target_message_id=event.target_message_id,
                mode=event.mode,
                checkpoint_id=event.checkpoint_id,
                temperature=event.temperature,
                top_p=event.top_p,
                top_k=event.top_k,
                max_tokens=event.max_tokens,
                repetition_penalty=event.repetition_penalty,
                privileged=connection.role == "ADMIN",
            )
            outcome, payload = await self.scheduler.submit(request, self.publish)
            if outcome == "existing" and isinstance(payload, dict):
                self.enqueue(connection, payload)
            elif outcome == "rejected" and isinstance(payload, dict):
                self.enqueue(
                    connection,
                    make_event(
                        "generation.failed",
                        request_id=event.request_id,
                        chat_id=event.chat_id,
                        sequence_id=1,
                        code=payload.get("code", "busy"),
                        detail=payload.get("detail", ""),
                    ),
                )
