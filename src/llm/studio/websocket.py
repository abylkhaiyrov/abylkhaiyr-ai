from __future__ import annotations

import asyncio
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Literal
from urllib.parse import urlsplit

from fastapi import WebSocket, WebSocketDisconnect
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from llm.studio.model_manager import ModelManager
from llm.studio.scheduler import GenerationRequest, GenerationScheduler, GenerationState
from llm.studio.service import StudioService

logger = logging.getLogger(__name__)

MAX_CONNECTIONS = 64
MAX_PAYLOAD_BYTES = 32768
MAX_EVENTS_PER_WINDOW = 50
RATE_WINDOW_SECONDS = 10


class ClientEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["subscribe", "ping", "generation.start", "generation.cancel"]
    channels: list[Literal["studio", "chat"]] = Field(default_factory=list, max_length=2)
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
    channels: set[str] = field(default_factory=set)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


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

    async def send(self, connection: Connection, payload: dict[str, Any]) -> None:
        try:
            async with connection.lock:
                await asyncio.wait_for(connection.socket.send_json(payload), timeout=5)
        except (RuntimeError, WebSocketDisconnect, TimeoutError):
            self.connections.discard(connection)

    async def publish(
        self, state: GenerationState, event_type: str, data: dict[str, Any]
    ) -> None:
        state.sequence += 1
        payload = {
            "type": event_type,
            "request_id": state.request_id,
            "chat_id": state.chat_id,
            "message_id": state.message_id,
            "sequence_id": state.sequence,
            **data,
        }
        await asyncio.gather(
            *(
                self.send(connection, payload)
                for connection in tuple(self.connections)
                if connection.user_id == state.user_id and "chat" in connection.channels
            )
        )

    async def broadcast_studio_snapshot(self, connection: Connection | None = None) -> None:
        dashboard, processes, metrics = await asyncio.gather(
            asyncio.to_thread(self.studio.dashboard),
            asyncio.to_thread(self.studio.list_processes),
            asyncio.to_thread(self.studio.metrics, None, 1000),
        )
        payload = {
            "type": "studio.snapshot",
            "data": {
                "dashboard": dashboard,
                "processes": processes,
                "metrics": metrics,
            },
        }
        if connection is not None:
            await self.send(connection, payload)
            return
        await asyncio.gather(
            *(
                self.send(item, payload)
                for item in tuple(self.connections)
                if "studio" in item.channels
            )
        )

    async def handle(self, socket: WebSocket) -> None:
        origin = socket.headers.get("origin")
        host = socket.headers.get("host")
        if (
            not origin
            or urlsplit(origin).netloc != host
            or urlsplit(origin).scheme not in {"http", "https"}
        ):
            await socket.close(code=1008)
            return
        if len(self.connections) >= MAX_CONNECTIONS:
            await socket.close(code=1013)
            return
        token = socket.cookies.get("abylkhaiyr_session")
        identity = await asyncio.to_thread(self.studio.auth.get_user, token)
        if identity is None:
            await socket.close(code=1008)
            return
        await socket.accept()
        connection = Connection(socket, identity[0].id)
        self.connections.add(connection)
        telemetry = asyncio.create_task(self.telemetry(connection, token))
        recent: list[float] = []
        try:
            while True:
                raw = await socket.receive_text()
                now = time.monotonic()
                recent = [stamp for stamp in recent if now - stamp < RATE_WINDOW_SECONDS]
                if (
                    len(raw.encode("utf-8")) > MAX_PAYLOAD_BYTES
                    or len(recent) >= MAX_EVENTS_PER_WINDOW
                ):
                    await socket.close(code=1008)
                    break
                recent.append(now)
                if await asyncio.to_thread(self.studio.auth.get_user, token) is None:
                    await socket.close(code=1008)
                    break
                try:
                    event = ClientEvent.model_validate_json(raw)
                    async with self.command_lock:
                        await self.dispatch(connection, event)
                except (ValidationError, ValueError):
                    await self.send(connection, {"type": "error", "code": "invalid_event"})
        except WebSocketDisconnect:
            pass
        finally:
            telemetry.cancel()
            self.connections.discard(connection)

    async def telemetry(self, connection: Connection, token: str | None) -> None:
        try:
            while True:
                await asyncio.sleep(5)
                if await asyncio.to_thread(self.studio.auth.get_user, token) is None:
                    await connection.socket.close(code=1008)
                    return
                if "studio" in connection.channels:
                    await self.broadcast_studio_snapshot(connection)
        except (WebSocketDisconnect, RuntimeError, asyncio.CancelledError):
            return

    async def dispatch(self, connection: Connection, event: ClientEvent) -> None:
        if event.type == "ping":
            await self.send(connection, {"type": "pong"})
        elif event.type == "subscribe":
            connection.channels = set(event.channels)
            if "chat" in connection.channels:
                for state in self.scheduler.active_states_for_user(connection.user_id):
                    await self.send(connection, self.scheduler._state_snapshot(state))
            if "studio" in connection.channels:
                await self.broadcast_studio_snapshot(connection)
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
            )
            outcome, payload = await self.scheduler.submit(request, self.publish)
            if outcome == "existing" and isinstance(payload, dict):
                await self.send(connection, payload)
            elif outcome == "rejected" and isinstance(payload, dict):
                await self.send(
                    connection,
                    {
                        "type": "generation.failed",
                        "request_id": event.request_id,
                        "chat_id": event.chat_id,
                        "sequence_id": 1,
                        "code": payload.get("code", "busy"),
                        "detail": payload.get("detail", ""),
                    },
                )
