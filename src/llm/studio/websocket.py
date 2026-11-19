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

from llm.inference.engine import InferenceEngine
from llm.inference.sampling import SamplingParams
from llm.studio.service import StudioService
from llm.tokenizer.bpe import Message

logger = logging.getLogger(__name__)


class ClientEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["subscribe", "ping", "generation.start", "generation.cancel"]
    channels: list[Literal["studio", "chat"]] = Field(default_factory=list, max_length=2)
    request_id: str | None = Field(default=None, min_length=1, max_length=64)
    chat_id: str | None = Field(default=None, min_length=1, max_length=64)
    content: str | None = Field(default=None, min_length=1, max_length=10000)


@dataclass(eq=False)
class Connection:
    socket: WebSocket
    user_id: int
    channels: set[str] = field(default_factory=set)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


@dataclass
class GenerationState:
    user_id: int
    chat_id: str
    request_id: str
    cancel: threading.Event = field(default_factory=threading.Event)
    status: str = "queued"
    message_id: str | None = None
    text: str = ""
    sequence: int = 0


class WebSocketManager:
    def __init__(self, studio: StudioService, engine: InferenceEngine | None, lock: threading.Lock):
        self.studio = studio
        self.engine = engine
        self.inference_lock = lock
        self.connections: set[Connection] = set()
        self.generations: dict[tuple[int, str], GenerationState] = {}
        self.worker: asyncio.Task[None] | None = None
        self.command_lock = asyncio.Lock()

    async def send(self, connection: Connection, payload: dict[str, Any]) -> None:
        try:
            async with connection.lock:
                await asyncio.wait_for(connection.socket.send_json(payload), timeout=5)
        except (RuntimeError, WebSocketDisconnect, TimeoutError):
            self.connections.discard(connection)

    async def publish(self, state: GenerationState, event_type: str, **data: Any) -> None:
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
                recent = [stamp for stamp in recent if now - stamp < 10]
                if len(raw.encode("utf-8")) > 16384 or len(recent) >= 50:
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
                if await asyncio.to_thread(self.studio.auth.get_user, token) is None:
                    await connection.socket.close(code=1008)
                    return
                if "studio" in connection.channels:
                    dashboard, processes, metrics = await asyncio.gather(
                        asyncio.to_thread(self.studio.dashboard),
                        asyncio.to_thread(self.studio.list_processes),
                        asyncio.to_thread(self.studio.metrics, None, 1000),
                    )
                    await self.send(
                        connection,
                        {
                            "type": "studio.snapshot",
                            "data": {
                                "dashboard": dashboard,
                                "processes": processes,
                                "metrics": metrics,
                            },
                        },
                    )
                await asyncio.sleep(5)
        except (WebSocketDisconnect, RuntimeError):
            return

    async def dispatch(self, connection: Connection, event: ClientEvent) -> None:
        if event.type == "ping":
            await self.send(connection, {"type": "pong"})
        elif event.type == "subscribe":
            connection.channels = set(event.channels)
            for state in self.generations.values():
                if (
                    state.user_id == connection.user_id
                    and state.status in {"queued", "running"}
                    and "chat" in connection.channels
                ):
                    await self.send(
                        connection,
                        {
                            "type": "generation.snapshot",
                            "request_id": state.request_id,
                            "chat_id": state.chat_id,
                            "message_id": state.message_id,
                            "sequence_id": state.sequence,
                            "status": state.status,
                            "text": state.text,
                        },
                    )
        elif event.type == "generation.cancel":
            cancelling = self.generations.get((connection.user_id, event.request_id or ""))
            if cancelling is not None:
                cancelling.cancel.set()
        elif event.type == "generation.start":
            if (
                not event.request_id
                or not event.chat_id
                or not event.content
                or not event.content.strip()
            ):
                raise ValueError("Missing generation fields")
            key = (connection.user_id, event.request_id)
            if key in self.generations:
                state = self.generations[key]
                await self.send(
                    connection,
                    {
                        "type": "generation.snapshot",
                        "request_id": state.request_id,
                        "chat_id": state.chat_id,
                        "message_id": state.message_id,
                        "sequence_id": state.sequence,
                        "status": state.status,
                        "text": state.text,
                    },
                )
                return
            if (
                self.engine is None
                or len(self.generations) >= 1000
                or (self.worker is not None and not self.worker.done())
            ):
                await self.send(
                    connection,
                    {
                        "type": "generation.failed",
                        "request_id": event.request_id,
                        "chat_id": event.chat_id,
                        "sequence_id": 1,
                        "code": "busy",
                    },
                )
                return
            chat = await asyncio.to_thread(self.studio.chat, connection.user_id, event.chat_id)
            if chat is None:
                await self.send(
                    connection,
                    {
                        "type": "generation.failed",
                        "request_id": event.request_id,
                        "code": "chat_not_found",
                        "chat_id": event.chat_id,
                        "sequence_id": 1,
                    },
                )
                return
            state = GenerationState(connection.user_id, event.chat_id, event.request_id)
            self.generations[key] = state
            self.worker = asyncio.create_task(self.generate(state, chat, event.content))

    async def generate(self, state: GenerationState, chat: dict[str, Any], content: str) -> None:
        engine = self.engine
        assert engine is not None
        try:
            saved = await asyncio.to_thread(
                self.studio.start_chat_generation, state.user_id, state.chat_id, content
            )
            if saved is None:
                raise ValueError("Chat not found")
            user, assistant = saved
            state.message_id = assistant["id"]
            state.status = "running"
            await self.publish(state, "generation.started", user=user)
            messages = [Message("system", chat["system_prompt"])] if chat["system_prompt"] else []
            messages.extend(Message(item["role"], item["content"]) for item in chat["messages"])
            messages.append(Message("user", content))
            await self.publish(state, "generation.prefill")
            loop = asyncio.get_running_loop()
            started = time.monotonic()

            def run() -> None:
                with self.inference_lock:
                    for piece in engine.stream_chat(
                        messages,
                        SamplingParams(max_new_tokens=min(256, max(1, engine.max_positions // 2))),
                    ):
                        if state.cancel.is_set():
                            break
                        future = asyncio.run_coroutine_threadsafe(self.delta(state, piece), loop)
                        future.result()

            await asyncio.to_thread(run)
            state.status = "cancelled" if state.cancel.is_set() else "completed"
            await self.publish(
                state,
                f"generation.{state.status}",
                elapsed_ms=int((time.monotonic() - started) * 1000),
                text=state.text,
            )
        except Exception:
            logger.exception("Generation failed")
            state.status = "failed"
            await self.publish(
                state, "generation.failed", code="generation_failed", text=state.text
            )

    async def delta(self, state: GenerationState, piece: str) -> None:
        state.text += piece
        await asyncio.to_thread(
            self.studio.update_message_content,
            state.user_id,
            state.chat_id,
            state.message_id,
            state.text,
        )
        await self.publish(state, "generation.delta", text=piece)
