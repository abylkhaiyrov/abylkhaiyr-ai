from __future__ import annotations

from typing import Any, Literal, get_args

ServerEventType = Literal[
    "hello",
    "heartbeat",
    "pong",
    "error",
    "chat.ready",
    "runtime.update",
    "system.update",
    "models.update",
    "metrics.append",
    "generation.queued",
    "generation.started",
    "generation.prefill",
    "generation.stage",
    "generation.delta",
    "generation.completed",
    "generation.cancelled",
    "generation.failed",
    "generation.snapshot",
]

SERVER_EVENT_TYPES = frozenset(get_args(ServerEventType))
DROPPABLE_EVENT_TYPES = frozenset({"heartbeat", "system.update", "metrics.append"})


def make_event(event_type: str, **data: Any) -> dict[str, Any]:
    if event_type not in SERVER_EVENT_TYPES:
        raise ValueError(f"unknown server event type: {event_type}")
    return {"type": event_type, **data}
