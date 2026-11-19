from __future__ import annotations

import json
import threading
import time
import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field

from llm.inference.engine import InferenceEngine
from llm.inference.sampling import SamplingParams
from llm.studio.api import create_studio_router
from llm.studio.service import StudioService
from llm.tokenizer.bpe import Message

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8000
DEFAULT_MAX_TOKENS = 256
ROLES = ("system", "user", "assistant", "tool")


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")
    model: str | None = None
    messages: list[ChatMessage] = Field(min_length=1)
    max_tokens: int = Field(default=DEFAULT_MAX_TOKENS, gt=0)
    temperature: float = Field(default=0.8, ge=0.0, le=2.0)
    top_p: float = Field(default=0.95, gt=0.0, le=1.0)
    top_k: int = Field(default=0, ge=0)
    repetition_penalty: float = Field(default=1.0, gt=0.0)
    stop: str | list[str] | None = None
    stream: bool = False
    seed: int | None = None
    n: int = Field(default=1)


def to_params(request: ChatRequest) -> SamplingParams:
    stops = [request.stop] if isinstance(request.stop, str) else list(request.stop or [])
    return SamplingParams(
        max_new_tokens=request.max_tokens,
        temperature=request.temperature,
        top_p=request.top_p,
        top_k=request.top_k,
        repetition_penalty=request.repetition_penalty,
        stop_sequences=stops,
        seed=request.seed,
    )


def sse(payload: dict | str) -> str:
    return f"data: {payload if isinstance(payload, str) else json.dumps(payload)}\n\n"


def create_app(
    engine: InferenceEngine,
    model_name: str,
    studio_root: Path | None = None,
) -> FastAPI:
    app = FastAPI(title="llm", docs_url=None, redoc_url=None)
    lock = threading.Lock()
    root = (studio_root or Path.cwd()).resolve()
    studio = StudioService(root, model_name)
    app.state.studio = studio
    app.include_router(create_studio_router(studio, engine, lock))

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/v1/models")
    def models() -> dict:
        return {
            "object": "list",
            "data": [{"id": model_name, "object": "model", "owned_by": "local"}],
        }

    @app.post("/v1/chat/completions")
    def chat(request: ChatRequest):
        if request.n != 1:
            raise HTTPException(status_code=400, detail="only n=1 is supported")
        messages = [Message(m.role, m.content) for m in request.messages]
        params = to_params(request)
        completion_id = f"chatcmpl-{uuid.uuid4().hex}"
        created = int(time.time())
        prompt_ids = engine.tokenizer.encode_conversation(messages, True)
        prompt_tokens = len(engine.fit_prompt(prompt_ids, params.max_new_tokens))
        if request.stream:
            return StreamingResponse(
                stream_chunks(engine, lock, messages, params, completion_id, created, model_name),
                media_type="text/event-stream",
            )
        with lock:
            reply = engine.chat(messages, params)
        return {
            "id": completion_id,
            "object": "chat.completion",
            "created": created,
            "model": model_name,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": reply.text},
                    "finish_reason": reply.finish_reason,
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": len(reply.token_ids),
                "total_tokens": prompt_tokens + len(reply.token_ids),
            },
        }

    frontend_dist = root / "frontend" / "dist"
    if frontend_dist.is_dir():
        assets = frontend_dist / "assets"
        if assets.is_dir():
            app.mount("/assets", StaticFiles(directory=assets), name="studio-assets")

        @app.get("/{path:path}", include_in_schema=False)
        def studio_frontend(path: str) -> FileResponse:
            requested = (frontend_dist / path).resolve()
            if path and frontend_dist in requested.parents and requested.is_file():
                return FileResponse(requested)
            return FileResponse(frontend_dist / "index.html")

    return app


def stream_chunks(
    engine: InferenceEngine,
    lock: threading.Lock,
    messages: list[Message],
    params: SamplingParams,
    completion_id: str,
    created: int,
    model_name: str,
) -> Iterator[str]:
    def chunk(delta: dict, finish: str | None = None) -> str:
        return sse(
            {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model_name,
                "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
            }
        )

    with lock:
        yield chunk({"role": "assistant"})
        for piece in engine.stream_chat(messages, params):
            yield chunk({"content": piece})
        yield chunk({}, "stop")
    yield sse("[DONE]")
