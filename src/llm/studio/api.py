import threading
import time
from collections import defaultdict, deque
from typing import Annotated, Any, Literal

from fastapi import APIRouter, Cookie, Depends, HTTPException, Request, Response, WebSocket, status
from pydantic import BaseModel, ConfigDict, Field

from llm.inference.sampling import SamplingParams
from llm.studio.auth import AuthenticationError
from llm.studio.model_manager import ModelManager, ModelManagerError
from llm.studio.processes import ProcessError
from llm.studio.service import StudioService
from llm.tokenizer.bpe import Message

SESSION_COOKIE = "abylkhaiyr_session"
CSRF_COOKIE = "abylkhaiyr_csrf"
CSRF_HEADER = "X-CSRF-Token"
LOGIN_ATTEMPTS = 5
LOGIN_WINDOW_SECONDS = 300


class LoginRequest(BaseModel):
    username: str = Field(min_length=1, max_length=128)
    password: str = Field(min_length=1, max_length=512)


class PasswordChangeRequest(BaseModel):
    current_password: str = Field(min_length=1, max_length=512)
    new_password: str = Field(min_length=12, max_length=512)


class TrainingStartRequest(BaseModel):
    config_path: str = Field(min_length=1, max_length=1024)
    stage: Literal["pretrain", "sft", "dpo"] | None = None


class ChatCreateRequest(BaseModel):
    title: str = Field(default="New chat", max_length=256)
    model_name: str | None = Field(default=None, max_length=256)
    checkpoint: str | None = Field(default=None, max_length=1024)
    system_prompt: str = Field(default="", max_length=20000)
    generation_settings: dict[str, Any] = Field(default_factory=dict)


class ChatUpdateRequest(BaseModel):
    title: str | None = Field(default=None, max_length=256)
    model_name: str | None = Field(default=None, max_length=256)
    checkpoint: str | None = Field(default=None, max_length=1024)
    system_prompt: str | None = Field(default=None, max_length=20000)
    generation_settings: dict[str, Any] | None = None


class ModelLoadRequest(BaseModel):
    checkpoint_id: str = Field(min_length=1, max_length=1024)
    device: Literal["auto", "cpu", "mps", "cuda"] | None = None


class ChatMessageRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    content: str = Field(min_length=1, max_length=100000)
    max_tokens: int = Field(default=256, gt=0, le=8192)
    temperature: float = Field(default=0.8, ge=0.0, le=2.0)
    top_p: float = Field(default=0.95, gt=0.0, le=1.0)
    top_k: int = Field(default=0, ge=0)
    repetition_penalty: float = Field(default=1.0, gt=0.0, le=2.0)
    seed: int | None = None


class LoginRateLimiter:
    def __init__(self) -> None:
        self.attempts: dict[str, deque[float]] = defaultdict(deque)
        self.lock = threading.Lock()

    def allow(self, key: str) -> bool:
        now = time.monotonic()
        with self.lock:
            attempts = self.attempts[key]
            while attempts and now - attempts[0] > LOGIN_WINDOW_SECONDS:
                attempts.popleft()
            return len(attempts) < LOGIN_ATTEMPTS

    def fail(self, key: str) -> None:
        with self.lock:
            self.attempts[key].append(time.monotonic())

    def clear(self, key: str) -> None:
        with self.lock:
            self.attempts.pop(key, None)


def create_studio_router(
    studio: StudioService,
    model_manager: ModelManager,
    inference_lock: threading.Lock,
) -> APIRouter:
    router = APIRouter(prefix="/api")
    rate_limiter = LoginRateLimiter()
    from llm.studio.websocket import WebSocketManager

    websocket_manager = WebSocketManager(studio, model_manager, inference_lock)

    @router.websocket("/ws")
    async def websocket_endpoint(socket: WebSocket) -> None:
        await websocket_manager.handle(socket)

    def current_identity(
        session_token: Annotated[str | None, Cookie(alias=SESSION_COOKIE)] = None,
    ) -> tuple[Any, str]:
        identity = studio.auth.get_user(session_token)
        if identity is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication required"
            )
        return identity

    def require_csrf(
        request: Request,
        identity: Annotated[tuple[Any, str], Depends(current_identity)],
        csrf_cookie: Annotated[str | None, Cookie(alias=CSRF_COOKIE)] = None,
    ) -> tuple[Any, str]:
        csrf_header = request.headers.get(CSRF_HEADER)
        if (
            csrf_cookie is None
            or csrf_header is None
            or csrf_header != csrf_cookie
            or csrf_header != identity[1]
        ):
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Invalid CSRF token")
        return identity

    @router.post("/auth/login")
    def login(payload: LoginRequest, request: Request, response: Response) -> dict[str, Any]:
        address = request.client.host if request.client is not None else "unknown"
        if not rate_limiter.allow(address):
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail="Too many login attempts"
            )
        try:
            token, csrf_token, user = studio.auth.authenticate(payload.username, payload.password)
        except AuthenticationError as error:
            rate_limiter.fail(address)
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED, detail=str(error)
            ) from error
        rate_limiter.clear(address)
        response.set_cookie(
            SESSION_COOKIE,
            token,
            httponly=True,
            secure=False,
            samesite="strict",
            max_age=12 * 60 * 60,
            path="/",
        )
        response.set_cookie(
            CSRF_COOKIE,
            csrf_token,
            httponly=False,
            secure=False,
            samesite="strict",
            max_age=12 * 60 * 60,
            path="/",
        )
        return {
            "username": user.username,
            "requires_password_change": user.requires_password_change,
        }

    @router.get("/auth/me")
    def me(identity: Annotated[tuple[Any, str], Depends(current_identity)]) -> dict[str, Any]:
        user, _ = identity
        return {
            "username": user.username,
            "requires_password_change": user.requires_password_change,
        }

    @router.post("/auth/logout", status_code=status.HTTP_204_NO_CONTENT)
    def logout(
        identity: Annotated[tuple[Any, str], Depends(require_csrf)],
        response: Response,
        session_token: Annotated[str | None, Cookie(alias=SESSION_COOKIE)] = None,
    ) -> Response:
        studio.auth.logout(session_token)
        response.delete_cookie(SESSION_COOKIE, path="/")
        response.delete_cookie(CSRF_COOKIE, path="/")
        response.status_code = status.HTTP_204_NO_CONTENT
        return response

    @router.put("/auth/password", status_code=status.HTTP_204_NO_CONTENT)
    def change_password(
        payload: PasswordChangeRequest,
        response: Response,
        identity: Annotated[tuple[Any, str], Depends(require_csrf)],
    ) -> Response:
        try:
            studio.auth.change_password(
                identity[0].id, payload.current_password, payload.new_password
            )
        except AuthenticationError as error:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail=str(error)
            ) from error
        response.delete_cookie(SESSION_COOKIE, path="/")
        response.delete_cookie(CSRF_COOKIE, path="/")
        response.status_code = status.HTTP_204_NO_CONTENT
        return response

    @router.get("/dashboard")
    def dashboard(_: Annotated[tuple[Any, str], Depends(current_identity)]) -> dict[str, Any]:
        return studio.dashboard()

    @router.get("/settings")
    def settings(identity: Annotated[tuple[Any, str], Depends(current_identity)]) -> dict[str, Any]:
        user, _ = identity
        return {
            "local_mode": True,
            "requires_password_change": user.requires_password_change,
            "session_ttl_seconds": 12 * 60 * 60,
        }

    @router.get("/system")
    def system(_: Annotated[tuple[Any, str], Depends(current_identity)]) -> dict[str, Any]:
        return studio.record_system_metric()

    @router.get("/system/history")
    def system_history(
        _: Annotated[tuple[Any, str], Depends(current_identity)],
        seconds: int = 300,
    ) -> list[dict[str, Any]]:
        return studio.system_history(min(max(seconds, 300), 86400))

    @router.get("/processes")
    def processes(_: Annotated[tuple[Any, str], Depends(current_identity)]) -> list[dict[str, Any]]:
        return studio.list_processes()

    @router.post("/training")
    def start_training(
        payload: TrainingStartRequest,
        _: Annotated[tuple[Any, str], Depends(require_csrf)],
    ) -> dict[str, Any]:
        try:
            if studio.runtime.pipeline() is not None:
                raise ProcessError("Another resource-intensive process is already active")
            return studio.processes.start_training(payload.config_path, payload.stage)
        except (ProcessError, ValueError) as error:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail=str(error)
            ) from error

    @router.post("/training/{process_id}/pause")
    def pause_training(
        process_id: str,
        _: Annotated[tuple[Any, str], Depends(require_csrf)],
    ) -> dict[str, Any]:
        try:
            return studio.processes.pause(process_id)
        except ProcessError as error:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(error)) from error

    @router.post("/training/{process_id}/stop")
    def stop_training(
        process_id: str,
        _: Annotated[tuple[Any, str], Depends(require_csrf)],
    ) -> dict[str, Any]:
        try:
            return studio.processes.stop(process_id)
        except ProcessError as error:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(error)) from error

    @router.get("/training/metrics")
    def training_metrics(
        _: Annotated[tuple[Any, str], Depends(current_identity)],
        run: str | None = None,
        limit: int = 1000,
    ) -> list[dict[str, float]]:
        return studio.metrics(run, limit)

    @router.get("/models")
    def models(_: Annotated[tuple[Any, str], Depends(current_identity)]) -> list[dict[str, Any]]:
        return studio.models()

    @router.get("/models/active")
    def active_model(
        _: Annotated[tuple[Any, str], Depends(current_identity)],
    ) -> dict[str, Any] | None:
        return model_manager.active_info()

    @router.post("/models/load")
    def load_model(
        payload: ModelLoadRequest,
        _: Annotated[tuple[Any, str], Depends(require_csrf)],
    ) -> dict[str, Any]:
        try:
            return model_manager.load(payload.checkpoint_id, payload.device)
        except ModelManagerError as error:
            status_code = (
                status.HTTP_404_NOT_FOUND
                if error.code == "checkpoint_not_found"
                else status.HTTP_409_CONFLICT
            )
            raise HTTPException(status_code=status_code, detail=error.detail) from error

    @router.post("/models/switch")
    def switch_model(
        payload: ModelLoadRequest,
        _: Annotated[tuple[Any, str], Depends(require_csrf)],
    ) -> dict[str, Any]:
        try:
            return model_manager.switch(payload.checkpoint_id, payload.device)
        except ModelManagerError as error:
            status_code = (
                status.HTTP_404_NOT_FOUND
                if error.code == "checkpoint_not_found"
                else status.HTTP_409_CONFLICT
            )
            raise HTTPException(status_code=status_code, detail=error.detail) from error

    @router.post("/models/unload")
    def unload_model(
        _: Annotated[tuple[Any, str], Depends(require_csrf)],
    ) -> dict[str, Any]:
        return model_manager.unload()

    @router.get("/checkpoints")
    def checkpoints(
        _: Annotated[tuple[Any, str], Depends(current_identity)],
    ) -> list[dict[str, Any]]:
        return studio.models()

    @router.get("/datasets")
    def datasets(_: Annotated[tuple[Any, str], Depends(current_identity)]) -> list[dict[str, Any]]:
        return studio.datasets()

    @router.get("/experiments")
    def experiments(
        _: Annotated[tuple[Any, str], Depends(current_identity)],
    ) -> list[dict[str, Any]]:
        return studio.experiments()

    @router.get("/evaluation")
    def evaluation(
        _: Annotated[tuple[Any, str], Depends(current_identity)],
    ) -> list[dict[str, Any]]:
        return studio.evaluations()

    @router.get("/logs")
    def logs(
        _: Annotated[tuple[Any, str], Depends(current_identity)],
        category: str | None = None,
        offset: int = 0,
        limit: int = 100,
    ) -> dict[str, Any]:
        return studio.logs(category, max(offset, 0), limit)

    @router.get("/chats")
    def chats(
        identity: Annotated[tuple[Any, str], Depends(current_identity)],
        query: str | None = None,
    ) -> list[dict[str, Any]]:
        return studio.chats(identity[0].id, query)

    @router.post("/chats", status_code=status.HTTP_201_CREATED)
    def create_chat(
        payload: ChatCreateRequest,
        identity: Annotated[tuple[Any, str], Depends(require_csrf)],
    ) -> dict[str, Any]:
        return studio.create_chat(
            identity[0].id,
            payload.title,
            payload.model_name,
            payload.checkpoint,
            payload.system_prompt,
            payload.generation_settings,
        )

    @router.get("/chats/{chat_id}")
    def get_chat(
        chat_id: str,
        identity: Annotated[tuple[Any, str], Depends(current_identity)],
    ) -> dict[str, Any]:
        chat = studio.chat(identity[0].id, chat_id)
        if chat is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Chat not found")
        return chat

    @router.patch("/chats/{chat_id}")
    def update_chat(
        chat_id: str,
        payload: ChatUpdateRequest,
        identity: Annotated[tuple[Any, str], Depends(require_csrf)],
    ) -> dict[str, Any]:
        chat = studio.update_chat(
            identity[0].id,
            chat_id,
            title=payload.title,
            model_name=payload.model_name,
            checkpoint=payload.checkpoint,
            system_prompt=payload.system_prompt,
            generation_settings=payload.generation_settings,
        )
        if chat is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Chat not found")
        return chat

    @router.delete("/chats/{chat_id}", status_code=status.HTTP_204_NO_CONTENT)
    def delete_chat(
        chat_id: str,
        identity: Annotated[tuple[Any, str], Depends(require_csrf)],
    ) -> Response:
        if not studio.delete_chat(identity[0].id, chat_id):
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Chat not found")
        return Response(status_code=status.HTTP_204_NO_CONTENT)

    @router.post("/chats/{chat_id}/messages")
    def generate_chat_message(
        chat_id: str,
        payload: ChatMessageRequest,
        identity: Annotated[tuple[Any, str], Depends(require_csrf)],
    ) -> dict[str, Any]:
        loaded = model_manager.active
        if loaded is None:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT, detail="No inference model is loaded"
            )
        engine = loaded.engine
        chat = studio.chat(identity[0].id, chat_id)
        if chat is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Chat not found")
        saved = studio.add_message(identity[0].id, chat_id, "user", payload.content)
        if saved is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Chat not found")
        messages = [Message("system", chat["system_prompt"])] if chat["system_prompt"] else []
        messages.extend(Message(item["role"], item["content"]) for item in chat["messages"])
        messages.append(Message("user", payload.content))
        params = SamplingParams(
            max_new_tokens=payload.max_tokens,
            temperature=payload.temperature,
            top_p=payload.top_p,
            top_k=payload.top_k,
            repetition_penalty=payload.repetition_penalty,
            seed=payload.seed,
        )
        params_dict = {
            "max_tokens": payload.max_tokens,
            "temperature": payload.temperature,
            "top_p": payload.top_p,
            "top_k": payload.top_k,
            "repetition_penalty": payload.repetition_penalty,
            "seed": payload.seed,
        }
        started = time.monotonic()
        with inference_lock:
            reply = engine.chat(messages, params)
        elapsed_ms = int((time.monotonic() - started) * 1000)
        output_tokens = len(reply.token_ids)
        assistant_message = studio.add_message(
            identity[0].id,
            chat_id,
            "assistant",
            reply.text,
            input_tokens=len(
                engine.fit_prompt(
                    engine.tokenizer.encode_conversation(messages, True), params.max_new_tokens
                )
            ),
            output_tokens=output_tokens,
            generation_ms=elapsed_ms,
            tokens_per_second=output_tokens / max(elapsed_ms / 1000, 0.001),
            model_id=loaded.model_id,
            checkpoint_id=loaded.checkpoint_id,
            generation_params=params_dict,
            model_config=loaded.config,
        )
        return {"user": saved, "assistant": assistant_message, "finish_reason": reply.finish_reason}

    return router
