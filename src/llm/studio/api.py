import asyncio
import threading
import time
from collections.abc import AsyncIterator, Callable, Sequence
from contextlib import AbstractAsyncContextManager, AsyncExitStack, asynccontextmanager
from typing import TYPE_CHECKING, Annotated, Any, Literal
from urllib.parse import quote

from fastapi import APIRouter, Depends, HTTPException, Query, Response, WebSocket, status
from pydantic import BaseModel, ConfigDict, Field, field_validator

from llm.inference.sampling import SamplingParams
from llm.studio.api_accounts import create_account_router
from llm.studio.evaluation_center import IncompatibleEvaluations
from llm.studio.evaluation_metrics import metric_catalog
from llm.studio.inference_profiles import PROFILE_IDS, ProfilePolicyError
from llm.studio.model_catalog import ModelPolicyError
from llm.studio.model_manager import ModelManager, ModelManagerError
from llm.studio.processes import ProcessError
from llm.studio.security import build_dependencies
from llm.studio.service import StudioService
from llm.tokenizer.bpe import Message

if TYPE_CHECKING:
    from llm.studio.websocket import WebSocketManager


class TrainingStartRequest(BaseModel):
    config_path: str = Field(min_length=1, max_length=1024)
    stage: Literal["pretrain", "sft", "dpo"] | None = None


def valid_profile_setting(settings: dict[str, Any] | None) -> dict[str, Any] | None:
    if settings is not None and settings.get("profile") not in {None, *PROFILE_IDS}:
        raise ValueError("unknown_inference_profile")
    return settings


class ChatCreateRequest(BaseModel):
    title: str = Field(default="New chat", max_length=256)
    model_name: str | None = Field(default=None, max_length=256)
    checkpoint: str | None = Field(default=None, max_length=1024)
    system_prompt: str = Field(default="", max_length=20000)
    generation_settings: dict[str, Any] = Field(default_factory=dict)

    _profile = field_validator("generation_settings")(valid_profile_setting)


class ChatUpdateRequest(BaseModel):
    title: str | None = Field(default=None, max_length=256)
    model_name: str | None = Field(default=None, max_length=256)
    checkpoint: str | None = Field(default=None, max_length=1024)
    system_prompt: str | None = Field(default=None, max_length=20000)
    generation_settings: dict[str, Any] | None = None
    pinned: bool | None = None
    archived: bool | None = None

    _profile = field_validator("generation_settings")(valid_profile_setting)


class AllowedModelsRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    ids: list[Annotated[str, Field(max_length=200)]] = Field(max_length=500)


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


def studio_lifespan(
    websocket_manager: "WebSocketManager", model_manager: ModelManager, studio: StudioService
) -> Callable[[Any], AbstractAsyncContextManager[None]]:
    @asynccontextmanager
    async def lifespan(_: Any) -> AsyncIterator[None]:
        async with AsyncExitStack() as stack:
            stack.push_async_callback(asyncio.to_thread, studio.close)
            stack.push_async_callback(asyncio.to_thread, model_manager.unload)
            stack.push_async_callback(websocket_manager.close)
            yield

    return lifespan


def create_studio_router(
    studio: StudioService,
    model_manager: ModelManager,
    inference_lock: threading.Lock,
) -> APIRouter:
    from llm.studio.websocket import WebSocketManager

    websocket_manager = WebSocketManager(studio, model_manager, inference_lock)
    router = APIRouter(
        prefix="/api", lifespan=studio_lifespan(websocket_manager, model_manager, studio)
    )
    deps = build_dependencies(studio)
    current_identity, require_csrf = deps.session, deps.csrf
    admin_identity, admin_csrf = deps.admin, deps.admin_csrf
    router.include_router(create_account_router(studio, deps))

    @router.websocket("/ws")
    async def websocket_endpoint(socket: WebSocket) -> None:
        await websocket_manager.handle(socket)

    def reject_during_generation() -> None:
        if inference_lock.locked():
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="generation_active")

    @router.get("/pipelines")
    def pipeline_history(
        _identity: Annotated[tuple[Any, str], Depends(admin_identity)],
    ) -> list[dict[str, Any]]:
        return studio.runtime.pipeline_history()

    @router.get("/catalog/runs")
    def run_catalog_page(
        _: Annotated[tuple[Any, str], Depends(admin_identity)],
        status_filter: Annotated[str | None, Query(alias="status")] = None,
        stage: str | None = None,
        version: str | None = None,
        q: Annotated[str | None, Query(max_length=100)] = None,
        sort: Annotated[Literal["newest", "oldest"], Query()] = "newest",
        limit: Annotated[int, Query(ge=1, le=200)] = 50,
        offset: Annotated[int, Query(ge=0)] = 0,
    ) -> dict[str, Any]:
        return studio.run_catalog.page(
            status=status_filter,
            stage=stage,
            version=version,
            query=q,
            oldest_first=sort == "oldest",
            limit=limit,
            offset=offset,
        )

    @router.get("/catalog/runs/{run_id:path}")
    def run_catalog_entry(
        run_id: str,
        _: Annotated[tuple[Any, str], Depends(admin_identity)],
    ) -> dict[str, Any]:
        entry = studio.run_catalog.get(run_id)
        if entry is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="run_not_found")
        pipeline = (
            next(
                (
                    item
                    for item in studio.runtime.pipeline_history()
                    if f"pipeline:{item['id']}" == run_id
                ),
                None,
            )
            if entry["kind"] == "pipeline"
            else None
        )
        detail = studio.catalog.detail(entry["name"]) if entry["kind"] == "training" else None
        return {"run": entry, "pipeline": pipeline, "detail": detail}

    @router.get("/runtime")
    def runtime_state(
        _: Annotated[tuple[Any, str], Depends(admin_identity)],
    ) -> dict[str, Any]:
        return studio.runtime_state()

    @router.get("/settings")
    def settings(identity: Annotated[tuple[Any, str], Depends(current_identity)]) -> dict[str, Any]:
        user, _ = identity
        return {
            "local_mode": True,
            "requires_password_change": user.requires_password_change,
            "session_ttl_seconds": 12 * 60 * 60,
        }

    @router.get("/system")
    def system(_: Annotated[tuple[Any, str], Depends(admin_identity)]) -> dict[str, Any]:
        return studio.record_system_metric()

    @router.get("/system/history")
    def system_history(
        _: Annotated[tuple[Any, str], Depends(admin_identity)],
        seconds: int = 300,
    ) -> list[dict[str, Any]]:
        return studio.system_history(min(max(seconds, 300), 86400))

    @router.get("/processes")
    def processes(_: Annotated[tuple[Any, str], Depends(admin_identity)]) -> list[dict[str, Any]]:
        return studio.list_processes()

    @router.get("/processes/{process_id}/events")
    def process_events(
        process_id: str,
        _: Annotated[tuple[Any, str], Depends(admin_identity)],
    ) -> Sequence[dict[str, Any]]:
        events = studio.processes.events(process_id)
        if events is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="process_not_found")
        return events

    @router.post("/training")
    def start_training(
        payload: TrainingStartRequest,
        _: Annotated[tuple[Any, str], Depends(admin_csrf)],
    ) -> dict[str, Any]:
        try:
            return studio.start_training_checked(payload.config_path, payload.stage)
        except (ProcessError, ValueError) as error:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail=str(error)
            ) from error

    @router.post("/pipeline/stop")
    def stop_pipeline(
        _: Annotated[tuple[Any, str], Depends(admin_csrf)],
    ) -> dict[str, Any]:
        try:
            return studio.stop_pipeline()
        except ProcessError as error:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(error)) from error

    @router.post("/training/{process_id}/pause")
    def pause_training(
        process_id: str,
        _: Annotated[tuple[Any, str], Depends(admin_csrf)],
    ) -> dict[str, Any]:
        try:
            return studio.processes.pause(process_id)
        except ProcessError as error:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(error)) from error

    @router.post("/training/{process_id}/stop")
    def stop_training(
        process_id: str,
        _: Annotated[tuple[Any, str], Depends(admin_csrf)],
    ) -> dict[str, Any]:
        try:
            return studio.processes.stop(process_id)
        except ProcessError as error:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(error)) from error

    @router.get("/training/metrics")
    def training_metrics(
        _: Annotated[tuple[Any, str], Depends(admin_identity)],
        run: str | None = None,
        max_points: int = 600,
    ) -> list[dict[str, float]]:
        return studio.metrics(run, max_points)

    @router.get("/training/configs")
    def training_configs(
        _: Annotated[tuple[Any, str], Depends(admin_identity)],
    ) -> list[dict[str, str]]:
        return studio.catalog.config_list()

    @router.get("/runs")
    def runs(_: Annotated[tuple[Any, str], Depends(admin_identity)]) -> list[dict[str, Any]]:
        return studio.catalog.runs()

    @router.get("/runs/{name}")
    def run_detail(
        name: str, _: Annotated[tuple[Any, str], Depends(admin_identity)]
    ) -> dict[str, Any]:
        detail = studio.catalog.detail(name)
        if detail is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Run not found")
        return detail

    @router.get("/runs/{name}/metrics")
    def run_metrics(
        name: str,
        _: Annotated[tuple[Any, str], Depends(admin_identity)],
        max_points: int = 600,
    ) -> list[dict[str, float]]:
        rows = studio.catalog.metrics(name, min(max(max_points, 10), 3000))
        if rows is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Run not found")
        return rows

    @router.get("/runs/{name}/logs")
    def run_logs(
        name: str,
        _: Annotated[tuple[Any, str], Depends(admin_identity)],
        limit: int = 300,
    ) -> list[str]:
        lines = studio.catalog.logs(name, max(limit, 1))
        if lines is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Run not found")
        return lines

    @router.get("/models")
    def models(_: Annotated[tuple[Any, str], Depends(admin_identity)]) -> list[dict[str, Any]]:
        return studio.models()

    @router.get("/models/registry")
    def model_registry(
        _: Annotated[tuple[Any, str], Depends(admin_identity)],
    ) -> dict[str, Any]:
        return model_manager.registry()

    @router.get("/models/checkpoints")
    def model_checkpoints(
        _: Annotated[tuple[Any, str], Depends(admin_identity)],
        q: Annotated[str | None, Query(max_length=100)] = None,
        status_filter: Annotated[str | None, Query(alias="status", max_length=32)] = None,
        stage: Annotated[str | None, Query(max_length=32)] = None,
        limit: Annotated[int, Query(ge=1, le=200)] = 50,
        offset: Annotated[int, Query(ge=0)] = 0,
    ) -> dict[str, Any]:
        return studio.checkpoint_page(q, status_filter, stage, offset, limit)

    @router.get("/models/catalog")
    def model_catalog(
        _: Annotated[tuple[Any, str], Depends(admin_identity)],
    ) -> list[dict[str, Any]]:
        return studio.model_catalog()

    @router.put("/models/allowed")
    def set_allowed_models(
        payload: AllowedModelsRequest,
        _: Annotated[tuple[Any, str], Depends(admin_csrf)],
    ) -> list[dict[str, Any]]:
        try:
            studio.set_allowed_models(payload.ids)
        except ModelPolicyError as error:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(error)
            ) from error
        return studio.model_catalog()

    @router.get("/chat/models")
    def chat_models(
        _: Annotated[tuple[Any, str], Depends(current_identity)],
    ) -> list[dict[str, Any]]:
        return studio.user_models()

    @router.get("/inference/profiles")
    def inference_profiles(
        identity: Annotated[tuple[Any, str], Depends(current_identity)],
    ) -> list[dict[str, Any]]:
        active = model_manager.active_info()
        context = None if active is None else active.get("context_length")
        return studio.inference_profiles(context, identity[0].role == "ADMIN")

    @router.put("/inference/profiles/allowed")
    def set_allowed_profiles(
        payload: AllowedModelsRequest,
        _: Annotated[tuple[Any, str], Depends(admin_csrf)],
    ) -> list[dict[str, Any]]:
        try:
            studio.set_allowed_profiles(payload.ids)
        except ProfilePolicyError as error:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(error)
            ) from error
        active = model_manager.active_info()
        return studio.inference_profiles(
            None if active is None else active.get("context_length"), True
        )

    @router.get("/models/active")
    def active_model(
        _: Annotated[tuple[Any, str], Depends(admin_identity)],
    ) -> dict[str, Any] | None:
        return model_manager.active_info()

    @router.post("/models/load")
    def load_model(
        payload: ModelLoadRequest,
        _: Annotated[tuple[Any, str], Depends(admin_csrf)],
    ) -> dict[str, Any]:
        reject_during_generation()
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
        _: Annotated[tuple[Any, str], Depends(admin_csrf)],
    ) -> dict[str, Any]:
        reject_during_generation()
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
        _: Annotated[tuple[Any, str], Depends(admin_csrf)],
    ) -> dict[str, Any]:
        reject_during_generation()
        return model_manager.unload()

    @router.get("/checkpoints")
    def checkpoints(
        _: Annotated[tuple[Any, str], Depends(admin_identity)],
    ) -> list[dict[str, Any]]:
        return studio.models()

    @router.get("/datasets")
    def datasets(_: Annotated[tuple[Any, str], Depends(admin_identity)]) -> list[dict[str, Any]]:
        return studio.datasets()

    @router.get("/experiments")
    def experiments(
        _: Annotated[tuple[Any, str], Depends(admin_identity)],
    ) -> list[dict[str, Any]]:
        return studio.experiments()

    @router.get("/evaluation")
    def evaluation(
        _: Annotated[tuple[Any, str], Depends(admin_identity)],
    ) -> list[dict[str, Any]]:
        return studio.evaluations()

    @router.get("/evaluation/metrics")
    def evaluation_metrics(
        _: Annotated[tuple[Any, str], Depends(admin_identity)],
    ) -> list[dict[str, Any]]:
        return metric_catalog()

    @router.get("/evaluation/groups")
    def evaluation_groups(
        _: Annotated[tuple[Any, str], Depends(admin_identity)],
        q: Annotated[str | None, Query(max_length=100)] = None,
        limit: Annotated[int, Query(ge=1, le=200)] = 50,
        offset: Annotated[int, Query(ge=0)] = 0,
    ) -> dict[str, Any]:
        return studio.evaluation_groups(q, offset, limit)

    @router.get("/evaluation/groups/{group_id}")
    def evaluation_group(
        group_id: str,
        _: Annotated[tuple[Any, str], Depends(admin_identity)],
    ) -> dict[str, Any]:
        group = studio.evaluation_group(group_id)
        if group is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="group_not_found")
        return group

    @router.get("/evaluation/compare")
    def evaluation_compare(
        _: Annotated[tuple[Any, str], Depends(admin_identity)],
        group: Annotated[str, Query(min_length=1, max_length=200)],
        checkpoint: Annotated[list[str], Query(min_length=2, max_length=8)],
    ) -> dict[str, Any]:
        try:
            result = studio.compare_evaluations(group, checkpoint)
        except IncompatibleEvaluations as error:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT, detail="incompatible_evaluations"
            ) from error
        if result is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="group_not_found")
        return result

    @router.get("/logs")
    def logs(
        _: Annotated[tuple[Any, str], Depends(admin_identity)],
        category: str | None = None,
        offset: int = 0,
        limit: int = 100,
    ) -> dict[str, Any]:
        return studio.logs(category, max(offset, 0), limit)

    @router.get("/chats")
    def chats(
        identity: Annotated[tuple[Any, str], Depends(current_identity)],
        query: str | None = None,
        archived: bool = False,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        return studio.chats(identity[0].id, query, archived, limit, offset)

    @router.get("/chat/model")
    def chat_model(
        identity: Annotated[tuple[Any, str], Depends(current_identity)],
    ) -> dict[str, Any] | None:
        active = model_manager.active_info()
        if active is None:
            return None
        return {
            "id": active.get("id"),
            "display_name": active.get("display_name") or active.get("name"),
            "stage": active.get("stage"),
            "context_length": active.get("context_length"),
            "device": active.get("device"),
        }

    def require_user_model(model_name: str | None, privileged: bool) -> str | None:
        if privileged or model_name is None:
            return None
        return model_name if studio.resolve_user_checkpoint(model_name) is not None else None

    @router.post("/chats", status_code=status.HTTP_201_CREATED)
    def create_chat(
        payload: ChatCreateRequest,
        identity: Annotated[tuple[Any, str], Depends(require_csrf)],
    ) -> dict[str, Any]:
        privileged = identity[0].role == "ADMIN"
        user_model = require_user_model(payload.model_name, privileged)
        return studio.create_chat(
            identity[0].id,
            payload.title,
            payload.model_name if privileged else user_model,
            payload.checkpoint if privileged else None,
            payload.system_prompt,
            payload.generation_settings,
        )

    @router.get("/chats/{chat_id}")
    def get_chat(
        chat_id: str,
        identity: Annotated[tuple[Any, str], Depends(current_identity)],
        limit: int = 100,
    ) -> dict[str, Any]:
        chat = studio.chat(identity[0].id, chat_id, limit)
        if chat is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Chat not found")
        return chat

    @router.patch("/chats/{chat_id}")
    def update_chat(
        chat_id: str,
        payload: ChatUpdateRequest,
        identity: Annotated[tuple[Any, str], Depends(require_csrf)],
    ) -> dict[str, Any]:
        privileged = identity[0].role == "ADMIN"
        user_model = require_user_model(payload.model_name, privileged)
        chat = studio.update_chat(
            identity[0].id,
            chat_id,
            title=payload.title,
            model_name=payload.model_name if privileged else user_model,
            checkpoint=payload.checkpoint if privileged else None,
            system_prompt=payload.system_prompt,
            generation_settings=payload.generation_settings,
            pinned=payload.pinned,
            archived=payload.archived,
        )
        if chat is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Chat not found")
        return chat

    @router.get("/chats/{chat_id}/export")
    def export_chat(
        chat_id: str,
        identity: Annotated[tuple[Any, str], Depends(current_identity)],
        format: Literal["md", "json"] = "md",
    ) -> Response:
        exported = studio.export_chat(identity[0].id, chat_id, format)
        if exported is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Chat not found")
        filename, media_type, body = exported
        return Response(
            body,
            media_type=media_type,
            headers={"Content-Disposition": f"attachment; filename*=UTF-8''{quote(filename)}"},
        )

    @router.post("/chats/{chat_id}/messages/{message_id}/variants/{index}")
    def select_variant(
        chat_id: str,
        message_id: str,
        index: int,
        identity: Annotated[tuple[Any, str], Depends(require_csrf)],
    ) -> dict[str, Any]:
        reject_during_generation()
        try:
            message = studio.select_variant(identity[0].id, chat_id, message_id, index)
        except ValueError as error:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="variant_not_found"
            ) from error
        if message is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="message_not_found")
        return message

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
