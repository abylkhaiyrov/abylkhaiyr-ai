from __future__ import annotations

import asyncio
import logging
import threading
import time
from collections import deque
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, Literal

from llm.inference.sampling import SamplingParams
from llm.reasoning.engine import Budget, ReasoningEngine, classify_task
from llm.reasoning.stages import executed_checks, phase_of
from llm.studio.inference_profiles import PROFILES, effective_output_tokens, unavailable_reason
from llm.studio.model_manager import LoadedModelState, ModelManager, ModelManagerError
from llm.studio.service import TASK_STATUS, StudioService
from llm.studio.streaming import PERSIST_INTERVAL_SECONDS, DeltaBuffer
from llm.tokenizer.bpe import Message

logger = logging.getLogger(__name__)

MAX_QUEUE_SIZE = 8
SHUTDOWN_TIMEOUT_SECONDS = 15.0
DEFAULT_TEMPERATURE = 0.7
DEFAULT_TOP_P = 0.9
DEFAULT_TOP_K = 40
DEFAULT_REPETITION_PENALTY = 1.1


@dataclass
class GenerationRequest:
    request_id: str
    user_id: int
    chat_id: str
    content: str | None = None
    action: Literal["send", "edit", "regenerate", "continue"] = "send"
    target_message_id: str | None = None
    mode: Literal["standard", "reasoning", "precise", "creative"] = "standard"
    profile: str | None = None
    checkpoint_id: str | None = None
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    max_tokens: int | None = None
    repetition_penalty: float | None = None
    privileged: bool = False


@dataclass
class GenerationState:
    user_id: int
    chat_id: str
    request_id: str
    action: str = "send"
    cancel: threading.Event = field(default_factory=threading.Event)
    timed_out: bool = False
    status: str = "queued"
    message_id: str | None = None
    text: str = ""
    sequence: int = 0
    queue_position: int = 0
    queue_reason: str | None = None
    model_id: str | None = None
    checkpoint_id: str | None = None
    device: str | None = None
    reasoning_events: list[dict[str, Any]] = field(default_factory=list)
    started_at: float = 0.0
    first_token_ms: int | None = None
    persisted_at: float = 0.0
    prompt_tokens: int = 0


PublishCallback = Callable[[GenerationState, str, dict[str, Any]], Awaitable[None]]


class GenerationScheduler:
    def __init__(
        self,
        studio: StudioService,
        model_manager: ModelManager,
        inference_lock: threading.Lock,
        max_queue_size: int = MAX_QUEUE_SIZE,
    ) -> None:
        self.studio = studio
        self.model_manager = model_manager
        self.inference_lock = inference_lock
        self.max_queue_size = max_queue_size
        self.states: dict[tuple[int, str], GenerationState] = {}
        self.pending: deque[tuple[GenerationState, GenerationRequest, PublishCallback]] = deque()
        self.active_state: GenerationState | None = None
        self.worker: asyncio.Task[None] | None = None

    async def close(self, timeout: float = SHUTDOWN_TIMEOUT_SECONDS) -> None:
        for state in self.states.values():
            state.cancel.set()
        self.pending.clear()
        worker = self.worker
        if worker is None or worker.done():
            return
        done, _ = await asyncio.wait({worker}, timeout=timeout)
        if not done:
            worker.cancel()
            await asyncio.gather(worker, return_exceptions=True)

    def active_states_for_user(self, user_id: int) -> list[GenerationState]:
        return [
            state
            for state in self.states.values()
            if state.user_id == user_id and state.status in {"queued", "running"}
        ]

    async def submit(
        self, request: GenerationRequest, publish: PublishCallback
    ) -> tuple[str, dict[str, Any] | GenerationState]:
        key = (request.user_id, request.request_id)
        if key in self.states:
            return "existing", self._state_snapshot(self.states[key])
        existing_job = await asyncio.to_thread(
            self.studio.get_generation_job, request.user_id, request.request_id
        )
        if existing_job is not None:
            return "existing", self._job_snapshot(existing_job)
        chat = await asyncio.to_thread(
            self.studio.chat, request.user_id, request.chat_id, None, False
        )
        if chat is None:
            return "rejected", {"code": "chat_not_found", "detail": "Chat not found"}
        if request.action in {"send", "edit"} and (
            not request.content or not request.content.strip()
        ):
            return "rejected", {"code": "invalid_event", "detail": "Message content is required"}
        request.profile = request.profile or (chat.get("generation_settings") or {}).get("profile")
        rejection = await self._model_rejection(request, chat) or self._profile_rejection(request)
        if rejection is not None:
            return "rejected", rejection
        if len(self.pending) >= self.max_queue_size:
            return "rejected", {
                "code": "queue_full",
                "detail": f"Generation queue is full ({self.max_queue_size} requests waiting)",
            }
        target_checkpoint = await self._target_checkpoint(request, chat)
        active_model = self.model_manager.active
        if active_model is None and not target_checkpoint:
            return "rejected", {
                "code": "model_not_loaded",
                "detail": "No inference model is loaded. Select a checkpoint to start chatting.",
            }
        training_active = await asyncio.to_thread(self.studio.runtime.training_active)
        queue_reason: str | None = None
        if training_active:
            config = active_model.config if active_model is not None else {}
            can_cpu, available_ram, required_ram = await asyncio.to_thread(
                self.model_manager.can_run_on_cpu, None, config
            )
            if not can_cpu:
                return "rejected", {
                    "code": "insufficient_memory_during_training",
                    "detail": (
                        "Training is active on MPS and available RAM is below the safety threshold "
                        f"for CPU inference ({available_ram} < {required_ram})"
                    ),
                }
            queue_reason = "cpu_fallback_training_active"
        if self.active_state is not None or len(self.pending) > 0:
            queue_reason = (
                "training_active_and_queue_busy"
                if training_active
                else "another_generation_running"
            )
        elif (
            target_checkpoint
            and active_model is not None
            and target_checkpoint != active_model.checkpoint_id
        ):
            queue_reason = "switching_model"
        job, created = await asyncio.to_thread(
            self.studio.create_generation_job,
            request.request_id,
            request.user_id,
            request.chat_id,
            request.action,
            active_model.model_id if active_model is not None else chat.get("model_name"),
            target_checkpoint or (active_model.checkpoint_id if active_model is not None else None),
            queue_reason,
        )
        if not created:
            return "existing", self._job_snapshot(job)
        self._evict_completed()
        state = GenerationState(
            user_id=request.user_id,
            chat_id=request.chat_id,
            request_id=request.request_id,
            action=request.action,
            status="queued",
            queue_position=len(self.pending) + (1 if self.active_state is not None else 0),
            queue_reason=queue_reason,
            model_id=active_model.model_id if active_model is not None else chat.get("model_name"),
            checkpoint_id=(
                target_checkpoint
                or (active_model.checkpoint_id if active_model is not None else None)
            ),
            device="cpu" if training_active else (active_model.device if active_model else None),
        )
        self.states[key] = state
        self.pending.append((state, request, publish))
        if state.queue_position > 0 or queue_reason is not None:
            await self._emit(
                publish,
                state,
                "generation.queued",
                {
                    "queue_position": state.queue_position,
                    "queue_reason": state.queue_reason,
                    "model_id": state.model_id,
                    "checkpoint_id": state.checkpoint_id,
                    "device": state.device,
                },
            )
        if self.worker is None or self.worker.done():
            self.worker = asyncio.create_task(self._worker_loop())
        return "queued", state

    async def cancel(self, user_id: int, request_id: str, publish: PublishCallback) -> bool:
        key = (user_id, request_id)
        state = self.states.get(key)
        if state is None:
            return False
        state.cancel.set()
        removed = False
        for index, (queued_state, _req, _pub) in enumerate(list(self.pending)):
            if queued_state is state:
                del self.pending[index]
                removed = True
                break
        if removed:
            state.status = "cancelled"
            await asyncio.to_thread(
                self.studio.update_generation_job,
                state.request_id,
                status="cancelled",
                sequence_id=state.sequence + 1,
                error_code="cancelled_in_queue",
            )
            await self._emit(
                publish,
                state,
                "generation.cancelled",
                {
                    "text": state.text,
                    "reason": "cancelled_in_queue",
                    "model_id": state.model_id,
                    "checkpoint_id": state.checkpoint_id,
                },
            )
            await self._broadcast_queue_positions()
        return True

    async def _broadcast_queue_positions(self) -> None:
        offset = 1 if self.active_state is not None else 0
        for index, (queued_state, _req, publish) in enumerate(list(self.pending)):
            queued_state.queue_position = index + offset
            await self._emit(
                publish,
                queued_state,
                "generation.queued",
                {
                    "queue_position": queued_state.queue_position,
                    "queue_reason": queued_state.queue_reason,
                    "model_id": queued_state.model_id,
                    "checkpoint_id": queued_state.checkpoint_id,
                    "device": queued_state.device,
                },
            )

    async def _worker_loop(self) -> None:
        while self.pending:
            state, request, publish = self.pending.popleft()
            if state.cancel.is_set():
                continue
            self.active_state = state
            await self._broadcast_queue_positions()
            try:
                await self._execute(state, request, publish)
            finally:
                self.active_state = None

    async def _resolve_loaded_model(
        self,
        state: GenerationState,
        request: GenerationRequest,
        chat: dict[str, Any],
        publish: PublishCallback,
    ) -> LoadedModelState:
        target_checkpoint = await self._target_checkpoint(request, chat)
        active = self.model_manager.active
        training_active = await asyncio.to_thread(self.studio.runtime.training_active)
        if target_checkpoint and (active is None or active.checkpoint_id != target_checkpoint):
            await self._record_stage(
                publish,
                state,
                "model_loading",
                "running",
                {"checkpoint_id": target_checkpoint, "training_active": training_active},
            )
            await asyncio.to_thread(
                self.model_manager.switch,
                target_checkpoint,
                "cpu" if training_active else None,
            )
            active = self.model_manager.active
            await self._record_stage(
                publish,
                state,
                "model_loading",
                "completed",
                {
                    "checkpoint_id": active.checkpoint_id if active else target_checkpoint,
                    "device": active.device if active else None,
                },
            )
        if active is None:
            raise ModelManagerError("model_not_loaded", "No inference model is loaded")
        if training_active and active.device == "mps":
            active = await asyncio.to_thread(self.model_manager.ensure_cpu_fallback)
        return active

    async def _execute(
        self, state: GenerationState, request: GenerationRequest, publish: PublishCallback
    ) -> None:
        try:
            chat = await asyncio.to_thread(
                self.studio.chat, state.user_id, state.chat_id, None, False
            )
            if chat is None:
                raise ModelManagerError("chat_not_found", "Chat not found")
            loaded = await self._resolve_loaded_model(state, request, chat, publish)
            engine = loaded.engine
            state.model_id = loaded.model_id
            state.checkpoint_id = loaded.checkpoint_id
            state.device = loaded.device
            params, params_dict = self._resolve_sampling_params(request, chat, engine.max_positions)
            saved = await asyncio.to_thread(
                self.studio.start_chat_generation,
                state.user_id,
                state.chat_id,
                request.content,
                request.action,
                request.target_message_id,
                loaded.model_id,
                loaded.checkpoint_id,
                params_dict,
                loaded.config,
            )
            if saved is None:
                raise ModelManagerError("chat_not_found", "Chat not found")
            user_message, assistant_message, history_records, continuation_prefix = saved
            state.message_id = assistant_message["id"]
            state.text = continuation_prefix
            state.status = "running"
            await asyncio.to_thread(
                self.studio.update_generation_job,
                state.request_id,
                status="running",
                user_message_id=user_message["id"] if user_message else None,
                assistant_message_id=state.message_id,
                partial_text=state.text,
                model_id=loaded.model_id,
                checkpoint_id=loaded.checkpoint_id,
            )
            await self._emit(
                publish,
                state,
                "generation.started",
                {
                    "user": user_message,
                    "assistant": assistant_message,
                    "action": request.action,
                    "model_id": loaded.model_id,
                    "checkpoint_id": loaded.checkpoint_id,
                    "stage": loaded.stage,
                    "device": loaded.device,
                    "context_limit": engine.max_positions,
                    "generation_params": params_dict,
                    "text": state.text,
                },
            )
            history = [Message("system", chat["system_prompt"])] if chat["system_prompt"] else []
            window = int((chat.get("generation_settings") or {}).get("context_messages") or 0)
            visible = history_records[-window:] if window > 0 else history_records
            history.extend(Message(item["role"], item["content"]) for item in visible)
            prefix_tokens = (
                len(engine.tokenizer.encode(continuation_prefix)) if continuation_prefix else 0
            )
            profile = PROFILES.get(str(params_dict.get("profile") or ""))
            reserved_tokens = engine.output_reserve(params.max_new_tokens) + prefix_tokens
            if profile is not None:
                reserved_tokens = max(
                    reserved_tokens, engine.max_positions - profile.context_budget
                )
            messages, dropped_messages = engine.fit_messages(history, reserved_tokens)
            input_tokens = (
                len(engine.tokenizer.encode_conversation(messages, add_generation_prompt=True))
                + prefix_tokens
            )
            state.prompt_tokens = input_tokens
            logger.info(
                "generation request=%s checkpoint=%s profile=%s messages=%d dropped=%d "
                "prompt_tokens=%d prompt_digest=%s sampling=%s",
                state.request_id,
                loaded.checkpoint_id,
                params_dict.get("profile"),
                len(messages),
                dropped_messages,
                input_tokens,
                engine.chat_digest(messages),
                {
                    key: params_dict[key]
                    for key in ("temperature", "top_p", "top_k", "max_tokens", "repetition_penalty")
                },
            )
            last_user_text = next(
                (item["content"] for item in reversed(history_records) if item["role"] == "user"),
                request.content or "",
            )
            await self._emit(
                publish,
                state,
                "generation.prefill",
                {
                    "input_tokens": input_tokens,
                    "context_limit": engine.max_positions,
                    "model_id": loaded.model_id,
                    "checkpoint_id": loaded.checkpoint_id,
                    "device": loaded.device,
                },
            )
            loop = asyncio.get_running_loop()
            started = time.monotonic()
            state.started_at = started
            state.persisted_at = started
            generated_chars_start = len(state.text)

            use_reasoning = profile.reasoning.enabled if profile else request.mode == "reasoning"
            timer = None
            if profile is not None:

                def expire() -> None:
                    state.timed_out = True
                    state.cancel.set()

                timer = threading.Timer(profile.generation_timeout_seconds, expire)
                timer.daemon = True
                timer.start()

            if use_reasoning and request.action != "continue":

                def run_reasoning() -> tuple[str, int]:
                    def on_reasoning_event(event: dict[str, Any]) -> None:
                        future = asyncio.run_coroutine_threadsafe(
                            self._record_stage(
                                publish,
                                state,
                                event["stage"],
                                event["status"],
                                event.get("data", {}),
                            ),
                            loop,
                        )
                        future.result()

                    with self.inference_lock:
                        reasoning_engine = ReasoningEngine(engine)
                        budget = (
                            profile.budget(params.max_new_tokens)
                            if profile is not None
                            else Budget(
                                max_candidates=2,
                                max_retries=1,
                                max_new_tokens_total=max(params.max_new_tokens * 3, 256),
                                max_seconds=30.0,
                                max_new_tokens_per_candidate=params.max_new_tokens,
                            )
                        )
                        solution = reasoning_engine.solve_auto(
                            last_user_text,
                            budget=budget,
                            history=messages,
                            on_event=on_reasoning_event,
                            cancel=state.cancel,
                            temperature=params.temperature,
                            top_p=params.top_p,
                            top_k=params.top_k,
                            repetition_penalty=params.repetition_penalty,
                        )
                        return solution.text, solution.tokens_used

                result_text, output_tokens = await asyncio.to_thread(run_reasoning)
                if result_text and not state.cancel.is_set():
                    chunk_size = 24
                    for offset in range(0, len(result_text), chunk_size):
                        if state.cancel.is_set():
                            break
                        await self._delta(publish, state, result_text[offset : offset + chunk_size])
            else:
                task_kind = classify_task(last_user_text)
                await self._record_stage(
                    publish,
                    state,
                    "task_classification",
                    "completed",
                    {"kind": task_kind},
                )
                await self._record_stage(
                    publish,
                    state,
                    "context_preparation",
                    "completed",
                    {
                        "input_tokens": input_tokens,
                        "context_limit": engine.max_positions,
                        "message_count": len(messages),
                        "dropped_messages": dropped_messages,
                    },
                )

                def run_stream() -> None:
                    buffer = DeltaBuffer()

                    def flush(text: str | None) -> None:
                        if text:
                            asyncio.run_coroutine_threadsafe(
                                self._delta(publish, state, text, engine.tokenizer.encode),
                                loop,
                            ).result()

                    with self.inference_lock:
                        iterator = (
                            engine.stream_chat_continuation(messages, continuation_prefix, params)
                            if request.action == "continue"
                            else engine.stream_chat(messages, params)
                        )
                        for piece in iterator:
                            if state.cancel.is_set():
                                break
                            flush(buffer.add(piece))
                        flush(buffer.drain())

                await asyncio.to_thread(run_stream)
                new_text = state.text[generated_chars_start:]
                output_tokens = len(engine.tokenizer.encode(new_text)) if new_text else 0
                if not state.cancel.is_set():
                    await self._record_stage(
                        publish,
                        state,
                        "answer_formation",
                        "completed",
                        {"output_tokens": output_tokens},
                    )

            if timer is not None:
                timer.cancel()
            if state.timed_out and profile is not None:
                await self._record_stage(
                    publish,
                    state,
                    "generation_timeout",
                    "failed",
                    {"seconds": profile.generation_timeout_seconds},
                )
            elapsed_ms = int((time.monotonic() - started) * 1000)
            tps = round(output_tokens / max(elapsed_ms / 1000, 0.001), 2) if output_tokens else 0.0
            state.status = "cancelled" if state.cancel.is_set() else "completed"
            await asyncio.to_thread(
                self.studio.finalize_message,
                state.user_id,
                state.chat_id,
                state.message_id,
                state.text,
                state.status,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                generation_ms=elapsed_ms,
                tokens_per_second=tps,
                model_id=loaded.model_id,
                checkpoint_id=loaded.checkpoint_id,
                generation_params=params_dict,
                model_config=loaded.config,
                reasoning_trace=state.reasoning_events,
            )
            await asyncio.to_thread(
                self.studio.update_generation_job,
                state.request_id,
                status=state.status,
                sequence_id=state.sequence + 1,
                partial_text=state.text,
                reasoning_events=state.reasoning_events,
            )
            await self._emit(
                publish,
                state,
                f"generation.{state.status}",
                {
                    "elapsed_ms": elapsed_ms,
                    "first_token_ms": state.first_token_ms,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "tokens_per_second": tps,
                    "context_limit": engine.max_positions,
                    "model_id": loaded.model_id,
                    "checkpoint_id": loaded.checkpoint_id,
                    "device": loaded.device,
                    "reasoning_events": state.reasoning_events,
                    "checks": executed_checks(state.reasoning_events),
                    "text": state.text,
                },
            )
        except ModelManagerError as error:
            state.status = "failed"
            await asyncio.to_thread(
                self.studio.update_generation_job,
                state.request_id,
                status="failed",
                sequence_id=state.sequence + 1,
                partial_text=state.text,
                error_code=error.code,
                error_detail=error.detail,
            )
            await self._emit(
                publish,
                state,
                "generation.failed",
                {"code": error.code, "detail": error.detail, "text": state.text},
            )
        except Exception as error:
            logger.exception("Generation failed")
            state.status = "failed"
            if state.message_id:
                await asyncio.to_thread(
                    self.studio.finalize_message,
                    state.user_id,
                    state.chat_id,
                    state.message_id,
                    state.text,
                    "failed",
                    reasoning_trace=state.reasoning_events,
                )
            await asyncio.to_thread(
                self.studio.update_generation_job,
                state.request_id,
                status="failed",
                sequence_id=state.sequence + 1,
                partial_text=state.text,
                error_code="generation_failed",
                error_detail=str(error),
            )
            await self._emit(
                publish,
                state,
                "generation.failed",
                {"code": "generation_failed", "detail": str(error), "text": state.text},
            )

    async def _delta(
        self,
        publish: PublishCallback,
        state: GenerationState,
        piece: str,
        encode: Callable[[str], list[int]] | None = None,
    ) -> None:
        now = time.monotonic()
        if state.first_token_ms is None and state.started_at:
            state.first_token_ms = int((now - state.started_at) * 1000)
        state.text += piece
        data: dict[str, Any] = {"text": piece, "first_token_ms": state.first_token_ms}
        if encode is not None and state.started_at:
            produced = len(encode(state.text))
            generating = max(now - state.started_at - (state.first_token_ms or 0) / 1000, 1e-3)
            data["output_tokens"] = produced
            data["tokens_per_second"] = round(produced / max(now - state.started_at, 1e-3), 2)
            data["generating_seconds"] = round(generating, 3)
        await self._emit(publish, state, "generation.delta", data)
        if now - state.persisted_at >= PERSIST_INTERVAL_SECONDS:
            state.persisted_at = now
            await asyncio.to_thread(
                self.studio.update_message_content,
                state.user_id,
                state.chat_id,
                state.message_id,
                state.text,
            )
            await asyncio.to_thread(
                self.studio.update_generation_job,
                state.request_id,
                sequence_id=state.sequence,
                partial_text=state.text,
            )

    async def _record_stage(
        self,
        publish: PublishCallback,
        state: GenerationState,
        stage: str,
        status: str,
        data: dict[str, Any],
    ) -> None:
        entry = {
            "stage": stage,
            "status": status,
            "phase": phase_of(stage, data),
            "data": data,
            "timestamp": time.time(),
        }
        replaced = False
        for index in range(len(state.reasoning_events) - 1, -1, -1):
            existing = state.reasoning_events[index]
            if existing["stage"] == stage and existing["status"] == "running":
                state.reasoning_events[index] = entry
                replaced = True
                break
        if not replaced:
            state.reasoning_events.append(entry)
        await self._emit(
            publish,
            state,
            "generation.stage",
            {
                "stage": stage,
                "stage_status": status,
                "stage_data": data,
                "reasoning_events": state.reasoning_events,
            },
        )
        await asyncio.to_thread(
            self.studio.update_generation_job,
            state.request_id,
            sequence_id=state.sequence,
            reasoning_events=state.reasoning_events,
        )

    async def _emit(
        self,
        publish: PublishCallback,
        state: GenerationState,
        event_type: str,
        data: dict[str, Any],
    ) -> None:
        await publish(state, event_type, data)
        status = event_type.removeprefix("generation.")
        if status in {"completed", "failed", "cancelled"}:
            await asyncio.to_thread(
                self.studio.notify_generation,
                state.user_id,
                state.chat_id,
                state.request_id,
                status,
            )

    async def _model_rejection(
        self, request: GenerationRequest, chat: dict[str, Any]
    ) -> dict[str, str] | None:
        if request.privileged:
            return None
        requested = request.checkpoint_id is not None
        model_name = chat.get("model_name")
        revoked = bool(model_name) and (
            await asyncio.to_thread(self.studio.resolve_user_checkpoint, str(model_name)) is None
        )
        if requested or revoked:
            return {
                "code": "model_not_allowed",
                "detail": "This model is not enabled for your account",
            }
        return None

    async def _target_checkpoint(
        self, request: GenerationRequest, chat: dict[str, Any]
    ) -> str | None:
        if request.privileged:
            return request.checkpoint_id or chat.get("checkpoint")
        model_name = chat.get("model_name")
        if not model_name:
            return None
        return await asyncio.to_thread(self.studio.resolve_user_checkpoint, model_name)

    def _profile_rejection(self, request: GenerationRequest) -> dict[str, str] | None:
        if request.profile is None:
            return None
        profile = PROFILES.get(request.profile)
        if profile is None:
            return {"code": "invalid_profile", "detail": "Unknown inference profile"}
        if not request.privileged and not self.studio.profile_allowed(profile.id):
            return {
                "code": "profile_not_allowed",
                "detail": "This inference profile is not enabled for your account",
            }
        active = self.model_manager.active
        reason = (
            None if active is None else unavailable_reason(profile, active.engine.max_positions)
        )
        if reason is not None:
            return {"code": "profile_unavailable", "detail": reason}
        return None

    @staticmethod
    def _resolve_sampling_params(
        request: GenerationRequest, chat: dict[str, Any], max_positions: int
    ) -> tuple[SamplingParams, dict[str, Any]]:
        saved = chat.get("generation_settings") or {}
        mode = request.mode or saved.get("mode") or "standard"
        profile = PROFILES.get(request.profile or saved.get("profile") or "")
        default_temp = DEFAULT_TEMPERATURE
        default_top_p = DEFAULT_TOP_P
        default_top_k = DEFAULT_TOP_K
        default_rep = DEFAULT_REPETITION_PENALTY
        greedy = False
        if mode == "precise":
            default_temp = 0.2
            default_top_p = 0.85
            default_top_k = 20
        elif mode == "creative":
            default_temp = 1.0
            default_top_p = 0.95
            default_top_k = 80
        temperature = float(
            request.temperature
            if request.temperature is not None
            else saved.get("temperature", default_temp)
        )
        top_p = float(
            request.top_p if request.top_p is not None else saved.get("top_p", default_top_p)
        )
        top_k = int(
            request.top_k if request.top_k is not None else saved.get("top_k", default_top_k)
        )
        repetition_penalty = float(
            request.repetition_penalty
            if request.repetition_penalty is not None
            else saved.get("repetition_penalty", default_rep)
        )
        max_cap = max(1, max_positions - 16)
        requested_max = int(
            request.max_tokens
            if request.max_tokens is not None
            else saved.get(
                "max_tokens",
                profile.max_output_tokens if profile else min(256, max(1, max_positions // 2)),
            )
        )
        max_new_tokens = max(1, min(requested_max, max_cap))
        if profile is not None:
            max_new_tokens = min(max_new_tokens, effective_output_tokens(profile, max_positions))
        greedy = temperature <= 1e-5
        params = SamplingParams(
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            greedy=greedy,
        )
        params_dict = {
            "mode": mode,
            "profile": None if profile is None else profile.id,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "max_tokens": max_new_tokens,
            "repetition_penalty": repetition_penalty,
        }
        return params, params_dict

    def _evict_completed(self) -> None:
        while len(self.states) >= 1000:
            completed = next(
                (
                    key
                    for key, item in self.states.items()
                    if item.status not in {"queued", "running"}
                ),
                None,
            )
            if completed is None:
                break
            del self.states[completed]

    @staticmethod
    def _state_snapshot(state: GenerationState) -> dict[str, Any]:
        return {
            "type": "generation.snapshot",
            "task_status": TASK_STATUS.get(state.status, "PROCESSING"),
            "request_id": state.request_id,
            "chat_id": state.chat_id,
            "message_id": state.message_id,
            "sequence_id": state.sequence,
            "status": state.status,
            "text": state.text,
            "model_id": state.model_id,
            "checkpoint_id": state.checkpoint_id,
            "device": state.device,
            "queue_position": state.queue_position,
            "queue_reason": state.queue_reason,
            "reasoning_events": state.reasoning_events,
        }

    @staticmethod
    def _job_snapshot(job: dict[str, Any]) -> dict[str, Any]:
        return {
            "type": "generation.snapshot",
            "request_id": job["request_id"],
            "chat_id": job["chat_id"],
            "message_id": job["assistant_message_id"],
            "sequence_id": job["sequence_id"],
            "task_status": job["task_status"],
            "status": job["status"],
            "text": job["partial_text"],
            "model_id": job["model_id"],
            "checkpoint_id": job["checkpoint_id"],
            "queue_reason": job["queue_reason"],
            "reasoning_events": job["reasoning_events"],
        }
