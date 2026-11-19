from __future__ import annotations

import json
import re
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sqlalchemy import select

from llm.core.config import load_experiment_config
from llm.core.errors import ConfigError
from llm.evaluation.registry import load_registry
from llm.studio.auth import AuthenticationService
from llm.studio.database import create_sqlite_engine, ensure_studio_schema, make_session_factory
from llm.studio.model_manager import ModelManager
from llm.studio.models import Chat, GenerationJob, MessageRecord, ProcessEvent, SystemMetric
from llm.studio.processes import ProcessManager
from llm.studio.runtime import RuntimeState
from llm.studio.telemetry import (
    checkpoint_inventory,
    latest_metrics,
    run_metrics,
    storage_snapshot,
    system_snapshot,
)

SECRET_PATTERN = re.compile(r"(?i)(authorization|password|token|secret)(\s*[=:]\s*)([^\s,]+)")
STORAGE_REFRESH_SECONDS = 30.0


class StudioService:
    def __init__(self, root: Path, inference_model: str | None = None) -> None:
        self.root = root.resolve()
        self.inference_model = inference_model
        self.model_manager: ModelManager | None = None
        database_path = self.root / "studio" / "studio.db"
        self.engine = create_sqlite_engine(database_path)
        ensure_studio_schema(self.engine)
        self.session_factory = make_session_factory(self.engine)
        self.auth = AuthenticationService(self.session_factory)
        self.auth.initialize()
        self.processes = ProcessManager(self.session_factory, self.root)
        self.runtime = RuntimeState(self.root)
        self.storage_metrics = storage_snapshot(self.root)
        self.storage_metrics_at = time.monotonic()
        self._reconcile_interrupted_jobs()

    def attach_model_manager(self, manager: ModelManager) -> None:
        self.model_manager = manager

    def _reconcile_interrupted_jobs(self) -> None:
        with self.session_factory.begin() as database:
            stale = database.scalars(
                select(GenerationJob).where(GenerationJob.status.in_(["queued", "running"]))
            ).all()
            for job in stale:
                job.status = "interrupted"
                job.error_code = "server_restarted"
                job.error_detail = "Backend restarted before generation completed"
                if job.assistant_message_id:
                    message = database.scalar(
                        select(MessageRecord).where(MessageRecord.id == job.assistant_message_id)
                    )
                    if message is not None and message.status in {"queued", "running"}:
                        message.status = "interrupted"

    def dashboard(self) -> dict[str, Any]:
        processes = self.processes.list()
        pipeline = self.runtime.pipeline()
        checkpoints = checkpoint_inventory(self.root)
        active = next(
            (item for item in processes if item["status"] in {"running", "pausing"}), None
        )
        if pipeline is not None:
            active = pipeline
        metrics: dict[str, float] = {}
        if pipeline is not None:
            metrics = pipeline["metrics"]
        elif active is not None and active.get("output_dir"):
            output_dir = (self.root / active["output_dir"]).resolve()
            if (
                output_dir == self.root / "experiments"
                or self.root / "experiments" in output_dir.parents
            ):
                from llm.studio.telemetry import read_metrics

                for record in read_metrics(output_dir / "metrics.jsonl"):
                    metrics.update(record)
        training = {
            "stage": None if active is None else active["kind"],
            "process_id": None if active is None else active["id"],
            "step": metrics.get("step"),
            "max_steps": pipeline["max_steps"] if pipeline else self._max_steps(active),
            "tokens_seen": metrics.get("perf/tokens_seen"),
            "train_loss": metrics.get("train/loss"),
            "validation_loss": metrics.get("validation/loss"),
            "perplexity": metrics.get("validation/perplexity", metrics.get("train/perplexity")),
            "throughput": metrics.get("perf/tokens_per_second"),
            "eta_seconds": metrics.get("perf/eta_seconds"),
            "started_at": None if active is None else active["started_at"],
            "model_name": pipeline["model_name"] if pipeline else None,
            "checkpoint": pipeline["checkpoint"] if pipeline else None,
        }
        active_model = self.model_manager.active_info() if self.model_manager is not None else None
        resolved_inference_model = (
            active_model["model_id"] if active_model is not None else self.inference_model
        )
        return {
            "system_status": "busy" if active is not None else "ready",
            "active_model": active_model,
            "inference_model": resolved_inference_model,
            "training": training,
            "active_process": active,
            "pipeline": self._pipeline(processes, checkpoints, pipeline),
            "system": self._system_snapshot(),
            "counts": {
                "active_processes": sum(item["status"] == "running" for item in processes)
                + int(pipeline is not None),
                "checkpoints": len(checkpoints),
            },
            "events": self.events(limit=12),
        }

    def record_system_metric(self) -> dict[str, Any]:
        snapshot = self._system_snapshot()
        with self.session_factory.begin() as database:
            database.add(SystemMetric(payload=json.dumps(snapshot)))
            self._prune_system_metrics(database)
        return snapshot

    def list_processes(self) -> list[dict[str, Any]]:
        processes = self.processes.list()
        pipeline = self.runtime.pipeline()
        if pipeline is not None:
            processes.insert(
                0,
                {
                    **pipeline,
                    "command": "scripts/run_pipeline.py",
                    "completed_at": None,
                    "exit_code": None,
                    "error_reason": None,
                    "cpu_percent": None,
                    "managed": False,
                },
            )
        return processes

    def system_history(self, seconds: int) -> list[dict[str, Any]]:
        cutoff = datetime.now(UTC).replace(tzinfo=None).timestamp() - seconds
        with self.session_factory() as database:
            rows = database.scalars(
                select(SystemMetric).order_by(SystemMetric.recorded_at.desc()).limit(5000)
            ).all()
        items = []
        for row in reversed(rows):
            if row.recorded_at.timestamp() < cutoff:
                continue
            try:
                items.append(json.loads(row.payload))
            except json.JSONDecodeError:
                continue
        return items

    def metrics(self, run_name: str | None, limit: int) -> list[dict[str, float]]:
        if run_name is None:
            pipeline = self.runtime.pipeline()
            if pipeline is not None:
                output_dir = (self.root / pipeline["output_dir"]).resolve()
                if self.root / "experiments" in output_dir.parents:
                    from llm.studio.telemetry import read_metrics

                    return read_metrics(output_dir / "metrics.jsonl", min(max(limit, 1), 3000))
            return []
        return run_metrics(self.root, run_name, min(max(limit, 1), 3000))

    def models(self) -> list[dict[str, Any]]:
        if self.model_manager is not None:
            return self.model_manager.list_models()
        metrics_by_run = {
            path.parent.name: latest_metrics(self.root, path.parent.name)
            for path in (self.root / "experiments").glob("*/metrics.jsonl")
        }
        models = checkpoint_inventory(self.root)
        for model in models:
            metadata = model["metadata"]
            run_name = model["name"]
            config = metadata.get("model_config", {})
            model["architecture"] = config
            model["parameters"] = self._parameter_count(config)
            model["vocab_size"] = config.get("vocab_size")
            model["context_length"] = config.get("max_position_embeddings")
            model["metrics"] = metrics_by_run.get(run_name, {})
        return models

    def datasets(self) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        for manifest_path in self.root.glob("data/**/manifest.json"):
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            results.append(
                {
                    "name": manifest_path.parent.name,
                    "path": str(manifest_path.parent.relative_to(self.root)),
                    "manifest": manifest,
                    "size_bytes": sum(
                        item.stat().st_size
                        for item in manifest_path.parent.rglob("*")
                        if item.is_file()
                    ),
                }
            )
        return results

    def experiments(self) -> list[dict[str, Any]]:
        return load_registry(self.root / "experiments" / "registry.jsonl")

    def evaluations(self) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        seen: set[Path] = set()
        for pattern in ("experiments/**/evaluation.json", "experiments/*_eval.json"):
            for path in sorted(self.root.glob(pattern)):
                if path in seen:
                    continue
                seen.add(path)
                try:
                    data = json.loads(path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    continue
                payload = data.get("metrics", data) if isinstance(data, dict) else {}
                numeric = {
                    key: float(value)
                    for key, value in payload.items()
                    if isinstance(value, (int, float))
                }
                results.append({"path": str(path.relative_to(self.root)), "results": numeric})
        return results

    def events(self, limit: int = 100) -> list[dict[str, Any]]:
        with self.session_factory() as database:
            rows = database.scalars(
                select(ProcessEvent).order_by(ProcessEvent.created_at.desc()).limit(min(limit, 500))
            ).all()
        return [
            {
                "process_id": row.process_id,
                "level": row.level,
                "message": self._redact(row.message),
                "created_at": row.created_at.isoformat(),
            }
            for row in rows
        ]

    def logs(self, category: str | None, offset: int, limit: int) -> dict[str, Any]:
        files = sorted(
            {
                *list((self.root / "experiments").glob("**/train.log")),
                *list((self.root / "experiments").glob("**/pipeline.log")),
                *list((self.root / "experiments").glob("**/run.log")),
            }
        )
        entries: list[dict[str, Any]] = []
        for path in files:
            log_category = "pipeline" if path.name in {"pipeline.log", "run.log"} else "training"
            for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
                entries.append(
                    {
                        "category": log_category,
                        "source": str(path.relative_to(self.root)),
                        "message": self._redact(line),
                    }
                )
        if category is not None:
            entries = [item for item in entries if item["category"] == category]
        entries.reverse()
        page = entries[offset : offset + min(max(limit, 1), 500)]
        return {"items": page, "total": len(entries), "offset": offset, "limit": limit}

    def create_chat(
        self,
        user_id: int,
        title: str,
        model_name: str | None,
        checkpoint: str | None,
        system_prompt: str,
        generation_settings: dict[str, Any],
    ) -> dict[str, Any]:
        chat = Chat(
            id=str(uuid.uuid4()),
            user_id=user_id,
            title=title.strip()[:256] or "New chat",
            model_name=model_name,
            checkpoint=checkpoint,
            system_prompt=system_prompt,
            generation_settings=json.dumps(generation_settings),
        )
        with self.session_factory.begin() as database:
            database.add(chat)
        return self._serialize_chat(chat)

    def chats(self, user_id: int, query: str | None = None) -> list[dict[str, Any]]:
        with self.session_factory() as database:
            statement = select(Chat).where(Chat.user_id == user_id).order_by(Chat.updated_at.desc())
            if query:
                statement = statement.where(Chat.title.ilike(f"%{query.strip()}%"))
            return [self._serialize_chat(item) for item in database.scalars(statement).all()]

    def chat(self, user_id: int, chat_id: str) -> dict[str, Any] | None:
        with self.session_factory() as database:
            chat = database.scalar(select(Chat).where(Chat.id == chat_id, Chat.user_id == user_id))
            if chat is None:
                return None
            data = self._serialize_chat(chat)
            messages = database.scalars(
                select(MessageRecord)
                .where(MessageRecord.chat_id == chat_id)
                .order_by(MessageRecord.created_at)
            ).all()
            data["messages"] = [self._serialize_message(item) for item in messages]
            return data

    def update_chat(
        self,
        user_id: int,
        chat_id: str,
        title: str | None = None,
        model_name: str | None = None,
        checkpoint: str | None = None,
        system_prompt: str | None = None,
        generation_settings: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        with self.session_factory.begin() as database:
            chat = database.scalar(select(Chat).where(Chat.id == chat_id, Chat.user_id == user_id))
            if chat is None:
                return None
            if title is not None:
                chat.title = title.strip()[:256] or chat.title
            if model_name is not None:
                chat.model_name = model_name
            if checkpoint is not None:
                chat.checkpoint = checkpoint
            if system_prompt is not None:
                chat.system_prompt = system_prompt
            if generation_settings is not None:
                chat.generation_settings = json.dumps(generation_settings)
            chat.updated_at = datetime.now(UTC).replace(tzinfo=None)
            database.flush()
            return self._serialize_chat(chat)

    def delete_chat(self, user_id: int, chat_id: str) -> bool:
        with self.session_factory.begin() as database:
            chat = database.scalar(select(Chat).where(Chat.id == chat_id, Chat.user_id == user_id))
            if chat is None:
                return False
            database.delete(chat)
            return True

    def add_message(
        self,
        user_id: int,
        chat_id: str,
        role: str,
        content: str,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
        generation_ms: int | None = None,
        tokens_per_second: float | None = None,
        model_id: str | None = None,
        checkpoint_id: str | None = None,
        generation_params: dict[str, Any] | None = None,
        model_config: dict[str, Any] | None = None,
        reasoning_trace: list[dict[str, Any]] | None = None,
        status: str = "completed",
    ) -> dict[str, Any] | None:
        with self.session_factory.begin() as database:
            chat = database.scalar(select(Chat).where(Chat.id == chat_id, Chat.user_id == user_id))
            if chat is None:
                return None
            message = MessageRecord(
                id=str(uuid.uuid4()),
                chat_id=chat_id,
                role=role,
                content=content,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                generation_ms=generation_ms,
                tokens_per_second=tokens_per_second,
                model_id=model_id,
                checkpoint_id=checkpoint_id,
                generation_params=(
                    json.dumps(generation_params) if generation_params is not None else None
                ),
                model_config_json=json.dumps(model_config) if model_config is not None else None,
                reasoning_trace=(
                    json.dumps(reasoning_trace) if reasoning_trace is not None else None
                ),
                status=status,
            )
            chat.updated_at = datetime.now(UTC).replace(tzinfo=None)
            database.add(message)
            database.flush()
            return self._serialize_message(message)

    def start_chat_generation(
        self,
        user_id: int,
        chat_id: str,
        content: str | None = None,
        action: str = "send",
        target_message_id: str | None = None,
        model_id: str | None = None,
        checkpoint_id: str | None = None,
        generation_params: dict[str, Any] | None = None,
        model_config: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any] | None, dict[str, Any], list[dict[str, Any]], str] | None:
        with self.session_factory.begin() as database:
            chat = database.scalar(select(Chat).where(Chat.id == chat_id, Chat.user_id == user_id))
            if chat is None:
                return None
            ordered = list(
                database.scalars(
                    select(MessageRecord)
                    .where(MessageRecord.chat_id == chat_id)
                    .order_by(MessageRecord.created_at)
                ).all()
            )
            resolved_model_id = model_id or chat.model_name or self.inference_model
            resolved_checkpoint_id = checkpoint_id or chat.checkpoint
            if resolved_model_id:
                chat.model_name = resolved_model_id
            if resolved_checkpoint_id:
                chat.checkpoint = resolved_checkpoint_id
            params_json = json.dumps(generation_params) if generation_params is not None else None
            config_json = json.dumps(model_config) if model_config is not None else None
            user_record: MessageRecord | None = None
            continuation_prefix = ""
            if action == "edit":
                if not target_message_id or not content or not content.strip():
                    raise ValueError("Edit requires target_message_id and content")
                index = next(
                    (i for i, item in enumerate(ordered) if item.id == target_message_id), None
                )
                if index is None or ordered[index].role != "user":
                    raise ValueError("User message not found")
                for trailing in ordered[index + 1 :]:
                    database.delete(trailing)
                target_user = ordered[index]
                target_user.content = content
                history_records = ordered[: index + 1]
                assistant = MessageRecord(
                    id=str(uuid.uuid4()),
                    chat_id=chat_id,
                    role="assistant",
                    content="",
                    model_id=resolved_model_id,
                    checkpoint_id=resolved_checkpoint_id,
                    generation_params=params_json,
                    model_config_json=config_json,
                    reasoning_trace="[]",
                    status="running",
                )
                database.add(assistant)
                user_record = target_user
            elif action == "regenerate":
                index = (
                    next(
                        (i for i, item in enumerate(ordered) if item.id == target_message_id), None
                    )
                    if target_message_id
                    else next(
                        (i for i in range(len(ordered) - 1, -1, -1) if ordered[i].role == "assistant"),
                        None,
                    )
                )
                if index is None or ordered[index].role != "assistant":
                    raise ValueError("Assistant message not found")
                for trailing in ordered[index:]:
                    database.delete(trailing)
                history_records = ordered[:index]
                if not any(item.role == "user" for item in history_records):
                    raise ValueError("No user prompt available to regenerate")
                assistant = MessageRecord(
                    id=str(uuid.uuid4()),
                    chat_id=chat_id,
                    role="assistant",
                    content="",
                    model_id=resolved_model_id,
                    checkpoint_id=resolved_checkpoint_id,
                    generation_params=params_json,
                    model_config_json=config_json,
                    reasoning_trace="[]",
                    status="running",
                )
                database.add(assistant)
            elif action == "continue":
                index = (
                    next(
                        (i for i, item in enumerate(ordered) if item.id == target_message_id), None
                    )
                    if target_message_id
                    else next(
                        (i for i in range(len(ordered) - 1, -1, -1) if ordered[i].role == "assistant"),
                        None,
                    )
                )
                if index is None or ordered[index].role != "assistant":
                    raise ValueError("Assistant message not found")
                assistant = ordered[index]
                continuation_prefix = assistant.content
                history_records = ordered[:index]
                assistant.model_id = resolved_model_id or assistant.model_id
                assistant.checkpoint_id = resolved_checkpoint_id or assistant.checkpoint_id
                assistant.generation_params = params_json or assistant.generation_params
                assistant.model_config_json = config_json or assistant.model_config_json
                assistant.status = "running"
            else:
                if not content or not content.strip():
                    raise ValueError("Message content is required")
                user = MessageRecord(
                    id=str(uuid.uuid4()), chat_id=chat_id, role="user", content=content
                )
                assistant = MessageRecord(
                    id=str(uuid.uuid4()),
                    chat_id=chat_id,
                    role="assistant",
                    content="",
                    model_id=resolved_model_id,
                    checkpoint_id=resolved_checkpoint_id,
                    generation_params=params_json,
                    model_config_json=config_json,
                    reasoning_trace="[]",
                    status="running",
                )
                if chat.title == "New chat":
                    chat.title = " ".join(content.split())[:80]
                database.add_all([user, assistant])
                history_records = [*ordered, user]
                user_record = user
            chat.updated_at = datetime.now(UTC).replace(tzinfo=None)
            database.flush()
            user_serialized = (
                self._serialize_message(user_record) if user_record is not None else None
            )
            return (
                user_serialized,
                self._serialize_message(assistant),
                [self._serialize_message(item) for item in history_records],
                continuation_prefix,
            )

    def update_message_content(
        self,
        user_id: int,
        chat_id: str,
        message_id: str | None,
        content: str,
        reasoning_trace: list[dict[str, Any]] | None = None,
        status: str | None = None,
    ) -> None:
        with self.session_factory.begin() as database:
            message = database.scalar(
                select(MessageRecord)
                .join(Chat, MessageRecord.chat_id == Chat.id)
                .where(Chat.id == chat_id, Chat.user_id == user_id, MessageRecord.id == message_id)
            )
            if message is None:
                raise ValueError("Message not found")
            message.content = content
            if reasoning_trace is not None:
                message.reasoning_trace = json.dumps(reasoning_trace)
            if status is not None:
                message.status = status

    def finalize_message(
        self,
        user_id: int,
        chat_id: str,
        message_id: str | None,
        content: str,
        status: str,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
        generation_ms: int | None = None,
        tokens_per_second: float | None = None,
        model_id: str | None = None,
        checkpoint_id: str | None = None,
        generation_params: dict[str, Any] | None = None,
        model_config: dict[str, Any] | None = None,
        reasoning_trace: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any] | None:
        with self.session_factory.begin() as database:
            message = database.scalar(
                select(MessageRecord)
                .join(Chat, MessageRecord.chat_id == Chat.id)
                .where(Chat.id == chat_id, Chat.user_id == user_id, MessageRecord.id == message_id)
            )
            if message is None:
                return None
            message.content = content
            message.status = status
            if input_tokens is not None:
                message.input_tokens = input_tokens
            if output_tokens is not None:
                message.output_tokens = output_tokens
            if generation_ms is not None:
                message.generation_ms = generation_ms
            if tokens_per_second is not None:
                message.tokens_per_second = tokens_per_second
            if model_id is not None:
                message.model_id = model_id
            if checkpoint_id is not None:
                message.checkpoint_id = checkpoint_id
            if generation_params is not None:
                message.generation_params = json.dumps(generation_params)
            if model_config is not None:
                message.model_config_json = json.dumps(model_config)
            if reasoning_trace is not None:
                message.reasoning_trace = json.dumps(reasoning_trace)
            database.flush()
            return self._serialize_message(message)

    def get_generation_job(self, user_id: int, request_id: str) -> dict[str, Any] | None:
        with self.session_factory() as database:
            job = database.scalar(
                select(GenerationJob).where(
                    GenerationJob.request_id == request_id, GenerationJob.user_id == user_id
                )
            )
            return None if job is None else self._serialize_job(job)

    def create_generation_job(
        self,
        request_id: str,
        user_id: int,
        chat_id: str,
        action: str = "send",
        model_id: str | None = None,
        checkpoint_id: str | None = None,
        queue_reason: str | None = None,
    ) -> tuple[dict[str, Any], bool]:
        with self.session_factory.begin() as database:
            existing = database.scalar(
                select(GenerationJob).where(GenerationJob.request_id == request_id)
            )
            if existing is not None:
                if existing.user_id != user_id:
                    raise ValueError("Duplicate request_id")
                return self._serialize_job(existing), False
            job = GenerationJob(
                request_id=request_id,
                user_id=user_id,
                chat_id=chat_id,
                action=action,
                status="queued",
                sequence_id=0,
                partial_text="",
                reasoning_events="[]",
                model_id=model_id,
                checkpoint_id=checkpoint_id,
                queue_reason=queue_reason,
            )
            database.add(job)
            database.flush()
            return self._serialize_job(job), True

    def update_generation_job(
        self,
        request_id: str,
        *,
        status: str | None = None,
        sequence_id: int | None = None,
        partial_text: str | None = None,
        user_message_id: str | None = None,
        assistant_message_id: str | None = None,
        reasoning_events: list[dict[str, Any]] | None = None,
        model_id: str | None = None,
        checkpoint_id: str | None = None,
        queue_reason: str | None = None,
        error_code: str | None = None,
        error_detail: str | None = None,
    ) -> None:
        with self.session_factory.begin() as database:
            job = database.scalar(
                select(GenerationJob).where(GenerationJob.request_id == request_id)
            )
            if job is None:
                return
            if status is not None:
                job.status = status
            if sequence_id is not None:
                job.sequence_id = sequence_id
            if partial_text is not None:
                job.partial_text = partial_text
            if user_message_id is not None:
                job.user_message_id = user_message_id
            if assistant_message_id is not None:
                job.assistant_message_id = assistant_message_id
            if reasoning_events is not None:
                job.reasoning_events = json.dumps(reasoning_events)
            if model_id is not None:
                job.model_id = model_id
            if checkpoint_id is not None:
                job.checkpoint_id = checkpoint_id
            if queue_reason is not None:
                job.queue_reason = queue_reason
            if error_code is not None:
                job.error_code = error_code
            if error_detail is not None:
                job.error_detail = error_detail
            job.updated_at = datetime.now(UTC).replace(tzinfo=None)

    def list_recoverable_jobs(self, user_id: int) -> list[dict[str, Any]]:
        with self.session_factory() as database:
            rows = database.scalars(
                select(GenerationJob)
                .where(
                    GenerationJob.user_id == user_id,
                    GenerationJob.status.in_(["queued", "running"]),
                )
                .order_by(GenerationJob.created_at)
            ).all()
            return [self._serialize_job(item) for item in rows]

    @staticmethod
    def _serialize_job(job: GenerationJob) -> dict[str, Any]:
        return {
            "request_id": job.request_id,
            "user_id": job.user_id,
            "chat_id": job.chat_id,
            "user_message_id": job.user_message_id,
            "assistant_message_id": job.assistant_message_id,
            "action": job.action,
            "status": job.status,
            "sequence_id": job.sequence_id,
            "partial_text": job.partial_text,
            "reasoning_events": json.loads(job.reasoning_events or "[]"),
            "model_id": job.model_id,
            "checkpoint_id": job.checkpoint_id,
            "queue_reason": job.queue_reason,
            "error_code": job.error_code,
            "error_detail": job.error_detail,
            "created_at": job.created_at.isoformat(),
            "updated_at": job.updated_at.isoformat(),
        }

    @staticmethod
    def _parameter_count(config: dict[str, Any]) -> int | None:
        values = ("vocab_size", "hidden_size", "num_hidden_layers", "intermediate_size")
        if not all(isinstance(config.get(key), int) for key in values):
            return None
        return int(
            config["vocab_size"] * config["hidden_size"]
            + config["num_hidden_layers"]
            * (
                4 * config["hidden_size"] ** 2
                + 3 * config["hidden_size"] * config["intermediate_size"]
            )
        )

    @staticmethod
    def _serialize_chat(chat: Chat) -> dict[str, Any]:
        return {
            "id": chat.id,
            "title": chat.title,
            "model_name": chat.model_name,
            "checkpoint": chat.checkpoint,
            "system_prompt": chat.system_prompt,
            "generation_settings": json.loads(chat.generation_settings),
            "created_at": chat.created_at.isoformat(),
            "updated_at": chat.updated_at.isoformat(),
        }

    @staticmethod
    def _serialize_message(message: MessageRecord) -> dict[str, Any]:
        return {
            "id": message.id,
            "role": message.role,
            "content": message.content,
            "input_tokens": message.input_tokens,
            "output_tokens": message.output_tokens,
            "generation_ms": message.generation_ms,
            "tokens_per_second": message.tokens_per_second,
            "model_id": message.model_id,
            "checkpoint_id": message.checkpoint_id,
            "generation_params": (
                json.loads(message.generation_params) if message.generation_params else None
            ),
            "model_config": (
                json.loads(message.model_config_json) if message.model_config_json else None
            ),
            "reasoning_trace": (
                json.loads(message.reasoning_trace) if message.reasoning_trace else []
            ),
            "status": message.status,
            "created_at": message.created_at.isoformat(),
        }

    def _max_steps(self, active: dict[str, Any] | None) -> int | None:
        if active is None or active.get("config_path") is None:
            return None
        try:
            config = load_experiment_config(self.root / active["config_path"])
            return config.train.max_steps
        except ConfigError:
            return None

    def _pipeline(
        self,
        processes: list[dict[str, Any]],
        checkpoints: list[dict[str, Any]],
        pipeline: dict[str, Any] | None = None,
    ) -> list[dict[str, str]]:
        has_data = any((self.root / "data").glob("**/manifest.json"))
        has_tokenizer = (self.root / "data" / "tokenizer" / "tokenizer.json").is_file()
        if pipeline is not None:
            stages = pipeline["stages"]
            eval_statuses = [
                stages.get(name, {}).get("status", "pending")
                for name in ("eval_pretrain", "eval_sft", "eval_final")
            ]
            eval_status = (
                "running"
                if "running" in eval_statuses
                else (
                    "failed"
                    if "failed" in eval_statuses
                    else (
                        "completed"
                        if stages.get("eval_final", {}).get("status") == "completed"
                        else "pending"
                    )
                )
            )
            return [
                {"id": "dataset", "status": "completed" if has_data else "pending"},
                {"id": "tokenizer", "status": "completed" if has_tokenizer else "pending"},
                {
                    "id": "pretraining",
                    "status": stages.get("pretrain", {}).get("status", "pending"),
                },
                {"id": "sft", "status": stages.get("sft", {}).get("status", "pending")},
                {"id": "dpo", "status": stages.get("dpo", {}).get("status", "pending")},
                {"id": "evaluation", "status": eval_status},
                {
                    "id": "model",
                    "status": stages.get("export_release", {}).get("status", "pending"),
                },
            ]
        active_kinds = {
            item["kind"] for item in processes if item["status"] in {"running", "pausing"}
        }
        has_pretrain = any(item["name"] not in {"debug_sft", "debug_dpo"} for item in checkpoints)
        has_sft = any("sft" in item["name"] for item in checkpoints)
        has_dpo = any("dpo" in item["name"] for item in checkpoints)
        has_evaluation = any((self.root / "experiments").glob("**/evaluation.json"))
        states = {
            "dataset": (
                "running" if "dataset" in active_kinds else "completed" if has_data else "pending"
            ),
            "tokenizer": (
                "running"
                if "tokenizer" in active_kinds
                else "completed"
                if has_tokenizer
                else "pending"
            ),
            "pretraining": (
                "running"
                if "pretrain" in active_kinds
                else "completed"
                if has_pretrain
                else "pending"
            ),
            "sft": "running" if "sft" in active_kinds else "completed" if has_sft else "pending",
            "dpo": "running" if "dpo" in active_kinds else "completed" if has_dpo else "pending",
            "evaluation": (
                "running"
                if "evaluation" in active_kinds
                else "completed"
                if has_evaluation
                else "pending"
            ),
            "model": "completed" if checkpoints else "pending",
        }
        return [{"id": key, "status": value} for key, value in states.items()]

    def _system_snapshot(self) -> dict[str, Any]:
        if time.monotonic() - self.storage_metrics_at >= STORAGE_REFRESH_SECONDS:
            self.storage_metrics = storage_snapshot(self.root)
            self.storage_metrics_at = time.monotonic()
        return system_snapshot(self.root, self.storage_metrics)

    @staticmethod
    def _redact(message: str) -> str:
        return SECRET_PATTERN.sub(r"\1\2[redacted]", message)

    @staticmethod
    def _prune_system_metrics(database) -> None:
        ids = database.scalars(
            select(SystemMetric.id).order_by(SystemMetric.recorded_at.desc()).offset(5000)
        ).all()
        if ids:
            database.query(SystemMetric).filter(SystemMetric.id.in_(ids)).delete(
                synchronize_session=False
            )
