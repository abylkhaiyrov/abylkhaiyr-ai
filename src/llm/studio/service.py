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
from llm.studio.database import Base, create_sqlite_engine, make_session_factory
from llm.studio.models import Chat, MessageRecord, ProcessEvent, SystemMetric
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
        database_path = self.root / "studio" / "studio.db"
        self.engine = create_sqlite_engine(database_path)
        Base.metadata.create_all(self.engine)
        self.session_factory = make_session_factory(self.engine)
        self.auth = AuthenticationService(self.session_factory)
        self.auth.initialize()
        self.processes = ProcessManager(self.session_factory, self.root)
        self.runtime = RuntimeState(self.root)
        self.storage_metrics = storage_snapshot(self.root)
        self.storage_metrics_at = time.monotonic()

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
        return {
            "system_status": "busy" if active is not None else "ready",
            "active_model": None,
            "inference_model": self.inference_model,
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

    def update_chat(self, user_id: int, chat_id: str, title: str | None) -> dict[str, Any] | None:
        with self.session_factory.begin() as database:
            chat = database.scalar(select(Chat).where(Chat.id == chat_id, Chat.user_id == user_id))
            if chat is None:
                return None
            if title is not None:
                chat.title = title.strip()[:256] or chat.title
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
            )
            chat.updated_at = datetime.now(UTC).replace(tzinfo=None)
            database.add(message)
            database.flush()
            return self._serialize_message(message)

    def start_chat_generation(
        self, user_id: int, chat_id: str, content: str
    ) -> tuple[dict[str, Any], dict[str, Any]] | None:
        with self.session_factory.begin() as database:
            chat = database.scalar(select(Chat).where(Chat.id == chat_id, Chat.user_id == user_id))
            if chat is None:
                return None
            user = MessageRecord(
                id=str(uuid.uuid4()), chat_id=chat_id, role="user", content=content
            )
            assistant = MessageRecord(
                id=str(uuid.uuid4()), chat_id=chat_id, role="assistant", content=""
            )
            if chat.title == "New chat":
                chat.title = " ".join(content.split())[:80]
            if chat.model_name is None:
                chat.model_name = self.inference_model
            chat.updated_at = datetime.now(UTC).replace(tzinfo=None)
            database.add_all([user, assistant])
            database.flush()
            return self._serialize_message(user), self._serialize_message(assistant)

    def update_message_content(
        self, user_id: int, chat_id: str, message_id: str | None, content: str
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
