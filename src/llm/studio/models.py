from __future__ import annotations

import uuid as uuid_module
from datetime import UTC, datetime

from sqlalchemy import Boolean, DateTime, ForeignKey, Index, Integer, String, Text, text
from sqlalchemy.orm import Mapped, mapped_column

from llm.studio.database import Base


def utcnow() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    username: Mapped[str] = mapped_column(String(128), unique=True, index=True)
    uuid: Mapped[str] = mapped_column(
        String(36), unique=True, index=True, default=lambda: str(uuid_module.uuid4())
    )
    first_name: Mapped[str] = mapped_column(String(64), default="", server_default="")
    last_name: Mapped[str] = mapped_column(String(64), default="", server_default="")
    email: Mapped[str | None] = mapped_column(String(254), unique=True, index=True)
    password_hash: Mapped[str] = mapped_column(String(512))
    role: Mapped[str] = mapped_column(String(16), default="USER", server_default="USER", index=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, server_default="1")
    requires_password_change: Mapped[bool] = mapped_column(Boolean, default=False)
    last_login_at: Mapped[datetime | None] = mapped_column(DateTime)
    last_active_at: Mapped[datetime | None] = mapped_column(DateTime)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, onupdate=utcnow)


class SessionRecord(Base):
    __tablename__ = "sessions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    token_hash: Mapped[str] = mapped_column(String(128), unique=True, index=True)
    csrf_token: Mapped[str] = mapped_column(String(128))
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    expires_at: Mapped[datetime] = mapped_column(DateTime, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    last_seen_at: Mapped[datetime | None] = mapped_column(DateTime)
    user_agent: Mapped[str | None] = mapped_column(String(256))


class Chat(Base):
    __tablename__ = "chats"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    title: Mapped[str] = mapped_column(String(256))
    model_name: Mapped[str | None] = mapped_column(String(256))
    checkpoint: Mapped[str | None] = mapped_column(String(1024))
    system_prompt: Mapped[str] = mapped_column(Text, default="")
    generation_settings: Mapped[str] = mapped_column(Text, default="{}")
    pinned: Mapped[bool] = mapped_column(Boolean, default=False, server_default="0")
    archived: Mapped[bool] = mapped_column(Boolean, default=False, server_default="0", index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=utcnow, onupdate=utcnow, index=True
    )


class MessageRecord(Base):
    __tablename__ = "messages"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    chat_id: Mapped[str] = mapped_column(ForeignKey("chats.id", ondelete="CASCADE"), index=True)
    role: Mapped[str] = mapped_column(String(16))
    content: Mapped[str] = mapped_column(Text)
    input_tokens: Mapped[int | None] = mapped_column(Integer)
    output_tokens: Mapped[int | None] = mapped_column(Integer)
    generation_ms: Mapped[int | None] = mapped_column(Integer)
    tokens_per_second: Mapped[float | None] = mapped_column()
    model_id: Mapped[str | None] = mapped_column(String(256))
    checkpoint_id: Mapped[str | None] = mapped_column(String(1024))
    generation_params: Mapped[str | None] = mapped_column(Text)
    model_config_json: Mapped[str | None] = mapped_column(Text)
    reasoning_trace: Mapped[str | None] = mapped_column(Text)
    variants: Mapped[str | None] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String(32), default="completed")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, index=True)


class GenerationJob(Base):
    __tablename__ = "generation_jobs"

    request_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    chat_id: Mapped[str] = mapped_column(ForeignKey("chats.id", ondelete="CASCADE"), index=True)
    user_message_id: Mapped[str | None] = mapped_column(String(36))
    assistant_message_id: Mapped[str | None] = mapped_column(String(36))
    action: Mapped[str] = mapped_column(String(32), default="send")
    status: Mapped[str] = mapped_column(String(32), index=True, default="queued")
    sequence_id: Mapped[int] = mapped_column(Integer, default=0)
    partial_text: Mapped[str] = mapped_column(Text, default="")
    reasoning_events: Mapped[str] = mapped_column(Text, default="[]")
    model_id: Mapped[str | None] = mapped_column(String(256))
    checkpoint_id: Mapped[str | None] = mapped_column(String(1024))
    queue_reason: Mapped[str | None] = mapped_column(String(128))
    started_at: Mapped[datetime | None] = mapped_column(DateTime)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime)
    error_code: Mapped[str | None] = mapped_column(String(64))
    error_detail: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, index=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, onupdate=utcnow)


class Notification(Base):
    __tablename__ = "notifications"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    uuid: Mapped[str] = mapped_column(
        String(36), unique=True, index=True, default=lambda: str(uuid_module.uuid4())
    )
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"))
    type: Mapped[str] = mapped_column(String(32))
    title: Mapped[str] = mapped_column(String(128))
    message: Mapped[str] = mapped_column(Text, default="", server_default="")
    chat_id: Mapped[str | None] = mapped_column(String(36))
    generation_id: Mapped[str | None] = mapped_column(String(64))
    is_read: Mapped[bool] = mapped_column(Boolean, default=False, server_default="0")
    read_at: Mapped[datetime | None] = mapped_column(DateTime)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)


class ManagedProcess(Base):
    __tablename__ = "processes"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    pid: Mapped[int | None] = mapped_column(Integer, unique=True)
    pid_started_at: Mapped[float | None] = mapped_column()
    kind: Mapped[str] = mapped_column(String(32), index=True)
    status: Mapped[str] = mapped_column(String(32), index=True)
    command: Mapped[str] = mapped_column(Text)
    config_path: Mapped[str | None] = mapped_column(String(1024))
    output_dir: Mapped[str | None] = mapped_column(String(1024))
    started_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime)
    exit_code: Mapped[int | None] = mapped_column(Integer)
    error_reason: Mapped[str | None] = mapped_column(Text)


class ProcessEvent(Base):
    __tablename__ = "process_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    process_id: Mapped[str] = mapped_column(
        ForeignKey("processes.id", ondelete="CASCADE"), index=True
    )
    level: Mapped[str] = mapped_column(String(16))
    message: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, index=True)


class SystemMetric(Base):
    __tablename__ = "system_metrics"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    recorded_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, index=True)
    payload: Mapped[str] = mapped_column(Text)


class AuditLog(Base):
    __tablename__ = "audit_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int | None] = mapped_column(
        ForeignKey("users.id", ondelete="SET NULL"), index=True
    )
    category: Mapped[str] = mapped_column(String(32), index=True)
    action: Mapped[str] = mapped_column(String(64))
    details: Mapped[str] = mapped_column(Text, default="{}")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, index=True)


class ApplicationSetting(Base):
    __tablename__ = "application_settings"

    key: Mapped[str] = mapped_column(String(128), primary_key=True)
    value: Mapped[str] = mapped_column(Text)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, onupdate=utcnow)


Index(
    "ix_notifications_user_read_created",
    Notification.user_id,
    Notification.is_read,
    Notification.created_at,
)
Index(
    "uq_notifications_user_generation",
    Notification.user_id,
    Notification.generation_id,
    unique=True,
    sqlite_where=text("generation_id IS NOT NULL"),
)
Index("ix_messages_chat_created", MessageRecord.chat_id, MessageRecord.created_at)
Index("ix_process_events_process_created", ProcessEvent.process_id, ProcessEvent.created_at)
