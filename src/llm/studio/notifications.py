from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import delete, func, select, update
from sqlalchemy.exc import IntegrityError

from llm.studio.models import Notification

TYPES = ("GENERATION_COMPLETED", "GENERATION_FAILED", "GENERATION_CANCELLED", "SYSTEM", "ACCOUNT")
MAX_PAGE = 100
Publisher = Callable[[int, dict[str, Any]], None]


def serialize(item: Notification) -> dict[str, Any]:
    return {
        "id": item.uuid,
        "type": item.type,
        "title": item.title,
        "message": item.message,
        "chat_id": item.chat_id,
        "generation_id": item.generation_id,
        "is_read": item.is_read,
        "read_at": None if item.read_at is None else item.read_at.isoformat(),
        "created_at": item.created_at.isoformat(),
    }


class NotificationService:
    def __init__(self, session_factory) -> None:
        self.session_factory = session_factory
        self.publishers: list[Publisher] = []

    def create(
        self,
        user_id: int,
        kind: str,
        title: str,
        message: str = "",
        chat_id: str | None = None,
        generation_id: str | None = None,
    ) -> dict[str, Any] | None:
        if kind not in TYPES:
            raise ValueError(f"unknown notification type: {kind}")
        try:
            with self.session_factory.begin() as database:
                item = Notification(
                    user_id=user_id,
                    type=kind,
                    title=title[:128],
                    message=message[:2000],
                    chat_id=chat_id,
                    generation_id=generation_id,
                )
                database.add(item)
                database.flush()
                payload = serialize(item)
        except IntegrityError:
            return None
        self._emit(user_id, {"type": "notification.created", "notification": payload})
        self._emit_count(user_id)
        return payload

    def list(
        self,
        user_id: int,
        unread_only: bool,
        kind: str | None,
        limit: int,
        offset: int,
        oldest_first: bool,
    ) -> dict[str, Any]:
        statement = select(Notification).where(Notification.user_id == user_id)
        if unread_only:
            statement = statement.where(Notification.is_read.is_(False))
        if kind in TYPES:
            statement = statement.where(Notification.type == kind)
        order = Notification.created_at.asc() if oldest_first else Notification.created_at.desc()
        with self.session_factory() as database:
            total = database.scalar(select(func.count()).select_from(statement.subquery())) or 0
            rows = database.scalars(
                statement.order_by(order, Notification.id.desc())
                .limit(min(max(limit, 1), MAX_PAGE))
                .offset(max(offset, 0))
            ).all()
            return {
                "items": [serialize(row) for row in rows],
                "total": total,
                "unread": self._unread(database, user_id),
            }

    def unread_count(self, user_id: int) -> int:
        with self.session_factory() as database:
            return self._unread(database, user_id)

    def get(self, user_id: int, uuid: str) -> dict[str, Any] | None:
        with self.session_factory() as database:
            item = database.scalar(
                select(Notification).where(
                    Notification.uuid == uuid, Notification.user_id == user_id
                )
            )
            return None if item is None else serialize(item)

    def mark_read(self, user_id: int, uuid: str) -> dict[str, Any] | None:
        with self.session_factory.begin() as database:
            item = database.scalar(
                select(Notification).where(
                    Notification.uuid == uuid, Notification.user_id == user_id
                )
            )
            if item is None:
                return None
            if not item.is_read:
                item.is_read, item.read_at = True, datetime.now(UTC).replace(tzinfo=None)
            payload = serialize(item)
        self._emit(user_id, {"type": "notification.read", "ids": [uuid]})
        self._emit_count(user_id)
        return payload

    def mark_all_read(self, user_id: int) -> int:
        with self.session_factory.begin() as database:
            ids = list(
                database.scalars(
                    select(Notification.uuid).where(
                        Notification.user_id == user_id, Notification.is_read.is_(False)
                    )
                )
            )
            database.execute(
                update(Notification)
                .where(Notification.user_id == user_id, Notification.is_read.is_(False))
                .values(is_read=True, read_at=datetime.now(UTC).replace(tzinfo=None))
            )
        if ids:
            self._emit(user_id, {"type": "notification.read", "ids": ids})
            self._emit_count(user_id)
        return len(ids)

    def delete(self, user_id: int, uuid: str) -> bool:
        with self.session_factory.begin() as database:
            removed = database.execute(
                delete(Notification).where(
                    Notification.uuid == uuid, Notification.user_id == user_id
                )
            ).rowcount
        if removed:
            self._emit(user_id, {"type": "notification.deleted", "ids": [uuid]})
            self._emit_count(user_id)
        return bool(removed)

    @staticmethod
    def _unread(database, user_id: int) -> int:
        return int(
            database.scalar(
                select(func.count())
                .select_from(Notification)
                .where(Notification.user_id == user_id, Notification.is_read.is_(False))
            )
            or 0
        )

    def _emit_count(self, user_id: int) -> None:
        self._emit(
            user_id, {"type": "notification.count.updated", "unread": self.unread_count(user_id)}
        )

    def _emit(self, user_id: int, event: dict[str, Any]) -> None:
        for publisher in tuple(self.publishers):
            publisher(user_id, event)
