from __future__ import annotations

import json
import secrets
from typing import Any

from sqlalchemy import func, or_, select
from sqlalchemy.exc import IntegrityError

from llm.studio.accounts import (
    ROLES,
    AuthenticationError,
    normalize_email,
    normalize_name,
    normalize_username,
    public_user,
)
from llm.studio.auth import AuthenticationService
from llm.studio.models import AuditLog, User
from llm.studio.notifications import NotificationService

MAX_PAGE = 100
TEMPORARY_PASSWORD_BYTES = 12


class UserDirectory:
    def __init__(
        self, session_factory, auth: AuthenticationService, notifications: NotificationService
    ) -> None:
        self.session_factory = session_factory
        self.auth = auth
        self.notifications = notifications

    def search(
        self, query: str | None, role: str | None, active: bool | None, limit: int, offset: int
    ) -> dict[str, Any]:
        statement = select(User)
        if query and query.strip():
            pattern = f"%{query.strip().lower()}%"
            statement = statement.where(
                or_(
                    func.lower(User.username).like(pattern),
                    func.lower(func.coalesce(User.email, "")).like(pattern),
                    func.lower(User.first_name).like(pattern),
                    func.lower(User.last_name).like(pattern),
                )
            )
        if role in ROLES:
            statement = statement.where(User.role == role)
        if active is not None:
            statement = statement.where(User.is_active.is_(active))
        with self.session_factory() as database:
            total = database.scalar(select(func.count()).select_from(statement.subquery())) or 0
            rows = database.scalars(
                statement.order_by(User.created_at.desc(), User.id.desc())
                .limit(min(max(limit, 1), MAX_PAGE))
                .offset(max(offset, 0))
            ).all()
            return {"items": [public_user(row) for row in rows], "total": total}

    def get(self, uuid: str) -> dict[str, Any] | None:
        with self.session_factory() as database:
            user = database.scalar(select(User).where(User.uuid == uuid))
            return None if user is None else public_user(user)

    def update(self, admin_id: int, uuid: str, changes: dict[str, Any]) -> dict[str, Any] | None:
        try:
            with self.session_factory.begin() as database:
                user = database.scalar(select(User).where(User.uuid == uuid))
                if user is None:
                    return None
                self._apply(database, user, changes)
                database.add(self._audit(admin_id, "user_updated", user, sorted(changes)))
                database.flush()
                result = public_user(user)
                revoked = not user.is_active
        except IntegrityError as error:
            raise AuthenticationError("account_conflict") from error
        if revoked:
            self.auth.revoke_sessions(user.id)
        return result

    def reset_password(self, admin_id: int, uuid: str) -> dict[str, Any] | None:
        temporary = secrets.token_urlsafe(TEMPORARY_PASSWORD_BYTES)
        with self.session_factory.begin() as database:
            user = database.scalar(select(User).where(User.uuid == uuid))
            if user is None:
                return None
            self.auth.set_password(user, temporary, temporary=True)
            database.add(self._audit(admin_id, "password_reset", user, []))
            target = user.id
        self.auth.revoke_sessions(target)
        self.notifications.create(target, "ACCOUNT", "password_reset")
        return {"temporary_password": temporary}

    def revoke_sessions(self, admin_id: int, uuid: str) -> int | None:
        with self.session_factory.begin() as database:
            user = database.scalar(select(User).where(User.uuid == uuid))
            if user is None:
                return None
            database.add(self._audit(admin_id, "sessions_revoked", user, []))
            target = user.id
        return self.auth.revoke_sessions(target)

    def _apply(self, database, user: User, changes: dict[str, Any]) -> None:
        if "first_name" in changes:
            user.first_name = normalize_name(changes["first_name"], "invalid_first_name")
        if "last_name" in changes:
            user.last_name = normalize_name(changes["last_name"], "invalid_last_name")
        if "username" in changes:
            user.username = normalize_username(changes["username"])
        if "email" in changes:
            user.email = normalize_email(changes["email"])
        new_role = changes.get("role", user.role)
        new_active = changes.get("is_active", user.is_active)
        if new_role not in ROLES:
            raise AuthenticationError("invalid_role")
        loses_admin = (
            user.role == "ADMIN" and user.is_active and (new_role != "ADMIN" or not new_active)
        )
        if loses_admin and self.auth.count_active_admins(database, excluding=user.id) == 0:
            raise AuthenticationError("last_admin")
        user.role, user.is_active = new_role, new_active

    @staticmethod
    def _audit(admin_id: int, action: str, user: User, fields: list[str]) -> AuditLog:
        return AuditLog(
            user_id=admin_id,
            category="admin",
            action=action,
            details=json.dumps({"target": user.uuid, "username": user.username, "fields": fields}),
        )
