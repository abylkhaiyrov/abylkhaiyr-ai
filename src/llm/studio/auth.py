from __future__ import annotations

import hashlib
import secrets
import threading
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from typing import Any

from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerificationError, VerifyMismatchError
from sqlalchemy import delete, func, select
from sqlalchemy.exc import IntegrityError

from llm.studio.accounts import (
    AuthenticationError,
    normalize_email,
    normalize_name,
    normalize_username,
    validate_password,
)
from llm.studio.models import AuditLog, SessionRecord, User

SESSION_TTL = timedelta(hours=12)
ACTIVITY_TOUCH_INTERVAL = timedelta(seconds=60)
DEFAULT_USERNAME = "admin"
DEFAULT_PASSWORD = "admin"

__all__ = ["DEFAULT_USERNAME", "AuthenticationError", "AuthenticationService"]

RevocationListener = Callable[[int, set[str] | None], None]


def utcnow() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


class AuthenticationService:
    def __init__(self, session_factory) -> None:
        self.session_factory = session_factory
        self.hasher = PasswordHasher()
        self.listeners: list[RevocationListener] = []
        self._register_lock = threading.Lock()

    def initialize(self) -> None:
        with self.session_factory.begin() as database:
            if database.scalar(select(User).where(User.username == DEFAULT_USERNAME)) is None:
                database.add(
                    User(
                        username=DEFAULT_USERNAME,
                        first_name="Administrator",
                        password_hash=self.hasher.hash(DEFAULT_PASSWORD),
                        role="ADMIN",
                        requires_password_change=True,
                    )
                )

    def register(
        self,
        first_name: str,
        last_name: str,
        email: str,
        username: str,
        password: str,
        user_agent: str | None,
    ) -> tuple[str, str, User]:
        normalized_username = normalize_username(username)
        normalized_email = normalize_email(email)
        first = normalize_name(first_name, "invalid_first_name")
        last = normalize_name(last_name, "invalid_last_name")
        validate_password(password, normalized_username)
        password_hash = self.hasher.hash(password)
        with self._register_lock:
            try:
                with self.session_factory.begin() as database:
                    if database.scalar(select(User.id).where(User.email == normalized_email)):
                        raise AuthenticationError("email_taken")
                    if database.scalar(select(User.id).where(User.username == normalized_username)):
                        raise AuthenticationError("username_taken")
                    user = User(
                        first_name=first,
                        last_name=last,
                        email=normalized_email,
                        username=normalized_username,
                        password_hash=password_hash,
                        role="USER",
                    )
                    database.add(user)
                    database.flush()
                    database.add(
                        AuditLog(user_id=user.id, category="authentication", action="registered")
                    )
                    token, csrf = self._open_session(database, user, user_agent)
                    database.expunge(user)
            except IntegrityError as error:
                raise AuthenticationError("account_conflict") from error
        return token, csrf, user

    def authenticate(
        self,
        identifier: str,
        password: str,
        user_agent: str | None,
        previous_token: str | None = None,
    ) -> tuple[str, str, User]:
        key = identifier.strip().lower()
        with self.session_factory.begin() as database:
            column = User.email if "@" in key else User.username
            user = database.scalar(select(User).where(column == key))
            if user is None or not self._verify(password, user.password_hash):
                raise AuthenticationError("invalid_credentials")
            if not user.is_active:
                raise AuthenticationError("account_disabled")
            if previous_token:
                database.execute(
                    delete(SessionRecord).where(
                        SessionRecord.token_hash == self.hash_token(previous_token)
                    )
                )
            user.last_login_at = utcnow()
            user.last_active_at = user.last_login_at
            token, csrf = self._open_session(database, user, user_agent)
            database.add(AuditLog(user_id=user.id, category="authentication", action="login"))
            database.flush()
            database.expunge(user)
            return token, csrf, user

    def get_user(self, token: str | None) -> tuple[User, str] | None:
        if not token:
            return None
        now = utcnow()
        with self.session_factory.begin() as database:
            record = database.scalar(
                select(SessionRecord).where(
                    SessionRecord.token_hash == self.hash_token(token),
                    SessionRecord.expires_at > now,
                )
            )
            if record is None:
                return None
            user = database.get(User, record.user_id)
            if user is None or not user.is_active:
                return None
            if record.last_seen_at is None or now - record.last_seen_at > ACTIVITY_TOUCH_INTERVAL:
                record.last_seen_at = now
                user.last_active_at = now
            database.flush()
            database.expunge(user)
            return user, record.csrf_token

    def logout(self, token: str | None) -> None:
        if not token:
            return
        token_hash = self.hash_token(token)
        with self.session_factory.begin() as database:
            record = database.scalar(
                select(SessionRecord).where(SessionRecord.token_hash == token_hash)
            )
            if record is None:
                return
            user_id = record.user_id
            database.delete(record)
        self._notify(user_id, {token_hash})

    def list_sessions(self, user_id: int, current_token: str | None) -> list[dict[str, Any]]:
        current = self.hash_token(current_token) if current_token else None
        with self.session_factory() as database:
            rows = database.scalars(
                select(SessionRecord)
                .where(SessionRecord.user_id == user_id, SessionRecord.expires_at > utcnow())
                .order_by(SessionRecord.created_at.desc())
            ).all()
            return [
                {
                    "id": row.id,
                    "created_at": row.created_at.isoformat(),
                    "last_seen_at": (
                        None if row.last_seen_at is None else row.last_seen_at.isoformat()
                    ),
                    "user_agent": row.user_agent,
                    "current": row.token_hash == current,
                }
                for row in rows
            ]

    def revoke_session(self, user_id: int, session_id: int) -> bool:
        with self.session_factory.begin() as database:
            record = database.scalar(
                select(SessionRecord).where(
                    SessionRecord.id == session_id, SessionRecord.user_id == user_id
                )
            )
            if record is None:
                return False
            token_hash = record.token_hash
            database.delete(record)
        self._notify(user_id, {token_hash})
        return True

    def revoke_sessions(self, user_id: int, keep_token: str | None = None) -> int:
        keep = self.hash_token(keep_token) if keep_token else None
        with self.session_factory.begin() as database:
            rows = database.scalars(
                select(SessionRecord).where(SessionRecord.user_id == user_id)
            ).all()
            revoked = {row.token_hash for row in rows if row.token_hash != keep}
            for row in rows:
                if row.token_hash in revoked:
                    database.delete(row)
        if revoked:
            self._notify(user_id, revoked)
        return len(revoked)

    def update_profile(
        self, user_id: int, first_name: str, last_name: str, email: str, username: str
    ) -> User:
        normalized_username = normalize_username(username)
        normalized_email = normalize_email(email)
        first = normalize_name(first_name, "invalid_first_name")
        last = normalize_name(last_name, "invalid_last_name")
        try:
            with self.session_factory.begin() as database:
                user = database.get(User, user_id)
                if user is None:
                    raise AuthenticationError("invalid_credentials")
                self._ensure_unique(database, user.id, normalized_email, normalized_username)
                user.first_name, user.last_name = first, last
                user.email, user.username = normalized_email, normalized_username
                database.add(
                    AuditLog(user_id=user.id, category="account", action="profile_updated")
                )
                database.flush()
                database.expunge(user)
        except IntegrityError as error:
            raise AuthenticationError("account_conflict") from error
        return user

    def change_password(self, user_id: int, current_password: str, new_password: str) -> None:
        with self.session_factory.begin() as database:
            user = database.get(User, user_id)
            if user is None or not self._verify(current_password, user.password_hash):
                raise AuthenticationError("invalid_password")
            validate_password(new_password, user.username)
            if self._verify(new_password, user.password_hash):
                raise AuthenticationError("weak_password")
            user.password_hash = self.hasher.hash(new_password)
            user.requires_password_change = False
            database.add(
                AuditLog(user_id=user.id, category="authentication", action="password_changed")
            )
        self.revoke_sessions(user_id)

    def set_password(self, user: User, password: str, temporary: bool) -> None:
        validate_password(password, user.username)
        user.password_hash = self.hasher.hash(password)
        user.requires_password_change = temporary

    def count_active_admins(self, database, excluding: int | None = None) -> int:
        statement = (
            select(func.count())
            .select_from(User)
            .where(User.role == "ADMIN", User.is_active.is_(True))
        )
        if excluding is not None:
            statement = statement.where(User.id != excluding)
        return int(database.scalar(statement) or 0)

    def notify_revoked(self, user_id: int, hashes: set[str] | None = None) -> None:
        self._notify(user_id, hashes)

    def _notify(self, user_id: int, hashes: set[str] | None) -> None:
        for listener in tuple(self.listeners):
            listener(user_id, hashes)

    def _open_session(self, database, user: User, user_agent: str | None) -> tuple[str, str]:
        token = secrets.token_urlsafe(48)
        csrf_token = secrets.token_urlsafe(32)
        database.add(
            SessionRecord(
                token_hash=self.hash_token(token),
                csrf_token=csrf_token,
                user_id=user.id,
                expires_at=utcnow() + SESSION_TTL,
                last_seen_at=utcnow(),
                user_agent=(user_agent or "")[:256] or None,
            )
        )
        return token, csrf_token

    @staticmethod
    def _ensure_unique(database, user_id: int, email: str, username: str) -> None:
        if database.scalar(select(User.id).where(User.email == email, User.id != user_id)):
            raise AuthenticationError("email_taken")
        if database.scalar(select(User.id).where(User.username == username, User.id != user_id)):
            raise AuthenticationError("username_taken")

    def _verify(self, password: str, password_hash: str) -> bool:
        try:
            return bool(self.hasher.verify(password_hash, password))
        except (InvalidHashError, VerificationError, VerifyMismatchError):
            return False

    @staticmethod
    def hash_token(token: str) -> str:
        return hashlib.sha256(token.encode("utf-8")).hexdigest()
