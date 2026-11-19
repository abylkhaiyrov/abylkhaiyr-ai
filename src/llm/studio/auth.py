from __future__ import annotations

import hashlib
import secrets
from datetime import UTC, datetime, timedelta

from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerificationError, VerifyMismatchError
from sqlalchemy import delete, select

from llm.studio.models import AuditLog, SessionRecord, User

SESSION_TTL = timedelta(hours=12)
DEFAULT_USERNAME = "admin"
DEFAULT_PASSWORD = "admin"


class AuthenticationError(Exception):
    pass


class AuthenticationService:
    def __init__(self, session_factory) -> None:
        self.session_factory = session_factory
        self.hasher = PasswordHasher()

    def initialize(self) -> None:
        with self.session_factory.begin() as database:
            if database.scalar(select(User).where(User.username == DEFAULT_USERNAME)) is None:
                database.add(
                    User(
                        username=DEFAULT_USERNAME,
                        password_hash=self.hasher.hash(DEFAULT_PASSWORD),
                        requires_password_change=True,
                    )
                )

    def authenticate(self, username: str, password: str) -> tuple[str, str, User]:
        with self.session_factory.begin() as database:
            user = database.scalar(select(User).where(User.username == username))
            if user is None or not self._verify(password, user.password_hash):
                raise AuthenticationError("Invalid login or password")
            token = secrets.token_urlsafe(48)
            csrf_token = secrets.token_urlsafe(32)
            database.add(
                SessionRecord(
                    token_hash=self._hash_token(token),
                    csrf_token=csrf_token,
                    user_id=user.id,
                    expires_at=datetime.now(UTC).replace(tzinfo=None) + SESSION_TTL,
                )
            )
            database.add(AuditLog(user_id=user.id, category="authentication", action="login"))
            return token, csrf_token, user

    def get_user(self, token: str | None) -> tuple[User, str] | None:
        if not token:
            return None
        now = datetime.now(UTC).replace(tzinfo=None)
        with self.session_factory.begin() as database:
            record = database.scalar(
                select(SessionRecord).where(
                    SessionRecord.token_hash == self._hash_token(token),
                    SessionRecord.expires_at > now,
                )
            )
            if record is None:
                return None
            user = database.get(User, record.user_id)
            if user is None:
                return None
            database.expunge(user)
            return user, record.csrf_token

    def logout(self, token: str | None) -> None:
        if not token:
            return
        with self.session_factory.begin() as database:
            database.execute(
                delete(SessionRecord).where(SessionRecord.token_hash == self._hash_token(token))
            )

    def change_password(self, user_id: int, current_password: str, new_password: str) -> None:
        if len(new_password) < 12:
            raise AuthenticationError("Password must contain at least 12 characters")
        with self.session_factory.begin() as database:
            user = database.get(User, user_id)
            if user is None or not self._verify(current_password, user.password_hash):
                raise AuthenticationError("Current password is invalid")
            user.password_hash = self.hasher.hash(new_password)
            user.requires_password_change = False
            database.execute(delete(SessionRecord).where(SessionRecord.user_id == user.id))
            database.add(
                AuditLog(user_id=user.id, category="authentication", action="password_changed")
            )

    def _verify(self, password: str, password_hash: str) -> bool:
        try:
            valid = self.hasher.verify(password_hash, password)
            if valid and self.hasher.check_needs_rehash(password_hash):
                return valid
            return valid
        except (InvalidHashError, VerificationError, VerifyMismatchError):
            return False

    @staticmethod
    def _hash_token(token: str) -> str:
        return hashlib.sha256(token.encode("utf-8")).hexdigest()
