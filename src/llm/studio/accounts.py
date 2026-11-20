from __future__ import annotations

import re
from typing import Any, Literal

from llm.studio.models import User

Role = Literal["ADMIN", "USER"]
ROLES: tuple[Role, ...] = ("ADMIN", "USER")
PASSWORD_MIN = 12
PASSWORD_MAX = 512
NAME_MAX = 64
EMAIL_PATTERN = re.compile(r"^[^@\s]{1,64}@[^@\s]+\.[^@\s]{2,}$")
USERNAME_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_.-]{2,31}$")


class AuthenticationError(Exception):
    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def normalize_email(value: str) -> str:
    email = value.strip().lower()
    if len(email) > 254 or not EMAIL_PATTERN.match(email):
        raise AuthenticationError("invalid_email")
    return email


def normalize_username(value: str) -> str:
    username = value.strip().lower()
    if not USERNAME_PATTERN.match(username):
        raise AuthenticationError("invalid_username")
    return username


def normalize_name(value: str, code: str) -> str:
    name = " ".join(value.split())
    if not name or len(name) > NAME_MAX:
        raise AuthenticationError(code)
    return name


def validate_password(password: str, username: str | None = None) -> None:
    if len(password) < PASSWORD_MIN or len(password) > PASSWORD_MAX:
        raise AuthenticationError("weak_password")
    if username and password.lower() == username.lower():
        raise AuthenticationError("weak_password")


def public_user(user: User) -> dict[str, Any]:
    return {
        "id": user.uuid,
        "first_name": user.first_name,
        "last_name": user.last_name,
        "email": user.email,
        "username": user.username,
        "role": user.role,
        "is_active": user.is_active,
        "requires_password_change": user.requires_password_change,
        "created_at": user.created_at.isoformat(),
        "last_login_at": None if user.last_login_at is None else user.last_login_at.isoformat(),
        "last_active_at": None if user.last_active_at is None else user.last_active_at.isoformat(),
    }
