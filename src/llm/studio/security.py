import threading
import time
from collections import defaultdict, deque
from collections.abc import Callable
from dataclasses import dataclass
from typing import Annotated, Any

from fastapi import Cookie, Depends, HTTPException, Request, status

from llm.studio.models import User
from llm.studio.protocol import API_VERSION, API_VERSION_HEADER
from llm.studio.service import StudioService

SESSION_COOKIE = "abylkhaiyr_session"
CSRF_COOKIE = "abylkhaiyr_csrf"
CSRF_HEADER = "X-CSRF-Token"

Identity = tuple[User, str]


class SlidingWindowLimiter:
    def __init__(self, limit: int, window_seconds: float) -> None:
        self.limit = limit
        self.window = window_seconds
        self.events: dict[str, deque[float]] = defaultdict(deque)
        self.lock = threading.Lock()

    def _trim(self, key: str, now: float) -> deque[float]:
        events = self.events[key]
        while events and now - events[0] > self.window:
            events.popleft()
        return events

    def allow(self, key: str) -> bool:
        with self.lock:
            return len(self._trim(key, time.monotonic())) < self.limit

    def hit(self, key: str) -> None:
        with self.lock:
            now = time.monotonic()
            self._trim(key, now).append(now)

    def clear(self, key: str) -> None:
        with self.lock:
            self.events.pop(key, None)


@dataclass(frozen=True)
class Dependencies:
    session: Callable[..., Identity]
    csrf: Callable[..., Identity]
    admin: Callable[..., Identity]
    admin_csrf: Callable[..., Identity]


def build_dependencies(studio: StudioService) -> Dependencies:
    def session(
        session_token: Annotated[str | None, Cookie(alias=SESSION_COOKIE)] = None,
    ) -> Identity:
        identity = studio.auth.get_user(session_token)
        if identity is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="authentication_required",
                headers={API_VERSION_HEADER: API_VERSION},
            )
        return identity

    def csrf(
        request: Request,
        identity: Annotated[Identity, Depends(session)],
        csrf_cookie: Annotated[str | None, Cookie(alias=CSRF_COOKIE)] = None,
    ) -> Identity:
        header = request.headers.get(CSRF_HEADER)
        if csrf_cookie is None or header is None or header != csrf_cookie or header != identity[1]:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="invalid_csrf")
        return identity

    def admin(identity: Annotated[Identity, Depends(session)]) -> Identity:
        if identity[0].role != "ADMIN":
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="admin_required")
        return identity

    def admin_csrf(identity: Annotated[Identity, Depends(csrf)]) -> Identity:
        if identity[0].role != "ADMIN":
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="admin_required")
        return identity

    return Dependencies(session, csrf, admin, admin_csrf)


def client_address(request: Request) -> str:
    return request.client.host if request.client is not None else "unknown"


def public_identity(identity: Identity) -> dict[str, Any]:
    from llm.studio.accounts import public_user

    return public_user(identity[0])
