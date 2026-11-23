from __future__ import annotations

from collections.abc import Sequence
from typing import Any, TypeVar

T = TypeVar("T")
MAX_PAGE = 200


def paginate(items: Sequence[T], offset: int, limit: int) -> dict[str, Any]:
    start = max(offset, 0)
    return {"items": list(items[start : start + min(max(limit, 1), MAX_PAGE)]), "total": len(items)}


def matches(query: str | None, *fields: object) -> bool:
    needle = (query or "").strip().lower()
    return not needle or needle in " ".join(str(item or "") for item in fields).lower()
