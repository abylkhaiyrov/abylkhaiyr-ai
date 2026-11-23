from __future__ import annotations

import json

from sqlalchemy.orm import Session, sessionmaker

from llm.studio.models import ApplicationSetting


class AccessPolicy:
    def __init__(self, session_factory: sessionmaker[Session], key: str) -> None:
        self.session_factory = session_factory
        self.key = key

    def allowed(self) -> set[str] | None:
        with self.session_factory() as database:
            row = database.get(ApplicationSetting, self.key)
            value = None if row is None else row.value
        if value is None:
            return None
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return set()
        if not isinstance(parsed, list):
            return set()
        return {item for item in parsed if isinstance(item, str)}

    def store(self, identifiers: set[str]) -> list[str]:
        ordered = sorted(identifiers)
        with self.session_factory.begin() as database:
            row = database.get(ApplicationSetting, self.key)
            if row is None:
                database.add(ApplicationSetting(key=self.key, value=json.dumps(ordered)))
            else:
                row.value = json.dumps(ordered)
        return ordered
