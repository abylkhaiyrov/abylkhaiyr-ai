from __future__ import annotations

import time

FLUSH_INTERVAL_SECONDS = 0.04
FLUSH_CHARACTERS = 24
PERSIST_INTERVAL_SECONDS = 1.0


class DeltaBuffer:
    def __init__(
        self,
        flush_interval: float = FLUSH_INTERVAL_SECONDS,
        flush_characters: int = FLUSH_CHARACTERS,
    ) -> None:
        self.flush_interval = flush_interval
        self.flush_characters = flush_characters
        self._pieces: list[str] = []
        self._size = 0
        self._last_flush = time.monotonic()

    def add(self, piece: str) -> str | None:
        self._pieces.append(piece)
        self._size += len(piece)
        due = time.monotonic() - self._last_flush >= self.flush_interval
        if due or self._size >= self.flush_characters:
            return self.drain()
        return None

    def drain(self) -> str | None:
        if not self._pieces:
            return None
        text = "".join(self._pieces)
        self._pieces.clear()
        self._size = 0
        self._last_flush = time.monotonic()
        return text
