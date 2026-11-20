from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from llm.tokenizer.bpe import BPETokenizer

JSON_ESCAPES = frozenset('"\\/bfnrt')
CONTROL_LIMIT = 0x20
DONE = -1


@dataclass(frozen=True)
class State:
    tool: str | None
    segment: int
    offset: int
    escaped: bool
    buffer: str


class ToolGrammar:
    def __init__(self, tokenizer: BPETokenizer, arguments: dict[str, tuple[str, ...]]) -> None:
        self.eos_id = tokenizer.eos_id
        self.texts = [tokenizer.decode([index]) for index in range(tokenizer.vocab_size)]
        self.special = {tokenizer.bos_id, tokenizer.pad_id, *tokenizer.role_ids.values()}
        self.segments = {name: self._segments(name, keys) for name, keys in arguments.items()}
        self.state = State(None, 0, 0, False, "")
        self.cache: dict[State, Tensor] = {}

    @staticmethod
    def _segments(name: str, keys: tuple[str, ...]) -> list[str | None]:
        if not keys:
            return [f'{{"tool": "{name}", "args": {{}}}}']
        parts: list[str | None] = [f'{{"tool": "{name}", "args": {{"{keys[0]}": "', None]
        for key in keys[1:]:
            parts.extend([f'", "{key}": "', None])
        parts.append('"}}')
        return parts

    def reset(self) -> None:
        self.state = State(None, 0, 0, False, "")

    @property
    def done(self) -> bool:
        return self.state.segment == DONE

    def _advance_char(self, state: State, char: str) -> State | None:
        if state.segment == DONE:
            return None
        if state.tool is None:
            buffer = state.buffer + char
            matches = [n for n, seg in self.segments.items() if str(seg[0]).startswith(buffer)]
            if not matches:
                return None
            complete = [n for n in matches if str(self.segments[n][0]) == buffer]
            if complete:
                return self._enter_next(State(complete[0], 0, len(buffer), False, ""))
            return State(None, 0, 0, False, buffer)
        parts = self.segments[state.tool]
        part = parts[state.segment]
        if part is None:
            return self._advance_string(state, char)
        if char != part[state.offset]:
            return None
        return self._enter_next(State(state.tool, state.segment, state.offset + 1, False, ""))

    def _enter_next(self, state: State) -> State:
        assert state.tool is not None
        part = self.segments[state.tool][state.segment]
        if part is not None and state.offset < len(part):
            return state
        following = state.segment + 1
        if following >= len(self.segments[state.tool]):
            return State(state.tool, DONE, 0, False, "")
        return State(state.tool, following, 0, False, "")

    def _advance_string(self, state: State, char: str) -> State | None:
        assert state.tool is not None
        if state.escaped:
            return State(state.tool, state.segment, 0, False, "") if char in JSON_ESCAPES else None
        if char == "\\":
            return State(state.tool, state.segment, 0, True, "")
        if char == '"':
            moved = State(state.tool, state.segment + 1, 0, False, "")
            return self._advance_char(moved, char)
        if ord(char) < CONTROL_LIMIT:
            return None
        return state

    def _consume(self, state: State, text: str) -> State | None:
        for char in text:
            next_state = self._advance_char(state, char)
            if next_state is None:
                return None
            state = next_state
        return state

    def mask(self) -> Tensor:
        state = self.state
        if state not in self.cache:
            allowed = torch.zeros(len(self.texts), dtype=torch.bool)
            if state.segment == DONE:
                allowed[self.eos_id] = True
            else:
                for index, text in enumerate(self.texts):
                    if index in self.special or index == self.eos_id or not text:
                        continue
                    allowed[index] = self._consume(state, text) is not None
            self.cache[state] = allowed
        return self.cache[state]

    def advance(self, token: int) -> None:
        if token == self.eos_id or self.state.segment == DONE:
            return
        advanced = self._consume(self.state, self.texts[token])
        if advanced is None:
            raise ValueError(f"token {token} violates the grammar")
        self.state = advanced
