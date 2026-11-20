from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

from llm.core.config import TokenizerConfig
from llm.core.errors import TokenizerError

SPECIAL_TOKENS = ("<pad>", "<bos>", "<eos>", "<system>", "<user>", "<assistant>", "<tool>")
TOKENIZER_FILE = "tokenizer.json"
META_FILE = "meta.json"
BATCH_SIZE = 1000
SPECIAL_PATTERN = re.compile("(" + "|".join(re.escape(t) for t in SPECIAL_TOKENS) + ")")


@dataclass(frozen=True)
class Message:
    role: str
    content: str


class BPETokenizer:
    def __init__(self, backend: Tokenizer) -> None:
        self.backend = backend
        self.pad_id = self._token_id("<pad>")
        self.bos_id = self._token_id("<bos>")
        self.eos_id = self._token_id("<eos>")
        self.role_ids = {
            "system": self._token_id("<system>"),
            "user": self._token_id("<user>"),
            "assistant": self._token_id("<assistant>"),
            "tool": self._token_id("<tool>"),
        }

    def _token_id(self, token: str) -> int:
        token_id = self.backend.token_to_id(token)
        if token_id is None:
            raise TokenizerError(f"missing special token: {token}")
        return token_id

    @property
    def vocab_size(self) -> int:
        return self.backend.get_vocab_size()

    def validate_checkpoint(self, vocab_size: int, version: object) -> None:
        if self.vocab_size != vocab_size:
            raise TokenizerError("checkpoint and tokenizer vocabulary sizes differ")
        if version is not None and version != self.version:
            raise TokenizerError("checkpoint tokenizer fingerprint differs")

    @property
    def version(self) -> str:
        return hashlib.sha256(self.backend.to_str().encode("utf-8")).hexdigest()[:16]

    @classmethod
    def train(cls, texts: Iterable[str], config: TokenizerConfig) -> BPETokenizer:
        backend = Tokenizer(models.BPE())
        backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        backend.decoder = decoders.ByteLevel()
        trainer = trainers.BpeTrainer(
            vocab_size=config.vocab_size,
            min_frequency=config.min_frequency,
            special_tokens=list(SPECIAL_TOKENS),
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            show_progress=False,
        )
        backend.train_from_iterator(texts, trainer=trainer)
        return cls(backend)

    def _encode_plain(self, text: str) -> list[int]:
        if not SPECIAL_PATTERN.search(text):
            return self.backend.encode(text, add_special_tokens=False).ids
        ids: list[int] = []
        for index, piece in enumerate(SPECIAL_PATTERN.split(text)):
            if index % 2 == 0:
                ids.extend(
                    self.backend.encode(piece, add_special_tokens=False).ids if piece else []
                )
            else:
                for part in (piece[:1], piece[1:]):
                    ids.extend(self.backend.encode(part, add_special_tokens=False).ids)
        return ids

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        ids = self._encode_plain(text)
        return ([self.bos_id] if add_bos else []) + ids + ([self.eos_id] if add_eos else [])

    def encode_batch(self, texts: list[str]) -> list[list[int]]:
        if any(SPECIAL_PATTERN.search(text) for text in texts):
            return [self._encode_plain(text) for text in texts]
        return [e.ids for e in self.backend.encode_batch(texts, add_special_tokens=False)]

    def encode_stream(self, texts: Iterable[str]) -> Iterator[list[int]]:
        batch: list[str] = []
        for text in texts:
            batch.append(text)
            if len(batch) == BATCH_SIZE:
                yield from self.encode_batch(batch)
                batch = []
        if batch:
            yield from self.encode_batch(batch)

    def decode(self, ids: list[int], skip_special_tokens: bool = False) -> str:
        return self.backend.decode(ids, skip_special_tokens=skip_special_tokens)

    def encode_message(self, message: Message) -> list[int]:
        role_id = self.role_ids.get(message.role)
        if role_id is None:
            raise TokenizerError(f"unknown role: {message.role}")
        return [role_id, *self.encode(message.content), self.eos_id]

    def encode_conversation(
        self, messages: list[Message], add_generation_prompt: bool = False
    ) -> list[int]:
        ids = [self.bos_id]
        for message in messages:
            ids.extend(self.encode_message(message))
        if add_generation_prompt:
            ids.append(self.role_ids["assistant"])
        return ids

    def encode_conversation_with_labels(
        self, messages: list[Message], ignore_index: int
    ) -> tuple[list[int], list[int]]:
        ids, labels = [self.bos_id], [ignore_index]
        for message in messages:
            piece = self.encode_message(message)
            ids.extend(piece)
            if message.role == "assistant":
                labels.extend([ignore_index] + piece[1:])
            else:
                labels.extend([ignore_index] * len(piece))
        return ids, labels

    def save(self, directory: str | Path) -> None:
        target = Path(directory)
        target.mkdir(parents=True, exist_ok=True)
        self.backend.save(str(target / TOKENIZER_FILE))
        meta = {"version": self.version, "vocab_size": self.vocab_size}
        (target / META_FILE).write_text(json.dumps(meta), encoding="utf-8")

    @classmethod
    def load(cls, directory: str | Path) -> BPETokenizer:
        file = Path(directory) / TOKENIZER_FILE
        if not file.is_file():
            raise TokenizerError(f"tokenizer not found: {file}")
        return cls(Tokenizer.from_file(str(file)))
