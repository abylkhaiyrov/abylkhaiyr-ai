from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from llm.core.errors import DataError
from llm.tokenizer.bpe import BPETokenizer, Message

UINT16_LIMIT = 1 << 16
FLUSH_TOKENS = 1 << 20
IGNORE_INDEX = -100


def with_extension(prefix: Path, extension: str) -> Path:
    return prefix.parent / f"{prefix.name}{extension}"


def token_dtype(vocab_size: int) -> np.dtype:
    return np.dtype(np.uint16 if vocab_size <= UINT16_LIMIT else np.uint32)


def read_jsonl(path: Path) -> Iterator[dict]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            yield json.loads(line)


def write_token_stream(path: Path, streams: Iterable[list[int]], dtype: np.dtype) -> int:
    total = 0
    buffer: list[int] = []
    with path.open("wb") as handle:
        for ids in streams:
            buffer.extend(ids)
            if len(buffer) >= FLUSH_TOKENS:
                np.asarray(buffer, dtype=dtype).tofile(handle)
                total += len(buffer)
                buffer = []
        if buffer:
            np.asarray(buffer, dtype=dtype).tofile(handle)
            total += len(buffer)
    return total


def tokenize_corpus_split(
    corpus_file: Path, output_file: Path, tokenizer: BPETokenizer
) -> int:
    texts = (row["text"] for row in read_jsonl(corpus_file))
    documents = ([tokenizer.bos_id, *ids, tokenizer.eos_id] for ids in tokenizer.encode_stream(texts))
    return write_token_stream(output_file, documents, token_dtype(tokenizer.vocab_size))


def tokenize_sft_split(sft_file: Path, prefix: Path, tokenizer: BPETokenizer) -> int:
    dtype = token_dtype(tokenizer.vocab_size)
    lengths: list[int] = []
    ids_path, labels_path = with_extension(prefix, ".ids.bin"), with_extension(prefix, ".labels.bin")
    with ids_path.open("wb") as ids_handle, labels_path.open("wb") as labels_handle:
        for row in read_jsonl(sft_file):
            messages = [Message(m["role"], m["content"]) for m in row["messages"]]
            ids, labels = tokenizer.encode_conversation_with_labels(messages, IGNORE_INDEX)
            np.asarray(ids, dtype=dtype).tofile(ids_handle)
            np.asarray(labels, dtype=np.int32).tofile(labels_handle)
            lengths.append(len(ids))
    np.save(with_extension(prefix, ".lengths.npy"), np.asarray(lengths, dtype=np.int64))
    return len(lengths)


class PackedTokenLoader:
    def __init__(
        self,
        path: str | Path,
        dtype: np.dtype,
        sequence_length: int,
        batch_size: int,
        seed: int,
        shuffle: bool = True,
    ) -> None:
        self.tokens = np.memmap(path, dtype=dtype, mode="r")
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.seed = seed
        self.shuffle = shuffle
        self.num_chunks = (len(self.tokens) - 1) // sequence_length
        if self.num_chunks < 1:
            raise DataError(f"not enough tokens in {path} for sequence length {sequence_length}")
        self.epoch = 0
        self.cursor = 0
        self.order = self._make_order()

    def _make_order(self) -> np.ndarray:
        if not self.shuffle:
            return np.arange(self.num_chunks)
        return np.random.default_rng(self.seed + self.epoch).permutation(self.num_chunks)

    def set_batch_size(self, batch_size: int) -> None:
        self.batch_size = batch_size

    def next_batch(self) -> tuple[Tensor, Tensor]:
        indices = []
        while len(indices) < self.batch_size:
            if self.cursor >= self.num_chunks:
                self.epoch += 1
                self.cursor = 0
                self.order = self._make_order()
            indices.append(int(self.order[self.cursor]))
            self.cursor += 1
        length = self.sequence_length
        window = np.stack(
            [self.tokens[i * length : i * length + length + 1].astype(np.int64) for i in indices]
        )
        tensor = torch.from_numpy(window)
        return tensor[:, :-1].contiguous(), tensor[:, 1:].contiguous()

    def state_dict(self) -> dict[str, int]:
        return {"epoch": self.epoch, "cursor": self.cursor}

    def load_state_dict(self, state: dict[str, int]) -> None:
        self.epoch, self.cursor = state["epoch"], state["cursor"]
        self.order = self._make_order()


class ConversationLoader:
    def __init__(
        self,
        prefix: str | Path,
        dtype: np.dtype,
        max_length: int,
        batch_size: int,
        pad_id: int,
        seed: int,
        shuffle: bool = True,
    ) -> None:
        prefix = Path(prefix)
        self.ids = np.memmap(with_extension(prefix, ".ids.bin"), dtype=dtype, mode="r")
        self.labels = np.memmap(with_extension(prefix, ".labels.bin"), dtype=np.int32, mode="r")
        lengths = np.load(with_extension(prefix, ".lengths.npy"))
        self.offsets = np.concatenate(([0], np.cumsum(lengths)))
        self.count = len(lengths)
        if self.count == 0:
            raise DataError(f"no conversations in {prefix}")
        self.max_length = max_length
        self.batch_size = batch_size
        self.pad_id = pad_id
        self.seed = seed
        self.shuffle = shuffle
        self.epoch = 0
        self.cursor = 0
        self.order = self._make_order()

    def _make_order(self) -> np.ndarray:
        if not self.shuffle:
            return np.arange(self.count)
        return np.random.default_rng(self.seed + self.epoch).permutation(self.count)

    def set_batch_size(self, batch_size: int) -> None:
        self.batch_size = batch_size

    def next_batch(self) -> tuple[Tensor, Tensor]:
        picked = []
        while len(picked) < self.batch_size:
            if self.cursor >= self.count:
                self.epoch += 1
                self.cursor = 0
                self.order = self._make_order()
            picked.append(int(self.order[self.cursor]))
            self.cursor += 1
        rows = []
        for index in picked:
            start, end = int(self.offsets[index]), int(self.offsets[index + 1])
            end = min(end, start + self.max_length)
            rows.append((self.ids[start:end].astype(np.int64), self.labels[start:end].astype(np.int64)))
        width = max(len(ids) for ids, _ in rows)
        inputs = torch.full((len(rows), width), self.pad_id, dtype=torch.long)
        targets = torch.full((len(rows), width), IGNORE_INDEX, dtype=torch.long)
        for row, (ids, labels) in enumerate(rows):
            inputs[row, : len(ids)] = torch.from_numpy(ids)
            targets[row, : len(labels)] = torch.from_numpy(labels)
        return inputs[:, :-1].contiguous(), targets[:, 1:].contiguous()

    def state_dict(self) -> dict[str, int]:
        return {"epoch": self.epoch, "cursor": self.cursor}

    def load_state_dict(self, state: dict[str, int]) -> None:
        self.epoch, self.cursor = state["epoch"], state["cursor"]
        self.order = self._make_order()
