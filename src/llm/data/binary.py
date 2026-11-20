from __future__ import annotations

import json
from collections.abc import Iterable, Iterator, Mapping
from pathlib import Path
from typing import BinaryIO, Protocol

import numpy as np
import torch
from torch import Tensor

from llm.core.errors import DataError
from llm.tokenizer.bpe import BPETokenizer, Message

UINT16_LIMIT = 1 << 16
FLUSH_TOKENS = 1 << 20
IGNORE_INDEX = -100
TOKENIZE_BATCH = 1000
PAD_MULTIPLE = 64


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


def tokenize_corpus_split(corpus_file: Path, output_file: Path, tokenizer: BPETokenizer) -> int:
    texts = (row["text"] for row in read_jsonl(corpus_file))
    documents = (
        [tokenizer.bos_id, *ids, tokenizer.eos_id] for ids in tokenizer.encode_stream(texts)
    )
    return write_token_stream(output_file, documents, token_dtype(tokenizer.vocab_size))


def tokenize_corpus_groups(
    corpus_file: Path, directory: Path, split: str, tokenizer: BPETokenizer
) -> dict[str, int]:
    dtype = token_dtype(tokenizer.vocab_size)
    counts: dict[str, int] = {}
    handles: dict[str, BinaryIO] = {}
    try:
        batch: list[dict] = []
        for row in [*read_jsonl(corpus_file), None]:
            if row is not None:
                batch.append(row)
            if len(batch) < TOKENIZE_BATCH and row is not None:
                continue
            encoded = tokenizer.encode_batch([item["text"] for item in batch])
            for item, ids in zip(batch, encoded, strict=True):
                group = item.get("group", "general")
                if group not in handles:
                    handles[group] = with_extension(directory / f"{split}.{group}", ".bin").open(
                        "wb"
                    )
                document = [tokenizer.bos_id, *ids, tokenizer.eos_id]
                np.asarray(document, dtype=dtype).tofile(handles[group])
                counts[group] = counts.get(group, 0) + len(document)
            batch = []
    finally:
        for handle in handles.values():
            handle.close()
    return counts


def tokenize_sft_split(sft_file: Path, prefix: Path, tokenizer: BPETokenizer) -> int:
    dtype = token_dtype(tokenizer.vocab_size)
    lengths: list[int] = []
    ids_path, labels_path = (
        with_extension(prefix, ".ids.bin"),
        with_extension(prefix, ".labels.bin"),
    )
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
        indices: list[int] = []
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
        pad_multiple: int = PAD_MULTIPLE,
        bucket_window: int = 0,
    ) -> None:
        prefix = Path(prefix)
        self.ids = np.memmap(with_extension(prefix, ".ids.bin"), dtype=dtype, mode="r")
        self.labels = np.memmap(with_extension(prefix, ".labels.bin"), dtype=np.int32, mode="r")
        self.lengths = np.load(with_extension(prefix, ".lengths.npy"))
        self.offsets = np.concatenate(([0], np.cumsum(self.lengths)))
        self.count = len(self.lengths)
        if self.count == 0:
            raise DataError(f"no conversations in {prefix}")
        self.max_length = max_length
        self.batch_size = batch_size
        self.pad_id = pad_id
        self.pad_multiple = pad_multiple
        self.bucket_window = bucket_window
        self.seed = seed
        self.shuffle = shuffle
        self.epoch = 0
        self.cursor = 0
        self.order = self._make_order()

    def _make_order(self) -> np.ndarray:
        if not self.shuffle:
            return np.arange(self.count)
        rng = np.random.default_rng(self.seed + self.epoch)
        order = rng.permutation(self.count)
        return self._bucketed(order, rng) if self.bucket_window else order

    def _bucketed(self, order: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        span = self.batch_size * self.bucket_window
        pieces: list[np.ndarray] = []
        for start in range(0, len(order), span):
            chunk = order[start : start + span]
            chunk = chunk[np.argsort(self.lengths[chunk], kind="stable")]
            batches = [
                chunk[i : i + self.batch_size] for i in range(0, len(chunk), self.batch_size)
            ]
            pieces.extend(batches[i] for i in rng.permutation(len(batches)))
        return np.concatenate(pieces)

    def set_batch_size(self, batch_size: int) -> None:
        self.batch_size = batch_size

    def next_batch(self) -> tuple[Tensor, ...]:
        picked: list[int] = []
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
            rows.append(
                (self.ids[start:end].astype(np.int64), self.labels[start:end].astype(np.int64))
            )
        longest = max(len(ids) for ids, _ in rows)
        multiple = self.pad_multiple
        width = min(-(-longest // multiple) * multiple, self.max_length)
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


class PackedConversationLoader(ConversationLoader):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.carry: int | None = None

    def _take(self) -> int:
        if self.cursor >= self.count:
            self.epoch += 1
            self.cursor = 0
            self.order = self._make_order()
        index = int(self.order[self.cursor])
        self.cursor += 1
        return index

    def _span(self, index: int) -> tuple[int, int]:
        start, end = int(self.offsets[index]), int(self.offsets[index + 1])
        return start, min(end, start + self.max_length)

    def _pack_row(self, carry: int | None) -> tuple[list[int], int | None]:
        members: list[int] = []
        used = 0
        pending = carry
        while True:
            index = pending if pending is not None else self._take()
            pending = None
            start, end = self._span(index)
            if members and used + (end - start) > self.max_length:
                return members, index
            members.append(index)
            used += end - start
            if used >= self.max_length:
                return members, None

    def next_batch(self) -> tuple[Tensor, ...]:
        width = self.max_length
        inputs = torch.full((self.batch_size, width), self.pad_id, dtype=torch.long)
        targets = torch.full((self.batch_size, width), IGNORE_INDEX, dtype=torch.long)
        segments = torch.zeros((self.batch_size, width), dtype=torch.long)
        carry = self.carry
        for row in range(self.batch_size):
            members, carry = self._pack_row(carry)
            offset = 0
            for number, index in enumerate(members, start=1):
                start, end = self._span(index)
                size = end - start
                inputs[row, offset : offset + size] = torch.from_numpy(
                    self.ids[start:end].astype(np.int64)
                )
                targets[row, offset : offset + size] = torch.from_numpy(
                    self.labels[start:end].astype(np.int64)
                )
                segments[row, offset : offset + size] = number
                offset += size
        self.carry = carry
        return (
            inputs[:, :-1].contiguous(),
            targets[:, 1:].contiguous(),
            segments[:, :-1].contiguous(),
        )

    def state_dict(self) -> dict[str, int]:
        return {**super().state_dict(), "carry": -1 if self.carry is None else self.carry}

    def load_state_dict(self, state: dict[str, int]) -> None:
        super().load_state_dict(state)
        carried = state.get("carry", -1)
        self.carry = None if carried < 0 else carried


class Loader(Protocol):
    def next_batch(self) -> tuple[Tensor, ...]: ...
    def set_batch_size(self, batch_size: int) -> None: ...
    def state_dict(self) -> dict[str, int]: ...
    def load_state_dict(self, state: dict[str, int]) -> None: ...


class MixtureLoader:
    def __init__(
        self,
        loaders: Mapping[str, Loader],
        weights: Mapping[str, float],
        seed: int,
        start_weights: Mapping[str, float] | None = None,
        ramp_batches: int = 0,
    ):
        if set(weights) != set(loaders):
            raise DataError("mixture weights must match the available groups")
        if start_weights is not None and set(start_weights) != set(loaders):
            raise DataError("curriculum weights must match the available groups")
        self.names = sorted(loaders)
        self.final = self._normalize(weights)
        self.start = self._normalize(start_weights) if start_weights is not None else self.final
        self.ramp_batches = ramp_batches
        self.loaders = loaders
        self.seed = seed
        self.counter = 0

    def _normalize(self, weights: Mapping[str, float]) -> np.ndarray:
        values = np.asarray([weights[name] for name in self.names], dtype=float)
        return values / values.sum()

    @property
    def probabilities(self) -> np.ndarray:
        if not self.ramp_batches:
            return self.final
        progress = min(self.counter / self.ramp_batches, 1.0)
        return self.start + (self.final - self.start) * progress

    def set_batch_size(self, batch_size: int) -> None:
        for loader in self.loaders.values():
            loader.set_batch_size(batch_size)

    def next_batch(self) -> tuple[Tensor, ...]:
        rng = np.random.default_rng((self.seed, self.counter))
        self.counter += 1
        name = self.names[int(rng.choice(len(self.names), p=self.probabilities))]
        return self.loaders[name].next_batch()

    def state_dict(self) -> dict[str, int]:
        state = {"counter": self.counter}
        for name, loader in self.loaders.items():
            state.update({f"{name}.{key}": value for key, value in loader.state_dict().items()})
        return state

    def load_state_dict(self, state: dict[str, int]) -> None:
        self.counter = state.get("counter", 0)
        for name, loader in self.loaders.items():
            prefix = f"{name}."
            restored = {k[len(prefix) :]: v for k, v in state.items() if k.startswith(prefix)}
            loader.load_state_dict({"epoch": 0, "cursor": 0, **restored})
