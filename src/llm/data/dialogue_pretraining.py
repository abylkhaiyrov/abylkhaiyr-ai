from __future__ import annotations

from collections.abc import Iterable, Iterator
from enum import StrEnum
from pathlib import Path

from llm.data.binary import read_jsonl, token_dtype, write_token_stream
from llm.tokenizer.bpe import BPETokenizer, Message


class StreamFormat(StrEnum):
    DIALOGUE = "dialogue"
    PLAIN = "plain"


def conversations(files: Iterable[Path]) -> Iterator[list[Message]]:
    for path in files:
        for row in read_jsonl(path):
            yield [Message(item["role"], item["content"]) for item in row["messages"]]


def document_ids(
    messages: list[Message], tokenizer: BPETokenizer, stream_format: StreamFormat
) -> list[int]:
    if stream_format is StreamFormat.DIALOGUE:
        return tokenizer.encode_conversation(messages)
    text = "\n".join(message.content for message in messages)
    return [tokenizer.bos_id, *tokenizer.encode(text), tokenizer.eos_id]


def write_dialogue_stream(
    files: Iterable[Path], target: Path, tokenizer: BPETokenizer, stream_format: StreamFormat
) -> int:
    documents = (
        document_ids(messages, tokenizer, stream_format) for messages in conversations(files)
    )
    return write_token_stream(target, documents, token_dtype(tokenizer.vocab_size))
