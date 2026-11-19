from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

from llm.core.config import TokenizerConfig
from llm.core.errors import DataError
from llm.data.binary import (
    read_jsonl,
    tokenize_corpus_groups,
    tokenize_corpus_split,
    tokenize_sft_split,
)
from llm.data.pipeline import MANIFEST_FILE, SPLITS
from llm.tokenizer.bpe import BPETokenizer

TOKENS_DIR = "tokens"
TRAIN_FILE = "train.jsonl"


def tokenizer_training_texts(data_dir: str | Path) -> Iterator[str]:
    root = Path(data_dir)
    corpus = root / "corpus" / TRAIN_FILE
    if not corpus.is_file():
        raise DataError(f"corpus not found: {corpus}; run prepare-data first")
    for row in read_jsonl(corpus):
        yield row["text"]
    sft = root / "sft" / TRAIN_FILE
    if sft.is_file():
        for row in read_jsonl(sft):
            for message in row["messages"]:
                yield message["content"]


def train_tokenizer(data_dir: str | Path, config: TokenizerConfig) -> BPETokenizer:
    tokenizer = BPETokenizer.train(tokenizer_training_texts(data_dir), config)
    tokenizer.save(config.path)
    return tokenizer


def tokenize_dataset(data_dir: str | Path, tokenizer: BPETokenizer) -> dict[str, dict]:
    root = Path(data_dir)
    (root / TOKENS_DIR).mkdir(exist_ok=True)
    summary: dict[str, dict] = {}
    for split in SPLITS:
        corpus_file = root / "corpus" / f"{split}.jsonl"
        sft_file = root / "sft" / f"{split}.jsonl"
        summary[split] = {
            "tokens": tokenize_corpus_split(
                corpus_file, root / TOKENS_DIR / f"{split}.bin", tokenizer
            ),
            "groups": tokenize_corpus_groups(corpus_file, root / TOKENS_DIR, split, tokenizer),
            "conversations": tokenize_sft_split(sft_file, root / "sft" / split, tokenizer),
        }
    manifest_path = root / MANIFEST_FILE
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["tokenizer"] = {"version": tokenizer.version, "vocab_size": tokenizer.vocab_size}
    manifest["tokenized"] = summary
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return summary


def roundtrip_check(tokenizer: BPETokenizer, texts: list[str]) -> bool:
    return all(tokenizer.decode(tokenizer.encode(text)) == text for text in texts)
