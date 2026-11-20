from __future__ import annotations

import json
from collections import Counter, defaultdict
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from itertools import islice
from pathlib import Path

import numpy as np

from llm.data.binary import read_jsonl, token_dtype, write_token_stream
from llm.data.dedup import ExactDeduplicator, MinHashDeduplicator, stable_hash
from llm.data.v4_curated import book_body, cpython_code, kazsandra_rows, prose_chunks
from llm.data.v4_quality import LANGUAGES, CyrillicLanguageFilter, clean_v4_text
from llm.data.v4_sources import Source, parquet_rows
from llm.tokenizer.bpe import BPETokenizer

SPLITS = ("train", "validation", "test")


@dataclass(frozen=True)
class Document:
    text: str
    source: str
    source_id: str
    url: str
    language: str
    category: str
    partition: str | None = None

    @property
    def split(self) -> str:
        if self.partition is not None:
            return self.partition
        bucket = stable_hash(f"{self.source}/{self.source_id}") % 100
        if bucket < 2:
            return "test"
        if bucket < 5:
            return "validation"
        return "train"


def wiki_rows(source: Source, root: Path, start: int, limit: int) -> Iterator[dict]:
    return islice(parquet_rows(source.local(root), ["id", "url", "text"]), start, start + limit)


def build_language_filter(sources: dict[str, Source], root: Path) -> CyrillicLanguageFilter:
    return CyrillicLanguageFilter(
        {
            lang: (row["text"] for row in wiki_rows(sources[f"wiki_{lang}"], root, 0, 1200))
            for lang in ("ru", "kk", "ky")
        }
    )


def collect_wiki(
    sources: dict[str, Source],
    root: Path,
    language_filter: CyrillicLanguageFilter,
    rejected: Counter[str],
) -> Iterator[Document]:
    for source in sources.values():
        if source.category not in {"encyclopedia", "education"} or source.language not in LANGUAGES:
            continue
        for row in wiki_rows(source, root, source.skip, source.limit):
            text = clean_v4_text(row["text"], 180)
            if text is None:
                rejected[f"{source.name}.quality"] += 1
                continue
            if not language_filter.accepts(text, source.language):
                rejected[f"{source.name}.language"] += 1
                continue
            yield Document(
                text, source.name, row["id"], row["url"], source.language, source.category
            )


def collect_curated(
    sources: dict[str, Source],
    root: Path,
    quality: CyrillicLanguageFilter,
    rejected: Counter[str],
) -> Iterator[Document]:
    for source in sources.values():
        if source.vendor != "github" or source.language is None:
            continue
        path = source.local(root)
        if source.category == "reviews":
            passages = (
                (row["custom_id"], row["text"])
                for row in islice(kazsandra_rows(path, semantic=False), source.limit)
            )
        else:
            raw = path.read_text(encoding=source.encoding)
            if source.category == "code":
                yield Document(
                    cpython_code(raw),
                    source.name,
                    source.path,
                    source.url,
                    source.language,
                    source.category,
                    "train",
                )
                continue
            if source.license == "Public-Domain":
                raw = book_body(raw)
            passages = ((str(index), text) for index, text in enumerate(prose_chunks(raw)))
        minimum = 100 if source.category == "reviews" else 180
        for identifier, passage in passages:
            text = clean_v4_text(passage, minimum)
            if text is None or not quality.accepts(text, source.language):
                rejected[f"{source.name}.quality_or_language"] += 1
                continue
            yield Document(
                text,
                source.name,
                identifier,
                source.url,
                source.language,
                source.category,
                "train",
            )


def write_splits(documents: Iterator[Document], target: Path) -> tuple[dict, dict]:
    groups: dict[str, list[Document]] = defaultdict(list)
    for document in documents:
        groups[document.split].append(document)
    exact = ExactDeduplicator()
    near = MinHashDeduplicator(48, 0.8, 4, 1234)
    counts: Counter[str] = Counter()
    rejected: Counter[str] = Counter()
    for split in reversed(SPLITS):
        with (target / f"{split}.jsonl").open("w", encoding="utf-8") as handle:
            for document in groups[split]:
                key = " ".join(document.text.casefold().split())
                if exact.is_duplicate(key) or near.is_duplicate(key):
                    rejected[split] += 1
                    continue
                row = asdict(document)
                row.pop("partition")
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                counts[f"{split}.{document.language}.{document.category}"] += 1
    return dict(counts), dict(rejected)


def write_tokens(root: Path, tokenizer: BPETokenizer) -> dict:
    directory = root / "tokens"
    directory.mkdir()
    result = {}
    dtype = token_dtype(tokenizer.vocab_size)
    for split in SPLITS:
        rows = list(read_jsonl(root / f"{split}.jsonl"))
        counts = {}
        for language in LANGUAGES:
            texts = (row["text"] for row in rows if row["language"] == language)
            sequences = (
                [tokenizer.bos_id, *ids, tokenizer.eos_id] for ids in tokenizer.encode_stream(texts)
            )
            counts[language] = write_token_stream(
                directory / f"{split}.{language}.bin", sequences, dtype
            )
        result[split] = summarize_token_streams(directory, split, counts, dtype)
    return result


def summarize_token_streams(
    directory: Path, split: str, counts: dict[str, int], dtype: np.dtype
) -> dict:
    blocks = 512
    streams = [
        np.memmap(directory / f"{split}.{lang}.bin", mode="r", dtype=dtype) for lang in LANGUAGES
    ]
    vocabulary: set[int] = set()
    with (directory / f"{split}.bin").open("wb") as handle:
        for start in range(0, max(map(len, streams)), blocks):
            for tokens in streams:
                if start < len(tokens):
                    tokens[start : start + blocks].tofile(handle)
        for tokens in streams:
            for start in range(0, len(tokens), 1 << 20):
                vocabulary.update(np.unique(tokens[start : start + (1 << 20)]).tolist())
    return {
        "stored_tokens": {**counts, "all": sum(counts.values())},
        "distinct_token_ids": len(vocabulary),
    }


def source_manifest(sources: dict[str, Source], root: Path) -> dict:
    return {
        name: {
            **source.model_dump(),
            "url": source.url,
            "license_url": source.license_url,
            "sha256": source.sha256,
        }
        for name, source in sources.items()
        if source.local(root).is_file()
    }
