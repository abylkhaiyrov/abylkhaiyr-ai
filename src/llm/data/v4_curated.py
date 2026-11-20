from __future__ import annotations

import csv
import io
import re
from collections.abc import Iterator
from pathlib import Path
from zipfile import ZipFile

from llm.data.cleaning import EMAIL_PATTERN
from llm.data.dedup import stable_hash

PG_START = re.compile(r"^\*\*\* START OF .*PROJECT GUTENBERG", re.MULTILINE)
PG_END = re.compile(r"^\*\*\* END OF .*PROJECT GUTENBERG", re.MULTILINE)
REVIEW_ARCHIVE_MEMBER = "01_pc_train_ib.csv"


def book_body(raw: str) -> str:
    start, end = PG_START.search(raw), PG_END.search(raw)
    if start is None or end is None or end.start() <= start.end():
        raise ValueError("Project Gutenberg boundaries not found")
    return raw[raw.index("\n", start.end()) + 1 : end.start()]


def prose_chunks(raw: str, limit: int = 3500) -> Iterator[str]:
    chunk: list[str] = []
    length = 0
    for paragraph in re.split(r"\n\s*\n", raw.strip()):
        if length + len(paragraph) >= limit and chunk:
            yield "\n\n".join(chunk)
            chunk = []
            length = 0
        chunk.append(paragraph)
        length += len(paragraph) + 2
    if chunk:
        yield "\n\n".join(chunk)


def cpython_code(raw: str) -> str:
    return EMAIL_PATTERN.sub("<email>", raw).strip("\n")


def kazsandra_rows(path: Path, *, semantic: bool) -> Iterator[dict[str, str]]:
    with ZipFile(path) as archive, archive.open(REVIEW_ARCHIVE_MEMBER) as raw:
        for row in csv.DictReader(io.TextIOWrapper(raw, encoding="utf-8")):
            semantic_partition = stable_hash(row["custom_id"]) % 4 == 0
            if semantic == semantic_partition:
                yield row
