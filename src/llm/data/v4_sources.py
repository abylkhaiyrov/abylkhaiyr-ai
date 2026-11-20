from __future__ import annotations

import json
import re
import subprocess
from collections.abc import Iterator
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Literal

from pyarrow.parquet import ParquetFile
from pydantic import BaseModel, Field

from llm.core.checksums import file_sha256

LICENSES = {"CC-BY-4.0", "CC-BY-SA-3.0", "CC-BY-SA-4.0", "Apache-2.0", "PSF-2.0", "Public-Domain"}


class Source(BaseModel, extra="forbid"):
    name: str = Field(pattern=r"^[a-z_]+$")
    repository: str = Field(pattern=r"^[\w-]+/[\w.-]+$")
    revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    vendor: Literal["huggingface", "github"] = "huggingface"
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    path: str
    attribution: str | None = None
    encoding: Literal["utf-8-sig", "latin-1"] = "utf-8-sig"
    license: str
    category: str
    language: str | None = None
    skip: int = Field(default=0, ge=0)
    limit: int = Field(default=0, ge=0)

    @property
    def url(self) -> str:
        if self.vendor == "github":
            return (
                f"https://raw.githubusercontent.com/{self.repository}/{self.revision}/{self.path}"
            )
        return (
            f"https://huggingface.co/datasets/{self.repository}/resolve/{self.revision}/{self.path}"
        )

    @property
    def license_url(self) -> str:
        if self.attribution is not None:
            return self.attribution
        if self.vendor == "github":
            return f"https://github.com/{self.repository}/blob/{self.revision}/LICENSE"
        return f"https://huggingface.co/datasets/{self.repository}/blob/{self.revision}/README.md"

    def local(self, root: Path) -> Path:
        suffix = Path(self.path).suffix
        if suffix not in {".parquet", ".jsonl", ".txt", ".py", ".rst", ".zip"}:
            raise ValueError(f"unsupported source format: {self.path}")
        return root / f"{self.name}{suffix}"


def load_sources(path: Path) -> dict[str, Source]:
    sources = [Source.model_validate(item) for item in json.loads(path.read_text(encoding="utf-8"))]
    indexed = {source.name: source for source in sources}
    if len(indexed) != len(sources):
        raise ValueError("duplicate source names")
    for source in sources:
        if source.license not in LICENSES:
            raise ValueError(f"{source.name}: unapproved license {source.license}")
        if (
            not re.fullmatch(r"[\w./-]+", source.path)
            or source.path.startswith("/")
            or ".." in Path(source.path).parts
        ):
            raise ValueError(f"invalid repository path: {source.path}")
    return indexed


def fetch_snapshot(url: str, target: Path, sha256: str) -> None:
    if target.is_file():
        if file_sha256(target) != sha256:
            raise ValueError(f"unexpected source checksum: {target}")
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=target.parent, prefix=f".{target.name}.", delete=False) as handle:
        staging = Path(handle.name)
    try:
        subprocess.run(
            ["curl", "-fLsS", "--retry", "2", "--max-time", "600", url, "-o", str(staging)],
            check=True,
        )
        if file_sha256(staging) != sha256:
            raise ValueError(f"unexpected download checksum: {target.name}")
        staging.replace(target)
    finally:
        staging.unlink(missing_ok=True)


def fetch(source: Source, root: Path) -> None:
    fetch_snapshot(source.url, source.local(root), source.sha256)


def parquet_rows(path: Path, columns: list[str] | None = None) -> Iterator[dict[str, Any]]:
    parquet = ParquetFile(path)
    for batch in parquet.iter_batches(batch_size=512, columns=columns):
        yield from batch.to_pylist()
