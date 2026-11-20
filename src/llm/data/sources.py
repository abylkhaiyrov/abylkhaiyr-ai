from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

from datasets import load_dataset

from llm.core.config import DataSourceConfig
from llm.core.errors import DataError


@dataclass(frozen=True)
class Record:
    source: str
    kind: str
    language: str | None
    text: str
    response: str | None = None
    min_chars: int | None = None
    check_language: bool = True
    group: str = "general"


class _SafeDict(dict[str, Any]):
    def __missing__(self, key: str) -> str:
        return ""


def _render(template: str, row: dict[str, Any]) -> str:
    return template.format_map(_SafeDict({k: v for k, v in row.items() if v is not None}))


def stream_source(
    source: DataSourceConfig, raw_dir: str, allowed_licenses: list[str] | None = None
) -> Iterator[Record]:
    try:
        file_options = {"data_files": source.data_files} if source.data_files else {}
        dataset = load_dataset(
            source.dataset,
            source.config,
            split=source.split,
            streaming=True,
            cache_dir=raw_dir,
            **file_options,
        )
    except Exception as error:
        raise DataError(f"cannot load {source.dataset}: {error}") from error
    produced = 0
    for row in dataset:
        if source.max_documents is not None and produced >= source.max_documents:
            return
        if not _row_allowed(source, row, allowed_licenses):
            continue
        record = _to_record(source, row)
        if record is not None:
            produced += 1
            yield record


def _row_allowed(
    source: DataSourceConfig, row: dict[str, Any], allowed_licenses: list[str] | None
) -> bool:
    if source.row_filter and any(str(row.get(k)) != v for k, v in source.row_filter.items()):
        return False
    if source.license_field is None or allowed_licenses is None:
        return True
    return str(row.get(source.license_field, "")).lower() in allowed_licenses


def _to_record(source: DataSourceConfig, row: dict[str, Any]) -> Record | None:
    if source.kind == "instruction":
        if source.template is None or source.response_field is None:
            raise DataError(f"instruction source {source.name} needs template and response_field")
        prompt = _render(source.template, row).strip()
        response = str(row.get(source.response_field) or "").strip()
        if not prompt or not response:
            return None
        return Record(
            source.name,
            source.kind,
            source.language,
            prompt,
            response,
            source.min_chars,
            source.check_language,
            source.group,
        )
    if source.template is not None:
        text = _render(source.template, row)
    elif source.text_field is not None:
        text = str(row.get(source.text_field) or "")
    else:
        raise DataError(f"source {source.name} needs text_field or template")
    return Record(
        source.name,
        source.kind,
        source.language,
        text[: source.max_chars],
        None,
        source.min_chars,
        source.check_language,
        source.group,
    )
