from __future__ import annotations

import json
from collections import Counter
from contextlib import ExitStack
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO

from llm.core.config import DataConfig
from llm.core.errors import ResourceError
from llm.core.hardware import free_disk_gb
from llm.data.cleaning import clean_for_kind, detect_language, repeated_line_ratio, symbol_ratio
from llm.data.dedup import ExactDeduplicator, MinHashDeduplicator, stable_hash
from llm.data.sources import Record, stream_source

SPLITS = ("train", "validation", "test")
HASH_RESOLUTION = 1_000_000
MANIFEST_FILE = "manifest.json"
EXEMPT_KINDS = {"code"}
PRETRAIN_KINDS = {"text", "math", "code"}
GROUPS = ("corpus", "sft")


@dataclass
class SourceStats:
    license: str
    dataset: str
    config: str | None
    seen: int = 0
    kept: int = 0
    rejected: Counter[str] = field(default_factory=Counter)


def assign_split(text: str, config: DataConfig) -> str:
    bucket = stable_hash(f"{config.seed}:{text}") % HASH_RESOLUTION
    test_edge = config.test_fraction * HASH_RESOLUTION
    validation_edge = test_edge + config.validation_fraction * HASH_RESOLUTION
    if bucket < test_edge:
        return "test"
    return "validation" if bucket < validation_edge else "train"


class QualityFilter:
    def __init__(self, config: DataConfig) -> None:
        self.config = config

    def rejection_reason(
        self, text: str, kind: str, min_chars: int | None = None, check_language: bool = True
    ) -> str | None:
        threshold = min_chars if min_chars is not None else self.config.min_chars
        if len(text) < threshold and kind in PRETRAIN_KINDS - {"math"}:
            return "too_short"
        if kind in EXEMPT_KINDS:
            return None
        if symbol_ratio(text) > self.config.max_symbol_ratio:
            return "symbols"
        if repeated_line_ratio(text) > self.config.max_repeated_line_ratio:
            return "repetition"
        if self.config.detect_language and check_language:
            if detect_language(text[: self.config.language_sample_chars]) not in self.config.allowed_languages:
                return "language"
        return None


class CorpusBuilder:
    def __init__(self, config: DataConfig) -> None:
        self.config = config
        self.quality = QualityFilter(config)
        self.exact = {group: ExactDeduplicator() for group in GROUPS}
        self.near = {
            group: MinHashDeduplicator(
                config.minhash_permutations,
                config.near_duplicate_threshold,
                config.shingle_size,
                config.seed,
            )
            for group in GROUPS
        }

    def build(self) -> dict[str, object]:
        output = Path(self.config.output_dir)
        if free_disk_gb(output) < self.config.min_free_disk_gb:
            raise ResourceError("not enough free disk space for data preparation")
        (output / "corpus").mkdir(parents=True, exist_ok=True)
        (output / "sft").mkdir(parents=True, exist_ok=True)
        stats: dict[str, SourceStats] = {}
        split_counts: Counter[str] = Counter()
        with ExitStack() as stack:
            handles = {
                (group, split): stack.enter_context(
                    (output / group / f"{split}.jsonl").open("w", encoding="utf-8")
                )
                for group in GROUPS
                for split in SPLITS
            }
            for source in self.config.sources:
                source_stats = stats[source.name] = SourceStats(
                    source.license, source.dataset, source.config
                )
                for record in stream_source(source, self.config.raw_dir):
                    source_stats.seen += 1
                    self._process(record, handles, source_stats, split_counts)
        manifest = {
            "sources": {
                name: {
                    "dataset": s.dataset,
                    "config": s.config,
                    "license": s.license,
                    "seen": s.seen,
                    "kept": s.kept,
                    "rejected": dict(s.rejected),
                }
                for name, s in stats.items()
            },
            "splits": dict(split_counts),
            "seed": self.config.seed,
            "near_duplicate_threshold": self.config.near_duplicate_threshold,
        }
        (output / MANIFEST_FILE).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return manifest

    def _process(
        self,
        record: Record,
        handles: dict[tuple[str, str], IO[str]],
        stats: SourceStats,
        split_counts: Counter[str],
    ) -> None:
        text = clean_for_kind(record.text, record.kind)
        response = clean_for_kind(record.response, record.kind) if record.response else None
        full_text = text if response is None else f"{text}\n{response}"
        reason = self.quality.rejection_reason(
            full_text, record.kind, record.min_chars, record.check_language
        )
        if reason is None and response is not None and not response:
            reason = "empty_response"
        group = "sft" if response is not None else "corpus"
        if reason is None and self.exact[group].is_duplicate(full_text):
            reason = "exact_duplicate"
        if reason is None and self.near[group].is_duplicate(full_text):
            reason = "near_duplicate"
        if reason is not None:
            stats.rejected[reason] += 1
            return
        split = assign_split(full_text, self.config)
        stats.kept += 1
        split_counts[split] += 1
        if response is None:
            payload = {"source": record.source, "language": record.language, "text": text}
            handles[("corpus", split)].write(json.dumps(payload, ensure_ascii=False) + "\n")
        else:
            messages = [
                {"role": "user", "content": text},
                {"role": "assistant", "content": response},
            ]
            payload = {"source": record.source, "messages": messages}
            handles[("sft", split)].write(json.dumps(payload, ensure_ascii=False) + "\n")
