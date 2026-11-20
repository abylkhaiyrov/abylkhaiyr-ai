from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass, field, replace
from pathlib import Path

from llm.core.config import ModelConfig
from llm.data.binary import read_jsonl, tokenize_sft_split
from llm.data.cleaning import detect_language
from llm.data.codegen import build_code_dataset
from llm.data.identity import build_identity_dataset, load_identity_facts
from llm.data.instructgen import build_instruction_dataset
from llm.data.sft_builder import SPLITS, write_group_files
from llm.data.sft_quality import (
    PromptBlocklist,
    QualityRules,
    SFTQualityFilter,
    equations_hold,
    repair_gsm8k,
)
from llm.data.textgen import build_reading_dataset
from llm.evaluation.prompts import evaluation_prompts
from llm.tokenizer.bpe import BPETokenizer

REAL_GROUPS = ("chat", "math", "code")
CHAT_LANGUAGES = ("en", "ru", "kk")
OUTPUT_GROUPS = (
    *(f"chat_{language}" for language in CHAT_LANGUAGES),
    "chat_other",
    *(f"reading_{language}" for language in CHAT_LANGUAGES),
    "dialog",
    "skills",
    "identity",
    "math",
    "word_math",
    "code",
)
KEEP_ORDER = {"math"}
GSM8K_SOURCE = "gsm8k_sft"
SQL_SOURCE = "sql_sft"
REFUSAL_SOURCE = "synthetic_refusal"
GENERATED_CODE_PREFIX = "synthetic_code"
MANIFEST = "manifest.json"


@dataclass
class InstructionBuildConfig:
    source_dir: str = "data/v2"
    output_dir: str = "data/v3_1"
    corpus_file: str = "data/v2/corpus/train.jsonl"
    benchmark_dir: str = "data/benchmarks"
    synthetic_count: int = 12000
    identity_count: int = 400
    sql_cap: int = 400
    reading_per_language: int = 3500
    code_count: int = 6000
    oasst_cap: int = 6000
    dolly_cap: int = 8000
    aya_cap: int = 6000
    prefix_cap: int = 600
    seed: int = 1
    version: str = "v3.1"
    rules: QualityRules = field(default_factory=QualityRules)

    def quality_rules(self) -> QualityRules:
        caps = {
            SQL_SOURCE: self.sql_cap,
            "oasst1": self.oasst_cap,
            "dolly_sft": self.dolly_cap,
            "aya_sft": self.aya_cap,
        }
        return replace(self.rules, source_caps=caps, prefix_cap=self.prefix_cap)


def read_group(source_dir: Path, group: str) -> Iterator[dict]:
    for split in SPLITS:
        path = source_dir / "sft_groups" / f"{split}.{group}.jsonl"
        if path.is_file():
            yield from read_jsonl(path)


def reassign(row: dict) -> dict | None:
    source = row.get("source", "")
    if source == GSM8K_SOURCE:
        answer = repair_gsm8k(row["messages"][-1]["content"])
        if not equations_hold(answer):
            return None
        row["messages"][-1]["content"] = answer
        return {**row, "group": "word_math", "category": "word_math"}
    if source == SQL_SOURCE:
        return {**row, "group": "code", "category": "sql"}
    if source == REFUSAL_SOURCE:
        return {**row, "group": "dialog", "category": "refusal"}
    if row["group"] == "chat":
        language = detect_language(row["messages"][-1]["content"])
        suffix = language if language in CHAT_LANGUAGES else "other"
        return {**row, "group": f"chat_{suffix}"}
    return row


def real_rows(source_dir: Path) -> Iterator[dict]:
    for group in REAL_GROUPS:
        for row in read_group(source_dir, group):
            if row.get("source", "").startswith(GENERATED_CODE_PREFIX):
                continue
            updated = reassign(row)
            if updated is not None:
                yield updated
            elif row.get("source") == GSM8K_SOURCE:
                yield {**row, "group": "rejected_gsm8k"}


def synthetic_rows(config: InstructionBuildConfig, model: ModelConfig) -> list[dict]:
    facts = load_identity_facts(model)
    rows = build_instruction_dataset(facts, config.synthetic_count, config.seed)
    rows.extend(build_identity_dataset(facts, config.identity_count, config.seed))
    rows.extend(build_code_dataset(config.code_count, config.seed))
    rows.extend(
        build_reading_dataset(Path(config.corpus_file), config.reading_per_language, config.seed)
    )
    return rows


def describe(rows: list[dict]) -> dict[str, object]:
    languages: Counter[str] = Counter()
    for row in rows:
        languages[row.get("language") or detect_language(row["messages"][-1]["content"])] += 1
    return {
        "groups": dict(Counter(row["group"] for row in rows)),
        "categories": dict(Counter(row.get("category", row["source"]) for row in rows)),
        "languages": dict(languages),
    }


def build_instruction_corpus(
    config: InstructionBuildConfig, tokenizer: BPETokenizer, model: ModelConfig
) -> dict[str, object]:
    source, output = Path(config.source_dir), Path(config.output_dir)
    groups_dir = output / "sft_groups"
    groups_dir.mkdir(parents=True, exist_ok=True)
    quality = SFTQualityFilter(config.quality_rules(), tokenizer)
    blocklist = PromptBlocklist(evaluation_prompts(config.benchmark_dir))
    candidates = [*real_rows(source), *synthetic_rows(config, model)]
    admitted: list[dict] = []
    contaminated = 0
    for row in candidates:
        if row["group"] == "rejected_gsm8k":
            quality.rejected[f"{GSM8K_SOURCE}:wrong_arithmetic"] += 1
        elif blocklist.blocks(row):
            contaminated += 1
        elif quality.admit(row):
            admitted.append(row)
    counts = write_group_files(admitted, groups_dir, config.seed, KEEP_ORDER)
    for group in OUTPUT_GROUPS:
        for split in SPLITS:
            path = groups_dir / f"{split}.{group}.jsonl"
            if path.is_file():
                tokenize_sft_split(path, groups_dir / f"{split}.{group}", tokenizer)
    manifest = {
        "version": config.version,
        "seed": config.seed,
        "candidates": len(candidates),
        "admitted": len(admitted),
        "benchmark_overlap_removed": contaminated,
        "rejected": dict(quality.rejected),
        "splits": counts,
        **describe(admitted),
    }
    (output / MANIFEST).write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return manifest
