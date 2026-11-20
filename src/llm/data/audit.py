from __future__ import annotations

import ast
import itertools
import json
import re
from collections import Counter
from collections.abc import Iterable
from pathlib import Path

import numpy as np

from llm.data.binary import read_jsonl, with_extension
from llm.data.cleaning import detect_language
from llm.data.sft_builder import SPLITS
from llm.data.sft_quality import PromptBlocklist, equations_hold, normalize
from llm.evaluation.prompts import evaluation_prompts

FENCE = re.compile(r"```(?:python)?\n(.*?)```", re.DOTALL)
PERCENTILES = (50, 90, 99)
TOP_PREFIXES = 5
NUMERIC_CATEGORIES = frozenset({"arithmetic", "fractions", "percent", "powers"})
PREFIX_WORDS = 3


def read_group_rows(groups_dir: Path, split: str, group: str) -> list[dict]:
    path = groups_dir / f"{split}.{group}.jsonl"
    return list(read_jsonl(path)) if path.is_file() else []


def length_profile(groups_dir: Path, split: str, group: str) -> dict[str, float]:
    path = with_extension(groups_dir / f"{split}.{group}", ".lengths.npy")
    if not path.is_file():
        return {}
    lengths = np.load(path)
    stats = {f"p{p}": float(np.percentile(lengths, p)) for p in PERCENTILES}
    return {"mean": float(lengths.mean()), "max": float(lengths.max()), **stats}


def eos_termination_rate(groups_dir: Path, split: str, group: str, eos_id: int) -> float:
    prefix = groups_dir / f"{split}.{group}"
    labels_path = with_extension(prefix, ".labels.bin")
    lengths_path = with_extension(prefix, ".lengths.npy")
    if not labels_path.is_file():
        return 0.0
    labels = np.memmap(labels_path, dtype=np.int32, mode="r")
    ends = np.cumsum(np.load(lengths_path)) - 1
    return float(np.mean(labels[ends] == eos_id))


def language_shares(rows: Iterable[dict]) -> dict[str, float]:
    counts = Counter(detect_language(row["messages"][-1]["content"]) for row in rows)
    total = sum(counts.values()) or 1
    return {language: count / total for language, count in sorted(counts.items())}


def duplicate_rate(rows: list[dict]) -> float:
    prompts = [normalize(row["messages"][0]["content"]) for row in rows]
    return 1 - len(set(prompts)) / max(len(prompts), 1)


def top_prefix_shares(rows: list[dict]) -> dict[str, float]:
    counts = Counter(
        " ".join(normalize(row["messages"][0]["content"]).split()[:PREFIX_WORDS]) for row in rows
    )
    total = sum(counts.values()) or 1
    return {prefix: count / total for prefix, count in counts.most_common(TOP_PREFIXES)}


def python_validity(rows: list[dict]) -> float | None:
    blocks = [block for row in rows for block in FENCE.findall(row["messages"][-1]["content"])]
    if not blocks:
        return None
    valid = 0
    for block in blocks:
        try:
            ast.parse(block)
            valid += 1
        except SyntaxError:
            continue
    return valid / len(blocks)


def arithmetic_validity(rows: list[dict]) -> float | None:
    checkable = [
        row
        for row in rows
        if row["group"] == "word_math" or row["source"].rsplit("_", 1)[-1] in NUMERIC_CATEGORIES
    ]
    if not checkable:
        return None
    return float(np.mean([equations_hold(row["messages"][-1]["content"]) for row in checkable]))


def curriculum_monotonicity(rows: list[dict]) -> float | None:
    levels = [row["level"] for row in rows if "level" in row]
    if len(levels) < 2:
        return None
    return float(np.mean([a <= b for a, b in itertools.pairwise(levels)]))


def audit_corpus(
    data_dir: str | Path, eos_id: int, benchmark_dir: str | Path = "data/benchmarks"
) -> dict[str, object]:
    groups_dir = Path(data_dir) / "sft_groups"
    blocklist = PromptBlocklist(evaluation_prompts(benchmark_dir))
    groups_report: dict[str, object] = {}
    eval_overlap = 0
    prompts_by_split: dict[str, set[str]] = {}
    groups = sorted({p.name.split(".")[1] for p in groups_dir.glob("train.*.jsonl")})
    for group in groups:
        train = read_group_rows(groups_dir, "train", group)
        entry: dict[str, object] = {
            "rows": {split: len(read_group_rows(groups_dir, split, group)) for split in SPLITS},
            "tokens": length_profile(groups_dir, "train", group),
            "eos_termination_rate": eos_termination_rate(groups_dir, "train", group, eos_id),
            "language_shares": language_shares(train),
            "duplicate_prompt_rate": duplicate_rate(train),
            "top_prefix_shares": top_prefix_shares(train),
            "python_validity": python_validity(train),
            "arithmetic_validity": arithmetic_validity(train)
            if group in {"math", "word_math"}
            else None,
            "curriculum_monotonicity": curriculum_monotonicity(train),
        }
        groups_report[group] = entry
        for split in SPLITS:
            rows = read_group_rows(groups_dir, split, group)
            eval_overlap += sum(blocklist.blocks(row) for row in rows)
            prompts_by_split.setdefault(split, set()).update(
                normalize(row["messages"][0]["content"]) for row in rows
            )
    split_overlap = len(prompts_by_split["train"] & prompts_by_split["test"]) + len(
        prompts_by_split["train"] & prompts_by_split["validation"]
    )
    return {"groups": groups_report, "eval_overlap": eval_overlap, "split_overlap": split_overlap}


def write_audit(report: dict[str, object], path: str | Path) -> None:
    Path(path).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
