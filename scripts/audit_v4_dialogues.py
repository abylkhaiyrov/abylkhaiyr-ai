from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from llm.data.binary import read_jsonl, supervised_token_count
from llm.data.v4_splits import normalized
from llm.tokenizer.bpe import BPETokenizer


def group_rows(directory: Path, split: str) -> dict[str, list[dict[str, Any]]]:
    return {
        path.stem.split(".", 1)[1]: list(read_jsonl(path))
        for path in sorted(directory.glob(f"{split}.*.jsonl"))
    }


def answer_lengths(rows: list[dict[str, Any]], tokenizer: BPETokenizer) -> dict[str, float]:
    sizes = sorted(len(tokenizer.encode(row["messages"][-1]["content"])) for row in rows)
    return {
        "p50_tokens": sizes[len(sizes) // 2],
        "p90_tokens": sizes[int(len(sizes) * 0.9)],
        "share_over_80": round(sum(size > 80 for size in sizes) / len(sizes), 3),
        "share_under_4": round(sum(size < 4 for size in sizes) / len(sizes), 3),
    }


def answer_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    answers = Counter(normalized(row["messages"][-1]["content"]) for row in rows)
    top, top_count = answers.most_common(1)[0]
    return {
        "examples": len(rows),
        "unique_answer_ratio": round(len(answers) / len(rows), 3),
        "top_answer": top[:40],
        "top_answer_share": round(top_count / len(rows), 3),
    }


def protected_overlap(train: dict[str, list[dict[str, Any]]], protected: Path) -> int:
    prompts = {
        normalized(turn["content"])
        for row in read_jsonl(protected)
        for turn in row["messages"]
        if turn["role"] == "user"
    }
    return sum(
        normalized(turn["content"]) in prompts
        for rows in train.values()
        for row in rows
        for turn in row["messages"]
        if turn["role"] == "user"
    )


def source_stats(rows: list[dict[str, Any]], tokenizer: BPETokenizer) -> dict[str, Any]:
    prompts = {normalized(row["messages"][-2]["content"]) for row in rows}
    openings = Counter(
        " ".join(normalized(row["messages"][-1]["content"]).split()[:3]) for row in rows
    )
    return {
        **answer_stats(rows),
        **answer_lengths(rows, tokenizer),
        "unique_prompt_ratio": round(len(prompts) / len(rows), 3),
        "top_opening_share": round(openings.most_common(1)[0][1] / len(rows), 3),
        "mean_answer_tokens": round(
            sum(len(tokenizer.encode(row["messages"][-1]["content"])) for row in rows) / len(rows),
            1,
        ),
        "languages": dict(Counter(row["language"] for row in rows)),
    }


def audit(directory: Path, protected: Path, tokenizer: BPETokenizer) -> dict[str, Any]:
    train = group_rows(directory, "train")
    validation = group_rows(directory, "validation")
    total_tokens = {name: supervised_token_count(directory / f"train.{name}") for name in train}
    grand = sum(total_tokens.values())
    by_cell: dict[str, Counter[str]] = defaultdict(Counter)
    by_source: dict[str, Counter[str]] = defaultdict(Counter)
    by_source_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for rows in train.values():
        for row in rows:
            by_cell[row["language"]][row["category"]] += 1
            by_source[row["language"]][row["source"]] += 1
            by_source_rows[row["source"]].append(row)
    train_prompts = {
        normalized(row["messages"][0]["content"]) for rows in train.values() for row in rows
    }
    train_families = {row["family"] for rows in train.values() for row in rows}
    leakage = {
        "validation_prompt_in_train": sum(
            normalized(row["messages"][0]["content"]) in train_prompts
            for rows in validation.values()
            for row in rows
        ),
        "validation_family_in_train": sum(
            row["family"] in train_families for rows in validation.values() for row in rows
        ),
        "validation_total": sum(len(rows) for rows in validation.values()),
        "protected_prompts_in_train": protected_overlap(train, protected),
    }
    language_tokens: Counter[str] = Counter()
    for name, count in total_tokens.items():
        language_tokens[
            name.rsplit("_", 1)[-1] if name.rsplit("_", 1)[-1] in {"en", "ru", "kk"} else "mixed"
        ] += count
    return {
        "supervised_tokens": total_tokens,
        "supervised_token_share_by_language": {
            key: round(value / grand, 4) for key, value in language_tokens.items()
        },
        "groups": {
            name: {**answer_stats(rows), **answer_lengths(rows, tokenizer)}
            for name, rows in train.items()
        },
        "sources": {
            source: source_stats(rows, tokenizer) for source, rows in sorted(by_source_rows.items())
        },
        "categories_by_language": {key: dict(value) for key, value in by_cell.items()},
        "sources_by_language": {key: dict(value) for key, value in by_source.items()},
        "leakage": leakage,
    }


def main() -> int:
    parser = argparse.ArgumentParser(prog="audit_v4_dialogues")
    parser.add_argument("--sft-groups", type=Path, default=Path("data/v4/sft_final/sft_groups"))
    parser.add_argument("--protected", type=Path, default=Path("configs/natural_eval_v4.jsonl"))
    parser.add_argument("--tokenizer", default="data/tokenizer")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.sft_groups, args.protected, BPETokenizer.load(args.tokenizer))
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(report, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
