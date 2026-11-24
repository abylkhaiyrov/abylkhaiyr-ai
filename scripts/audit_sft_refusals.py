from __future__ import annotations

import argparse
import json
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from llm.core.config import ExperimentConfig, load_experiment_config
from llm.data.binary import IGNORE_INDEX, read_jsonl, with_extension
from llm.data.refusals import DETECTORS, KINDS, assistant_answers, phrasing_counts
from llm.tokenizer.bpe import BPETokenizer, Message
from llm.training.runner import _build_loaders
from llm.training.spec import load_spec


def encode(row: dict, tokenizer: BPETokenizer, masked: str | None) -> tuple[list[int], list[int]]:
    messages = []
    for turn in row["messages"]:
        hidden = (
            masked is not None
            and turn["role"] == "assistant"
            and DETECTORS[masked](row, turn["content"])
        )
        messages.append(Message(turn["role"], turn["content"], not hidden))
    return tokenizer.encode_conversation_with_labels(messages, IGNORE_INDEX)


def group_statistics(directory: Path, tokenizer: BPETokenizer) -> dict[str, Any]:
    report: dict[str, Any] = {}
    for path in sorted(directory.glob("train.*.jsonl")):
        name = path.name.split(".")[1]
        rows = list(read_jsonl(path))
        labels = np.memmap(
            with_extension(path.with_suffix(""), ".labels.bin"), dtype=np.int32, mode="r"
        )
        supervised = int((labels != IGNORE_INDEX).sum())
        entry: dict[str, Any] = {
            "examples": len(rows),
            "unique_examples": len(
                {json.dumps(row["messages"], ensure_ascii=False) for row in rows}
            ),
            "unique_answers": len({answer for row in rows for answer in assistant_answers(row)}),
            "supervised_tokens": supervised,
            "languages": dict(Counter(row.get("language", "n/a") for row in rows)),
            "kinds": {},
        }
        for kind in KINDS:
            flagged = [
                row for row in rows if any(DETECTORS[kind](row, a) for a in assistant_answers(row))
            ]
            if not flagged:
                continue
            tokens = 0
            for row in flagged:
                _, full = encode(row, tokenizer, None)
                _, hidden = encode(row, tokenizer, kind)
                tokens += sum(a != IGNORE_INDEX for a in full) - sum(
                    a != IGNORE_INDEX for a in hidden
                )
            templates = phrasing_counts(rows, kind)
            entry["kinds"][kind] = {
                "examples": len(flagged),
                "share_of_group_examples": len(flagged) / len(rows),
                "supervised_tokens": tokens,
                "share_of_group_supervised_tokens": tokens / supervised,
                "distinct_phrasings": len(templates),
                "top_phrasings": templates.most_common(4),
                "languages": dict(Counter(row.get("language", "n/a") for row in flagged)),
            }
        report[name] = entry
    return report


def exposed_tokens(
    config: ExperimentConfig, root: Path, tokenizer: BPETokenizer, steps: int
) -> int:
    config = config.model_copy(
        update={"train": config.train.model_copy(update={"data_dir": str(root)})}
    )
    source, _ = _build_loaders(config, tokenizer)
    total = 0
    for _ in range(steps * config.train.gradient_accumulation_steps):
        _, targets, *segments = source.next_batch()
        if segments:
            total += int((targets != IGNORE_INDEX).sum())
    return total


def masked_root(base: Path, scratch: Path, tokenizer: BPETokenizer, kind: str) -> Path:
    groups = scratch / "sft_groups"
    groups.mkdir(parents=True)
    rewritten: set[Path] = set()
    for path in sorted((base / "sft_groups").glob("train.*.jsonl")):
        rows = list(read_jsonl(path))
        if not any(any(DETECTORS[kind](row, a) for a in assistant_answers(row)) for row in rows):
            continue
        prefix = path.with_suffix("")
        ids, labels = [], []
        for row in rows:
            row_ids, row_labels = encode(row, tokenizer, kind)
            ids.extend(row_ids)
            labels.extend(row_labels)
        original = np.memmap(with_extension(prefix, ".ids.bin"), dtype=np.uint16, mode="r")
        if len(original) != len(ids) or not np.array_equal(
            original, np.asarray(ids, dtype=np.uint16)
        ):
            raise ValueError(f"token alignment differs for {path.name}")
        np.asarray(labels, dtype=np.int32).tofile(groups / f"{prefix.name}.labels.bin")
        rewritten.add(with_extension(Path(prefix.name), ".labels.bin"))
    for source in (base / "sft_groups").iterdir():
        target = groups / source.name
        if not target.exists():
            target.symlink_to(source.resolve())
    return scratch


def exposure(
    config: ExperimentConfig, base: Path, tokenizer: BPETokenizer, steps: int
) -> dict[str, Any]:
    total = exposed_tokens(config, base, tokenizer, steps)
    result: dict[str, Any] = {"assistant_tokens_seen": total, "steps": steps, "kinds": {}}
    for kind in KINDS:
        with tempfile.TemporaryDirectory() as scratch:
            hidden = exposed_tokens(
                config,
                masked_root(base, Path(scratch) / "root", tokenizer, kind),
                tokenizer,
                steps,
            )
        result["kinds"][kind] = {"tokens_seen": total - hidden, "share": (total - hidden) / total}
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/tiny_v31_sft_stage11.yaml"))
    parser.add_argument("--spec", type=Path)
    parser.add_argument("--job")
    parser.add_argument("--data", type=Path, default=Path("data/v3_1"))
    parser.add_argument("--tokenizer", type=Path, default=Path("data/tokenizer"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    tokenizer = BPETokenizer.load(args.tokenizer)
    config = (
        load_spec(args.spec).job_configuration(args.job)
        if args.spec
        else load_experiment_config(args.config)
    )
    report = {
        "groups": group_statistics(args.data / "sft_groups", tokenizer),
        "mixture": dict(config.train.mixture),
        "replay_fraction": config.train.sft_replay_fraction,
        "exposure": exposure(config, args.data, tokenizer, config.train.max_steps),
    }
    args.output.write_text(
        json.dumps(report, indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report["exposure"], indent=1))


if __name__ == "__main__":
    main()
