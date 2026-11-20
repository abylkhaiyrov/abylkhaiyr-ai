from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np

from llm.data.binary import IGNORE_INDEX, read_jsonl, token_dtype
from llm.data.sft_quality import normalize
from llm.evaluation.dialogue_v4 import PROBES, benchmark_cases
from llm.tokenizer.bpe import BPETokenizer


def audit(root: Path, tokenizer: BPETokenizer) -> dict[str, object]:
    counters: Counter[str] = Counter()
    split_prompts: dict[str, set[str]] = {"train": set(), "validation": set(), "test": set()}
    histories: dict[str, set[tuple[str, ...]]] = {split: set() for split in split_prompts}
    answer_lengths: list[int] = []
    for source in sorted(root.glob("*.jsonl")):
        if source.name.endswith(".grounded.jsonl"):
            continue
        split = source.name.split(".", 1)[0]
        if split not in split_prompts:
            continue
        prefix = source.with_suffix("")
        ids = np.fromfile(f"{prefix}.ids.bin", dtype=token_dtype(tokenizer.vocab_size))
        labels = np.fromfile(f"{prefix}.labels.bin", dtype=np.int32)
        lengths = np.load(f"{prefix}.lengths.npy", allow_pickle=False)
        if len(ids) != len(labels) or int(lengths.sum()) != len(ids):
            raise ValueError(f"label or length count mismatch: {source}")
        offset = 0
        for length, row in zip(lengths, read_jsonl(source), strict=True):
            messages = row["messages"]
            histories[split].add(tuple(normalize(message["content"]) for message in messages))
            counters[f"{split}_rows"] += 1
            counters[f"{split}_multi_turn"] += len(messages) > 2
            counters["tokens"] += int(length)
            offset_next = offset + int(length)
            conversation = ids[offset:offset_next]
            actual = labels[offset:offset_next]
            offsets = np.flatnonzero(conversation == tokenizer.eos_id)
            pos = 1
            if conversation[0] != tokenizer.bos_id or actual[0] != IGNORE_INDEX:
                raise ValueError(f"BOS label mismatch: {source}")
            for eos, message in zip(offsets, messages, strict=True):
                role = message["role"]
                if conversation[pos] != tokenizer.role_ids[role] or actual[pos] != IGNORE_INDEX:
                    raise ValueError(f"role marker audit failed: {source}")
                if role == "assistant":
                    answer_lengths.append(int(eos) - pos - 1)
                    if np.any(actual[pos + 1 : eos + 1] != conversation[pos + 1 : eos + 1]):
                        raise ValueError(f"assistant labels mismatch: {source}")
                    counters["supervised"] += int(eos) - pos
                elif np.any(actual[pos + 1 : eos + 1] != IGNORE_INDEX):
                    raise ValueError(f"non-assistant labels are not masked: {source}")
                if role == "user":
                    split_prompts[split].add(normalize(message["content"]))
                pos = int(eos) + 1
            if pos != length:
                raise ValueError(f"missing final EOS: {source}")
            offset = offset_next
    reserved = {
        normalize(text)
        for case in benchmark_cases()
        for text in (case.prompt, *(question for question, _ in case.history))
    }
    probe_prompts = {
        normalize(message.content)
        for conversation in PROBES.values()
        for message in conversation
        if message.role == "user"
    }
    return {
        **dict(counters),
        "train_validation_prompt_overlap": len(
            split_prompts["train"] & split_prompts["validation"]
        ),
        "train_test_prompt_overlap": len(split_prompts["train"] & split_prompts["test"]),
        "train_validation_dialogue_overlap": len(histories["train"] & histories["validation"]),
        "train_benchmark_prompt_overlap": len(split_prompts["train"] & reserved),
        "train_open_probe_overlap": len(split_prompts["train"] & probe_prompts),
        "assistant_length_quantiles": dict(
            zip(
                ("p10", "p50", "p90"),
                np.percentile(answer_lengths, [10, 50, 90]).tolist(),
                strict=True,
            )
        ),
    }


def main() -> None:
    tokenizer = BPETokenizer.load("data/tokenizer")
    output = {
        name: audit(Path(f"data/{name}/sft_groups"), tokenizer)
        for name in ("gpu_chat", "dialogue_v4")
    }
    report = json.dumps(output, indent=2)
    Path("experiments/dialogue_v4/label_audit.json").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
