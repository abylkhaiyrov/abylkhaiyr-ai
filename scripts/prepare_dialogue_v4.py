from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from llm.data.binary import read_jsonl, token_dtype, tokenize_sft_split, write_token_stream
from llm.data.dedup import stable_hash
from llm.data.dialogues import MessageTree, conversations
from llm.data.sft_quality import PromptBlocklist, QualityRules, SFTQualityFilter, normalize
from llm.evaluation.dialogue_v4 import PROBES, benchmark_cases
from llm.evaluation.prompts import evaluation_prompts
from llm.tokenizer.bpe import BPETokenizer, Message

Row = dict[str, Any]
LANGUAGES = ("ru", "en", "kk")
GROUPS = (
    "dialog",
    "identity",
    "skills",
    "reading_ru",
    "reading_kk",
    "reading_en",
    "chat_en",
    "code",
)


def human_rows(path: Path) -> Iterator[tuple[str, str, Row]]:
    rows = []
    for batch in pq.ParquetFile(path).iter_batches(batch_size=2048):
        for row in batch.to_pylist():
            if row["deleted"] or row["synthetic"] or row["lang"] not in LANGUAGES:
                continue
            if row["review_result"] is False or row["tree_state"] != "ready_for_export":
                continue
            labels = row.get("labels") or {}
            scores = dict(zip(labels.get("name") or [], labels.get("value") or [], strict=True))
            if any(
                scores.get(key, 0) > 0.25
                for key in ("pii", "spam", "not_appropriate", "lang_mismatch")
            ):
                continue
            if row["role"] == "assistant" and scores.get("quality", 0.5) < 0.55:
                continue
            rows.append(row)
    tree = MessageTree(rows)
    for row in conversations(tree):
        question = row["messages"][0]["content"]
        split = "validation" if stable_hash(normalize(question)) % 20 == 0 else "train"
        yield split, f"human_{row['language']}", row


def candidates() -> Iterator[tuple[str, str, Row]]:
    yield from human_rows(Path("data/raw/oasst1_train.parquet"))
    for split in ("train", "validation"):
        for group in GROUPS:
            source = Path(f"data/gpu_chat/sft_groups/{split}.{group}.jsonl")
            if source.is_file():
                for row in read_jsonl(source):
                    yield split, group, row
    curated = Path("data/raw/dialogue_v4_reviewed.jsonl")
    if curated.is_file():
        for row in read_jsonl(curated):
            first = normalize(row["messages"][0]["content"])
            split = "validation" if stable_hash(first) % 8 == 0 else "train"
            yield split, "reviewed", row


def main() -> None:
    root = Path("data/dialogue_v4")
    groups = root / "sft_groups"
    groups.mkdir(parents=True, exist_ok=True)
    tokenizer = BPETokenizer.load("data/tokenizer")
    suite = benchmark_cases()
    reserved = {
        normalize(text)
        for case in suite
        for text in (case.prompt, *(question for question, _ in case.history))
    }
    reserved.update(
        normalize(message.content)
        for messages in PROBES.values()
        for message in messages
        if message.role == "user"
    )
    blocklist = PromptBlocklist(evaluation_prompts("data/benchmarks"))
    rows = sorted(candidates(), key=lambda item: item[0] != "validation")
    quality = SFTQualityFilter(QualityRules(max_tokens=384), tokenizer)
    assigned: dict[str, str] = {}
    seen: set[tuple[str, ...]] = set()
    output: dict[tuple[str, str], list[Row]] = {}
    rejected: Counter[str] = Counter()
    for split, group, row in rows:
        prompts = {normalize(msg["content"]) for msg in row["messages"] if msg["role"] == "user"}
        if prompts & reserved or blocklist.blocks(row):
            rejected["benchmark_overlap"] += 1
            continue
        if any(prompt in assigned and assigned[prompt] != split for prompt in prompts):
            rejected["cross_split"] += 1
            continue
        identity = tuple(normalize(msg["content"]) for msg in row["messages"])
        if identity in seen:
            rejected["duplicate"] += 1
            continue
        if (group.startswith("human_") or group == "reviewed") and not quality.admit(row):
            continue
        encoded = tokenizer.encode_conversation(
            [Message(m["role"], m["content"]) for m in row["messages"]]
        )
        if len(encoded) > 384:
            rejected["length"] += 1
            continue
        seen.add(identity)
        assigned.update(dict.fromkeys(prompts, split))
        output.setdefault((split, group), []).append(row)
    report: dict[str, Any] = {
        "accepted": {},
        "rejected": dict(rejected),
        "quality_rejected": dict(quality.rejected),
    }
    for (split, group), items in sorted(output.items()):
        prefix = groups / f"{split}.{group}"
        source = groups / f"{split}.{group}.jsonl"
        with source.open("w", encoding="utf-8") as handle:
            for row in items:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        count = tokenize_sft_split(source, prefix, tokenizer)
        report["accepted"][f"{split}.{group}"] = {
            "rows": count,
            "multi_turn": sum(len(row["messages"]) > 2 for row in items),
            "languages": dict(Counter(row.get("language", "unspecified") for row in items)),
        }
    replay = root / "replay"
    replay.mkdir(exist_ok=True)
    for language in LANGUAGES:
        texts = (
            row["text"]
            for row in read_jsonl(Path("data/processed/corpus/train.jsonl"))
            if row.get("language") == language
        )
        streams = (
            [tokenizer.bos_id, *tokens, tokenizer.eos_id]
            for tokens in tokenizer.encode_stream(texts)
        )
        report.setdefault("replay_tokens", {})[language] = write_token_stream(
            replay / f"train.{language}.bin", streams, token_dtype(tokenizer.vocab_size)
        )
    (root / "manifest.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
