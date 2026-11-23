from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import numpy as np

from llm.core.checksums import file_sha256
from llm.data.binary import read_jsonl, token_dtype
from llm.data.dedup import stable_hash
from llm.data.v4_quality import CORRUPTION
from llm.evaluation.diagnostic_bank import items
from llm.tokenizer.bpe import BPETokenizer

WORDS = re.compile(r"[^\W_]+|[+%*/×÷\[\]-]")


def canonical(text: str) -> str:
    return " ".join(WORDS.findall(text.casefold()))


def audit_items(
    rows: list[dict], tokenizer: BPETokenizer, train: Path, *, strict: bool = True
) -> dict:
    choices: dict[str, Counter[int]] = {}
    prompts: dict[str, str] = {}
    shortest = longest = equal_length = rank_shortcut = 0
    duplicates = []
    for row in rows:
        key = canonical(row["prefix"])
        if key in prompts:
            duplicates.append([prompts[key], row["id"]])
            if strict:
                raise ValueError(f"duplicate prompts: {prompts[key]} and {row['id']}")
        prompts[key] = row["id"]
        options = row["options"]
        answer = row["answer"]
        if len(set(map(canonical, options))) != 4 or answer not in range(4):
            raise ValueError(f"invalid alternatives: {row['id']}")
        ids = tokenizer.encode_batch([row.get("separator", " ") + option for option in options])
        lengths = [len(option) for option in ids]
        if min(lengths) < 1 or len(set(map(tuple, ids))) != 4:
            raise ValueError(f"invalid tokenized alternatives: {row['id']}")
        shortest += int(lengths[answer] == min(lengths) and lengths.count(min(lengths)) == 1)
        longest += int(lengths[answer] == max(lengths) and lengths.count(max(lengths)) == 1)
        equal_length += int(len(set(lengths)) == 1)
        if all(option.isdecimal() for option in options):
            best = sorted(range(4), key=lambda j: int(options[j]))[1]
            rank_shortcut += best == answer
        cell = f"{row['language']}/{row['task']}"
        choices.setdefault(cell, Counter())[answer] += 1
    hits = []
    prompt_hashes = {stable_hash(prompt): name for prompt, name in prompts.items()}
    windows = {len(prompt.split()) for prompt in prompts}
    for row in read_jsonl(train):
        words = canonical(row["text"]).split()
        if len(words) in windows:
            name = prompt_hashes.get(stable_hash(" ".join(words)))
            if name is not None:
                hits.append(name)
    return {
        "items": len(rows),
        "position_histograms": choices,
        "duplicate_prefixes": duplicates,
        "families": len({row.get("family", row["id"]) for row in rows}),
        "unique_shortest_gold": shortest,
        "unique_longest_gold": longest,
        "equal_token_lengths": equal_length,
        "second_smallest_numeric_gold": rank_shortcut,
        "exact_full_document_prompt_hits": hits,
    }


def audit_corpus(root: Path, tokenizer: BPETokenizer) -> dict:
    seen: dict[str, set[int]] = {}
    sources: dict[str, Counter[str]] = {}
    languages: dict[str, Counter[str]] = {}
    snapshot = {"tokenizer": tokenizer.version, "splits": {}}
    dtype = token_dtype(tokenizer.vocab_size)
    for split in ("train", "validation", "test"):
        seen[split] = set()
        sources[split] = Counter()
        languages[split] = Counter()
        duplicates = broken = documents = 0
        path = root / f"{split}.jsonl"
        for row in read_jsonl(path):
            normalized = canonical(row["text"])
            key = stable_hash(normalized)
            duplicates += key in seen[split]
            seen[split].add(key)
            broken += bool(CORRUPTION.search(row["text"]))
            sources[split][f"{row['source']}/{row['category']}"] += 1
            languages[split][row["language"]] += 1
            documents += 1
        binary = root / "tokens" / f"{split}.bin"
        tokens = np.memmap(binary, mode="r", dtype=dtype)
        frequencies = np.zeros(tokenizer.vocab_size, dtype=np.int64)
        for start in range(0, len(tokens), 1 << 20):
            frequencies += np.bincount(
                tokens[start : start + (1 << 20)], minlength=tokenizer.vocab_size
            )
        snapshot["splits"][split] = {
            "sha256": file_sha256(binary),
            "jsonl_bytes": path.stat().st_size,
            "stored_tokens": len(tokens),
            "distinct_token_ids": int(np.count_nonzero(frequencies)),
            "documents": documents,
            "exact_duplicate_documents": duplicates,
            "corruption_marker_documents": broken,
            "sources": sources[split],
            "languages": languages[split],
        }
    snapshot["exact_train_validation"] = len(seen["train"] & seen["validation"])
    snapshot["exact_train_test"] = len(seen["train"] & seen["test"])
    return snapshot


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("experiments/stage17"))
    parser.add_argument("--corpus", type=Path, default=Path("data/v4_expanded"))
    parser.add_argument("--tokenizer", type=Path, default=Path("data/tokenizer"))
    args = parser.parse_args()
    tokenizer = BPETokenizer.load(args.tokenizer)
    args.root.mkdir(parents=True, exist_ok=True)
    rows = [asdict(item) for item in items()]
    diagnostic = audit_items(rows, tokenizer, args.corpus / "train.jsonl")
    legacy = list(read_jsonl(Path("configs/diagnostic_v1.jsonl")))
    path = args.root / "diagnostic_v2.jsonl"
    serialized = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows)
    if path.exists():
        if path.read_text(encoding="utf-8") != serialized:
            parser.error("refusing to replace the frozen diagnostic bank")
    else:
        with path.open("x", encoding="utf-8") as handle:
            handle.write(serialized)
    report = {
        "bank_sha256": file_sha256(path),
        "v2": diagnostic,
        "v1": audit_items(legacy, tokenizer, args.corpus / "train.jsonl", strict=False),
        "corpus": audit_corpus(args.corpus, tokenizer),
    }
    (args.root / "preflight_data.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
