from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from llm.data.binary import read_jsonl
from llm.data.dedup import stable_hash
from llm.data.v4_corpus import build_language_filter
from llm.data.v4_dialogues import Dialogue, oasst_dialogues, support_dialogues
from llm.data.v4_sources import fetch, load_sources
from llm.data.v4_splits import legacy_prompts, normalized
from llm.tokenizer.bpe import BPETokenizer

HAND_WRITTEN = (Path("configs/natural_eval_v4.jsonl"), Path("configs/natural_eval_v5.jsonl"))


def dialogue_case(row: Dialogue) -> dict:
    return {
        "id": row.key,
        "source": row.source,
        "language": row.language,
        "category": "support" if row.source == "kk_support" else row.category,
        "kind": "free",
        "accept": None,
        "reject": [],
        "messages": [{"role": item.role, "content": item.content} for item in row.messages],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sources", type=Path, default=Path("configs/data_v4_sources.json"))
    parser.add_argument("--inputs", type=Path, default=Path("data/v4_inputs"))
    parser.add_argument("--tokenizer", type=Path, default=Path("data/tokenizer"))
    parser.add_argument("--train", nargs="*", type=Path, default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"output already exists: {args.output}")
    sources = load_sources(args.sources)
    needed = {
        name: sources[name] for name in ("oasst_val", "kk_support", "wiki_ru", "wiki_kk", "wiki_ky")
    }
    for source in needed.values():
        fetch(source, args.inputs)
    quality = build_language_filter(sources, args.inputs)
    tokenizer = BPETokenizer.load(args.tokenizer)
    cases = [row for path in HAND_WRITTEN for row in read_jsonl(path)]
    cases += [
        dialogue_case(row)
        for row in oasst_dialogues(
            "oasst_val", needed["oasst_val"].local(args.inputs), quality, tokenizer
        )
    ]
    cases += [
        dialogue_case(row)
        for row in support_dialogues(needed["kk_support"].local(args.inputs), quality, tokenizer)
    ]
    cases.sort(key=lambda case: stable_hash(case["id"]))
    if len({case["id"] for case in cases}) != len(cases):
        raise ValueError("duplicate validation IDs")
    train_prompts = legacy_prompts(
        path for root in args.train for path in sorted(root.glob("train.*.jsonl"))
    )
    overlap = sorted(
        case["id"]
        for case in cases
        if any(
            stable_hash(normalized(turn["content"])) in train_prompts
            for turn in case["messages"]
            if turn["role"] == "user"
        )
    )
    args.output.mkdir(parents=True)
    target = args.output / "validation.jsonl"
    with target.open("w", encoding="utf-8") as handle:
        for case in cases:
            handle.write(json.dumps(case, ensure_ascii=False) + "\n")
    report = {
        "cases": len(cases),
        "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
        "by_language": dict(Counter(case["language"] for case in cases)),
        "by_source": dict(Counter(case["source"] for case in cases)),
        "by_category": dict(sorted(Counter(case["category"] for case in cases).items())),
        "single_answer": sum(case["kind"] == "single" for case in cases),
        "prompts_found_in_training": len(overlap),
    }
    (args.output / "contaminated_ids.json").write_text(json.dumps(overlap, indent=1) + "\n")
    (args.output / "manifest.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    )
    print(json.dumps(report, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
