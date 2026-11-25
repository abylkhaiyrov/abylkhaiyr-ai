from __future__ import annotations

import argparse
import json
from pathlib import Path

from llm.data.verified_contrast import contrastive_pairs, write_pairs
from llm.data.verified_corpus import build_corpus, verified_rows
from llm.tokenizer.bpe import BPETokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("data/verified_v1"))
    parser.add_argument("--tokenizer", type=Path, default=Path("data/tokenizer"))
    parser.add_argument(
        "--benchmark", type=Path, default=Path("configs/acceptance/benchmark_v1.json")
    )
    parser.add_argument("--version", default="verified-v1")
    parser.add_argument("--pairs", action="store_true")
    args = parser.parse_args()
    tokenizer = BPETokenizer.load(args.tokenizer)
    manifest = build_corpus(args.output, tokenizer, args.benchmark, args.version)
    summary = {key: manifest[key] for key in ("collected", "kept", "rejected")}
    if args.pairs:
        rows, _ = verified_rows(args.benchmark)
        summary["pairs"] = write_pairs(contrastive_pairs(rows), args.output, tokenizer)
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
