from __future__ import annotations

import argparse
import json
from pathlib import Path

from llm.data.dialogue_pretraining import StreamFormat, write_dialogue_stream
from llm.tokenizer.bpe import BPETokenizer

DEFAULT_FILES = (
    "data/v5/sft_nodefs/sft_groups/train.dialogue_en.jsonl",
    "data/v5/sft_nodefs/sft_groups/train.dialogue_ru.jsonl",
    "data/v5/sft_nodefs/sft_groups/train.semantic_en.jsonl",
    "data/v5/sft_nodefs/sft_groups/train.semantic_ru.jsonl",
    "data/v5/sft_defs/sft_groups/train.dialogue_kk.jsonl",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--format", type=StreamFormat, choices=list(StreamFormat), required=True)
    parser.add_argument(
        "--files", nargs="*", type=Path, default=[Path(item) for item in DEFAULT_FILES]
    )
    parser.add_argument("--base-tokens", type=Path, default=Path("data/v4/tokens"))
    parser.add_argument("--tokenizer", type=Path, default=Path("data/tokenizer"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"output already exists: {args.output}")
    tokenizer = BPETokenizer.load(args.tokenizer)
    tokens = args.output / "tokens"
    tokens.mkdir(parents=True)
    for source in args.base_tokens.glob("*.bin"):
        (tokens / source.name).symlink_to(source.resolve())
    count = write_dialogue_stream(args.files, tokens / "train.dialog.bin", tokenizer, args.format)
    report = {
        "format": args.format.value,
        "tokens": count,
        "files": [str(path) for path in args.files],
        "tokenizer": tokenizer.version,
    }
    (args.output / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
