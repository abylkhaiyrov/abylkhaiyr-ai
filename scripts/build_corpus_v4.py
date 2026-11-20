from __future__ import annotations

import argparse
import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from itertools import chain
from pathlib import Path
from tempfile import TemporaryDirectory

from llm.data.v4_corpus import (
    build_language_filter,
    collect_curated,
    collect_wiki,
    source_manifest,
    write_splits,
    write_tokens,
)
from llm.data.v4_sources import fetch, load_sources
from llm.tokenizer.bpe import BPETokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sources", type=Path, default=Path("configs/data_v4_sources.json"))
    parser.add_argument("--inputs", type=Path, default=Path("data/v4_inputs"))
    parser.add_argument("--output", type=Path, default=Path("data/v4"))
    parser.add_argument("--tokenizer", type=Path, default=Path("data/tokenizer"))
    parser.add_argument("--expanded", action="store_true")
    args = parser.parse_args()
    sources = {
        name: source
        for name, source in load_sources(args.sources).items()
        if name.startswith("wiki_")
        or (args.expanded and source.vendor == "github" and source.category != "tokenizer_holdout")
    }
    if args.output.exists():
        parser.error(f"output already exists: {args.output}")
    with ThreadPoolExecutor(max_workers=5) as pool:
        list(pool.map(partial(fetch, root=args.inputs), sources.values()))
    tokenizer = BPETokenizer.load(args.tokenizer)
    quality = build_language_filter(sources, args.inputs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=args.output.parent) as staging:
        root = Path(staging) / "corpus"
        root.mkdir()
        rejected: Counter[str] = Counter()
        counts, duplicates = write_splits(
            chain(
                collect_wiki(sources, args.inputs, quality, rejected),
                collect_curated(sources, args.inputs, quality, rejected),
            ),
            root,
        )
        manifest = {
            "version": "v4-expanded" if args.expanded else "v4",
            "sources": source_manifest(sources, args.inputs),
            "documents": counts,
            "rejected": rejected,
            "duplicates": duplicates,
            "tokenizer": tokenizer.version,
            "tokens": write_tokens(root, tokenizer),
        }
        report = json.dumps(manifest, indent=2, sort_keys=True)
        (root / "manifest.json").write_text(report + "\n", encoding="utf-8")
        root.rename(args.output)
        print(report)


if __name__ == "__main__":
    main()
