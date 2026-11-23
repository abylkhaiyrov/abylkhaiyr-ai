from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import asdict
from itertools import chain
from pathlib import Path
from tempfile import TemporaryDirectory

from llm.core.checksums import file_sha256
from llm.data.corpus_overlap import HoldoutIndex, paragraphs
from llm.data.corpus_quality_study import (
    balanced,
    clean_candidates,
    encoded_counts,
    human_dialogues,
    passage_holdouts,
    read_documents,
)
from llm.data.dedup import stable_hash
from llm.data.v4_corpus import Document, build_language_filter, write_tokens
from llm.data.v4_sources import load_sources
from llm.tokenizer.bpe import BPETokenizer


def write_rows(target: Path, rows: list[Document]) -> None:
    with target.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(asdict(row), ensure_ascii=False) + "\n")


def hashes(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)): file_sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--origin", type=Path, default=Path("data/v4_expanded"))
    parser.add_argument("--inputs", type=Path, default=Path("data/v4_inputs"))
    parser.add_argument("--tokenizer", type=Path, default=Path("data/tokenizer"))
    parser.add_argument("--target", type=Path, default=Path("data/stage18"))
    args = parser.parse_args()
    if args.target.exists():
        parser.error("corpus already exists")
    tokenizer = BPETokenizer.load(args.tokenizer)
    sources = load_sources(Path("configs/data_v4_sources.json"))
    historical = json.loads((args.origin / "manifest.json").read_text(encoding="utf-8"))
    required = set(historical["sources"]) | {
        "oasst_train",
        "belebele_en",
        "belebele_ru",
        "belebele_kk",
    }
    provenance = {}
    for key in sorted(required):
        source = sources[key]
        if file_sha256(source.local(args.inputs)) != source.sha256:
            parser.error(f"source checksum mismatch: {key}")
        provenance[key] = {
            "revision": source.revision,
            "license": source.license,
            "url": source.url,
            "license_url": source.license_url,
            "sha256": source.sha256,
        }

    old_splits = (args.origin / f"{split}.jsonl" for split in ("validation", "test"))
    holdout = passage_holdouts(sources, args.inputs)
    validation_index = HoldoutIndex(holdout["validation"])
    original_test = len(holdout["test"])
    holdout["test"] = [row for row in holdout["test"] if not validation_index.matches(row)]
    del validation_index
    excluded = HoldoutIndex(
        chain.from_iterable(chain(map(read_documents, old_splits), holdout.values()))
    )
    seal = {"holdout_test_removed": original_test - len(holdout["test"]), "sources": provenance}
    args.target.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=args.target.parent, prefix=".stage18-") as staging:
        corpus_root = Path(staging) / "corpus"
        corpus_root.mkdir()
        for label in ("a", "b"):
            counts: Counter[str] = Counter()
            candidates = read_documents(args.origin / "train.jsonl")
            if label == "b":
                language_filter = build_language_filter(sources, args.inputs)
                extra = human_dialogues(
                    sources["oasst_train"], args.inputs, language_filter, tokenizer
                )
                candidates = chain(candidates, extra)
            encoded = encoded_counts(
                list(clean_candidates(candidates, excluded, counts, curated=label == "b")),
                tokenizer,
            )
            available: Counter[str] = Counter()
            for item in encoded:
                available[f"{item.row.language}/{item.row.category}"] += item.tokens
            selected = balanced(encoded) if label == "b" else encoded
            train = [item.row for item in selected]
            target = corpus_root / label
            target.mkdir()
            write_rows(target / "train.jsonl", train)
            for split, rows in holdout.items():
                write_rows(target / f"{split}.jsonl", rows)
            tokens_by_cell: Counter[str] = Counter()
            docs_by_source: Counter[str] = Counter()
            unique_paragraphs: set[int] = set()
            paragraph_count = 0
            for item in selected:
                row = item.row
                tokens_by_cell[f"{row.language}/{row.category}"] += item.tokens
                docs_by_source[row.source] += 1
                for block in paragraphs(row.text):
                    unique_paragraphs.add(stable_hash(block))
                    paragraph_count += 1
            seal[label] = {
                "documents": len(train),
                "unique_documents": len(train),
                "tokens": write_tokens(target, tokenizer),
                "candidate_tokens_by_language_category": available,
                "selected_tokens_by_language_category": tokens_by_cell,
                "documents_by_source": docs_by_source,
                "unique_paragraphs": len(unique_paragraphs),
                "paragraph_occurrences": paragraph_count,
                "rejected": counts,
                "sha256": hashes(target),
            }
            print(
                f"{label}: {len(train)} documents; {sum(tokens_by_cell.values())} tokens",
                flush=True,
            )
        seal["tokenizer"] = tokenizer.version
        seal["tokenizer_sha256"] = file_sha256(args.tokenizer / "tokenizer.json")
        (corpus_root / "manifest.json").write_text(
            json.dumps(seal, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        corpus_root.rename(args.target)


if __name__ == "__main__":
    main()
