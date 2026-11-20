from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from llm.data.binary import read_jsonl
from llm.data.dedup import stable_hash
from llm.data.v4_corpus import build_language_filter, wiki_rows
from llm.data.v4_quality import clean_v4_text
from llm.data.v4_sources import fetch, load_sources


def article_key(text: str) -> int:
    return stable_hash(" ".join(text.casefold().split()))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, default=Path("data/v4"))
    parser.add_argument("--sources", type=Path, default=Path("configs/data_v4_sources.json"))
    parser.add_argument("--inputs", type=Path, default=Path("data/v4_inputs"))
    parser.add_argument("--legacy", type=Path, default=Path("data/v2/corpus"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sources = load_sources(args.sources)
    for language in ("ru", "kk", "ky"):
        fetch(sources[f"wiki_{language}"], args.inputs)
    language_filter = build_language_filter(sources, args.inputs)
    confusion: dict[str, Counter] = {}
    for language in ("ru", "kk", "ky"):
        counter: Counter[str] = Counter()
        samples = wiki_rows(sources[f"wiki_{language}"], args.inputs, 70000, 400)
        for sample in samples:
            counter["sampled"] += 1
            text = clean_v4_text(sample["text"], 180)
            if text is None:
                counter["quality_rejected"] += 1
                continue
            for target in ("ru", "kk"):
                counter[target] += language_filter.accepts(text, target)
        confusion[language] = counter
    rows = list(read_jsonl(args.corpus / "validation.jsonl"))
    validation = {article_key(row["text"]): (row["language"], row["source_id"]) for row in rows}
    matches = set()
    scanned = 0
    for name in ("train", "validation", "test"):
        for row in read_jsonl(args.legacy / f"{name}.jsonl"):
            scanned += 1
            text = row["text"]
            key = article_key(text)
            if key in validation:
                matches.add(key)
    result = json.dumps(
        {
            "language_check": confusion,
            "legacy_documents": scanned,
            "validation_documents": len(rows),
            "exact_legacy_overlap": sorted(validation[key] for key in matches),
        },
        indent=2,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(result + "\n", encoding="utf-8")
    print(result)


if __name__ == "__main__":
    main()
