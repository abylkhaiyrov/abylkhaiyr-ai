from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from llm.data.binary import read_jsonl
from llm.data.cleaning import detect_language
from llm.data.dedup import stable_hash
from llm.tokenizer.bpe import BPETokenizer

LENGTH_PERCENTILES = (5, 25, 50, 75, 95, 99)
REPORT_FILE = "corpus_report.json"


def corpus_report(data_dir: str | Path, tokenizer: BPETokenizer) -> dict[str, object]:
    root = Path(data_dir)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    licenses = {name: info["license"] for name, info in manifest["sources"].items()}
    tokens: Counter[str] = Counter()
    documents: Counter[str] = Counter()
    languages: Counter[str] = Counter()
    lengths: dict[str, list[int]] = defaultdict(list)
    hashes: set[int] = set()
    for row in read_jsonl(root / "corpus" / "train.jsonl"):
        count = len(tokenizer.encode(row["text"]))
        source = row["source"]
        tokens[source] += count
        documents[source] += 1
        languages[row.get("language") or detect_language(row["text"][:2000])] += count
        lengths[source].append(count)
        hashes.add(stable_hash(row["text"]))
    total = sum(tokens.values())
    return {
        "total_tokens": total,
        "unique_documents": len(hashes),
        "documents": sum(documents.values()),
        "tokens_by_language": dict(languages),
        "sources": {
            name: {
                "tokens": tokens[name],
                "share": tokens[name] / total,
                "documents": documents[name],
                "license": licenses.get(name),
                "length_percentiles": {
                    str(p): float(np.percentile(lengths[name], p)) for p in LENGTH_PERCENTILES
                },
            }
            for name in tokens
        },
    }
