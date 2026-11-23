from __future__ import annotations

import argparse
import json
import subprocess
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator
from pathlib import Path
from tempfile import NamedTemporaryFile

from llm.data.binary import read_jsonl
from llm.data.corpus_overlap import HoldoutIndex
from llm.data.corpus_quality_study import read_documents
from llm.data.v4_corpus import Document

MIN_FLANK = 8


def texts(root: Path) -> Iterator[Document]:
    for row in read_jsonl(root / "diagnostic.jsonl"):
        yield Document(row["prefix"], "original_diagnostic", row["id"], "", row["language"], "eval")
    for row in read_jsonl(root / "free_cases.jsonl"):
        for index, message in enumerate(row["messages"]):
            if message["role"] == "user":
                yield Document(
                    message["content"],
                    "original_generation",
                    f"{row['id']}/{index}",
                    "",
                    row["language"],
                    "eval",
                )


def matches(index: HoldoutIndex, candidates: Iterable[Document]) -> dict[str, int]:
    report: Counter[str] = Counter()
    for row in candidates:
        reasons = index.matches(row)
        if reasons:
            report["documents"] += 1
            report.update(reasons)
    return dict(report)


def anchors(documents: Iterable[Document], *, stride: int = 1) -> dict[str, set[str]]:
    by_phrase: dict[str, set[str]] = defaultdict(set)
    for row in documents:
        words = row.text.split()
        for index in range(0, len(words) - MIN_FLANK + 1, stride):
            phrase = " ".join(words[index : index + MIN_FLANK]).casefold()
            if '" ' in phrase or " \\" in phrase or "\\" in phrase:
                continue
            by_phrase[phrase].add(f"{row.source}/{row.source_id}")
    return dict(by_phrase)


def find_historical(phrases: dict[str, set[str]], paths: Iterable[Path]) -> dict:
    result = {}
    with NamedTemporaryFile(mode="w", encoding="utf-8", suffix=".txt") as patterns:
        patterns.write("\n".join(sorted(phrases)) + "\n")
        patterns.flush()
        for path in paths:
            scan = subprocess.run(
                ["rg", "-ioF", "-f", patterns.name, "--", str(path)],
                capture_output=True,
                text=True,
                check=False,
            )
            if scan.returncode > 1:
                raise ValueError(f"historical search failed for {path}: {scan.stderr}")
            affected = {
                key for line in scan.stdout.splitlines() for key in phrases.get(line.casefold(), ())
            }
            result[str(path)] = sorted(affected)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--historical", action="store_true")
    args = parser.parse_args()
    root = Path("experiments/stage18")
    corpus = Path("data/stage18")
    passages = [
        row
        for split in ("validation", "test")
        for row in read_documents(corpus / "a" / f"{split}.jsonl")
    ]
    diagnostic = list(texts(root))
    original_anchors = anchors(diagnostic)
    if args.historical:
        phrases = anchors(passages, stride=4)
        for key, values in original_anchors.items():
            phrases.setdefault(key, set()).update(values)
        requested = [
            Path("data/processed/corpus/train.jsonl"),
            Path("data/v2/corpus/train.jsonl"),
            *sorted(Path("data/v3_1/sft_groups").glob("train.*.jsonl")),
        ]
        scanned = [path for path in requested if path.is_file()]
        result = {
            "historical_matches": find_historical(phrases, scanned),
            "missing": [str(path) for path in requested if path not in scanned],
        }
        output = "historical_overlap.json"
    else:
        index = HoldoutIndex([*passages, *diagnostic])
        result = {}
        for arm in ("a", "b"):
            candidate = corpus / arm / "train.jsonl"
            result[arm] = matches(index, read_documents(candidate))
        result["short_original_anchors"] = find_historical(
            original_anchors, (corpus / arm / "train.jsonl" for arm in ("a", "b"))
        )
        output = "verified_overlap.json"
    (root / output).write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
