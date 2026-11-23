from __future__ import annotations

import argparse
import json
import math
import zlib
from collections import Counter
from pathlib import Path

import numpy as np

from llm.data.binary import read_jsonl, token_dtype

LANGUAGES = ("en", "ru", "kk")
SAMPLE_TOKENS = 2_000_000
BUCKETS = 512
CLUSTERS = 20


def entropy(counts: np.ndarray) -> float:
    probabilities = counts[counts > 0] / counts.sum()
    return float(-(probabilities * np.log2(probabilities)).sum())


def distinct_ngrams(tokens: np.ndarray, size: int) -> float:
    grams = np.lib.stride_tricks.sliding_window_view(tokens.astype(np.int64), size)
    keys = np.zeros(len(grams), dtype=np.int64)
    for column in range(size):
        keys = keys * 8209 + grams[:, column]
    return float(len(np.unique(keys)) / len(keys))


def stream_statistics(root: Path, vocab: int) -> dict:
    dtype = token_dtype(vocab)
    result: dict = {}
    for language in LANGUAGES:
        tokens = np.memmap(root / f"train.{language}.bin", dtype=dtype, mode="r")
        counts = np.bincount(tokens, minlength=vocab)
        start = len(tokens) // 4
        window = np.asarray(tokens[start : start + min(SAMPLE_TOKENS, len(tokens) - start)])
        result[language] = {
            "tokens": len(tokens),
            "distinct_ids": int((counts > 0).sum()),
            "unigram_entropy_bits": entropy(counts),
            "distinct_4gram_share": distinct_ngrams(window, 4),
            "distinct_8gram_share": distinct_ngrams(window, 8),
        }
    return result


def hashed_vectors(texts: list[str]) -> np.ndarray:
    matrix = np.zeros((len(texts), BUCKETS), dtype=np.float32)
    for row, text in enumerate(texts):
        for word in text[:3000].lower().split():
            matrix[row, zlib.crc32(word.encode()) % BUCKETS] += 1.0
    matrix = np.log1p(matrix)
    return matrix / np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-9)


def topic_clusters(
    vectors: np.ndarray, centers: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    generator = np.random.default_rng(0)
    centers = (
        vectors[generator.choice(len(vectors), CLUSTERS, replace=False)]
        if centers is None
        else centers
    )
    for _ in range(15):
        assignment = (vectors @ centers.T).argmax(axis=1)
        for index in range(CLUSTERS):
            members = vectors[assignment == index]
            if len(members):
                centers[index] = members.mean(axis=0) / max(
                    np.linalg.norm(members.mean(axis=0)), 1e-9
                )
    return assignment, centers


def document_statistics(path: Path) -> dict:
    rows = list(read_jsonl(path))
    characters = np.asarray([len(row["text"]) for row in rows])
    by_source: Counter[str] = Counter()
    by_category: Counter[str] = Counter()
    documents: Counter[str] = Counter(row["source"] for row in rows)
    for row in rows:
        by_source[row["source"]] += len(row["text"])
        by_category[row["category"]] += len(row["text"])
    total = sum(by_source.values())
    paragraphs = [
        paragraph
        for row in rows
        for paragraph in row["text"].split("\n\n")
        if len(paragraph.split()) >= 10
    ]
    kazakh = [len(row["text"]) for row in rows if row["language"] == "kk"]
    return {
        "documents": len(rows),
        "unique_texts": len({row["text"] for row in rows}),
        "unique_urls": len({row["url"] for row in rows}),
        "unique_paragraphs": len(set(paragraphs)),
        "paragraphs": len(paragraphs),
        "characters": int(characters.sum()),
        "median_characters": int(np.median(characters)),
        "share_under_500_characters": float((characters < 500).mean()),
        "kazakh_median_characters": int(np.median(kazakh)),
        "kazakh_share_under_500_characters": float(np.mean([size < 500 for size in kazakh])),
        "source_character_share": {
            key: round(value / total, 4) for key, value in by_source.most_common()
        },
        "source_entropy_bits": entropy(np.asarray(list(by_source.values()), dtype=float)),
        "category_entropy_bits": entropy(np.asarray(list(by_category.values()), dtype=float)),
        "documents_by_source": dict(documents),
        "oasst_character_share": by_source["oasst_train"] / total,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path("data/stage18")
    report: dict = {}
    rows = {arm: list(read_jsonl(root / arm / "train.jsonl")) for arm in ("a", "b")}
    vectors = {arm: hashed_vectors([row["text"] for row in rows[arm]]) for arm in rows}
    _, centers = topic_clusters(np.concatenate(list(vectors.values())))
    for arm in ("a", "b"):
        assignment, _ = topic_clusters(vectors[arm], centers.copy())
        counts = np.bincount(assignment, minlength=CLUSTERS)
        report[arm] = {
            "streams": stream_statistics(root / arm / "tokens", 8192),
            "documents": document_statistics(root / arm / "train.jsonl"),
            "topic_cluster_entropy_bits": entropy(counts.astype(float)),
            "topic_cluster_max_bits": math.log2(CLUSTERS),
            "largest_topic_cluster_share": int(np.max(counts)) / int(np.sum(counts)),
            "empty_topic_clusters": int((counts == 0).sum()),
        }
    processed = 4_997_120
    for arm in ("a", "b"):
        total = sum(item["tokens"] for item in report[arm]["streams"].values())
        report[arm]["processed_tokens"] = processed
        report[arm]["corpus_tokens"] = total
        report[arm]["coverage_of_corpus_by_one_run"] = processed / total
    args.output.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    print(f"written {args.output}")


if __name__ == "__main__":
    main()
