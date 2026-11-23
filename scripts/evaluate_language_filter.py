from __future__ import annotations

import argparse
import json
import re
from collections.abc import Iterator
from pathlib import Path

from llm.data.binary import read_jsonl
from llm.data.v4_corpus import build_language_filter, wiki_rows
from llm.data.v4_quality import CyrillicLanguageFilter
from llm.data.v4_sources import load_sources

SENTENCE = re.compile(r"(?<=[.!?])\s+")
LATIN_WORDS = ("Python", "Wi-Fi", "iPhone", "email", "Google")
HELD_OUT_START, HELD_OUT_COUNT = 1200, 500


def first_sentences(rows: Iterator[dict], limit: int) -> list[str]:
    result = []
    for row in rows:
        sentences = [item for item in SENTENCE.split(row["text"].strip()) if 20 <= len(item) <= 120]
        if sentences:
            result.append(sentences[0])
        if len(result) == limit:
            break
    return result


def with_latin(texts: list[str]) -> list[str]:
    return [f"{text} {LATIN_WORDS[index % len(LATIN_WORDS)]}" for index, text in enumerate(texts)]


def rate(quality: CyrillicLanguageFilter, texts: list[str], expected: str, short: bool) -> float:
    return sum(quality.accepts(text, expected, allow_short=short) for text in texts) / len(texts)


def main() -> int:
    parser = argparse.ArgumentParser(prog="evaluate_language_filter")
    parser.add_argument("--sources", type=Path, default=Path("configs/data_v4_sources.json"))
    parser.add_argument("--inputs", type=Path, default=Path("data/v4_inputs"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sources = load_sources(args.sources)
    quality = build_language_filter(sources, args.inputs)
    held = {
        lang: list(wiki_rows(sources[f"wiki_{lang}"], args.inputs, HELD_OUT_START, HELD_OUT_COUNT))
        for lang in ("ru", "kk", "ky")
    }
    pools: dict[str, dict[str, list[str]]] = {
        "article": {lang: [row["text"][:1600] for row in rows] for lang, rows in held.items()},
        "sentence": {lang: first_sentences(iter(rows), 300) for lang, rows in held.items()},
    }
    for lang in ("ru", "kk"):
        questions = [row["question"] for row in read_jsonl(args.inputs / f"belebele_{lang}.jsonl")]
        pools["question"] = {**pools.get("question", {}), lang: questions}
    pools["question"]["kk"] += [
        row["question"] for row in read_jsonl(args.inputs / "kk_support.jsonl")
    ]
    pools["short_phrase"] = {
        lang: [" ".join(text.split()[:4]) for text in pools["sentence"][lang]]
        for lang in ("ru", "kk", "ky")
    }
    pools["latin_mixed"] = {
        lang: with_latin(pools["sentence"][lang]) for lang in ("ru", "kk", "ky")
    }
    report: dict[str, dict] = {}
    for pool, languages in pools.items():
        for short in (False, True):
            if pool == "article" and short:
                continue
            cell = {}
            for expected in ("ru", "kk"):
                cell[f"accept_as_{expected}"] = {
                    truth: round(rate(quality, texts, expected, short), 3)
                    for truth, texts in languages.items()
                }
            report[f"{pool}.allow_short={short}"] = {
                "n": {truth: len(texts) for truth, texts in languages.items()},
                **cell,
            }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
