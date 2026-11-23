from __future__ import annotations

import argparse
import json
import statistics
import time
from collections.abc import Iterable
from itertools import islice
from pathlib import Path

import psutil

from llm.core.config import TokenizerConfig
from llm.core.profiles import build_model_config, count_parameters
from llm.data.binary import read_jsonl
from llm.data.v4_quality import LANGUAGES
from llm.data.v4_sources import fetch, load_sources
from llm.tokenizer.bpe import BPETokenizer


def training_texts(root: Path, documents: int) -> Iterable[str]:
    for language in LANGUAGES:
        rows = (row for row in read_jsonl(root / "train.jsonl") if row["language"] == language)
        for row in islice(rows, documents):
            yield row["text"][:4000]
    for row in read_jsonl(root / "train.jsonl"):
        if row["category"] == "code":
            yield row["text"]


def measure_tokenizer(tokenizer: BPETokenizer, samples: list[str]) -> dict:
    durations = []
    for _ in range(5):
        started = time.perf_counter()
        encoded = tokenizer.encode_batch(samples)
        durations.append(time.perf_counter() - started)
    bytes_used = sum(len(text.encode("utf-8")) for text in samples)
    characters = sum(len(text) for text in samples)
    tokens = sum(map(len, encoded))
    return {
        "samples": len(samples),
        "utf8_bytes": bytes_used,
        "tokens": tokens,
        "bytes_per_token": bytes_used / tokens,
        "characters_per_token": characters / tokens,
        "utf8_to_uint16_ratio": bytes_used / (2 * tokens),
        "encode_mib_per_second": bytes_used / 2**20 / statistics.median(durations),
        "lossless": all(
            tokenizer.decode(ids) == text for ids, text in zip(encoded, samples, strict=True)
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("data/v4_expanded"))
    parser.add_argument("--inputs", type=Path, default=Path("data/v4_inputs"))
    parser.add_argument("--sources", type=Path, default=Path("configs/data_v4_sources.json"))
    parser.add_argument("--documents-per-language", type=int, default=1000)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--vocab-sizes",
        type=int,
        nargs="+",
        choices=(8192, 16384, 32768),
        default=(8192, 16384, 32768),
    )
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    args = parser.parse_args()
    if args.documents_per_language < 1:
        parser.error("--documents-per-language must be positive")
    source = load_sources(args.sources)["python_graphlib"]
    fetch(source, args.inputs)
    samples = {
        language: [
            row["text"]
            for row in islice(
                (
                    row
                    for row in read_jsonl(args.root / f"{args.split}.jsonl")
                    if row["language"] == language
                ),
                100,
            )
        ]
        for language in LANGUAGES
    }
    samples["code"] = [source.local(args.inputs).read_text(encoding="utf-8")]
    output = []
    for requested in args.vocab_sizes:
        rss_before = psutil.Process().memory_info().rss
        started = time.perf_counter()
        tokenizer = BPETokenizer.train(
            training_texts(args.root, args.documents_per_language),
            TokenizerConfig(vocab_size=requested),
        )
        trained_seconds = time.perf_counter() - started
        rss_after = psutil.Process().memory_info().rss
        target = args.output.parent / f"bpe-{requested}"
        tokenizer.save(target)
        architectures = {}
        for profile in ("tiny", "small", "small_plus"):
            model = build_model_config(profile, tokenizer.vocab_size)
            embedding = model.hidden_size * tokenizer.vocab_size
            architectures[profile] = {
                "parameters": count_parameters(model),
                "embedding_parameters": embedding,
                "embedding_mib_fp32": 4 * embedding / 2**20,
                "forward_output_flops_per_token": 2 * embedding,
            }
        output.append(
            {
                "requested": requested,
                "actual": tokenizer.vocab_size,
                "version": tokenizer.version,
                "training_seconds": trained_seconds,
                "rss_before_training_mib": rss_before / 2**20,
                "rss_after_training_mib": rss_after / 2**20,
                "evaluation_split": args.split,
                "architectures": architectures,
                "samples": {
                    name: measure_tokenizer(tokenizer, texts) for name, texts in samples.items()
                },
            }
        )
    result = json.dumps(output, indent=2, allow_nan=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(result + "\n", encoding="utf-8")
    print(result)


if __name__ == "__main__":
    main()
