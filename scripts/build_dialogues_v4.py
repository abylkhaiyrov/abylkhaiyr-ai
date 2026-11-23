from __future__ import annotations

import argparse
import json
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from itertools import chain
from pathlib import Path
from tempfile import TemporaryDirectory

from llm.data.v4_corpus import build_language_filter, source_manifest
from llm.data.v4_dialogues import (
    Dialogue,
    aya_dialogues,
    belebele_dialogues,
    curated_dialogues,
    dolly_dialogues,
    kazsandra_dialogues,
    oasst_dialogues,
    squad_dialogues,
    support_dialogues,
    wiki_definition_dialogues,
)
from llm.data.v4_quality import LANGUAGES
from llm.data.v4_sources import fetch, load_sources
from llm.data.v4_splits import (
    SEMANTIC,
    legacy_prompts,
    select_training,
    validation_set,
    write_rows,
    write_sft_groups,
)
from llm.data.v4_translation import (
    FLORES_LICENSE_URL,
    FLORES_SHA256,
    FLORES_URL,
    fetch_flores,
    flores_dialogues,
)
from llm.tokenizer.bpe import BPETokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sources", type=Path, default=Path("configs/data_v4_sources.json"))
    parser.add_argument("--inputs", type=Path, default=Path("data/v4_inputs"))
    parser.add_argument("--output", type=Path, default=Path("data/v4/dialogues"))
    parser.add_argument("--tokenizer", type=Path, default=Path("data/tokenizer"))
    parser.add_argument(
        "--legacy-sft",
        nargs="*",
        type=Path,
        default=[Path("data/v3_1/sft_groups"), Path("data/v3_3/sft_groups")],
    )
    parser.add_argument(
        "--protected",
        nargs="*",
        type=Path,
        default=[Path("configs/natural_eval_v4.jsonl")],
    )
    parser.add_argument("--sentiment-limit", type=int, default=0)
    parser.add_argument("--definitions", action="store_true")
    parser.add_argument("--definitions-limit", type=int, default=900)
    parser.add_argument("--curated", type=Path)
    parser.add_argument("--validation-only", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"output already exists: {args.output}")
    for directory in args.legacy_sft:
        if not list(directory.glob("train.*.jsonl")):
            parser.error(f"legacy SFT files unavailable: {directory}")
    all_sources = load_sources(args.sources)
    names = {"wiki_ru", "wiki_kk", "wiki_ky", "oasst_val", "kk_support"}
    names.update(f"belebele_{lang}" for lang in LANGUAGES)
    if not args.validation_only:
        names.update(("oasst_train", "aya_train", "dolly", "squad_train", "kazsandra"))
        if args.definitions:
            names.add("wiki_en")
    sources = {name: source for name, source in all_sources.items() if name in names}
    with ThreadPoolExecutor(max_workers=5) as pool:
        list(pool.map(partial(fetch, root=args.inputs), sources.values()))
    quality = build_language_filter(sources, args.inputs)
    tokenizer = BPETokenizer.load(args.tokenizer)
    flores = fetch_flores(args.inputs)
    paths = {name: source.local(args.inputs) for name, source in sources.items()}
    candidates = chain(
        oasst_dialogues("oasst_val", paths["oasst_val"], quality, tokenizer),
        support_dialogues(paths["kk_support"], quality, tokenizer),
        chain.from_iterable(
            belebele_dialogues(
                f"belebele_{lang}",
                paths[f"belebele_{lang}"],
                lang,
                quality,
                tokenizer,
            )
            for lang in LANGUAGES
        ),
        chain.from_iterable(
            flores_dialogues(flores, "dev", lang, quality, tokenizer) for lang in LANGUAGES
        ),
    )
    legacy = legacy_prompts(
        path for directory in args.legacy_sft for path in sorted(directory.glob("train.*.jsonl"))
    )
    protected = legacy_prompts(args.protected)
    holdout = validation_set(candidates, legacy)
    holdout.index.block(protected)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=args.output.parent) as staging:
        root = Path(staging) / "dialogues"
        root.mkdir()
        report = {
            "tokenizer": tokenizer.version,
            "validation": write_rows(root / "validation.jsonl", holdout.rows),
            "sources": source_manifest(sources, args.inputs),
            "legacy_prompt_hashes": len(legacy),
            "flores": {
                "url": FLORES_URL,
                "license_url": FLORES_LICENSE_URL,
                "license": "CC-BY-SA-4.0",
                "sha256": FLORES_SHA256,
            },
        }
        if not args.validation_only:
            train: Iterator[Dialogue] = select_training(
                chain(
                    oasst_dialogues("oasst_train", paths["oasst_train"], quality, tokenizer),
                    aya_dialogues(paths["aya_train"], quality, tokenizer),
                    dolly_dialogues(paths["dolly"], quality, tokenizer),
                    squad_dialogues("squad_train", paths["squad_train"], quality, tokenizer),
                    kazsandra_dialogues(
                        paths["kazsandra"], quality, tokenizer, args.sentiment_limit
                    ),
                    *([curated_dialogues(args.curated, tokenizer)] if args.curated else []),
                    *(
                        wiki_definition_dialogues(
                            f"wiki_{lang}",
                            paths[f"wiki_{lang}"],
                            lang,
                            quality,
                            tokenizer,
                            args.definitions_limit,
                        )
                        for lang in LANGUAGES
                        if args.definitions
                    ),
                ),
                holdout.index,
            )
            finished = list(train)
            report["sft_supervised_tokens"] = write_sft_groups(root, finished, tokenizer)
            report["conversational_train"] = write_rows(
                root / "conversational_train.jsonl",
                (row for row in finished if row.category not in SEMANTIC),
            )
            report["semantic_train"] = write_rows(
                root / "semantic_train.jsonl", (row for row in finished if row.category in SEMANTIC)
            )
        output = json.dumps(report, indent=2, sort_keys=True)
        (root / "manifest.json").write_text(output + "\n", encoding="utf-8")
        root.rename(args.output)
        print(output)


if __name__ == "__main__":
    main()
