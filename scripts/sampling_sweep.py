from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from llm.evaluation.comparison import (
    default_variants,
    load_engine,
    sweep_validation,
    validation_cases,
    word_list,
)
from llm.evaluation.conversation import engine_responder, run_conversation_benchmark


def main() -> int:
    parser = argparse.ArgumentParser(prog="sampling_sweep")
    parser.add_argument("checkpoint")
    parser.add_argument("--tokenizer", default="data/tokenizer")
    parser.add_argument("--validation-dir", default="data/v3")
    parser.add_argument("--data-dir", default="data/v2")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--per-group", type=int, default=30)
    parser.add_argument("--validation-only", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--variants", nargs="+")
    parser.add_argument("--output", default="experiments/sampling_sweep.json")
    args = parser.parse_args()
    engine = load_engine(args.checkpoint, args.tokenizer, args.device)
    variants = default_variants(args.max_new_tokens)
    if args.variants:
        requested = set(args.variants)
        variants = [variant for variant in variants if variant.name in requested]
        if {variant.name for variant in variants} != requested:
            parser.error("unknown sampling variant")
    result = sweep_validation(
        engine, variants, validation_cases(args.validation_dir, args.per_group)
    )
    best = result.best_by_group()
    by_name = {variant.name: variant for variant in variants}
    words = word_list(args.data_dir)
    benchmark = (
        {}
        if args.validation_only
        else {
            name: run_conversation_benchmark(
                engine_responder(engine, by_name[name].params), words
            ).metrics
            for name in sorted(
                {row.variant for row in best.values()} | ({"greedy_rp1.1"} & by_name.keys())
            )
        }
    )
    payload = {
        "selection": "validation split (token F1, completion, repetition)",
        "rows": [asdict(row) for row in result.rows],
        "best_by_group": {group: asdict(row) for group, row in best.items()},
        "benchmark_of_selected": benchmark,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload["best_by_group"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
