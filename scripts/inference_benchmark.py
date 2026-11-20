from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from llm.evaluation.comparison import load_engine
from llm.evaluation.inference_bench import measure_latency


def main() -> int:
    parser = argparse.ArgumentParser(prog="inference_benchmark")
    parser.add_argument("checkpoint")
    parser.add_argument("--tokenizer", default="data/tokenizer")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--prompt-tokens", type=int, nargs="+", default=[32, 128, 384])
    parser.add_argument("--new-tokens", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", default="experiments/inference_benchmark.json")
    args = parser.parse_args()
    engine = load_engine(args.checkpoint, args.tokenizer, args.device)
    rows = []
    for prompt_tokens in args.prompt_tokens:
        for kind in ("concat", "preallocated"):
            budget = min(args.new_tokens, engine.max_positions - prompt_tokens)
            row = asdict(measure_latency(engine, prompt_tokens, budget, args.repeats, kind))
            rows.append(row)
            print(json.dumps(row), flush=True)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(rows, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
