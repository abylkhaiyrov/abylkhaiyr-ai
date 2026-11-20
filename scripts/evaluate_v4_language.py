from __future__ import annotations

import argparse
import json
from pathlib import Path

from llm.evaluation.comparison import load_engine
from llm.evaluation.v4_language import language_losses


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoints", nargs="+")
    parser.add_argument("--tokens", type=Path, default=Path("data/v4/tokens"))
    parser.add_argument("--tokenizer", default="data/tokenizer")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--batches", type=int, default=128)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.batches < 1:
        parser.error("--batches must be positive")
    scores = {}
    for checkpoint in args.checkpoints:
        engine = load_engine(checkpoint, args.tokenizer, args.device)
        scores[checkpoint] = language_losses(engine, args.tokens, args.batches)
        del engine
    report = json.dumps(scores, indent=2, allow_nan=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report + "\n", encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
