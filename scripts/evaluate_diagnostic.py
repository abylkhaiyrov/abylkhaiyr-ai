from __future__ import annotations

import argparse
import json
from pathlib import Path

from llm.data.binary import read_jsonl
from llm.evaluation.comparison import load_engine
from llm.evaluation.diagnostic import score_item, summarize


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("--items", type=Path, default=Path("configs/diagnostic_v1.jsonl"))
    parser.add_argument("--tokenizer", default="data/tokenizer")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    engine = load_engine(args.checkpoint, args.tokenizer, args.device)
    items = list(read_jsonl(args.items))
    report = summarize(score_item(engine, item) for item in items)
    report["checkpoint"] = args.checkpoint
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    overall = report["overall"]
    print(
        f"{args.checkpoint}: {overall['rate']:.3f} (chance {overall['chance']:.2f}, n={overall['n']})"
    )


if __name__ == "__main__":
    main()
