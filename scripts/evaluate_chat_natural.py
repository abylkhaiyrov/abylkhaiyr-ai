from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from llm.data.binary import read_jsonl
from llm.evaluation.chat_natural import CONFIGS, SEEDS, runs_for, summarize
from llm.evaluation.comparison import load_engine


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("models", nargs="+", help="name=checkpoint")
    parser.add_argument("--cases", type=Path, default=Path("configs/chat_eval_natural_v1.jsonl"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    cases = list(read_jsonl(args.cases))
    for entry in args.models:
        name, checkpoint = entry.split("=", 1)
        engine = load_engine(checkpoint, "data/tokenizer", args.device)
        report: dict[str, Any] = {"checkpoint": checkpoint, "configs": {}, "rows": {}}
        for label, config in CONFIGS.items():
            runs = runs_for(engine, cases, config, SEEDS)
            pooled = [row for rows in runs for row in rows]
            report["configs"][label] = {**summarize(pooled), "runs": len(runs)}
            report["rows"][label] = runs[0]
        (args.output / f"{name}.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
        )
        print(name, "done", flush=True)


if __name__ == "__main__":
    main()
