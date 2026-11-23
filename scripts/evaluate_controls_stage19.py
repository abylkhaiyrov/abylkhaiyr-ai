from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from llm.data.binary import read_jsonl
from llm.evaluation.capacity_metrics import unigram_probabilities
from llm.evaluation.comparison import load_engine
from llm.evaluation.diagnostic_bank import Item
from llm.evaluation.diagnostic_controls import control_report, score_controls

BANKS = {
    "diagnostic144": Path("experiments/stage18/diagnostic.jsonl"),
    "historical1152": Path("experiments/stage17/diagnostic_v2.jsonl"),
}
UNIGRAM = Path("data/v4_expanded/tokens/train.bin")


def load_items(path: Path) -> list[Item]:
    return [Item(**{**row, "options": tuple(row["options"])}) for row in read_jsonl(path)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("models", nargs="+", help="name=checkpoint")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--output", type=Path, default=Path("experiments/stage19/controls"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    for entry in args.models:
        name, checkpoint = entry.split("=", 1)
        engine = load_engine(checkpoint, "data/tokenizer", args.device)
        unigram = unigram_probabilities(UNIGRAM, engine.tokenizer)
        for label, path in BANKS.items():
            rows = list(score_controls(engine, load_items(path), unigram))
            payload = {
                "checkpoint": checkpoint,
                "bank": label,
                "report": control_report(rows),
                "rows": [asdict(row) for row in rows],
            }
            (args.output / f"{name}.{label}.json").write_text(
                json.dumps(payload, ensure_ascii=False) + "\n", encoding="utf-8"
            )
        print(f"{name} done", flush=True)


if __name__ == "__main__":
    main()
