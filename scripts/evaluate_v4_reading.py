from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from llm.data.binary import read_jsonl
from llm.data.v4_quality import LANGUAGES
from llm.evaluation.comparison import load_engine
from llm.evaluation.v4_review import proportion
from llm.inference.engine import InferenceEngine
from llm.tokenizer.bpe import Message


def reading_case(row: dict) -> tuple[list[Message], int]:
    context, reference = row["messages"][:-1], row["messages"][-1]["content"]
    options = context[-1]["content"].splitlines()[-4:]
    for index, option in enumerate(options, 1):
        if option == f"{index}. {reference}":
            return [Message(**item) for item in context], index - 1
    raise ValueError(f"no option matching reference: {row['id']}")


@torch.inference_mode()
def evaluate(engine: InferenceEngine, rows: list[dict]) -> dict:
    labels = [engine.tokenizer.encode(str(index)) for index in range(1, 5)]
    if any(len(ids) != 1 for ids in labels):
        raise ValueError("numbered choice labels must encode to one token each")
    column = [ids[0] for ids in labels]
    results = {}
    for language in LANGUAGES:
        cases = [reading_case(row) for row in rows if row["language"] == language]
        matched = 0
        for offset in range(0, len(cases), 8):
            batch = cases[offset : offset + 8]
            inputs = [
                engine.tokenizer.encode_conversation(
                    engine.fit_messages(messages, 1)[0],
                    add_generation_prompt=True,
                )
                for messages, _ in batch
            ]
            lengths = torch.tensor([len(ids) - 1 for ids in inputs], device=engine.device)
            ids = torch.nn.utils.rnn.pad_sequence(
                [torch.tensor(item) for item in inputs],
                batch_first=True,
                padding_value=engine.tokenizer.pad_id,
            ).to(engine.device)
            logits = engine.model(ids)[torch.arange(len(batch), device=engine.device), lengths]
            choices = logits[:, column].argmax(dim=1).tolist()
            matched += sum(
                choice == answer for choice, (_, answer) in zip(choices, batch, strict=True)
            )
        results[language] = proportion(matched, len(cases))
    total = sum(result["n"] for result in results.values())
    hit = sum(result["successes"] for result in results.values())
    return {"overall": proportion(hit, total), "languages": results}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoints", nargs="+")
    parser.add_argument("--cases", type=Path, default=Path("data/v4/sft_final/validation.jsonl"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--tokenizer", default="data/tokenizer")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = [
        row
        for row in read_jsonl(args.cases)
        if row["category"] == "reading" and row["source"].startswith("belebele_")
    ]
    results = {
        checkpoint: evaluate(load_engine(checkpoint, args.tokenizer, args.device), rows)
        for checkpoint in args.checkpoints
    }
    report = json.dumps(results, indent=2, allow_nan=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report + "\n", encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
