from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch

from llm.evaluation.comparison import load_engine
from llm.tokenizer.bpe import Message
from llm.training.losses import IGNORE_INDEX, contrastive_pair_terms


def encode(engine, messages: list[dict[str, str]]) -> tuple[torch.Tensor, torch.Tensor]:
    ids, labels = engine.tokenizer.encode_conversation_with_labels(
        [Message(m["role"], m["content"]) for m in messages], IGNORE_INDEX
    )
    tensor = torch.tensor([ids], device=engine.device)
    return tensor[:, :-1], torch.tensor([labels], device=engine.device)[:, 1:]


@torch.inference_mode()
def evaluate(engine, pairs: list[dict]) -> dict:
    buckets: dict[str, list[float]] = defaultdict(list)
    for item in pairs:
        chosen_in, chosen_tg = encode(engine, item["chosen"])
        rejected_in, rejected_tg = encode(engine, item["rejected"])
        terms = contrastive_pair_terms(
            engine.model(chosen_in), chosen_tg, engine.model(rejected_in), rejected_tg, 1.0, 0.0
        )
        if not bool(terms.valid.item()):
            continue
        delta = float(terms.delta.item())
        for key in ("all", f"kind:{item['kind']}", f"language:{item['language']}"):
            buckets[key].append(delta)
    return {
        key: {
            "n": len(values),
            "accuracy": sum(value > 0 for value in values) / len(values),
            "mean_margin": sum(values) / len(values),
        }
        for key, values in sorted(buckets.items())
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument(
        "--pairs", type=Path, default=Path("data/verified_v1/preference/validation.jsonl")
    )
    parser.add_argument("--tokenizer", default="data/tokenizer")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    engine = load_engine(args.checkpoint, args.tokenizer, args.device)
    pairs = [json.loads(line) for line in args.pairs.read_text(encoding="utf-8").splitlines()]
    report = evaluate(engine, pairs)
    args.output.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(report["all"]))


if __name__ == "__main__":
    main()
