from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional

from llm.data.binary import read_jsonl
from llm.evaluation.comparison import load_engine
from llm.tokenizer.bpe import BPETokenizer, Message
from llm.training.losses import IGNORE_INDEX

BUCKETS = ((0, 16), (16, 64), (64, 160), (160, 10**6))


def bucket(length: int) -> str:
    low, high = next(item for item in BUCKETS if item[0] <= length < item[1])
    return f"{low}-{high if high < 10**6 else 'inf'}"


def special_tokens(tokenizer: BPETokenizer, rows: list[dict]) -> dict:
    ids = {
        "bos": tokenizer.bos_id,
        "eos": tokenizer.eos_id,
        "pad": tokenizer.pad_id,
        **{f"role_{k}": v for k, v in tokenizer.role_ids.items()},
    }
    wrong_final = inner = 0
    for row in rows:
        messages = [Message(item["role"], item["content"]) for item in row["messages"]]
        sequence, labels = tokenizer.encode_conversation_with_labels(messages, IGNORE_INDEX)
        supervised = [index for index, label in enumerate(labels) if label != IGNORE_INDEX]
        wrong_final += int(sequence[supervised[-1]] != tokenizer.eos_id)
        inner += int(sequence.count(tokenizer.eos_id) != len(messages))
    return {
        "ids": ids,
        "distinct": len(set(ids.values())) == len(ids),
        "conversations": len(rows),
        "last_supervised_token_is_eos_violations": wrong_final,
        "eos_count_differs_from_message_count": inner,
    }


def truncation(groups: Path, window: int) -> dict:
    result = {}
    for path in sorted(groups.glob("train.*.lengths.npy")):
        lengths = np.load(path)
        result[path.name.split(".")[1]] = {
            "conversations": len(lengths),
            "share_over_window": round(float((lengths > window).mean()), 4),
            "p50": int(np.median(lengths)),
            "p90": int(np.percentile(lengths, 90)),
        }
    return result


@torch.inference_mode()
def eos_by_category(engine, rows: list[dict]) -> dict:
    tokenizer = engine.tokenizer
    window = engine.max_positions
    cells: dict[str, list[tuple[float, bool]]] = defaultdict(list)
    for row in rows:
        messages = [Message(item["role"], item["content"]) for item in row["messages"]]
        ids, labels = tokenizer.encode_conversation_with_labels(messages, IGNORE_INDEX)
        if len(ids) > window:
            continue
        logits = engine.model(torch.tensor([ids[:-1]], device=engine.device))[0].float().cpu()
        probs = functional.softmax(logits, dim=-1)
        starts = [
            i
            for i, label in enumerate(labels[1:])
            if label == tokenizer.eos_id and labels[i + 1] != IGNORE_INDEX
        ]
        start = next((i for i, label in enumerate(labels) if label != IGNORE_INDEX), 0)
        for position in starts:
            length = position + 1 - start
            probability = float(probs[position, tokenizer.eos_id])
            top = int(probs[position].argmax()) == tokenizer.eos_id
            for key in (
                f"category/{row.get('category', 'n/a')}",
                f"length/{bucket(length)}",
                "all",
            ):
                cells[key].append((probability, top))
    return {
        key: {
            "n": len(values),
            "mean_eos_probability": round(float(np.mean([p for p, _ in values])), 3),
            "eos_is_top1": round(float(np.mean([t for _, t in values])), 3),
        }
        for key, values in sorted(cells.items())
        if len(values) >= 5
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("--groups", type=Path, default=Path("data/v5/sft_nodefs/sft_groups"))
    parser.add_argument("--tokenizer", default="data/tokenizer")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    engine = load_engine(args.checkpoint, args.tokenizer, args.device)
    rows = [
        row for path in sorted(args.groups.glob("validation.*.jsonl")) for row in read_jsonl(path)
    ]
    report = {
        "checkpoint": args.checkpoint,
        "special_tokens": special_tokens(engine.tokenizer, rows),
        "truncation": truncation(args.groups, engine.max_positions),
        "eos": eos_by_category(engine, rows),
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("special_tokens",)}, indent=1))
    print(json.dumps(report["eos"].get("all"), indent=1))


if __name__ == "__main__":
    main()
