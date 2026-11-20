from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch

from llm.inference.engine import InferenceEngine
from llm.inference.sampling import SamplingParams
from llm.tokenizer.bpe import BPETokenizer, Message
from llm.training.checkpoint import load_model_for_inference
from llm.training.losses import IGNORE_INDEX, causal_lm_loss_sum


def load_pairs(path: Path, limit: int, seed: int) -> list[tuple[str, str]]:
    pairs = []
    with path.open() as handle:
        for line in handle:
            messages = json.loads(line)["messages"]
            if len(messages) == 2:
                pairs.append((messages[0]["content"], messages[1]["content"]))
    random.Random(seed).shuffle(pairs)
    return pairs[:limit]


def encode(tokenizer: BPETokenizer, pair: tuple[str, str]) -> tuple[list[int], list[int]]:
    messages = [Message("user", pair[0]), Message("assistant", pair[1])]
    return tokenizer.encode_conversation_with_labels(messages, IGNORE_INDEX)


def f1(prediction: list[str], reference: list[str]) -> float:
    common = sum(min(prediction.count(w), reference.count(w)) for w in set(prediction))
    if not common:
        return 0.0
    precision, recall = common / len(prediction), common / len(reference)
    return 2 * precision * recall / (precision + recall)


def evaluate(engine: InferenceEngine, pairs: list[tuple[str, str]]) -> dict[str, float]:
    params = SamplingParams(max_new_tokens=64, temperature=0.0)
    f1s, echoes, exact = [], 0, 0
    for prompt, gold in pairs:
        text = engine.chat([Message("user", prompt)], params).text
        f1s.append(f1(text.lower().split(), gold.lower().split()))
        echoes += int(text.strip().lower() == prompt.strip().lower())
        exact += int(text.strip() == gold.strip())
    count = len(pairs)
    return {
        "token_f1": round(sum(f1s) / count, 3),
        "exact_match": round(exact / count, 3),
        "prompt_echo_rate": round(echoes / count, 3),
    }


def main() -> int:
    parser = argparse.ArgumentParser(prog="overfit_test")
    parser.add_argument("--init", default="checkpoints/tiny_v2/tiny_best")
    parser.add_argument("--tokenizer", default="data/tokenizer")
    parser.add_argument("--train-file", default="data/v3_3/sft_groups/train.dialog.jsonl")
    parser.add_argument("--heldout-file", default="data/v3_3/sft_groups/validation.dialog.jsonl")
    parser.add_argument("--examples", type=int, default=64)
    parser.add_argument("--steps", type=int, default=150)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    device = torch.device(args.device)
    tokenizer = BPETokenizer.load(args.tokenizer)
    model, _, _ = load_model_for_inference(args.init)
    model = model.to(device)
    train_pairs = load_pairs(Path(args.train_file), args.examples, 1)
    heldout = load_pairs(Path(args.heldout_file), 64, 2)
    train_ids = {pair[0] for pair in train_pairs}
    heldout = [pair for pair in heldout if pair[0] not in train_ids]
    encoded = [encode(tokenizer, pair) for pair in train_pairs]
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
    history = []
    for step in range(1, args.steps + 1):
        model.train()
        total, tokens = 0.0, 0
        optimizer.zero_grad()
        for ids, labels in encoded:
            logits = model(torch.tensor([ids[:-1]], device=device))
            loss, count = causal_lm_loss_sum(logits, torch.tensor([labels[1:]], device=device))
            loss.backward()
            total += float(loss)
            tokens += count
        for parameter in model.parameters():
            if parameter.grad is not None:
                parameter.grad /= tokens
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        history.append(total / tokens)
        if step % 25 == 0:
            print(f"step {step} loss {history[-1]:.4f}", flush=True)
    model.eval()
    engine = InferenceEngine(model, tokenizer, device)
    report = {
        "examples": len(train_pairs),
        "steps": args.steps,
        "final_train_loss": round(history[-1], 4),
        "train": evaluate(engine, train_pairs),
        "heldout": evaluate(engine, heldout),
    }
    Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
