"""Save reproducible raw model responses for a manual chat quality check."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from llm.inference.engine import InferenceEngine
from llm.inference.sampling import SamplingParams
from llm.tokenizer.bpe import BPETokenizer, Message
from llm.training.checkpoint import load_model_for_inference

PROMPTS = (
    "Что ты умеешь делать?",
    "Какие задачи тебе можно поручить?",
    "Привет!",
    "Как тебя зовут?",
    "Объясни простыми словами, что такое алгоритм.",
    "Сәлем! Сен кімсің?",
    "В какой галактике находится вымышленная планета Зоримба?",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/tiny_gpu_sft/best")
    parser.add_argument("--output", default="experiments/tiny_gpu_sft/chat_check.json")
    args = parser.parse_args()
    torch.set_num_threads(4)
    model, _, meta = load_model_for_inference(Path(args.checkpoint))
    engine = InferenceEngine(model, BPETokenizer.load("data/tokenizer"), torch.device("cuda"))
    params = SamplingParams(
        max_new_tokens=128, temperature=0.3, top_k=40, top_p=0.9, repetition_penalty=1.1, seed=42
    )
    responses = []
    for prompt in PROMPTS:
        answer = engine.chat([Message("user", prompt)], params)
        responses.append(
            {"prompt": prompt, "answer": answer.text, "finish_reason": answer.finish_reason}
        )
        print(json.dumps(responses[-1], ensure_ascii=False), flush=True)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {"checkpoint": args.checkpoint, "step": meta["step"], "responses": responses},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
