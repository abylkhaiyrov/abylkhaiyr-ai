from __future__ import annotations

import argparse
from pathlib import Path

import torch

from llm.developer_agent import DeveloperTask, solve
from llm.evaluation.comparison import load_engine


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("task", type=Path)
    parser.add_argument("--image", required=True)
    parser.add_argument("--checkpoint", default="checkpoints/tiny_gpu_sft/best")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    task = DeveloperTask.model_validate_json(args.task.read_bytes())
    engine = load_engine(args.checkpoint, "data/tokenizer", args.device)
    result = solve(engine, task, args.output, args.image)
    result_json = result.model_dump_json(indent=2)
    (args.output / "result.json").write_text(result_json, encoding="utf-8")
    print(result_json)
    return 0 if result.status == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
