from __future__ import annotations

import json
from pathlib import Path

import torch

from llm.core.profiles import count_parameters
from llm.data.binary import read_jsonl
from llm.evaluation.comparison import load_engine
from llm.evaluation.engine import EvaluationConfig, Evaluator
from llm.training.checkpoint import CheckpointManager


def main() -> None:
    torch.set_num_threads(4)
    output = Path("experiments/baseline_checkpoints.json")
    report: dict[str, object] = {}
    for path in sorted(Path("checkpoints").glob("*/*/meta.json")):
        checkpoint = str(path.parent)
        meta = CheckpointManager.verify(path.parent)
        engine = load_engine(checkpoint, "data/tokenizer", "cuda")
        corpus = Path("data/processed")
        evaluator = Evaluator(engine, EvaluationConfig(str(corpus), max_windows=32))
        samples = [
            row["text"] for _, row in zip(range(30), read_jsonl(corpus / "corpus/test.jsonl"))
        ]
        report[checkpoint] = {
            "step": meta["step"],
            "tokens_seen": meta["tokens_seen"],
            "parameters": count_parameters(engine.model.config),
            "architecture": engine.model.config.model_dump(),
            "tokenizer_roundtrips": sum(
                engine.tokenizer.decode(engine.tokenizer.encode(sample)) == sample
                for sample in samples
            ),
            **evaluator.language_modeling(),
            **evaluator.conversation(),
        }
        output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
        print(checkpoint, report[checkpoint], flush=True)


if __name__ == "__main__":
    main()
