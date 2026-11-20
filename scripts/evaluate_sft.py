from __future__ import annotations

import argparse
import json
from pathlib import Path

from llm.evaluation.comparison import load_engine
from llm.evaluation.engine import EvaluationConfig, Evaluator
from llm.optimization.memory import release_device_memory


def main() -> int:
    parser = argparse.ArgumentParser(prog="evaluate_sft")
    parser.add_argument("models", nargs="+", help="name=checkpoint pairs")
    parser.add_argument("--data-dir", default="data/v2")
    parser.add_argument("--tokenizer", default="data/tokenizer")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--split", choices=("validation", "test"), required=True)
    parser.add_argument("--retention-only", action="store_true")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config = EvaluationConfig(args.data_dir, corpus_split=args.split)
    reports = {}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    for model in args.models:
        name, checkpoint = model.split("=", 1)
        engine = load_engine(checkpoint, args.tokenizer, args.device)
        evaluator = Evaluator(engine, config)
        if args.split == "test" and not args.retention_only:
            metrics = evaluator.run(include_dataset_suites=False)
        else:
            metrics = {**evaluator.language_modeling(), **evaluator.multilingual_perplexity()}
            if args.split == "test":
                metrics.update(evaluator.conversation())
        reports[name] = {"checkpoint": checkpoint, "split": args.split, "metrics": metrics}
        output.write_text(json.dumps(reports, indent=2, allow_nan=False) + "\n")
        print(json.dumps({"model": name, **metrics}), flush=True)
        device = engine.device
        del evaluator, engine
        release_device_memory(device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
