from __future__ import annotations

import argparse
import json
from pathlib import Path

from llm.core.config import load_experiment_config
from llm.evaluation.registry import record_experiment
from llm.tokenizer.bpe import BPETokenizer
from llm.training.experiments import EfficiencyVariant, train_candidate


def main() -> int:
    parser = argparse.ArgumentParser(prog="train_sft_budget")
    parser.add_argument("--config", default="configs/tiny_v31_sft.yaml")
    parser.add_argument("--sft-dir", default="data/v3_2")
    parser.add_argument("--corpus-dir", default="data/v2")
    parser.add_argument("--checkpoints", default="checkpoints/tiny_v32_sft")
    parser.add_argument("--output", default="experiments/sft_budget.json")
    parser.add_argument(
        "trials", nargs="+", help="name:learning_rate:budget_tokens:replay_fraction"
    )
    args = parser.parse_args()
    base = load_experiment_config(args.config)
    base = base.model_copy(
        update={
            "train": base.train.model_copy(
                update={"data_dir": args.sft_dir, "sft_replay_dir": f"{args.corpus_dir}/tokens"}
            )
        }
    )
    tokenizer = BPETokenizer.load(base.tokenizer.path)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    results = json.loads(output.read_text()) if output.is_file() else []
    for trial in args.trials:
        name, rate, budget, replay, *extra = trial.split(":")
        seed = int(extra[0]) if extra else base.train.seed
        variant = EfficiencyVariant(
            name,
            base.train.mixture,
            pack=True,
            learning_rate=float(rate),
            replay_fraction=float(replay),
        )
        seeded = base.model_copy(update={"train": base.train.model_copy(update={"seed": seed})})
        row = train_candidate(
            seeded, variant, int(float(budget)), tokenizer, args.corpus_dir, Path(args.checkpoints)
        )
        row.update(
            {
                "learning_rate": float(rate),
                "budget_tokens": int(float(budget)),
                "replay": float(replay),
                "seed": seed,
            }
        )
        results.append(row)
        output.write_text(json.dumps(results, indent=2, allow_nan=False) + "\n")
        print(json.dumps(row), flush=True)
        record_experiment(
            f"sft_budget_{name}",
            {"kind": "sft_budget", "sft_dir": args.sft_dir},
            {k: float(v) for k, v in row.items() if isinstance(v, (int, float))},
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
