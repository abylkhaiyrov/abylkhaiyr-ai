from __future__ import annotations

import argparse
import json
from pathlib import Path

from llm.core.config import load_experiment_config
from llm.evaluation.registry import record_experiment
from llm.training.scaling_probe import run_probe


def main() -> int:
    parser = argparse.ArgumentParser(prog="scaling_probe")
    parser.add_argument("profiles", nargs="+")
    parser.add_argument("--config", default="configs/tiny_v2.yaml")
    parser.add_argument("--budget-tokens", type=int, default=6_000_000)
    parser.add_argument("--output", default="experiments/scaling_probe.json")
    args = parser.parse_args()
    base = load_experiment_config(args.config)
    results = []
    for profile in args.profiles:
        row = run_probe(base, profile, args.budget_tokens)
        results.append(row)
        print(json.dumps(row), flush=True)
        record_experiment(
            f"scaling_probe_{profile}",
            {"kind": "scaling_probe", "budget_tokens": args.budget_tokens},
            {k: float(v) for k, v in row.items() if isinstance(v, (int, float))},
        )
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(results, indent=2, allow_nan=False) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
