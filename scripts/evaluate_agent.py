from __future__ import annotations

import argparse
import json
from pathlib import Path

from llm.agent.benchmark import evaluate_agent
from llm.evaluation.comparison import load_engine
from llm.optimization.memory import release_device_memory


def main() -> int:
    parser = argparse.ArgumentParser(prog="evaluate_agent")
    parser.add_argument("models", nargs="+", help="name=checkpoint pairs")
    parser.add_argument("--tokenizer", default="data/tokenizer")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    reports: dict[str, dict[str, dict[str, float]]] = {}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    for model in args.models:
        name, checkpoint = model.split("=", 1)
        engine = load_engine(checkpoint, args.tokenizer, args.device)
        reports[name] = {
            "unconstrained": evaluate_agent(engine, constrained=False, limit=args.limit),
            "constrained": evaluate_agent(engine, constrained=True, limit=args.limit),
        }
        output.write_text(json.dumps(reports, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({name: reports[name]}), flush=True)
        device = engine.device
        del engine
        release_device_memory(device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
