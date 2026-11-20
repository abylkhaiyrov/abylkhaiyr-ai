from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from llm.evaluation.comparison import default_variants, load_engine, sweep_sampling, word_list


def main() -> int:
    parser = argparse.ArgumentParser(prog="sampling_sweep")
    parser.add_argument("checkpoint")
    parser.add_argument("--tokenizer", default="data/tokenizer")
    parser.add_argument("--data-dir", default="data/v2")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--output", default="experiments/sampling_sweep.json")
    args = parser.parse_args()
    engine = load_engine(args.checkpoint, args.tokenizer, args.device)
    result = sweep_sampling(engine, default_variants(), words=word_list(args.data_dir))
    best = {group: asdict(row) for group, row in result.best_by_group().items()}
    payload = {"rows": [asdict(row) for row in result.rows], "best_by_group": best}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(best, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
