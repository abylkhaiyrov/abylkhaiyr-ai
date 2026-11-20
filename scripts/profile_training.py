from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from llm.core.config import load_experiment_config
from llm.optimization.profiling import Variant, measure_throughput, profile_phases

VARIANTS = {
    "baseline": Variant("baseline", foreach=False),
    "foreach": Variant("foreach"),
    "native_gqa": Variant("native_gqa", native_gqa=True),
    "fast_causal": Variant("fast_causal", fast_causal=True),
    "gqa_causal": Variant("gqa_causal", native_gqa=True, fast_causal=True),
    "packed": Variant("packed", pack=True),
    "packed_gqa_causal": Variant("packed_gqa_causal", pack=True, native_gqa=True),
    "bf16": Variant("bf16", precision="bf16"),
    "ckpt_mb16": Variant("ckpt_mb16", micro_batch=16, accumulation=2, checkpointing=True),
    "mb4": Variant("mb4", micro_batch=4, accumulation=8),
    "mb16": Variant("mb16", micro_batch=16, accumulation=2),
}


def main() -> int:
    parser = argparse.ArgumentParser(prog="profile_training")
    parser.add_argument("--config", default="configs/tiny_v3_sft.yaml")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--variants", nargs="+", default=["baseline", "foreach"])
    parser.add_argument("--steps", type=int, default=6)
    parser.add_argument("--phases", action="store_true")
    parser.add_argument("--output", default="experiments/profile_training.json")
    args = parser.parse_args()
    config = load_experiment_config(args.config)
    config = config.model_copy(
        update={"train": config.train.model_copy(update={"device": args.device})}
    )
    results = []
    for name in args.variants:
        variant = VARIANTS[name]
        row = asdict(measure_throughput(config, variant, args.steps))
        if args.phases:
            row["phases"] = profile_phases(config, variant, args.steps)
        results.append(row)
        print(json.dumps(row), flush=True)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(results, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
