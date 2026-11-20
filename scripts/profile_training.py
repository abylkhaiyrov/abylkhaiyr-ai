from __future__ import annotations

import argparse
import json
import statistics
from dataclasses import asdict
from pathlib import Path

from llm.core.config import load_experiment_config
from llm.optimization.profiling import (
    Variant,
    loss_trajectory,
    max_trajectory_gap,
    measure_throughput,
    profile_phases,
)

VARIANTS = {
    "baseline": Variant("baseline", foreach=False),
    "foreach": Variant("foreach"),
    "native_gqa": Variant("native_gqa", native_gqa=True),
    "fast_causal": Variant("fast_causal", fast_causal=True),
    "gqa_causal": Variant("gqa_causal", native_gqa=True, fast_causal=True),
    "exact_pad": Variant("exact_pad", pad_multiple=1),
    "packed": Variant("packed", pack=True),
    "packed_gqa_causal": Variant("packed_gqa_causal", pack=True, native_gqa=True),
    "bf16": Variant("bf16", precision="bf16"),
    "fp16": Variant("fp16", precision="fp16"),
    "bf16_causal": Variant("bf16_causal", precision="bf16", fast_causal=True),
    "bf16_causal_mb16": Variant(
        "bf16_causal_mb16", precision="bf16", fast_causal=True, micro_batch=16, accumulation=2
    ),
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
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--stability-steps", type=int, default=0)
    parser.add_argument("--output", default="experiments/profile_training.json")
    args = parser.parse_args()
    config = load_experiment_config(args.config)
    config = config.model_copy(
        update={"train": config.train.model_copy(update={"device": args.device})}
    )
    results = []
    reference = None
    for name in args.variants:
        variant = VARIANTS[name]
        runs = [
            asdict(measure_throughput(config, variant, args.steps)) for _ in range(args.repeats)
        ]
        row = {
            **runs[len(runs) // 2],
            "useful_tokens_per_second": statistics.median(
                r["useful_tokens_per_second"] for r in runs
            ),
            "steps_per_second": statistics.median(r["steps_per_second"] for r in runs),
            "runs_useful_tokens_per_second": [r["useful_tokens_per_second"] for r in runs],
            "peak_device_gb": max(r["peak_device_gb"] for r in runs),
        }
        if args.phases:
            row["phases"] = profile_phases(config, variant, args.steps)
        if args.stability_steps and not variant.pack:
            losses = loss_trajectory(config, variant, args.stability_steps)
            reference = reference or losses
            row["max_loss_gap_vs_first"] = max_trajectory_gap(reference, losses)
        results.append(row)
        print(json.dumps(row), flush=True)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(results, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
