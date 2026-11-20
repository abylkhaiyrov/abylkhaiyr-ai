from __future__ import annotations

import argparse
import json
import statistics
import sys
from dataclasses import asdict
from pathlib import Path

import torch

from llm.core.config import ExperimentConfig, load_experiment_config
from llm.optimization.gpu_monitor import GpuSampler
from llm.optimization.profiling import (
    Variant,
    loss_trajectory,
    max_trajectory_gap,
    measure_throughput,
    profile_phases,
)


def plan(config: ExperimentConfig) -> list[Variant]:
    base = Variant("fp32", foreach=False)
    causal = {"precision": "bf16", "fast_causal": True}
    variants = [
        base,
        Variant("tf32", tf32=True, foreach=False),
        Variant("bf16", precision="bf16", foreach=False),
        Variant("fp16", precision="fp16", foreach=False),
        Variant("bf16_causal", **causal),
        Variant("bf16_causal_gqa", native_gqa=True, **causal),
        Variant("bf16_causal_flash", sdpa="flash", **causal),
        Variant("bf16_causal_efficient", sdpa="efficient", **causal),
        Variant("bf16_causal_math", sdpa="math", **causal),
        Variant("bf16_causal_foreach", foreach=True, **causal),
        Variant("bf16_causal_mb16", micro_batch=16, accumulation=2, **causal),
        Variant("bf16_causal_mb32", micro_batch=32, accumulation=1, **causal),
        Variant(
            "bf16_causal_ckpt_mb32", micro_batch=32, accumulation=1, checkpointing=True, **causal
        ),
        Variant("bf16_causal_compile", compile=True, **causal),
    ]
    if config.train.stage == "sft":
        variants += [
            Variant("bf16_packed", pack=True, **causal),
            Variant("bf16_packed_mb16", pack=True, micro_batch=16, accumulation=2, **causal),
        ]
    return variants


def main() -> int:
    parser = argparse.ArgumentParser(prog="cuda_benchmark")
    parser.add_argument("--config", default="configs/tiny_gpu.yaml")
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--stability-steps", type=int, default=6)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output", default="experiments/cuda_benchmark.json")
    args = parser.parse_args()
    config = load_experiment_config(args.config)
    variants = plan(config)
    if args.dry_run:
        print(json.dumps([asdict(v) for v in variants], indent=1))
        return 0
    if not torch.cuda.is_available():
        print("NOT RUN: CUDA is unavailable on this machine", file=sys.stderr)
        return 2
    config = config.model_copy(update={"train": config.train.model_copy(update={"device": "cuda"})})
    header = {"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__}
    rows = []
    reference = None
    for variant in variants:
        with GpuSampler() as sampler:
            runs = [
                asdict(measure_throughput(config, variant, args.steps)) for _ in range(args.repeats)
            ]
        usage = sampler.usage()
        row = {
            **runs[len(runs) // 2],
            "useful_tokens_per_second": statistics.median(
                r["useful_tokens_per_second"] for r in runs
            ),
            "runs_useful_tokens_per_second": [r["useful_tokens_per_second"] for r in runs],
            "gpu_usage": None if usage is None else asdict(usage),
            "phases": profile_phases(config, variant, max(args.steps // 2, 2)),
        }
        if not variant.pack:
            losses = loss_trajectory(config, variant, args.stability_steps)
            reference = reference or losses
            row["max_loss_gap_vs_fp32"] = max_trajectory_gap(reference, losses)
        rows.append(row)
        print(json.dumps({k: v for k, v in row.items() if k != "phases"}), flush=True)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps({**header, "rows": rows}, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
