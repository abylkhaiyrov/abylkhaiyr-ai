from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from llm.core.hardware import resolve_device, resolve_dtype
from llm.core.profiles import build_model_config, count_parameters, estimate_training_memory_gb
from llm.core.scaling import flops_per_token
from llm.optimization.benchmark import BenchmarkCase, run_case

PROFILES = ("tiny", "small", "small_plus")
BUDGETS = (5_000_000, 20_000_000, 50_000_000, 100_000_000)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="mps")
    parser.add_argument("--precision", default="fp32")
    parser.add_argument("--micro-batch", type=int, default=2)
    parser.add_argument("--sequence-length", type=int, default=512)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(args.micro_batch, args.sequence_length, args.steps) < 1:
        parser.error("batch, context and steps must be positive")
    device = resolve_device(args.device)
    dtype = resolve_dtype(args.precision, device)
    torch.set_num_threads(4)
    case = BenchmarkCase(args.micro_batch, args.sequence_length, dtype)
    results = []
    tiny_flops = flops_per_token(build_model_config("tiny", 8192), args.sequence_length)
    for name in PROFILES:
        model = build_model_config(name, 8192)
        flops = flops_per_token(model, args.sequence_length)
        result = run_case(model, case, device, args.steps)
        speed = result.tokens_per_second
        results.append(
            {
                "profile": name,
                "parameters": count_parameters(model),
                "flops_per_token": flops,
                "memory_estimate_gib": estimate_training_memory_gb(
                    model, case.micro_batch, case.sequence_length
                ),
                "device_memory_gib": result.peak_device_gb,
                "swap_gib": result.swap_used_gb,
                "micro_batch": case.micro_batch,
                "sequence_length": case.sequence_length,
                "success": result.ok,
                "tokens_per_second": speed,
                "budgets": [
                    {
                        "equal_tokens": tokens,
                        "equal_flops_tokens": int(tokens * tiny_flops / flops),
                        "equal_token_seconds": tokens / speed if result.ok else None,
                        "equal_flops_seconds": (
                            tokens * tiny_flops / flops / speed if result.ok else None
                        ),
                    }
                    for tokens in BUDGETS
                ],
            }
        )
    report = json.dumps(
        {"device": str(device), "precision": args.precision, "profiles": results},
        indent=2,
        allow_nan=False,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report + "\n", encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
