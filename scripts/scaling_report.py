from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from llm.core.config import load_experiment_config
from llm.core.profiles import PROFILES, build_model_config
from llm.core.scaling import ScalingRow, calibrate_flops, scaling_row


def markdown(rows: list[ScalingRow]) -> str:
    header = (
        "| model | params M | MFLOPs/token | train mem GB (mb=1) | KV cache MB | tok/s (est.) |"
        " Chinchilla tokens B | Chinchilla hours | hours / 1B tokens |"
    )
    lines = [header, "|---|" + "---|" * 8]
    for row in rows:
        lines.append(
            f"| {row.name} | {row.parameters_m:.1f} | {row.flops_per_token_m:.0f} |"
            f" {row.training_memory_gb:.1f} | {row.kv_cache_mb:.1f} | {row.tokens_per_second:.0f} |"
            f" {row.chinchilla_tokens_b:.2f} | {row.chinchilla_hours:.0f} | {row.hours_per_billion_tokens:.0f} |"
        )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(prog="scaling_report")
    parser.add_argument("--reference", default="configs/tiny_v2.yaml")
    parser.add_argument("--tokens-per-second", type=float, required=True)
    parser.add_argument("--vocab", type=int, default=8192)
    parser.add_argument("--sequence-length", type=int, default=512)
    parser.add_argument("--output", default="experiments/scaling_report.json")
    args = parser.parse_args()
    reference = load_experiment_config(args.reference).model
    achieved = calibrate_flops(reference, args.sequence_length, args.tokens_per_second)
    rows = [
        scaling_row(name, build_model_config(name, args.vocab), args.sequence_length, 1, achieved)
        for name in PROFILES
    ]
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    payload = {"achieved_tflops": achieved / 1e12, "rows": [asdict(row) for row in rows]}
    Path(args.output).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"achieved {achieved / 1e12:.2f} TFLOPs (calibrated on {args.reference})")
    print(markdown(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
