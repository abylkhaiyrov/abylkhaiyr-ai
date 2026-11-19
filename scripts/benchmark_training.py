import json
import sys
from pathlib import Path

import torch

from llm.core.config import load_experiment_config
from llm.core.hardware import resolve_device
from llm.optimization.benchmark import BenchmarkCase, run_case

config = load_experiment_config(sys.argv[1])
device = resolve_device("auto")
length = config.train.sequence_length
cases = [
    BenchmarkCase(4, length),
    BenchmarkCase(8, length),
    BenchmarkCase(12, length),
    BenchmarkCase(16, length, activation_checkpointing=True),
    BenchmarkCase(24, length, activation_checkpointing=True),
    BenchmarkCase(8, length, torch.bfloat16),
    BenchmarkCase(16, length, torch.bfloat16),
    BenchmarkCase(8, length, torch.float16),
]
results = []
for case in cases:
    try:
        r = run_case(config.model, case, device, steps=6)
    except (RuntimeError, ValueError) as error:
        print("ERR", case, type(error).__name__, str(error)[:100], flush=True)
        continue
    row = {
        "micro_batch": case.micro_batch,
        "dtype": str(case.dtype).split(".")[-1],
        "ckpt": case.activation_checkpointing,
        "tok_s": round(r.tokens_per_second),
        "peak_gb": round(r.peak_device_gb, 2),
        "swap_gb": round(r.swap_used_gb, 2),
        "ok": r.ok,
    }
    print(row, flush=True)
    results.append(row)
Path("experiments/benchmark_tiny.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
