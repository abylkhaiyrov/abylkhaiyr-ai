from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from llm.evaluation.comparison import load_engine
from llm.evaluation.stress import growth, inference_memory_profile


def main() -> int:
    parser = argparse.ArgumentParser(prog="stress_inference")
    parser.add_argument("checkpoint")
    parser.add_argument("--tokenizer", default="data/tokenizer")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--requests", type=int, default=600)
    parser.add_argument("--max-prompt-words", type=int, default=240)
    parser.add_argument("--every", type=int, default=50)
    parser.add_argument("--output", default="experiments/stress_inference.json")
    args = parser.parse_args()
    engine = load_engine(args.checkpoint, args.tokenizer, args.device)
    samples = inference_memory_profile(engine, args.requests, args.max_prompt_words, args.every)
    payload = {
        "requests": args.requests,
        "samples": [asdict(sample) for sample in samples],
        "rss_growth_mb": growth(samples, "rss_mb"),
        "device_growth_gb": growth(samples, "device_gb"),
        "tensor_growth": growth(samples, "live_tensors"),
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in payload.items() if k != "samples"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
