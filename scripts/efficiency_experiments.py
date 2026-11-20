from __future__ import annotations

import argparse
import json
from pathlib import Path

from llm.core.config import load_experiment_config
from llm.evaluation.registry import record_experiment
from llm.tokenizer.bpe import BPETokenizer
from llm.training.experiments import EfficiencyVariant, run_variant

EASY_FIRST = {
    "dialog": 3.0,
    "identity": 3.0,
    "skills": 2.0,
    "reading_en": 1.5,
    "reading_ru": 1.5,
    "reading_kk": 1.5,
    "chat_en": 0.3,
    "chat_ru": 0.3,
    "chat_kk": 0.1,
    "chat_other": 0.1,
    "math": 0.3,
    "word_math": 0.3,
    "code": 0.3,
}


def variants(plain: dict[str, float], weighted: dict[str, float]) -> dict[str, EfficiencyVariant]:
    return {
        "padded_plain": EfficiencyVariant("padded_plain", plain),
        "padded_weighted": EfficiencyVariant("padded_weighted", weighted),
        "bucketed_weighted": EfficiencyVariant("bucketed_weighted", weighted, bucket_window=8),
        "packed_weighted": EfficiencyVariant("packed_weighted", weighted, pack=True),
        "curriculum_weighted": EfficiencyVariant(
            "curriculum_weighted", weighted, curriculum_start=EASY_FIRST, curriculum_batches=2400
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(prog="efficiency_experiments")
    parser.add_argument("--config", default="configs/tiny_v31_sft.yaml")
    parser.add_argument("--plain-config", default="configs/tiny_v31_plain.json")
    parser.add_argument("--variants", nargs="+", default=["padded_plain", "padded_weighted"])
    parser.add_argument("--budget-tokens", type=int, default=2_000_000)
    parser.add_argument("--quality", action="store_true")
    parser.add_argument("--corpus-dir", default="data/v2")
    parser.add_argument("--output", default="experiments/efficiency.json")
    args = parser.parse_args()
    config = load_experiment_config(args.config)
    plain = json.loads(Path(args.plain_config).read_text(encoding="utf-8"))
    available = variants(plain, dict(config.train.mixture))
    tokenizer = BPETokenizer.load(config.tokenizer.path)
    results = []
    for name in args.variants:
        row = run_variant(
            config, available[name], args.budget_tokens, tokenizer, args.quality, args.corpus_dir
        )
        results.append(row)
        print(json.dumps(row), flush=True)
        record_experiment(
            f"efficiency_{name}",
            {"kind": "efficiency", "budget_tokens": args.budget_tokens, "config": args.config},
            {k: float(v) for k, v in row.items() if isinstance(v, (int, float))},
        )
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(results, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
