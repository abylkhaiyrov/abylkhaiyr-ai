from __future__ import annotations

import argparse
import json

from llm.core.config import load_experiment_config
from llm.data.instruction_corpus import InstructionBuildConfig, build_instruction_corpus
from llm.tokenizer.bpe import BPETokenizer


def main() -> int:
    parser = argparse.ArgumentParser(prog="build_instruction_corpus")
    parser.add_argument("--config", default="configs/tiny_v2.yaml")
    parser.add_argument("--tokenizer", default="data/tokenizer")
    parser.add_argument("--source-dir", default="data/v2")
    parser.add_argument("--output-dir", default="data/v3")
    parser.add_argument("--synthetic-count", type=int, default=12000)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()
    build = InstructionBuildConfig(
        source_dir=args.source_dir,
        output_dir=args.output_dir,
        synthetic_count=args.synthetic_count,
        seed=args.seed,
    )
    model = load_experiment_config(args.config).model
    manifest = build_instruction_corpus(build, BPETokenizer.load(args.tokenizer), model)
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
