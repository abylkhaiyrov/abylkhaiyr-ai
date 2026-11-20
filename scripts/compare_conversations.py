from __future__ import annotations

import argparse

from llm.evaluation.comparison import (
    category_table,
    compare_checkpoints,
    markdown_table,
    save_comparison,
)


def main() -> int:
    parser = argparse.ArgumentParser(prog="compare_conversations")
    parser.add_argument("models", nargs="+", help="name=checkpoint_path pairs")
    parser.add_argument("--tokenizer", default="data/tokenizer")
    parser.add_argument("--data-dir", default="data/v2")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--output", default="experiments/conversation_compare.json")
    args = parser.parse_args()
    models = dict(item.split("=", 1) for item in args.models)
    reports = compare_checkpoints(models, args.tokenizer, args.device, args.data_dir)
    save_comparison(reports, args.output)
    print(markdown_table(reports))
    print()
    print(category_table(reports))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
