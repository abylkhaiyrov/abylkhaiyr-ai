from __future__ import annotations

import argparse
from pathlib import Path

from llm.evaluation.report import build_report


def main() -> int:
    parser = argparse.ArgumentParser(prog="build_report")
    parser.add_argument("--results", default="experiments")
    parser.add_argument("--output", default="docs/stage10_report.md")
    parser.add_argument("--title", default="Stage 10 measured results")
    args = parser.parse_args()
    Path(args.output).write_text(build_report(Path(args.results), args.title), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
