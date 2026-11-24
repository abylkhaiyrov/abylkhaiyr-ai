from __future__ import annotations

import sys

from llm.cli.main import main

SPEC = "configs/experiments/stage19.yaml"


def translate(argv: list[str]) -> list[str]:
    if argv[:1] == ["--run"]:
        return ["experiment", "run", SPEC]
    if argv[:1] == ["--job"] and len(argv) > 1:
        return ["experiment", "job", SPEC, argv[1]]
    return ["experiment", *argv[:1], SPEC, *argv[1:]]


if __name__ == "__main__":
    raise SystemExit(main(translate(sys.argv[1:])))
