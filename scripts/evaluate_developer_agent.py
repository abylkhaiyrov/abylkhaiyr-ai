from __future__ import annotations

import argparse
from pathlib import Path

import torch
from pydantic import BaseModel

from llm.developer_agent import AcceptanceCase, DeveloperResult, DeveloperTask, solve
from llm.evaluation.comparison import load_engine


class AgentBenchmark(BaseModel):
    checkpoint: str
    tasks: dict[str, DeveloperResult]
    pass_at_1: float
    task_completion_rate: float


def task(instruction: str, pairs: tuple[tuple[str, str], ...]) -> DeveloperTask:
    return DeveloperTask(
        instruction=instruction,
        cases=[AcceptanceCase(stdin=stdin, stdout=stdout) for stdin, stdout in pairs],
        attempts=3,
    )


TASKS = {
    "sum": task(
        "Read two whitespace-separated integers and print their sum as a single integer.",
        (("3 4\n", "7"), ("-6 2\n", "-4"), ("12 0\n", "12")),
    ),
    "algorithm": task(
        "Read one line of integers and print the unique integers in ascending order, separated by spaces.",
        (("3 1 2 1\n", "1 2 3"), ("-6 2 -6\n", "-6 2"), ("1 1 1\n", "1")),
    ),
    "bug_fix": task(
        "Fix this Python CLI: print(input()[1:]). It should print the entire input line reversed.",
        (("beta\n", "ateb"), ("q\n", "q"), ("az 31\n", "13 za")),
    ),
    "sql": task(
        "Use sqlite3 in memory. Read a line of integers into a table and print the SQL COUNT of positive rows.",
        (("3 -4 2 0\n", "2"), ("-3 -3 -1\n", "0"), ("3 4 2 1\n", "4")),
    ),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--checkpoint", default="checkpoints/tiny_gpu_sft/best")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    engine = load_engine(args.checkpoint, "data/tokenizer", "cpu")
    results: dict[str, DeveloperResult] = {}
    for name, spec in TASKS.items():
        results[name] = solve(engine, spec, args.output / name, args.image)
    report = AgentBenchmark(
        checkpoint=args.checkpoint,
        tasks=results,
        pass_at_1=sum(
            result.status == "passed" and len(result.attempts) == 1 for result in results.values()
        )
        / len(TASKS),
        task_completion_rate=sum(result.status == "passed" for result in results.values())
        / len(TASKS),
    )
    output = report.model_dump_json(indent=2)
    (args.output / "report.json").write_text(output, encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
