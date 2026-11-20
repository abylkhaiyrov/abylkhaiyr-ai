from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from llm.core.errors import ResourceError
from llm.data.mathgen import build_dataset, read_dataset
from llm.training.continuous import Example, rollback, run_cycle
from llm.training.session import exclusive_session


def seed_math(inbox: Path) -> None:
    inbox.mkdir(parents=True, exist_ok=True)
    manifest = inbox / "sources.json"
    if manifest.exists():
        raise FileExistsError(manifest)
    excluded = {problem.key for problem in read_dataset(Path("data/benchmarks/math_bench.jsonl"))}
    problems = build_dataset(480, seed=190721, exclude=excluded, levels=(1, 2, 3))
    examples = (
        Example(
            prompt="Solve this mathematical problem and show the answer. " + item.question,
            answer=item.solution,
        )
        for item in problems
    )
    (inbox / "verified_math.jsonl").write_text(
        "".join(example.model_dump_json() + "\n" for example in examples), encoding="utf-8"
    )
    manifest.write_text(
        json.dumps([{"file": "verified_math.jsonl", "license": "MIT"}]), encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inbox", type=Path, default=Path("experiments/learning_inbox"))
    parser.add_argument("--work", type=Path, default=Path("experiments/continuous"))
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--reference", default="checkpoints/tiny_gpu_sft/best")
    parser.add_argument("--evaluation-data", default="data/processed")
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--polls", type=int, choices=range(1, 25), default=1)
    parser.add_argument("--interval", type=int, default=600)
    parser.add_argument("--seed-math", action="store_true")
    parser.add_argument("--rollback", action="store_true")
    args = parser.parse_args()
    if args.interval < 60:
        parser.error("--interval must be at least 60 seconds")
    if args.rollback:
        print(rollback(args.work))
        return
    args.inbox.mkdir(parents=True, exist_ok=True)
    if args.seed_math:
        seed_math(args.inbox)
    with exclusive_session(args.work / ".service.lock"):
        for index in range(args.polls):
            if (args.work / "STOP").exists():
                break
            try:
                status = run_cycle(
                    args.inbox,
                    args.work,
                    args.steps,
                    args.device,
                    reference=args.reference,
                    evaluation_data=args.evaluation_data,
                )
                print(status, flush=True)
            except ResourceError as error:
                print(f"deferred: {error}", flush=True)
            if index + 1 < args.polls:
                deadline = time.monotonic() + args.interval
                while time.monotonic() < deadline and not (args.work / "STOP").exists():
                    time.sleep(min(1, max(0, deadline - time.monotonic())))


if __name__ == "__main__":
    main()
