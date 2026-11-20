from __future__ import annotations

import argparse
import json
from pathlib import Path

from llm.training.promotion import Candidate, promote


def main() -> int:
    parser = argparse.ArgumentParser(prog="promote_model")
    parser.add_argument("results", help="JSON containing checkpoint-level test results")
    parser.add_argument("candidates", nargs="+", help="name=checkpoint_dir=stage")
    parser.add_argument("--reference", default="", help="name of the pretrained reference model")
    parser.add_argument("--root", default=".")
    args = parser.parse_args()
    results = json.loads(Path(args.results).read_text(encoding="utf-8"))
    root = Path(args.root)
    candidates = []
    for item in args.candidates:
        name, checkpoint, stage = item.split("=", 2)
        report = results[name]
        if report.get("split", "test") != "test":
            parser.error(f"{name} must use test metrics")
        recorded = report.get("checkpoint")
        if recorded and (root / recorded).resolve() != (root / checkpoint).resolve():
            parser.error(f"{name} metrics do not match the requested checkpoint")
        candidates.append(Candidate(checkpoint, stage, report["metrics"]))
    reference = results.get(args.reference, {})
    if reference.get("split", "test") != "test":
        parser.error("the reference must use test metrics")
    print(
        json.dumps(
            promote(root, candidates, reference_metrics=reference.get("metrics", {})), indent=2
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
