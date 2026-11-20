from __future__ import annotations

import argparse
import json
from pathlib import Path

from llm.training.promotion import Candidate, promote, rollback


def main() -> int:
    parser = argparse.ArgumentParser(prog="promote_model")
    parser.add_argument("results", nargs="?", help="JSON containing checkpoint-level test results")
    parser.add_argument("candidates", nargs="*", help="name=checkpoint_dir=stage")
    parser.add_argument("--reference", default="", help="name of the pretrained reference model")
    parser.add_argument("--root", default=".")
    parser.add_argument("--rollback", action="store_true")
    args = parser.parse_args()
    root = Path(args.root)
    if args.rollback:
        if args.results or args.candidates:
            parser.error("--rollback does not take results or candidates")
        print(rollback(root))
        return 0
    if not args.results or not args.candidates:
        parser.error("results and at least one candidate are required")
    results = json.loads(Path(args.results).read_text(encoding="utf-8"))
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
