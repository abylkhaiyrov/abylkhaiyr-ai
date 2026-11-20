from __future__ import annotations

import argparse
import json
from pathlib import Path

from llm.training.promotion import Candidate, promote


def main() -> int:
    parser = argparse.ArgumentParser(prog="promote_model")
    parser.add_argument("results", help="JSON written by compare_conversations.py")
    parser.add_argument("candidates", nargs="+", help="name=checkpoint_dir=stage")
    parser.add_argument("--reference", default="", help="name of the pretrained reference model")
    parser.add_argument("--root", default=".")
    args = parser.parse_args()
    results = json.loads(Path(args.results).read_text(encoding="utf-8"))
    candidates = []
    for item in args.candidates:
        name, checkpoint, stage = item.split("=", 2)
        candidates.append(Candidate(checkpoint, stage, results[name]["metrics"]))
    reference = results.get(args.reference, {}).get("metrics", {}).get("conv_pass_rate", 0.0)
    print(json.dumps(promote(Path(args.root), candidates, reference), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
