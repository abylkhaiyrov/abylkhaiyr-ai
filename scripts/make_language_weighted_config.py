from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from llm.training.mixture import adjust_for_languages, language_scores


def main() -> int:
    parser = argparse.ArgumentParser(prog="make_language_weighted_config")
    parser.add_argument("config")
    parser.add_argument(
        "metrics", help="JSON with conv_lang_* scores (a pipeline state or metrics)"
    )
    parser.add_argument("output")
    parser.add_argument("--strength", type=float, default=1.0)
    args = parser.parse_args()
    metrics = json.loads(Path(args.metrics).read_text(encoding="utf-8"))
    metrics = metrics.get("artifacts", {}).get("sft_metrics", metrics)
    scores = language_scores(metrics)
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    mixture = config["train"]["mixture"]
    config["train"]["mixture"] = {
        k: round(v, 4) for k, v in adjust_for_languages(mixture, scores, args.strength).items()
    }
    Path(args.output).write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    print(json.dumps({"scores": scores, "mixture": config["train"]["mixture"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
