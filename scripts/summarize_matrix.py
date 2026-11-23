from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

SENTIMENT = ("жағымды", "жағымсыз")


def kazakh_sentiment_rate(predictions: Path) -> float:
    rows = [json.loads(line) for line in predictions.read_text().splitlines()]
    kazakh = [row for row in rows if row.get("language") == "kk" and "response" in row]
    hits = sum(row["response"].strip().casefold().startswith(SENTIMENT) for row in kazakh)
    return hits / len(kazakh)


def spread(values: list[float]) -> str:
    return f"{statistics.fmean(values):.3f} [{min(values):.3f},{max(values):.3f}]"


def main() -> int:
    parser = argparse.ArgumentParser(prog="summarize_matrix")
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    groups: dict[str, list[Path]] = defaultdict(list)
    for path in sorted(args.directory.glob("[A-Z]*_s[0-9].json")):
        groups[path.stem.rsplit("_s", 1)[0]].append(path)
    print(
        "| variant | seeds | single-answer | kk single | incomplete | echo | kk sentiment label |"
    )
    print("|---|---:|---|---|---|---|---|")
    for name, paths in groups.items():
        reports = [json.loads(path.read_text()) for path in paths]
        single = [r["overall"]["single_answer_accuracy"]["rate"] for r in reports]
        kk = [r["by_language"]["kk"]["single_answer_accuracy"]["rate"] for r in reports]
        incomplete = [r["overall"]["incomplete"]["rate"] for r in reports]
        echo = [r["overall"]["prompt_echo"]["rate"] for r in reports]
        sentiment = [kazakh_sentiment_rate(path.with_suffix(".jsonl")) for path in paths]
        print(
            f"| {name} | {len(paths)} | {spread(single)} | {spread(kk)} | {spread(incomplete)}"
            f" | {spread(echo)} | {spread(sentiment)} |"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
