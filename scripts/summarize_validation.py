from __future__ import annotations

import argparse
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path

from llm.data.binary import read_jsonl
from llm.evaluation.v4_review import verbatim_leak

COLUMNS = (
    ("single", "single_answer_accuracy"),
    ("incomplete", "incomplete"),
    ("echo", "prompt_echo"),
    ("script", "script_match"),
)


def spread(values: list[float]) -> str:
    return f"{statistics.fmean(values):.3f} [{min(values):.3f},{max(values):.3f}]"


def leak_rate(path: Path, answers: list[str], excluded: set[str]) -> float:
    rows = [row for row in read_jsonl(path) if "response" in row and row["id"] not in excluded]
    return sum(verbatim_leak(row["response"], answers) for row in rows) / len(rows)


def main() -> int:
    parser = argparse.ArgumentParser(prog="summarize_validation")
    parser.add_argument("directory", type=Path)
    parser.add_argument("--suffix", default=".clean.json")
    parser.add_argument("--curated", type=Path)
    parser.add_argument("--exclude-ids", type=Path)
    args = parser.parse_args()
    groups: dict[str, list[dict]] = defaultdict(list)
    leaks: dict[str, list[float]] = defaultdict(list)
    answers = (
        [row["messages"][-1]["content"] for row in read_jsonl(args.curated)] if args.curated else []
    )
    excluded = set(json.loads(args.exclude_ids.read_text())) if args.exclude_ids else set()
    for path in sorted(args.directory.glob(f"*{args.suffix}")):
        stem = path.name.removesuffix(args.suffix)
        name = re.sub(r"_s\d+$", "", stem)
        groups[name].append(json.loads(path.read_text()))
        if answers:
            leaks[name].append(leak_rate(args.directory / f"{stem}.jsonl", answers, excluded))
    header = ["model", "seeds", "cases"] + [label for label, _ in COLUMNS] + ["repetition"]
    header += [f"single {lang}" for lang in ("en", "ru", "kk")]
    header += ["curated leak"] * bool(answers)
    print("| " + " | ".join(header) + " |")
    print("|" + "---|" * len(header))
    for name, reports in groups.items():
        cells = [name, str(len(reports)), str(reports[0]["overall"]["incomplete"]["n"])]
        for _, key in COLUMNS:
            cells.append(spread([r["overall"][key]["rate"] for r in reports]))
        cells.append(spread([r["overall"]["mean_repetition"] for r in reports]))
        for lang in ("en", "ru", "kk"):
            values = [
                r["by_language"][lang]["single_answer_accuracy"]["rate"]
                for r in reports
                if lang in r["by_language"]
            ]
            cells.append(spread(values) if values else "n/a")
        if answers:
            cells.append(spread(leaks[name]))
        print("| " + " | ".join(cells) + " |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
