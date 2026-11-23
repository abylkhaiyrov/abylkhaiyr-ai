from __future__ import annotations

import argparse
import json
from pathlib import Path


def cell(entry: dict) -> str:
    if not entry["n"]:
        return "n/a"
    low, high = entry["wilson_95"]
    return f"{entry['rate']:.2f} [{low:.2f},{high:.2f}] n={entry['n']}"


def main() -> int:
    parser = argparse.ArgumentParser(prog="compare_natural")
    parser.add_argument("reports", nargs="+", type=Path)
    parser.add_argument("--metric", default="single_answer_accuracy")
    args = parser.parse_args()
    print("| model | overall | en | ru | kk | echo | incomplete | script | repetition |")
    print("|---|---|---|---|---|---|---|---|---|")
    for path in args.reports:
        report = json.loads(path.read_text())
        overall = report["overall"]
        languages = report["by_language"]
        print(
            f"| {path.stem} | {cell(overall[args.metric])} | "
            + " | ".join(cell(languages[lang][args.metric]) for lang in ("en", "ru", "kk"))
            + f" | {cell(overall['prompt_echo'])} | {cell(overall['incomplete'])}"
            f" | {cell(overall['script_match'])} | {overall['mean_repetition']:.3f} |"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
