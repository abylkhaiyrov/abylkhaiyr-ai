from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from llm.evaluation.capacity_metrics import paired_summary
from llm.evaluation.diagnostic_controls import METHODS, ControlRow, choices

BANKS = ("diagnostic144", "historical1152")
SHOWN = ("normalized", "total", "option_only", "neutral", "calibrated", "mismatched", "unigram")


def load(root: Path, model: str, bank: str) -> dict:
    return json.loads((root / f"{model}.{bank}.json").read_text(encoding="utf-8"))


def rows_of(payload: dict) -> list[ControlRow]:
    return [ControlRow(**{**row, "lengths": tuple(row["lengths"])}) for row in payload["rows"]]


def cell(entry: dict) -> str:
    low, high = entry["wilson_95"]
    return f"{entry['rate']:.3f} [{low:.2f},{high:.2f}]"


def accuracy_table(root: Path, models: Sequence[str], bank: str) -> None:
    print(f"\n#### {bank}\n")
    print("| модель | " + " | ".join(SHOWN) + " |")
    print("|---|" + "---|" * len(SHOWN))
    for model in models:
        report = load(root, model, bank)["report"]["accuracy"]
        print(f"| {model} | " + " | ".join(cell(report[method]) for method in SHOWN) + " |")


def interval(report: dict, key: str) -> str:
    entry = report[key]
    low, high = entry["cluster_bootstrap_95"]
    return f"{entry['absolute_difference']:+.3f} [{low:+.3f},{high:+.3f}]"


def within_table(root: Path, models: Sequence[str], bank: str) -> None:
    print(f"\n#### {bank}: внутри модели, разница доли верных\n")
    print(
        "| модель | вопрос против чужого вопроса | вопрос против только варианта | калибровка против нормировки | нормировка и сумма совпадают | выбран кратчайший (норм., сумма) |"
    )
    print("|---|---|---|---|---|---|")
    for model in models:
        report = load(root, model, bank)["report"]

        shortest = report["chosen_shortest"]
        print(
            f"| {model} | {interval(report, 'question_use')} | {interval(report, 'prompt_gain_over_option_only')} | "
            f"{interval(report, 'calibrated_vs_normalized')} | {report['agreement_normalized_total']:.3f} | "
            f"{shortest['normalized']:.2f}, {shortest['total']:.2f} |"
        )


def between(root: Path, bank: str, tasks: set[str] | None, repetitions: int) -> dict:
    result = {}
    for method in ("normalized", "total", "calibrated", "option_only"):
        pairs = []
        for first, second in (("a1", "b1"), ("a2", "b2")):
            left, right = rows_of(load(root, first, bank)), rows_of(load(root, second, bank))
            if tasks is not None:
                left = [row for row in left if row.task in tasks]
                right = [row for row in right if row.task in tasks]
            pairs.append((choices(left, method), choices(right, method)))
        summary = paired_summary(pairs, method="choice", repetitions=repetitions)
        low, high = summary["cluster_bootstrap_95"]
        result[method] = (
            f"{summary['absolute_difference'] * 100:+.2f} п.п. [{low * 100:+.1f},{high * 100:+.1f}], n={summary['n']}"
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("experiments/stage19/controls"))
    parser.add_argument(
        "--models", nargs="+", default=["v2_start", "a1", "a2", "b1", "b2", "L1", "L3", "L4", "v31"]
    )
    parser.add_argument("--repetitions", type=int, default=5000)
    args = parser.parse_args()
    for bank in BANKS:
        accuracy_table(args.root, args.models, bank)
        within_table(args.root, args.models, bank)
    print("\n#### B − A по методам (среднее по двум seed, кластерный бутстреп по семействам)\n")
    for bank, tasks in (
        ("diagnostic144", {"knowledge"}),
        ("diagnostic144", {"semantic", "negation", "causal"}),
        ("diagnostic144", None),
        ("historical1152", {"knowledge"}),
        ("historical1152", None),
    ):
        label = "все" if tasks is None else "+".join(sorted(tasks))
        print(
            f"- {bank}, {label}:",
            json.dumps(between(args.root, bank, tasks, args.repetitions), ensure_ascii=False),
        )
    print("\nМетоды:", ", ".join(METHODS))


if __name__ == "__main__":
    main()
