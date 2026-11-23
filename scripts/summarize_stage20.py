from __future__ import annotations

import argparse
import json
import math
import statistics
from collections.abc import Sequence
from pathlib import Path

from llm.evaluation.capacity_metrics import paired_summary
from llm.evaluation.diagnostic_controls import ControlRow, choices

BANKS = ("historical1152", "diagnostic144")
LANGUAGES = ("en", "ru", "kk")
PPL_SETS = (
    "s18_validation",
    "s18_test",
    "v2_validation",
    "v2_test",
    "v2_train_head",
    "v4_validation",
)
ARMS = ("A", "B", "R")
JOBS = {
    "A": {1: "L1", 2: "A-2", 3: "A-3"},
    "B": {s: f"B-{s}" for s in (1, 2, 3)},
    "R": {s: f"R-{s}" for s in (1, 2, 3)},
}
CONTROLS = Path("experiments/stage19/controls")
REPETITIONS = 10_000


def rows_of(path: Path) -> list[ControlRow]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [ControlRow(**{**row, "lengths": tuple(row["lengths"])}) for row in payload["rows"]]


def control_path(root: Path, job: str, bank: str) -> Path:
    local = root / f"{job}.{bank}.json"
    return local if local.exists() else CONTROLS / f"{job}.{bank}.json"


def complete_seeds(root: Path) -> list[int]:
    return [
        seed
        for seed in (1, 2, 3)
        if all(control_path(root, JOBS[arm][seed], BANKS[0]).exists() for arm in ARMS)
    ]


def accuracy(rows: Sequence[ControlRow], method: str) -> float:
    return sum(getattr(row, method) == row.answer for row in rows) / len(rows)


def subset(
    rows: Sequence[ControlRow], tasks: set[str] | None, language: str | None
) -> list[ControlRow]:
    return [
        row
        for row in rows
        if (tasks is None or row.task in tasks) and (language is None or row.language == language)
    ]


def ratios(grid: dict, label: str) -> dict[str, float]:
    return {
        name: math.exp(
            sum(
                math.log(
                    grid[label][name][lang]["perplexity"]
                    / grid["v2_start"][name][lang]["perplexity"]
                )
                for lang in LANGUAGES
            )
            / len(LANGUAGES)
        )
        for name in PPL_SETS
    }


def spread(values: list[float], digits: int = 3) -> str:
    return (
        f"{statistics.fmean(values):.{digits}f} [{min(values):.{digits}f},{max(values):.{digits}f}]"
    )


def paired(
    root: Path,
    bank: str,
    first: str,
    second: str,
    method: str,
    seeds: list[int],
    tasks: set[str] | None,
    language: str | None,
) -> dict:
    pairs = [
        (
            choices(
                subset(rows_of(control_path(root, JOBS[first][seed], bank)), tasks, language),
                method,
            ),
            choices(
                subset(rows_of(control_path(root, JOBS[second][seed], bank)), tasks, language),
                method,
            ),
        )
        for seed in seeds
    ]
    return paired_summary(pairs, method="choice", repetitions=REPETITIONS)


def describe(summary: dict, confidence: str = "cluster_bootstrap_95") -> str:
    low, high = summary[confidence]
    return f"{summary['absolute_difference'] * 100:+.2f} п.п. [{low * 100:+.1f}, {high * 100:+.1f}], n={summary['n']}, seeds={summary['seeds']}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("experiments/stage20/controls"))
    parser.add_argument(
        "--perplexity",
        type=Path,
        nargs="+",
        default=[
            Path("experiments/stage20/perplexity.json"),
            Path("experiments/stage20/perplexity_l1.json"),
        ],
    )
    args = parser.parse_args()
    seeds = complete_seeds(args.root)
    print("complete seeds:", seeds)
    grid: dict = {}
    for path in args.perplexity:
        grid.update(json.loads(path.read_text(encoding="utf-8"))["perplexity"])
    print("\n| PPL к v2 | " + " | ".join(PPL_SETS) + " |\n|---|" + "---|" * len(PPL_SETS))
    table = {arm: [ratios(grid, JOBS[arm][seed]) for seed in seeds] for arm in ARMS}
    for arm in ARMS:
        print(
            f"| {arm} | "
            + " | ".join(spread([r[name] for r in table[arm]]) for name in PPL_SETS)
            + " |"
        )
    mean_v2 = {
        arm: statistics.fmean((r["v2_validation"] + r["v2_test"]) / 2 for r in table[arm])
        for arm in ARMS
    }
    print(
        "\nHB2 (R ниже B на v2 validation+test не менее чем на 0,01):",
        mean_v2["B"] - mean_v2["R"] >= 0.01,
        {k: round(v, 4) for k, v in mean_v2.items()},
    )
    print(
        "HB4 (все отношения <= 1,03 на v2 val/test и s18 val):",
        all(
            r[name] <= 1.03
            for arm in ARMS
            for r in table[arm]
            for name in ("v2_validation", "v2_test", "s18_validation")
        ),
    )
    print("\n#### Калиброванная и нормированная точность (среднее по seed [мин, макс])")
    for bank in BANKS:
        for method in ("calibrated", "normalized"):
            line = []
            for arm in ARMS:
                values = [
                    accuracy(rows_of(control_path(args.root, JOBS[arm][seed], bank)), method)
                    for seed in seeds
                ]
                line.append(f"{arm} {spread(values)}")
            print(f"- {bank}, {method}:", "; ".join(line))
    print("\n#### Парные сравнения (кластерный бутстреп по семействам, 95%)")
    for first, second in (("A", "B"), ("B", "R"), ("A", "R")):
        for bank, tasks, label in (
            ("historical1152", None, "P1 все"),
            ("diagnostic144", {"knowledge", "semantic"}, "P2 знания+семантика"),
            ("historical1152", {"knowledge"}, "знания 1152"),
            ("diagnostic144", None, "все 144"),
        ):
            for method in ("calibrated", "normalized"):
                print(
                    f"- {second} − {first}, {label}, {method}:",
                    describe(paired(args.root, bank, first, second, method, seeds, tasks, None)),
                )
    print("\n#### По языкам, калиброванная, P1-банк 1152 (разность относительно A)")
    for language in LANGUAGES:
        for other in ("B", "R"):
            print(
                f"- {other} − A, {language}:",
                describe(
                    paired(
                        args.root, "historical1152", "A", other, "calibrated", seeds, None, language
                    )
                ),
            )
    v2 = rows_of(CONTROLS / "v2_start.historical1152.json")
    print("\nHB5 (точность по языкам относительно v2, калибр., 1152):")
    for language in LANGUAGES:
        base = accuracy(subset(v2, None, language), "calibrated")
        for arm in ARMS:
            values = [
                accuracy(
                    subset(
                        rows_of(control_path(args.root, JOBS[arm][seed], "historical1152")),
                        None,
                        language,
                    ),
                    "calibrated",
                )
                for seed in seeds
            ]
            print(f"- {arm} {language}: {statistics.fmean(values) - base:+.3f}")


if __name__ == "__main__":
    main()
