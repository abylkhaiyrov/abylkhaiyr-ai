from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from llm.evaluation.capacity_metrics import Prediction, paired_summary
from llm.training.capacity_study import SEEDS

SLICES = {
    "all": set(),
    "knowledge": {"knowledge"},
    "semantic": {"semantic"},
    "knowledge_semantic": {"knowledge", "semantic"},
    "instruction": {"instruction"},
    "context": {"context"},
}


def comparisons(
    first: list[list[Prediction]],
    second: list[list[Prediction]],
    *,
    method: str = "normalized",
) -> dict:
    if not first or len(first) != len(second):
        raise ValueError("expected one ordered pair of outputs per seed")
    result = {}
    slices = {**SLICES, "without_grammar": {row.task for row in first[0]} - {"grammar"}}
    for group, tasks in slices.items():
        pairs = [
            (
                [row for row in left if not tasks or row.task in tasks],
                [row for row in right if not tasks or row.task in tasks],
            )
            for left, right in zip(first, second, strict=True)
        ]
        result[group] = paired_summary(pairs, method=method)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("experiments/stage17"))
    args = parser.parse_args()
    bank = json.loads((args.root / "preflight_data.json").read_text(encoding="utf-8"))
    directory = args.root / "diagnostic"
    reports = {}
    for path in sorted(directory.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        if data["bank_sha256"] != bank["bank_sha256"]:
            parser.error(f"stale diagnostic report: {path}")
        reports[path.stem] = [Prediction(**item) for item in data["items"]]
    aggregate = {}
    for prefix, left, right in (
        ("equal_tokens", "tiny-5m", "small-5m"),
        ("equal_flops", "tiny-5m", "small-equivalent"),
        ("tiny_learning", "tiny-init", "tiny-5m"),
        ("small_learning", "small-init", "small-5m"),
        ("small_more_tokens", "small-equivalent", "small-5m"),
    ):
        paired = [
            (reports[f"{left}-{seed}"], reports[f"{right}-{seed}"])
            for seed in SEEDS
            if f"{left}-{seed}" in reports and f"{right}-{seed}" in reports
        ]
        if paired:
            first_runs = [first for first, _ in paired]
            second_runs = [second for _, second in paired]
            aggregate[prefix] = comparisons(first_runs, second_runs)
            if prefix in {"equal_tokens", "equal_flops"}:
                aggregate[f"{prefix}_total_logprob"] = comparisons(
                    first_runs, second_runs, method="total"
                )
    for prefix in ("tiny-5m", "small-5m", "small-equivalent"):
        model = [reports[f"{prefix}-{seed}"] for seed in SEEDS if f"{prefix}-{seed}" in reports]
        if model:
            independent = [
                [replace(row, normalized=row.option_only) for row in run] for run in model
            ]
            frequency = [[replace(row, normalized=row.unigram) for row in run] for run in model]
            aggregate[f"{prefix}_vs_option_only"] = comparisons(independent, model)
            aggregate[f"{prefix}_vs_unigram"] = comparisons(frequency, model)
    report = json.dumps(aggregate, indent=2, allow_nan=False)
    (args.root / "paired_comparisons.json").write_text(report + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
