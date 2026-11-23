from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

from llm.data.v4_quality import LANGUAGES
from llm.evaluation.capacity_metrics import Prediction, paired_summary
from llm.training.capacity_study import SEEDS

ROOT = Path("experiments/stage18")
PRIMARY = {"knowledge", "semantic"}


def gate_accepts(effect: dict) -> bool:
    return (
        effect["absolute_difference"] >= 0.20
        and effect["cluster_bootstrap_97.5"][0] > 0
        and all(delta > 0 for delta in effect["seed_differences"])
    )


def main() -> None:
    reports = {
        (arm, seed): json.loads(
            (ROOT / "evaluation" / f"{arm}-seed-{seed}.json").read_text(encoding="utf-8")
        )
        for arm in ("a", "b")
        for seed in SEEDS
    }
    bank = {report["diagnostic_sha256"] for report in reports.values()}
    if len(bank) != 1:
        raise ValueError("inconsistent diagnostic fingerprints")
    rows = {
        key: [Prediction(**row) for row in item["predictions"]] for key, item in reports.items()
    }
    comparisons = {}
    pairs: list[tuple[list[Prediction], list[Prediction]]]
    for task in ("all", *sorted({row.task for row in rows["a", SEEDS[0]]})):
        confidence = 0.975 if task in PRIMARY else 0.95
        pairs = [
            (
                [row for row in rows["a", seed] if task in {"all", row.task}],
                [row for row in rows["b", seed] if task in {"all", row.task}],
            )
            for seed in SEEDS
        ]
        comparisons[task] = {
            method: paired_summary(pairs, method=method, confidence=confidence)
            for method in ("normalized", "total")
        }
    languages = {
        language: paired_summary(
            [
                (
                    [row for row in rows["a", seed] if row.language == language],
                    [row for row in rows["b", seed] if row.language == language],
                )
                for seed in SEEDS
            ]
        )
        for language in LANGUAGES
    }
    option_only = {}
    for task in sorted(PRIMARY):
        conditioned = [[row for row in rows["b", seed] if row.task == task] for seed in SEEDS]
        pairs = [
            ([replace(row, normalized=row.option_only) for row in items], items)
            for items in conditioned
        ]
        option_only[task] = paired_summary(pairs)
    ratios = {
        split: {
            lang: [
                reports["b", seed]["language"][split][lang]["perplexity"]
                / reports["a", seed]["language"][split][lang]["perplexity"]
                for seed in SEEDS
            ]
            for lang in LANGUAGES
        }
        for split in ("validation", "test")
    }
    qualified = [
        task
        for task in sorted(PRIMARY)
        if gate_accepts(comparisons[task]["normalized"])
        and all(delta > 0 for delta in option_only[task]["seed_differences"])
    ]
    ppl_gate = all(
        ratio <= 1.10
        for languages in ratios.values()
        for seed_values in languages.values()
        for ratio in seed_values
    )
    result = {
        "n_seeds": len(SEEDS),
        "diagnostic": comparisons,
        "by_language": languages,
        "b_vs_b_option_only": option_only,
        "perplexity_ratios": ratios,
        "perplexity_gate": ppl_gate,
        "qualified_tasks": qualified,
        "proceed_to_sft": bool(qualified and ppl_gate),
    }
    output = ROOT / "paired_comparisons.json"
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"qualified_tasks": qualified, "perplexity_gate": ppl_gate}))


if __name__ == "__main__":
    main()
