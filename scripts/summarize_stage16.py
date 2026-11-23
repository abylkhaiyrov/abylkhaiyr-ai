from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

HARD_GROUPS = (
    "parametric_knowledge",
    "commonsense",
    "semantics",
    "instruction",
    "language_completion",
)
CONTEXT_CATEGORIES = ("context", "context_multi", "self_correction", "correction")
LANGUAGES = ("en", "ru", "kk")
VARIANTS = ("A", "B", "C")


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def diag_hard(report: dict) -> float:
    cells = [report["by_group"][group] for group in HARD_GROUPS]
    return sum(cell["successes"] for cell in cells) / sum(cell["n"] for cell in cells)


def context_accuracy(report: dict) -> float:
    cells = [
        report["by_category"][name]["single_answer_accuracy"]
        for name in CONTEXT_CATEGORIES
        if name in report["by_category"]
    ]
    return sum(cell["successes"] for cell in cells) / sum(cell["n"] for cell in cells)


def perplexity(report: dict) -> dict[str, float]:
    (entry,) = report.values()
    return {language: entry[language]["perplexity"] for language in LANGUAGES}


def spread(values: list[float], digits: int = 3) -> str:
    return (
        f"{statistics.fmean(values):.{digits}f} [{min(values):.{digits}f},{max(values):.{digits}f}]"
    )


def collect(root: Path, seeds: list[int]) -> dict[str, dict[str, list]]:
    table: dict[str, dict[str, list]] = {}
    for variant in VARIANTS:
        rows: dict[str, list] = {
            key: []
            for key in (
                "p1",
                "p2_pre",
                "p2_post",
                "context",
                "incomplete",
                "echo",
                "repetition",
                "ppl",
                "grammar_pre",
                "extraction_pre",
            )
        }
        for seed in seeds:
            validation = load(root / "val" / f"{variant}_s{seed}.clean.json")
            pre = load(root / "diag" / f"v6_{variant}_s{seed}_pre.json")
            post = load(root / "diag" / f"{variant}_s{seed}_sft.json")
            overall = validation["overall"]
            rows["p1"].append(overall["single_answer_accuracy"]["rate"])
            rows["context"].append(context_accuracy(validation))
            rows["incomplete"].append(overall["incomplete"]["rate"])
            rows["echo"].append(overall["prompt_echo"]["rate"])
            rows["repetition"].append(overall["mean_repetition"])
            rows["p2_pre"].append(diag_hard(pre))
            rows["p2_post"].append(diag_hard(post))
            rows["grammar_pre"].append(pre["by_group"]["grammar"]["rate"])
            rows["extraction_pre"].append(pre["by_group"]["in_context_extraction"]["rate"])
            rows["ppl"].append(perplexity(load(root / "lang" / f"v6_{variant}_s{seed}.json")))
        table[variant] = rows
    return table


def separated(first: list[float], second: list[float], margin: float) -> bool:
    return statistics.fmean(first) - statistics.fmean(second) >= margin and min(first) > max(second)


def pooled(reports: list[dict], section: str, key: str, metric: str) -> tuple[int, int]:
    cells = [report[section][key][metric] for report in reports if key in report[section]]
    return sum(cell["successes"] for cell in cells), sum(cell["n"] for cell in cells)


def group_table(root: Path, seeds: list[int], section: str) -> None:
    models = {
        variant: [load(root / "val" / f"{variant}_s{seed}.clean.json") for seed in seeds]
        for variant in VARIANTS
    }
    models["v31"] = [load(root / "val" / "v31.clean.json")]
    keys = sorted(
        {key for reports in models.values() for report in reports for key in report[section]}
    )
    print(
        f"| {section[3:]} | "
        + " | ".join(f"{name} single / incomplete / echo" for name in models)
        + " |"
    )
    print("|---|" + "---|" * len(models))
    for key in keys:
        cells = []
        for reports in models.values():
            single = pooled(reports, section, key, "single_answer_accuracy")
            incomplete = pooled(reports, section, key, "incomplete")
            echo = pooled(reports, section, key, "prompt_echo")
            single_text = f"{single[0]}/{single[1]}" if single[1] else "-"
            cells.append(f"{single_text} / {incomplete[0]}/{incomplete[1]} / {echo[0]}/{echo[1]}")
        print(f"| {key} | " + " | ".join(cells) + " |")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("experiments/v6"))
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2])
    parser.add_argument("--tables", action="store_true")
    args = parser.parse_args()
    table = collect(args.root, args.seeds)
    print(
        "| variant | PPL EN | PPL RU | PPL KK | diag_hard pre | grammar pre | extraction pre | diag_hard post | P1 single | context | incomplete | echo | repetition |"
    )
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for variant, rows in table.items():
        ppl = [spread([item[language] for item in rows["ppl"]], 2) for language in LANGUAGES]
        cells = [
            spread(rows[key])
            for key in (
                "p2_pre",
                "grammar_pre",
                "extraction_pre",
                "p2_post",
                "p1",
                "context",
                "incomplete",
                "echo",
                "repetition",
            )
        ]
        print(f"| {variant} | " + " | ".join([*ppl, *cells]) + " |")
    a, b, c = table["A"], table["B"], table["C"]
    ratio = {
        name: {
            language: statistics.fmean(i[language] for i in rows["ppl"])
            / statistics.fmean(i[language] for i in a["ppl"])
            for language in LANGUAGES
        }
        for name, rows in (("B", b), ("C", c))
    }
    verdict = {
        "H2_B_over_A_P1": separated(b["p1"], a["p1"], 0.05),
        "H3_B_over_A_diag_hard_pre": separated(b["p2_pre"], a["p2_pre"], 0.05),
        "H4_format_only_P1": abs(statistics.fmean(b["p1"]) - statistics.fmean(c["p1"])) < 0.03,
        "H4_format_only_diag_hard": abs(
            statistics.fmean(b["p2_pre"]) - statistics.fmean(c["p2_pre"])
        )
        < 0.03,
        "H5_ppl_ratio_B": {k: round(v, 3) for k, v in ratio["B"].items()},
        "H5_ppl_ratio_C": {k: round(v, 3) for k, v in ratio["C"].items()},
        "H5_holds_B": all(v <= 1.10 for v in ratio["B"].values()),
        "H5_holds_C": all(v <= 1.10 for v in ratio["C"].values()),
    }
    print(json.dumps(verdict, indent=1))
    if args.tables:
        group_table(args.root, args.seeds, "by_group")
        group_table(args.root, args.seeds, "by_language")


if __name__ == "__main__":
    main()
