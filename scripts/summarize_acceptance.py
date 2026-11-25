from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from llm.evaluation.acceptance import Grade, aggregate


def load_grades(path: Path) -> dict[str, Grade]:
    raw = json.loads(path.read_text(encoding="utf-8"))["grades"]
    return {
        item_id: Grade(item_id, entry["scores"], entry["cause"], entry["note"], entry["passed"])
        for item_id, entry in raw.items()
    }


def disagreements(rows: list[dict[str, Any]], grades: dict[str, Grade]) -> list[str]:
    return [
        f"{row['item_id']}: auto={row['auto']['passed']} manual={grades[row['item_id']].passed}"
        for row in rows
        if row["item_id"] in grades
        and row["auto"]["passed"] is not None
        and row["auto"]["passed"] != grades[row["item_id"]].passed
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("report", type=Path)
    parser.add_argument("--grades", type=Path, default=None)
    args = parser.parse_args()
    report = json.loads(args.report.read_text(encoding="utf-8"))
    grades = load_grades(args.grades) if args.grades else {}
    summary = aggregate(report["rows"], grades)
    summary["benchmark_sha256"] = report["benchmark_sha256"]
    summary["finish_reasons"] = {}
    for row in report["rows"]:
        reason = row["turns"][-1]["finish_reason"]
        summary["finish_reasons"][reason] = summary["finish_reasons"].get(reason, 0) + 1
    summary["auto_manual_disagreements"] = disagreements(report["rows"], grades)
    print(json.dumps(summary, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
