from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any

from llm.studio.evaluation_metrics import BY_KEY
from llm.studio.evaluations import evaluation_reports
from llm.training.promotion import INSTRUCTION_STAGES, MAX_REPETITION, READY_PASS_RATE

DECISIVE_METRIC = "conv_pass_rate"
MIN_DECISIVE_GAP = 0.05


class IncompatibleEvaluations(ValueError):
    pass


def group_id(experiment: str, dataset: str | None, methodology: str | None) -> str:
    raw = f"{experiment}|{dataset or ''}|{methodology or ''}"
    return f"{experiment}:{hashlib.sha1(raw.encode()).hexdigest()[:8]}"


def quality_gates(stage: str | None, results: dict[str, float]) -> list[dict[str, Any]]:
    pass_rate = results.get("conv_pass_rate")
    repetition = results.get("conv_repetition_ratio")
    return [
        {
            "id": "instruction_stage",
            "status": "not_verified"
            if stage is None
            else "passed"
            if stage in INSTRUCTION_STAGES
            else "failed",
            "value": stage,
            "threshold": "/".join(INSTRUCTION_STAGES),
        },
        {
            "id": "conversational_pass_rate",
            "status": "not_verified"
            if pass_rate is None
            else "passed"
            if pass_rate >= READY_PASS_RATE
            else "failed",
            "value": pass_rate,
            "threshold": READY_PASS_RATE,
        },
        {
            "id": "repetition",
            "status": "not_verified"
            if repetition is None
            else "passed"
            if repetition <= MAX_REPETITION
            else "failed",
            "value": repetition,
            "threshold": MAX_REPETITION,
        },
    ]


def gate_summary(gates: list[dict[str, Any]]) -> str:
    statuses = {item["status"] for item in gates}
    if "failed" in statuses:
        return "failed"
    return "not_verified" if "not_verified" in statuses else "passed"


def build_groups(root: Path) -> dict[str, dict[str, Any]]:
    groups: dict[str, dict[str, Any]] = {}
    for report in evaluation_reports(root):
        identifier = group_id(report["experiment"], report["dataset"], report.get("methodology"))
        group = groups.setdefault(
            identifier,
            {
                "id": identifier,
                "experiment": report["experiment"],
                "dataset": report["dataset"],
                "methodology": report.get("methodology"),
                "split": report.get("split"),
                "rows": {},
            },
        )
        key = report["checkpoint"] or report["path"]
        row = group["rows"].setdefault(
            key,
            {
                "checkpoint": report["checkpoint"],
                "model": report["model"],
                "stage": report["stage"],
                "evaluated_at": report.get("evaluated_at"),
                "sources": [],
                "results": {},
            },
        )
        row["results"].update(report["results"])
        row["sources"].append(report["path"])
        row["stage"] = row["stage"] or report["stage"]
        if (report.get("evaluated_at") or "") > (row["evaluated_at"] or ""):
            row["evaluated_at"] = report["evaluated_at"]
    return groups


def infer_stage(checkpoint: str | None) -> str | None:
    name = (checkpoint or "").lower()
    return next((stage for stage in ("dpo", "sft") if stage in name), None)


def finalize_row(row: dict[str, Any]) -> dict[str, Any]:
    stage = row["stage"] or infer_stage(row["checkpoint"])
    gates = quality_gates(stage, row["results"])
    return {**row, "stage": stage, "gates": gates, "gate_status": gate_summary(gates)}


def summarize(group: dict[str, Any]) -> dict[str, Any]:
    rows = [finalize_row(item) for item in group["rows"].values()]
    return {
        "id": group["id"],
        "experiment": group["experiment"],
        "dataset": group["dataset"],
        "methodology": group["methodology"],
        "split": group["split"],
        "checkpoints": len(rows),
        "evaluated_at": max((item["evaluated_at"] or "" for item in rows), default="") or None,
        "metric_count": len({key for item in rows for key in item["results"]}),
        "gate_status": gate_summary([{"status": item["gate_status"]} for item in rows]),
    }


def list_groups(root: Path, query: str | None, offset: int, limit: int) -> dict[str, Any]:
    needle = (query or "").strip().lower()
    items = [
        summarize(group)
        for group in build_groups(root).values()
        if not needle or needle in f"{group['experiment']} {group['dataset'] or ''}".lower()
    ]
    items.sort(key=lambda item: item["evaluated_at"] or "", reverse=True)
    return {"items": items[offset : offset + limit], "total": len(items)}


def group_detail(root: Path, identifier: str) -> dict[str, Any] | None:
    group = build_groups(root).get(identifier)
    if group is None:
        return None
    rows = sorted(
        (finalize_row(item) for item in group["rows"].values()),
        key=lambda item: (item["stage"] or "", item["checkpoint"] or ""),
    )
    return {**summarize(group), "rows": rows}


def relative_change(base: float, value: float) -> float | None:
    return None if base == 0 or not math.isfinite(base) else (value - base) / abs(base)


def compare(root: Path, identifier: str, checkpoints: list[str]) -> dict[str, Any] | None:
    detail = group_detail(root, identifier)
    if detail is None:
        return None
    rows = {item["checkpoint"]: item for item in detail["rows"]}
    missing = [item for item in checkpoints if item not in rows]
    if missing:
        raise IncompatibleEvaluations(",".join(missing))
    chosen = [rows[item] for item in checkpoints]
    keys = [key for key in BY_KEY if any(key in row["results"] for row in chosen)]
    baseline = chosen[0]
    metrics = []
    for key in keys:
        definition = BY_KEY[key]
        values = [row["results"].get(key) for row in chosen]
        base = baseline["results"].get(key)
        entries = []
        for value in values:
            comparable = base is not None and value is not None
            delta = value - base if comparable else None
            better = None
            if delta is not None and delta != 0:
                better = (delta > 0) == (definition.direction == "higher")
            entries.append(
                {
                    "value": value,
                    "delta": delta,
                    "relative": relative_change(base, value) if comparable else None,
                    "improved": better,
                    "ci": None,
                    "samples": None,
                }
            )
        metrics.append(
            {
                "key": key,
                "direction": definition.direction,
                "section": definition.section,
                "entries": entries,
            }
        )
    return {
        "group": {
            key: detail[key] for key in ("id", "experiment", "dataset", "methodology", "split")
        },
        "checkpoints": [
            {
                key: row[key]
                for key in ("checkpoint", "model", "stage", "gates", "gate_status", "evaluated_at")
            }
            for row in chosen
        ],
        "metrics": metrics,
        "caveats": [
            caveat
            for caveat, active in (
                ("dataset_not_recorded", detail["dataset"] is None),
                ("methodology_not_recorded", detail["methodology"] is None),
                ("samples_not_recorded", True),
                ("confidence_intervals_not_computed", True),
            )
            if active
        ],
        "verdict": verdict(chosen),
    }


def verdict(rows: list[dict[str, Any]]) -> dict[str, Any]:
    scored = [(row, row["results"].get(DECISIVE_METRIC)) for row in rows]
    if len(rows) < 2 or any(score is None for _, score in scored):
        return {
            "status": "insufficient_data",
            "checkpoint": None,
            "reason": "missing_decisive_metric",
        }
    ordered = sorted(scored, key=lambda item: item[1], reverse=True)
    top, runner = ordered[0], ordered[1]
    if top[1] - runner[1] < MIN_DECISIVE_GAP:
        return {
            "status": "inconclusive",
            "checkpoint": None,
            "reason": "difference_below_threshold",
        }
    if top[0]["gate_status"] == "failed":
        return {
            "status": "inconclusive",
            "checkpoint": None,
            "reason": "leader_fails_quality_gates",
        }
    return {"status": "leader", "checkpoint": top[0]["checkpoint"], "reason": None}
