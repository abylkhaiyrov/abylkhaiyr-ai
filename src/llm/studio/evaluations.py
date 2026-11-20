from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from llm.core.config import load_experiment_config
from llm.core.errors import ConfigError
from llm.studio.model_registry import relative_id
from llm.studio.pipeline_view import finite

EVALUATION_STAGES = {"eval_pretrain": "pretrain", "eval_sft": "sft", "eval_final": "dpo"}


@lru_cache(maxsize=64)
def _read_report(path: Path, stamp: int) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def read_report(path: Path) -> dict[str, Any]:
    try:
        stamp = path.stat().st_mtime_ns
    except OSError:
        return {}
    return _read_report(path, stamp)


def result_metrics(data: object) -> dict[str, float]:
    if not isinstance(data, dict):
        return {}
    return {
        key: value
        for key, raw in data.items()
        if not isinstance(raw, bool) and (value := finite(raw)) is not None
    }


def pipeline_dataset(root: Path, experiment: str, state: dict[str, Any]) -> str | None:
    config = state.get("config") or {}
    dataset = config.get("data_dir") or state.get("artifacts", {}).get("data_dir")
    if isinstance(dataset, str):
        return dataset
    config_path = config.get("pretrain_config") or (
        f"configs/{experiment.removeprefix('pipeline_')}.yaml"
    )
    try:
        return load_experiment_config(root / config_path).train.data_dir
    except ConfigError:
        return None


def evaluation_reports(root: Path) -> list[dict[str, Any]]:
    reports = []
    for path in sorted((root / "experiments").glob("*/state.json")):
        state = read_report(path)
        experiment = path.parent.name
        source = str(path.relative_to(root))
        dataset = pipeline_dataset(root, experiment, state)
        for key, stage in EVALUATION_STAGES.items():
            entry = state.get("stages", {}).get(key, {})
            if entry.get("status") != "completed":
                continue
            if key == "eval_final" and entry.get("source") != "dpo":
                continue
            checkpoints = entry.get("candidates") if key == "eval_pretrain" else None
            if checkpoints is None:
                checkpoints = {
                    entry.get("checkpoint") or entry.get("selected"): entry.get("metrics")
                }
            for checkpoint, metrics in checkpoints.items():
                identifier = relative_id(root, checkpoint)
                results = result_metrics(metrics)
                if identifier and results:
                    reports.append(
                        {
                            "path": source,
                            "experiment": experiment,
                            "dataset": dataset,
                            "stage": stage,
                            "checkpoint": identifier,
                            "model": Path(identifier).parent.name,
                            "results": results,
                        }
                    )
    files = set(root.glob("experiments/**/evaluation.json")) | set(
        root.glob("experiments/*_eval.json")
    )
    for path in sorted(files):
        data = read_report(path)
        results = result_metrics(data.get("metrics", data))
        if results:
            checkpoint = relative_id(root, data.get("checkpoint"))
            reports.append(
                {
                    "path": str(path.relative_to(root)),
                    "experiment": path.parent.name if path.name == "evaluation.json" else path.stem,
                    "dataset": data.get("data_dir"),
                    "stage": data.get("stage"),
                    "checkpoint": checkpoint,
                    "model": Path(checkpoint).parent.name if checkpoint else None,
                    "results": results,
                }
            )
    return reports
