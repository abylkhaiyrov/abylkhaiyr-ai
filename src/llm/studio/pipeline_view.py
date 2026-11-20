from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from llm.progress import read_progress
from llm.studio.telemetry import read_metrics

PROGRESS_FRESH_SECONDS = 180.0
EVAL_HEADLINE = (
    "test_perplexity",
    "test_loss",
    "next_token_accuracy",
    "arithmetic_exact_match",
    "math_bench_model_accuracy",
    "math_bench_tool_accuracy",
    "instruction_following_accuracy",
    "code_pass_at_1",
    "repeated_trigram_ratio_greedy",
    "generation_tokens_per_second",
)
TRAIN_RESULT_KEYS = {
    "train/loss": "train_loss",
    "validation/loss": "validation_loss",
    "validation/perplexity": "validation_perplexity",
    "train/learning_rate": "learning_rate",
    "train/grad_norm": "grad_norm",
    "perf/tokens_per_second": "tokens_per_second",
    "perf/tokens_seen": "tokens_seen",
    "dpo/reward_margin": "reward_margin",
    "dpo/reward_accuracy": "reward_accuracy",
}
FINISHED = {"completed", "skipped"}


@dataclass(frozen=True)
class StageConfig:
    output_dir: str
    checkpoint_dir: str
    max_steps: int
    data_dir: str


@dataclass(frozen=True)
class StageSpec:
    key: str
    kind: str
    config_key: str | None = None


STAGE_SPECS = (
    StageSpec("data_preparation", "data"),
    StageSpec("tokenization", "tokenize"),
    StageSpec("pretrain", "train", "pretrain"),
    StageSpec("eval_pretrain", "eval"),
    StageSpec("select_best_pretrain", "select"),
    StageSpec("sft", "train", "sft"),
    StageSpec("eval_sft", "eval"),
    StageSpec("dpo", "train", "dpo"),
    StageSpec("eval_final", "eval"),
    StageSpec("export_release", "export"),
)


def parse_time(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def iso_from_mtime(path: Path) -> str | None:
    try:
        return datetime.fromtimestamp(path.stat().st_mtime, UTC).isoformat()
    except OSError:
        return None


def finite(value: Any) -> float | None:
    return float(value) if isinstance(value, int | float) and math.isfinite(value) else None


def percent_of(done: float | None, total: float | None) -> float | None:
    if done is None or not total or total <= 0:
        return None
    return min(100.0, max(0.0, done / total * 100))


def blank_stage(spec: StageSpec) -> dict[str, Any]:
    return {
        "key": spec.key,
        "kind": spec.kind,
        "status": "pending",
        "started_at": None,
        "finished_at": None,
        "duration_seconds": None,
        "done": None,
        "total": None,
        "unit": None,
        "percent": None,
        "indeterminate": False,
        "eta_seconds": None,
        "phase": None,
        "detail": {},
        "results": {},
        "error": None,
        "note": None,
        "output_dir": None,
        "checkpoint_dir": None,
    }


def apply_state_entry(stage: dict[str, Any], entry: dict[str, Any], alive: bool) -> None:
    status = str(entry.get("status", "pending"))
    if status == "running" and not alive:
        status = "interrupted"
    stage["status"] = status
    stage["started_at"] = entry.get("started_at")
    if status in FINISHED | {"failed"}:
        stage["finished_at"] = entry.get("updated_at")
    stage["error"] = entry.get("error")
    stage["note"] = entry.get("reason")
    regression = re.fullmatch(
        r"SFT perplexity ([\d.]+) regressed > ([\d.]+)x pretrain ([\d.]+)",
        stage["note"] or "",
    )
    if regression:
        sft, limit, pretrain = map(float, regression.groups())
        stage["note_code"] = "sft_perplexity_regression"
        stage["note_values"] = {"sft": sft, "limit": limit, "pretrain": pretrain}
    started, ended = parse_time(stage["started_at"]), parse_time(stage["finished_at"])
    if started and ended:
        stage["duration_seconds"] = max((ended - started).total_seconds(), 0.0)


def stamp_running_duration(stage: dict[str, Any], now: datetime) -> None:
    started = parse_time(stage["started_at"])
    if stage["status"] == "running" and started:
        stage["duration_seconds"] = max((now - started).total_seconds(), 0.0)


def train_stage_info(stage: dict[str, Any], root: Path, config: StageConfig, now: datetime) -> None:
    max_steps = config.max_steps
    stage["output_dir"] = config.output_dir
    stage["checkpoint_dir"] = config.checkpoint_dir
    directory = root / config.output_dir
    records = read_metrics(directory / "metrics.jsonl", 100000)
    latest: dict[str, float] = {}
    for record in records:
        latest.update(record)
    results = {
        alias: finite(latest.get(source))
        for source, alias in TRAIN_RESULT_KEYS.items()
        if finite(latest.get(source)) is not None
    }
    stage["unit"] = "steps"
    stage["total"] = max_steps
    step = latest.get("step")
    tracker = read_progress(directory).get("train")
    fresh = False
    if isinstance(tracker, dict):
        updated = parse_time(tracker.get("updated_at"))
        fresh = updated is not None and (now - updated).total_seconds() < PROGRESS_FRESH_SECONDS
        if stage["status"] == "running" and fresh:
            stage["phase"] = tracker.get("detail", {}).get("phase")
            stage["detail"] = tracker.get("detail", {})
            step = max(float(step or 0), float(tracker.get("done") or 0))
            stage["eta_seconds"] = finite(tracker.get("eta_seconds"))
    if stage["status"] == "completed":
        final = stage["results"].get("final_step")
        step = final if isinstance(final, int | float) else max_steps
    if stage["status"] == "running" and stage["eta_seconds"] is None:
        stage["eta_seconds"] = finite(latest.get("perf/eta_seconds"))
    stage["done"] = step
    stage["percent"] = percent_of(step, max_steps)
    stage["results"] = {**stage["results"], **results}


def eval_stage_info(stage: dict[str, Any], tracked: dict[str, Any] | None, now: datetime) -> None:
    stage["unit"] = "benchmarks"
    if tracked is None:
        stage["indeterminate"] = stage["status"] == "running"
        return
    updated = parse_time(tracked.get("updated_at"))
    stale = (
        stage["status"] == "running"
        and updated is not None
        and (now - updated).total_seconds() > PROGRESS_FRESH_SECONDS * 4
    )
    stage["done"] = tracked.get("done")
    stage["total"] = tracked.get("total")
    stage["percent"] = percent_of(tracked.get("done"), tracked.get("total"))
    stage["detail"] = {**tracked.get("detail", {}), "rate": tracked.get("rate")}
    stage["indeterminate"] = stage["total"] is None
    if stage["status"] == "running" and not stale:
        stage["eta_seconds"] = finite(tracked.get("eta_seconds"))


def headline_metrics(metrics: dict[str, Any]) -> dict[str, float]:
    return {
        key: float(metrics[key]) for key in EVAL_HEADLINE if finite(metrics.get(key)) is not None
    }


def eval_results(entry: dict[str, Any]) -> dict[str, Any]:
    results: dict[str, Any] = {}
    candidates = entry.get("candidates")
    metrics = entry.get("metrics")
    if isinstance(candidates, dict) and candidates:
        scored = {
            name: item
            for name, item in candidates.items()
            if isinstance(item, dict) and finite(item.get("test_perplexity")) is not None
        }
        results["checkpoints_evaluated"] = len(candidates) + (1 if entry.get("baseline") else 0)
        if scored:
            best = min(scored, key=lambda name: scored[name]["test_perplexity"])
            results["best_checkpoint"] = Path(best).name
            results.update(headline_metrics(scored[best]))
        baseline = entry.get("baseline")
        if isinstance(baseline, dict) and finite(baseline.get("test_perplexity")) is not None:
            results["baseline_test_perplexity"] = float(baseline["test_perplexity"])
    elif isinstance(metrics, dict):
        results.update(headline_metrics(metrics))
    if entry.get("selected"):
        results["checkpoint"] = Path(str(entry["selected"])).name
        results["source"] = entry.get("source")
    return results


def stage_results_from_entry(spec: StageSpec, entry: dict[str, Any]) -> dict[str, Any]:
    if spec.kind == "train":
        metrics = entry.get("metrics")
        return dict(metrics) if isinstance(metrics, dict) else {}
    if spec.kind == "eval":
        return eval_results(entry)
    if spec.kind == "select":
        keys = ("test_perplexity", "baseline_perplexity")
        results: dict[str, Any] = {
            key: finite(entry.get(key)) for key in keys if finite(entry.get(key)) is not None
        }
        if entry.get("selected"):
            results["checkpoint"] = Path(str(entry["selected"])).name
        return results
    if spec.kind == "export" and entry.get("release_path"):
        return {"release_path": entry["release_path"]}
    return {}


def data_stages(root: Path, data_dir: str | None) -> tuple[dict[str, Any], dict[str, Any]]:
    prepared = blank_stage(STAGE_SPECS[0])
    tokenized = blank_stage(STAGE_SPECS[1])
    if not data_dir:
        return prepared, tokenized
    manifest_path = root / data_dir / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return prepared, tokenized
    prepared["status"] = "completed"
    prepared["finished_at"] = iso_from_mtime(manifest_path)
    prepared["results"] = {
        "dataset": manifest.get("version") or data_dir,
        "sources": len(manifest.get("sources", {})),
        "documents": sum(manifest.get("splits", {}).values()),
    }
    train_tokens = finite(manifest.get("tokenized", {}).get("train", {}).get("tokens"))
    if train_tokens is not None:
        tokenized["status"] = "completed"
        tokenized["finished_at"] = prepared["finished_at"]
        tokenized["results"] = {"train_tokens": train_tokens}
    return prepared, tokenized


def overall_progress(stages: list[dict[str, Any]], running: bool, now: datetime) -> dict[str, Any]:
    finished = sum(stage["status"] in FINISHED for stage in stages)
    current = next((stage for stage in stages if stage["status"] == "running"), None)
    fraction = 0.0
    if current is not None and current["percent"] is not None:
        fraction = current["percent"] / 100
    first = next((parse_time(stage["started_at"]) for stage in stages if stage["started_at"]), None)
    return {
        "finished": finished,
        "total": len(stages),
        "percent": (finished + fraction) / len(stages) * 100,
        "elapsed_seconds": (now - first).total_seconds() if first and running else None,
    }


def build_pipeline_view(
    root: Path,
    state: dict[str, Any],
    state_dir: Path,
    configs: dict[str, StageConfig | None],
    alive: bool,
    now: datetime,
) -> list[dict[str, Any]]:
    entries = state.get("stages", {})
    tracked = read_progress(state_dir)
    pretrain = configs.get("pretrain")
    prepared, tokenized = data_stages(root, pretrain.data_dir if pretrain else None)
    stages: list[dict[str, Any]] = [prepared, tokenized]
    for spec in STAGE_SPECS[2:]:
        stage = blank_stage(spec)
        entry = entries.get(spec.key, {})
        apply_state_entry(stage, entry, alive)
        stage["results"] = stage_results_from_entry(spec, entry)
        if spec.kind == "train":
            config = configs.get(spec.key)
            if config is not None and stage["status"] != "pending":
                train_stage_info(stage, root, config, now)
        elif spec.kind == "eval":
            eval_stage_info(stage, tracked.get(spec.key), now)
        stamp_running_duration(stage, now)
        stages.append(stage)
    return stages
