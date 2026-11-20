from __future__ import annotations

from pathlib import Path
from typing import Any

from llm.training.promotion import production_checkpoint, read_manifest

ROLE_ORDER = ("production", "best_evaluated", "latest_trained", "loaded")


def relative_id(root: Path, value: str | None) -> str | None:
    if not value:
        return None
    path = Path(value)
    try:
        return str(path.resolve().relative_to(root)) if path.is_absolute() else value
    except ValueError:
        return value


def stage_status(pipeline: dict[str, Any] | None, key: str) -> str:
    if pipeline is None:
        return "pending"
    return next((item["status"] for item in pipeline["stages"] if item["key"] == key), "pending")


def verified_checkpoints(root: Path, pipeline: dict[str, Any] | None) -> dict[str, str]:
    if pipeline is None:
        return {}
    artifacts = pipeline["artifacts"]
    verified: dict[str, str] = {}
    candidates = (
        ("best_pretrain_checkpoint", "select_best_pretrain", "pretrain"),
        ("sft_checkpoint", "eval_sft", "sft"),
        ("dpo_checkpoint", "eval_final", "dpo"),
    )
    for artifact, gate, stage in candidates:
        identifier = relative_id(root, artifacts.get(artifact))
        if identifier and stage_status(pipeline, gate) == "completed":
            verified[identifier] = stage
    return verified


def role_assignments(
    root: Path,
    pipeline: dict[str, Any] | None,
    models: list[dict[str, Any]],
    loaded_id: str | None,
) -> dict[str, str | None]:
    artifacts = pipeline["artifacts"] if pipeline else {}
    best_pretrain = relative_id(root, artifacts.get("best_pretrain_checkpoint"))
    directories = (
        {item["checkpoint_dir"] for item in pipeline["stages"] if item["checkpoint_dir"]}
        if pipeline
        else set()
    )
    trained = [
        item
        for item in models
        if not directories
        or any(item["id"].startswith(f"{directory}/") for directory in directories)
    ]
    latest = max(trained, key=lambda item: item["created_at"], default=None)
    production = production_info(root)
    approved = production_checkpoint(root) if production and production["ready"] else None
    return {
        "production": relative_id(root, approved),
        "best_evaluated": best_pretrain,
        "latest_trained": None if latest is None else latest["id"],
        "loaded": loaded_id,
    }


def annotate_models(
    root: Path,
    pipeline: dict[str, Any] | None,
    models: list[dict[str, Any]],
    loaded_id: str | None,
) -> dict[str, str | None]:
    assignments = role_assignments(root, pipeline, models, loaded_id)
    verified = verified_checkpoints(root, pipeline)
    stages = {item["key"]: item for item in pipeline["stages"]} if pipeline else {}
    artifacts = pipeline["artifacts"] if pipeline else {}
    sft = relative_id(root, artifacts.get("sft_checkpoint"))
    gate = stages.get("dpo", {})
    failed_sft = gate.get("note_code") == "sft_perplexity_regression"
    for model in models:
        model["roles"] = [role for role in ROLE_ORDER if assignments[role] == model["id"]]
        model["verified_stage"] = verified.get(model["id"])
        model["evaluated"] = model["id"] in verified
        model["quality_failed"] = failed_sft and model["id"] == sft
        model["verified"] = model["evaluated"] and not model["quality_failed"]
        model["trained"] = model["step"] is not None and model["step"] > 0
        model["display_name"] = f"{model['name']} · {model.get('step') or '—'}"
    return assignments


def production_info(root: Path) -> dict[str, Any] | None:
    production = read_manifest(root).get("production")
    if not isinstance(production, dict) or not production.get("checkpoint"):
        return None
    return {
        "checkpoint": production["checkpoint"],
        "stage": production.get("stage"),
        "ready": bool(production.get("ready")),
        "pass_rate": production.get("conv_pass_rate"),
        "limitations": [str(item) for item in production.get("limitations", [])],
    }
