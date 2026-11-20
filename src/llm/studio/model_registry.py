from __future__ import annotations

from pathlib import Path
from typing import Any

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
    final = relative_id(root, artifacts.get("final_checkpoint"))
    best_pretrain = relative_id(root, artifacts.get("best_pretrain_checkpoint"))
    exported = stage_status(pipeline, "export_release") == "completed"
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
    return {
        "production": final if exported else None,
        "best_evaluated": final or best_pretrain,
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
    for model in models:
        model["roles"] = [role for role in ROLE_ORDER if assignments[role] == model["id"]]
        model["verified_stage"] = verified.get(model["id"])
        model["verified"] = model["id"] in verified
        model["display_name"] = f"{model['name']} · {model.get('step') or '—'}"
    return assignments
