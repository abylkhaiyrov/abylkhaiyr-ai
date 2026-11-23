from __future__ import annotations

import json
from collections.abc import Callable, Iterable
from typing import Any

from sqlalchemy.orm import Session, sessionmaker

from llm.studio.access_policy import AccessPolicy
from llm.studio.run_catalog import version_of

ALLOWED_KEY = "models.allowed"
STATUS_RANK = {"production": 4, "validated": 3, "candidate": 2, "experimental": 1, "rejected": 0}
CHAT_STAGES = {"sft", "dpo"}
USER_FIELDS = (
    "id",
    "display_name",
    "family",
    "version",
    "parameters",
    "context_length",
    "quality_status",
    "capabilities",
    "available",
    "recommended_use",
)


class ModelPolicyError(ValueError):
    pass


def checkpoint_status(model: dict[str, Any]) -> str:
    if "production" in model["roles"]:
        return "production"
    if model["quality_failed"]:
        return "rejected"
    if model["verified"]:
        return "validated"
    if model["evaluated"] or {"best_evaluated", "latest_trained"} & set(model["roles"]):
        return "candidate"
    return "experimental"


def family_of(name: str) -> str:
    return name.split("_")[0]


def display_name(name: str) -> str:
    words = name.replace("_", " ").strip()
    return words[:1].upper() + words[1:]


def representative(checkpoints: list[dict[str, Any]]) -> dict[str, Any]:
    return max(checkpoints, key=lambda item: (STATUS_RANK[item["status"]], item["step"] or 0))


def capabilities(stage: str | None) -> list[str]:
    return ["chat"] if stage in CHAT_STAGES else ["text_completion"]


def recommended_use(entry_status: str, stage: str | None) -> str:
    if stage in CHAT_STAGES and entry_status in {"production", "validated"}:
        return "chat"
    return "research"


def build_catalog(
    models: Iterable[dict[str, Any]], allowed: set[str] | None
) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for model in models:
        groups.setdefault(model["model_id"], []).append(model)
    entries = []
    for identifier, checkpoints in groups.items():
        best = representative(checkpoints)
        status = best["status"]
        entries.append(
            {
                "id": identifier,
                "display_name": display_name(identifier),
                "family": family_of(identifier),
                "version": version_of(identifier),
                "parameters": best["parameters"],
                "context_length": best["context_length"],
                "quality_status": status,
                "stage": best["stage"],
                "capabilities": capabilities(best["stage"]),
                "available": bool(best["compatible"] and best["available"]),
                "availability_reason": best["availability_reason"],
                "recommended_use": recommended_use(status, best["stage"]),
                "representative_checkpoint": best["id"],
                "checkpoint_count": len(checkpoints),
                "allowed": allowed_by_policy(identifier, status, allowed),
            }
        )
    return sorted(entries, key=lambda item: (-STATUS_RANK[item["quality_status"]], item["id"]))


def allowed_by_policy(identifier: str, status: str, allowed: set[str] | None) -> bool:
    if allowed is None:
        return status == "production"
    return identifier in allowed and status != "rejected"


def user_view(entries: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {key: entry[key] for key in USER_FIELDS}
        for entry in entries
        if entry["allowed"] and entry["available"]
    ]


class ModelAccessPolicy(AccessPolicy):
    def __init__(self, session_factory: sessionmaker[Session]) -> None:
        super().__init__(session_factory, ALLOWED_KEY)

    def update(
        self, identifiers: list[str], catalog: Callable[[], list[dict[str, Any]]]
    ) -> list[str]:
        known = {item["id"]: item for item in catalog()}
        unknown = sorted(set(identifiers) - known.keys())
        rejected = sorted(
            item
            for item in identifiers
            if item in known and known[item]["quality_status"] == "rejected"
        )
        if unknown or rejected:
            raise ModelPolicyError(json.dumps({"unknown": unknown, "rejected": rejected}))
        return self.store(set(identifiers))
