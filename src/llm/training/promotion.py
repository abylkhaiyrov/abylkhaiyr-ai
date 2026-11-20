from __future__ import annotations

import json
import os
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

MANIFEST_PATH = Path("checkpoints") / "production.json"
STAGE_ORDER = ("base", "pretrain", "sft", "dpo")
INSTRUCTION_STAGES = ("sft", "dpo")
READY_PASS_RATE = 0.5
MAX_REPETITION = 0.45
KNOWN_LIMITATIONS = (
    "small model (about 16M parameters): may give wrong or made-up facts",
    "context limited to 512 tokens; no internet access; no memory between chats",
    "weakest in Kazakh and in multi-step reasoning; verify important answers",
)


@dataclass(frozen=True)
class Candidate:
    checkpoint: str
    stage: str
    metrics: Mapping[str, float]

    @property
    def pass_rate(self) -> float:
        return float(self.metrics.get("conv_pass_rate", 0.0))

    @property
    def repetition(self) -> float:
        return float(self.metrics.get("conv_repetition_ratio", 1.0))


def manifest_file(root: Path) -> Path:
    return root / MANIFEST_PATH


def read_manifest(root: Path) -> dict[str, Any]:
    path = manifest_file(root)
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def production_checkpoint(root: Path) -> str | None:
    production = read_manifest(root).get("production") or {}
    checkpoint = production.get("checkpoint")
    return checkpoint if isinstance(checkpoint, str) and (root / checkpoint).exists() else None


def eligible(candidate: Candidate, reference_pass_rate: float) -> bool:
    return (
        candidate.stage in INSTRUCTION_STAGES
        and candidate.pass_rate > reference_pass_rate
        and candidate.repetition <= MAX_REPETITION
    )


def select_production(
    candidates: list[Candidate], reference_pass_rate: float = 0.0
) -> Candidate | None:
    pool = [item for item in candidates if eligible(item, reference_pass_rate)]
    if not pool:
        return None
    stage_rank = {stage: index for index, stage in enumerate(STAGE_ORDER)}
    return max(pool, key=lambda item: (item.pass_rate, -item.repetition, stage_rank[item.stage]))


def promote(
    root: Path, candidates: list[Candidate], reference_pass_rate: float = 0.0
) -> dict[str, Any]:
    chosen = select_production(candidates, reference_pass_rate)
    manifest: dict[str, Any] = {
        "updated_at": datetime.now(UTC).isoformat(),
        "reference_pass_rate": reference_pass_rate,
        "models": {
            item.checkpoint: {"stage": item.stage, "metrics": dict(item.metrics)}
            for item in candidates
        },
        "production": None
        if chosen is None
        else {
            "checkpoint": chosen.checkpoint,
            "stage": chosen.stage,
            "conv_pass_rate": chosen.pass_rate,
            "ready": chosen.pass_rate >= READY_PASS_RATE,
            "limitations": list(KNOWN_LIMITATIONS),
        },
    }
    path = manifest_file(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary, path)
    return manifest
