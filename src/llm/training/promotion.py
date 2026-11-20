from __future__ import annotations

import json
import math
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
MAX_CORPUS_PERPLEXITY_RATIO = 1.5
KNOWN_LIMITATIONS = (
    "small model (about 16M parameters): may give wrong or made-up facts",
    "context limited to 512 tokens; no internet access; no memory between chats",
    "multilingual and multi-step reasoning quality varies by task; verify important answers",
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
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def production_checkpoint(root: Path) -> str | None:
    production = read_manifest(root).get("production") or {}
    if not isinstance(production, dict) or production.get("ready") is not True:
        return None
    checkpoint = production.get("checkpoint")
    if isinstance(checkpoint, str) and (root / checkpoint / "model.safetensors").is_file():
        return checkpoint
    return None


def eligible(candidate: Candidate, reference_pass_rate: float) -> bool:
    return (
        candidate.stage in INSTRUCTION_STAGES
        and 0 <= reference_pass_rate < candidate.pass_rate <= 1
        and 0 <= candidate.repetition <= MAX_REPETITION
    )


def select_experimental(
    candidates: list[Candidate], reference_pass_rate: float = 0.0
) -> Candidate | None:
    pool = [item for item in candidates if eligible(item, reference_pass_rate)]
    if not pool:
        return None
    stage_rank = {stage: index for index, stage in enumerate(STAGE_ORDER)}
    return max(pool, key=lambda item: (item.pass_rate, -item.repetition, stage_rank[item.stage]))


def quality_failures(candidate: Candidate, reference: Mapping[str, float]) -> list[str]:
    failures: list[str] = []
    baseline_rate = reference.get("conv_pass_rate", float("nan"))
    if not math.isfinite(baseline_rate) or not eligible(candidate, baseline_rate):
        failures.append("conversational_improvement_or_repetition_gate_failed")
    if candidate.pass_rate < READY_PASS_RATE:
        failures.append("conversational_pass_rate_below_0.5")
    for key in ("test_perplexity", "perplexity_en", "perplexity_ru", "perplexity_kk"):
        baseline = reference.get(key, float("nan"))
        value = candidate.metrics.get(key, float("nan"))
        if not math.isfinite(baseline) or not math.isfinite(value) or min(baseline, value) <= 0:
            failures.append(f"missing_comparable_{key}")
        elif value > baseline * MAX_CORPUS_PERPLEXITY_RATIO:
            failures.append(f"regression_{key}")
    return failures


def select_production(
    candidates: list[Candidate], reference: Mapping[str, float]
) -> Candidate | None:
    eligible_candidates = [item for item in candidates if not quality_failures(item, reference)]
    return select_experimental(eligible_candidates, reference.get("conv_pass_rate", 0.0))


def promote(
    root: Path,
    candidates: list[Candidate],
    reference_pass_rate: float = 0.0,
    *,
    reference_metrics: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    reference = dict(reference_metrics or {})
    baseline_rate = reference.get("conv_pass_rate", reference_pass_rate)
    checks = {item.checkpoint: quality_failures(item, reference) for item in candidates}
    available = [
        item for item in candidates if (root / item.checkpoint / "model.safetensors").is_file()
    ]
    chosen = select_production(available, reference)
    experimental = select_experimental(
        [
            item
            for item in available
            if not any(failure.startswith("regression_") for failure in checks[item.checkpoint])
        ],
        baseline_rate,
    )
    manifest: dict[str, Any] = {
        "updated_at": datetime.now(UTC).isoformat(),
        "reference_pass_rate": baseline_rate,
        "reference_metrics": reference,
        "models": {
            item.checkpoint: {
                "stage": item.stage,
                "metrics": dict(item.metrics),
                "quality_failures": checks[item.checkpoint],
            }
            for item in candidates
        },
        "experimental": (
            None
            if experimental is None or chosen
            else {
                "checkpoint": experimental.checkpoint,
                "conv_pass_rate": experimental.pass_rate,
                "limitations": list(KNOWN_LIMITATIONS),
            }
        ),
        "production": (
            None
            if chosen is None
            else {
                "checkpoint": chosen.checkpoint,
                "stage": chosen.stage,
                "conv_pass_rate": chosen.pass_rate,
                "ready": True,
                "limitations": list(KNOWN_LIMITATIONS),
            }
        ),
    }
    path = manifest_file(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8"
    )
    os.replace(temporary, path)
    return manifest
