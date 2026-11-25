from __future__ import annotations

from llm.core.errors import ResourceError
from llm.training.checkpoint import CheckpointManager
from llm.training.spec import ExperimentSpec


def completed_step(spec: ExperimentSpec, job: str) -> int:
    manager = CheckpointManager(spec.checkpoint_dir(job), keep=2)
    state = manager.latest_valid()
    if state is None:
        if manager.list_steps():
            raise ResourceError(f"no restorable checkpoints for {job}")
        return 0
    return int(CheckpointManager.verify(state)["step"])


def is_complete(spec: ExperimentSpec, job: str, until: int | None = None) -> bool:
    return completed_step(spec, job) >= spec.goal(until)
