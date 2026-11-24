from __future__ import annotations

import json
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from llm.core.errors import ConfigError
from llm.training.assessment import assess
from llm.training.budget import BudgetClock, TimeLedger
from llm.training.checkpoint import CheckpointManager
from llm.training.preflight import frozen_hashes, verify_corpus, verify_lock
from llm.training.progress import completed_step, is_complete
from llm.training.resources import resource_gate, resources_available
from llm.training.series import run_series
from llm.training.session import exclusive_session, run_training
from llm.training.spec import ExperimentSpec, load_spec, spec_digest

CALIBRATION_KEY = "peak_device_gb"


def verified_inputs(spec: ExperimentSpec) -> dict[str, str]:
    hashes = frozen_hashes(spec.frozen_inputs)
    if spec.lock_file is not None:
        verify_lock(hashes, spec.lock_file, spec.lock_labels)
    if spec.corpus is not None:
        verify_corpus(
            spec.corpus, spec.initial_checkpoint, spec.target_steps * spec.tokens_per_step
        )
    return {**hashes, "specification": spec_digest(spec)}


def build_plan(spec: ExperimentSpec) -> dict[str, Any]:
    return {
        "frozen": verified_inputs(spec),
        "limits": spec.limits.model_dump(),
        "resources": spec.resources.model_dump(),
        "jobs": {
            job: spec.job_configuration(job).model_dump(mode="json") for job in spec.job_names()
        },
    }


def calibrated_peak(spec: ExperimentSpec) -> float:
    if spec.calibration_file is None:
        return 0.0
    profile = json.loads(Path(spec.calibration_file).read_text(encoding="utf-8"))[0]
    return float(profile[CALIBRATION_KEY])


def load_state(spec: ExperimentSpec, plan: dict[str, Any]) -> dict[str, Any] | None:
    if not spec.state_path.exists():
        return None
    state: dict[str, Any] = json.loads(spec.state_path.read_text(encoding="utf-8"))
    if state["plan"] != plan:
        raise ConfigError("resume requires the registered configurations, inputs and budget")
    return state


def healthy(spec: ExperimentSpec, peak: float, job: str) -> bool:
    if spec.resources.require_mps and not torch.backends.mps.is_available():
        return False
    manager = CheckpointManager(spec.checkpoint_dir(job), keep=2)
    if manager.list_steps() and manager.latest_valid() is None:
        return False
    return resources_available(peak, spec.resources, spec.checkpoint_dir(job).parent)


def run_campaign(spec: ExperimentSpec, spec_path: Path) -> str | None:
    plan = build_plan(spec)
    jobs = spec.runnable_jobs()
    if all(is_complete(spec, job) for job in jobs):
        return None
    spec.state_path.parent.mkdir(parents=True, exist_ok=True)
    peak = calibrated_peak(spec)
    gate = resource_gate(peak, len(jobs), spec.resources, Path(spec.checkpoint_root))
    state = load_state(spec, plan)
    if state is None:
        state = {"plan": plan, "resource_gate": gate, "created_at": time.time()}
        spec.state_path.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(state["resource_gate"]), flush=True)
    ledger = TimeLedger.load(spec.ledger_path) or TimeLedger()
    clock = BudgetClock(spec.limits, ledger)
    clock.check()
    with exclusive_session(Path(spec.output_root) / ".series.lock"):
        reason = run_series(
            jobs,
            clock,
            lambda job, remaining: [
                sys.executable,
                "-m",
                "llm.cli.main",
                "experiment",
                "job",
                str(spec_path),
                job,
                "--active-seconds",
                str(remaining),
            ],
            lambda job: is_complete(spec, job),
            lambda job: healthy(spec, peak, job),
            spec.ledger_path,
        )
    clock.ledger.save(spec.ledger_path)
    return reason


def evaluate_campaign(spec: ExperimentSpec) -> dict[str, Any]:
    return assess(spec)


def run_job(spec: ExperimentSpec, job: str, active_seconds: float | None) -> dict[str, Any]:
    plan = build_plan(spec)
    if load_state(spec, plan) is None:
        raise ConfigError("start through the series runner to verify the entire series")
    step = completed_step(spec, job)
    if step >= spec.target_steps:
        return {"job": job, "final_step": step}
    deadline = None if active_seconds is None else time.monotonic() + active_seconds
    result = run_training(
        spec.job_configuration(job), step > 0, spec.target_steps, deadline=deadline
    )
    return {"job": job, **asdict(result)}


def status(spec: ExperimentSpec) -> dict[str, Any]:
    ledger = TimeLedger.load(spec.ledger_path)
    return {
        "name": spec.name,
        "target_steps": spec.target_steps,
        "jobs": {job: completed_step(spec, job) for job in spec.job_names()},
        "reused": sorted(spec.reuse),
        "budget": asdict(ledger) if ledger else None,
        "limits": spec.limits.model_dump(),
        "evaluation_suites": [suite.name for suite in spec.evaluation.suites]
        if spec.evaluation
        else [],
    }


def open_spec(path: str) -> tuple[ExperimentSpec, Path]:
    return load_spec(path), Path(path)
