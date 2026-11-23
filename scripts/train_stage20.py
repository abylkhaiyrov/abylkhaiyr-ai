from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

from prepare_stage20 import MIXTURE
from prepare_stage20 import ROOT as REPLAY_ROOT
from train_stage18 import ROOT as STAGE18_ROOT
from train_stage18 import TARGET, configuration, resource_gate, verify_data

from llm.core.checksums import file_sha256
from llm.core.config import ExperimentConfig
from llm.core.errors import ResourceError
from llm.training.checkpoint import CheckpointManager
from llm.training.series import run_series
from llm.training.session import exclusive_session, run_training

ROOT = Path("experiments/stage20")
CHECKPOINTS = Path("checkpoints/stage20")
REGISTRATION = Path("docs/stage20_preregistration.md")
REUSED = {"A-1": Path("checkpoints/stage19/L1")}
WALL_SECONDS = 3900
RAM_MARGIN_GIB = 0.25
LEARNING_RATE = 5e-5
JOBS = ("B-1", "R-1", "A-2", "B-2", "R-2", "A-3", "B-3", "R-3")
DATA = {"A": "data/stage18/a", "B": "data/stage18/b", "R": str(REPLAY_ROOT)}


def checkpoint_dir(job: str) -> Path:
    return REUSED.get(job, CHECKPOINTS / job)


def job_configuration(job: str) -> ExperimentConfig:
    arm, seed = job.split("-")
    base = configuration(f"a-seed-{seed}")
    optimizer = base.train.optimizer.model_copy(update={"learning_rate": LEARNING_RATE})
    train = base.train.model_copy(
        update={
            "data_dir": DATA[arm],
            "dataset_version": f"stage20-{arm.lower()}",
            "mixture": dict(MIXTURE) if arm == "R" else {},
            "optimizer": optimizer,
            "checkpoint_dir": str(checkpoint_dir(job)),
            "output_dir": str(ROOT / job),
        }
    )
    return base.model_copy(update={"name": f"stage20-{job}", "train": train})


def step_of(job: str) -> int:
    manager = CheckpointManager(checkpoint_dir(job), keep=2)
    state = manager.latest_valid()
    return 0 if state is None else int(CheckpointManager.verify(state)["step"])


def frozen_inputs() -> dict[str, str]:
    files = {"registration": REGISTRATION, "replay": REPLAY_ROOT / "manifest.json"}
    return {label: file_sha256(path) for label, path in files.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    commands = parser.add_mutually_exclusive_group(required=True)
    commands.add_argument("--run", action="store_true")
    commands.add_argument("--job", choices=JOBS)
    parser.add_argument("--state", type=Path, default=ROOT / "series.json")
    args = parser.parse_args()
    plan = {
        "frozen": {**verify_data(), **frozen_inputs()},
        "jobs": {job: job_configuration(job).model_dump(mode="json") for job in JOBS},
    }
    if args.run:
        ROOT.mkdir(parents=True, exist_ok=True)
        profile = json.loads((STAGE18_ROOT / "profile.json").read_text(encoding="utf-8"))[0]
        resources = resource_gate(float(profile["peak_device_gb"]), RAM_MARGIN_GIB)
        if args.state.exists():
            state = json.loads(args.state.read_text(encoding="utf-8"))
            if state["plan"] != plan:
                parser.error("resume requires the registered configurations and inputs")
        else:
            state = {"plan": plan, "deadline": time.time() + WALL_SECONDS, "resources": resources}
            args.state.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(state["resources"]), flush=True)
        with exclusive_session(ROOT / ".series.lock"):
            run_series(
                JOBS,
                float(state["deadline"]),
                lambda job: [
                    sys.executable,
                    str(Path(__file__)),
                    "--job",
                    job,
                    "--state",
                    str(args.state),
                ],
                lambda job: step_of(job) >= TARGET,
            )
        return
    state = json.loads(args.state.read_text(encoding="utf-8"))
    if state["plan"] != plan:
        raise ResourceError("registered configuration or input changed")
    step = step_of(args.job)
    if step == TARGET:
        return
    deadline = time.monotonic() + float(state["deadline"]) - time.time()
    result = run_training(job_configuration(args.job), step > 0, TARGET, deadline=deadline)
    print(json.dumps({"job": args.job, **asdict(result)}), flush=True)


if __name__ == "__main__":
    main()
