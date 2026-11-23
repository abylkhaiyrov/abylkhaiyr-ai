from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

from train_stage18 import (
    ROOT as STAGE18_ROOT,
)
from train_stage18 import (
    TARGET,
    configuration,
    resource_gate,
    verify_data,
)

from llm.core.config import ExperimentConfig
from llm.training.checkpoint import CheckpointManager
from llm.training.series import run_series
from llm.training.session import exclusive_session, run_training

ROOT = Path("experiments/stage19")
CHECKPOINTS = Path("checkpoints/stage19")
WALL_SECONDS = 1800
ARMS: dict[str, tuple[float, float, bool]] = {
    "L1": (5e-5, 1.0, False),
    "L2": (5e-5, 1.0, True),
    "L3": (1e-4, 1.0, False),
    "L4": (2e-4, 0.25, False),
}
BASE_JOB = "a-seed-1"


def arm_configuration(arm: str) -> ExperimentConfig:
    rate, floor, restore = ARMS[arm]
    base = configuration(BASE_JOB)
    optimizer = base.train.optimizer.model_copy(
        update={"learning_rate": rate, "min_lr_ratio": floor}
    )
    train = base.train.model_copy(
        update={
            "restore_optimizer": restore,
            "optimizer": optimizer,
            "checkpoint_dir": str(CHECKPOINTS / arm),
            "output_dir": str(ROOT / arm),
        }
    )
    return base.model_copy(update={"name": f"stage19-{arm}", "train": train})


def run_queue(deadline: float, arms: tuple[str, ...], state_file: Path) -> None:
    run_series(
        arms,
        deadline,
        lambda arm: [sys.executable, str(Path(__file__)), "--job", arm, "--state", str(state_file)],
        lambda arm: step_of(arm) >= TARGET,
    )


def step_of(arm: str) -> int:
    manager = CheckpointManager(CHECKPOINTS / arm, keep=2)
    state = manager.latest_valid()
    return 0 if state is None else int(CheckpointManager.verify(state)["step"])


def main() -> None:
    parser = argparse.ArgumentParser()
    commands = parser.add_mutually_exclusive_group(required=True)
    commands.add_argument("--run", action="store_true")
    commands.add_argument("--job", choices=tuple(ARMS))
    parser.add_argument("--arms", nargs="+", choices=tuple(ARMS), default=tuple(ARMS))
    parser.add_argument("--state", type=Path, default=ROOT / "lr_series.json")
    parser.add_argument("--wall", type=int, default=WALL_SECONDS)
    args = parser.parse_args()
    state_file = args.state
    frozen = verify_data()
    plan = {
        "frozen": frozen,
        "arms": {arm: arm_configuration(arm).model_dump(mode="json") for arm in ARMS},
    }
    if args.run:
        ROOT.mkdir(parents=True, exist_ok=True)
        profile = json.loads((STAGE18_ROOT / "profile.json").read_text(encoding="utf-8"))[0]
        resources = resource_gate(float(profile["peak_device_gb"]))
        if state_file.exists():
            state = json.loads(state_file.read_text(encoding="utf-8"))
            if state["plan"] != plan:
                parser.error("resume requires the registered configurations")
        else:
            state = {"plan": plan, "deadline": time.time() + args.wall, "resources": resources}
            state_file.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(state["resources"]), flush=True)
        with exclusive_session(ROOT / ".series.lock"):
            run_queue(float(state["deadline"]), tuple(args.arms), state_file)
        return
    state = json.loads(state_file.read_text(encoding="utf-8"))
    step = step_of(args.job)
    if step == TARGET:
        return
    deadline = time.monotonic() + float(state["deadline"]) - time.time()
    result = run_training(arm_configuration(args.job), step > 0, TARGET, deadline=deadline)
    print(json.dumps({"job": args.job, **asdict(result)}), flush=True)


if __name__ == "__main__":
    main()
