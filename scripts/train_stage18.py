from __future__ import annotations

import argparse
import json
import signal
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path
from threading import Event

import psutil
import torch

from llm.core.checksums import file_sha256
from llm.core.config import ExperimentConfig, load_experiment_config
from llm.core.errors import ResourceError
from llm.core.hardware import free_disk_gb
from llm.training.capacity_study import BATCH, BUDGETS, SEEDS, TOKENS_PER_STEP, budget_step
from llm.training.checkpoint import CheckpointManager
from llm.training.session import exclusive_session, graceful_shutdown, run_training

ROOT = Path("experiments/stage18")
DATA = Path("data/stage18")
CHECKPOINTS = Path("checkpoints/stage18")
INITIAL = Path("checkpoints/tiny_v2/step-00006000")
TOKENIZER = Path("data/tokenizer/tokenizer.json")
REGISTRATION = Path("docs/stage18_preregistration.md")
LOCK = Path("configs/stage18_pilot_lock.json")
JOBS = tuple(f"{arm}-seed-{seed}" for seed in SEEDS for arm in ("a", "b"))
TARGET = budget_step(BUDGETS[0])
WALL_SECONDS = 3300
DISK_RESERVE = 20.0
GIB = 2**30


def configuration(job: str) -> ExperimentConfig:
    source = load_experiment_config("configs/tiny_v2.yaml")
    arm, _, seed = job.split("-")
    train = {
        "init_from": str(INITIAL),
        "restore_optimizer": False,
        "seed": int(seed),
        "data_dir": str(DATA / arm),
        "dataset_version": f"stage18-{arm}",
        "checkpoint_dir": str(CHECKPOINTS / job),
        "output_dir": str(ROOT / job),
        "micro_batch_size": BATCH,
        "gradient_accumulation_steps": TOKENS_PER_STEP // (BATCH * 512),
        "dynamic_batch_size": True,
        "min_micro_batch_size": 2,
        "max_steps": TARGET,
        "num_threads": 4,
        "eval_interval": 61,
        "eval_batches": 6,
        "log_interval": 10,
        "checkpoint_interval": 61,
        "protected_steps": [TARGET],
        "keep_checkpoints": 2,
        "checkpoint_alias": None,
        "fast_causal": True,
        "mixture": {},
        "max_ram_fraction": 0.90,
        "min_free_disk_gb": DISK_RESERVE,
        "optimizer": source.train.optimizer.model_copy(
            update={"learning_rate": 2e-4, "warmup_steps": 10, "min_lr_ratio": 1.0, "foreach": True}
        ),
    }
    return source.model_copy(
        update={"name": f"stage18-{job}", "train": source.train.model_copy(update=train)}
    )


def frozen_inputs() -> dict[str, str]:
    files = {
        "corpus": DATA / "manifest.json",
        "diagnosis": ROOT / "diagnostic.jsonl",
        "generation": ROOT / "free_cases.jsonl",
        "overlap_audit": ROOT / "verified_overlap.json",
        "historical_diagnostic": Path("experiments/stage17/diagnostic_v2.jsonl"),
        "historical_generation": Path("configs/natural_eval_v4.jsonl"),
        "unigram": Path("data/v4_expanded/tokens/train.bin"),
        "initial_weights": INITIAL / "model.safetensors",
        "initial_metadata": INITIAL / "meta.json",
        "tokenizer": TOKENIZER,
        "registration": REGISTRATION,
    }
    return {label: file_sha256(path) for label, path in files.items()}


def verify_data() -> dict[str, str]:
    inputs = frozen_inputs()
    locked: dict[str, str] = json.loads(LOCK.read_text(encoding="utf-8"))
    if locked != inputs:
        raise ResourceError("frozen pilot input changed")
    audit = json.loads((ROOT / "verified_overlap.json").read_text(encoding="utf-8"))
    if audit["a"] or audit["b"] or any(audit["short_original_anchors"].values()):
        raise ResourceError("evaluation overlap detected")
    initial = CheckpointManager.verify(INITIAL)
    corpus = json.loads((DATA / "manifest.json").read_text(encoding="utf-8"))
    if corpus["tokenizer"] != initial["tokenizer_version"]:
        raise ResourceError("initial checkpoint and corpus tokenizers differ")
    for arm in ("a", "b"):
        details = corpus[arm]
        if details["tokens"]["train"]["stored_tokens"]["all"] <= TARGET * TOKENS_PER_STEP:
            raise ResourceError(f"arm {arm} would require repeated exposures")
        for name, digest in details["sha256"].items():
            if file_sha256(DATA / arm / name) != digest:
                raise ResourceError(f"arm {arm} contains changed data: {name}")
    for name, digest in corpus["a"]["sha256"].items():
        if (
            not name.startswith(("train", "tokens/train"))
            and corpus["b"]["sha256"].get(name) != digest
        ):
            raise ResourceError(f"arm holdouts differ: {name}")
    return inputs


def completed_step(job: str) -> int:
    manager = CheckpointManager(CHECKPOINTS / job, keep=2)
    state = manager.latest_valid()
    if state is None:
        if manager.list_steps():
            raise ResourceError(f"no restorable checkpoints for {job}")
        return 0
    return int(CheckpointManager.verify(state)["step"])


def resource_gate(expected_peak: float) -> dict[str, float]:
    ram = psutil.virtual_memory()
    swap = psutil.swap_memory()
    disk = free_disk_gb(CHECKPOINTS)
    if not torch.backends.mps.is_available() or ram.available / GIB < expected_peak + 0.75:
        raise ResourceError("MPS or RAM preflight failed")
    needed_disk = DISK_RESERVE + len(JOBS) * 3 * 0.20 + 0.50
    if disk < needed_disk:
        raise ResourceError(f"disk preflight needs {needed_disk:.2f} GiB")
    return {
        "unified_ram_gib": ram.total / GIB,
        "available_ram_gib": ram.available / GIB,
        "swap_used_gib": swap.used / GIB,
        "swap_total_gib": swap.total / GIB,
        "disk_free_gib": disk,
        "disk_required_gib": needed_disk,
        "calibrated_mps_peak_gib": expected_peak,
    }


def run_queue(steps: dict[str, int]) -> None:
    stop = Event()
    with graceful_shutdown(stop):
        for job, step in steps.items():
            if step >= TARGET:
                continue
            if stop.is_set():
                return
            command = [sys.executable, str(Path(__file__)), "--job", job]
            with subprocess.Popen(command) as child:
                while child.poll() is None:
                    if stop.wait(0.5):
                        child.send_signal(signal.SIGTERM)
                        child.wait()
                        return
                if child.returncode:
                    raise subprocess.CalledProcessError(child.returncode, command)
            if completed_step(job) != TARGET:
                raise ResourceError(f"job {job} stopped before its frozen comparison point")


def main() -> None:
    parser = argparse.ArgumentParser()
    commands = parser.add_mutually_exclusive_group(required=True)
    commands.add_argument("--profile-config", action="store_true")
    commands.add_argument("--run", action="store_true")
    commands.add_argument("--job", choices=JOBS)
    args = parser.parse_args()
    if args.profile_config:
        print(configuration(JOBS[0]).model_dump_json(indent=2))
        return
    frozen = verify_data()
    state = ROOT / "pilot.json"
    configs = {job: configuration(job).model_dump(mode="json") for job in JOBS}
    plan = {"frozen": frozen, "configs": configs, "seconds": WALL_SECONDS}
    previous = json.loads(state.read_text(encoding="utf-8")) if state.exists() else None
    if previous is not None and previous["plan"] != plan:
        parser.error("resume requires the original frozen inputs and configurations")
    if args.run:
        profile = json.loads((ROOT / "profile.json").read_text(encoding="utf-8"))[0]
        throughput = float(profile["useful_tokens_per_second"])
        peak = float(profile["peak_device_gb"])
        if profile["variant"] != "fast_causal" or throughput <= 0 or peak <= 0:
            parser.error("valid FP32 MPS throughput calibration required")
        steps = {job: completed_step(job) for job in JOBS}
        remaining_steps = sum(max(0, TARGET - step) for step in steps.values())
        if not remaining_steps:
            return
        resources = resource_gate(peak)
        conservative_seconds = remaining_steps * TOKENS_PER_STEP / (0.65 * throughput) + 180
        now = time.time()
        deadline = now + WALL_SECONDS if previous is None else previous["deadline"]
        if conservative_seconds > deadline - now:
            parser.error("time budget does not fit all remaining comparable runs")
        if previous is None:
            previous = {
                "plan": plan,
                "deadline": deadline,
                "resource_gate": resources,
                "seconds_estimated": conservative_seconds,
                "tokens_per_second": throughput,
            }
            with state.open("x", encoding="utf-8") as handle:
                handle.write(json.dumps(previous, indent=2) + "\n")
        print(json.dumps(previous["resource_gate"]), flush=True)
        with exclusive_session(ROOT / ".pilot.lock"):
            run_queue({job: completed_step(job) for job in JOBS})
        return
    if previous is None:
        parser.error("start through --run to verify the entire series")
    resource_gate(previous["resource_gate"]["calibrated_mps_peak_gib"])
    step = completed_step(args.job)
    if step == TARGET:
        return
    deadline = time.monotonic() + previous["deadline"] - time.time()
    result = run_training(configuration(args.job), step > 0, TARGET, deadline=deadline)
    print(json.dumps({"job": args.job, **asdict(result)}), flush=True)


if __name__ == "__main__":
    main()
