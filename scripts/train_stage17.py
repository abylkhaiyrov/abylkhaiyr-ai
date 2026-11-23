from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import psutil
import torch

from llm.core.checksums import file_sha256
from llm.core.errors import ResourceError
from llm.core.hardware import free_disk_gb
from llm.optimization.memory import release_device_memory
from llm.tokenizer.bpe import BPETokenizer
from llm.training.capacity_study import (
    ARCHITECTURES,
    BATCH,
    BUDGETS,
    CONTEXT,
    INITIAL_WALL_SECONDS,
    MIN_DISK_GIB,
    SEEDS,
    budget_step,
    capacity_config,
)
from llm.training.checkpoint import CheckpointManager
from llm.training.session import run_training

BYTES_PER_GIB = 2**30


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("experiments/stage17"))
    parser.add_argument("--checkpoints", type=Path, default=Path("checkpoints/stage17"))
    parser.add_argument("--corpus", type=Path, default=Path("data/v4_expanded"))
    parser.add_argument("--tokenizer", type=Path, default=Path("data/tokenizer"))
    parser.add_argument("--profile", type=Path)
    parser.add_argument(
        "--job",
        required=True,
        choices=[f"{model}-seed-{seed}" for seed in SEEDS for model in ARCHITECTURES],
    )
    args = parser.parse_args()
    output = args.output
    profile = args.profile if args.profile else output / "profile_mb8.json"
    calibration = json.loads(profile.read_text(encoding="utf-8"))
    measured = {row["profile"]: row for row in calibration["profiles"]}
    if calibration["device"] != "mps" or calibration["precision"] != "fp32":
        parser.error("expected a successful FP32 MPS calibration")
    for name in ARCHITECTURES:
        row = measured.get(name)
        if (
            not row
            or not row["success"]
            or row["micro_batch"] != BATCH
            or row["sequence_length"] != CONTEXT
            or row["tokens_per_second"] <= 0
        ):
            parser.error(f"calibration shape or result does not match {name}")
    frozen = json.loads((output / "preflight_data.json").read_text(encoding="utf-8"))
    if file_sha256(output / "diagnostic_v2.jsonl") != frozen["bank_sha256"]:
        parser.error("diagnostic bank changed")
    if BPETokenizer.load(args.tokenizer).version != frozen["corpus"]["tokenizer"]:
        parser.error("tokenizer version differs from the frozen corpus")
    for split, details in frozen["corpus"]["splits"].items():
        if file_sha256(args.corpus / "tokens" / f"{split}.bin") != details["sha256"]:
            parser.error(f"{split} token binary changed")
    if not torch.backends.mps.is_available():
        parser.error("MPS unavailable")
    study_state = output / "study.json"
    plan = {
        "bank_sha256": frozen["bank_sha256"],
        "corpus": {key: value["sha256"] for key, value in frozen["corpus"]["splits"].items()},
        "tokenizer_sha256": file_sha256(args.tokenizer / "tokenizer.json"),
        "preregistration_sha256": file_sha256(Path("docs/stage17_preregistration.md")),
        "max_wall_seconds": INITIAL_WALL_SECONDS,
        "initial_tokens": BUDGETS[0],
        "configurations": {
            config.name: config.model_dump(mode="json")
            for seed in SEEDS
            for model in ARCHITECTURES
            for config in [
                capacity_config(model, seed, output, args.checkpoints, args.corpus, args.tokenizer)
            ]
        },
    }
    saved = json.loads(study_state.read_text(encoding="utf-8")) if study_state.exists() else None
    if saved is None:
        with study_state.open("x", encoding="utf-8") as handle:
            saved = {"absolute_deadline": time.time() + INITIAL_WALL_SECONDS, "plan": plan}
            handle.write(json.dumps(saved, indent=2) + "\n")
    elif saved["plan"] != plan:
        parser.error("resume contract differs from recorded plan")
    deadline = time.monotonic() + saved["absolute_deadline"] - time.time()
    profile_name, seed_value = args.job.split("-seed-")
    seed = int(seed_value)
    config = capacity_config(
        profile_name, seed, output, args.checkpoints, args.corpus, args.tokenizer
    )
    target = budget_step(BUDGETS[0])
    manager = CheckpointManager(config.train.checkpoint_dir, keep=2)
    previous = manager.latest_valid()
    if previous is None and manager.list_steps():
        raise ResourceError(f"no valid checkpoint can restore {config.name}")
    step = CheckpointManager.verify(previous)["step"] if previous is not None else 0
    if step >= target:
        return
    available = psutil.virtual_memory().available / BYTES_PER_GIB
    disk = free_disk_gb(config.train.checkpoint_dir)
    needed = measured[profile_name]["device_memory_gib"] + 0.75
    remaining = deadline - time.monotonic()
    if disk < MIN_DISK_GIB or available < needed or remaining <= 0:
        raise ResourceError(
            f"resource gate: RAM available={available:.2f} GiB, "
            f"RAM required={needed:.2f} GiB, disk={disk:.2f} GiB, "
            f"remaining={remaining:.0f} s"
        )
    result = run_training(config, previous is not None, target, deadline=deadline)
    print(json.dumps({"profile": profile_name, "seed": seed, **asdict(result)}), flush=True)
    release_device_memory(torch.device("mps"))


if __name__ == "__main__":
    main()
