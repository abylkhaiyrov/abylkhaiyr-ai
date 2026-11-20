from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from llm.core.config import load_experiment_config
from llm.training.session import run_training

SOURCES = {
    "pretrain": ("checkpoints/tiny_gpu/best", 0.00025),
    "sft": ("checkpoints/tiny_gpu_sft/best", 0.00012),
    "verified": ("checkpoints/tiny_gpu_sft/best", 0.00009),
}
MIXTURE = {
    "human_ru": 0.19,
    "human_en": 0.15,
    "human_kk": 0.06,
    "reviewed": 0.09,
    "dialog": 0.12,
    "identity": 0.07,
    "skills": 0.13,
    "reading_ru": 0.05,
    "reading_kk": 0.04,
    "reading_en": 0.03,
    "chat_en": 0.04,
    "code": 0.03,
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=650)
    parser.add_argument("--start", choices=tuple(SOURCES), nargs="+", default=["pretrain", "sft"])
    args = parser.parse_args()
    for start in args.start:
        reference, learning_rate = SOURCES[start]
        config = load_experiment_config("configs/tiny_gpu_sft.yaml")
        root = Path("experiments/dialogue_v4") / start
        config.name = f"dialogue_v4_{start}"
        mixture = MIXTURE
        if start == "verified":
            mixture = {
                "grounded_disjoint": 0.30,
                "reviewed": 0.10,
                "human_ru": 0.08,
                "human_en": 0.06,
                "dialog": 0.13,
                "identity": 0.08,
                "skills": 0.17,
                "reading_ru": 0.03,
                "reading_kk": 0.03,
                "code": 0.02,
            }
        config.train = config.train.model_copy(
            update={
                "data_dir": "data/dialogue_v4",
                "init_from": reference,
                "output_dir": str(root),
                "checkpoint_dir": str(root / "weights"),
                "max_steps": args.steps,
                "micro_batch_size": 16,
                "gradient_accumulation_steps": 2,
                "sequence_length": 383,
                "pad_multiple": 16,
                "mixture": mixture,
                "sft_replay_dir": "data/dialogue_v4/replay",
                "sft_replay_fraction": 0.12,
                "eval_interval": 75,
                "eval_batches": 70,
                "checkpoint_interval": 150,
                "keep_checkpoints": 4,
                "early_stopping_patience": 4,
            }
        )
        config.train.optimizer.learning_rate = learning_rate
        config.train.optimizer.warmup_steps = 40
        result = run_training(config, resume=True, stop_file=Path("experiments/dialogue_v4/STOP"))
        (root / "result.json").write_text(json.dumps(asdict(result), indent=2), encoding="utf-8")
        if result.interrupted:
            break


if __name__ == "__main__":
    main()
